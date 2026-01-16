//! Hybrid optimization strategies combining multiple methods.
//!
//! Based on Soares et al. 2021 recommendations for combining:
//! - Fast 2D beam analysis for initial exploration
//! - 3D FEM for accurate evaluation
//! - Evolutionary algorithms for global search
//! - Surrogate optimization for expensive function minimization
//!
//! The key insight is that 2D beam models are ~100x faster than 3D FEM,
//! so using 2D to find a good starting region, then refining with 3D surrogate
//! optimization, dramatically reduces total computation time.

use crate::beam2d_solver::compute_modal_frequencies_2d_from_cuts;
use crate::cuts::{generate_element_heights, genes_to_cuts};
use crate::mesh::generate_bar_mesh_3d;
use crate::optimization::objective::{compute_tuning_error, compute_volume_penalty};
use crate::optimization::surrogate::{run_surrogate_optimization, SurrogateConfig};
use crate::optimization::types::{
    AnalysisMode, BarParameters, BoundsConstraints, EAParameters, Material, PenaltyType,
    VariableBounds,
};
use crate::optimization::{run_optimization, EAConfig};
use crate::solver::compute_modal_frequencies_with_solver;
use crate::types::EigenSolver;

/// Optimization strategy selection.
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    /// Pure evolutionary algorithm (existing implementation)
    Evolutionary(EAParameters),

    /// Pure surrogate optimization with 3D FEM
    Surrogate3D(SurrogateConfig),

    /// Hybrid: 2D EA exploration → 3D surrogate refinement (recommended)
    Hybrid2Dto3D {
        /// EA parameters for 2D phase
        ea_params: EAParameters,
        /// Surrogate config for 3D phase
        surrogate_config: SurrogateConfig,
        /// Frequency correction factor for 2D→3D (typically 0.98-1.02)
        frequency_correction: f64,
    },

    /// Two-phase: EA exploration → surrogate refinement (same analysis mode)
    TwoPhase {
        /// EA parameters for exploration phase
        ea_params: EAParameters,
        /// Number of EA generations for exploration
        ea_generations: usize,
        /// Surrogate config for refinement phase
        surrogate_config: SurrogateConfig,
    },
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        OptimizationStrategy::Hybrid2Dto3D {
            ea_params: EAParameters {
                analysis_mode: AnalysisMode::Beam2D,
                max_generations: 50,
                population_size: 40,
                ..Default::default()
            },
            surrogate_config: SurrogateConfig::default(),
            frequency_correction: 1.0,
        }
    }
}

/// Configuration for hybrid optimization.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Bar geometry parameters
    pub bar: BarParameters,
    /// Material properties
    pub material: Material,
    /// Target frequencies (Hz)
    pub target_frequencies: Vec<f64>,
    /// Number of cuts
    pub num_cuts: usize,
    /// Optimization strategy
    pub strategy: OptimizationStrategy,
    /// Penalty type
    pub penalty_type: PenaltyType,
    /// Penalty weight (alpha)
    pub penalty_alpha: f64,
    /// f1 priority weight
    pub f1_priority: f64,
    /// Bounds constraints
    pub bounds_constraints: Option<BoundsConstraints>,
    /// Verbose output
    pub verbose: bool,
}

impl HybridConfig {
    /// Create a new hybrid configuration.
    pub fn new(
        bar: BarParameters,
        material: Material,
        target_frequencies: Vec<f64>,
        num_cuts: usize,
    ) -> Self {
        Self {
            bar,
            material,
            target_frequencies,
            num_cuts,
            strategy: OptimizationStrategy::default(),
            penalty_type: PenaltyType::None,
            penalty_alpha: 0.0,
            f1_priority: 1.0,
            bounds_constraints: None,
            verbose: false,
        }
    }

    /// Set optimization strategy.
    pub fn with_strategy(mut self, strategy: OptimizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set penalty type and weight.
    pub fn with_penalty(mut self, penalty_type: PenaltyType, alpha: f64) -> Self {
        self.penalty_type = penalty_type;
        self.penalty_alpha = alpha;
        self
    }

    /// Set verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set bounds constraints.
    pub fn with_bounds_constraints(mut self, constraints: BoundsConstraints) -> Self {
        self.bounds_constraints = Some(constraints);
        self
    }
}

/// Result from hybrid optimization.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Best parameters found (genes)
    pub best_genes: Vec<f64>,
    /// Best fitness value
    pub best_fitness: f64,
    /// Computed frequencies at best solution
    pub computed_frequencies: Vec<f64>,
    /// Target frequencies
    pub target_frequencies: Vec<f64>,
    /// Errors in cents
    pub errors_cents: Vec<f64>,
    /// Maximum error in cents
    pub max_error_cents: f64,
    /// Volume penalty percentage
    pub volume_percent: f64,
    /// Total evaluations (2D + 3D)
    pub total_evaluations_2d: usize,
    /// Total 3D evaluations
    pub total_evaluations_3d: usize,
    /// Phase 1 result (if applicable)
    pub phase1_fitness: Option<f64>,
    /// Converged flag
    pub converged: bool,
}

/// Run hybrid optimization.
///
/// This is the main entry point for the hybrid optimization system.
pub fn run_hybrid_optimization(config: &HybridConfig) -> HybridResult {
    match &config.strategy {
        OptimizationStrategy::Evolutionary(ea_params) => {
            run_evolutionary_strategy(config, ea_params)
        }
        OptimizationStrategy::Surrogate3D(surrogate_config) => {
            run_surrogate_3d_strategy(config, surrogate_config)
        }
        OptimizationStrategy::Hybrid2Dto3D {
            ea_params,
            surrogate_config,
            frequency_correction,
        } => run_hybrid_2d_to_3d_strategy(config, ea_params, surrogate_config, *frequency_correction),
        OptimizationStrategy::TwoPhase {
            ea_params,
            ea_generations,
            surrogate_config,
        } => run_two_phase_strategy(config, ea_params, *ea_generations, surrogate_config),
    }
}

/// Run pure evolutionary strategy (wrapper around existing implementation).
fn run_evolutionary_strategy(config: &HybridConfig, ea_params: &EAParameters) -> HybridResult {
    let mut params = ea_params.clone();
    params.f1_priority = config.f1_priority;

    let ea_config = EAConfig::new(
        config.bar.clone(),
        config.material.clone(),
        config.target_frequencies.clone(),
        config.num_cuts,
    )
    .with_params(params)
    .with_penalty(config.penalty_type, config.penalty_alpha);

    let result = run_optimization(&ea_config);

    let errors_cents: Vec<f64> = result
        .computed_frequencies
        .iter()
        .zip(result.target_frequencies.iter())
        .map(|(c, t)| 1200.0 * (c / t).log2())
        .collect();

    let max_error_cents = errors_cents
        .iter()
        .map(|e| e.abs())
        .fold(0.0, f64::max);

    HybridResult {
        best_genes: result.best_individual.genes,
        best_fitness: result.tuning_error,
        computed_frequencies: result.computed_frequencies,
        target_frequencies: result.target_frequencies,
        errors_cents,
        max_error_cents,
        volume_percent: result.volume_percent,
        total_evaluations_2d: if ea_params.analysis_mode == AnalysisMode::Beam2D {
            result.generations * ea_params.population_size
        } else {
            0
        },
        total_evaluations_3d: if ea_params.analysis_mode == AnalysisMode::Solid3D {
            result.generations * ea_params.population_size
        } else {
            0
        },
        phase1_fitness: None,
        converged: result.tuning_error < ea_params.target_error,
    }
}

/// Run pure surrogate 3D strategy.
fn run_surrogate_3d_strategy(
    config: &HybridConfig,
    surrogate_config: &SurrogateConfig,
) -> HybridResult {
    let bounds = compute_bounds(config);
    let num_genes = config.num_cuts * 2;

    let evaluate_fn = |genes: &[f64]| -> f64 {
        evaluate_3d(
            genes,
            config,
            num_genes,
        )
    };

    let result = run_surrogate_optimization(surrogate_config, bounds, evaluate_fn);

    // Compute final frequencies at best solution
    let (computed_frequencies, fitness) = evaluate_3d_with_frequencies(&result.best_params, config, num_genes);

    let errors_cents: Vec<f64> = computed_frequencies
        .iter()
        .zip(config.target_frequencies.iter())
        .map(|(c, t)| 1200.0 * (c / t).log2())
        .collect();

    let max_error_cents = errors_cents.iter().map(|e| e.abs()).fold(0.0, f64::max);

    let cuts = genes_to_cuts(&result.best_params[..num_genes.min(result.best_params.len())]);
    let volume_percent = compute_volume_penalty(&cuts, config.bar.length, config.bar.h0);

    HybridResult {
        best_genes: result.best_params,
        best_fitness: fitness,
        computed_frequencies,
        target_frequencies: config.target_frequencies.clone(),
        errors_cents,
        max_error_cents,
        volume_percent,
        total_evaluations_2d: 0,
        total_evaluations_3d: result.evaluations,
        phase1_fitness: None,
        converged: result.converged,
    }
}

/// Run hybrid 2D→3D strategy (recommended approach from Soares paper).
fn run_hybrid_2d_to_3d_strategy(
    config: &HybridConfig,
    ea_params: &EAParameters,
    surrogate_config: &SurrogateConfig,
    frequency_correction: f64,
) -> HybridResult {
    if config.verbose {
        println!("=== Phase 1: 2D Evolutionary Optimization ===");
    }

    // Phase 1: Fast 2D optimization
    let mut params_2d = ea_params.clone();
    params_2d.analysis_mode = AnalysisMode::Beam2D;
    params_2d.f1_priority = config.f1_priority;

    // Adjust targets for 2D (apply correction factor)
    let targets_2d: Vec<f64> = config
        .target_frequencies
        .iter()
        .map(|f| f * frequency_correction)
        .collect();

    let ea_config_2d = EAConfig::new(
        config.bar.clone(),
        config.material.clone(),
        targets_2d,
        config.num_cuts,
    )
    .with_params(params_2d.clone())
    .with_penalty(config.penalty_type, config.penalty_alpha);

    let result_2d = run_optimization(&ea_config_2d);
    let phase1_fitness = result_2d.tuning_error;
    let phase1_genes = result_2d.best_individual.genes.clone();

    if config.verbose {
        println!(
            "Phase 1 complete: fitness = {:.6}, {} generations",
            phase1_fitness, result_2d.generations
        );
        println!("\n=== Phase 2: 3D Surrogate Refinement ===");
    }

    // Phase 2: 3D surrogate refinement starting from 2D solution
    let bounds = compute_bounds(config);
    let num_genes = config.num_cuts * 2;

    let mut surrogate_config_3d = surrogate_config.clone();
    surrogate_config_3d.initial_point = Some(phase1_genes.clone());
    surrogate_config_3d.verbose = config.verbose;

    let evaluate_fn = |genes: &[f64]| -> f64 {
        evaluate_3d(genes, config, num_genes)
    };

    let result_3d = run_surrogate_optimization(&surrogate_config_3d, bounds, evaluate_fn);

    // Compute final frequencies
    let (computed_frequencies, fitness) =
        evaluate_3d_with_frequencies(&result_3d.best_params, config, num_genes);

    let errors_cents: Vec<f64> = computed_frequencies
        .iter()
        .zip(config.target_frequencies.iter())
        .map(|(c, t)| 1200.0 * (c / t).log2())
        .collect();

    let max_error_cents = errors_cents.iter().map(|e| e.abs()).fold(0.0, f64::max);

    let cuts = genes_to_cuts(&result_3d.best_params[..num_genes.min(result_3d.best_params.len())]);
    let volume_percent = compute_volume_penalty(&cuts, config.bar.length, config.bar.h0);

    if config.verbose {
        println!(
            "Phase 2 complete: fitness = {:.6}, {} evaluations",
            fitness, result_3d.evaluations
        );
        println!("Final max error: {:.1} cents", max_error_cents);
    }

    HybridResult {
        best_genes: result_3d.best_params,
        best_fitness: fitness,
        computed_frequencies,
        target_frequencies: config.target_frequencies.clone(),
        errors_cents,
        max_error_cents,
        volume_percent,
        total_evaluations_2d: result_2d.generations * params_2d.population_size,
        total_evaluations_3d: result_3d.evaluations,
        phase1_fitness: Some(phase1_fitness),
        converged: result_3d.converged,
    }
}

/// Run two-phase EA→surrogate strategy.
fn run_two_phase_strategy(
    config: &HybridConfig,
    ea_params: &EAParameters,
    ea_generations: usize,
    surrogate_config: &SurrogateConfig,
) -> HybridResult {
    if config.verbose {
        println!("=== Phase 1: EA Exploration ({} generations) ===", ea_generations);
    }

    // Phase 1: Limited EA exploration
    let mut params = ea_params.clone();
    params.max_generations = ea_generations;
    params.f1_priority = config.f1_priority;

    let ea_config = EAConfig::new(
        config.bar.clone(),
        config.material.clone(),
        config.target_frequencies.clone(),
        config.num_cuts,
    )
    .with_params(params.clone())
    .with_penalty(config.penalty_type, config.penalty_alpha);

    let result_ea = run_optimization(&ea_config);
    let phase1_fitness = result_ea.tuning_error;

    if config.verbose {
        println!("Phase 1 complete: fitness = {:.6}", phase1_fitness);
        println!("\n=== Phase 2: Surrogate Refinement ===");
    }

    // Phase 2: Surrogate refinement
    let bounds = compute_bounds(config);
    let num_genes = config.num_cuts * 2;

    let mut surrogate_config_refined = surrogate_config.clone();
    surrogate_config_refined.initial_point = Some(result_ea.best_individual.genes.clone());
    surrogate_config_refined.verbose = config.verbose;

    let is_3d = params.analysis_mode == AnalysisMode::Solid3D;
    let evaluate_fn = |genes: &[f64]| -> f64 {
        if is_3d {
            evaluate_3d(genes, config, num_genes)
        } else {
            evaluate_2d(genes, config, num_genes)
        }
    };

    let result_surrogate = run_surrogate_optimization(&surrogate_config_refined, bounds, evaluate_fn);

    // Compute final frequencies
    let (computed_frequencies, fitness) = if is_3d {
        evaluate_3d_with_frequencies(&result_surrogate.best_params, config, num_genes)
    } else {
        evaluate_2d_with_frequencies(&result_surrogate.best_params, config, num_genes)
    };

    let errors_cents: Vec<f64> = computed_frequencies
        .iter()
        .zip(config.target_frequencies.iter())
        .map(|(c, t)| 1200.0 * (c / t).log2())
        .collect();

    let max_error_cents = errors_cents.iter().map(|e| e.abs()).fold(0.0, f64::max);

    let cuts = genes_to_cuts(&result_surrogate.best_params[..num_genes.min(result_surrogate.best_params.len())]);
    let volume_percent = compute_volume_penalty(&cuts, config.bar.length, config.bar.h0);

    HybridResult {
        best_genes: result_surrogate.best_params,
        best_fitness: fitness,
        computed_frequencies,
        target_frequencies: config.target_frequencies.clone(),
        errors_cents,
        max_error_cents,
        volume_percent,
        total_evaluations_2d: if !is_3d {
            ea_generations * params.population_size + result_surrogate.evaluations
        } else {
            0
        },
        total_evaluations_3d: if is_3d {
            ea_generations * params.population_size + result_surrogate.evaluations
        } else {
            0
        },
        phase1_fitness: Some(phase1_fitness),
        converged: result_surrogate.converged,
    }
}

/// Compute variable bounds from config.
fn compute_bounds(config: &HybridConfig) -> Vec<(f64, f64)> {
    let var_bounds = VariableBounds::from_bar(
        &config.bar,
        config.num_cuts,
        config.bounds_constraints.as_ref(),
    );

    let mut bounds = Vec::with_capacity(config.num_cuts * 2);
    for _ in 0..config.num_cuts {
        bounds.push((var_bounds.lambda_min, var_bounds.lambda_max));
        bounds.push((var_bounds.h_min, var_bounds.h_max));
    }
    bounds
}

/// Evaluate fitness using 3D FEM.
fn evaluate_3d(genes: &[f64], config: &HybridConfig, num_genes: usize) -> f64 {
    let (_, fitness) = evaluate_3d_with_frequencies(genes, config, num_genes);
    fitness
}

/// Evaluate fitness using 3D FEM and return frequencies.
fn evaluate_3d_with_frequencies(
    genes: &[f64],
    config: &HybridConfig,
    num_genes: usize,
) -> (Vec<f64>, f64) {
    let cuts = genes_to_cuts(&genes[..num_genes.min(genes.len())]);
    let num_elements_x = 30; // Could be configurable
    let heights = generate_element_heights(&cuts, config.bar.length, config.bar.h0, num_elements_x);
    let mesh = generate_bar_mesh_3d(
        config.bar.length,
        config.bar.width,
        &heights,
        num_elements_x,
        2,
        2,
    );

    let num_modes = config.target_frequencies.len();
    let frequencies = compute_modal_frequencies_with_solver(
        &mesh,
        config.material.e,
        config.material.nu,
        config.material.rho,
        num_modes,
        EigenSolver::Auto,
    );

    let tuning_error = compute_tuning_error(&frequencies, &config.target_frequencies, config.f1_priority);

    let fitness = match config.penalty_type {
        PenaltyType::None => tuning_error,
        PenaltyType::Volume => {
            let volume_penalty = compute_volume_penalty(&cuts, config.bar.length, config.bar.h0);
            (1.0 - config.penalty_alpha) * tuning_error + config.penalty_alpha * volume_penalty
        }
        PenaltyType::Roughness => {
            let roughness = crate::optimization::objective::compute_roughness_penalty(&cuts, config.bar.h0);
            (1.0 - config.penalty_alpha) * tuning_error + config.penalty_alpha * roughness
        }
    };

    (frequencies, fitness)
}

/// Evaluate fitness using 2D beam model.
fn evaluate_2d(genes: &[f64], config: &HybridConfig, num_genes: usize) -> f64 {
    let (_, fitness) = evaluate_2d_with_frequencies(genes, config, num_genes);
    fitness
}

/// Evaluate fitness using 2D beam model and return frequencies.
fn evaluate_2d_with_frequencies(
    genes: &[f64],
    config: &HybridConfig,
    num_genes: usize,
) -> (Vec<f64>, f64) {
    let cuts = genes_to_cuts(&genes[..num_genes.min(genes.len())]);
    let num_modes = config.target_frequencies.len();

    let frequencies = compute_modal_frequencies_2d_from_cuts(
        &cuts,
        config.bar.length,
        config.bar.width,
        config.bar.h0,
        config.material.e,
        config.material.nu,
        config.material.rho,
        150, // num_elements
        num_modes,
    );

    let tuning_error = compute_tuning_error(&frequencies, &config.target_frequencies, config.f1_priority);

    let fitness = match config.penalty_type {
        PenaltyType::None => tuning_error,
        PenaltyType::Volume => {
            let volume_penalty = compute_volume_penalty(&cuts, config.bar.length, config.bar.h0);
            (1.0 - config.penalty_alpha) * tuning_error + config.penalty_alpha * volume_penalty
        }
        PenaltyType::Roughness => {
            let roughness = crate::optimization::objective::compute_roughness_penalty(&cuts, config.bar.h0);
            (1.0 - config.penalty_alpha) * tuning_error + config.penalty_alpha * roughness
        }
    };

    (frequencies, fitness)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> HybridConfig {
        let bar = BarParameters::new(0.35, 0.05, 0.01, 0.002);
        let material = Material::aluminum();
        // Simple 1:4 tuning ratio
        let target_frequencies = vec![175.0, 700.0];

        HybridConfig::new(bar, material, target_frequencies, 2)
    }

    #[test]
    fn test_compute_bounds() {
        let config = test_config();
        let bounds = compute_bounds(&config);

        assert_eq!(bounds.len(), 4); // 2 cuts * 2 params (lambda, h)

        // Lambda bounds
        assert!(bounds[0].0 >= 0.0);
        assert!(bounds[0].1 <= config.bar.length / 2.0);

        // Height bounds
        assert!(bounds[1].0 >= config.bar.h_min);
        assert!(bounds[1].1 <= config.bar.h0);
    }

    #[test]
    fn test_evaluate_2d() {
        let config = test_config();
        let genes = vec![0.05, 0.006, 0.1, 0.005]; // 2 cuts

        let (freqs, fitness) = evaluate_2d_with_frequencies(&genes, &config, 4);

        assert!(!freqs.is_empty());
        assert!(fitness.is_finite());
        assert!(fitness >= 0.0);
    }

    #[test]
    fn test_surrogate_strategy_creation() {
        let config = SurrogateConfig::fast();
        assert_eq!(config.initial_samples, 10);
        assert_eq!(config.max_evaluations, 50);

        let config = SurrogateConfig::thorough();
        assert_eq!(config.initial_samples, 25);
        assert_eq!(config.max_evaluations, 150);
    }

    #[test]
    fn test_optimization_strategy_default() {
        let strategy = OptimizationStrategy::default();
        match strategy {
            OptimizationStrategy::Hybrid2Dto3D { .. } => {}
            _ => panic!("Default should be Hybrid2Dto3D"),
        }
    }
}
