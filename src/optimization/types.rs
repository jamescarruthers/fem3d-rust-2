//! Type definitions for evolutionary optimization.

use std::fmt;

/// Analysis mode for FEM computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnalysisMode {
    /// 2D Timoshenko beam elements (fast, good for slender bars).
    #[default]
    Beam2D,
    /// 3D hexahedral elements (accurate, slower).
    Solid3D,
}

/// Material properties for bar tuning.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    /// Young's modulus (Pa)
    pub e: f64,
    /// Density (kg/m³)
    pub rho: f64,
    /// Poisson's ratio
    pub nu: f64,
}

impl Material {
    pub fn new(name: impl Into<String>, e: f64, rho: f64, nu: f64) -> Self {
        Self {
            name: name.into(),
            e,
            rho,
            nu,
        }
    }

    /// Sapele wood - commonly used for marimbas
    pub fn sapele() -> Self {
        Self::new("Sapele", 12.0e9, 640.0, 0.35)
    }

    /// Aluminum 6061
    pub fn aluminum() -> Self {
        Self::new("Aluminum 6061", 68.9e9, 2700.0, 0.33)
    }

    /// Rosewood - premium marimba tonewood
    pub fn rosewood() -> Self {
        Self::new("Honduran Rosewood", 12.5e9, 850.0, 0.37)
    }
}

/// Bar geometry parameters.
#[derive(Debug, Clone)]
pub struct BarParameters {
    /// Length (m)
    pub length: f64,
    /// Width (m)
    pub width: f64,
    /// Original thickness (m)
    pub h0: f64,
    /// Minimum thickness (m) - typically 10% of h0
    pub h_min: f64,
}

impl BarParameters {
    pub fn new(length: f64, width: f64, h0: f64, h_min: f64) -> Self {
        Self {
            length,
            width,
            h0,
            h_min,
        }
    }
}

/// Variable bounds for optimization.
#[derive(Debug, Clone)]
pub struct VariableBounds {
    /// Minimum lambda (distance from center)
    pub lambda_min: f64,
    /// Maximum lambda (L/2)
    pub lambda_max: f64,
    /// Minimum height
    pub h_min: f64,
    /// Maximum height (h0)
    pub h_max: f64,
    /// Minimum spacing between cut boundaries (m)
    pub min_cut_width: f64,
    /// Maximum cut width (2*lambda) (m), 0 = no limit
    pub max_cut_width: f64,
    /// Minimum depth (h0 - h) (m)
    pub min_cut_depth: f64,
    /// Maximum depth (h0 - h) (m)
    pub max_cut_depth: f64,
    /// Maximum trim from each end (m)
    pub max_length_trim: f64,
    /// Maximum extension from each end (m)
    pub max_length_extend: f64,
}

impl VariableBounds {
    /// Create bounds from bar parameters.
    pub fn from_bar(bar: &BarParameters, _num_cuts: usize, constraints: Option<&BoundsConstraints>) -> Self {
        let constraints = constraints.cloned().unwrap_or_default();

        let lambda_min = 0.0;
        let lambda_max = bar.length / 2.0;

        // Height bounds (depth = h0 - h, so larger depth means smaller h)
        let h_max = if constraints.min_cut_depth > 0.0 {
            bar.h_min.max(bar.h0 - constraints.min_cut_depth)
        } else {
            bar.h0
        };

        let h_min = if constraints.max_cut_depth > 0.0 {
            bar.h_min.max(bar.h0 - constraints.max_cut_depth)
        } else {
            bar.h_min
        };

        Self {
            lambda_min,
            lambda_max,
            h_min,
            h_max,
            min_cut_width: constraints.min_cut_width,
            max_cut_width: constraints.max_cut_width,
            min_cut_depth: constraints.min_cut_depth,
            max_cut_depth: constraints.max_cut_depth,
            max_length_trim: constraints.max_length_trim,
            max_length_extend: constraints.max_length_extend,
        }
    }

    /// Check if length adjustment is enabled.
    pub fn has_length_adjust(&self) -> bool {
        self.max_length_trim > 0.0 || self.max_length_extend > 0.0
    }
}

/// Constraint options for creating bounds.
#[derive(Debug, Clone, Default)]
pub struct BoundsConstraints {
    /// Minimum spacing between cut boundaries (m)
    pub min_cut_width: f64,
    /// Maximum cut width = 2*lambda (m), 0 = no limit
    pub max_cut_width: f64,
    /// Minimum depth = h0 - h (m), 0 = no limit
    pub min_cut_depth: f64,
    /// Maximum depth = h0 - h (m), 0 = use bar.h_min
    pub max_cut_depth: f64,
    /// Maximum trim from each end (m), 0 = no trimming
    pub max_length_trim: f64,
    /// Maximum extension from each end (m), 0 = no extension
    pub max_length_extend: f64,
}

/// Individual in EA population.
#[derive(Debug, Clone)]
pub struct Individual {
    /// Genes: [lambda_1, h_1, lambda_2, h_2, ..., length_adjust?]
    pub genes: Vec<f64>,
    /// Fitness value (lower is better)
    pub fitness: f64,
    /// Sigmas for self-adaptive Gaussian mutation
    pub sigmas: Option<Vec<f64>>,
}

impl Individual {
    pub fn new(genes: Vec<f64>) -> Self {
        Self {
            genes,
            fitness: f64::INFINITY,
            sigmas: None,
        }
    }

    pub fn with_fitness(genes: Vec<f64>, fitness: f64) -> Self {
        Self {
            genes,
            fitness,
            sigmas: None,
        }
    }

    pub fn with_sigmas(genes: Vec<f64>, sigmas: Vec<f64>) -> Self {
        Self {
            genes,
            fitness: f64::INFINITY,
            sigmas: Some(sigmas),
        }
    }
}

/// Evolutionary algorithm parameters.
#[derive(Debug, Clone)]
pub struct EAParameters {
    /// Population size (Npop)
    pub population_size: usize,
    /// Elitism percentage (Pe, 0-100)
    pub elitism_percent: f64,
    /// Crossover percentage (Pc, 0-100)
    pub crossover_percent: f64,
    /// Mutation percentage (Pm, 0-100, Pe + Pc + Pm = 100)
    pub mutation_percent: f64,
    /// Mutation strength (sigma for uniform mutation, 0.05-0.2)
    pub mutation_strength: f64,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Target error percentage (stopping criterion)
    pub target_error: f64,
    /// Number of FEM elements (Ne for discretization)
    pub num_elements: usize,
    /// Weight multiplier for f1
    pub f1_priority: f64,
    /// Minimum width between cut boundaries (m)
    pub min_cut_width: f64,
    /// Maximum cut width (2*lambda) (m), 0 = no limit
    pub max_cut_width: f64,
    /// Minimum cut depth (h0 - h) (m), 0 = no limit
    pub min_cut_depth: f64,
    /// Maximum cut depth (h0 - h) (m), 0 = no limit
    pub max_cut_depth: f64,
    /// Max trim from each end (m), 0 = no trimming
    pub max_length_trim: f64,
    /// Max extension from each end (m), 0 = no extension
    pub max_length_extend: f64,
    /// Max worker threads (0 = auto)
    pub max_workers: usize,
    /// Analysis mode (2D beam or 3D solid)
    pub analysis_mode: AnalysisMode,
    /// 3D mesh: elements in width direction
    pub num_elements_y: usize,
    /// 3D mesh: elements in thickness direction
    pub num_elements_z: usize,
    /// Frequency offset for 2D/3D calibration
    pub frequency_offset: f64,
}

impl Default for EAParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            elitism_percent: 10.0,
            crossover_percent: 30.0,
            mutation_percent: 60.0,
            mutation_strength: 0.1,
            max_generations: 100,
            target_error: 0.01,
            num_elements: 150,
            f1_priority: 1.0,
            min_cut_width: 0.0,
            max_cut_width: 0.0,
            min_cut_depth: 0.0,
            max_cut_depth: 0.0,
            max_length_trim: 0.0,
            max_length_extend: 0.0,
            max_workers: 0,
            analysis_mode: AnalysisMode::Beam2D,
            num_elements_y: 2,
            num_elements_z: 2,
            frequency_offset: 0.0,
        }
    }
}

impl EAParameters {
    /// Create default parameters for a given number of cuts.
    pub fn for_num_cuts(num_cuts: usize) -> Self {
        Self {
            population_size: 30.max(num_cuts * 10),
            ..Default::default()
        }
    }

    /// Get bounds constraints from parameters.
    pub fn to_bounds_constraints(&self) -> BoundsConstraints {
        BoundsConstraints {
            min_cut_width: self.min_cut_width,
            max_cut_width: self.max_cut_width,
            min_cut_depth: self.min_cut_depth,
            max_cut_depth: self.max_cut_depth,
            max_length_trim: self.max_length_trim,
            max_length_extend: self.max_length_extend,
        }
    }
}

/// Population statistics.
#[derive(Debug, Clone)]
pub struct PopulationStats {
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub average_fitness: f64,
    pub median_fitness: f64,
    pub standard_deviation: f64,
}

/// Detailed evaluation results.
#[derive(Debug, Clone)]
pub struct DetailedEvaluation {
    pub computed_frequencies: Vec<f64>,
    pub target_frequencies: Vec<f64>,
    pub tuning_error: f64,
    pub volume_penalty: f64,
    pub roughness_penalty: f64,
    pub combined_fitness: f64,
    pub cents_errors: Vec<f64>,
    pub max_cents_error: f64,
}

/// Progress update during optimization.
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub generation: usize,
    pub best_fitness: f64,
    pub best_individual: Individual,
    pub average_fitness: f64,
    pub computed_frequencies: Option<Vec<f64>>,
    pub errors_in_cents: Option<Vec<f64>>,
    pub length_trim: f64,
}

/// Optimization result.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_individual: Individual,
    pub cuts: Vec<crate::cuts::Cut>,
    pub computed_frequencies: Vec<f64>,
    pub target_frequencies: Vec<f64>,
    pub tuning_error: f64,
    pub max_error_cents: f64,
    pub errors_in_cents: Vec<f64>,
    pub volume_percent: f64,
    pub roughness_percent: f64,
    pub generations: usize,
    pub length_trim: f64,
    pub effective_length: f64,
}

impl fmt::Display for OptimizationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Optimization Result:")?;
        writeln!(f, "  Generations: {}", self.generations)?;
        writeln!(f, "  Tuning error: {:.4}%", self.tuning_error)?;
        writeln!(f, "  Max cents error: {:.1} cents", self.max_error_cents)?;
        writeln!(f, "  Volume removed: {:.1}%", self.volume_percent)?;
        writeln!(f, "  Length trim: {:.4} m", self.length_trim)?;
        writeln!(f, "  Effective length: {:.4} m", self.effective_length)?;
        writeln!(f, "  Frequencies (Hz):")?;
        for (i, (comp, target)) in self
            .computed_frequencies
            .iter()
            .zip(self.target_frequencies.iter())
            .enumerate()
        {
            let cents = if i < self.errors_in_cents.len() {
                self.errors_in_cents[i]
            } else {
                0.0
            };
            writeln!(
                f,
                "    f{}: {:.1} Hz (target: {:.1} Hz, error: {:.1} cents)",
                i + 1,
                comp,
                target,
                cents
            )?;
        }
        writeln!(f, "  Cuts:")?;
        for (i, cut) in self.cuts.iter().enumerate() {
            writeln!(
                f,
                "    Cut {}: lambda={:.4} m, h={:.4} m",
                i + 1,
                cut.lambda,
                cut.h
            )?;
        }
        Ok(())
    }
}

/// Penalty type for optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PenaltyType {
    /// No penalty
    #[default]
    None,
    /// Volume-based penalty
    Volume,
    /// Roughness-based penalty
    Roughness,
}

/// Selection method for parent selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionMethod {
    /// Roulette wheel selection (fitness proportional)
    #[default]
    Roulette,
    /// Tournament selection
    Tournament,
    /// Rank-based selection
    Rank,
}

/// Crossover method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrossoverMethod {
    /// Heuristic crossover from paper (Eq. 16)
    #[default]
    Heuristic,
    /// Single-point crossover
    SinglePoint,
    /// Two-point crossover
    TwoPoint,
    /// Uniform crossover
    Uniform,
    /// Blend crossover (BLX-alpha)
    Blend,
}

/// Mutation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MutationMethod {
    /// Uniform random mutation
    #[default]
    Uniform,
    /// Self-adaptive Gaussian mutation
    GaussianAdaptive,
    /// Polynomial mutation
    Polynomial,
}
