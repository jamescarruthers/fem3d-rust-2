//! Evolutionary Algorithm for Bar Tuning Optimization.
//!
//! Main optimization loop implementing the algorithm from Section 3 of the paper.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::beam2d_solver::compute_modal_frequencies_2d_from_cuts;
use crate::cuts::genes_to_cuts;
use crate::mesh::generate_bar_mesh_3d;
use crate::solver::compute_modal_frequencies;

use super::crossover::heuristic_crossover;
use super::mutation::{adaptive_length_mutation, uniform_mutation, FrequencyError};
use super::objective::{compute_cents_errors, compute_tuning_error, evaluate_detailed};
use super::population::{
    calculate_population_stats, clone_individual, create_uncut_bar_individual,
    get_best_individual, get_length_adjust_from_genes, initialize_population,
};
use super::selection::{select_elite, select_mating_pairs};
use super::types::{
    AnalysisMode, BarParameters, EAParameters, Individual, Material, OptimizationResult,
    PenaltyType, ProgressUpdate, SelectionMethod, VariableBounds,
};

/// Configuration for the evolutionary algorithm.
#[derive(Debug, Clone)]
pub struct EAConfig {
    pub bar: BarParameters,
    pub material: Material,
    pub target_frequencies: Vec<f64>,
    pub num_cuts: usize,
    pub penalty_type: PenaltyType,
    pub penalty_weight: f64,
    pub ea_params: EAParameters,
    pub seed_genes: Option<Vec<f64>>,
}

impl EAConfig {
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
            penalty_type: PenaltyType::None,
            penalty_weight: 0.0,
            ea_params: EAParameters::for_num_cuts(num_cuts),
            seed_genes: None,
        }
    }

    pub fn with_params(mut self, params: EAParameters) -> Self {
        self.ea_params = params;
        self
    }

    pub fn with_penalty(mut self, penalty_type: PenaltyType, weight: f64) -> Self {
        self.penalty_type = penalty_type;
        self.penalty_weight = weight;
        self
    }

    pub fn with_seed(mut self, seed_genes: Vec<f64>) -> Self {
        self.seed_genes = Some(seed_genes);
        self
    }
}

/// Compute frequencies for given genes.
fn compute_frequencies_from_genes(
    genes: &[f64],
    bar: &BarParameters,
    material: &Material,
    num_modes: usize,
    num_elements: usize,
    num_cuts: usize,
    analysis_mode: AnalysisMode,
    ny: usize,
    nz: usize,
) -> Vec<f64> {
    let length_adjust = get_length_adjust_from_genes(genes, num_cuts);
    let effective_length = bar.length - 2.0 * length_adjust;

    let cuts = genes_to_cuts(&genes[..num_cuts * 2]);

    match analysis_mode {
        AnalysisMode::Beam2D => compute_modal_frequencies_2d_from_cuts(
            &cuts,
            effective_length,
            bar.width,
            bar.h0,
            material.e,
            material.nu,
            material.rho,
            num_elements,
            num_modes,
        ),
        AnalysisMode::Solid3D => {
            let heights =
                crate::cuts::generate_element_heights(&cuts, effective_length, bar.h0, num_elements);
            let mesh = generate_bar_mesh_3d(effective_length, bar.width, &heights, num_elements, ny, nz);
            compute_modal_frequencies(&mesh, material.e, material.nu, material.rho, num_modes)
        }
    }
}

/// Evaluate a single individual's fitness.
fn evaluate_individual(
    individual: &Individual,
    bar: &BarParameters,
    material: &Material,
    target_frequencies: &[f64],
    params: &EAParameters,
    num_cuts: usize,
    penalty_type: PenaltyType,
    penalty_weight: f64,
) -> f64 {
    let computed_freq = compute_frequencies_from_genes(
        &individual.genes,
        bar,
        material,
        target_frequencies.len(),
        params.num_elements,
        num_cuts,
        params.analysis_mode,
        params.num_elements_y,
        params.num_elements_z,
    );

    if computed_freq.is_empty() {
        return f64::INFINITY;
    }

    let tuning_error = compute_tuning_error(&computed_freq, target_frequencies, params.f1_priority);

    if penalty_type == PenaltyType::None || penalty_weight == 0.0 {
        return tuning_error;
    }

    let length_adjust = get_length_adjust_from_genes(&individual.genes, num_cuts);
    let effective_length = bar.length - 2.0 * length_adjust;
    let cuts = genes_to_cuts(&individual.genes[..num_cuts * 2]);

    let penalty = match penalty_type {
        PenaltyType::Volume => {
            super::objective::compute_volume_penalty(&cuts, effective_length, bar.h0)
        }
        PenaltyType::Roughness => super::objective::compute_roughness_penalty(&cuts, bar.h0),
        PenaltyType::None => 0.0,
    };

    (1.0 - penalty_weight) * tuning_error + penalty_weight * penalty
}

/// Batch evaluate population fitness.
/// Uses parallel processing when the `parallel` feature is enabled.
#[cfg(feature = "parallel")]
fn batch_evaluate_population(
    population: Vec<Individual>,
    bar: &BarParameters,
    material: &Material,
    target_frequencies: &[f64],
    params: &EAParameters,
    num_cuts: usize,
    penalty_type: PenaltyType,
    penalty_weight: f64,
) -> Vec<Individual> {
    population
        .into_par_iter()
        .map(|ind| {
            let fitness = evaluate_individual(
                &ind,
                bar,
                material,
                target_frequencies,
                params,
                num_cuts,
                penalty_type,
                penalty_weight,
            );

            Individual {
                genes: ind.genes,
                fitness,
                sigmas: ind.sigmas,
            }
        })
        .collect()
}

/// Batch evaluate population fitness (sequential, for WASM).
#[cfg(not(feature = "parallel"))]
fn batch_evaluate_population(
    population: Vec<Individual>,
    bar: &BarParameters,
    material: &Material,
    target_frequencies: &[f64],
    params: &EAParameters,
    num_cuts: usize,
    penalty_type: PenaltyType,
    penalty_weight: f64,
) -> Vec<Individual> {
    population
        .into_iter()
        .map(|ind| {
            let fitness = evaluate_individual(
                &ind,
                bar,
                material,
                target_frequencies,
                params,
                num_cuts,
                penalty_type,
                penalty_weight,
            );

            Individual {
                genes: ind.genes,
                fitness,
                sigmas: ind.sigmas,
            }
        })
        .collect()
}

/// Run the evolutionary algorithm.
///
/// # Arguments
/// * `config` - Algorithm configuration
/// * `on_progress` - Optional callback for progress updates
/// * `should_stop` - Optional callback to check if optimization should stop
///
/// # Returns
/// Optimization result with best solution found.
pub fn run_evolutionary_algorithm<F, S>(
    config: &EAConfig,
    mut on_progress: Option<F>,
    should_stop: Option<S>,
) -> OptimizationResult
where
    F: FnMut(ProgressUpdate),
    S: Fn() -> bool,
{
    let bar = &config.bar;
    let material = &config.material;
    let num_cuts = config.num_cuts;
    let penalty_type = config.penalty_type;
    let penalty_weight = config.penalty_weight;
    let params = &config.ea_params;

    // Apply frequency offset for 2D/3D calibration
    let target_frequencies: Vec<f64> = config
        .target_frequencies
        .iter()
        .map(|f| f * (1.0 + params.frequency_offset))
        .collect();

    let constraints = params.to_bounds_constraints();
    let bounds = VariableBounds::from_bar(bar, num_cuts, Some(&constraints));

    let has_length_adjust = params.max_length_trim > 0.0 || params.max_length_extend > 0.0;

    // Report Generation 0: uncut bar baseline
    if let Some(ref mut progress) = on_progress {
        let uncut_bar = create_uncut_bar_individual(num_cuts, &bounds, bar.h0);
        let fitness = evaluate_individual(
            &uncut_bar,
            bar,
            material,
            &target_frequencies,
            params,
            num_cuts,
            penalty_type,
            penalty_weight,
        );
        let computed = compute_frequencies_from_genes(
            &uncut_bar.genes,
            bar,
            material,
            target_frequencies.len(),
            params.num_elements,
            num_cuts,
            params.analysis_mode,
            params.num_elements_y,
            params.num_elements_z,
        );
        let cents = compute_cents_errors(&computed, &target_frequencies);

        progress(ProgressUpdate {
            generation: 0,
            best_fitness: fitness,
            best_individual: Individual::with_fitness(uncut_bar.genes, fitness),
            average_fitness: fitness,
            computed_frequencies: Some(computed),
            errors_in_cents: Some(cents),
            length_trim: 0.0,
        });
    }

    // Initialize population
    let population = initialize_population(
        params.population_size,
        num_cuts,
        &bounds,
        config.seed_genes.as_deref(),
    );

    // Evaluate initial population
    let mut population = batch_evaluate_population(
        population,
        bar,
        material,
        &target_frequencies,
        params,
        num_cuts,
        penalty_type,
        penalty_weight,
    );

    // Calculate operation counts
    let num_elite = (params.population_size as f64 * params.elitism_percent / 100.0)
        .max(1.0) as usize;
    let num_crossover = (params.population_size as f64 * params.crossover_percent / 100.0) as usize;
    let num_crossover_pairs = (num_crossover + 1) / 2;

    let mut best_ever = clone_individual(get_best_individual(&population));
    let mut generation = 0;

    // Main evolution loop
    while generation < params.max_generations {
        // Check stopping condition
        if let Some(ref stop_fn) = should_stop {
            if stop_fn() {
                break;
            }
        }

        // Check if target error reached
        if best_ever.fitness <= params.target_error {
            break;
        }

        // Create next generation
        let mut next_generation: Vec<Individual> = Vec::with_capacity(params.population_size);

        // 1. Elitism: Keep best individuals unchanged
        let elite = select_elite(&population, num_elite);
        next_generation.extend(elite);

        // 2. Crossover: Select parents and create children
        let mut new_offspring: Vec<Individual> = Vec::new();
        if num_crossover > 0 {
            let mating_pairs =
                select_mating_pairs(&population, num_crossover_pairs, SelectionMethod::Roulette);

            for (parent1, parent2) in mating_pairs {
                let (child1, child2) = heuristic_crossover(&parent1, &parent2, &bounds, num_cuts);
                new_offspring.push(child1);

                if next_generation.len() + new_offspring.len() < params.population_size {
                    new_offspring.push(child2);
                }
            }
        }

        // 3. Mutation: Select individuals and mutate
        let mut sorted_pop = population.clone();
        sorted_pop.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));

        while next_generation.len() + new_offspring.len() < params.population_size {
            let idx = ((sorted_pop.len() as f64
                * (num_elite + num_crossover) as f64 / params.population_size as f64
                * 0.5)
                * (1.0
                    + 0.5
                        * (1.0 - next_generation.len() as f64 / params.population_size as f64)))
            .min(sorted_pop.len() as f64 - 1.0) as usize;

            let parent = &sorted_pop[idx];

            let mutant = if has_length_adjust {
                // Use adaptive mutation if length adjustment is enabled
                let parent_freqs = compute_frequencies_from_genes(
                    &parent.genes,
                    bar,
                    material,
                    1,
                    params.num_elements,
                    num_cuts,
                    params.analysis_mode,
                    params.num_elements_y,
                    params.num_elements_z,
                );
                let f1_error = if !parent_freqs.is_empty() {
                    parent_freqs[0] - target_frequencies[0]
                } else {
                    0.0
                };

                adaptive_length_mutation(
                    parent,
                    params.mutation_strength,
                    &bounds,
                    num_cuts,
                    Some(FrequencyError { f1_error }),
                    0.7,
                )
            } else {
                uniform_mutation(parent, params.mutation_strength, &bounds, num_cuts)
            };

            new_offspring.push(mutant);
        }

        // Batch evaluate all new offspring
        if !new_offspring.is_empty() {
            let evaluated = batch_evaluate_population(
                new_offspring,
                bar,
                material,
                &target_frequencies,
                params,
                num_cuts,
                penalty_type,
                penalty_weight,
            );
            next_generation.extend(evaluated);
        }

        // Update population
        population = next_generation;

        // Update best ever
        let current_best = get_best_individual(&population);
        if current_best.fitness < best_ever.fitness {
            best_ever = clone_individual(current_best);
        }

        generation += 1;

        // Report progress
        if let Some(ref mut progress) = on_progress {
            let stats = calculate_population_stats(&population);
            let computed = compute_frequencies_from_genes(
                &best_ever.genes,
                bar,
                material,
                target_frequencies.len(),
                params.num_elements,
                num_cuts,
                params.analysis_mode,
                params.num_elements_y,
                params.num_elements_z,
            );
            let cents = compute_cents_errors(&computed, &target_frequencies);
            let length_trim = get_length_adjust_from_genes(&best_ever.genes, num_cuts);

            progress(ProgressUpdate {
                generation,
                best_fitness: best_ever.fitness,
                best_individual: clone_individual(&best_ever),
                average_fitness: stats.average_fitness,
                computed_frequencies: Some(computed),
                errors_in_cents: Some(cents),
                length_trim,
            });
        }
    }

    // Build final result
    let length_adjust = get_length_adjust_from_genes(&best_ever.genes, num_cuts);
    let effective_length = bar.length - 2.0 * length_adjust;
    let cuts = genes_to_cuts(&best_ever.genes[..num_cuts * 2]);

    // Evaluate against ORIGINAL targets (not offset-adjusted) for accurate reporting
    let computed_frequencies = compute_frequencies_from_genes(
        &best_ever.genes,
        bar,
        material,
        config.target_frequencies.len(),
        params.num_elements,
        num_cuts,
        params.analysis_mode,
        params.num_elements_y,
        params.num_elements_z,
    );

    let detailed = evaluate_detailed(
        &computed_frequencies,
        &config.target_frequencies,
        &best_ever.genes,
        effective_length,
        bar.h0,
        penalty_type,
        penalty_weight,
        num_cuts,
    );

    OptimizationResult {
        best_individual: best_ever,
        cuts,
        computed_frequencies: detailed.computed_frequencies,
        target_frequencies: config.target_frequencies.clone(),
        tuning_error: detailed.tuning_error,
        max_error_cents: detailed.max_cents_error,
        errors_in_cents: detailed.cents_errors,
        volume_percent: detailed.volume_penalty,
        roughness_percent: detailed.roughness_penalty,
        generations: generation,
        length_trim: length_adjust,
        effective_length,
    }
}

/// Run optimization without callbacks (simpler API).
pub fn run_optimization(config: &EAConfig) -> OptimizationResult {
    run_evolutionary_algorithm::<fn(ProgressUpdate), fn() -> bool>(config, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> EAConfig {
        let bar = BarParameters::new(0.5, 0.03, 0.024, 0.01);
        let material = Material::sapele();
        let target_frequencies = vec![350.0, 700.0, 1400.0, 2100.0];

        let mut params = EAParameters::for_num_cuts(2);
        params.population_size = 10;
        params.max_generations = 5;
        params.num_elements = 50;
        params.analysis_mode = AnalysisMode::Beam2D;

        EAConfig::new(bar, material, target_frequencies, 2).with_params(params)
    }

    #[test]
    fn optimization_runs_without_panic() {
        let config = test_config();
        let result = run_optimization(&config);

        assert!(result.generations > 0);
        assert!(!result.computed_frequencies.is_empty());
        assert!(!result.cuts.is_empty());
    }

    #[test]
    fn optimization_improves_fitness() {
        let config = test_config();

        let mut best_fitnesses = Vec::new();
        let result = run_evolutionary_algorithm(
            &config,
            Some(|update: ProgressUpdate| {
                best_fitnesses.push(update.best_fitness);
            }),
            None::<fn() -> bool>,
        );

        // Fitness should generally decrease (improve)
        if best_fitnesses.len() > 2 {
            let first = best_fitnesses[1]; // Skip generation 0 (uncut bar)
            let last = *best_fitnesses.last().unwrap();
            assert!(
                last <= first * 1.1, // Allow some tolerance
                "Fitness should improve: first={}, last={}",
                first,
                last
            );
        }
    }

    #[test]
    fn progress_callback_receives_updates() {
        let config = test_config();

        let mut updates = Vec::new();
        run_evolutionary_algorithm(
            &config,
            Some(|update: ProgressUpdate| {
                updates.push(update.generation);
            }),
            None::<fn() -> bool>,
        );

        assert!(!updates.is_empty());
        assert_eq!(updates[0], 0); // Should start with generation 0
    }

    #[test]
    fn stop_callback_stops_optimization() {
        let mut config = test_config();
        config.ea_params.max_generations = 100;

        let result = run_evolutionary_algorithm(
            &config,
            None::<fn(ProgressUpdate)>,
            Some(|| true), // Always request stop
        );

        // Should stop early
        assert!(result.generations < config.ea_params.max_generations);
    }
}
