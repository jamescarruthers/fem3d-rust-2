//! Population management for evolutionary algorithm.

use rand::Rng;

use super::types::{Individual, PopulationStats, VariableBounds};

/// Create a random individual within bounds.
pub fn create_random_individual(num_cuts: usize, bounds: &VariableBounds) -> Individual {
    let mut rng = rand::thread_rng();
    let mut genes = Vec::with_capacity(num_cuts * 2 + 1);

    let min_width = bounds.min_cut_width;
    let max_width = bounds.max_cut_width;

    // Generate lambdas with spacing constraints
    let mut lambdas = Vec::with_capacity(num_cuts);

    if num_cuts == 1 {
        let mut lambda = bounds.lambda_min + rng.r#gen::<f64>() * (bounds.lambda_max - bounds.lambda_min);
        if max_width > 0.0 {
            lambda = lambda.min(max_width);
        }
        lambdas.push(lambda);
    } else {
        let mut current_max = bounds.lambda_max;

        for i in 0..num_cuts {
            let remaining_cuts = num_cuts - i - 1;
            let reserved_space = remaining_cuts as f64 * min_width;
            let available_max = current_max - reserved_space;

            let mut cut_min = bounds.lambda_min + reserved_space;
            let mut cut_max = available_max;

            if max_width > 0.0 && i > 0 {
                let prev_lambda = lambdas[i - 1];
                cut_min = cut_min.max(prev_lambda - max_width);
            }

            if cut_max < cut_min {
                cut_max = cut_min;
            }

            let lambda = cut_min + rng.r#gen::<f64>() * (cut_max - cut_min);
            lambdas.push(lambda);

            current_max = lambda - min_width;
        }
    }

    // Sort lambdas descending (outermost first)
    lambdas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Build genes array with lambdas and random heights
    for lambda in lambdas {
        genes.push(lambda);
        let h = bounds.h_min + rng.r#gen::<f64>() * (bounds.h_max - bounds.h_min);
        genes.push(h);
    }

    // Add length adjustment gene if enabled
    if bounds.has_length_adjust() {
        let min_val = -bounds.max_length_extend;
        let max_val = bounds.max_length_trim;
        let length_adjust = min_val + rng.r#gen::<f64>() * (max_val - min_val);
        genes.push(length_adjust);
    }

    Individual::new(genes)
}

/// Create an "uncut bar" individual - baseline with no modifications.
pub fn create_uncut_bar_individual(num_cuts: usize, bounds: &VariableBounds, h0: f64) -> Individual {
    let mut genes = Vec::with_capacity(num_cuts * 2 + 1);

    // All cuts have lambda=0 and h=h0 (no material removed)
    for _ in 0..num_cuts {
        genes.push(0.0); // lambda = 0 (no cut extent)
        genes.push(h0); // h = h0 (full original thickness)
    }

    // Add length adjustment gene if enabled (set to 0 = no trim/extend)
    if bounds.has_length_adjust() {
        genes.push(0.0);
    }

    Individual::new(genes)
}

/// Initialize a population of random individuals.
pub fn initialize_population(
    population_size: usize,
    num_cuts: usize,
    bounds: &VariableBounds,
    seed_genes: Option<&[f64]>,
) -> Vec<Individual> {
    let mut population = Vec::with_capacity(population_size);

    // If seed genes provided, create an individual from them
    if let Some(seed) = seed_genes {
        if !seed.is_empty() {
            let clamped = clamp_to_bounds(seed, bounds, num_cuts);
            population.push(Individual::new(clamped));

            // Add some mutated variants of the seed for diversity
            let mut rng = rand::thread_rng();
            let num_variants = (population_size as f64 * 0.2).min(10.0) as usize;

            for _ in 0..num_variants {
                if population.len() >= population_size {
                    break;
                }
                let variant_genes: Vec<f64> = seed
                    .iter()
                    .map(|&g| g * (0.95 + rng.r#gen::<f64>() * 0.1))
                    .collect();
                let clamped_variant = clamp_to_bounds(&variant_genes, bounds, num_cuts);
                population.push(Individual::new(clamped_variant));
            }
        }
    }

    // Fill remaining slots with random individuals
    while population.len() < population_size {
        population.push(create_random_individual(num_cuts, bounds));
    }

    population
}

/// Get the best individual from a population (lowest fitness).
pub fn get_best_individual(population: &[Individual]) -> &Individual {
    population
        .iter()
        .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
        .expect("Population should not be empty")
}

/// Get the N best individuals from a population.
pub fn get_top_individuals(population: &[Individual], n: usize) -> Vec<&Individual> {
    let mut sorted: Vec<_> = population.iter().collect();
    sorted.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));
    sorted.into_iter().take(n).collect()
}

/// Calculate population statistics.
pub fn calculate_population_stats(population: &[Individual]) -> PopulationStats {
    let fitnesses: Vec<f64> = population
        .iter()
        .filter(|ind| ind.fitness.is_finite())
        .map(|ind| ind.fitness)
        .collect();

    if fitnesses.is_empty() {
        return PopulationStats {
            best_fitness: f64::INFINITY,
            worst_fitness: f64::INFINITY,
            average_fitness: f64::INFINITY,
            median_fitness: f64::INFINITY,
            standard_deviation: 0.0,
        };
    }

    let mut sorted = fitnesses.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let average = sorted.iter().sum::<f64>() / sorted.len() as f64;

    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let variance = sorted.iter().map(|f| (f - average).powi(2)).sum::<f64>() / sorted.len() as f64;
    let std_dev = variance.sqrt();

    PopulationStats {
        best_fitness: sorted[0],
        worst_fitness: *sorted.last().unwrap(),
        average_fitness: average,
        median_fitness: median,
        standard_deviation: std_dev,
    }
}

/// Clone an individual.
pub fn clone_individual(individual: &Individual) -> Individual {
    Individual {
        genes: individual.genes.clone(),
        fitness: individual.fitness,
        sigmas: individual.sigmas.clone(),
    }
}

/// Clamp genes to bounds and enforce constraints.
pub fn clamp_to_bounds(genes: &[f64], bounds: &VariableBounds, num_cuts: usize) -> Vec<f64> {
    let mut clamped = genes.to_vec();
    let min_width = bounds.min_cut_width;
    let max_width = bounds.max_cut_width;

    let has_length_adjust = bounds.has_length_adjust();
    let cut_genes_length = num_cuts * 2;

    // First pass: clamp individual cut values (lambda and h)
    for i in (0..cut_genes_length).step_by(2) {
        if i < clamped.len() {
            // Clamp lambda
            clamped[i] = clamped[i].clamp(bounds.lambda_min, bounds.lambda_max);
        }
        if i + 1 < clamped.len() {
            // Clamp height
            clamped[i + 1] = clamped[i + 1].clamp(bounds.h_min, bounds.h_max);
        }
    }

    // Clamp length adjustment gene if present
    if has_length_adjust && clamped.len() > cut_genes_length {
        let min_val = -bounds.max_length_extend;
        let max_val = bounds.max_length_trim;
        clamped[cut_genes_length] = clamped[cut_genes_length].clamp(min_val, max_val);
    }

    // Second pass: enforce min/max spacing between lambdas
    if (min_width > 0.0 || max_width > 0.0) && num_cuts >= 1 {
        // Extract lambdas with their original indices
        let mut lambdas_with_idx: Vec<(f64, usize)> = (0..num_cuts)
            .filter_map(|i| {
                let idx = i * 2;
                if idx < clamped.len() {
                    Some((clamped[idx], idx))
                } else {
                    None
                }
            })
            .collect();

        // Sort by lambda descending (outermost first)
        lambdas_with_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Enforce constraints from outside in
        for i in 1..lambdas_with_idx.len() {
            let outer_lambda = lambdas_with_idx[i - 1].0;
            let mut inner_lambda = lambdas_with_idx[i].0;

            // Enforce minimum spacing
            if min_width > 0.0 {
                let max_allowed = outer_lambda - min_width;
                if inner_lambda > max_allowed {
                    inner_lambda = bounds.lambda_min.max(max_allowed);
                }
            }

            // Enforce maximum spacing
            if max_width > 0.0 {
                let min_allowed = outer_lambda - max_width;
                if inner_lambda < min_allowed {
                    inner_lambda = bounds.lambda_min.max(min_allowed);
                }
            }

            lambdas_with_idx[i].0 = inner_lambda;
        }

        // Write back to clamped array
        for (lambda_val, idx) in lambdas_with_idx {
            clamped[idx] = lambda_val;
        }
    }

    clamped
}

/// Extract length adjustment from genes array.
pub fn get_length_adjust_from_genes(genes: &[f64], num_cuts: usize) -> f64 {
    let expected_cut_genes = num_cuts * 2;
    if genes.len() > expected_cut_genes {
        genes[expected_cut_genes]
    } else {
        0.0
    }
}

/// Calculate diversity measure for population.
pub fn calculate_diversity(population: &[Individual]) -> f64 {
    if population.len() < 2 {
        return 0.0;
    }

    let num_genes = population[0].genes.len();
    let mut total_variance = 0.0;

    for g in 0..num_genes {
        let gene_values: Vec<f64> = population.iter().map(|ind| ind.genes[g]).collect();
        let mean = gene_values.iter().sum::<f64>() / gene_values.len() as f64;
        let variance = gene_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / gene_values.len() as f64;
        total_variance += variance;
    }

    (total_variance / num_genes as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bounds() -> VariableBounds {
        VariableBounds {
            lambda_min: 0.0,
            lambda_max: 0.25,
            h_min: 0.01,
            h_max: 0.024,
            min_cut_width: 0.0,
            max_cut_width: 0.0,
            min_cut_depth: 0.0,
            max_cut_depth: 0.0,
            max_length_trim: 0.0,
            max_length_extend: 0.0,
        }
    }

    #[test]
    fn random_individual_has_correct_gene_count() {
        let bounds = test_bounds();
        let ind = create_random_individual(2, &bounds);
        assert_eq!(ind.genes.len(), 4); // 2 cuts * 2 genes each
    }

    #[test]
    fn random_individual_genes_within_bounds() {
        let bounds = test_bounds();
        for _ in 0..100 {
            let ind = create_random_individual(2, &bounds);

            // Check lambdas (even indices)
            for i in (0..ind.genes.len()).step_by(2) {
                assert!(
                    ind.genes[i] >= bounds.lambda_min && ind.genes[i] <= bounds.lambda_max,
                    "Lambda {} out of bounds: {}",
                    i / 2,
                    ind.genes[i]
                );
            }

            // Check heights (odd indices)
            for i in (1..ind.genes.len()).step_by(2) {
                assert!(
                    ind.genes[i] >= bounds.h_min && ind.genes[i] <= bounds.h_max,
                    "Height {} out of bounds: {}",
                    i / 2,
                    ind.genes[i]
                );
            }
        }
    }

    #[test]
    fn uncut_bar_has_zero_lambdas() {
        let bounds = test_bounds();
        let h0 = 0.024;
        let ind = create_uncut_bar_individual(2, &bounds, h0);

        assert_eq!(ind.genes.len(), 4);
        assert_eq!(ind.genes[0], 0.0); // lambda_1
        assert_eq!(ind.genes[1], h0); // h_1
        assert_eq!(ind.genes[2], 0.0); // lambda_2
        assert_eq!(ind.genes[3], h0); // h_2
    }

    #[test]
    fn population_initialization() {
        let bounds = test_bounds();
        let pop = initialize_population(20, 2, &bounds, None);

        assert_eq!(pop.len(), 20);
        for ind in &pop {
            assert_eq!(ind.genes.len(), 4);
        }
    }

    #[test]
    fn clamping_works() {
        let bounds = test_bounds();
        let genes = vec![0.5, 0.03, -0.1, 0.005]; // Out of bounds
        let clamped = clamp_to_bounds(&genes, &bounds, 2);

        assert_eq!(clamped[0], bounds.lambda_max);
        assert_eq!(clamped[1], bounds.h_max);
        assert_eq!(clamped[2], bounds.lambda_min);
        assert_eq!(clamped[3], bounds.h_min);
    }

    #[test]
    fn population_stats_calculation() {
        let pop = vec![
            Individual::with_fitness(vec![0.1, 0.02], 1.0),
            Individual::with_fitness(vec![0.15, 0.018], 2.0),
            Individual::with_fitness(vec![0.2, 0.015], 3.0),
        ];

        let stats = calculate_population_stats(&pop);

        assert_eq!(stats.best_fitness, 1.0);
        assert_eq!(stats.worst_fitness, 3.0);
        assert_eq!(stats.average_fitness, 2.0);
        assert_eq!(stats.median_fitness, 2.0);
    }
}
