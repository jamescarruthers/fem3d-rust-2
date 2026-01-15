//! Selection operators for evolutionary algorithm.

use rand::Rng;

use super::population::clone_individual;
use super::types::{Individual, SelectionMethod};

/// Roulette wheel selection (fitness proportional) from Eq. 15.
///
/// p_i = (1/e_i) / sum(1/e_j)
///
/// Lower fitness = higher probability of selection.
pub fn roulette_selection(population: &[Individual], num_selections: usize) -> Vec<Individual> {
    let mut rng = rand::thread_rng();

    // Filter out individuals with invalid fitness
    let valid_population: Vec<&Individual> = population
        .iter()
        .filter(|ind| ind.fitness.is_finite() && ind.fitness > 0.0)
        .collect();

    if valid_population.is_empty() {
        // Fallback: return random selection from original population
        return (0..num_selections)
            .map(|_| clone_individual(&population[rng.gen_range(0..population.len())]))
            .collect();
    }

    // Calculate selection probabilities (inverse of fitness)
    let inverse_fitnesses: Vec<f64> = valid_population.iter().map(|ind| 1.0 / ind.fitness).collect();
    let sum_inverse: f64 = inverse_fitnesses.iter().sum();
    let probabilities: Vec<f64> = inverse_fitnesses.iter().map(|inv| inv / sum_inverse).collect();

    // Calculate cumulative probabilities
    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut cum_sum = 0.0;
    for prob in &probabilities {
        cum_sum += prob;
        cumulative.push(cum_sum);
    }

    // Select individuals using roulette wheel
    let mut selected = Vec::with_capacity(num_selections);
    for _ in 0..num_selections {
        let r: f64 = rng.r#gen();

        let mut selected_index = 0;
        for (j, &cum) in cumulative.iter().enumerate() {
            if r <= cum {
                selected_index = j;
                break;
            }
        }

        selected.push(clone_individual(valid_population[selected_index]));
    }

    selected
}

/// Tournament selection.
pub fn tournament_selection(
    population: &[Individual],
    num_selections: usize,
    tournament_size: usize,
) -> Vec<Individual> {
    let mut rng = rand::thread_rng();
    let mut selected = Vec::with_capacity(num_selections);

    for _ in 0..num_selections {
        // Pick random individuals for tournament
        let tournament: Vec<&Individual> = (0..tournament_size)
            .map(|_| &population[rng.gen_range(0..population.len())])
            .collect();

        // Select the best from tournament
        let winner = tournament
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        selected.push(clone_individual(winner));
    }

    selected
}

/// Rank-based selection.
pub fn rank_selection(
    population: &[Individual],
    num_selections: usize,
    selection_pressure: f64,
) -> Vec<Individual> {
    let mut rng = rand::thread_rng();

    // Sort by fitness (ascending - best first)
    let mut sorted: Vec<&Individual> = population.iter().collect();
    sorted.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len() as f64;

    // Calculate rank-based probabilities
    let probabilities: Vec<f64> = (0..sorted.len())
        .map(|i| {
            let rank = i as f64;
            (2.0 - selection_pressure) / n
                + 2.0 * (selection_pressure - 1.0) * (n - 1.0 - rank) / (n * (n - 1.0))
        })
        .collect();

    // Normalize probabilities
    let sum_prob: f64 = probabilities.iter().sum();
    let normalized: Vec<f64> = probabilities.iter().map(|p| p / sum_prob).collect();

    // Calculate cumulative probabilities
    let mut cumulative = Vec::with_capacity(normalized.len());
    let mut cum_sum = 0.0;
    for prob in &normalized {
        cum_sum += prob;
        cumulative.push(cum_sum);
    }

    // Select individuals
    let mut selected = Vec::with_capacity(num_selections);
    for _ in 0..num_selections {
        let r: f64 = rng.r#gen();

        let mut selected_index = 0;
        for (j, &cum) in cumulative.iter().enumerate() {
            if r <= cum {
                selected_index = j;
                break;
            }
        }

        selected.push(clone_individual(sorted[selected_index]));
    }

    selected
}

/// Select parents using the specified method.
pub fn select_parents(
    population: &[Individual],
    num_selections: usize,
    method: SelectionMethod,
) -> Vec<Individual> {
    match method {
        SelectionMethod::Roulette => roulette_selection(population, num_selections),
        SelectionMethod::Tournament => tournament_selection(population, num_selections, 3),
        SelectionMethod::Rank => rank_selection(population, num_selections, 1.5),
    }
}

/// Select mating pairs for crossover.
pub fn select_mating_pairs(
    population: &[Individual],
    num_pairs: usize,
    method: SelectionMethod,
) -> Vec<(Individual, Individual)> {
    let mut rng = rand::thread_rng();
    let mut pairs = Vec::with_capacity(num_pairs);

    for _ in 0..num_pairs {
        let parents = select_parents(population, 2, method);
        let parent1 = parents.into_iter().next().unwrap();

        // Keep selecting second parent until it's different
        let mut parent2 = select_parents(population, 1, method).into_iter().next().unwrap();
        let mut attempts = 0;

        while parent2.genes == parent1.genes && attempts < 10 {
            parent2 = select_parents(population, 1, method).into_iter().next().unwrap();
            attempts += 1;
        }

        // If we couldn't find a different parent, just use a random one
        if parent2.genes == parent1.genes {
            let idx = rng.gen_range(0..population.len());
            parent2 = clone_individual(&population[idx]);
        }

        pairs.push((parent1, parent2));
    }

    pairs
}

/// Elitism: select the best individuals to pass unchanged to next generation.
pub fn select_elite(population: &[Individual], num_elite: usize) -> Vec<Individual> {
    let mut sorted: Vec<&Individual> = population.iter().collect();
    sorted.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(std::cmp::Ordering::Equal));

    sorted.into_iter().take(num_elite).map(clone_individual).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_population() -> Vec<Individual> {
        vec![
            Individual::with_fitness(vec![0.1, 0.02], 1.0),
            Individual::with_fitness(vec![0.15, 0.018], 2.0),
            Individual::with_fitness(vec![0.2, 0.015], 3.0),
            Individual::with_fitness(vec![0.12, 0.019], 1.5),
            Individual::with_fitness(vec![0.18, 0.016], 2.5),
        ]
    }

    #[test]
    fn roulette_selection_returns_correct_count() {
        let pop = test_population();
        let selected = roulette_selection(&pop, 3);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn tournament_selection_returns_correct_count() {
        let pop = test_population();
        let selected = tournament_selection(&pop, 3, 2);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn rank_selection_returns_correct_count() {
        let pop = test_population();
        let selected = rank_selection(&pop, 3, 1.5);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn elite_selection_returns_best() {
        let pop = test_population();
        let elite = select_elite(&pop, 2);

        assert_eq!(elite.len(), 2);
        assert_eq!(elite[0].fitness, 1.0);
        assert_eq!(elite[1].fitness, 1.5);
    }

    #[test]
    fn mating_pairs_returns_correct_count() {
        let pop = test_population();
        let pairs = select_mating_pairs(&pop, 3, SelectionMethod::Roulette);

        assert_eq!(pairs.len(), 3);
        for (p1, p2) in &pairs {
            assert!(!p1.genes.is_empty());
            assert!(!p2.genes.is_empty());
        }
    }
}
