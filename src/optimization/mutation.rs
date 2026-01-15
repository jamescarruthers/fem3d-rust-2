//! Mutation operators for evolutionary algorithm.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use super::population::clamp_to_bounds;
use super::types::{Individual, MutationMethod, VariableBounds};

/// Generate a random number from standard Gaussian distribution.
fn gaussian_random() -> f64 {
    StandardNormal.sample(&mut rand::thread_rng())
}

/// Uniform random mutation.
///
/// As described in paper Section 3.3:
/// 1. Randomly select number of genes to mutate (1 to 2N)
/// 2. Randomly select which genes to mutate
/// 3. Mutate with: c = p + sigma * r, where r is uniform [-1, 1]
pub fn uniform_mutation(
    individual: &Individual,
    sigma: f64,
    bounds: &VariableBounds,
    num_cuts: usize,
) -> Individual {
    let mut rng = rand::thread_rng();
    let mut genes = individual.genes.clone();
    let num_genes = genes.len();

    // Step 1: Random number of genes to mutate
    let num_mutate = rng.gen_range(1..=num_genes);

    // Step 2: Select which genes to mutate
    let mut indices: Vec<usize> = (0..num_genes).collect();
    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    let indices_to_mutate: Vec<usize> = indices.into_iter().take(num_mutate).collect();

    let has_length_adjust = bounds.has_length_adjust();
    let cut_genes_count = num_cuts * 2;

    // Step 3: Mutate selected genes
    for idx in indices_to_mutate {
        let range_val = if has_length_adjust && idx == cut_genes_count {
            // Length adjustment gene
            bounds.max_length_trim + bounds.max_length_extend
        } else {
            // Cut genes: alternating lambda and h
            let is_lambda = idx % 2 == 0;
            if is_lambda {
                bounds.lambda_max - bounds.lambda_min
            } else {
                bounds.h_max - bounds.h_min
            }
        };

        // Random mutation: r is uniform [-1, 1]
        let r: f64 = rng.r#gen::<f64>() * 2.0 - 1.0;
        genes[idx] += sigma * range_val * r;
    }

    // Clamp to bounds
    let mutated_genes = clamp_to_bounds(&genes, bounds, num_cuts);

    Individual {
        genes: mutated_genes,
        fitness: f64::INFINITY,
        sigmas: individual.sigmas.clone(),
    }
}

/// Frequency error info for adaptive length mutation.
#[derive(Debug, Clone, Copy)]
pub struct FrequencyError {
    /// f1_computed - f1_target (positive = too high, negative = too low)
    pub f1_error: f64,
}

/// Adaptive mutation with gradient-aware length adjustment.
///
/// When mutating the length gene, biases the direction based on f1 error:
/// - If f1 is too high (positive error), bias toward extending (negative length adjust)
/// - If f1 is too low (negative error), bias toward trimming (positive length adjust)
///
/// Physics: f1 is proportional to 1/L², so shorter bar = higher frequency.
pub fn adaptive_length_mutation(
    individual: &Individual,
    sigma: f64,
    bounds: &VariableBounds,
    num_cuts: usize,
    freq_error: Option<FrequencyError>,
    adaptive_bias: f64,
) -> Individual {
    let mut rng = rand::thread_rng();
    let mut genes = individual.genes.clone();
    let num_genes = genes.len();

    let num_mutate = rng.gen_range(1..=num_genes);

    let mut indices: Vec<usize> = (0..num_genes).collect();
    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }
    let indices_to_mutate: Vec<usize> = indices.into_iter().take(num_mutate).collect();

    let has_length_adjust = bounds.has_length_adjust();
    let cut_genes_count = num_cuts * 2;
    let length_gene_idx = cut_genes_count;

    for idx in indices_to_mutate {
        if has_length_adjust && idx == length_gene_idx {
            // Length adjustment gene - use adaptive mutation
            let range_val = bounds.max_length_trim + bounds.max_length_extend;

            if let Some(err) = freq_error {
                if err.f1_error.abs() > 0.001 {
                    // Use gradient-aware mutation for length gene
                    // f1_error > 0 means f1 is too high, need to extend (negative length adjust)
                    // f1_error < 0 means f1 is too low, need to trim (positive length adjust)
                    let desired_direction = if err.f1_error < 0.0 { 1.0 } else { -1.0 };

                    let r: f64 = if rng.r#gen::<f64>() < adaptive_bias {
                        // Biased: move in the desired direction
                        desired_direction * rng.r#gen::<f64>()
                    } else {
                        // Random exploration
                        rng.r#gen::<f64>() * 2.0 - 1.0
                    };

                    genes[idx] += sigma * range_val * r;
                    continue;
                }
            }

            // No frequency error info, use standard random mutation
            let r: f64 = rng.r#gen::<f64>() * 2.0 - 1.0;
            genes[idx] += sigma * range_val * r;
        } else {
            // Cut genes - standard random mutation
            let is_lambda = idx % 2 == 0;
            let range_val = if is_lambda {
                bounds.lambda_max - bounds.lambda_min
            } else {
                bounds.h_max - bounds.h_min
            };

            let r: f64 = rng.r#gen::<f64>() * 2.0 - 1.0;
            genes[idx] += sigma * range_val * r;
        }
    }

    let mutated_genes = clamp_to_bounds(&genes, bounds, num_cuts);

    Individual {
        genes: mutated_genes,
        fitness: f64::INFINITY,
        sigmas: individual.sigmas.clone(),
    }
}

/// Self-adaptive Gaussian mutation from Eq. 17-18.
///
/// sigma_k^{t+1} = sigma_k^t * exp(tau1 * z1 + tau2 * z2)
/// c_k = p_k + sigma_k^{t+1} * z3
///
/// where:
/// - tau1 = 1 / sqrt(2 * sqrt(2n))
/// - tau2 = 1 / sqrt(4n)
/// - z1, z2, z3 are Gaussian random numbers with standard deviation phi
pub fn gaussian_self_adaptive_mutation(
    individual: &Individual,
    phi: f64,
    bounds: &VariableBounds,
    num_cuts: usize,
) -> Individual {
    let mut genes = individual.genes.clone();
    let num_genes = genes.len();
    let n = num_genes as f64;

    // Initialize sigmas if not present
    let mut sigmas = individual.sigmas.clone().unwrap_or_else(|| vec![0.2; num_genes]);

    // Learning rates from Eq. 18
    let tau1 = 1.0 / (2.0 * (2.0 * n).sqrt()).sqrt();
    let tau2 = 1.0 / (4.0 * n).sqrt();

    // Common random factor (same for all genes in this mutation)
    let z1 = gaussian_random() * phi;

    let has_length_adjust = bounds.has_length_adjust();
    let cut_genes_count = num_cuts * 2;

    for k in 0..num_genes {
        // Gene-specific random factors
        let z2 = gaussian_random() * phi;
        let z3 = gaussian_random();

        // Update sigma (Eq. 17, first line)
        sigmas[k] = sigmas[k] * (tau1 * z1 + tau2 * z2).exp();

        // Clamp sigma to reasonable range
        sigmas[k] = sigmas[k].clamp(0.001, 1.0);

        // Mutate gene (Eq. 17, second line)
        let range_val = if has_length_adjust && k == cut_genes_count {
            bounds.max_length_trim + bounds.max_length_extend
        } else {
            let is_lambda = k % 2 == 0;
            if is_lambda {
                bounds.lambda_max - bounds.lambda_min
            } else {
                bounds.h_max - bounds.h_min
            }
        };

        genes[k] += sigmas[k] * range_val * z3;
    }

    let mutated_genes = clamp_to_bounds(&genes, bounds, num_cuts);

    Individual {
        genes: mutated_genes,
        fitness: f64::INFINITY,
        sigmas: Some(sigmas),
    }
}

/// Polynomial mutation.
///
/// Non-uniform mutation that can produce values close to parents or at bounds.
pub fn polynomial_mutation(
    individual: &Individual,
    mutation_prob: f64,
    eta: f64,
    bounds: &VariableBounds,
    num_cuts: usize,
) -> Individual {
    let mut rng = rand::thread_rng();
    let mut genes = individual.genes.clone();
    let num_genes = genes.len();

    let has_length_adjust = bounds.has_length_adjust();
    let cut_genes_count = num_cuts * 2;

    for i in 0..num_genes {
        if rng.r#gen::<f64>() > mutation_prob {
            continue;
        }

        let (min_val, max_val) = if has_length_adjust && i == cut_genes_count {
            (-bounds.max_length_extend, bounds.max_length_trim)
        } else {
            let is_lambda = i % 2 == 0;
            if is_lambda {
                (bounds.lambda_min, bounds.lambda_max)
            } else {
                (bounds.h_min, bounds.h_max)
            }
        };

        let x = genes[i];
        let range = max_val - min_val;

        if range.abs() < 1e-10 {
            continue;
        }

        let delta1 = (x - min_val) / range;
        let delta2 = (max_val - x) / range;

        let r: f64 = rng.r#gen();

        let delta = if r < 0.5 {
            let xy = 1.0 - delta1;
            let val = 2.0 * r + (1.0 - 2.0 * r) * xy.powf(eta + 1.0);
            val.powf(1.0 / (eta + 1.0)) - 1.0
        } else {
            let xy = 1.0 - delta2;
            let val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * xy.powf(eta + 1.0);
            1.0 - val.powf(1.0 / (eta + 1.0))
        };

        genes[i] = x + delta * range;
    }

    Individual {
        genes: clamp_to_bounds(&genes, bounds, num_cuts),
        fitness: f64::INFINITY,
        sigmas: individual.sigmas.clone(),
    }
}

/// Perform mutation using the specified method.
pub fn perform_mutation(
    individual: &Individual,
    bounds: &VariableBounds,
    num_cuts: usize,
    method: MutationMethod,
    sigma: f64,
) -> Individual {
    match method {
        MutationMethod::Uniform => uniform_mutation(individual, sigma, bounds, num_cuts),
        MutationMethod::GaussianAdaptive => {
            gaussian_self_adaptive_mutation(individual, sigma, bounds, num_cuts)
        }
        MutationMethod::Polynomial => polynomial_mutation(individual, 0.1, 20.0, bounds, num_cuts),
    }
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
    fn uniform_mutation_changes_genes() {
        let bounds = test_bounds();
        let ind = Individual::with_fitness(vec![0.1, 0.02, 0.05, 0.015], 1.0);

        let mut changed = false;
        for _ in 0..100 {
            let mutant = uniform_mutation(&ind, 0.1, &bounds, 2);

            if mutant.genes != ind.genes {
                changed = true;
                break;
            }
        }

        assert!(changed, "Mutation should change genes");
    }

    #[test]
    fn mutant_within_bounds() {
        let bounds = test_bounds();
        let ind = Individual::with_fitness(vec![0.1, 0.02, 0.05, 0.015], 1.0);

        for _ in 0..100 {
            let mutant = uniform_mutation(&ind, 0.5, &bounds, 2);

            for i in (0..mutant.genes.len()).step_by(2) {
                assert!(
                    mutant.genes[i] >= bounds.lambda_min && mutant.genes[i] <= bounds.lambda_max,
                    "Lambda out of bounds: {}",
                    mutant.genes[i]
                );
            }

            for i in (1..mutant.genes.len()).step_by(2) {
                assert!(
                    mutant.genes[i] >= bounds.h_min && mutant.genes[i] <= bounds.h_max,
                    "Height out of bounds: {}",
                    mutant.genes[i]
                );
            }
        }
    }

    #[test]
    fn gaussian_mutation_updates_sigmas() {
        let bounds = test_bounds();
        let ind = Individual::with_sigmas(vec![0.1, 0.02, 0.05, 0.015], vec![0.2, 0.2, 0.2, 0.2]);

        let mutant = gaussian_self_adaptive_mutation(&ind, 0.1, &bounds, 2);

        assert!(mutant.sigmas.is_some());
        let sigmas = mutant.sigmas.unwrap();
        assert_eq!(sigmas.len(), 4);

        // Sigmas should be within [0.001, 1.0]
        for s in sigmas {
            assert!(s >= 0.001 && s <= 1.0);
        }
    }

    #[test]
    fn polynomial_mutation_works() {
        let bounds = test_bounds();
        let ind = Individual::with_fitness(vec![0.1, 0.02, 0.05, 0.015], 1.0);

        let mutant = polynomial_mutation(&ind, 1.0, 20.0, &bounds, 2);

        assert_eq!(mutant.genes.len(), 4);
        assert!(mutant.fitness.is_infinite());
    }
}
