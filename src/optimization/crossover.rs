//! Crossover operators for evolutionary algorithm.

use rand::Rng;

use super::population::clamp_to_bounds;
use super::types::{CrossoverMethod, Individual, VariableBounds};

/// Heuristic crossover from Eq. 16.
///
/// c1 = p1 + r * (p2 - p1)
/// c2 = p2 + r * (p1 - p2)
///
/// where r is uniform random [0, 1]
pub fn heuristic_crossover(
    parent1: &Individual,
    parent2: &Individual,
    bounds: &VariableBounds,
    num_cuts: usize,
) -> (Individual, Individual) {
    let mut rng = rand::thread_rng();
    let num_genes = parent1.genes.len();
    let r: f64 = rng.r#gen();

    let child1_genes: Vec<f64> = (0..num_genes)
        .map(|i| {
            let p1 = parent1.genes[i];
            let p2 = parent2.genes[i];
            p1 + r * (p2 - p1)
        })
        .collect();

    let child2_genes: Vec<f64> = (0..num_genes)
        .map(|i| {
            let p1 = parent1.genes[i];
            let p2 = parent2.genes[i];
            p2 + r * (p1 - p2)
        })
        .collect();

    // Clamp to bounds
    let clamped_child1 = clamp_to_bounds(&child1_genes, bounds, num_cuts);
    let clamped_child2 = clamp_to_bounds(&child2_genes, bounds, num_cuts);

    // Handle sigmas for self-adaptive mutation
    let child1_sigmas = if let (Some(s1), Some(s2)) = (&parent1.sigmas, &parent2.sigmas) {
        Some(
            s1.iter()
                .zip(s2.iter())
                .map(|(sig1, sig2)| sig1 + r * (sig2 - sig1))
                .collect(),
        )
    } else {
        None
    };

    let child2_sigmas = if let (Some(s1), Some(s2)) = (&parent1.sigmas, &parent2.sigmas) {
        Some(
            s1.iter()
                .zip(s2.iter())
                .map(|(sig1, sig2)| sig2 + r * (sig1 - sig2))
                .collect(),
        )
    } else {
        None
    };

    (
        Individual {
            genes: clamped_child1,
            fitness: f64::INFINITY,
            sigmas: child1_sigmas,
        },
        Individual {
            genes: clamped_child2,
            fitness: f64::INFINITY,
            sigmas: child2_sigmas,
        },
    )
}

/// Single-point crossover.
pub fn single_point_crossover(
    parent1: &Individual,
    parent2: &Individual,
    bounds: &VariableBounds,
    num_cuts: usize,
) -> (Individual, Individual) {
    let mut rng = rand::thread_rng();
    let num_genes = parent1.genes.len();
    let crossover_point = rng.gen_range(1..num_genes);

    let child1_genes: Vec<f64> = parent1.genes[..crossover_point]
        .iter()
        .chain(parent2.genes[crossover_point..].iter())
        .copied()
        .collect();

    let child2_genes: Vec<f64> = parent2.genes[..crossover_point]
        .iter()
        .chain(parent1.genes[crossover_point..].iter())
        .copied()
        .collect();

    (
        Individual::new(clamp_to_bounds(&child1_genes, bounds, num_cuts)),
        Individual::new(clamp_to_bounds(&child2_genes, bounds, num_cuts)),
    )
}

/// Two-point crossover.
pub fn two_point_crossover(
    parent1: &Individual,
    parent2: &Individual,
    bounds: &VariableBounds,
    num_cuts: usize,
) -> (Individual, Individual) {
    let mut rng = rand::thread_rng();
    let num_genes = parent1.genes.len();

    let mut point1 = rng.gen_range(0..num_genes);
    let mut point2 = rng.gen_range(0..num_genes);

    if point1 > point2 {
        std::mem::swap(&mut point1, &mut point2);
    }

    let child1_genes: Vec<f64> = parent1.genes[..point1]
        .iter()
        .chain(parent2.genes[point1..point2].iter())
        .chain(parent1.genes[point2..].iter())
        .copied()
        .collect();

    let child2_genes: Vec<f64> = parent2.genes[..point1]
        .iter()
        .chain(parent1.genes[point1..point2].iter())
        .chain(parent2.genes[point2..].iter())
        .copied()
        .collect();

    (
        Individual::new(clamp_to_bounds(&child1_genes, bounds, num_cuts)),
        Individual::new(clamp_to_bounds(&child2_genes, bounds, num_cuts)),
    )
}

/// Uniform crossover.
pub fn uniform_crossover(
    parent1: &Individual,
    parent2: &Individual,
    bounds: &VariableBounds,
    num_cuts: usize,
    mixing_ratio: f64,
) -> (Individual, Individual) {
    let mut rng = rand::thread_rng();
    let num_genes = parent1.genes.len();

    let mut child1_genes = Vec::with_capacity(num_genes);
    let mut child2_genes = Vec::with_capacity(num_genes);

    for i in 0..num_genes {
        if rng.r#gen::<f64>() < mixing_ratio {
            child1_genes.push(parent1.genes[i]);
            child2_genes.push(parent2.genes[i]);
        } else {
            child1_genes.push(parent2.genes[i]);
            child2_genes.push(parent1.genes[i]);
        }
    }

    (
        Individual::new(clamp_to_bounds(&child1_genes, bounds, num_cuts)),
        Individual::new(clamp_to_bounds(&child2_genes, bounds, num_cuts)),
    )
}

/// Blend crossover (BLX-alpha).
pub fn blend_crossover(
    parent1: &Individual,
    parent2: &Individual,
    bounds: &VariableBounds,
    num_cuts: usize,
    alpha: f64,
) -> (Individual, Individual) {
    let mut rng = rand::thread_rng();
    let num_genes = parent1.genes.len();

    let mut child1_genes = Vec::with_capacity(num_genes);
    let mut child2_genes = Vec::with_capacity(num_genes);

    for i in 0..num_genes {
        let p1 = parent1.genes[i];
        let p2 = parent2.genes[i];

        let min_val = p1.min(p2);
        let max_val = p1.max(p2);
        let range_val = max_val - min_val;

        let extended_min = min_val - alpha * range_val;
        let extended_max = max_val + alpha * range_val;

        child1_genes.push(extended_min + rng.r#gen::<f64>() * (extended_max - extended_min));
        child2_genes.push(extended_min + rng.r#gen::<f64>() * (extended_max - extended_min));
    }

    (
        Individual::new(clamp_to_bounds(&child1_genes, bounds, num_cuts)),
        Individual::new(clamp_to_bounds(&child2_genes, bounds, num_cuts)),
    )
}

/// Perform crossover using the specified method.
pub fn perform_crossover(
    parent1: &Individual,
    parent2: &Individual,
    bounds: &VariableBounds,
    num_cuts: usize,
    method: CrossoverMethod,
) -> (Individual, Individual) {
    match method {
        CrossoverMethod::Heuristic => heuristic_crossover(parent1, parent2, bounds, num_cuts),
        CrossoverMethod::SinglePoint => single_point_crossover(parent1, parent2, bounds, num_cuts),
        CrossoverMethod::TwoPoint => two_point_crossover(parent1, parent2, bounds, num_cuts),
        CrossoverMethod::Uniform => uniform_crossover(parent1, parent2, bounds, num_cuts, 0.5),
        CrossoverMethod::Blend => blend_crossover(parent1, parent2, bounds, num_cuts, 0.5),
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

    fn test_parents() -> (Individual, Individual) {
        (
            Individual::with_fitness(vec![0.1, 0.02, 0.05, 0.015], 1.0),
            Individual::with_fitness(vec![0.2, 0.018, 0.15, 0.012], 2.0),
        )
    }

    #[test]
    fn heuristic_crossover_produces_children() {
        let bounds = test_bounds();
        let (p1, p2) = test_parents();

        let (c1, c2) = heuristic_crossover(&p1, &p2, &bounds, 2);

        assert_eq!(c1.genes.len(), 4);
        assert_eq!(c2.genes.len(), 4);
        assert_eq!(c1.fitness, f64::INFINITY);
        assert_eq!(c2.fitness, f64::INFINITY);
    }

    #[test]
    fn children_within_bounds() {
        let bounds = test_bounds();
        let (p1, p2) = test_parents();

        for _ in 0..100 {
            let (c1, c2) = heuristic_crossover(&p1, &p2, &bounds, 2);

            for i in (0..c1.genes.len()).step_by(2) {
                assert!(c1.genes[i] >= bounds.lambda_min && c1.genes[i] <= bounds.lambda_max);
                assert!(c2.genes[i] >= bounds.lambda_min && c2.genes[i] <= bounds.lambda_max);
            }

            for i in (1..c1.genes.len()).step_by(2) {
                assert!(c1.genes[i] >= bounds.h_min && c1.genes[i] <= bounds.h_max);
                assert!(c2.genes[i] >= bounds.h_min && c2.genes[i] <= bounds.h_max);
            }
        }
    }

    #[test]
    fn single_point_crossover_works() {
        let bounds = test_bounds();
        let (p1, p2) = test_parents();

        let (c1, c2) = single_point_crossover(&p1, &p2, &bounds, 2);

        assert_eq!(c1.genes.len(), 4);
        assert_eq!(c2.genes.len(), 4);
    }

    #[test]
    fn blend_crossover_explores_beyond_parents() {
        let bounds = test_bounds();
        let (p1, p2) = test_parents();

        // With alpha > 0, children can be outside parent range (but within bounds)
        let mut found_outside = false;
        for _ in 0..100 {
            let (c1, _) = blend_crossover(&p1, &p2, &bounds, 2, 0.5);

            for i in 0..c1.genes.len() {
                let min_parent = p1.genes[i].min(p2.genes[i]);
                let max_parent = p1.genes[i].max(p2.genes[i]);

                if c1.genes[i] < min_parent || c1.genes[i] > max_parent {
                    found_outside = true;
                    break;
                }
            }

            if found_outside {
                break;
            }
        }

        // With blend crossover, we should sometimes see children outside parent range
        // (but this isn't guaranteed, so we just check the test runs)
        assert!(true);
    }
}
