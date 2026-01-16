//! Sampling methods for optimization initialization.
//!
//! Provides Latin Hypercube Sampling (LHS) and other space-filling designs
//! for efficient initialization of surrogate optimization.

use rand::seq::SliceRandom;
use rand::Rng;

/// Generate Latin Hypercube samples within the given bounds.
///
/// Latin Hypercube Sampling ensures good coverage of the parameter space
/// by dividing each dimension into n equal intervals and placing exactly
/// one sample in each interval per dimension.
///
/// # Arguments
/// * `bounds` - Parameter bounds [(min, max), ...]
/// * `n_samples` - Number of samples to generate
///
/// # Returns
/// Vector of sample points, each a vector of parameter values.
pub fn latin_hypercube_sample(bounds: &[(f64, f64)], n_samples: usize) -> Vec<Vec<f64>> {
    if n_samples == 0 || bounds.is_empty() {
        return Vec::new();
    }

    let n_dims = bounds.len();
    let mut rng = rand::thread_rng();

    // For each dimension, create a permutation of interval indices
    let mut permutations: Vec<Vec<usize>> = Vec::with_capacity(n_dims);
    for _ in 0..n_dims {
        let mut perm: Vec<usize> = (0..n_samples).collect();
        perm.shuffle(&mut rng);
        permutations.push(perm);
    }

    // Generate samples
    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut sample = Vec::with_capacity(n_dims);
        for (dim, perm) in permutations.iter().enumerate() {
            let (lo, hi) = bounds[dim];
            let interval = perm[i];

            // Random point within the interval
            let interval_size = (hi - lo) / (n_samples as f64);
            let interval_lo = lo + (interval as f64) * interval_size;
            let value = interval_lo + rng.gen_range(0.0..1.0) * interval_size;

            sample.push(value);
        }
        samples.push(sample);
    }

    samples
}

/// Generate random uniform samples within bounds.
///
/// # Arguments
/// * `bounds` - Parameter bounds [(min, max), ...]
/// * `n_samples` - Number of samples to generate
pub fn random_sample(bounds: &[(f64, f64)], n_samples: usize) -> Vec<Vec<f64>> {
    if bounds.is_empty() || n_samples == 0 {
        return Vec::new();
    }

    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let sample: Vec<f64> = bounds
            .iter()
            .map(|(lo, hi)| rng.gen_range(*lo..*hi))
            .collect();
        samples.push(sample);
    }

    samples
}

/// Generate grid samples (for small dimensions).
///
/// # Arguments
/// * `bounds` - Parameter bounds [(min, max), ...]
/// * `points_per_dim` - Number of points per dimension
pub fn grid_sample(bounds: &[(f64, f64)], points_per_dim: usize) -> Vec<Vec<f64>> {
    if bounds.is_empty() || points_per_dim == 0 {
        return Vec::new();
    }

    let n_dims = bounds.len();
    let total_points = points_per_dim.pow(n_dims as u32);
    let mut samples = Vec::with_capacity(total_points);

    // Generate all combinations
    let mut indices = vec![0usize; n_dims];
    loop {
        // Convert indices to parameter values
        let sample: Vec<f64> = indices
            .iter()
            .zip(bounds.iter())
            .map(|(&idx, (lo, hi))| {
                if points_per_dim == 1 {
                    (lo + hi) / 2.0
                } else {
                    lo + (idx as f64) * (hi - lo) / ((points_per_dim - 1) as f64)
                }
            })
            .collect();
        samples.push(sample);

        // Increment indices (like counting in base points_per_dim)
        let mut carry = true;
        for i in 0..n_dims {
            if carry {
                indices[i] += 1;
                if indices[i] >= points_per_dim {
                    indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }

        if carry {
            break; // All indices wrapped around
        }
    }

    samples
}

/// Generate Sobol sequence samples (quasi-random, low-discrepancy).
///
/// This is a simplified implementation for up to 6 dimensions.
/// For more dimensions, consider using a dedicated library.
///
/// # Arguments
/// * `bounds` - Parameter bounds [(min, max), ...]
/// * `n_samples` - Number of samples to generate
pub fn sobol_sample(bounds: &[(f64, f64)], n_samples: usize) -> Vec<Vec<f64>> {
    if bounds.is_empty() || n_samples == 0 {
        return Vec::new();
    }

    let n_dims = bounds.len().min(6); // Limited to 6 dimensions

    // Direction numbers for Sobol sequence (simplified)
    let direction_numbers: [[u32; 32]; 6] = [
        [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1,
        ],
        [
            1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1,
            3, 1, 3,
        ],
        [
            1, 3, 5, 15, 1, 3, 5, 15, 1, 3, 5, 15, 1, 3, 5, 15, 1, 3, 5, 15, 1, 3, 5, 15, 1, 3, 5,
            15, 1, 3, 5, 15,
        ],
        [
            1, 1, 7, 11, 1, 1, 7, 11, 1, 1, 7, 11, 1, 1, 7, 11, 1, 1, 7, 11, 1, 1, 7, 11, 1, 1, 7,
            11, 1, 1, 7, 11,
        ],
        [
            1, 3, 7, 5, 1, 3, 7, 5, 1, 3, 7, 5, 1, 3, 7, 5, 1, 3, 7, 5, 1, 3, 7, 5, 1, 3, 7, 5, 1,
            3, 7, 5,
        ],
        [
            1, 1, 5, 3, 1, 1, 5, 3, 1, 1, 5, 3, 1, 1, 5, 3, 1, 1, 5, 3, 1, 1, 5, 3, 1, 1, 5, 3, 1,
            1, 5, 3,
        ],
    ];

    let mut samples = Vec::with_capacity(n_samples);
    let mut x = vec![0u32; n_dims];

    for i in 0..n_samples {
        // Find rightmost zero bit position
        let mut c = 0;
        let mut value = i;
        while value & 1 == 1 {
            value >>= 1;
            c += 1;
        }

        // Update x using direction numbers
        for (dim, x_dim) in x.iter_mut().enumerate().take(n_dims) {
            *x_dim ^= direction_numbers[dim][c.min(31)] << (31 - c.min(31));
        }

        // Convert to [0, 1] and scale to bounds
        let sample: Vec<f64> = x
            .iter()
            .zip(bounds.iter())
            .map(|(&xi, (lo, hi))| {
                let unit = (xi as f64) / (u32::MAX as f64);
                lo + unit * (hi - lo)
            })
            .collect();

        samples.push(sample);
    }

    // If we have more dimensions than supported, fill rest with random
    if bounds.len() > 6 {
        let mut rng = rand::thread_rng();
        for sample in &mut samples {
            for (lo, hi) in bounds.iter().skip(6) {
                sample.push(rng.gen_range(*lo..*hi));
            }
        }
    }

    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latin_hypercube_coverage() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let samples = latin_hypercube_sample(&bounds, 10);

        assert_eq!(samples.len(), 10);

        // Check all samples are within bounds
        for sample in &samples {
            assert_eq!(sample.len(), 2);
            for (i, &val) in sample.iter().enumerate() {
                assert!(
                    val >= bounds[i].0 && val <= bounds[i].1,
                    "Sample value {} out of bounds",
                    val
                );
            }
        }
    }

    #[test]
    fn test_latin_hypercube_stratification() {
        let bounds = vec![(0.0, 10.0)];
        let samples = latin_hypercube_sample(&bounds, 10);

        // Each interval [0,1), [1,2), ..., [9,10) should have exactly one sample
        let mut intervals = vec![false; 10];
        for sample in &samples {
            let interval = (sample[0].floor() as usize).min(9);
            assert!(
                !intervals[interval],
                "Multiple samples in interval {}",
                interval
            );
            intervals[interval] = true;
        }

        // All intervals should be covered
        assert!(intervals.iter().all(|&x| x), "Not all intervals covered");
    }

    #[test]
    fn test_random_sample() {
        let bounds = vec![(-5.0, 5.0), (0.0, 100.0)];
        let samples = random_sample(&bounds, 100);

        assert_eq!(samples.len(), 100);
        for sample in &samples {
            assert!(sample[0] >= -5.0 && sample[0] <= 5.0);
            assert!(sample[1] >= 0.0 && sample[1] <= 100.0);
        }
    }

    #[test]
    fn test_grid_sample() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let samples = grid_sample(&bounds, 3);

        // 3x3 = 9 samples
        assert_eq!(samples.len(), 9);

        // Check corners exist
        let has_origin = samples.iter().any(|s| s[0] < 0.01 && s[1] < 0.01);
        let has_corner = samples.iter().any(|s| s[0] > 0.99 && s[1] > 0.99);
        assert!(has_origin, "Should have origin corner");
        assert!(has_corner, "Should have (1,1) corner");
    }

    #[test]
    fn test_sobol_sample() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let samples = sobol_sample(&bounds, 16);

        assert_eq!(samples.len(), 16);

        // All samples should be in bounds
        for sample in &samples {
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
            assert!(sample[1] >= 0.0 && sample[1] <= 1.0);
        }
    }

    #[test]
    fn test_empty_bounds() {
        let bounds: Vec<(f64, f64)> = vec![];
        assert!(latin_hypercube_sample(&bounds, 10).is_empty());
        assert!(random_sample(&bounds, 10).is_empty());
        assert!(grid_sample(&bounds, 3).is_empty());
    }

    #[test]
    fn test_zero_samples() {
        let bounds = vec![(0.0, 1.0)];
        assert!(latin_hypercube_sample(&bounds, 0).is_empty());
        assert!(random_sample(&bounds, 0).is_empty());
    }
}
