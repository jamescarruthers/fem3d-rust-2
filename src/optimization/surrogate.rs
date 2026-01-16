//! Surrogate optimization using Radial Basis Function (RBF) interpolation.
//!
//! Based on Soares et al. 2021: "Tuning of bending and torsional modes of bars
//! used in mallet percussion instruments" (Section IV.A).
//!
//! The surrogate optimization algorithm alternates between:
//! 1. Construction of a surrogate model (RBF interpolation)
//! 2. Search for minimum using the surrogate as guide
//!
//! This approach significantly reduces the number of expensive 3D FEM evaluations
//! needed to find optimal solutions (typically 50-100 vs 1000+ for pure EA).
//!
//! ## Parallelization
//!
//! When the `parallel` feature is enabled, the RBF matrix construction and
//! candidate evaluation are parallelized using Rayon. This provides significant
//! speedups for large numbers of samples or candidates.

use nalgebra::{DMatrix, DVector};
use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Radial basis function kernel types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RbfKernel {
    /// Cubic kernel: φ(r) = r³ (paper default, recommended)
    Cubic,
    /// Thin plate spline: φ(r) = r² ln(r)
    ThinPlateSpline,
    /// Gaussian kernel: φ(r) = exp(-r²/σ²)
    Gaussian(f64),
    /// Multiquadric: φ(r) = sqrt(r² + c²)
    Multiquadric(f64),
    /// Linear: φ(r) = r
    Linear,
}

impl Default for RbfKernel {
    fn default() -> Self {
        RbfKernel::Cubic
    }
}

impl RbfKernel {
    /// Evaluate the radial basis function at distance r.
    pub fn evaluate(&self, r: f64) -> f64 {
        match self {
            RbfKernel::Cubic => r.powi(3),
            RbfKernel::ThinPlateSpline => {
                if r > 1e-14 {
                    r * r * r.ln()
                } else {
                    0.0
                }
            }
            RbfKernel::Gaussian(sigma) => (-r * r / (sigma * sigma)).exp(),
            RbfKernel::Multiquadric(c) => (r * r + c * c).sqrt(),
            RbfKernel::Linear => r,
        }
    }
}

/// Surrogate model using Radial Basis Function interpolation.
///
/// The surrogate function s(x) approximates the expensive objective function f(x)
/// using the formula:
///
/// s(x) = Σᵢ βᵢ φ(‖x - xᵢ‖)
///
/// where φ is the radial basis function and βᵢ are coefficients determined
/// by solving a linear system.
#[derive(Debug, Clone)]
pub struct SurrogateModel {
    /// Evaluated sample points (parameter vectors)
    samples: Vec<Vec<f64>>,
    /// Objective values at sample points
    values: Vec<f64>,
    /// RBF coefficients (computed by refit)
    coefficients: DVector<f64>,
    /// Kernel type
    kernel: RbfKernel,
    /// Parameter bounds [(min, max), ...]
    bounds: Vec<(f64, f64)>,
    /// Regularization parameter for numerical stability
    regularization: f64,
}

impl SurrogateModel {
    /// Create a new surrogate model with the given bounds and kernel.
    pub fn new(bounds: Vec<(f64, f64)>, kernel: RbfKernel) -> Self {
        Self {
            samples: Vec::new(),
            values: Vec::new(),
            coefficients: DVector::zeros(0),
            kernel,
            bounds,
            regularization: 1e-10,
        }
    }

    /// Create a new surrogate model with default cubic kernel.
    pub fn with_bounds(bounds: Vec<(f64, f64)>) -> Self {
        Self::new(bounds, RbfKernel::Cubic)
    }

    /// Set the regularization parameter.
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Get the number of samples in the model.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Get the parameter bounds.
    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }

    /// Get the best (minimum) value found so far.
    pub fn best_value(&self) -> Option<f64> {
        self.values.iter().cloned().reduce(f64::min)
    }

    /// Get the sample point with the best (minimum) value.
    pub fn best_sample(&self) -> Option<&Vec<f64>> {
        self.values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| &self.samples[i])
    }

    /// Add a sample point and refit the model.
    pub fn add_sample(&mut self, params: Vec<f64>, value: f64) {
        self.samples.push(params);
        self.values.push(value);
        self.refit();
    }

    /// Add multiple samples and refit once.
    pub fn add_samples(&mut self, samples: Vec<(Vec<f64>, f64)>) {
        for (params, value) in samples {
            self.samples.push(params);
            self.values.push(value);
        }
        self.refit();
    }

    /// Refit RBF coefficients by solving linear system Φβ = f.
    /// Uses parallel computation for matrix construction when available.
    fn refit(&mut self) {
        let m = self.samples.len();
        if m == 0 {
            self.coefficients = DVector::zeros(0);
            return;
        }

        // Build interpolation matrix Φ where Φ[i,j] = φ(‖xᵢ - xⱼ‖)
        let phi = self.build_rbf_matrix();

        // Solve Φ β = f for coefficients β
        let f = DVector::from_vec(self.values.clone());

        // Try LU decomposition first
        if let Some(solution) = phi.clone().lu().solve(&f) {
            self.coefficients = solution;
        } else if let Some(solution) = phi.clone().svd(true, true).solve(&f, 1e-10).ok() {
            // Fall back to SVD for ill-conditioned matrices
            self.coefficients = solution;
        } else {
            // Last resort: use pseudo-inverse
            self.coefficients = DVector::zeros(m);
        }
    }

    /// Build the RBF interpolation matrix (sequential version).
    #[cfg(not(feature = "parallel"))]
    fn build_rbf_matrix(&self) -> DMatrix<f64> {
        let m = self.samples.len();
        let mut phi = DMatrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                let dist = euclidean_distance(&self.samples[i], &self.samples[j]);
                phi[(i, j)] = self.kernel.evaluate(dist);
            }
            // Add regularization to diagonal for numerical stability
            phi[(i, i)] += self.regularization;
        }
        phi
    }

    /// Build the RBF interpolation matrix (parallel version).
    /// Parallelizes row computation for O(n²) speedup on multicore systems.
    #[cfg(feature = "parallel")]
    fn build_rbf_matrix(&self) -> DMatrix<f64> {
        let m = self.samples.len();

        // Compute each row in parallel
        let rows: Vec<Vec<f64>> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut row = Vec::with_capacity(m);
                for j in 0..m {
                    let dist = euclidean_distance(&self.samples[i], &self.samples[j]);
                    let mut val = self.kernel.evaluate(dist);
                    // Add regularization to diagonal
                    if i == j {
                        val += self.regularization;
                    }
                    row.push(val);
                }
                row
            })
            .collect();

        // Assemble into matrix
        let mut phi = DMatrix::zeros(m, m);
        for (i, row) in rows.into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                phi[(i, j)] = val;
            }
        }
        phi
    }

    /// Predict surrogate value at a new point.
    pub fn predict(&self, x: &[f64]) -> f64 {
        if self.samples.is_empty() || self.coefficients.len() == 0 {
            return f64::INFINITY;
        }

        let mut sum = 0.0;
        for (i, sample) in self.samples.iter().enumerate() {
            let dist = euclidean_distance(x, sample);
            sum += self.coefficients[i] * self.kernel.evaluate(dist);
        }
        sum
    }

    /// Find minimum distance from point x to any evaluated sample.
    pub fn min_distance_to_samples(&self, x: &[f64]) -> f64 {
        self.samples
            .iter()
            .map(|s| euclidean_distance(x, s))
            .fold(f64::INFINITY, f64::min)
    }

    /// Get range of objective values (min, max).
    pub fn value_range(&self) -> (f64, f64) {
        let min = self.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }

    /// Compute the merit function M(x) = α·S(x) + (1-α)·D(x) (Eq. 8 from paper).
    ///
    /// This balances exploitation (minimizing surrogate) vs exploration (distance from samples).
    ///
    /// # Arguments
    /// * `x` - Point to evaluate
    /// * `alpha` - Weight factor (0-1). Higher = more exploitation, lower = more exploration.
    pub fn merit(&self, x: &[f64], alpha: f64) -> f64 {
        if self.samples.is_empty() {
            return f64::INFINITY;
        }

        let s_x = self.predict(x);
        let d_x = self.min_distance_to_samples(x);

        // Get normalization ranges
        let (s_min, s_max) = self.value_range();

        // Compute diagonal of bounds for distance normalization
        let d_max: f64 = self
            .bounds
            .iter()
            .map(|(lo, hi)| (hi - lo).powi(2))
            .sum::<f64>()
            .sqrt();

        // Normalized surrogate value (Eq. 9a)
        let s_norm = if s_max > s_min + 1e-14 {
            (s_x - s_min) / (s_max - s_min)
        } else {
            0.5
        };

        // Normalized distance (Eq. 9b) - inverted so smaller distance = higher value
        let d_norm = if d_max > 1e-14 {
            (d_max - d_x) / d_max
        } else {
            0.5
        };

        // Merit function (Eq. 8)
        alpha * s_norm + (1.0 - alpha) * d_norm
    }

    /// Suggest next sample point by minimizing merit function.
    ///
    /// # Arguments
    /// * `num_candidates` - Number of random candidates to evaluate
    /// * `alpha` - Merit function weight (higher = more exploitation)
    pub fn suggest_next(&self, num_candidates: usize, alpha: f64) -> Vec<f64> {
        self.suggest_next_impl(num_candidates, alpha)
    }

    /// Sequential implementation of suggest_next.
    #[cfg(not(feature = "parallel"))]
    fn suggest_next_impl(&self, num_candidates: usize, alpha: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        let mut best_candidate: Vec<f64> = self
            .bounds
            .iter()
            .map(|(lo, hi)| rng.gen_range(*lo..*hi))
            .collect();
        let mut best_merit = self.merit(&best_candidate, alpha);

        for _ in 0..num_candidates {
            let candidate: Vec<f64> = self
                .bounds
                .iter()
                .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                .collect();

            let merit = self.merit(&candidate, alpha);
            if merit < best_merit {
                best_merit = merit;
                best_candidate = candidate;
            }
        }

        best_candidate
    }

    /// Parallel implementation of suggest_next.
    /// Evaluates candidates in parallel batches for better performance.
    #[cfg(feature = "parallel")]
    fn suggest_next_impl(&self, num_candidates: usize, alpha: f64) -> Vec<f64> {
        use rand::SeedableRng;

        // Generate all candidates in parallel with thread-local RNGs
        let candidates: Vec<Vec<f64>> = (0..num_candidates)
            .into_par_iter()
            .map_init(
                || rand::rngs::SmallRng::from_entropy(),
                |rng, _| {
                    self.bounds
                        .iter()
                        .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                        .collect()
                },
            )
            .collect();

        // Evaluate merit in parallel and find the best
        candidates
            .into_par_iter()
            .map(|candidate| {
                let merit = self.merit(&candidate, alpha);
                (candidate, merit)
            })
            .reduce(
                || {
                    // Initial value: generate a random candidate
                    let mut rng = rand::thread_rng();
                    let candidate: Vec<f64> = self
                        .bounds
                        .iter()
                        .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                        .collect();
                    let merit = self.merit(&candidate, alpha);
                    (candidate, merit)
                },
                |best, current| {
                    if current.1 < best.1 {
                        current
                    } else {
                        best
                    }
                },
            )
            .0
    }

    /// Suggest next sample using gradient-enhanced search around best point.
    ///
    /// Combines random search with local refinement around the current best.
    pub fn suggest_next_enhanced(
        &self,
        num_candidates: usize,
        alpha: f64,
        local_fraction: f64,
    ) -> Vec<f64> {
        self.suggest_next_enhanced_impl(num_candidates, alpha, local_fraction)
    }

    /// Sequential implementation of suggest_next_enhanced.
    #[cfg(not(feature = "parallel"))]
    fn suggest_next_enhanced_impl(
        &self,
        num_candidates: usize,
        alpha: f64,
        local_fraction: f64,
    ) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let num_local = ((num_candidates as f64) * local_fraction) as usize;
        let num_global = num_candidates - num_local;

        let mut best_candidate: Vec<f64> = self
            .bounds
            .iter()
            .map(|(lo, hi)| rng.gen_range(*lo..*hi))
            .collect();
        let mut best_merit = self.merit(&best_candidate, alpha);

        // Global random search
        for _ in 0..num_global {
            let candidate: Vec<f64> = self
                .bounds
                .iter()
                .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                .collect();

            let merit = self.merit(&candidate, alpha);
            if merit < best_merit {
                best_merit = merit;
                best_candidate = candidate;
            }
        }

        // Local search around current best sample
        if let Some(best_sample) = self.best_sample() {
            for _ in 0..num_local {
                let candidate: Vec<f64> = best_sample
                    .iter()
                    .zip(self.bounds.iter())
                    .map(|(&x, (lo, hi))| {
                        let range = hi - lo;
                        let perturbation = rng.gen_range(-0.1..0.1) * range;
                        (x + perturbation).clamp(*lo, *hi)
                    })
                    .collect();

                let merit = self.merit(&candidate, alpha);
                if merit < best_merit {
                    best_merit = merit;
                    best_candidate = candidate;
                }
            }
        }

        best_candidate
    }

    /// Parallel implementation of suggest_next_enhanced.
    #[cfg(feature = "parallel")]
    fn suggest_next_enhanced_impl(
        &self,
        num_candidates: usize,
        alpha: f64,
        local_fraction: f64,
    ) -> Vec<f64> {
        use rand::SeedableRng;

        let num_local = ((num_candidates as f64) * local_fraction) as usize;
        let num_global = num_candidates - num_local;

        // Generate global candidates in parallel
        let global_candidates: Vec<Vec<f64>> = (0..num_global)
            .into_par_iter()
            .map_init(
                || rand::rngs::SmallRng::from_entropy(),
                |rng, _| {
                    self.bounds
                        .iter()
                        .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                        .collect()
                },
            )
            .collect();

        // Generate local candidates around best sample in parallel
        let local_candidates: Vec<Vec<f64>> = if let Some(best_sample) = self.best_sample() {
            (0..num_local)
                .into_par_iter()
                .map_init(
                    || rand::rngs::SmallRng::from_entropy(),
                    |rng, _| {
                        best_sample
                            .iter()
                            .zip(self.bounds.iter())
                            .map(|(&x, (lo, hi))| {
                                let range = hi - lo;
                                let perturbation = rng.gen_range(-0.1..0.1) * range;
                                (x + perturbation).clamp(*lo, *hi)
                            })
                            .collect()
                    },
                )
                .collect()
        } else {
            Vec::new()
        };

        // Combine and evaluate all candidates in parallel
        let all_candidates: Vec<Vec<f64>> = global_candidates
            .into_iter()
            .chain(local_candidates)
            .collect();

        all_candidates
            .into_par_iter()
            .map(|candidate| {
                let merit = self.merit(&candidate, alpha);
                (candidate, merit)
            })
            .reduce(
                || {
                    // Initial value
                    let mut rng = rand::thread_rng();
                    let candidate: Vec<f64> = self
                        .bounds
                        .iter()
                        .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                        .collect();
                    let merit = self.merit(&candidate, alpha);
                    (candidate, merit)
                },
                |best, current| {
                    if current.1 < best.1 {
                        current
                    } else {
                        best
                    }
                },
            )
            .0
    }
}

/// Compute Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Alpha schedule for balancing exploration vs exploitation.
#[derive(Debug, Clone, Copy)]
pub enum AlphaSchedule {
    /// Constant alpha throughout optimization
    Constant(f64),
    /// Linear increase from start to end (more exploitation over time)
    Linear { start: f64, end: f64 },
    /// Cosine annealing schedule
    Cosine { min: f64, max: f64 },
}

impl Default for AlphaSchedule {
    fn default() -> Self {
        AlphaSchedule::Linear {
            start: 0.3,
            end: 0.95,
        }
    }
}

impl AlphaSchedule {
    /// Get alpha value for current iteration.
    pub fn get_alpha(&self, iteration: usize, max_iterations: usize) -> f64 {
        let progress = if max_iterations > 0 {
            (iteration as f64) / (max_iterations as f64)
        } else {
            0.0
        };

        match self {
            AlphaSchedule::Constant(alpha) => *alpha,
            AlphaSchedule::Linear { start, end } => start + (end - start) * progress,
            AlphaSchedule::Cosine { min, max } => {
                let cos_val = (std::f64::consts::PI * progress).cos();
                min + (max - min) * (1.0 - cos_val) / 2.0
            }
        }
    }
}

/// Configuration for surrogate optimization.
#[derive(Debug, Clone)]
pub struct SurrogateConfig {
    /// Number of initial samples (Latin Hypercube or random)
    pub initial_samples: usize,
    /// Maximum number of expensive function evaluations
    pub max_evaluations: usize,
    /// Convergence tolerance (stop when fitness < tol)
    pub convergence_tol: f64,
    /// Alpha schedule for merit function
    pub alpha_schedule: AlphaSchedule,
    /// Number of candidate points to evaluate per iteration
    pub candidate_pool_size: usize,
    /// Fraction of candidates for local search (0-1)
    pub local_search_fraction: f64,
    /// RBF kernel type
    pub kernel: RbfKernel,
    /// Optional initial point (e.g., from 2D optimization)
    pub initial_point: Option<Vec<f64>>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for SurrogateConfig {
    fn default() -> Self {
        Self {
            initial_samples: 15,
            max_evaluations: 100,
            convergence_tol: 0.01,
            alpha_schedule: AlphaSchedule::default(),
            candidate_pool_size: 5000,
            local_search_fraction: 0.3,
            kernel: RbfKernel::Cubic,
            initial_point: None,
            verbose: false,
        }
    }
}

impl SurrogateConfig {
    /// Create configuration for fast optimization (fewer evaluations).
    pub fn fast() -> Self {
        Self {
            initial_samples: 10,
            max_evaluations: 50,
            convergence_tol: 0.05,
            candidate_pool_size: 3000,
            ..Default::default()
        }
    }

    /// Create configuration for thorough optimization.
    pub fn thorough() -> Self {
        Self {
            initial_samples: 25,
            max_evaluations: 150,
            convergence_tol: 0.001,
            candidate_pool_size: 10000,
            local_search_fraction: 0.4,
            ..Default::default()
        }
    }

    /// Set initial point (e.g., from 2D pre-optimization).
    pub fn with_initial_point(mut self, point: Vec<f64>) -> Self {
        self.initial_point = Some(point);
        self
    }

    /// Set verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// Result of surrogate optimization.
#[derive(Debug, Clone)]
pub struct SurrogateResult {
    /// Best parameters found
    pub best_params: Vec<f64>,
    /// Best fitness value
    pub best_fitness: f64,
    /// Total number of expensive evaluations
    pub evaluations: usize,
    /// History of best fitness per evaluation
    pub fitness_history: Vec<f64>,
    /// All evaluated samples (params, fitness)
    pub all_samples: Vec<(Vec<f64>, f64)>,
    /// Whether convergence tolerance was reached
    pub converged: bool,
}

/// Run surrogate optimization.
///
/// # Arguments
/// * `config` - Surrogate optimization configuration
/// * `bounds` - Parameter bounds [(min, max), ...]
/// * `evaluate_fn` - Expensive objective function to minimize
///
/// # Returns
/// Optimization result with best parameters and fitness.
pub fn run_surrogate_optimization<F>(
    config: &SurrogateConfig,
    bounds: Vec<(f64, f64)>,
    mut evaluate_fn: F,
) -> SurrogateResult
where
    F: FnMut(&[f64]) -> f64,
{
    let mut surrogate = SurrogateModel::new(bounds.clone(), config.kernel);
    let mut fitness_history = Vec::with_capacity(config.max_evaluations);
    let mut all_samples = Vec::with_capacity(config.max_evaluations);

    // Phase 1: Initial sampling
    let initial_points = if let Some(ref init_point) = config.initial_point {
        // Include initial point plus random samples
        let mut points = crate::optimization::sampling::latin_hypercube_sample(
            &bounds,
            config.initial_samples - 1,
        );
        points.insert(0, init_point.clone());
        points
    } else {
        crate::optimization::sampling::latin_hypercube_sample(&bounds, config.initial_samples)
    };

    let mut best_fitness = f64::INFINITY;
    let mut best_params = vec![];

    for point in initial_points {
        let fitness = evaluate_fn(&point);
        all_samples.push((point.clone(), fitness));
        surrogate.add_sample(point.clone(), fitness);

        if fitness < best_fitness {
            best_fitness = fitness;
            best_params = point;
        }
        fitness_history.push(best_fitness);

        if config.verbose {
            println!(
                "[Surrogate] Init sample {}/{}: fitness = {:.6}",
                all_samples.len(),
                config.initial_samples,
                fitness
            );
        }

        if fitness < config.convergence_tol {
            return SurrogateResult {
                best_params,
                best_fitness,
                evaluations: all_samples.len(),
                fitness_history,
                all_samples,
                converged: true,
            };
        }
    }

    // Phase 2: Iterative refinement
    for eval in config.initial_samples..config.max_evaluations {
        let alpha = config
            .alpha_schedule
            .get_alpha(eval - config.initial_samples, config.max_evaluations - config.initial_samples);

        // Suggest next point using merit function
        let candidate = surrogate.suggest_next_enhanced(
            config.candidate_pool_size,
            alpha,
            config.local_search_fraction,
        );

        // Expensive evaluation
        let fitness = evaluate_fn(&candidate);
        all_samples.push((candidate.clone(), fitness));
        surrogate.add_sample(candidate.clone(), fitness);

        if fitness < best_fitness {
            best_fitness = fitness;
            best_params = candidate;
        }
        fitness_history.push(best_fitness);

        if config.verbose {
            println!(
                "[Surrogate] Eval {}/{}: fitness = {:.6}, best = {:.6}, alpha = {:.3}",
                eval + 1,
                config.max_evaluations,
                fitness,
                best_fitness,
                alpha
            );
        }

        if fitness < config.convergence_tol {
            return SurrogateResult {
                best_params,
                best_fitness,
                evaluations: all_samples.len(),
                fitness_history,
                all_samples,
                converged: true,
            };
        }
    }

    SurrogateResult {
        best_params,
        best_fitness,
        evaluations: all_samples.len(),
        fitness_history,
        all_samples,
        converged: best_fitness < config.convergence_tol,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_kernel_cubic() {
        let kernel = RbfKernel::Cubic;
        assert!((kernel.evaluate(0.0) - 0.0).abs() < 1e-10);
        assert!((kernel.evaluate(1.0) - 1.0).abs() < 1e-10);
        assert!((kernel.evaluate(2.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_rbf_kernel_gaussian() {
        let kernel = RbfKernel::Gaussian(1.0);
        assert!((kernel.evaluate(0.0) - 1.0).abs() < 1e-10);
        assert!(kernel.evaluate(1.0) < 0.4); // exp(-1) ≈ 0.368
        assert!(kernel.evaluate(2.0) < 0.02); // exp(-4) ≈ 0.018
    }

    #[test]
    fn test_surrogate_model_basic() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut model = SurrogateModel::with_bounds(bounds);

        // Add some samples
        model.add_sample(vec![0.0, 0.0], 1.0);
        model.add_sample(vec![1.0, 1.0], 1.0);
        model.add_sample(vec![0.5, 0.5], 0.0); // Minimum at center

        assert_eq!(model.num_samples(), 3);
        assert!((model.best_value().unwrap() - 0.0).abs() < 1e-10);

        // Prediction at minimum should be close to 0
        let pred = model.predict(&[0.5, 0.5]);
        assert!(pred.abs() < 0.1, "Prediction at minimum: {}", pred);
    }

    #[test]
    fn test_surrogate_model_interpolation() {
        let bounds = vec![(-1.0, 1.0)];
        let mut model = SurrogateModel::with_bounds(bounds);

        // Quadratic function: f(x) = x²
        model.add_sample(vec![-1.0], 1.0);
        model.add_sample(vec![0.0], 0.0);
        model.add_sample(vec![1.0], 1.0);

        // Should interpolate exactly at sample points
        assert!((model.predict(&[-1.0]) - 1.0).abs() < 0.1);
        assert!((model.predict(&[0.0]) - 0.0).abs() < 0.1);
        assert!((model.predict(&[1.0]) - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_merit_function() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut model = SurrogateModel::with_bounds(bounds);

        model.add_sample(vec![0.5, 0.5], 0.0);
        model.add_sample(vec![0.0, 0.0], 1.0);

        // With high alpha (exploitation), merit should favor low surrogate value
        let merit_at_min = model.merit(&[0.5, 0.5], 0.9);
        let merit_at_max = model.merit(&[0.0, 0.0], 0.9);
        assert!(merit_at_min < merit_at_max);

        // With low alpha (exploration), merit should favor distance from samples
        let merit_far = model.merit(&[1.0, 1.0], 0.1);
        let merit_near = model.merit(&[0.5, 0.5], 0.1);
        assert!(merit_far < merit_near);
    }

    #[test]
    fn test_alpha_schedule_linear() {
        let schedule = AlphaSchedule::Linear {
            start: 0.3,
            end: 0.9,
        };

        assert!((schedule.get_alpha(0, 100) - 0.3).abs() < 1e-10);
        assert!((schedule.get_alpha(50, 100) - 0.6).abs() < 1e-10);
        assert!((schedule.get_alpha(100, 100) - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_surrogate_optimization_sphere() {
        // Optimize sphere function: f(x) = Σxᵢ²
        let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let config = SurrogateConfig {
            initial_samples: 10,
            max_evaluations: 30,
            convergence_tol: 0.01,
            verbose: false,
            ..Default::default()
        };

        let result = run_surrogate_optimization(&config, bounds, |x| {
            x.iter().map(|xi| xi * xi).sum()
        });

        // Should find minimum near (0, 0) with value near 0
        assert!(
            result.best_fitness < 0.5,
            "Best fitness {} should be < 0.5",
            result.best_fitness
        );
        assert!(
            result.best_params[0].abs() < 1.0,
            "x[0] = {} should be near 0",
            result.best_params[0]
        );
        assert!(
            result.best_params[1].abs() < 1.0,
            "x[1] = {} should be near 0",
            result.best_params[1]
        );
    }

    #[test]
    fn test_surrogate_optimization_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
        // Minimum at (1, 1) with value 0
        let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];
        let config = SurrogateConfig {
            initial_samples: 15,
            max_evaluations: 60,
            convergence_tol: 0.1,
            verbose: false,
            ..Default::default()
        };

        let result = run_surrogate_optimization(&config, bounds, |x| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        });

        // Should get reasonably close to minimum
        assert!(
            result.best_fitness < 5.0,
            "Best fitness {} should be < 5.0",
            result.best_fitness
        );
    }

    #[test]
    fn test_suggest_next() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let mut model = SurrogateModel::with_bounds(bounds.clone());

        model.add_sample(vec![0.5, 0.5], 0.0);
        model.add_sample(vec![0.0, 0.0], 1.0);

        let suggestion = model.suggest_next(1000, 0.7);

        // Suggestion should be within bounds
        for (i, &val) in suggestion.iter().enumerate() {
            assert!(
                val >= bounds[i].0 && val <= bounds[i].1,
                "Suggestion {} = {} out of bounds",
                i,
                val
            );
        }
    }
}
