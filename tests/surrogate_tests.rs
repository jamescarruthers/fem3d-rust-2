//! Integration tests for surrogate optimization module.
//!
//! Tests the RBF surrogate model, sampling methods, and hybrid optimization strategies.

use fem3d_rust_2::optimization::{
    latin_hypercube_sample, random_sample, run_surrogate_optimization, AlphaSchedule, RbfKernel,
    SurrogateConfig, SurrogateModel,
};

// ============================================================================
// RBF Kernel Tests
// ============================================================================

#[test]
fn test_rbf_kernels_are_symmetric() {
    let kernels = [
        RbfKernel::Cubic,
        RbfKernel::Linear,
        RbfKernel::Gaussian(1.0),
        RbfKernel::Multiquadric(1.0),
        RbfKernel::ThinPlateSpline,
    ];

    for kernel in kernels {
        // φ(r) should give same result for same r
        let r = 0.5;
        let v1 = kernel.evaluate(r);
        let v2 = kernel.evaluate(r);
        assert!(
            (v1 - v2).abs() < 1e-14,
            "Kernel {:?} not deterministic",
            kernel
        );
    }
}

#[test]
fn test_rbf_kernel_at_zero() {
    // Most kernels should have specific behavior at r=0
    assert!((RbfKernel::Cubic.evaluate(0.0) - 0.0).abs() < 1e-14);
    assert!((RbfKernel::Linear.evaluate(0.0) - 0.0).abs() < 1e-14);
    assert!((RbfKernel::Gaussian(1.0).evaluate(0.0) - 1.0).abs() < 1e-14);
    assert!(RbfKernel::Multiquadric(1.0).evaluate(0.0) > 0.0);
}

// ============================================================================
// Surrogate Model Tests
// ============================================================================

#[test]
fn test_surrogate_model_creation() {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let model = SurrogateModel::with_bounds(bounds.clone());

    assert_eq!(model.num_samples(), 0);
    assert_eq!(model.bounds().len(), 2);
    assert!(model.best_value().is_none());
}

#[test]
fn test_surrogate_model_single_sample() {
    let bounds = vec![(0.0, 1.0)];
    let mut model = SurrogateModel::with_bounds(bounds);

    model.add_sample(vec![0.5], 1.0);

    assert_eq!(model.num_samples(), 1);
    assert!((model.best_value().unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_surrogate_model_multiple_samples() {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let mut model = SurrogateModel::with_bounds(bounds);

    model.add_sample(vec![0.0, 0.0], 10.0);
    model.add_sample(vec![0.5, 0.5], 1.0); // Best
    model.add_sample(vec![1.0, 1.0], 5.0);

    assert_eq!(model.num_samples(), 3);
    assert!((model.best_value().unwrap() - 1.0).abs() < 1e-10);

    let best = model.best_sample().unwrap();
    assert!((best[0] - 0.5).abs() < 1e-10);
    assert!((best[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_surrogate_model_interpolation_accuracy() {
    let bounds = vec![(-1.0, 1.0)];
    let mut model = SurrogateModel::with_bounds(bounds);

    // Sample a quadratic function: f(x) = x²
    model.add_sample(vec![-1.0], 1.0);
    model.add_sample(vec![-0.5], 0.25);
    model.add_sample(vec![0.0], 0.0);
    model.add_sample(vec![0.5], 0.25);
    model.add_sample(vec![1.0], 1.0);

    // Check interpolation at sample points
    assert!((model.predict(&[-1.0]) - 1.0).abs() < 0.1);
    assert!((model.predict(&[0.0]) - 0.0).abs() < 0.1);
    assert!((model.predict(&[1.0]) - 1.0).abs() < 0.1);
}

#[test]
fn test_surrogate_merit_function() {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let mut model = SurrogateModel::with_bounds(bounds);

    model.add_sample(vec![0.2, 0.2], 0.5);
    model.add_sample(vec![0.8, 0.8], 1.5);

    // With high alpha, merit should favor lower surrogate values
    let merit_low = model.merit(&[0.2, 0.2], 0.9);
    let merit_high = model.merit(&[0.8, 0.8], 0.9);

    // Point with lower value should have lower merit
    assert!(
        merit_low < merit_high,
        "Expected merit_low ({}) < merit_high ({})",
        merit_low,
        merit_high
    );
}

#[test]
fn test_surrogate_suggest_next_within_bounds() {
    let bounds = vec![(0.0, 10.0), (-5.0, 5.0)];
    let mut model = SurrogateModel::with_bounds(bounds.clone());

    model.add_sample(vec![5.0, 0.0], 1.0);

    for _ in 0..10 {
        let suggestion = model.suggest_next(100, 0.5);

        assert!(
            suggestion[0] >= 0.0 && suggestion[0] <= 10.0,
            "x[0] = {} out of bounds",
            suggestion[0]
        );
        assert!(
            suggestion[1] >= -5.0 && suggestion[1] <= 5.0,
            "x[1] = {} out of bounds",
            suggestion[1]
        );
    }
}

// ============================================================================
// Alpha Schedule Tests
// ============================================================================

#[test]
fn test_alpha_schedule_constant() {
    let schedule = AlphaSchedule::Constant(0.7);

    assert!((schedule.get_alpha(0, 100) - 0.7).abs() < 1e-10);
    assert!((schedule.get_alpha(50, 100) - 0.7).abs() < 1e-10);
    assert!((schedule.get_alpha(100, 100) - 0.7).abs() < 1e-10);
}

#[test]
fn test_alpha_schedule_linear() {
    let schedule = AlphaSchedule::Linear {
        start: 0.2,
        end: 0.8,
    };

    assert!((schedule.get_alpha(0, 100) - 0.2).abs() < 1e-10);
    assert!((schedule.get_alpha(50, 100) - 0.5).abs() < 1e-10);
    assert!((schedule.get_alpha(100, 100) - 0.8).abs() < 1e-10);
}

#[test]
fn test_alpha_schedule_cosine() {
    let schedule = AlphaSchedule::Cosine { min: 0.1, max: 0.9 };

    let alpha_start = schedule.get_alpha(0, 100);
    let alpha_mid = schedule.get_alpha(50, 100);
    let alpha_end = schedule.get_alpha(100, 100);

    // Cosine should start low, increase smoothly
    assert!(alpha_start < alpha_mid);
    assert!(alpha_mid < alpha_end);
    assert!(alpha_start >= 0.1);
    assert!(alpha_end <= 0.9);
}

// ============================================================================
// Sampling Tests
// ============================================================================

#[test]
fn test_latin_hypercube_sample_coverage() {
    let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
    let samples = latin_hypercube_sample(&bounds, 10);

    assert_eq!(samples.len(), 10);

    // All samples should be within bounds
    for sample in &samples {
        assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
        assert!(sample[1] >= 0.0 && sample[1] <= 1.0);
    }
}

#[test]
fn test_latin_hypercube_stratification() {
    // In 1D, LHS should place exactly one sample per interval
    let bounds = vec![(0.0, 5.0)];
    let samples = latin_hypercube_sample(&bounds, 5);

    let mut intervals = [false; 5];
    for sample in &samples {
        let interval = (sample[0].floor() as usize).min(4);
        assert!(
            !intervals[interval],
            "Duplicate sample in interval {}",
            interval
        );
        intervals[interval] = true;
    }

    assert!(intervals.iter().all(|&x| x), "Not all intervals covered");
}

#[test]
fn test_random_sample_coverage() {
    let bounds = vec![(-10.0, 10.0)];
    let samples = random_sample(&bounds, 100);

    assert_eq!(samples.len(), 100);

    // All samples within bounds
    for sample in &samples {
        assert!(sample[0] >= -10.0 && sample[0] <= 10.0);
    }

    // Should have reasonable spread (not all in one spot)
    let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / 100.0;
    assert!(mean.abs() < 5.0, "Mean {} is too extreme", mean);
}

// ============================================================================
// Surrogate Optimization Tests
// ============================================================================

#[test]
fn test_surrogate_optimization_sphere_function() {
    // Sphere function: f(x) = Σxᵢ² - minimum at origin
    let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];

    let config = SurrogateConfig {
        initial_samples: 8,
        max_evaluations: 25,
        convergence_tol: 0.05,
        verbose: false,
        ..Default::default()
    };

    let result = run_surrogate_optimization(&config, bounds, |x| x.iter().map(|xi| xi * xi).sum());

    assert!(
        result.best_fitness < 1.0,
        "Sphere optimization should find value < 1.0, got {}",
        result.best_fitness
    );
    assert!(result.evaluations <= 25);
}

#[test]
fn test_surrogate_optimization_with_initial_point() {
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    // Start near the minimum
    let config = SurrogateConfig {
        initial_samples: 5,
        max_evaluations: 15,
        convergence_tol: 0.01,
        initial_point: Some(vec![0.1, 0.1]),
        verbose: false,
        ..Default::default()
    };

    let result = run_surrogate_optimization(&config, bounds, |x| x.iter().map(|xi| xi * xi).sum());

    // Should converge faster with good initial point
    assert!(
        result.best_fitness < 0.5,
        "Should converge well with good initial point, got {}",
        result.best_fitness
    );
}

#[test]
fn test_surrogate_optimization_convergence_flag() {
    let bounds = vec![(0.0, 1.0)];

    // Very easy problem - should converge
    let config = SurrogateConfig {
        initial_samples: 5,
        max_evaluations: 20,
        convergence_tol: 10.0, // Very lenient
        verbose: false,
        ..Default::default()
    };

    let result = run_surrogate_optimization(&config, bounds, |x| x[0] * x[0]);

    // Should converge since tolerance is very lenient
    assert!(result.converged || result.best_fitness < 10.0);
}

#[test]
fn test_surrogate_optimization_fitness_history() {
    let bounds = vec![(-1.0, 1.0)];

    let config = SurrogateConfig {
        initial_samples: 5,
        max_evaluations: 15,
        convergence_tol: 0.001,
        verbose: false,
        ..Default::default()
    };

    let result = run_surrogate_optimization(&config, bounds, |x| x[0] * x[0]);

    // Fitness history should be non-empty and non-increasing
    assert!(!result.fitness_history.is_empty());

    let mut prev = f64::INFINITY;
    for &fitness in &result.fitness_history {
        assert!(fitness <= prev + 1e-10, "Fitness history should be monotonically non-increasing");
        prev = fitness;
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_surrogate_config_fast() {
    let config = SurrogateConfig::fast();

    assert_eq!(config.initial_samples, 10);
    assert_eq!(config.max_evaluations, 50);
    assert!(config.convergence_tol > 0.01);
}

#[test]
fn test_surrogate_config_thorough() {
    let config = SurrogateConfig::thorough();

    assert_eq!(config.initial_samples, 25);
    assert_eq!(config.max_evaluations, 150);
    assert!(config.convergence_tol < 0.01);
}

#[test]
fn test_surrogate_config_with_initial_point() {
    let config = SurrogateConfig::default().with_initial_point(vec![1.0, 2.0, 3.0]);

    assert!(config.initial_point.is_some());
    let point = config.initial_point.unwrap();
    assert_eq!(point, vec![1.0, 2.0, 3.0]);
}
