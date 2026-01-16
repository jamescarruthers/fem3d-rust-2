//! Demonstration of Surrogate Optimization for Bar Tuning
//!
//! This example demonstrates the surrogate optimization approach from
//! Soares et al. 2021 for efficiently tuning percussion bar frequencies.
//!
//! The key insight is that 3D FEM evaluations are expensive (1-10 seconds each),
//! so surrogate optimization dramatically reduces the number of evaluations needed
//! by building an RBF interpolation model that guides the search.
//!
//! Usage:
//!   cargo run --example surrogate_demo --release
//!
//! Strategies demonstrated:
//! 1. Pure surrogate 3D optimization
//! 2. Hybrid 2D→3D optimization (recommended)
//! 3. Comparison with pure EA

use fem3d_rust_2::optimization::{
    run_hybrid_optimization, run_optimization, AnalysisMode, BarParameters, EAConfig,
    EAParameters, HybridConfig, Material, OptimizationStrategy, PenaltyType, SurrogateConfig,
};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Surrogate Optimization Demo (Soares et al. 2021)             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Define bar geometry (similar to F3 vibraphone bar, ~175 Hz)
    let bar = BarParameters::new(
        0.35,  // length: 350mm
        0.05,  // width: 50mm
        0.01,  // h0: 10mm
        0.002, // h_min: 2mm
    );

    // Aluminum properties (from paper: E=69.8 GPa, ρ=2748 kg/m³, ν=0.28)
    let material = Material::aluminum();

    // Target frequencies: 1:4 tuning ratio (fundamental + double octave)
    // F3 ≈ 175 Hz, so f2 = 700 Hz
    let target_frequencies = vec![175.0, 700.0];

    let num_cuts = 2;

    println!("Bar Configuration:");
    println!("  Length:    {} mm", bar.length * 1000.0);
    println!("  Width:     {} mm", bar.width * 1000.0);
    println!("  Thickness: {} mm", bar.h0 * 1000.0);
    println!("  Material:  {}", material.name);
    println!("\nTarget Frequencies:");
    for (i, f) in target_frequencies.iter().enumerate() {
        println!("  f{}: {} Hz", i + 1, f);
    }
    println!("\n");

    // ========================================================================
    // Strategy 1: Hybrid 2D→3D Optimization (Recommended)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Strategy 1: Hybrid 2D→3D Optimization (Recommended)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Phase 1: Fast 2D EA exploration");
    println!("Phase 2: 3D surrogate refinement using 2D result as starting point\n");

    let start = Instant::now();

    let hybrid_config = HybridConfig::new(
        bar.clone(),
        material.clone(),
        target_frequencies.clone(),
        num_cuts,
    )
    .with_strategy(OptimizationStrategy::Hybrid2Dto3D {
        ea_params: EAParameters {
            population_size: 30,
            max_generations: 40,
            target_error: 0.01,
            ..Default::default()
        },
        surrogate_config: SurrogateConfig {
            initial_samples: 10,
            max_evaluations: 40,
            convergence_tol: 0.01,
            verbose: false,
            ..Default::default()
        },
        frequency_correction: 1.0,
    })
    .with_verbose(true);

    let result_hybrid = run_hybrid_optimization(&hybrid_config);
    let elapsed_hybrid = start.elapsed();

    println!("\nHybrid Results:");
    println!("  Best fitness:     {:.6}", result_hybrid.best_fitness);
    println!("  Max error:        {:.1} cents", result_hybrid.max_error_cents);
    println!("  2D evaluations:   {}", result_hybrid.total_evaluations_2d);
    println!("  3D evaluations:   {}", result_hybrid.total_evaluations_3d);
    println!("  Time:             {:.2}s", elapsed_hybrid.as_secs_f64());
    println!("  Converged:        {}", result_hybrid.converged);
    println!("\nFrequencies:");
    for (i, (comp, target)) in result_hybrid
        .computed_frequencies
        .iter()
        .zip(result_hybrid.target_frequencies.iter())
        .enumerate()
    {
        let error = result_hybrid.errors_cents.get(i).unwrap_or(&0.0);
        println!(
            "  f{}: {:.1} Hz (target: {:.1} Hz, error: {:+.1} cents)",
            i + 1,
            comp,
            target,
            error
        );
    }

    // ========================================================================
    // Strategy 2: Pure 2D EA (Fast baseline)
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Strategy 2: Pure 2D Evolutionary Algorithm (Fast Baseline)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let start = Instant::now();

    let mut params_2d = EAParameters::default();
    params_2d.population_size = 40;
    params_2d.max_generations = 80;
    params_2d.analysis_mode = AnalysisMode::Beam2D;

    let config_2d = EAConfig::new(
        bar.clone(),
        material.clone(),
        target_frequencies.clone(),
        num_cuts,
    )
    .with_params(params_2d);

    let result_2d = run_optimization(&config_2d);
    let elapsed_2d = start.elapsed();

    println!("2D EA Results:");
    println!("  Best fitness:     {:.6}", result_2d.tuning_error);
    println!("  Max error:        {:.1} cents", result_2d.max_error_cents);
    println!("  Generations:      {}", result_2d.generations);
    println!("  Time:             {:.2}s", elapsed_2d.as_secs_f64());
    println!("\nFrequencies:");
    for (i, (comp, target)) in result_2d
        .computed_frequencies
        .iter()
        .zip(result_2d.target_frequencies.iter())
        .enumerate()
    {
        let error = result_2d.errors_in_cents.get(i).unwrap_or(&0.0);
        println!(
            "  f{}: {:.1} Hz (target: {:.1} Hz, error: {:+.1} cents)",
            i + 1,
            comp,
            target,
            error
        );
    }

    // ========================================================================
    // Strategy 3: Pure Surrogate 3D (no 2D warmup)
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Strategy 3: Pure Surrogate 3D Optimization");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Uses only 3D FEM with surrogate guidance (no 2D warmup)\n");

    let start = Instant::now();

    let surrogate_3d_config = HybridConfig::new(
        bar.clone(),
        material.clone(),
        target_frequencies.clone(),
        num_cuts,
    )
    .with_strategy(OptimizationStrategy::Surrogate3D(SurrogateConfig {
        initial_samples: 15,
        max_evaluations: 50,
        convergence_tol: 0.01,
        verbose: true,
        ..Default::default()
    }));

    let result_surrogate_3d = run_hybrid_optimization(&surrogate_3d_config);
    let elapsed_surrogate_3d = start.elapsed();

    println!("\nPure Surrogate 3D Results:");
    println!("  Best fitness:     {:.6}", result_surrogate_3d.best_fitness);
    println!(
        "  Max error:        {:.1} cents",
        result_surrogate_3d.max_error_cents
    );
    println!(
        "  3D evaluations:   {}",
        result_surrogate_3d.total_evaluations_3d
    );
    println!("  Time:             {:.2}s", elapsed_surrogate_3d.as_secs_f64());
    println!("  Converged:        {}", result_surrogate_3d.converged);

    // ========================================================================
    // Summary Comparison
    // ========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        Summary Comparison                        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<25} {:>12} {:>12} {:>12}",
        "Strategy", "Max Error", "3D Evals", "Time"
    );
    println!("{:-<25} {:->12} {:->12} {:->12}", "", "", "", "");
    println!(
        "{:<25} {:>10.1} ct {:>12} {:>10.2}s",
        "Hybrid 2D→3D",
        result_hybrid.max_error_cents,
        result_hybrid.total_evaluations_3d,
        elapsed_hybrid.as_secs_f64()
    );
    println!(
        "{:<25} {:>10.1} ct {:>12} {:>10.2}s",
        "Pure 2D EA",
        result_2d.max_error_cents,
        0,
        elapsed_2d.as_secs_f64()
    );
    println!(
        "{:<25} {:>10.1} ct {:>12} {:>10.2}s",
        "Pure Surrogate 3D",
        result_surrogate_3d.max_error_cents,
        result_surrogate_3d.total_evaluations_3d,
        elapsed_surrogate_3d.as_secs_f64()
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Insights from Soares et al. 2021:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("• 2D beam models are ~100x faster but miss torsional modes");
    println!("• 3D FEM captures all modes but is expensive (1-10s per eval)");
    println!("• Surrogate optimization reduces 3D evals from 1000s to 50-100");
    println!("• Hybrid 2D→3D combines fast exploration with accurate refinement");
    println!("• Tuning tolerance: <15 cents is typically acceptable for musicians");
    println!();
}
