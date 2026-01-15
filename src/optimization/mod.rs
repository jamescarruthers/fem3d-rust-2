//! Evolutionary optimization for bar tuning.
//!
//! This module provides a complete evolutionary algorithm implementation
//! for optimizing undercut bar geometries to achieve target frequencies.
//!
//! # Example
//!
//! ```
//! use fem3d_rust_2::optimization::{
//!     EAConfig, EAParameters, BarParameters, Material, AnalysisMode,
//!     run_optimization,
//! };
//!
//! // Define bar and material
//! let bar = BarParameters::new(0.5, 0.03, 0.024, 0.01);
//! let material = Material::sapele();
//!
//! // Target frequencies (Hz)
//! let targets = vec![350.0, 700.0, 1400.0, 2100.0];
//!
//! // Configure optimization
//! let mut params = EAParameters::default();
//! params.population_size = 30;
//! params.max_generations = 50;
//! params.analysis_mode = AnalysisMode::Beam2D;
//!
//! let config = EAConfig::new(bar, material, targets, 2)
//!     .with_params(params);
//!
//! // Run optimization
//! let result = run_optimization(&config);
//!
//! println!("Best tuning error: {:.4}%", result.tuning_error);
//! for cut in &result.cuts {
//!     println!("  Cut: lambda={:.4}m, h={:.4}m", cut.lambda, cut.h);
//! }
//! ```

pub mod algorithm;
pub mod crossover;
pub mod mutation;
pub mod objective;
pub mod population;
pub mod selection;
pub mod types;

// Re-export commonly used items
pub use algorithm::{run_evolutionary_algorithm, run_optimization, EAConfig};
pub use crossover::{
    blend_crossover, heuristic_crossover, perform_crossover, single_point_crossover,
    two_point_crossover, uniform_crossover,
};
pub use mutation::{
    adaptive_length_mutation, gaussian_self_adaptive_mutation, perform_mutation,
    polynomial_mutation, uniform_mutation, FrequencyError,
};
pub use objective::{
    combined_objective_roughness, combined_objective_volume, compute_cents_error,
    compute_cents_errors, compute_roughness_penalty, compute_tuning_error, compute_volume_penalty,
    evaluate_detailed, evaluate_fitness,
};
pub use population::{
    calculate_diversity, calculate_population_stats, clamp_to_bounds, clone_individual,
    create_random_individual, create_uncut_bar_individual, get_best_individual,
    get_length_adjust_from_genes, get_top_individuals, initialize_population,
};
pub use selection::{
    rank_selection, roulette_selection, select_elite, select_mating_pairs, select_parents,
    tournament_selection,
};
pub use types::{
    AnalysisMode, BarParameters, BoundsConstraints, CrossoverMethod, DetailedEvaluation,
    EAParameters, Individual, Material, MutationMethod, OptimizationResult, PenaltyType,
    PopulationStats, ProgressUpdate, SelectionMethod, VariableBounds,
};
