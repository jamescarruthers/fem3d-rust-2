//! Optimization module for bar tuning.
//!
//! This module provides multiple optimization strategies for finding optimal
//! undercut bar geometries to achieve target frequencies:
//!
//! - **Evolutionary Algorithm (EA)**: Global search using genetic operators
//! - **Surrogate Optimization**: RBF-based surrogate model for expensive 3D FEM
//! - **Hybrid Strategies**: Combining fast 2D analysis with 3D refinement
//!
//! The surrogate and hybrid approaches are based on Soares et al. 2021:
//! "Tuning of bending and torsional modes of bars used in mallet percussion instruments"
//!
//! # Example: Evolutionary Algorithm
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
//!
//! # Example: Hybrid 2D→3D Optimization (Recommended)
//!
//! ```ignore
//! use fem3d_rust_2::optimization::{
//!     HybridConfig, OptimizationStrategy, run_hybrid_optimization,
//!     BarParameters, Material, EAParameters, SurrogateConfig,
//! };
//!
//! let bar = BarParameters::new(0.35, 0.05, 0.01, 0.002);
//! let material = Material::aluminum();
//! let targets = vec![175.0, 700.0, 1750.0];
//!
//! let config = HybridConfig::new(bar, material, targets, 2)
//!     .with_strategy(OptimizationStrategy::Hybrid2Dto3D {
//!         ea_params: EAParameters::default(),
//!         surrogate_config: SurrogateConfig::default(),
//!         frequency_correction: 1.0,
//!     })
//!     .with_verbose(true);
//!
//! let result = run_hybrid_optimization(&config);
//! println!("Max error: {:.1} cents", result.max_error_cents);
//! ```

pub mod algorithm;
pub mod crossover;
pub mod hybrid;
pub mod materials;
pub mod mutation;
pub mod objective;
pub mod population;
pub mod sampling;
pub mod selection;
pub mod surrogate;
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
    EAParameters, Individual, Material, MaterialCategory, MutationMethod, OptimizationResult,
    PenaltyType, PopulationStats, ProgressUpdate, SelectionMethod, VariableBounds,
};
pub use materials::{
    calculate_shear_modulus, get_all_materials, get_material, get_materials_by_category,
    KAPPA, MATERIAL_KEYS,
};

// Surrogate optimization (Soares et al. 2021)
pub use surrogate::{
    run_surrogate_optimization, AlphaSchedule, RbfKernel, SurrogateConfig, SurrogateModel,
    SurrogateResult,
};

// Sampling methods
pub use sampling::{grid_sample, latin_hypercube_sample, random_sample, sobol_sample};

// Hybrid optimization strategies
pub use hybrid::{
    run_hybrid_optimization, HybridConfig, HybridResult, OptimizationStrategy,
};
