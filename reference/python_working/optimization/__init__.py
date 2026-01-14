"""Optimization module for evolutionary algorithm implementation."""

from .population import (
    create_bounds,
    create_uncut_bar_individual,
    create_random_individual,
    initialize_population,
    get_best_individual,
    get_top_individuals,
    calculate_population_stats,
    clone_individual,
    clamp_to_bounds,
    get_length_adjust_from_genes,
    calculate_diversity,
    BoundsConstraints,
    PopulationStats,
)

from .selection import (
    roulette_selection,
    tournament_selection,
    rank_selection,
    select_mating_pairs,
    select_elite,
)

from .crossover import (
    heuristic_crossover,
    single_point_crossover,
    two_point_crossover,
    uniform_crossover,
    blend_crossover,
    perform_crossover,
)

from .mutation import (
    uniform_mutation,
    adaptive_length_mutation,
    gaussian_self_adaptive_mutation,
    polynomial_mutation,
    perform_mutation,
    FrequencyError,
)

from .penalties import (
    compute_volume_penalty,
    compute_roughness_penalty,
    compute_both_penalties,
)

from .objective import (
    compute_tuning_error,
    compute_max_tuning_error,
    combined_objective_volume,
    combined_objective_roughness,
    evaluate_fitness,
    evaluate_population,
    evaluate_detailed,
)

from .algorithm import (
    run_evolutionary_algorithm,
    run_adaptive_evolution,
    get_default_ea_parameters,
    EAConfig,
)

__all__ = [
    # Population
    "create_bounds",
    "create_uncut_bar_individual",
    "create_random_individual",
    "initialize_population",
    "get_best_individual",
    "get_top_individuals",
    "calculate_population_stats",
    "clone_individual",
    "clamp_to_bounds",
    "get_length_adjust_from_genes",
    "calculate_diversity",
    "BoundsConstraints",
    "PopulationStats",
    # Selection
    "roulette_selection",
    "tournament_selection",
    "rank_selection",
    "select_mating_pairs",
    "select_elite",
    # Crossover
    "heuristic_crossover",
    "single_point_crossover",
    "two_point_crossover",
    "uniform_crossover",
    "blend_crossover",
    "perform_crossover",
    # Mutation
    "uniform_mutation",
    "adaptive_length_mutation",
    "gaussian_self_adaptive_mutation",
    "polynomial_mutation",
    "perform_mutation",
    "FrequencyError",
    # Penalties
    "compute_volume_penalty",
    "compute_roughness_penalty",
    "compute_both_penalties",
    # Objective
    "compute_tuning_error",
    "compute_max_tuning_error",
    "combined_objective_volume",
    "combined_objective_roughness",
    "evaluate_fitness",
    "evaluate_population",
    "evaluate_detailed",
    # Algorithm
    "run_evolutionary_algorithm",
    "run_adaptive_evolution",
    "get_default_ea_parameters",
    "EAConfig",
]
