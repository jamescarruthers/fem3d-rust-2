"""
Evolutionary Algorithm for Bar Tuning Optimization

Main optimization loop implementing the algorithm from Section 3 of the paper.
Combines elitism, selection, crossover, and mutation to find optimal
undercut geometries for target frequencies.

Uses multithreading for parallel fitness evaluation.
"""

from typing import List, Optional, Callable, Literal
from dataclasses import dataclass
import math

from ..types import (
    Individual,
    BarParameters,
    Material,
    EAParameters,
    OptimizationResult,
    ProgressUpdate,
    AnalysisMode,
)
from ..physics.frequencies import compute_frequencies_from_genes, batch_compute_fitness
from ..physics.bar_profile import genes_to_cuts

from .population import (
    create_bounds,
    create_uncut_bar_individual,
    initialize_population,
    get_best_individual,
    calculate_population_stats,
    clone_individual,
    get_length_adjust_from_genes,
    BoundsConstraints,
)
from .selection import select_elite, select_mating_pairs
from .crossover import heuristic_crossover
from .mutation import uniform_mutation, gaussian_self_adaptive_mutation, adaptive_length_mutation, FrequencyError
from .penalties import compute_volume_penalty, compute_roughness_penalty
from .objective import evaluate_detailed


@dataclass
class EAConfig:
    """Configuration for the evolutionary algorithm."""
    bar: BarParameters
    material: Material
    target_frequencies: List[float]
    num_cuts: int
    penalty_type: Literal['volume', 'roughness', 'none'] = 'none'
    penalty_weight: float = 0.0
    ea_params: Optional[EAParameters] = None
    seed_genes: Optional[List[float]] = None
    on_progress: Optional[Callable[[ProgressUpdate], None]] = None
    should_stop: Optional[Callable[[], bool]] = None


def _compute_frequencies_and_errors(
    genes: List[float],
    bar: BarParameters,
    material: Material,
    target_frequencies: List[float],
    num_elements: int,
    num_cuts: int,
    analysis_mode: AnalysisMode = AnalysisMode.BEAM_2D,
    ny: int = 2,
    nz: int = 2
) -> dict:
    """Compute frequencies and cents errors for an individual."""
    try:
        length_trim = get_length_adjust_from_genes(genes, num_cuts)
        computed_frequencies = compute_frequencies_from_genes(
            genes,
            bar,
            material,
            len(target_frequencies),
            num_elements,
            num_cuts,
            analysis_mode,
            ny,
            nz
        )

        errors_in_cents = []
        for i, comp in enumerate(computed_frequencies):
            target = target_frequencies[i] if i < len(target_frequencies) else 0
            if target and target > 0:
                errors_in_cents.append(1200 * math.log2(comp / target))
            else:
                errors_in_cents.append(0.0)

        return {
            "computed_frequencies": computed_frequencies,
            "errors_in_cents": errors_in_cents,
            "length_trim": length_trim
        }
    except Exception:
        return {
            "computed_frequencies": [],
            "errors_in_cents": [],
            "length_trim": 0.0
        }


def _batch_evaluate_population(
    population: List[Individual],
    bar: BarParameters,
    material: Material,
    target_frequencies: List[float],
    penalty_type: Literal['volume', 'roughness', 'none'],
    penalty_weight: float,
    num_elements: int,
    f1_priority: float = 1.0,
    num_cuts: int = 1,
    max_workers: int = 0,
    analysis_mode: AnalysisMode = AnalysisMode.BEAM_2D,
    ny: int = 2,
    nz: int = 2
) -> List[Individual]:
    """
    Batch evaluate population fitness using multithreading.
    """
    genes_array = [ind.genes for ind in population]
    tuning_errors = batch_compute_fitness(
        genes_array,
        bar,
        material,
        target_frequencies,
        num_elements,
        f1_priority,
        num_cuts,
        max_workers,
        analysis_mode,
        ny,
        nz
    )

    # Apply penalties if needed
    result: List[Individual] = []
    for i, ind in enumerate(population):
        fitness = tuning_errors[i]

        if penalty_type != 'none' and penalty_weight > 0:
            # Extract cut genes only (exclude length trim)
            cut_genes = ind.genes[:num_cuts * 2]
            cuts = genes_to_cuts(cut_genes)

            # Get effective bar length if length adjustment is used
            length_adjust = get_length_adjust_from_genes(ind.genes, num_cuts)
            effective_L = bar.L - 2 * length_adjust

            if penalty_type == 'volume':
                penalty = compute_volume_penalty(cuts, effective_L, bar.h0)
                fitness = (1 - penalty_weight) * fitness + penalty_weight * penalty
            else:
                penalty = compute_roughness_penalty(cuts, bar.h0)
                fitness = (1 - penalty_weight) * fitness + penalty_weight * penalty

        result.append(Individual(
            genes=ind.genes.copy(),
            fitness=fitness,
            sigmas=ind.sigmas.copy() if ind.sigmas else None
        ))

    return result


def run_evolutionary_algorithm(config: EAConfig) -> OptimizationResult:
    """
    Run the evolutionary algorithm.

    Args:
        config: Algorithm configuration

    Returns:
        Optimization result
    """
    bar = config.bar
    material = config.material
    original_target_frequencies = config.target_frequencies
    num_cuts = config.num_cuts
    penalty_type = config.penalty_type
    penalty_weight = config.penalty_weight
    ea_params = config.ea_params or get_default_ea_parameters(num_cuts)
    seed_genes = config.seed_genes
    on_progress = config.on_progress
    should_stop = config.should_stop

    # Apply frequency offset for 2D/3D calibration
    # If offset is positive, we aim for higher 2D frequencies to match 3D
    offset = ea_params.frequency_offset
    target_frequencies = [f * (1 + offset) for f in original_target_frequencies]

    bounds_constraints = BoundsConstraints(
        min_cut_width=ea_params.min_cut_width,
        max_cut_width=ea_params.max_cut_width,
        min_cut_depth=ea_params.min_cut_depth,
        max_cut_depth=ea_params.max_cut_depth,
        max_length_trim=ea_params.max_length_trim,
        max_length_extend=ea_params.max_length_extend
    )
    bounds = create_bounds(bar, num_cuts, bounds_constraints)

    f1_priority = ea_params.f1_priority
    has_length_adjust = ea_params.max_length_trim > 0 or ea_params.max_length_extend > 0
    max_workers = ea_params.max_workers
    analysis_mode = ea_params.analysis_mode
    ny = ea_params.num_elements_y
    nz = ea_params.num_elements_z

    # Report Generation 0: uncut bar baseline
    if on_progress:
        uncut_bar = create_uncut_bar_individual(num_cuts, bounds, bar.h0)
        [evaluated_uncut] = _batch_evaluate_population(
            [uncut_bar], bar, material, target_frequencies,
            penalty_type, penalty_weight, ea_params.num_elements, f1_priority, num_cuts, max_workers,
            analysis_mode, ny, nz
        )
        freq_data = _compute_frequencies_and_errors(
            evaluated_uncut.genes, bar, material, target_frequencies, ea_params.num_elements, num_cuts,
            analysis_mode, ny, nz
        )
        on_progress(ProgressUpdate(
            generation=0,
            best_fitness=evaluated_uncut.fitness,
            best_individual=evaluated_uncut,
            average_fitness=evaluated_uncut.fitness,
            computed_frequencies=freq_data["computed_frequencies"],
            errors_in_cents=freq_data["errors_in_cents"],
            length_trim=freq_data["length_trim"]
        ))

    # Initialize population (with optional seed)
    population = initialize_population(ea_params.population_size, num_cuts, bounds, seed_genes)

    # Evaluate initial population
    population = _batch_evaluate_population(
        population, bar, material, target_frequencies,
        penalty_type, penalty_weight, ea_params.num_elements, f1_priority, num_cuts, max_workers,
        analysis_mode, ny, nz
    )

    # Calculate percentages for different operations
    num_elite = max(1, int(ea_params.population_size * ea_params.elitism_percent / 100))
    num_crossover = int(ea_params.population_size * ea_params.crossover_percent / 100)
    num_crossover_pairs = (num_crossover + 1) // 2

    best_ever = get_best_individual(population)
    generation = 0

    # Main evolution loop
    while generation < ea_params.max_generations:
        # Check stopping condition
        if should_stop and should_stop():
            break

        # Check if target error reached
        if best_ever.fitness <= ea_params.target_error:
            break

        # Create next generation
        next_generation: List[Individual] = []

        # 1. Elitism: Keep best individuals unchanged
        elite = select_elite(population, num_elite)
        next_generation.extend(elite)

        # 2. Crossover: Select parents and create children
        new_offspring: List[Individual] = []
        if num_crossover > 0:
            mating_pairs = select_mating_pairs(population, num_crossover_pairs, 'roulette')

            for parent1, parent2 in mating_pairs:
                child1, child2 = heuristic_crossover(parent1, parent2, bounds)
                new_offspring.append(child1)
                if len(next_generation) + len(new_offspring) < ea_params.population_size:
                    new_offspring.append(child2)

        # 3. Mutation: Select individuals and mutate
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        while len(next_generation) + len(new_offspring) < ea_params.population_size:
            idx = int(len(sorted_pop) * min(0.5, (num_elite + num_crossover) / ea_params.population_size) *
                     (1 + 0.5 * (1 - len(next_generation) / ea_params.population_size)))
            idx = min(idx, len(sorted_pop) - 1)
            parent = sorted_pop[idx]

            # Use adaptive mutation if length adjustment is enabled
            if has_length_adjust:
                parent_freqs = compute_frequencies_from_genes(
                    parent.genes, bar, material, 1, ea_params.num_elements, num_cuts,
                    analysis_mode, ny, nz
                )
                f1_error = parent_freqs[0] - target_frequencies[0] if parent_freqs else 0
                freq_error = FrequencyError(f1_error=f1_error)
                mutant = adaptive_length_mutation(parent, ea_params.mutation_strength, bounds, freq_error)
            else:
                mutant = uniform_mutation(parent, ea_params.mutation_strength, bounds)
            new_offspring.append(mutant)

        # Batch evaluate all new offspring at once
        if new_offspring:
            evaluated_offspring = _batch_evaluate_population(
                new_offspring, bar, material, target_frequencies,
                penalty_type, penalty_weight, ea_params.num_elements, f1_priority, num_cuts, max_workers,
                analysis_mode, ny, nz
            )
            next_generation.extend(evaluated_offspring)

        # Update population
        population = next_generation

        # Update best ever
        current_best = get_best_individual(population)
        if current_best.fitness < best_ever.fitness:
            best_ever = clone_individual(current_best)

        generation += 1

        # Report progress
        if on_progress:
            stats = calculate_population_stats(population)
            freq_data = _compute_frequencies_and_errors(
                best_ever.genes, bar, material, target_frequencies, ea_params.num_elements, num_cuts,
                analysis_mode, ny, nz
            )
            on_progress(ProgressUpdate(
                generation=generation,
                best_fitness=best_ever.fitness,
                best_individual=clone_individual(best_ever),
                average_fitness=stats.average_fitness,
                computed_frequencies=freq_data["computed_frequencies"],
                errors_in_cents=freq_data["errors_in_cents"],
                length_trim=freq_data["length_trim"]
            ))

    # Get detailed results for best solution
    length_adjust = get_length_adjust_from_genes(best_ever.genes, num_cuts)
    effective_length = bar.L - 2 * length_adjust

    # Create effective bar for detailed evaluation
    effective_bar = BarParameters(
        L=effective_length,
        b=bar.b,
        h0=bar.h0,
        hMin=bar.hMin
    ) if length_adjust != 0 else bar

    # Extract only cut genes
    cut_genes = best_ever.genes[:num_cuts * 2]

    # Evaluate against ORIGINAL targets (not offset-adjusted) for accurate reporting
    detailed = evaluate_detailed(
        cut_genes,
        effective_bar,
        material,
        original_target_frequencies,
        penalty_type,
        penalty_weight,
        ea_params.num_elements,
        num_cuts
    )

    return OptimizationResult(
        best_individual=best_ever,
        cuts=genes_to_cuts(cut_genes),
        computed_frequencies=detailed.computed_frequencies,
        target_frequencies=original_target_frequencies,  # Report original targets
        tuning_error=detailed.tuning_error,
        max_error_cents=detailed.max_cents_error,
        errors_in_cents=detailed.cents_errors,
        volume_percent=detailed.volume_penalty,
        roughness_percent=detailed.roughness_penalty,
        generations=generation,
        length_trim=length_adjust,
        effective_length=effective_length
    )


def run_adaptive_evolution(config: EAConfig) -> OptimizationResult:
    """
    Run optimization with adaptive mutation.
    Uses self-adaptive Gaussian mutation for potentially better convergence.
    """
    bar = config.bar
    material = config.material
    original_target_frequencies = config.target_frequencies
    num_cuts = config.num_cuts
    penalty_type = config.penalty_type
    penalty_weight = config.penalty_weight
    ea_params = config.ea_params or get_default_ea_parameters(num_cuts)

    # Apply frequency offset for 2D/3D calibration
    offset = ea_params.frequency_offset
    target_frequencies = [f * (1 + offset) for f in original_target_frequencies]
    on_progress = config.on_progress
    should_stop = config.should_stop

    bounds_constraints = BoundsConstraints(
        min_cut_width=ea_params.min_cut_width,
        max_cut_width=ea_params.max_cut_width,
        min_cut_depth=ea_params.min_cut_depth,
        max_cut_depth=ea_params.max_cut_depth,
        max_length_trim=ea_params.max_length_trim,
        max_length_extend=ea_params.max_length_extend
    )
    bounds = create_bounds(bar, num_cuts, bounds_constraints)

    f1_priority = ea_params.f1_priority
    has_length_adjust = ea_params.max_length_trim > 0 or ea_params.max_length_extend > 0
    num_genes = num_cuts * 2 + 1 if has_length_adjust else num_cuts * 2
    max_workers = ea_params.max_workers
    analysis_mode = ea_params.analysis_mode
    ny = ea_params.num_elements_y
    nz = ea_params.num_elements_z

    # Report Generation 0: uncut bar baseline
    if on_progress:
        uncut_bar = create_uncut_bar_individual(num_cuts, bounds, bar.h0)
        [evaluated_uncut] = _batch_evaluate_population(
            [uncut_bar], bar, material, target_frequencies,
            penalty_type, penalty_weight, ea_params.num_elements, f1_priority, num_cuts, max_workers,
            analysis_mode, ny, nz
        )
        freq_data = _compute_frequencies_and_errors(
            evaluated_uncut.genes, bar, material, target_frequencies, ea_params.num_elements, num_cuts,
            analysis_mode, ny, nz
        )
        on_progress(ProgressUpdate(
            generation=0,
            best_fitness=evaluated_uncut.fitness,
            best_individual=evaluated_uncut,
            average_fitness=evaluated_uncut.fitness,
            computed_frequencies=freq_data["computed_frequencies"],
            errors_in_cents=freq_data["errors_in_cents"],
            length_trim=freq_data["length_trim"]
        ))

    # Initialize population with sigmas
    population = initialize_population(ea_params.population_size, num_cuts, bounds)
    for ind in population:
        ind.sigmas = [0.2] * num_genes

    # Evaluate initial population
    population = _batch_evaluate_population(
        population, bar, material, target_frequencies,
        penalty_type, penalty_weight, ea_params.num_elements, f1_priority, num_cuts, max_workers,
        analysis_mode, ny, nz
    )

    num_elite = max(1, int(ea_params.population_size * ea_params.elitism_percent / 100))

    best_ever = get_best_individual(population)
    generation = 0

    while generation < ea_params.max_generations:
        if should_stop and should_stop():
            break
        if best_ever.fitness <= ea_params.target_error:
            break

        next_generation: List[Individual] = []

        # Elitism
        elite = select_elite(population, num_elite)
        next_generation.extend(elite)

        # Generate offspring through mutation only (mu + lambda strategy)
        new_offspring: List[Individual] = []
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)

        while len(next_generation) + len(new_offspring) < ea_params.population_size:
            idx = int(len(sorted_pop) * 0.5 * (1 - len(new_offspring) / ea_params.population_size))
            idx = min(idx, len(sorted_pop) - 1)
            parent = sorted_pop[idx]
            mutant = gaussian_self_adaptive_mutation(parent, ea_params.mutation_strength, bounds)
            new_offspring.append(mutant)

        # Batch evaluate all new offspring at once
        if new_offspring:
            evaluated_offspring = _batch_evaluate_population(
                new_offspring, bar, material, target_frequencies,
                penalty_type, penalty_weight, ea_params.num_elements, f1_priority, num_cuts, max_workers,
                analysis_mode, ny, nz
            )
            next_generation.extend(evaluated_offspring)

        population = next_generation

        current_best = get_best_individual(population)
        if current_best.fitness < best_ever.fitness:
            best_ever = clone_individual(current_best)

        generation += 1

        if on_progress:
            stats = calculate_population_stats(population)
            freq_data = _compute_frequencies_and_errors(
                best_ever.genes, bar, material, target_frequencies, ea_params.num_elements, num_cuts,
                analysis_mode, ny, nz
            )
            on_progress(ProgressUpdate(
                generation=generation,
                best_fitness=best_ever.fitness,
                best_individual=clone_individual(best_ever),
                average_fitness=stats.average_fitness,
                computed_frequencies=freq_data["computed_frequencies"],
                errors_in_cents=freq_data["errors_in_cents"],
                length_trim=freq_data["length_trim"]
            ))

    # Get detailed results
    length_adjust = get_length_adjust_from_genes(best_ever.genes, num_cuts)
    effective_length = bar.L - 2 * length_adjust

    effective_bar = BarParameters(
        L=effective_length,
        b=bar.b,
        h0=bar.h0,
        hMin=bar.hMin
    ) if length_adjust != 0 else bar

    cut_genes = best_ever.genes[:num_cuts * 2]

    # Evaluate against ORIGINAL targets (not offset-adjusted) for accurate reporting
    detailed = evaluate_detailed(
        cut_genes,
        effective_bar,
        material,
        original_target_frequencies,
        penalty_type,
        penalty_weight,
        ea_params.num_elements,
        num_cuts
    )

    return OptimizationResult(
        best_individual=best_ever,
        cuts=genes_to_cuts(cut_genes),
        computed_frequencies=detailed.computed_frequencies,
        target_frequencies=original_target_frequencies,  # Report original targets
        tuning_error=detailed.tuning_error,
        max_error_cents=detailed.max_cents_error,
        errors_in_cents=detailed.cents_errors,
        volume_percent=detailed.volume_penalty,
        roughness_percent=detailed.roughness_penalty,
        generations=generation,
        length_trim=length_adjust,
        effective_length=effective_length
    )


def get_default_ea_parameters(num_cuts: int) -> EAParameters:
    """Get default EA parameters based on problem size."""
    return EAParameters(
        population_size=max(30, num_cuts * 10),
        elitism_percent=10.0,
        crossover_percent=30.0,
        mutation_percent=60.0,
        mutation_strength=0.1,
        max_generations=100,
        target_error=0.01,  # 0.01% error
        num_elements=150,
        f1_priority=1.0,
        min_cut_width=0.0,
        max_cut_width=0.0,
        min_cut_depth=0.0,
        max_cut_depth=0.0,
        max_length_trim=0.0,
        max_length_extend=0.0,
        max_workers=0  # 0 = auto
    )
