"""
Population Management for Evolutionary Algorithm

Handles creation, manipulation, and analysis of populations of individuals.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
import random
import math

from ..types import Individual, VariableBounds, BarParameters


@dataclass
class BoundsConstraints:
    """Constraint options for creating bounds."""
    min_cut_width: float = 0.0      # Minimum spacing between cut boundaries (m)
    max_cut_width: float = 0.0      # Maximum cut width = 2*lambda (m), 0 = no limit
    min_cut_depth: float = 0.0      # Minimum depth = h0 - h (m), 0 = no limit
    max_cut_depth: float = 0.0      # Maximum depth = h0 - h (m), 0 = use bar.hMin
    max_length_trim: float = 0.0    # Maximum trim from each end (m), 0 = no trimming
    max_length_extend: float = 0.0  # Maximum extension from each end (m), 0 = no extension


@dataclass
class PopulationStats:
    """Population statistics."""
    best_fitness: float
    worst_fitness: float
    average_fitness: float
    median_fitness: float
    standard_deviation: float


def create_bounds(
    bar: BarParameters,
    num_cuts: int,
    constraints: Optional[BoundsConstraints] = None
) -> VariableBounds:
    """
    Create bounds for optimization variables based on bar parameters.

    Args:
        bar: Bar parameters
        num_cuts: Number of cuts
        constraints: Optional constraint parameters

    Returns:
        Variable bounds
    """
    if constraints is None:
        constraints = BoundsConstraints()

    # Lambda bounds
    lambda_min = 0.0
    lambda_max = bar.L / 2

    # Height bounds (remember: depth = h0 - h, so larger depth means smaller h)
    h_max = max(bar.hMin, bar.h0 - constraints.min_cut_depth) if constraints.min_cut_depth > 0 else bar.h0
    h_min = max(bar.hMin, bar.h0 - constraints.max_cut_depth) if constraints.max_cut_depth > 0 else bar.hMin

    return VariableBounds(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        h_min=h_min,
        h_max=h_max,
        min_cut_width=constraints.min_cut_width,
        max_cut_width=constraints.max_cut_width,
        min_cut_depth=constraints.min_cut_depth,
        max_cut_depth=constraints.max_cut_depth,
        max_length_trim=constraints.max_length_trim,
        max_length_extend=constraints.max_length_extend
    )


def create_uncut_bar_individual(
    num_cuts: int,
    bounds: VariableBounds,
    h0: float
) -> Individual:
    """
    Create an "uncut bar" individual - baseline with no modifications.
    All lambdas = 0 (no cuts), all heights = h0 (full thickness).

    Args:
        num_cuts: Number of cuts (determines gene array size)
        bounds: Variable bounds
        h0: Original bar thickness

    Returns:
        Individual representing an uncut bar
    """
    genes: List[float] = []

    # All cuts have lambda=0 and h=h0 (no material removed)
    for _ in range(num_cuts):
        genes.append(0.0)   # lambda = 0 (no cut extent)
        genes.append(h0)    # h = h0 (full original thickness)

    # Add length adjustment gene if enabled (set to 0 = no trim/extend)
    has_length_adjust = bounds.max_length_trim > 0 or bounds.max_length_extend > 0
    if has_length_adjust:
        genes.append(0.0)

    return Individual(genes=genes, fitness=float('inf'))


def create_random_individual(num_cuts: int, bounds: VariableBounds) -> Individual:
    """
    Create a random individual within bounds, respecting min/max cut width constraints.

    Args:
        num_cuts: Number of cuts (2 genes per cut: lambda, h)
        bounds: Variable bounds

    Returns:
        New individual with random genes
    """
    genes: List[float] = []
    min_width = bounds.min_cut_width or 0.0
    max_width = bounds.max_cut_width or 0.0

    # Generate lambdas that respect both min and max cut width constraints
    lambdas: List[float] = []

    if num_cuts == 1:
        # Single cut
        lambda_ = bounds.lambda_min + random.random() * (bounds.lambda_max - bounds.lambda_min)
        if max_width > 0:
            lambda_ = min(lambda_, max_width)
        lambdas.append(lambda_)
    else:
        # Multiple cuts: generate with spacing constraints
        current_max = bounds.lambda_max

        for i in range(num_cuts):
            remaining_cuts = num_cuts - i - 1
            reserved_space = remaining_cuts * min_width
            available_max = current_max - reserved_space

            cut_min = bounds.lambda_min + reserved_space
            cut_max = available_max

            if max_width > 0 and i > 0:
                cut_min = max(cut_min, lambdas[i - 1] - max_width)

            if cut_max < cut_min:
                cut_max = cut_min

            lambda_ = cut_min + random.random() * (cut_max - cut_min)
            lambdas.append(lambda_)

            current_max = lambda_ - min_width

    # Ensure lambdas are sorted descending
    lambdas.sort(reverse=True)

    # Build genes array with lambdas and random heights
    for i in range(num_cuts):
        genes.append(lambdas[i])
        h = bounds.h_min + random.random() * (bounds.h_max - bounds.h_min)
        genes.append(h)

    # Add length adjustment gene if enabled
    has_length_adjust = bounds.max_length_trim > 0 or bounds.max_length_extend > 0
    if has_length_adjust:
        min_val = -bounds.max_length_extend
        max_val = bounds.max_length_trim
        length_adjust = min_val + random.random() * (max_val - min_val)
        genes.append(length_adjust)

    return Individual(genes=genes, fitness=float('inf'))


def initialize_population(
    population_size: int,
    num_cuts: int,
    bounds: VariableBounds,
    seed_genes: Optional[List[float]] = None
) -> List[Individual]:
    """
    Initialize a population of random individuals, optionally seeded with initial genes.

    Args:
        population_size: Number of individuals
        num_cuts: Number of cuts per individual
        bounds: Variable bounds
        seed_genes: Optional seed genes to use for initial individual(s)

    Returns:
        Array of individuals
    """
    population: List[Individual] = []

    # If seed genes provided, create an individual from them
    if seed_genes and len(seed_genes) > 0:
        clamped_genes = clamp_to_bounds(seed_genes, bounds)
        population.append(Individual(genes=clamped_genes, fitness=float('inf')))

        # Also add some mutated variants of the seed for diversity
        num_variants = min(int(population_size * 0.2), 10)
        for _ in range(num_variants):
            if len(population) >= population_size:
                break
            variant_genes = [g * (0.95 + random.random() * 0.1) for g in clamped_genes]
            population.append(Individual(
                genes=clamp_to_bounds(variant_genes, bounds),
                fitness=float('inf')
            ))

    # Fill remaining slots with random individuals
    while len(population) < population_size:
        population.append(create_random_individual(num_cuts, bounds))

    return population


def get_best_individual(population: List[Individual]) -> Individual:
    """
    Get the best individual from a population.
    (Lowest fitness is best - we're minimizing)
    """
    return min(population, key=lambda ind: ind.fitness)


def get_top_individuals(population: List[Individual], n: int) -> List[Individual]:
    """Get the N best individuals from a population."""
    return sorted(population, key=lambda ind: ind.fitness)[:n]


def calculate_population_stats(population: List[Individual]) -> PopulationStats:
    """Calculate population statistics."""
    fitnesses = [ind.fitness for ind in population if math.isfinite(ind.fitness)]

    if not fitnesses:
        return PopulationStats(
            best_fitness=float('inf'),
            worst_fitness=float('inf'),
            average_fitness=float('inf'),
            median_fitness=float('inf'),
            standard_deviation=0.0
        )

    fitnesses.sort()

    average = sum(fitnesses) / len(fitnesses)

    if len(fitnesses) % 2 == 0:
        median = (fitnesses[len(fitnesses) // 2 - 1] + fitnesses[len(fitnesses) // 2]) / 2
    else:
        median = fitnesses[len(fitnesses) // 2]

    squared_diffs = [(f - average) ** 2 for f in fitnesses]
    variance = sum(squared_diffs) / len(fitnesses)
    std_dev = math.sqrt(variance)

    return PopulationStats(
        best_fitness=fitnesses[0],
        worst_fitness=fitnesses[-1],
        average_fitness=average,
        median_fitness=median,
        standard_deviation=std_dev
    )


def clone_individual(individual: Individual) -> Individual:
    """Deep clone an individual."""
    return Individual(
        genes=individual.genes.copy(),
        fitness=individual.fitness,
        sigmas=individual.sigmas.copy() if individual.sigmas else None
    )


def clamp_to_bounds(genes: List[float], bounds: VariableBounds) -> List[float]:
    """
    Check if genes are within bounds and clamp if necessary.
    Enforces both minimum and maximum cut width constraints.
    """
    clamped = genes.copy()
    min_width = bounds.min_cut_width or 0.0
    max_width = bounds.max_cut_width or 0.0

    # Determine if length adjustment gene is present
    has_length_adjust = bounds.max_length_trim > 0 or bounds.max_length_extend > 0
    num_cuts = int((len(clamped) - 1) / 2) if has_length_adjust else int(len(clamped) / 2)
    cut_genes_length = num_cuts * 2

    # First pass: clamp individual cut values (lambda and h)
    for i in range(0, cut_genes_length, 2):
        # Clamp lambda
        clamped[i] = max(bounds.lambda_min, min(bounds.lambda_max, clamped[i]))
        # Clamp height
        clamped[i + 1] = max(bounds.h_min, min(bounds.h_max, clamped[i + 1]))

    # Clamp length adjustment gene if present
    if has_length_adjust and len(clamped) > cut_genes_length:
        min_val = -bounds.max_length_extend
        max_val = bounds.max_length_trim
        clamped[cut_genes_length] = max(min_val, min(max_val, clamped[cut_genes_length]))

    # Second pass: enforce min/max spacing between lambdas
    if (min_width > 0 or max_width > 0) and num_cuts >= 1:
        # Extract lambdas with their original indices
        lambdas_with_idx = [(clamped[i * 2], i * 2) for i in range(num_cuts)]

        # Sort by lambda descending (outermost first)
        lambdas_with_idx.sort(key=lambda x: x[0], reverse=True)

        # Enforce constraints from outside in
        for i in range(1, len(lambdas_with_idx)):
            outer_lambda = lambdas_with_idx[i - 1][0]
            inner_lambda = lambdas_with_idx[i][0]

            # Enforce minimum spacing
            if min_width > 0:
                max_allowed = outer_lambda - min_width
                if inner_lambda > max_allowed:
                    inner_lambda = max(bounds.lambda_min, max_allowed)

            # Enforce maximum spacing
            if max_width > 0:
                min_allowed = outer_lambda - max_width
                if inner_lambda < min_allowed:
                    inner_lambda = max(bounds.lambda_min, min_allowed)

            lambdas_with_idx[i] = (inner_lambda, lambdas_with_idx[i][1])

        # Write back to clamped array
        for lambda_val, idx in lambdas_with_idx:
            clamped[idx] = lambda_val

    return clamped


def get_length_adjust_from_genes(genes: List[float], num_cuts: int) -> float:
    """
    Extract length adjustment from genes array.
    Returns 0 if length adjustment is not enabled.

    Args:
        genes: Gene array [lambda_1, h_1, ..., length_adjust?]
        num_cuts: Number of cuts

    Returns:
        Length adjustment value (m): positive = trim, negative = extend, 0 if not present
    """
    expected_cut_genes = num_cuts * 2
    if len(genes) > expected_cut_genes:
        return genes[expected_cut_genes]
    return 0.0


def calculate_diversity(population: List[Individual]) -> float:
    """
    Calculate diversity measure for population.
    Higher values indicate more diverse population.
    """
    if len(population) < 2:
        return 0.0

    num_genes = len(population[0].genes)
    total_variance = 0.0

    for g in range(num_genes):
        gene_values = [ind.genes[g] for ind in population]
        mean = sum(gene_values) / len(gene_values)
        variance = sum((v - mean) ** 2 for v in gene_values) / len(gene_values)
        total_variance += variance

    return math.sqrt(total_variance / num_genes)
