"""
Crossover Operators for Evolutionary Algorithm

Implements heuristic crossover (Eq. 16 from paper)
and other crossover strategies.
"""

from typing import List, Tuple, Literal
import random

from ..types import Individual, VariableBounds
from .population import clamp_to_bounds


def heuristic_crossover(
    parent1: Individual,
    parent2: Individual,
    bounds: VariableBounds
) -> Tuple[Individual, Individual]:
    """
    Heuristic crossover from Eq. 16.

    c1 = p1 + r * (p2 - p1)
    c2 = p2 + r * (p1 - p2)

    where r is uniform random [0, 1]

    Args:
        parent1: First parent
        parent2: Second parent
        bounds: Variable bounds for clamping

    Returns:
        Two children
    """
    num_genes = len(parent1.genes)
    r = random.random()

    child1_genes: List[float] = []
    child2_genes: List[float] = []

    for i in range(num_genes):
        p1 = parent1.genes[i]
        p2 = parent2.genes[i]

        child1_genes.append(p1 + r * (p2 - p1))
        child2_genes.append(p2 + r * (p1 - p2))

    # Clamp to bounds
    clamped_child1 = clamp_to_bounds(child1_genes, bounds)
    clamped_child2 = clamp_to_bounds(child2_genes, bounds)

    # Handle sigmas for self-adaptive mutation
    child1_sigmas = None
    child2_sigmas = None

    if parent1.sigmas and parent2.sigmas:
        child1_sigmas = [s1 + r * (s2 - s1) for s1, s2 in zip(parent1.sigmas, parent2.sigmas)]
        child2_sigmas = [s2 + r * (s1 - s2) for s1, s2 in zip(parent1.sigmas, parent2.sigmas)]

    return (
        Individual(genes=clamped_child1, fitness=float('inf'), sigmas=child1_sigmas),
        Individual(genes=clamped_child2, fitness=float('inf'), sigmas=child2_sigmas)
    )


def single_point_crossover(
    parent1: Individual,
    parent2: Individual,
    bounds: VariableBounds
) -> Tuple[Individual, Individual]:
    """
    Single-point crossover.
    Genes are swapped after a random crossover point.

    Args:
        parent1: First parent
        parent2: Second parent
        bounds: Variable bounds

    Returns:
        Two children
    """
    num_genes = len(parent1.genes)
    crossover_point = random.randint(1, num_genes - 1)

    child1_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
    child2_genes = parent2.genes[:crossover_point] + parent1.genes[crossover_point:]

    return (
        Individual(genes=clamp_to_bounds(child1_genes, bounds), fitness=float('inf')),
        Individual(genes=clamp_to_bounds(child2_genes, bounds), fitness=float('inf'))
    )


def two_point_crossover(
    parent1: Individual,
    parent2: Individual,
    bounds: VariableBounds
) -> Tuple[Individual, Individual]:
    """
    Two-point crossover.
    Genes between two random points are swapped.
    """
    num_genes = len(parent1.genes)

    point1 = random.randint(0, num_genes - 1)
    point2 = random.randint(0, num_genes - 1)

    if point1 > point2:
        point1, point2 = point2, point1

    child1_genes = parent1.genes[:point1] + parent2.genes[point1:point2] + parent1.genes[point2:]
    child2_genes = parent2.genes[:point1] + parent1.genes[point1:point2] + parent2.genes[point2:]

    return (
        Individual(genes=clamp_to_bounds(child1_genes, bounds), fitness=float('inf')),
        Individual(genes=clamp_to_bounds(child2_genes, bounds), fitness=float('inf'))
    )


def uniform_crossover(
    parent1: Individual,
    parent2: Individual,
    bounds: VariableBounds,
    mixing_ratio: float = 0.5
) -> Tuple[Individual, Individual]:
    """
    Uniform crossover.
    Each gene is randomly chosen from either parent.
    """
    num_genes = len(parent1.genes)

    child1_genes: List[float] = []
    child2_genes: List[float] = []

    for i in range(num_genes):
        if random.random() < mixing_ratio:
            child1_genes.append(parent1.genes[i])
            child2_genes.append(parent2.genes[i])
        else:
            child1_genes.append(parent2.genes[i])
            child2_genes.append(parent1.genes[i])

    return (
        Individual(genes=clamp_to_bounds(child1_genes, bounds), fitness=float('inf')),
        Individual(genes=clamp_to_bounds(child2_genes, bounds), fitness=float('inf'))
    )


def blend_crossover(
    parent1: Individual,
    parent2: Individual,
    bounds: VariableBounds,
    alpha: float = 0.5
) -> Tuple[Individual, Individual]:
    """
    Blend crossover (BLX-alpha).
    Children are generated in an extended range around parents.

    Args:
        alpha: Extension parameter (default 0.5)
    """
    num_genes = len(parent1.genes)

    child1_genes: List[float] = []
    child2_genes: List[float] = []

    for i in range(num_genes):
        p1 = parent1.genes[i]
        p2 = parent2.genes[i]

        min_val = min(p1, p2)
        max_val = max(p1, p2)
        range_val = max_val - min_val

        extended_min = min_val - alpha * range_val
        extended_max = max_val + alpha * range_val

        child1_genes.append(extended_min + random.random() * (extended_max - extended_min))
        child2_genes.append(extended_min + random.random() * (extended_max - extended_min))

    return (
        Individual(genes=clamp_to_bounds(child1_genes, bounds), fitness=float('inf')),
        Individual(genes=clamp_to_bounds(child2_genes, bounds), fitness=float('inf'))
    )


def perform_crossover(
    pairs: List[Tuple[Individual, Individual]],
    bounds: VariableBounds,
    method: Literal['heuristic', 'single', 'two', 'uniform', 'blend'] = 'heuristic'
) -> List[Individual]:
    """
    Perform crossover on multiple parent pairs.

    Args:
        pairs: Array of parent pairs
        bounds: Variable bounds
        method: Crossover method to use

    Returns:
        Array of children
    """
    crossover_fn = {
        'heuristic': heuristic_crossover,
        'single': single_point_crossover,
        'two': two_point_crossover,
        'uniform': uniform_crossover,
        'blend': blend_crossover
    }[method]

    children: List[Individual] = []

    for parent1, parent2 in pairs:
        child1, child2 = crossover_fn(parent1, parent2, bounds)
        children.extend([child1, child2])

    return children
