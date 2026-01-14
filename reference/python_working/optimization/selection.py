"""
Selection Operators for Evolutionary Algorithm

Implements roulette wheel selection (Eq. 15 from paper)
and other selection strategies.
"""

from typing import List, Tuple, Literal
import random
import math

from ..types import Individual


def roulette_selection(
    population: List[Individual],
    num_selections: int
) -> List[Individual]:
    """
    Roulette wheel selection (fitness proportional) from Eq. 15.

    p_i = (1/e_i) / sum(1/e_j)

    Lower fitness = higher probability of selection.

    Args:
        population: Array of individuals with fitness values
        num_selections: Number of individuals to select

    Returns:
        Selected individuals
    """
    # Filter out individuals with invalid fitness
    valid_population = [
        ind for ind in population
        if math.isfinite(ind.fitness) and ind.fitness > 0
    ]

    if not valid_population:
        # Fallback: return random selection from original population
        return [random.choice(population) for _ in range(num_selections)]

    # Calculate selection probabilities (inverse of fitness)
    inverse_fitnesses = [1.0 / ind.fitness for ind in valid_population]
    sum_inverse = sum(inverse_fitnesses)
    probabilities = [inv / sum_inverse for inv in inverse_fitnesses]

    # Calculate cumulative probabilities
    cumulative: List[float] = []
    cum_sum = 0.0
    for prob in probabilities:
        cum_sum += prob
        cumulative.append(cum_sum)

    # Select individuals using roulette wheel
    selected: List[Individual] = []
    for _ in range(num_selections):
        r = random.random()

        # Find the individual corresponding to this random number
        selected_index = 0
        for j, cum in enumerate(cumulative):
            if r <= cum:
                selected_index = j
                break

        selected.append(valid_population[selected_index])

    return selected


def tournament_selection(
    population: List[Individual],
    num_selections: int,
    tournament_size: int = 3
) -> List[Individual]:
    """
    Tournament selection.
    Selects the best individual from a random subset of the population.

    Args:
        population: Array of individuals
        num_selections: Number of individuals to select
        tournament_size: Size of each tournament (default: 3)

    Returns:
        Selected individuals
    """
    selected: List[Individual] = []

    for _ in range(num_selections):
        # Pick random individuals for tournament
        tournament = [random.choice(population) for _ in range(tournament_size)]

        # Select the best from tournament
        winner = min(tournament, key=lambda ind: ind.fitness)
        selected.append(winner)

    return selected


def rank_selection(
    population: List[Individual],
    num_selections: int,
    selection_pressure: float = 1.5
) -> List[Individual]:
    """
    Rank-based selection.
    Selection probability based on rank rather than absolute fitness.

    Args:
        population: Array of individuals
        num_selections: Number of individuals to select
        selection_pressure: Higher values favor better individuals (1.0-2.0)

    Returns:
        Selected individuals
    """
    # Sort by fitness (ascending - best first)
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    n = len(sorted_pop)

    # Calculate rank-based probabilities
    probabilities = []
    for i in range(n):
        rank = i
        prob = (2 - selection_pressure) / n + \
               2 * (selection_pressure - 1) * (n - 1 - rank) / (n * (n - 1))
        probabilities.append(prob)

    # Normalize probabilities
    sum_prob = sum(probabilities)
    normalized_probs = [p / sum_prob for p in probabilities]

    # Calculate cumulative probabilities
    cumulative: List[float] = []
    cum_sum = 0.0
    for prob in normalized_probs:
        cum_sum += prob
        cumulative.append(cum_sum)

    # Select individuals
    selected: List[Individual] = []
    for _ in range(num_selections):
        r = random.random()
        selected_index = 0
        for j, cum in enumerate(cumulative):
            if r <= cum:
                selected_index = j
                break
        selected.append(sorted_pop[selected_index])

    return selected


def select_mating_pairs(
    population: List[Individual],
    num_pairs: int,
    selection_method: Literal['roulette', 'tournament', 'rank'] = 'roulette'
) -> List[Tuple[Individual, Individual]]:
    """
    Select pairs for mating.
    Ensures each pair contains two different individuals.

    Args:
        population: Array of individuals
        num_pairs: Number of pairs to select
        selection_method: Selection method to use

    Returns:
        Array of pairs (parent1, parent2)
    """
    pairs: List[Tuple[Individual, Individual]] = []

    select_fn = {
        'roulette': roulette_selection,
        'tournament': tournament_selection,
        'rank': rank_selection
    }[selection_method]

    for _ in range(num_pairs):
        # Select two parents
        [parent1] = select_fn(population, 1)

        # Keep selecting second parent until it's different
        parent2 = parent1
        attempts = 0
        while parent2 is parent1 and attempts < 10:
            [parent2] = select_fn(population, 1)
            attempts += 1

        pairs.append((parent1, parent2))

    return pairs


def select_elite(population: List[Individual], num_elite: int) -> List[Individual]:
    """
    Elitism: select the best individuals to pass unchanged to next generation.

    Args:
        population: Current population
        num_elite: Number of elite individuals

    Returns:
        Array of elite individuals (cloned)
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    return [
        Individual(
            genes=ind.genes.copy(),
            fitness=ind.fitness,
            sigmas=ind.sigmas.copy() if ind.sigmas else None
        )
        for ind in sorted_pop[:num_elite]
    ]
