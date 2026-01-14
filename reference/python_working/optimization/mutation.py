"""
Mutation Operators for Evolutionary Algorithm

Implements uniform random mutation and self-adaptive Gaussian mutation
from Section 3.3 of the paper (Eq. 17-18).
"""

from typing import List, Optional
from dataclasses import dataclass
import random
import math

from ..types import Individual, VariableBounds
from .population import clamp_to_bounds


@dataclass
class FrequencyError:
    """Frequency error info for adaptive length mutation."""
    f1_error: float  # f1_computed - f1_target (positive = too high, negative = too low)


def gaussian_random() -> float:
    """
    Generate a random number from standard Gaussian distribution.
    Using Box-Muller transform.
    """
    u1 = random.random()
    u2 = random.random()

    # Avoid log(0)
    while u1 == 0:
        u1 = random.random()

    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def uniform_mutation(
    individual: Individual,
    sigma: float,
    bounds: VariableBounds
) -> Individual:
    """
    Uniform random mutation.
    As described in paper Section 3.3:
    1. Randomly select number of genes to mutate (1 to 2N)
    2. Randomly select which genes to mutate
    3. Mutate with: c = p + sigma * r, where r is uniform [-1, 1]

    Args:
        individual: Individual to mutate
        sigma: Mutation strength (normalized to bounds)
        bounds: Variable bounds

    Returns:
        Mutated individual
    """
    genes = individual.genes.copy()
    num_genes = len(genes)

    # Step 1: Random number of genes to mutate
    num_mutate = random.randint(1, num_genes)

    # Step 2: Select which genes to mutate
    indices_to_mutate = set(random.sample(range(num_genes), num_mutate))

    # Determine if length adjustment gene is present
    has_length_adjust = bounds.max_length_trim > 0 or bounds.max_length_extend > 0
    num_cuts = int((num_genes - 1) / 2) if has_length_adjust else int(num_genes / 2)
    cut_genes_count = num_cuts * 2

    # Step 3: Mutate selected genes
    for idx in indices_to_mutate:
        # Determine bound range for this gene
        if has_length_adjust and idx == cut_genes_count:
            # This is the length adjustment gene
            range_val = bounds.max_length_trim + bounds.max_length_extend
        else:
            # Cut genes: alternating lambda and h
            is_lambda = idx % 2 == 0
            range_val = (bounds.lambda_max - bounds.lambda_min) if is_lambda else (bounds.h_max - bounds.h_min)

        # Random mutation: r is uniform [-1, 1]
        r = random.random() * 2 - 1
        genes[idx] += sigma * range_val * r

    # Clamp to bounds
    mutated_genes = clamp_to_bounds(genes, bounds)

    return Individual(
        genes=mutated_genes,
        fitness=float('inf'),
        sigmas=individual.sigmas.copy() if individual.sigmas else None
    )


def adaptive_length_mutation(
    individual: Individual,
    sigma: float,
    bounds: VariableBounds,
    freq_error: Optional[FrequencyError] = None,
    adaptive_bias: float = 0.7
) -> Individual:
    """
    Adaptive mutation with gradient-aware length adjustment.

    When mutating the length gene, biases the direction based on f1 error:
    - If f1 is too high (positive error), bias toward trimming (positive length adjust)
    - If f1 is too low (negative error), bias toward extending (negative length adjust)

    Physics: f1 is proportional to 1/L^2, so shorter bar = higher frequency.

    Args:
        individual: Individual to mutate
        sigma: Mutation strength (normalized to bounds)
        bounds: Variable bounds
        freq_error: Optional frequency error info for adaptive length mutation
        adaptive_bias: How strongly to bias toward the correct direction (0-1, default 0.7)

    Returns:
        Mutated individual
    """
    genes = individual.genes.copy()
    num_genes = len(genes)

    # Step 1: Random number of genes to mutate
    num_mutate = random.randint(1, num_genes)

    # Step 2: Select which genes to mutate
    indices_to_mutate = set(random.sample(range(num_genes), num_mutate))

    # Determine if length adjustment gene is present
    has_length_adjust = bounds.max_length_trim > 0 or bounds.max_length_extend > 0
    num_cuts = int((num_genes - 1) / 2) if has_length_adjust else int(num_genes / 2)
    cut_genes_count = num_cuts * 2
    length_gene_idx = cut_genes_count

    # Step 3: Mutate selected genes
    for idx in indices_to_mutate:
        if has_length_adjust and idx == length_gene_idx:
            # This is the length adjustment gene - use adaptive mutation
            range_val = bounds.max_length_trim + bounds.max_length_extend

            if freq_error and abs(freq_error.f1_error) > 0.001:
                # Use gradient-aware mutation for length gene
                # f1_error > 0 means f1 is too high, need to extend (negative length adjust)
                # f1_error < 0 means f1 is too low, need to trim (positive length adjust)
                desired_direction = 1 if freq_error.f1_error < 0 else -1

                # Generate biased random value
                if random.random() < adaptive_bias:
                    # Biased: move in the desired direction
                    r = desired_direction * random.random()
                else:
                    # Random exploration
                    r = random.random() * 2 - 1

                genes[idx] += sigma * range_val * r
            else:
                # No frequency error info, use standard random mutation
                r = random.random() * 2 - 1
                genes[idx] += sigma * range_val * r
        else:
            # Cut genes: alternating lambda and h - standard random mutation
            is_lambda = idx % 2 == 0
            range_val = (bounds.lambda_max - bounds.lambda_min) if is_lambda else (bounds.h_max - bounds.h_min)

            r = random.random() * 2 - 1
            genes[idx] += sigma * range_val * r

    # Clamp to bounds
    mutated_genes = clamp_to_bounds(genes, bounds)

    return Individual(
        genes=mutated_genes,
        fitness=float('inf'),
        sigmas=individual.sigmas.copy() if individual.sigmas else None
    )


def gaussian_self_adaptive_mutation(
    individual: Individual,
    phi: float,
    bounds: VariableBounds
) -> Individual:
    """
    Self-adaptive Gaussian mutation from Eq. 17-18.

    sigma_k^{t+1} = sigma_k^t * exp(tau1 * z1 + tau2 * z2)
    c_k = p_k + sigma_k^{t+1} * z3

    where:
    - tau1 = 1 / sqrt(2 * sqrt(2n))
    - tau2 = 1 / sqrt(4n)
    - z1, z2, z3 are Gaussian random numbers with standard deviation phi

    Args:
        individual: Individual to mutate (must have sigmas)
        phi: Base standard deviation for random numbers
        bounds: Variable bounds

    Returns:
        Mutated individual with updated sigmas
    """
    genes = individual.genes.copy()
    num_genes = len(genes)
    n = num_genes

    # Initialize sigmas if not present
    sigmas = individual.sigmas.copy() if individual.sigmas else [0.2] * num_genes

    # Learning rates from Eq. 18
    tau1 = 1.0 / math.sqrt(2 * math.sqrt(2 * n))
    tau2 = 1.0 / math.sqrt(4 * n)

    # Common random factor (same for all genes in this mutation)
    z1 = gaussian_random() * phi

    # Determine if length adjustment gene is present
    has_length_adjust = bounds.max_length_trim > 0 or bounds.max_length_extend > 0
    num_cuts = int((num_genes - 1) / 2) if has_length_adjust else int(num_genes / 2)
    cut_genes_count = num_cuts * 2

    for k in range(num_genes):
        # Gene-specific random factors
        z2 = gaussian_random() * phi
        z3 = gaussian_random()

        # Update sigma (Eq. 17, first line)
        sigmas[k] = sigmas[k] * math.exp(tau1 * z1 + tau2 * z2)

        # Clamp sigma to reasonable range
        sigmas[k] = max(0.001, min(1.0, sigmas[k]))

        # Mutate gene (Eq. 17, second line)
        if has_length_adjust and k == cut_genes_count:
            range_val = bounds.max_length_trim + bounds.max_length_extend
        else:
            is_lambda = k % 2 == 0
            range_val = (bounds.lambda_max - bounds.lambda_min) if is_lambda else (bounds.h_max - bounds.h_min)

        genes[k] += sigmas[k] * range_val * z3

    # Clamp to bounds
    mutated_genes = clamp_to_bounds(genes, bounds)

    return Individual(
        genes=mutated_genes,
        fitness=float('inf'),
        sigmas=sigmas
    )


def polynomial_mutation(
    individual: Individual,
    mutation_prob: float,
    eta: float,
    bounds: VariableBounds
) -> Individual:
    """
    Polynomial mutation.
    Non-uniform mutation that can produce values close to parents or at bounds.

    Args:
        individual: Individual to mutate
        mutation_prob: Probability of mutating each gene
        eta: Distribution index (higher = closer to parent)
        bounds: Variable bounds
    """
    genes = individual.genes.copy()
    num_genes = len(genes)

    has_length_adjust = bounds.max_length_trim > 0 or bounds.max_length_extend > 0
    num_cuts = int((num_genes - 1) / 2) if has_length_adjust else int(num_genes / 2)
    cut_genes_count = num_cuts * 2

    for i in range(num_genes):
        if random.random() > mutation_prob:
            continue

        if has_length_adjust and i == cut_genes_count:
            min_val = -bounds.max_length_extend
            max_val = bounds.max_length_trim
        else:
            is_lambda = i % 2 == 0
            min_val = bounds.lambda_min if is_lambda else bounds.h_min
            max_val = bounds.lambda_max if is_lambda else bounds.h_max

        x = genes[i]
        delta1 = (x - min_val) / (max_val - min_val) if max_val != min_val else 0
        delta2 = (max_val - x) / (max_val - min_val) if max_val != min_val else 0

        r = random.random()

        if r < 0.5:
            xy = 1 - delta1
            val = 2 * r + (1 - 2 * r) * (xy ** (eta + 1))
            delta = (val ** (1 / (eta + 1))) - 1
        else:
            xy = 1 - delta2
            val = 2 * (1 - r) + 2 * (r - 0.5) * (xy ** (eta + 1))
            delta = 1 - (val ** (1 / (eta + 1)))

        genes[i] = x + delta * (max_val - min_val)

    return Individual(
        genes=clamp_to_bounds(genes, bounds),
        fitness=float('inf'),
        sigmas=individual.sigmas.copy() if individual.sigmas else None
    )


def perform_mutation(
    individuals: List[Individual],
    bounds: VariableBounds,
    method: str = 'uniform',
    sigma: float = 0.1
) -> List[Individual]:
    """
    Perform mutation on a set of individuals.

    Args:
        individuals: Individuals to mutate
        bounds: Variable bounds
        method: Mutation method ('uniform' or 'gaussian')
        sigma: Mutation strength (for uniform) or phi (for gaussian)

    Returns:
        Mutated individuals
    """
    if method == 'uniform':
        return [uniform_mutation(ind, sigma, bounds) for ind in individuals]
    else:
        return [gaussian_self_adaptive_mutation(ind, sigma, bounds) for ind in individuals]
