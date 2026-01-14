"""
Frequency Computation Module

Main functions for computing natural frequencies of undercut bars
using the FEM model. Supports both 2D Timoshenko beam and 3D solid
element analysis.
"""

from typing import List, Optional
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from ..types import BarParameters, Material, AnalysisMode
from .bar_profile import genes_to_cuts
from .fem_assembly import assemble_global_matrices, solve_generalized_eigenvalue
from .fem_3d import compute_frequencies_3d


def compute_frequencies(
    element_heights: List[float],
    le: float,
    b: float,
    E: float,
    rho: float,
    nu: float,
    num_modes: int,
    analysis_mode: AnalysisMode = AnalysisMode.BEAM_2D,
    ny: int = 2,
    nz: int = 2
) -> List[float]:
    """
    Compute natural frequencies for a bar with given element heights.

    Args:
        element_heights: Height of each finite element (m)
        le: Length of each element (m)
        b: Bar width (m)
        E: Young's modulus (Pa)
        rho: Density (kg/m^3)
        nu: Poisson's ratio
        num_modes: Number of modes to extract
        analysis_mode: BEAM_2D (fast) or SOLID_3D (accurate)
        ny: Number of elements in width direction (3D only)
        nz: Number of elements in thickness direction (3D only)

    Returns:
        List of natural frequencies in Hz
    """
    if analysis_mode == AnalysisMode.SOLID_3D:
        # 3D solid element analysis
        length = le * len(element_heights)
        return compute_frequencies_3d(
            element_heights, length, b, E, rho, nu, num_modes, ny, nz
        )
    else:
        # 2D Timoshenko beam analysis (default)
        K, M = assemble_global_matrices(element_heights, le, b, E, rho, nu)
        return solve_generalized_eigenvalue(K, M, num_modes)


def compute_frequencies_from_genes(
    genes: List[float],
    bar: BarParameters,
    material: Material,
    num_modes: int,
    num_elements: int,
    num_cuts: int = 0,
    analysis_mode: AnalysisMode = AnalysisMode.BEAM_2D,
    ny: int = 2,
    nz: int = 2
) -> List[float]:
    """
    Compute frequencies directly from cut parameters (genes).
    Combines profile generation and frequency computation.

    Args:
        genes: Flat array [lambda_1, h_1, lambda_2, h_2, ..., length_adjust?]
        bar: Bar parameters
        material: Material properties
        num_modes: Number of modes to extract
        num_elements: Number of finite elements
        num_cuts: Number of cuts (for determining length adjustment gene)
        analysis_mode: BEAM_2D (fast) or SOLID_3D (accurate)
        ny: Number of elements in width direction (3D only)
        nz: Number of elements in thickness direction (3D only)

    Returns:
        List of natural frequencies in Hz
    """
    # Handle length adjustment if present
    bar_length = bar.L
    if num_cuts > 0 and len(genes) > num_cuts * 2:
        length_adjust = genes[num_cuts * 2]
        bar_length = bar.L - 2 * length_adjust

    # Parse genes into cuts
    cut_genes = genes[:num_cuts * 2] if num_cuts > 0 else genes
    cuts = genes_to_cuts(cut_genes)

    # Sort by lambda descending (largest first)
    cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    # Generate element heights
    le = bar_length / num_elements
    center_x = bar_length / 2

    element_heights: List[float] = []
    for e in range(num_elements):
        x_mid = (e + 0.5) * le
        dist_from_center = abs(x_mid - center_x)

        # Find all cuts that contain this point, return the innermost one's height
        innermost_h = bar.h0
        for cut in cuts:
            if cut.lambda_ > 0 and dist_from_center <= cut.lambda_:
                innermost_h = cut.h

        element_heights.append(innermost_h)

    return compute_frequencies(
        element_heights,
        le,
        bar.b,
        material.E,
        material.rho,
        material.nu,
        num_modes,
        analysis_mode,
        ny,
        nz
    )


def _compute_single_fitness(
    genes: List[float],
    bar_length: float,
    bar_width: float,
    h0: float,
    num_elements: int,
    E: float,
    rho: float,
    nu: float,
    target_frequencies: List[float],
    f1_priority: float,
    num_cuts: int,
    analysis_mode: AnalysisMode = AnalysisMode.BEAM_2D,
    ny: int = 2,
    nz: int = 2
) -> float:
    """
    Compute fitness for a single individual.
    Internal function used by batch_compute_fitness.
    """
    # Handle length adjustment if present
    effective_length = bar_length
    if num_cuts > 0 and len(genes) > num_cuts * 2:
        length_adjust = genes[num_cuts * 2]
        effective_length = bar_length - 2 * length_adjust

    # Parse genes into cuts
    cut_genes = genes[:num_cuts * 2] if num_cuts > 0 else genes
    cuts = genes_to_cuts(cut_genes)
    cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    # Generate element heights
    le = effective_length / num_elements
    center_x = effective_length / 2

    element_heights: List[float] = []
    for e in range(num_elements):
        x_mid = (e + 0.5) * le
        dist_from_center = abs(x_mid - center_x)

        innermost_h = h0
        for cut in cuts:
            if cut.lambda_ > 0 and dist_from_center <= cut.lambda_:
                innermost_h = cut.h

        element_heights.append(innermost_h)

    # Compute frequencies
    try:
        frequencies = compute_frequencies(
            element_heights,
            le,
            bar_width,
            E,
            rho,
            nu,
            len(target_frequencies),
            analysis_mode,
            ny,
            nz
        )
    except Exception:
        return float('inf')

    # Compute weighted tuning error
    num_modes = len(target_frequencies)
    if len(frequencies) < num_modes:
        return float('inf')

    weighted_sum_sq_error = 0.0
    total_weight = 0.0
    for m in range(num_modes):
        weight = f1_priority if m == 0 else 1.0
        rel_error = (frequencies[m] - target_frequencies[m]) / target_frequencies[m]
        weighted_sum_sq_error += weight * rel_error * rel_error
        total_weight += weight

    if total_weight > 0:
        return 100.0 * weighted_sum_sq_error / total_weight
    else:
        return float('inf')


def batch_compute_fitness(
    genes_array: List[List[float]],
    bar: BarParameters,
    material: Material,
    target_frequencies: List[float],
    num_elements: int,
    f1_priority: float = 1.0,
    num_cuts: int = 1,
    max_workers: int = 0,
    analysis_mode: AnalysisMode = AnalysisMode.BEAM_2D,
    ny: int = 2,
    nz: int = 2
) -> List[float]:
    """
    Batch compute fitness for entire population using multithreading.

    Args:
        genes_array: List of gene arrays, one per individual
        bar: Bar parameters
        material: Material properties
        target_frequencies: Target frequencies (Hz)
        num_elements: Number of FEM elements
        f1_priority: Weight multiplier for f1 (>1 prioritizes f1)
        num_cuts: Number of cuts per individual
        max_workers: Maximum number of worker threads (0 = auto)
        analysis_mode: BEAM_2D (fast) or SOLID_3D (accurate)
        ny: Number of elements in width direction (3D only)
        nz: Number of elements in thickness direction (3D only)

    Returns:
        List of fitness values for each individual
    """
    if max_workers <= 0:
        max_workers = min(os.cpu_count() or 4, len(genes_array))

    # Use multithreading for parallel fitness evaluation
    fitness_values = [float('inf')] * len(genes_array)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                _compute_single_fitness,
                genes,
                bar.L,
                bar.b,
                bar.h0,
                num_elements,
                material.E,
                material.rho,
                material.nu,
                target_frequencies,
                f1_priority,
                num_cuts,
                analysis_mode,
                ny,
                nz
            ): idx
            for idx, genes in enumerate(genes_array)
        }

        # Collect results
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                fitness_values[idx] = future.result()
            except Exception:
                fitness_values[idx] = float('inf')

    return fitness_values
