"""
Geometric Penalties for Optimization

Implements volumetric and roughness penalties from the paper
to generate more manufacture-friendly geometries.
"""

from typing import List, Dict
from ..types import Cut


def compute_volume_penalty(cuts: List[Cut], L: float, h0: float) -> float:
    """
    Compute the volumetric penalty (Eq. 10 from paper).

    V = 100 * (2 / h0*L) * integral(h0 - H(x)) dx

    This represents the percentage of volume extracted from the bar.
    For rectangular cuts, this simplifies to sum of cut volumes.

    Args:
        cuts: Array of rectangular cuts
        L: Bar length (m)
        h0: Original bar height (m)

    Returns:
        Volume penalty as percentage (0-100)
    """
    if not cuts:
        return 0.0

    # Sort cuts by lambda (descending - largest first)
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    # Original volume (half bar, since symmetric)
    half_bar_volume = (L / 2) * h0

    # Calculate extracted volume from symmetric undercuts
    extracted_volume = 0.0
    prev_lambda = 0.0

    # Process cuts from innermost (smallest lambda) to outermost (largest lambda)
    for cut in reversed(sorted_cuts):
        if cut.lambda_ > prev_lambda and cut.lambda_ > 0:
            # Width of this cut region
            width = cut.lambda_ - prev_lambda

            # Height removed in this region
            height_removed = h0 - cut.h

            # Volume removed (just one side, will account for symmetry later)
            extracted_volume += width * height_removed

            prev_lambda = cut.lambda_

    # Calculate percentage (divide by half-bar volume)
    volume_percent = 100.0 * (extracted_volume / half_bar_volume)

    return max(0.0, min(100.0, volume_percent))


def compute_roughness_penalty(cuts: List[Cut], h0: float) -> float:
    """
    Compute the roughness penalty (Eq. 12 from paper).

    S = 100 * (1/N) * sum(|h_{n-1} - h_n| / h0)

    This represents the average height change per discontinuity,
    normalized to the original height.

    Args:
        cuts: Array of rectangular cuts
        h0: Original bar height (m)

    Returns:
        Roughness penalty as percentage
    """
    if not cuts:
        return 0.0

    # Sort cuts by lambda (descending - largest first)
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    # Calculate sum of height changes
    sum_height_changes = 0.0
    num_discontinuities = 0

    # First discontinuity: from h0 to first (outermost) cut
    if sorted_cuts:
        sum_height_changes += abs(h0 - sorted_cuts[0].h)
        num_discontinuities += 1

    # Discontinuities between consecutive cuts
    for i in range(1, len(sorted_cuts)):
        # Only count if this cut is actually visible
        if sorted_cuts[i].lambda_ < sorted_cuts[i - 1].lambda_:
            sum_height_changes += abs(sorted_cuts[i - 1].h - sorted_cuts[i].h)
            num_discontinuities += 1

    if num_discontinuities == 0:
        return 0.0

    # Calculate average roughness as percentage
    roughness_percent = 100.0 * (sum_height_changes / (num_discontinuities * h0))

    return roughness_percent


def compute_both_penalties(cuts: List[Cut], L: float, h0: float) -> Dict[str, float]:
    """Compute both penalties at once for efficiency."""
    return {
        "volume": compute_volume_penalty(cuts, L, h0),
        "roughness": compute_roughness_penalty(cuts, h0)
    }


def estimate_minimum_volume(target_ratios: List[float], num_cuts: int) -> float:
    """
    Estimate the minimum volume needed to achieve a tuning target.
    Based on empirical observations from the paper.

    This is just a rough estimate - actual minimum depends on target ratio.
    """
    # Uniform bar ratios (approximately)
    uniform_ratios = [1, 2.76, 5.40, 8.93, 13.34]

    deviation_sum = 0.0
    for i in range(min(len(target_ratios), len(uniform_ratios))):
        deviation_sum += abs(target_ratios[i] - uniform_ratios[i]) / uniform_ratios[i]

    # Rough estimate: 5-50% volume based on deviation
    base_volume = 5.0
    deviation_factor = min(deviation_sum * 10, 45.0)

    return base_volume + deviation_factor
