"""
Bar Profile Generation

Implements the symmetric rectangular undercut model.
The undercut is defined by N cuts, each with a length (lambda) and height (h).

Key equations from paper:
- Profile H(x) defined by Eq. 3
- Quadratic interpolation for discontinuities (Eq. 6)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import math
from ..types import Cut, BarParameters


def compute_height(
    x: float,
    cuts: List[Cut],
    L: float,
    h0: float
) -> float:
    """
    Compute the bar profile H(x) for a given x position.
    Based on Eq. 3 from the paper.

    The profile is symmetric about x = L/2 (center of bar).
    Cuts are nested: lambda_1 > lambda_2 > ... > lambda_N > 0.
    The innermost cut (smallest lambda) that contains the point determines the height.

    Args:
        x: Position along bar (m), 0 <= x <= L
        cuts: Array of cuts (will be sorted internally)
        L: Bar length (m)
        h0: Original bar height (m)

    Returns:
        Height H(x) at position x
    """
    # Distance from center
    dist_from_center = abs(x - L / 2)

    # Sort cuts by lambda descending (largest first = outermost)
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    # Find all cuts that contain this point, then return the innermost one's height
    containing_cuts = [
        cut for cut in sorted_cuts
        if cut.lambda_ > 0 and dist_from_center <= cut.lambda_
    ]

    if not containing_cuts:
        # Outside all cuts - return original height
        return h0

    # Return innermost (smallest lambda) containing cut
    return containing_cuts[-1].h


def generate_element_heights(
    cuts: List[Cut],
    L: float,
    h0: float,
    Ne: int
) -> List[float]:
    """
    Generate element heights for FEM discretization with quadratic interpolation
    at discontinuities (Eq. 6 from paper).

    The quadratic weighting: H_i = sqrt((h_{n-1}^2 * dx1 + h_n^2 * dx2) / (dx1 + dx2))

    Args:
        cuts: Array of cuts
        L: Bar length (m)
        h0: Original height (m)
        Ne: Number of finite elements

    Returns:
        Array of element heights (length Ne)
    """
    Le = L / Ne  # Element length
    heights: List[float] = []
    center_x = L / 2

    # Sort cuts by lambda (descending - largest/outermost first)
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    # Build list of discontinuity positions with heights on each side
    discontinuities: List[dict] = []

    for cut in sorted_cuts:
        if cut.lambda_ <= 0:
            continue

        left_boundary = center_x - cut.lambda_
        right_boundary = center_x + cut.lambda_

        # At left boundary: compute heights just before and after
        h_outside_left = compute_height(left_boundary - 0.0001, sorted_cuts, L, h0)
        h_inside_left = compute_height(left_boundary + 0.0001, sorted_cuts, L, h0)
        if abs(h_outside_left - h_inside_left) > 1e-9:
            discontinuities.append({
                "pos": left_boundary,
                "h_before": h_outside_left,
                "h_after": h_inside_left
            })

        # At right boundary: compute heights just before and after
        h_inside_right = compute_height(right_boundary - 0.0001, sorted_cuts, L, h0)
        h_outside_right = compute_height(right_boundary + 0.0001, sorted_cuts, L, h0)
        if abs(h_inside_right - h_outside_right) > 1e-9:
            discontinuities.append({
                "pos": right_boundary,
                "h_before": h_inside_right,
                "h_after": h_outside_right
            })

    # Sort discontinuities by position
    discontinuities.sort(key=lambda d: d["pos"])

    # For each element, compute the appropriate height
    for i in range(Ne):
        x_start = i * Le
        x_end = (i + 1) * Le
        x_mid = (x_start + x_end) / 2

        # Check if element contains a discontinuity
        found_discontinuity = False
        for disc in discontinuities:
            if disc["pos"] > x_start and disc["pos"] < x_end:
                # Element contains a discontinuity - use quadratic interpolation (Eq. 6)
                dx1 = disc["pos"] - x_start
                dx2 = x_end - disc["pos"]
                h1 = disc["h_before"]
                h2 = disc["h_after"]

                # Quadratic weighting from Eq. 6
                heights.append(math.sqrt((h1 * h1 * dx1 + h2 * h2 * dx2) / (dx1 + dx2)))
                found_discontinuity = True
                break

        if not found_discontinuity:
            # No discontinuity in this element - use height at midpoint
            heights.append(compute_height(x_mid, sorted_cuts, L, h0))

    return heights


def genes_to_cuts(genes: List[float]) -> List[Cut]:
    """
    Convert genes array to cuts array.
    Genes format: [lambda_1, h_1, lambda_2, h_2, ..., length_adjust?]
    Note: genes may have an optional trailing length_adjust value that should be ignored.

    Args:
        genes: Flat array of optimization variables

    Returns:
        Array of Cut objects
    """
    cuts: List[Cut] = []
    # Process pairs of genes (lambda, h) - stop before incomplete pair
    i = 0
    while i + 1 < len(genes):
        lambda_ = genes[i]
        h = genes[i + 1]
        # Only add valid cuts
        if isinstance(lambda_, (int, float)) and isinstance(h, (int, float)):
            if not math.isnan(lambda_) and not math.isnan(h):
                cuts.append(Cut(lambda_=lambda_, h=h))
        i += 2

    # Sort by lambda descending (largest first)
    return sorted(cuts, key=lambda c: c.lambda_, reverse=True)


def cuts_to_genes(cuts: List[Cut]) -> List[float]:
    """
    Convert cuts array to genes array.
    Cuts are sorted by lambda (descending) before conversion.

    Args:
        cuts: Array of Cut objects

    Returns:
        Flat array of genes [lambda_1, h_1, lambda_2, h_2, ...]
    """
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)
    genes: List[float] = []
    for cut in sorted_cuts:
        genes.extend([cut.lambda_, cut.h])
    return genes


def count_effective_cuts(cuts: List[Cut]) -> int:
    """
    Compute the effective number of cuts (ignoring cuts that are inside others).
    If lambda_n >= lambda_{n-1}, the outer cut becomes inconsequential.
    """
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    effective_count = 0
    last_lambda = float('inf')

    for cut in sorted_cuts:
        if cut.lambda_ < last_lambda and cut.lambda_ > 0:
            effective_count += 1
            last_lambda = cut.lambda_

    return effective_count


def validate_cuts(cuts: List[Cut], bar: BarParameters) -> Tuple[bool, Optional[str]]:
    """
    Validate cuts are within bounds.

    Returns:
        Tuple of (valid, message)
    """
    for i, cut in enumerate(cuts):
        if cut.lambda_ < 0 or cut.lambda_ > bar.L / 2:
            return (False, f"Cut {i + 1} lambda ({cut.lambda_}) out of bounds [0, {bar.L / 2}]")

        if cut.h < bar.hMin or cut.h > bar.h0:
            return (False, f"Cut {i + 1} height ({cut.h}) out of bounds [{bar.hMin}, {bar.h0}]")

    return (True, None)


def generate_adaptive_mesh_1d(
    cuts: List[Cut],
    L: float,
    h0: float,
    base_elements: int = 60,
    refinement_factor: int = 4,
    transition_width: float = 0.02
) -> Tuple[List[float], List[float]]:
    """
    Generate adaptive 1D mesh with refinement at cut boundaries.

    Uses finer elements near discontinuities (cut boundaries) and coarser
    elements in uniform regions for better accuracy with fewer total elements.

    Args:
        cuts: Array of cuts
        L: Bar length (m)
        h0: Original height (m)
        base_elements: Number of elements if mesh were uniform
        refinement_factor: How many times finer the mesh is at boundaries
        transition_width: Width of transition zone as fraction of L

    Returns:
        Tuple of (x_positions, element_heights):
        - x_positions: List of element starting x-coordinates (length n)
        - element_heights: Height at each element (length n)
    """
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)
    center_x = L / 2

    # Find all discontinuity positions
    discontinuities: List[float] = []
    for cut in sorted_cuts:
        if cut.lambda_ <= 0:
            continue
        left_boundary = center_x - cut.lambda_
        right_boundary = center_x + cut.lambda_
        discontinuities.append(left_boundary)
        discontinuities.append(right_boundary)

    discontinuities.sort()

    # Define refinement zones around each discontinuity
    transition_dist = transition_width * L

    def is_near_discontinuity(x: float) -> bool:
        """Check if position is near any discontinuity."""
        for disc in discontinuities:
            if abs(x - disc) < transition_dist:
                return True
        return False

    # Generate adaptive element positions
    # Base element size
    base_dx = L / base_elements
    fine_dx = base_dx / refinement_factor

    x_positions: List[float] = [0.0]
    current_x = 0.0

    while current_x < L - 1e-10:
        # Determine element size based on proximity to discontinuity
        if is_near_discontinuity(current_x) or is_near_discontinuity(current_x + base_dx):
            dx = fine_dx
        else:
            dx = base_dx

        # Don't overshoot the bar length
        if current_x + dx > L:
            dx = L - current_x

        current_x += dx
        if current_x <= L:
            x_positions.append(current_x)

    # Ensure last position is exactly L
    if abs(x_positions[-1] - L) > 1e-10:
        x_positions[-1] = L

    # Generate heights for each element
    num_elements = len(x_positions) - 1
    element_heights: List[float] = []

    for i in range(num_elements):
        x_start = x_positions[i]
        x_end = x_positions[i + 1]
        x_mid = (x_start + x_end) / 2

        # Check if element contains a discontinuity
        found_discontinuity = False
        for disc_x in discontinuities:
            if disc_x > x_start and disc_x < x_end:
                # Element contains a discontinuity - use quadratic interpolation
                dx1 = disc_x - x_start
                dx2 = x_end - disc_x
                h1 = compute_height(disc_x - 0.0001, sorted_cuts, L, h0)
                h2 = compute_height(disc_x + 0.0001, sorted_cuts, L, h0)

                # Quadratic weighting from Eq. 6
                element_heights.append(math.sqrt((h1 * h1 * dx1 + h2 * h2 * dx2) / (dx1 + dx2)))
                found_discontinuity = True
                break

        if not found_discontinuity:
            # No discontinuity - use height at midpoint
            element_heights.append(compute_height(x_mid, sorted_cuts, L, h0))

    return x_positions, element_heights


@dataclass
class MeshResolutionWarning:
    """Warning about mesh resolution vs cut geometry."""
    feature_type: str       # 'cut_width', 'cut_spacing', 'boundary_region'
    feature_size_mm: float  # Size of the geometric feature in mm
    element_size_mm: float  # Size of the mesh element in mm
    ratio: float            # element_size / feature_size (>1 means under-resolved)
    message: str


def analyze_mesh_resolution(
    cuts: List[Cut],
    L: float,
    h0: float,
    num_elements: int,
    adaptive: bool = False,
    refinement_factor: int = 4,
    transition_width: float = 0.02
) -> Tuple[List[MeshResolutionWarning], dict]:
    """
    Analyze whether mesh resolution is adequate for the cut geometry.

    Checks if element sizes are small enough to accurately capture:
    - Narrow cuts (small lambda values)
    - Spacing between adjacent cut boundaries
    - Sharp transitions at cut edges

    Args:
        cuts: List of cuts
        L: Bar length (m)
        h0: Original height (m)
        num_elements: Number of elements for uniform mesh
        adaptive: Whether adaptive meshing is used
        refinement_factor: Refinement factor for adaptive mesh
        transition_width: Transition width for adaptive mesh

    Returns:
        Tuple of (warnings, stats):
        - warnings: List of resolution warnings
        - stats: Dictionary with mesh statistics
    """
    warnings: List[MeshResolutionWarning] = []
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    # Calculate element sizes
    L_mm = L * 1000
    base_element_size_mm = L_mm / num_elements

    if adaptive:
        fine_element_size_mm = base_element_size_mm / refinement_factor
        min_element_size_mm = fine_element_size_mm
    else:
        min_element_size_mm = base_element_size_mm

    # Collect all boundary positions (in mm from center)
    boundaries_mm = []
    for cut in sorted_cuts:
        if cut.lambda_ > 0:
            boundaries_mm.append(cut.lambda_ * 1000)

    boundaries_mm.sort()

    stats = {
        'num_elements': num_elements,
        'base_element_size_mm': base_element_size_mm,
        'min_element_size_mm': min_element_size_mm,
        'adaptive': adaptive,
        'num_cuts': len([c for c in sorted_cuts if c.lambda_ > 0]),
        'boundaries_mm': boundaries_mm,
        'smallest_feature_mm': float('inf'),
        'resolution_adequate': True
    }

    if not boundaries_mm:
        return warnings, stats

    # Check 1: Smallest cut region (innermost cut width = 2 * smallest lambda)
    smallest_cut_width_mm = 2 * boundaries_mm[0]
    stats['smallest_feature_mm'] = min(stats['smallest_feature_mm'], smallest_cut_width_mm)

    if smallest_cut_width_mm < min_element_size_mm * 2:
        ratio = min_element_size_mm / (smallest_cut_width_mm / 2)
        warnings.append(MeshResolutionWarning(
            feature_type='cut_width',
            feature_size_mm=smallest_cut_width_mm,
            element_size_mm=min_element_size_mm,
            ratio=ratio,
            message=f"Innermost cut region ({smallest_cut_width_mm:.1f}mm wide) may be under-resolved. "
                    f"Element size {min_element_size_mm:.1f}mm should be <{smallest_cut_width_mm/2:.1f}mm for accuracy."
        ))
        stats['resolution_adequate'] = False

    # Check 2: Spacing between adjacent cut boundaries
    for i in range(1, len(boundaries_mm)):
        spacing_mm = boundaries_mm[i] - boundaries_mm[i-1]
        stats['smallest_feature_mm'] = min(stats['smallest_feature_mm'], spacing_mm)

        if spacing_mm < min_element_size_mm * 2:
            ratio = min_element_size_mm / (spacing_mm / 2)
            warnings.append(MeshResolutionWarning(
                feature_type='cut_spacing',
                feature_size_mm=spacing_mm,
                element_size_mm=min_element_size_mm,
                ratio=ratio,
                message=f"Cut boundary spacing ({spacing_mm:.1f}mm) may be under-resolved. "
                        f"Element size {min_element_size_mm:.1f}mm should be <{spacing_mm/2:.1f}mm."
            ))
            stats['resolution_adequate'] = False

    # Check 3: Recommend minimum elements based on geometry
    min_feature_mm = stats['smallest_feature_mm']
    if min_feature_mm < float('inf'):
        recommended_element_size = min_feature_mm / 3  # At least 3 elements per feature
        recommended_num_elements = int(L_mm / recommended_element_size)

        stats['recommended_element_size_mm'] = recommended_element_size
        stats['recommended_num_elements'] = recommended_num_elements

        if recommended_num_elements > num_elements:
            if adaptive:
                # For adaptive, recommend higher refinement
                needed_refinement = int(base_element_size_mm / recommended_element_size) + 1
                stats['recommended_refinement'] = needed_refinement
            else:
                stats['recommended_num_elements'] = recommended_num_elements

    return warnings, stats


def check_mesh_resolution(
    cuts: List[Cut],
    L: float,
    num_elements: int,
    adaptive: bool = False,
    refinement_factor: int = 4,
    verbose: bool = True
) -> bool:
    """
    Quick check if mesh resolution is adequate. Prints warnings if verbose.

    Args:
        cuts: List of cuts
        L: Bar length (m)
        num_elements: Number of elements
        adaptive: Whether adaptive meshing is used
        refinement_factor: Refinement factor for adaptive mesh
        verbose: Print warnings

    Returns:
        True if resolution is adequate, False if there are warnings
    """
    warnings, stats = analyze_mesh_resolution(
        cuts, L, 0.024,  # h0 doesn't matter for resolution check
        num_elements, adaptive, refinement_factor
    )

    if verbose and warnings:
        print(f"\n⚠️  Mesh Resolution Warnings:")
        for w in warnings:
            print(f"  - {w.message}")

        if 'recommended_num_elements' in stats and not adaptive:
            print(f"\n  Recommendation: Use {stats['recommended_num_elements']} elements "
                  f"(currently {num_elements})")
        elif 'recommended_refinement' in stats and adaptive:
            print(f"\n  Recommendation: Use refinement_factor={stats['recommended_refinement']} "
                  f"(currently {refinement_factor})")
        print()

    return len(warnings) == 0


def generate_profile_points(
    cuts: List[Cut],
    L: float,
    h0: float,
    num_points: int = 200
) -> List[Tuple[float, float]]:
    """
    Generate profile points for visualization.
    Returns an array of (x, h) pairs representing the profile shape.

    Args:
        cuts: Array of cuts
        L: Bar length (m)
        h0: Original height (m)
        num_points: Number of points for smooth visualization

    Returns:
        List of (x, h) coordinate pairs
    """
    points: List[Tuple[float, float]] = []
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)

    # Generate points from left to right
    for i in range(num_points + 1):
        x = (i / num_points) * L
        h = compute_height(x, sorted_cuts, L, h0)
        points.append((x, h))

    # Add extra points at discontinuities for crisp edges
    extra_points: List[Tuple[float, float]] = []
    for cut in sorted_cuts:
        left_edge = L / 2 - cut.lambda_
        right_edge = L / 2 + cut.lambda_

        epsilon = L / 10000

        if 0 < left_edge < L:
            extra_points.append((left_edge - epsilon, compute_height(left_edge - epsilon, sorted_cuts, L, h0)))
            extra_points.append((left_edge + epsilon, compute_height(left_edge + epsilon, sorted_cuts, L, h0)))

        if 0 < right_edge < L:
            extra_points.append((right_edge - epsilon, compute_height(right_edge - epsilon, sorted_cuts, L, h0)))
            extra_points.append((right_edge + epsilon, compute_height(right_edge + epsilon, sorted_cuts, L, h0)))

    # Merge and sort all points by x
    all_points = points + extra_points
    all_points.sort(key=lambda p: p[0])

    return all_points
