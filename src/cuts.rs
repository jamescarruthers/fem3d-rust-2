//! Cut geometry and profile generation for undercut bar profiles.
//!
//! This module provides functions for defining and computing the height profile
//! of a bar with rectangular undercuts (cuts).

/// Single rectangular cut defining the undercut profile.
///
/// Each cut has a position (lambda) from the bar center and a height (h).
/// Cuts are nested: larger lambda values are outermost.
/// The profile is symmetric about the bar center.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cut {
    /// Distance from center of bar (m). Must be between 0 and L/2.
    pub lambda: f64,
    /// Height at this cut position (m).
    pub h: f64,
}

impl Cut {
    /// Create a new cut.
    pub fn new(lambda: f64, h: f64) -> Self {
        Cut { lambda, h }
    }
}

/// Compute the bar profile height H(x) at a given position.
///
/// Based on Eq. 3 from the reference paper. The profile is symmetric about x = L/2.
/// Cuts are nested: lambda_1 > lambda_2 > ... > lambda_N > 0.
/// The innermost cut (smallest lambda) that contains the point determines the height.
///
/// # Arguments
/// * `x` - Position along bar (m), 0 <= x <= L
/// * `cuts` - Slice of cuts (will be sorted internally by lambda descending)
/// * `length` - Bar length (m)
/// * `h0` - Original bar height (m)
///
/// # Returns
/// Height H(x) at position x
pub fn compute_height(x: f64, cuts: &[Cut], length: f64, h0: f64) -> f64 {
    let dist_from_center = (x - length / 2.0).abs();

    // Find all cuts that contain this point
    let mut containing_cuts: Vec<Cut> = cuts
        .iter()
        .filter(|cut| cut.lambda > 0.0 && dist_from_center <= cut.lambda)
        .copied()
        .collect();

    // Sort by lambda descending (largest first = outermost)
    containing_cuts.sort_by(|a, b| {
        b.lambda
            .partial_cmp(&a.lambda)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return innermost (smallest lambda) containing cut's height, or h0 if outside all cuts
    containing_cuts.last().map(|cut| cut.h).unwrap_or(h0)
}

/// Generate element heights for FEM discretization with quadratic interpolation
/// at discontinuities (Eq. 6 from paper).
///
/// The quadratic weighting: H_i = sqrt((h_{n-1}^2 * dx1 + h_n^2 * dx2) / (dx1 + dx2))
///
/// # Arguments
/// * `cuts` - Slice of cuts
/// * `length` - Bar length (m)
/// * `h0` - Original height (m)
/// * `num_elements` - Number of finite elements
///
/// # Returns
/// Vector of element heights (length num_elements)
pub fn generate_element_heights(cuts: &[Cut], length: f64, h0: f64, num_elements: usize) -> Vec<f64> {
    let element_length = length / num_elements as f64;
    let center_x = length / 2.0;

    // Sort cuts by lambda descending (outermost first)
    let mut sorted_cuts: Vec<Cut> = cuts.to_vec();
    sorted_cuts.sort_by(|a, b| {
        b.lambda
            .partial_cmp(&a.lambda)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build list of discontinuity positions with heights on each side
    let mut discontinuities: Vec<(f64, f64, f64)> = Vec::new();

    for cut in &sorted_cuts {
        if cut.lambda <= 0.0 {
            continue;
        }

        let left_boundary = center_x - cut.lambda;
        let right_boundary = center_x + cut.lambda;

        // At left boundary: compute heights just before and after
        let h_outside_left = compute_height(left_boundary - 0.0001, &sorted_cuts, length, h0);
        let h_inside_left = compute_height(left_boundary + 0.0001, &sorted_cuts, length, h0);
        if (h_outside_left - h_inside_left).abs() > 1e-9 {
            discontinuities.push((left_boundary, h_outside_left, h_inside_left));
        }

        // At right boundary: compute heights just before and after
        let h_inside_right = compute_height(right_boundary - 0.0001, &sorted_cuts, length, h0);
        let h_outside_right = compute_height(right_boundary + 0.0001, &sorted_cuts, length, h0);
        if (h_inside_right - h_outside_right).abs() > 1e-9 {
            discontinuities.push((right_boundary, h_inside_right, h_outside_right));
        }
    }

    // Sort discontinuities by position
    discontinuities.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // For each element, compute the appropriate height
    let mut heights = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let x_start = i as f64 * element_length;
        let x_end = (i + 1) as f64 * element_length;
        let x_mid = (x_start + x_end) / 2.0;

        // Check if element contains a discontinuity
        let mut found_discontinuity = false;
        for &(disc_x, h_before, h_after) in &discontinuities {
            if disc_x > x_start && disc_x < x_end {
                // Element contains a discontinuity - use quadratic interpolation (Eq. 6)
                let dx1 = disc_x - x_start;
                let dx2 = x_end - disc_x;

                // Quadratic weighting from Eq. 6
                let height =
                    ((h_before * h_before * dx1 + h_after * h_after * dx2) / (dx1 + dx2)).sqrt();
                heights.push(height);
                found_discontinuity = true;
                break;
            }
        }

        if !found_discontinuity {
            // No discontinuity in this element - use height at midpoint
            heights.push(compute_height(x_mid, &sorted_cuts, length, h0));
        }
    }

    heights
}

/// Convert genes array to cuts vector.
///
/// Genes format: [lambda_1, h_1, lambda_2, h_2, ...]
/// Note: genes may have an optional trailing length_adjust value that should be ignored.
///
/// # Arguments
/// * `genes` - Flat array of optimization variables
///
/// # Returns
/// Vector of Cut objects sorted by lambda descending
pub fn genes_to_cuts(genes: &[f64]) -> Vec<Cut> {
    let mut cuts = Vec::new();

    // Process pairs of genes (lambda, h)
    let mut i = 0;
    while i + 1 < genes.len() {
        let lambda = genes[i];
        let h = genes[i + 1];

        // Only add valid cuts (non-NaN)
        if lambda.is_finite() && h.is_finite() {
            cuts.push(Cut::new(lambda, h));
        }
        i += 2;
    }

    // Sort by lambda descending (largest first)
    cuts.sort_by(|a, b| {
        b.lambda
            .partial_cmp(&a.lambda)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    cuts
}

/// Convert cuts vector to genes array.
///
/// Cuts are sorted by lambda (descending) before conversion.
///
/// # Arguments
/// * `cuts` - Slice of Cut objects
///
/// # Returns
/// Flat array of genes [lambda_1, h_1, lambda_2, h_2, ...]
pub fn cuts_to_genes(cuts: &[Cut]) -> Vec<f64> {
    let mut sorted_cuts = cuts.to_vec();
    sorted_cuts.sort_by(|a, b| {
        b.lambda
            .partial_cmp(&a.lambda)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut genes = Vec::with_capacity(sorted_cuts.len() * 2);
    for cut in sorted_cuts {
        genes.push(cut.lambda);
        genes.push(cut.h);
    }
    genes
}

/// Generate adaptive 1D mesh with refinement at cut boundaries.
///
/// Uses finer elements near discontinuities (cut boundaries) and coarser
/// elements in uniform regions for better accuracy with fewer total elements.
///
/// # Arguments
/// * `cuts` - Slice of cuts
/// * `length` - Bar length (m)
/// * `h0` - Original height (m)
/// * `base_elements` - Number of elements if mesh were uniform
/// * `refinement_factor` - How many times finer the mesh is at boundaries (default: 4)
/// * `transition_width` - Width of transition zone as fraction of length (default: 0.02)
///
/// # Returns
/// Tuple of (x_positions, element_heights):
/// - x_positions: Vector of element boundary x-coordinates (length n+1)
/// - element_heights: Height at each element (length n)
pub fn generate_adaptive_mesh_1d(
    cuts: &[Cut],
    length: f64,
    h0: f64,
    base_elements: usize,
    refinement_factor: usize,
    transition_width: f64,
) -> (Vec<f64>, Vec<f64>) {
    let center_x = length / 2.0;

    // Sort cuts by lambda descending
    let mut sorted_cuts: Vec<Cut> = cuts.to_vec();
    sorted_cuts.sort_by(|a, b| {
        b.lambda
            .partial_cmp(&a.lambda)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find all discontinuity positions
    let mut discontinuities: Vec<f64> = Vec::new();
    for cut in &sorted_cuts {
        if cut.lambda <= 0.0 {
            continue;
        }
        let left_boundary = center_x - cut.lambda;
        let right_boundary = center_x + cut.lambda;
        discontinuities.push(left_boundary);
        discontinuities.push(right_boundary);
    }
    discontinuities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Define refinement zones around each discontinuity
    let transition_dist = transition_width * length;

    let is_near_discontinuity = |x: f64| -> bool {
        discontinuities
            .iter()
            .any(|&disc| (x - disc).abs() < transition_dist)
    };

    // Generate adaptive element positions
    let base_dx = length / base_elements as f64;
    let fine_dx = base_dx / refinement_factor as f64;

    let mut x_positions: Vec<f64> = vec![0.0];
    let mut current_x = 0.0;

    while current_x < length - 1e-10 {
        // Determine element size based on proximity to discontinuity
        let dx = if is_near_discontinuity(current_x) || is_near_discontinuity(current_x + base_dx) {
            fine_dx
        } else {
            base_dx
        };

        // Don't overshoot the bar length
        let next_x = if current_x + dx > length {
            length
        } else {
            current_x + dx
        };

        current_x = next_x;
        x_positions.push(current_x);
    }

    // Ensure last position is exactly length
    if let Some(last) = x_positions.last_mut() {
        if (*last - length).abs() > 1e-10 {
            *last = length;
        }
    }

    // Generate heights for each element
    let num_elements = x_positions.len() - 1;
    let mut element_heights = Vec::with_capacity(num_elements);

    // Build discontinuities with heights for interpolation
    let mut disc_with_heights: Vec<(f64, f64, f64)> = Vec::new();
    for cut in &sorted_cuts {
        if cut.lambda <= 0.0 {
            continue;
        }

        let left_boundary = center_x - cut.lambda;
        let right_boundary = center_x + cut.lambda;

        // At left boundary
        let h_outside_left = compute_height(left_boundary - 0.0001, &sorted_cuts, length, h0);
        let h_inside_left = compute_height(left_boundary + 0.0001, &sorted_cuts, length, h0);
        if (h_outside_left - h_inside_left).abs() > 1e-9 {
            disc_with_heights.push((left_boundary, h_outside_left, h_inside_left));
        }

        // At right boundary
        let h_inside_right = compute_height(right_boundary - 0.0001, &sorted_cuts, length, h0);
        let h_outside_right = compute_height(right_boundary + 0.0001, &sorted_cuts, length, h0);
        if (h_inside_right - h_outside_right).abs() > 1e-9 {
            disc_with_heights.push((right_boundary, h_inside_right, h_outside_right));
        }
    }
    disc_with_heights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..num_elements {
        let x_start = x_positions[i];
        let x_end = x_positions[i + 1];
        let x_mid = (x_start + x_end) / 2.0;

        // Check if element contains a discontinuity
        let mut found_discontinuity = false;
        for &(disc_x, h_before, h_after) in &disc_with_heights {
            if disc_x > x_start && disc_x < x_end {
                // Element contains a discontinuity - use quadratic interpolation
                let dx1 = disc_x - x_start;
                let dx2 = x_end - disc_x;

                // Quadratic weighting from Eq. 6
                let height =
                    ((h_before * h_before * dx1 + h_after * h_after * dx2) / (dx1 + dx2)).sqrt();
                element_heights.push(height);
                found_discontinuity = true;
                break;
            }
        }

        if !found_discontinuity {
            // No discontinuity - use height at midpoint
            element_heights.push(compute_height(x_mid, &sorted_cuts, length, h0));
        }
    }

    (x_positions, element_heights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_creation() {
        let cut = Cut::new(0.1, 0.02);
        assert_eq!(cut.lambda, 0.1);
        assert_eq!(cut.h, 0.02);
    }

    #[test]
    fn compute_height_uniform_bar() {
        // No cuts - should return h0 everywhere
        let cuts = [];
        let length = 0.5;
        let h0 = 0.024;

        assert_eq!(compute_height(0.0, &cuts, length, h0), h0);
        assert_eq!(compute_height(0.25, &cuts, length, h0), h0);
        assert_eq!(compute_height(0.5, &cuts, length, h0), h0);
    }

    #[test]
    fn compute_height_single_cut() {
        // Single cut at center
        let cuts = [Cut::new(0.1, 0.015)];
        let length = 0.5;
        let h0 = 0.024;
        let center = length / 2.0;

        // Inside cut (within lambda from center)
        assert_eq!(compute_height(center, &cuts, length, h0), 0.015);
        assert_eq!(compute_height(center - 0.05, &cuts, length, h0), 0.015);
        assert_eq!(compute_height(center + 0.05, &cuts, length, h0), 0.015);

        // Outside cut
        assert_eq!(compute_height(0.0, &cuts, length, h0), h0);
        assert_eq!(compute_height(length, &cuts, length, h0), h0);
    }

    #[test]
    fn compute_height_nested_cuts() {
        // Two nested cuts
        let cuts = [
            Cut::new(0.2, 0.020), // Outer cut
            Cut::new(0.1, 0.015), // Inner cut
        ];
        let length = 0.5;
        let h0 = 0.024;
        let center = length / 2.0;

        // At center: innermost cut applies
        assert_eq!(compute_height(center, &cuts, length, h0), 0.015);

        // Between cuts: outer cut applies
        assert_eq!(compute_height(center + 0.15, &cuts, length, h0), 0.020);

        // Outside all cuts: original height
        assert_eq!(compute_height(0.0, &cuts, length, h0), h0);
    }

    #[test]
    fn generate_element_heights_uniform() {
        let cuts = [];
        let length = 0.5;
        let h0 = 0.024;
        let num_elements = 10;

        let heights = generate_element_heights(&cuts, length, h0, num_elements);

        assert_eq!(heights.len(), num_elements);
        for height in heights {
            assert!((height - h0).abs() < 1e-10);
        }
    }

    #[test]
    fn generate_element_heights_with_cut() {
        let cuts = [Cut::new(0.1, 0.015)];
        let length = 0.5;
        let h0 = 0.024;
        let num_elements = 20;

        let heights = generate_element_heights(&cuts, length, h0, num_elements);

        assert_eq!(heights.len(), num_elements);

        // Elements at the ends should be close to h0
        assert!((heights[0] - h0).abs() < 1e-3);
        assert!((heights[num_elements - 1] - h0).abs() < 1e-3);

        // Elements near center should be close to cut height
        let mid = num_elements / 2;
        assert!((heights[mid] - 0.015).abs() < 1e-3);
    }

    #[test]
    fn genes_to_cuts_conversion() {
        let genes = vec![0.2, 0.020, 0.1, 0.015];
        let cuts = genes_to_cuts(&genes);

        assert_eq!(cuts.len(), 2);
        // Should be sorted by lambda descending
        assert_eq!(cuts[0].lambda, 0.2);
        assert_eq!(cuts[0].h, 0.020);
        assert_eq!(cuts[1].lambda, 0.1);
        assert_eq!(cuts[1].h, 0.015);
    }

    #[test]
    fn genes_to_cuts_filters_nan() {
        let genes = vec![0.2, 0.020, f64::NAN, 0.015, 0.1, 0.012];
        let cuts = genes_to_cuts(&genes);

        // Should skip the NaN pair and include valid cuts
        assert_eq!(cuts.len(), 2);
        assert_eq!(cuts[0].lambda, 0.2);
        assert_eq!(cuts[1].lambda, 0.1);
    }

    #[test]
    fn genes_to_cuts_odd_length() {
        // Odd length genes - should ignore the last unpaired value
        let genes = vec![0.2, 0.020, 0.1, 0.015, 0.05];
        let cuts = genes_to_cuts(&genes);

        assert_eq!(cuts.len(), 2);
    }

    #[test]
    fn cuts_to_genes_conversion() {
        let cuts = vec![Cut::new(0.1, 0.015), Cut::new(0.2, 0.020)];
        let genes = cuts_to_genes(&cuts);

        // Should be sorted by lambda descending
        assert_eq!(genes.len(), 4);
        assert_eq!(genes[0], 0.2);
        assert_eq!(genes[1], 0.020);
        assert_eq!(genes[2], 0.1);
        assert_eq!(genes[3], 0.015);
    }

    #[test]
    fn cuts_genes_roundtrip() {
        let original_genes = vec![0.2, 0.020, 0.1, 0.015];
        let cuts = genes_to_cuts(&original_genes);
        let roundtrip_genes = cuts_to_genes(&cuts);

        assert_eq!(original_genes, roundtrip_genes);
    }

    #[test]
    fn generate_element_heights_with_discontinuity() {
        // Test quadratic interpolation at discontinuities
        let cuts = [Cut::new(0.237, 0.012)];
        let length = 1.0;
        let h0 = 0.024;
        let num_elements = 50;

        let heights = generate_element_heights(&cuts, length, h0, num_elements);

        assert_eq!(heights.len(), num_elements);

        // Check that we have both cut height and original height
        let has_cut_height = heights.iter().any(|&h| (h - 0.012).abs() < 1e-3);
        let has_original_height = heights.iter().any(|&h| (h - 0.024).abs() < 1e-3);

        assert!(has_cut_height, "Should have elements at cut height");
        assert!(has_original_height, "Should have elements at original height");

        // At boundaries, some elements should have intermediate heights from interpolation
        let intermediate_heights: Vec<f64> = heights
            .iter()
            .filter(|&&h| h > 0.013 && h < 0.023)
            .copied()
            .collect();

        assert!(
            !intermediate_heights.is_empty(),
            "Should find interpolated heights at boundaries. Got heights range: [{:.6}, {:.6}]",
            heights.iter().cloned().fold(f64::INFINITY, f64::min),
            heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
    }

    #[test]
    fn adaptive_mesh_1d_has_variable_spacing() {
        let cuts = [Cut::new(0.15, 0.015)];
        let length = 0.5;
        let h0 = 0.024;
        let base_elements = 20;
        let refinement_factor = 4;
        let transition_width = 0.02;

        let (x_positions, element_heights) = generate_adaptive_mesh_1d(
            &cuts,
            length,
            h0,
            base_elements,
            refinement_factor,
            transition_width,
        );

        // Should have more elements than base due to refinement
        assert!(x_positions.len() > base_elements);
        assert_eq!(element_heights.len(), x_positions.len() - 1);

        // First and last positions should be 0 and length
        assert!((x_positions[0] - 0.0).abs() < 1e-10);
        assert!((x_positions[x_positions.len() - 1] - length).abs() < 1e-10);

        // Check for variable element sizes
        let mut element_sizes: Vec<f64> = Vec::new();
        for i in 0..x_positions.len() - 1 {
            element_sizes.push(x_positions[i + 1] - x_positions[i]);
        }

        let min_size = element_sizes
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_size = element_sizes
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Should have varying element sizes due to refinement
        assert!(
            (max_size / min_size) > 2.0,
            "Should have significant element size variation. Min: {:.6}, Max: {:.6}",
            min_size,
            max_size
        );
    }
}
