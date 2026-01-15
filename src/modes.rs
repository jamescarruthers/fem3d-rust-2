//! Mode classification for vibration analysis.
//!
//! This module provides functions for classifying vibration modes
//! using the Soares top-corner displacement method.

use std::collections::HashMap;

use nalgebra::{DMatrix, Vector3};

use crate::types::{ModeType, DEFAULT_CORNER_TOL, DOF_PER_NODE, Z_DIR_INDEX};

/// Find the two corner nodes at the x=0 end on the top surface (max z) with max/min y.
///
/// Returns the indices of the top-surface nodes at x ≈ 0 having the maximum and
/// minimum y-coordinates, respectively. `tol` controls the coordinate tolerance
/// when comparing x/z locations.
pub fn find_corner_nodes(nodes: &[Vector3<f64>], tol: f64) -> Option<(usize, usize)> {
    if nodes.is_empty() {
        return None;
    }

    let x_min = nodes.iter().map(|n| n.x).fold(f64::INFINITY, f64::min);
    let end_indices: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| (n.x - x_min).abs() < tol)
        .map(|(i, _)| i)
        .collect();

    if end_indices.is_empty() {
        return None;
    }

    let z_max = end_indices
        .iter()
        .map(|&i| nodes[i].z)
        .fold(f64::NEG_INFINITY, f64::max);
    let top_indices: Vec<usize> = end_indices
        .iter()
        .cloned()
        .filter(|&i| (nodes[i].z - z_max).abs() < tol)
        .collect();
    if top_indices.len() < 2 {
        return None;
    }

    let s1 = top_indices
        .iter()
        .max_by(|&&a, &&b| nodes[a].y.total_cmp(&nodes[b].y))
        .copied()?;
    let s2 = top_indices
        .iter()
        .min_by(|&&a, &&b| nodes[a].y.total_cmp(&nodes[b].y))
        .copied()?;

    Some((s1, s2))
}

/// Classify a single mode using Soares' corner displacement method.
///
/// The displacement at two corner nodes on the x=0 top surface is examined. The
/// dominant displacement direction at the first corner chooses lateral (y) vs
/// axial (x) vs bending/torsion (z). If z dominates, same sign between the two
/// corners ⇒ vertical bending; opposite signs ⇒ torsional.
pub fn classify_mode_soares(
    mode_shape: &[f64],
    _nodes: &[Vector3<f64>], // kept for potential future shape filtering/normalization
    corner_indices: (usize, usize),
) -> ModeType {
    let (s1_idx, s2_idx) = corner_indices;
    let need = s1_idx.max(s2_idx) * DOF_PER_NODE + DOF_PER_NODE;
    if mode_shape.len() < need {
        return ModeType::Unknown;
    }

    let s1 = &mode_shape[s1_idx * DOF_PER_NODE..s1_idx * DOF_PER_NODE + DOF_PER_NODE];
    let s2 = &mode_shape[s2_idx * DOF_PER_NODE..s2_idx * DOF_PER_NODE + DOF_PER_NODE];

    let abs_s1 = [s1[0].abs(), s1[1].abs(), s1[2].abs()];
    let max_dir = abs_s1
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(Z_DIR_INDEX);

    match max_dir {
        1 => ModeType::Lateral,
        0 => ModeType::Axial,
        _ => {
            if s1[2].signum() == s2[2].signum() {
                ModeType::VerticalBending
            } else {
                ModeType::Torsional
            }
        }
    }
}

/// Classify all modes given frequencies and mode shapes.
///
/// Returns a map from `ModeType` to a vector of tuples
/// `(frequency_hz, mode_index, family_rank)`, where `mode_index` is the column
/// index in `mode_shapes` and `family_rank` is the 1-based order within its
/// family after sorting by frequency.
pub fn classify_all_modes(
    frequencies: &[f64],
    mode_shapes: &DMatrix<f64>,
    nodes: &[Vector3<f64>],
) -> HashMap<ModeType, Vec<(f64, usize, usize)>> {
    let mut families: HashMap<ModeType, Vec<(f64, usize, usize)>> = HashMap::new();
    families.insert(ModeType::VerticalBending, Vec::new());
    families.insert(ModeType::Torsional, Vec::new());
    families.insert(ModeType::Lateral, Vec::new());
    families.insert(ModeType::Axial, Vec::new());
    families.insert(ModeType::Unknown, Vec::new());

    let Some(corner_indices) = find_corner_nodes(nodes, DEFAULT_CORNER_TOL) else {
        return families;
    };
    let mode_count = mode_shapes.ncols();

    for (idx, freq) in frequencies.iter().copied().enumerate() {
        if idx >= mode_count {
            break;
        }
        let shape_col = mode_shapes.column(idx);
        let mode_type = classify_mode_soares(shape_col.as_slice(), nodes, corner_indices);
        families.entry(mode_type).or_default().push((freq, idx, 0));
    }

    for modes in families.values_mut() {
        modes.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (i, mode) in modes.iter_mut().enumerate() {
            mode.2 = i + 1;
        }
    }

    families
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soares_classification_distinguishes_modes() {
        let nodes = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(0.0, 1.0, 1.0),
        ];

        let mut shapes = DMatrix::<f64>::zeros(24, 4);

        // Vertical bending: both top corners positive z
        shapes[(4 * 3 + 2, 0)] = 1.0; // node4 z
        shapes[(7 * 3 + 2, 0)] = 1.0; // node7 z

        // Torsional: opposite z signs
        shapes[(4 * 3 + 2, 1)] = 1.0;
        shapes[(7 * 3 + 2, 1)] = -1.0;

        // Lateral: y dominant at s1
        shapes[(7 * 3 + 1, 2)] = 1.0;

        // Axial: x dominant at s1
        shapes[(7 * 3, 3)] = 1.0;

        let freqs = [100.0, 200.0, 300.0, 400.0];
        let families = classify_all_modes(&freqs, &shapes, &nodes);

        assert_eq!(families[&ModeType::VerticalBending][0].2, 1);
        assert_eq!(families[&ModeType::Torsional][0].2, 1);
        assert_eq!(families[&ModeType::Lateral][0].2, 1);
        assert_eq!(families[&ModeType::Axial][0].2, 1);
    }

    #[test]
    fn classification_with_real_mesh_frequencies_and_shapes() {
        use crate::mesh::generate_bar_mesh_3d;
        use crate::solver::compute_modal_frequencies_with_shapes;

        // Create a bar mesh with enough elements for modal analysis
        let heights: Vec<f64> = vec![0.024; 10]; // 10 elements along length
        let mesh = generate_bar_mesh_3d(0.2, 0.03, &heights, 10, 2, 2);

        // Compute frequencies and mode shapes
        let (freqs, mode_shapes) = compute_modal_frequencies_with_shapes(
            &mesh,
            12e9,  // Young's modulus (Pa) - wood-like
            0.35,  // Poisson's ratio
            640.0, // Density (kg/m³)
            8,     // Request 8 modes
        );

        assert!(
            !freqs.is_empty(),
            "Should compute at least one frequency"
        );
        assert_eq!(
            freqs.len(),
            mode_shapes.ncols(),
            "Number of frequencies should match number of mode shape columns"
        );

        // Classify the modes
        let families = classify_all_modes(&freqs, &mode_shapes, &mesh.nodes);

        // Check that at least some modes were classified
        let total_classified: usize = families.values().map(|v| v.len()).sum();
        assert_eq!(
            total_classified,
            freqs.len(),
            "All modes should be classified"
        );

        // Verify that the mode indices in families correctly reference the mode shapes
        for (mode_type, modes) in &families {
            for (freq, mode_idx, family_rank) in modes {
                assert!(
                    *mode_idx < mode_shapes.ncols(),
                    "Mode index {} for {:?} should be within mode_shapes columns ({})",
                    mode_idx,
                    mode_type,
                    mode_shapes.ncols()
                );
                // Verify the frequency matches
                let expected_freq = freqs[*mode_idx];
                assert!(
                    (*freq - expected_freq).abs() < 1e-6,
                    "Frequency mismatch: family has {} but freqs[{}] = {}",
                    freq,
                    mode_idx,
                    expected_freq
                );
                assert!(
                    *family_rank >= 1,
                    "Family rank should be 1-indexed, got {}",
                    family_rank
                );
            }
        }

        // Print classification results for debugging
        println!("Mode classification results:");
        for (mode_type, modes) in &families {
            if !modes.is_empty() {
                println!(
                    "  {:?}: {} modes, frequencies: {:?}",
                    mode_type,
                    modes.len(),
                    modes.iter().map(|(f, _, _)| *f).collect::<Vec<_>>()
                );
            }
        }
    }
}
