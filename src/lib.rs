use std::collections::HashMap;

use nalgebra::linalg::SymmetricEigen;
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, Vector3};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use serde::{Deserialize, Serialize};

#[cfg(feature = "sprs-backend")]
use sprs::{CsMat, TriMat};

type NodeCoords = SMatrix<f64, 8, 3>;
type Matrix6 = SMatrix<f64, 6, 6>;
type Matrix6x24 = SMatrix<f64, 6, 24>;
type Matrix3x24 = SMatrix<f64, 3, 24>;
type Matrix3x8 = SMatrix<f64, 3, 8>;
const DOF_PER_NODE: usize = 3;
const DEFAULT_CORNER_TOL: f64 = 1e-6;
const Z_DIR_INDEX: usize = 2;
// f64 representation of 1/sqrt(3) Gauss point coordinate for 2×2×2 quadrature
const GAUSS_G: f64 = 0.577_350_269_189_625_8;
const MIN_DET_J: f64 = 1e-12;
pub const LAMBDA_TOL: f64 = 1e-12;
/// Eigenvalue threshold used to discard the near-zero rigid-body modes of a free-free bar
/// (matches the Python reference's 100.0 cutoff).
pub const RIGID_BODY_LAMBDA_THRESHOLD: f64 = 100.0;
/// Frequency threshold (Hz) to remove any residual near-rigid-body modes that survive the
/// lambda cutoff on large DOF problems.
pub const MIN_FREQUENCY_HZ: f64 = 5.0;
/// Default shift for shift-invert Lanczos (targets eigenvalues near this value).
pub const DEFAULT_SHIFT: f64 = 1.0;
/// DOF threshold above which sparse solver is used automatically.
pub const SPARSE_DOF_THRESHOLD: usize = 500;
/// Maximum Lanczos iterations.
const MAX_LANCZOS_ITER: usize = 300;
/// Convergence tolerance for Lanczos.
const LANCZOS_TOL: f64 = 1e-10;

/// Solver type for eigenvalue computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EigenSolver {
    /// Dense solver using full matrix eigendecomposition. Best for small problems (< 500 DOF).
    Dense,
    /// Sparse solver using shift-invert Lanczos. Best for large problems (> 500 DOF).
    Sparse,
    /// Automatically choose based on problem size.
    #[default]
    Auto,
}

/// Backend for sparse matrix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseBackend {
    /// Use nalgebra-sparse (always available).
    #[default]
    NalgebraSparse,
    /// Use sprs crate (requires "sprs-backend" feature).
    #[cfg(feature = "sprs-backend")]
    Sprs,
}

/// Mode classification following Soares top-corner displacement method
/// (see reference/details.md section 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModeType {
    VerticalBending,
    Torsional,
    Lateral,
    Axial,
    Unknown,
}

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
    containing_cuts.sort_by(|a, b| b.lambda.partial_cmp(&a.lambda).unwrap_or(std::cmp::Ordering::Equal));

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
    sorted_cuts.sort_by(|a, b| b.lambda.partial_cmp(&a.lambda).unwrap_or(std::cmp::Ordering::Equal));

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
                let height = ((h_before * h_before * dx1 + h_after * h_after * dx2) / (dx1 + dx2)).sqrt();
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
    cuts.sort_by(|a, b| b.lambda.partial_cmp(&a.lambda).unwrap_or(std::cmp::Ordering::Equal));
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
    sorted_cuts.sort_by(|a, b| b.lambda.partial_cmp(&a.lambda).unwrap_or(std::cmp::Ordering::Equal));
    
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
    sorted_cuts.sort_by(|a, b| b.lambda.partial_cmp(&a.lambda).unwrap_or(std::cmp::Ordering::Equal));

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
        let dx = if is_near_discontinuity(current_x)
            || is_near_discontinuity(current_x + base_dx)
        {
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

/// Return 2×2×2 Gauss points and weights.
pub fn gauss_points_3d() -> ([Vector3<f64>; 8], SVector<f64, 8>) {
    let points = [
        Vector3::new(-GAUSS_G, -GAUSS_G, -GAUSS_G),
        Vector3::new(GAUSS_G, -GAUSS_G, -GAUSS_G),
        Vector3::new(GAUSS_G, GAUSS_G, -GAUSS_G),
        Vector3::new(-GAUSS_G, GAUSS_G, -GAUSS_G),
        Vector3::new(-GAUSS_G, -GAUSS_G, GAUSS_G),
        Vector3::new(GAUSS_G, -GAUSS_G, GAUSS_G),
        Vector3::new(GAUSS_G, GAUSS_G, GAUSS_G),
        Vector3::new(-GAUSS_G, GAUSS_G, GAUSS_G),
    ];
    let weights = SVector::<f64, 8>::from_element(1.0);
    (points, weights)
}

/// Hex8 shape functions at (xi, eta, zeta).
pub fn shape_functions_hex8(xi: f64, eta: f64, zeta: f64) -> SVector<f64, 8> {
    SVector::<f64, 8>::from_row_slice(&[
        (1.0 - xi) * (1.0 - eta) * (1.0 - zeta),
        (1.0 + xi) * (1.0 - eta) * (1.0 - zeta),
        (1.0 + xi) * (1.0 + eta) * (1.0 - zeta),
        (1.0 - xi) * (1.0 + eta) * (1.0 - zeta),
        (1.0 - xi) * (1.0 - eta) * (1.0 + zeta),
        (1.0 + xi) * (1.0 - eta) * (1.0 + zeta),
        (1.0 + xi) * (1.0 + eta) * (1.0 + zeta),
        (1.0 - xi) * (1.0 + eta) * (1.0 + zeta),
    ]) * 0.125
}

/// Shape function derivatives w.r.t. natural coordinates.
pub fn shape_function_derivatives_hex8(xi: f64, eta: f64, zeta: f64) -> Matrix3x8 {
    let mut d_n = Matrix3x8::zeros();

    d_n[(0, 0)] = -0.125 * (1.0 - eta) * (1.0 - zeta);
    d_n[(0, 1)] = 0.125 * (1.0 - eta) * (1.0 - zeta);
    d_n[(0, 2)] = 0.125 * (1.0 + eta) * (1.0 - zeta);
    d_n[(0, 3)] = -0.125 * (1.0 + eta) * (1.0 - zeta);
    d_n[(0, 4)] = -0.125 * (1.0 - eta) * (1.0 + zeta);
    d_n[(0, 5)] = 0.125 * (1.0 - eta) * (1.0 + zeta);
    d_n[(0, 6)] = 0.125 * (1.0 + eta) * (1.0 + zeta);
    d_n[(0, 7)] = -0.125 * (1.0 + eta) * (1.0 + zeta);

    d_n[(1, 0)] = -0.125 * (1.0 - xi) * (1.0 - zeta);
    d_n[(1, 1)] = -0.125 * (1.0 + xi) * (1.0 - zeta);
    d_n[(1, 2)] = 0.125 * (1.0 + xi) * (1.0 - zeta);
    d_n[(1, 3)] = 0.125 * (1.0 - xi) * (1.0 - zeta);
    d_n[(1, 4)] = -0.125 * (1.0 - xi) * (1.0 + zeta);
    d_n[(1, 5)] = -0.125 * (1.0 + xi) * (1.0 + zeta);
    d_n[(1, 6)] = 0.125 * (1.0 + xi) * (1.0 + zeta);
    d_n[(1, 7)] = 0.125 * (1.0 - xi) * (1.0 + zeta);

    d_n[(2, 0)] = -0.125 * (1.0 - xi) * (1.0 - eta);
    d_n[(2, 1)] = -0.125 * (1.0 + xi) * (1.0 - eta);
    d_n[(2, 2)] = -0.125 * (1.0 + xi) * (1.0 + eta);
    d_n[(2, 3)] = -0.125 * (1.0 - xi) * (1.0 + eta);
    d_n[(2, 4)] = 0.125 * (1.0 - xi) * (1.0 - eta);
    d_n[(2, 5)] = 0.125 * (1.0 + xi) * (1.0 - eta);
    d_n[(2, 6)] = 0.125 * (1.0 + xi) * (1.0 + eta);
    d_n[(2, 7)] = 0.125 * (1.0 - xi) * (1.0 + eta);

    d_n
}

/// 3D isotropic elasticity matrix (6×6).
pub fn elasticity_matrix_3d(e: f64, nu: f64) -> Matrix6 {
    let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mut d = Matrix6::zeros();

    d[(0, 0)] = factor * (1.0 - nu);
    d[(1, 1)] = factor * (1.0 - nu);
    d[(2, 2)] = factor * (1.0 - nu);

    d[(0, 1)] = factor * nu;
    d[(1, 0)] = factor * nu;
    d[(0, 2)] = factor * nu;
    d[(2, 0)] = factor * nu;
    d[(1, 2)] = factor * nu;
    d[(2, 1)] = factor * nu;

    let shear = factor * (1.0 - 2.0 * nu) / 2.0;
    d[(3, 3)] = shear;
    d[(4, 4)] = shear;
    d[(5, 5)] = shear;

    d
}

/// Compute stiffness and mass matrices for an 8-node hexahedron.
pub fn compute_hex8_matrices(
    node_coords: &NodeCoords,
    e: f64,
    nu: f64,
    rho: f64,
) -> (SMatrix<f64, 24, 24>, SMatrix<f64, 24, 24>) {
    let d = elasticity_matrix_3d(e, nu);
    let mut ke = SMatrix::<f64, 24, 24>::zeros();
    let mut me = SMatrix::<f64, 24, 24>::zeros();

    let (points, weights) = gauss_points_3d();

    for gp in 0..8 {
        let point = points[gp];
        let (xi, eta, zeta) = (point.x, point.y, point.z);
        let w = weights[gp];

        let n = shape_functions_hex8(xi, eta, zeta);
        let d_n_nat = shape_function_derivatives_hex8(xi, eta, zeta);

        let j: Matrix3<f64> = d_n_nat * node_coords;
        let det_j = j.determinant();
        let det_j_abs = det_j.abs();
        if det_j <= 0.0 || det_j_abs <= MIN_DET_J {
            continue;
        }
        let Some(j_inv) = j.try_inverse() else {
            continue;
        };

        let d_n_phys = j_inv * d_n_nat;
        let weight = w * det_j_abs;

        let mut b = Matrix6x24::zeros();
        for i in 0..8 {
            let col = 3 * i;
            let dndx = d_n_phys[(0, i)];
            let dndy = d_n_phys[(1, i)];
            let dndz = d_n_phys[(2, i)];

            b[(0, col)] = dndx;
            b[(1, col + 1)] = dndy;
            b[(2, col + 2)] = dndz;

            b[(3, col)] = dndy;
            b[(3, col + 1)] = dndx;
            b[(4, col + 1)] = dndz;
            b[(4, col + 2)] = dndy;
            b[(5, col)] = dndz;
            b[(5, col + 2)] = dndx;
        }

        ke += weight * (b.transpose() * d * b);

        let mut n_mat = Matrix3x24::zeros();
        for i in 0..8 {
            let col = 3 * i;
            let val = n[i];
            n_mat[(0, col)] = val;
            n_mat[(1, col + 1)] = val;
            n_mat[(2, col + 2)] = val;
        }

        me += (weight * rho) * (n_mat.transpose() * n_mat);
    }

    ke = 0.5 * (ke + ke.transpose());
    me = 0.5 * (me + me.transpose());

    (ke, me)
}

/// Simple sparse assembly using 3 DOF per node.
pub fn assemble_global_sparse(
    num_dofs: usize,
    elements: &[[usize; 8]],
    element_matrices: &[SMatrix<f64, 24, 24>],
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(num_dofs, num_dofs);

    for (elem_idx, nodes) in elements.iter().enumerate() {
        let dof_map: Vec<usize> = nodes
            .iter()
            .flat_map(|n| [3 * n, 3 * n + 1, 3 * n + 2])
            .collect();
        let local = &element_matrices[elem_idx];

        for i in 0..24 {
            for j in 0..24 {
                let val = local[(i, j)];
                if val.abs() > f64::EPSILON {
                    coo.push(dof_map[i], dof_map[j], val);
                }
            }
        }
    }

    CsrMatrix::from(&coo)
}

/// Dense assembly variant used for small test problems.
pub fn assemble_global_dense(
    num_dofs: usize,
    elements: &[[usize; 8]],
    element_matrices: &[SMatrix<f64, 24, 24>],
) -> DMatrix<f64> {
    let mut mat = DMatrix::<f64>::zeros(num_dofs, num_dofs);

    for (elem_idx, nodes) in elements.iter().enumerate() {
        let dof_map: Vec<usize> = nodes
            .iter()
            .flat_map(|n| [3 * n, 3 * n + 1, 3 * n + 2])
            .collect();
        let local = &element_matrices[elem_idx];

        for i in 0..24 {
            for j in 0..24 {
                let val = local[(i, j)];
                if val.abs() > f64::EPSILON {
                    mat[(dof_map[i], dof_map[j])] += val;
                }
            }
        }
    }

    mat
}

/// Mesh representation for the 3D bar.
#[derive(Debug, Clone)]
pub struct Mesh3d {
    pub nodes: Vec<Vector3<f64>>,
    pub elements: Vec<[usize; 8]>,
    pub heights_per_element: Vec<f64>,
}

/// Serializable 3D node position for frontend visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableNode {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Serializable hexahedral element for frontend visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableElement {
    /// Indices of the 8 nodes that form this hexahedral element.
    /// Node ordering: bottom face (z-), then top face (z+), counterclockwise.
    pub nodes: [usize; 8],
}

/// Serializable mesh data for frontend visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableMesh {
    /// Node positions in 3D space.
    pub nodes: Vec<SerializableNode>,
    /// Hexahedral elements defined by node indices.
    pub elements: Vec<SerializableElement>,
    /// Height value for each element (useful for coloring/visualization).
    pub element_heights: Vec<f64>,
    /// Mesh metadata.
    pub metadata: MeshMetadata,
}

/// Metadata about the mesh for frontend display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshMetadata {
    /// Total number of nodes.
    pub num_nodes: usize,
    /// Total number of elements.
    pub num_elements: usize,
    /// Total degrees of freedom (3 * num_nodes).
    pub num_dof: usize,
    /// Bounding box minimum coordinates.
    pub bbox_min: [f64; 3],
    /// Bounding box maximum coordinates.
    pub bbox_max: [f64; 3],
}

impl Mesh3d {
    /// Convert the mesh to a serializable format for frontend visualization.
    ///
    /// # Returns
    /// A `SerializableMesh` that can be serialized to JSON for frontend consumption.
    ///
    /// # Example
    /// ```
    /// use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d};
    ///
    /// let cuts = vec![Cut::new(0.1, 0.015)];
    /// let heights = generate_element_heights(&cuts, 0.5, 0.024, 20);
    /// let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 20, 2, 2);
    ///
    /// let serializable = mesh.to_serializable();
    /// let json = serde_json::to_string(&serializable).unwrap();
    /// ```
    pub fn to_serializable(&self) -> SerializableMesh {
        // Convert nodes
        let nodes: Vec<SerializableNode> = self
            .nodes
            .iter()
            .map(|v| SerializableNode {
                x: v.x,
                y: v.y,
                z: v.z,
            })
            .collect();

        // Convert elements
        let elements: Vec<SerializableElement> = self
            .elements
            .iter()
            .map(|&elem| SerializableElement { nodes: elem })
            .collect();

        // Compute bounding box
        let mut bbox_min = [f64::INFINITY; 3];
        let mut bbox_max = [f64::NEG_INFINITY; 3];

        for node in &self.nodes {
            bbox_min[0] = bbox_min[0].min(node.x);
            bbox_min[1] = bbox_min[1].min(node.y);
            bbox_min[2] = bbox_min[2].min(node.z);
            bbox_max[0] = bbox_max[0].max(node.x);
            bbox_max[1] = bbox_max[1].max(node.y);
            bbox_max[2] = bbox_max[2].max(node.z);
        }

        let metadata = MeshMetadata {
            num_nodes: self.nodes.len(),
            num_elements: self.elements.len(),
            num_dof: self.nodes.len() * 3,
            bbox_min,
            bbox_max,
        };

        SerializableMesh {
            nodes,
            elements,
            element_heights: self.heights_per_element.clone(),
            metadata,
        }
    }

    /// Export mesh to JSON string for frontend visualization.
    ///
    /// # Returns
    /// A JSON string representation of the mesh.
    ///
    /// # Example
    /// ```
    /// use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d};
    ///
    /// let cuts = vec![Cut::new(0.1, 0.015)];
    /// let heights = generate_element_heights(&cuts, 0.5, 0.024, 20);
    /// let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 20, 2, 2);
    ///
    /// let json = mesh.to_json().unwrap();
    /// println!("{}", json);
    /// ```
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let serializable = self.to_serializable();
        serde_json::to_string(&serializable)
    }

    /// Export mesh to pretty-printed JSON string for frontend visualization.
    ///
    /// # Returns
    /// A pretty-printed JSON string representation of the mesh.
    ///
    /// # Example
    /// ```
    /// use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d};
    ///
    /// let cuts = vec![Cut::new(0.1, 0.015)];
    /// let heights = generate_element_heights(&cuts, 0.5, 0.024, 20);
    /// let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 20, 2, 2);
    ///
    /// let json = mesh.to_json_pretty().unwrap();
    /// println!("{}", json);
    /// ```
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        let serializable = self.to_serializable();
        serde_json::to_string_pretty(&serializable)
    }
}

/// Generate uniform 3D mesh following the Python reference.
pub fn generate_bar_mesh_3d(
    length: f64,
    width: f64,
    element_heights: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Mesh3d {
    let expected_nx = element_heights.len();
    assert_eq!(nx, expected_nx, "nx must match element_heights length");
    let dx = length / expected_nx as f64;
    let dy = width / ny as f64;

    let nnx = expected_nx + 1;
    let nny = ny + 1;
    let nnz = nz + 1;

    let mut nodes = Vec::with_capacity(nnx * nny * nnz);
    let mut node_index = HashMap::new();

    for ix in 0..nnx {
        let x = ix as f64 * dx;
        let h = if ix == 0 {
            element_heights[0]
        } else if ix == expected_nx {
            element_heights[element_heights.len() - 1]
        } else {
            (element_heights[ix - 1] + element_heights[ix]) / 2.0
        };
        let dz = h / nz as f64;

        for iy in 0..nny {
            let y = iy as f64 * dy;
            for iz in 0..nnz {
                let z = iz as f64 * dz;
                let idx = nodes.len();
                node_index.insert((ix, iy, iz), idx);
                nodes.push(Vector3::new(x, y, z));
            }
        }
    }

    let mut elements = Vec::with_capacity(nx * ny * nz);
    let mut heights = Vec::with_capacity(nx * ny * nz);

    for ix in 0..expected_nx {
        let h = element_heights[ix];
        for iy in 0..ny {
            for iz in 0..nz {
                let n0 = node_index[&(ix, iy, iz)];
                let n1 = node_index[&(ix + 1, iy, iz)];
                let n2 = node_index[&(ix + 1, iy + 1, iz)];
                let n3 = node_index[&(ix, iy + 1, iz)];
                let n4 = node_index[&(ix, iy, iz + 1)];
                let n5 = node_index[&(ix + 1, iy, iz + 1)];
                let n6 = node_index[&(ix + 1, iy + 1, iz + 1)];
                let n7 = node_index[&(ix, iy + 1, iz + 1)];

                elements.push([n0, n1, n2, n3, n4, n5, n6, n7]);
                heights.push(h);
            }
        }
    }

    Mesh3d {
        nodes,
        elements,
        heights_per_element: heights,
    }
}

/// Generate adaptive mesh using supplied x-positions.
pub fn generate_bar_mesh_3d_adaptive(
    length: f64,
    width: f64,
    x_positions: &[f64],
    element_heights: &[f64],
    ny: usize,
    nz: usize,
) -> Mesh3d {
    let nx = element_heights.len();
    assert_eq!(
        x_positions.len(),
        nx + 1,
        "x_positions must have length len(element_heights)+1"
    );
    let end = *x_positions
        .last()
        .expect("x_positions must contain at least one position");
    assert!(
        (end - length).abs() < 1e-9,
        "x_positions must span the provided length"
    );

    let dy = width / ny as f64;
    let nnx = nx + 1;
    let nny = ny + 1;
    let nnz = nz + 1;

    let mut nodes = Vec::with_capacity(nnx * nny * nnz);
    let mut node_index = HashMap::new();

    for ix in 0..nnx {
        let x = x_positions[ix];
        let h = if ix == 0 {
            element_heights[0]
        } else if ix == nx {
            element_heights[element_heights.len() - 1]
        } else {
            (element_heights[ix - 1] + element_heights[ix]) / 2.0
        };
        let dz = h / nz as f64;

        for iy in 0..nny {
            let y = iy as f64 * dy;
            for iz in 0..nnz {
                let z = iz as f64 * dz;
                let idx = nodes.len();
                node_index.insert((ix, iy, iz), idx);
                nodes.push(Vector3::new(x, y, z));
            }
        }
    }

    let mut elements = Vec::with_capacity(nx * ny * nz);
    let mut heights = Vec::with_capacity(nx * ny * nz);
    for ix in 0..nx {
        let h = element_heights[ix];
        for iy in 0..ny {
            for iz in 0..nz {
                let n0 = node_index[&(ix, iy, iz)];
                let n1 = node_index[&(ix + 1, iy, iz)];
                let n2 = node_index[&(ix + 1, iy + 1, iz)];
                let n3 = node_index[&(ix, iy + 1, iz)];
                let n4 = node_index[&(ix, iy, iz + 1)];
                let n5 = node_index[&(ix + 1, iy, iz + 1)];
                let n6 = node_index[&(ix + 1, iy + 1, iz + 1)];
                let n7 = node_index[&(ix, iy + 1, iz + 1)];

                elements.push([n0, n1, n2, n3, n4, n5, n6, n7]);
                heights.push(h);
            }
        }
    }

    Mesh3d {
        nodes,
        elements,
        heights_per_element: heights,
    }
}

/// Build dense global stiffness and mass matrices for a mesh (intended for small test cases).
pub fn compute_global_matrices_dense(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let mut ke_list = Vec::with_capacity(mesh.elements.len());
    let mut me_list = Vec::with_capacity(mesh.elements.len());

    for nodes in &mesh.elements {
        let mut coords = NodeCoords::zeros();
        for (i, node_idx) in nodes.iter().copied().enumerate() {
            assert!(
                node_idx < mesh.nodes.len(),
                "element references invalid node index"
            );
            let node = mesh.nodes[node_idx];
            coords[(i, 0)] = node.x;
            coords[(i, 1)] = node.y;
            coords[(i, 2)] = node.z;
        }
        let (ke, me) = compute_hex8_matrices(&coords, e, nu, rho);
        ke_list.push(ke);
        me_list.push(me);
    }

    let num_dofs = mesh.nodes.len() * DOF_PER_NODE;
    (
        assemble_global_dense(num_dofs, &mesh.elements, &ke_list),
        assemble_global_dense(num_dofs, &mesh.elements, &me_list),
    )
}

fn filter_and_truncate_frequencies(mut freqs: Vec<f64>, num_modes: usize) -> Vec<f64> {
    freqs.retain(|f| *f > MIN_FREQUENCY_HZ);
    freqs.sort_by(|a, b| a.total_cmp(b));
    freqs.truncate(num_modes.min(freqs.len()));
    freqs
}

/// Compute modal frequencies (Hz) for a small mesh using dense generalized eigenvalue solve.
///
/// Returns an empty vector if the mass matrix is not positive definite.
pub fn compute_modal_frequencies(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
    num_modes: usize,
) -> Vec<f64> {
    let (k, m) = compute_global_matrices_dense(mesh, e, nu, rho);
    let Some(chol) = m.cholesky() else {
        return Vec::new();
    };
    let Some(l_inv) = chol.l().try_inverse() else {
        return Vec::new();
    };
    // Dense transform is acceptable here because this helper targets small test meshes.
    let a = l_inv.transpose() * k * l_inv;

    let eig = SymmetricEigen::new(a);
    let freqs: Vec<f64> = eig
        .eigenvalues
        .iter()
        .copied()
        .filter(|lambda| *lambda > RIGID_BODY_LAMBDA_THRESHOLD)
        .map(|lambda| lambda.sqrt() / (2.0 * std::f64::consts::PI))
        .collect();

    filter_and_truncate_frequencies(freqs, num_modes)
}

// ============================================================================
// Sparse Eigenvalue Solver (Shift-Invert Lanczos)
// ============================================================================

/// Build sparse global stiffness and mass matrices.
pub fn compute_global_matrices_sparse(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let mut ke_list = Vec::with_capacity(mesh.elements.len());
    let mut me_list = Vec::with_capacity(mesh.elements.len());

    for nodes in &mesh.elements {
        let mut coords = NodeCoords::zeros();
        for (i, node_idx) in nodes.iter().copied().enumerate() {
            let node = mesh.nodes[node_idx];
            coords[(i, 0)] = node.x;
            coords[(i, 1)] = node.y;
            coords[(i, 2)] = node.z;
        }
        let (ke, me) = compute_hex8_matrices(&coords, e, nu, rho);
        ke_list.push(ke);
        me_list.push(me);
    }

    let num_dofs = mesh.nodes.len() * DOF_PER_NODE;
    (
        assemble_global_sparse(num_dofs, &mesh.elements, &ke_list),
        assemble_global_sparse(num_dofs, &mesh.elements, &me_list),
    )
}

/// Sparse matrix-vector product: y = A * x
fn spmv(a: &CsrMatrix<f64>, x: &DVector<f64>) -> DVector<f64> {
    let n = a.nrows();
    let mut y = DVector::zeros(n);
    for (i, row) in a.row_iter().enumerate() {
        let mut sum = 0.0;
        for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
            sum += val * x[col];
        }
        y[i] = sum;
    }
    y
}

/// Build the shifted matrix (K - sigma * M) as a dense matrix for factorization.
/// For truly large systems, this should use sparse Cholesky, but dense is simpler
/// and works for moderate sizes.
fn build_shifted_matrix_dense(k: &CsrMatrix<f64>, m: &CsrMatrix<f64>, sigma: f64) -> DMatrix<f64> {
    let n = k.nrows();
    let mut a = DMatrix::zeros(n, n);

    // Add K
    for (i, row) in k.row_iter().enumerate() {
        for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
            a[(i, col)] += val;
        }
    }

    // Subtract sigma * M
    for (i, row) in m.row_iter().enumerate() {
        for (&col, &val) in row.col_indices().iter().zip(row.values().iter()) {
            a[(i, col)] -= sigma * val;
        }
    }

    a
}

/// Shift-invert Lanczos algorithm for generalized eigenvalue problem K*x = λ*M*x.
///
/// Uses shift-invert transformation: (K - σM)^(-1) * M * x = θ * x
/// where θ = 1/(λ - σ), so eigenvalues near σ become largest.
///
/// Returns (eigenvalues, eigenvectors) sorted by eigenvalue magnitude.
pub fn lanczos_shift_invert(
    k: &CsrMatrix<f64>,
    m: &CsrMatrix<f64>,
    num_modes: usize,
    sigma: f64,
) -> (Vec<f64>, DMatrix<f64>) {
    let n = k.nrows();
    let num_lanczos = (num_modes + 10).min(n - 1).min(MAX_LANCZOS_ITER);

    // Build and factor the shifted matrix A = K - sigma * M
    let a_shifted = build_shifted_matrix_dense(k, m, sigma);
    let chol = match a_shifted.clone().cholesky() {
        Some(c) => c,
        None => {
            // If Cholesky fails, try with regularization
            let mut a_reg = a_shifted;
            for i in 0..n {
                a_reg[(i, i)] += 1e-10 * a_reg[(i, i)].abs().max(1e-10);
            }
            match a_reg.cholesky() {
                Some(c) => c,
                None => return (Vec::new(), DMatrix::zeros(n, 0)),
            }
        }
    };

    // Initialize Lanczos vectors
    let mut v_prev = DVector::zeros(n);
    let mut v_curr = DVector::from_fn(n, |i, _| ((i * 7 + 13) % 101) as f64 / 100.0 - 0.5);

    // M-orthonormalize initial vector
    let mv = spmv(m, &v_curr);
    let norm = v_curr.dot(&mv).sqrt();
    if norm < 1e-14 {
        return (Vec::new(), DMatrix::zeros(n, 0));
    }
    v_curr /= norm;

    // Lanczos vectors storage
    let mut v_matrix = DMatrix::zeros(n, num_lanczos);
    let mut alpha = Vec::with_capacity(num_lanczos);
    let mut beta = Vec::with_capacity(num_lanczos);

    for j in 0..num_lanczos {
        v_matrix.set_column(j, &v_curr);

        // w = (K - sigma*M)^(-1) * M * v_curr
        let mv_curr = spmv(m, &v_curr);
        let w = chol.solve(&mv_curr);

        // alpha_j = w^T * M * v_curr
        let mw = spmv(m, &w);
        let alpha_j = v_curr.dot(&mw);
        alpha.push(alpha_j);

        // Orthogonalize: w = w - alpha_j * v_curr - beta_{j-1} * v_prev
        let mut w_orth = w - alpha_j * &v_curr;
        if j > 0 {
            w_orth -= beta[j - 1] * &v_prev;
        }

        // Full reorthogonalization (important for numerical stability)
        for k in 0..=j {
            let v_k = v_matrix.column(k);
            let mv_k = spmv(m, &v_k.clone_owned());
            let coeff = w_orth.dot(&mv_k);
            w_orth -= coeff * &v_k;
        }

        // beta_j = ||w||_M
        let mw_orth = spmv(m, &w_orth);
        let beta_j = w_orth.dot(&mw_orth).sqrt();

        if beta_j < LANCZOS_TOL {
            // Invariant subspace found
            alpha.truncate(j + 1);
            break;
        }

        beta.push(beta_j);
        v_prev = v_curr;
        v_curr = w_orth / beta_j;
    }

    let m_lanczos = alpha.len();
    if m_lanczos == 0 {
        return (Vec::new(), DMatrix::zeros(n, 0));
    }

    // Build tridiagonal matrix T
    let mut t_mat = DMatrix::zeros(m_lanczos, m_lanczos);
    for i in 0..m_lanczos {
        t_mat[(i, i)] = alpha[i];
        if i < beta.len() && i + 1 < m_lanczos {
            t_mat[(i, i + 1)] = beta[i];
            t_mat[(i + 1, i)] = beta[i];
        }
    }

    // Solve eigenvalue problem for tridiagonal matrix
    let eig = SymmetricEigen::new(t_mat);
    let theta = eig.eigenvalues;
    let s = eig.eigenvectors;

    // Convert theta back to lambda: lambda = sigma + 1/theta
    let mut eigen_pairs: Vec<(f64, DVector<f64>)> = Vec::new();
    for i in 0..m_lanczos {
        if theta[i].abs() > 1e-14 {
            let lambda = sigma + 1.0 / theta[i];

            // Only keep positive eigenvalues above rigid body threshold
            if lambda > RIGID_BODY_LAMBDA_THRESHOLD {
                // Reconstruct eigenvector: y = V * s_i
                let s_col = s.column(i);
                let mut y = DVector::zeros(n);
                for j in 0..m_lanczos {
                    y += s_col[j] * v_matrix.column(j);
                }

                // Normalize
                let norm = y.norm();
                if norm > 1e-14 {
                    y /= norm;
                    eigen_pairs.push((lambda, y));
                }
            }
        }
    }

    // Sort by eigenvalue (ascending)
    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Extract results
    let num_found = eigen_pairs.len().min(num_modes);
    let eigenvalues: Vec<f64> = eigen_pairs.iter().take(num_found).map(|(l, _)| *l).collect();
    let mut eigenvectors = DMatrix::zeros(n, num_found);
    for (i, (_, v)) in eigen_pairs.iter().take(num_found).enumerate() {
        eigenvectors.set_column(i, v);
    }

    (eigenvalues, eigenvectors)
}

/// Compute modal frequencies using sparse shift-invert Lanczos solver.
///
/// This is more efficient for large problems (> 500 DOF).
pub fn compute_modal_frequencies_sparse(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
    num_modes: usize,
) -> Vec<f64> {
    let (k, m) = compute_global_matrices_sparse(mesh, e, nu, rho);
    let (eigenvalues, _) = lanczos_shift_invert(&k, &m, num_modes, DEFAULT_SHIFT);

    let freqs: Vec<f64> = eigenvalues
        .iter()
        .map(|&lambda| lambda.sqrt() / (2.0 * std::f64::consts::PI))
        .collect();

    filter_and_truncate_frequencies(freqs, num_modes)
}

/// Compute modal frequencies with automatic solver selection.
///
/// Automatically chooses between dense and sparse solver based on problem size,
/// or uses the specified solver type.
///
/// # Arguments
/// * `mesh` - The 3D mesh
/// * `e` - Young's modulus (Pa)
/// * `nu` - Poisson's ratio
/// * `rho` - Density (kg/m³)
/// * `num_modes` - Number of modes to compute
/// * `solver` - Solver type (Dense, Sparse, or Auto)
///
/// # Returns
/// Vector of frequencies in Hz, sorted ascending.
pub fn compute_modal_frequencies_with_solver(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
    num_modes: usize,
    solver: EigenSolver,
) -> Vec<f64> {
    let num_dofs = mesh.nodes.len() * DOF_PER_NODE;

    let use_sparse = match solver {
        EigenSolver::Dense => false,
        EigenSolver::Sparse => true,
        EigenSolver::Auto => num_dofs > SPARSE_DOF_THRESHOLD,
    };

    if use_sparse {
        compute_modal_frequencies_sparse(mesh, e, nu, rho, num_modes)
    } else {
        compute_modal_frequencies(mesh, e, nu, rho, num_modes)
    }
}

// ============================================================================
// SPRS Backend (optional, requires "sprs-backend" feature)
// ============================================================================

/// Build sparse global matrices using sprs.
#[cfg(feature = "sprs-backend")]
pub fn compute_global_matrices_sprs(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (CsMat<f64>, CsMat<f64>) {
    let mut ke_list = Vec::with_capacity(mesh.elements.len());
    let mut me_list = Vec::with_capacity(mesh.elements.len());

    for nodes in &mesh.elements {
        let mut coords = NodeCoords::zeros();
        for (i, node_idx) in nodes.iter().copied().enumerate() {
            let node = mesh.nodes[node_idx];
            coords[(i, 0)] = node.x;
            coords[(i, 1)] = node.y;
            coords[(i, 2)] = node.z;
        }
        let (ke, me) = compute_hex8_matrices(&coords, e, nu, rho);
        ke_list.push(ke);
        me_list.push(me);
    }

    let num_dofs = mesh.nodes.len() * DOF_PER_NODE;
    (
        assemble_global_sprs(num_dofs, &mesh.elements, &ke_list),
        assemble_global_sprs(num_dofs, &mesh.elements, &me_list),
    )
}

/// Assemble global matrix using sprs triplet format.
#[cfg(feature = "sprs-backend")]
fn assemble_global_sprs(
    num_dofs: usize,
    elements: &[[usize; 8]],
    element_matrices: &[SMatrix<f64, 24, 24>],
) -> CsMat<f64> {
    let mut tri = TriMat::new((num_dofs, num_dofs));

    for (elem_idx, nodes) in elements.iter().enumerate() {
        let dof_map: Vec<usize> = nodes
            .iter()
            .flat_map(|n| [3 * n, 3 * n + 1, 3 * n + 2])
            .collect();
        let local = &element_matrices[elem_idx];

        for i in 0..24 {
            for j in 0..24 {
                let val = local[(i, j)];
                if val.abs() > f64::EPSILON {
                    tri.add_triplet(dof_map[i], dof_map[j], val);
                }
            }
        }
    }

    tri.to_csr()
}

/// Sparse matrix-vector product using sprs: y = A * x
#[cfg(feature = "sprs-backend")]
fn spmv_sprs(a: &CsMat<f64>, x: &DVector<f64>) -> DVector<f64> {
    let n = a.rows();
    let mut y = DVector::zeros(n);
    for (row_idx, row) in a.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (col_idx, &val) in row.iter() {
            sum += val * x[col_idx];
        }
        y[row_idx] = sum;
    }
    y
}

/// Build shifted matrix (K - σM) as dense matrix from sprs matrices.
#[cfg(feature = "sprs-backend")]
fn build_shifted_matrix_dense_sprs(k: &CsMat<f64>, m: &CsMat<f64>, sigma: f64) -> DMatrix<f64> {
    let n = k.rows();
    let mut a = DMatrix::zeros(n, n);

    // Add K
    for (row_idx, row) in k.outer_iterator().enumerate() {
        for (col_idx, &val) in row.iter() {
            a[(row_idx, col_idx)] += val;
        }
    }

    // Subtract sigma * M
    for (row_idx, row) in m.outer_iterator().enumerate() {
        for (col_idx, &val) in row.iter() {
            a[(row_idx, col_idx)] -= sigma * val;
        }
    }

    a
}

/// Shift-invert Lanczos using sprs matrices.
#[cfg(feature = "sprs-backend")]
pub fn lanczos_shift_invert_sprs(
    k: &CsMat<f64>,
    m: &CsMat<f64>,
    num_modes: usize,
    sigma: f64,
) -> (Vec<f64>, DMatrix<f64>) {
    let n = k.rows();
    let num_lanczos = (num_modes + 10).min(n - 1).min(MAX_LANCZOS_ITER);

    // Build and factor the shifted matrix A = K - sigma * M
    let a_shifted = build_shifted_matrix_dense_sprs(k, m, sigma);
    let chol = match a_shifted.clone().cholesky() {
        Some(c) => c,
        None => {
            let mut a_reg = a_shifted;
            for i in 0..n {
                a_reg[(i, i)] += 1e-10 * a_reg[(i, i)].abs().max(1e-10);
            }
            match a_reg.cholesky() {
                Some(c) => c,
                None => return (Vec::new(), DMatrix::zeros(n, 0)),
            }
        }
    };

    // Initialize Lanczos vectors
    let mut v_prev = DVector::zeros(n);
    let mut v_curr = DVector::from_fn(n, |i, _| ((i * 7 + 13) % 101) as f64 / 100.0 - 0.5);

    // M-orthonormalize initial vector
    let mv = spmv_sprs(m, &v_curr);
    let norm = v_curr.dot(&mv).sqrt();
    if norm < 1e-14 {
        return (Vec::new(), DMatrix::zeros(n, 0));
    }
    v_curr /= norm;

    // Lanczos vectors storage
    let mut v_matrix = DMatrix::zeros(n, num_lanczos);
    let mut alpha = Vec::with_capacity(num_lanczos);
    let mut beta = Vec::with_capacity(num_lanczos);

    for j in 0..num_lanczos {
        v_matrix.set_column(j, &v_curr);

        // w = (K - sigma*M)^(-1) * M * v_curr
        let mv_curr = spmv_sprs(m, &v_curr);
        let w = chol.solve(&mv_curr);

        // alpha_j = w^T * M * v_curr
        let mw = spmv_sprs(m, &w);
        let alpha_j = v_curr.dot(&mw);
        alpha.push(alpha_j);

        // Orthogonalize
        let mut w_orth = w - alpha_j * &v_curr;
        if j > 0 {
            w_orth -= beta[j - 1] * &v_prev;
        }

        // Full reorthogonalization
        for k in 0..=j {
            let v_k = v_matrix.column(k);
            let mv_k = spmv_sprs(m, &v_k.clone_owned());
            let coeff = w_orth.dot(&mv_k);
            w_orth -= coeff * &v_k;
        }

        // beta_j = ||w||_M
        let mw_orth = spmv_sprs(m, &w_orth);
        let beta_j = w_orth.dot(&mw_orth).sqrt();

        if beta_j < LANCZOS_TOL {
            alpha.truncate(j + 1);
            break;
        }

        beta.push(beta_j);
        v_prev = v_curr;
        v_curr = w_orth / beta_j;
    }

    let m_lanczos = alpha.len();
    if m_lanczos == 0 {
        return (Vec::new(), DMatrix::zeros(n, 0));
    }

    // Build tridiagonal matrix T
    let mut t_mat = DMatrix::zeros(m_lanczos, m_lanczos);
    for i in 0..m_lanczos {
        t_mat[(i, i)] = alpha[i];
        if i < beta.len() && i + 1 < m_lanczos {
            t_mat[(i, i + 1)] = beta[i];
            t_mat[(i + 1, i)] = beta[i];
        }
    }

    // Solve eigenvalue problem for tridiagonal matrix
    let eig = SymmetricEigen::new(t_mat);
    let theta = eig.eigenvalues;
    let s = eig.eigenvectors;

    // Convert theta back to lambda
    let mut eigen_pairs: Vec<(f64, DVector<f64>)> = Vec::new();
    for i in 0..m_lanczos {
        if theta[i].abs() > 1e-14 {
            let lambda = sigma + 1.0 / theta[i];

            if lambda > RIGID_BODY_LAMBDA_THRESHOLD {
                let s_col = s.column(i);
                let mut y = DVector::zeros(n);
                for j in 0..m_lanczos {
                    y += s_col[j] * v_matrix.column(j);
                }

                let norm = y.norm();
                if norm > 1e-14 {
                    y /= norm;
                    eigen_pairs.push((lambda, y));
                }
            }
        }
    }

    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let num_found = eigen_pairs.len().min(num_modes);
    let eigenvalues: Vec<f64> = eigen_pairs.iter().take(num_found).map(|(l, _)| *l).collect();
    let mut eigenvectors = DMatrix::zeros(n, num_found);
    for (i, (_, v)) in eigen_pairs.iter().take(num_found).enumerate() {
        eigenvectors.set_column(i, v);
    }

    (eigenvalues, eigenvectors)
}

/// Compute modal frequencies using sprs backend.
#[cfg(feature = "sprs-backend")]
pub fn compute_modal_frequencies_sprs(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
    num_modes: usize,
) -> Vec<f64> {
    let (k, m) = compute_global_matrices_sprs(mesh, e, nu, rho);
    let (eigenvalues, _) = lanczos_shift_invert_sprs(&k, &m, num_modes, DEFAULT_SHIFT);

    let freqs: Vec<f64> = eigenvalues
        .iter()
        .map(|&lambda| lambda.sqrt() / (2.0 * std::f64::consts::PI))
        .collect();

    filter_and_truncate_frequencies(freqs, num_modes)
}

/// Compute modal frequencies with configurable solver and backend.
///
/// # Arguments
/// * `mesh` - The 3D mesh
/// * `e` - Young's modulus (Pa)
/// * `nu` - Poisson's ratio
/// * `rho` - Density (kg/m³)
/// * `num_modes` - Number of modes to compute
/// * `solver` - Solver type (Dense, Sparse, or Auto)
/// * `backend` - Sparse backend (NalgebraSparse or Sprs)
///
/// # Returns
/// Vector of frequencies in Hz, sorted ascending.
pub fn compute_modal_frequencies_full(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
    num_modes: usize,
    solver: EigenSolver,
    backend: SparseBackend,
) -> Vec<f64> {
    let num_dofs = mesh.nodes.len() * DOF_PER_NODE;

    let use_sparse = match solver {
        EigenSolver::Dense => false,
        EigenSolver::Sparse => true,
        EigenSolver::Auto => num_dofs > SPARSE_DOF_THRESHOLD,
    };

    if !use_sparse {
        return compute_modal_frequencies(mesh, e, nu, rho, num_modes);
    }

    match backend {
        SparseBackend::NalgebraSparse => {
            compute_modal_frequencies_sparse(mesh, e, nu, rho, num_modes)
        }
        #[cfg(feature = "sprs-backend")]
        SparseBackend::Sprs => {
            compute_modal_frequencies_sprs(mesh, e, nu, rho, num_modes)
        }
    }
}

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
        let mode_type =
            classify_mode_soares(shape_col.as_slice(), nodes, corner_indices);
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

    const TEST_TOL: f64 = 1e-12;
    const ELASTICITY_TOL: f64 = 1e-9;

    #[test]
    fn shape_functions_sum_to_one() {
        let n = shape_functions_hex8(0.2, -0.3, 0.1);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn gauss_points_match_reference() {
        let (points, weights) = gauss_points_3d();
        let expected = [
            (-GAUSS_G, -GAUSS_G, -GAUSS_G),
            (GAUSS_G, -GAUSS_G, -GAUSS_G),
            (GAUSS_G, GAUSS_G, -GAUSS_G),
            (-GAUSS_G, GAUSS_G, -GAUSS_G),
            (-GAUSS_G, -GAUSS_G, GAUSS_G),
            (GAUSS_G, -GAUSS_G, GAUSS_G),
            (GAUSS_G, GAUSS_G, GAUSS_G),
            (-GAUSS_G, GAUSS_G, GAUSS_G),
        ];

        for (p, (ex, ey, ez)) in points.iter().zip(expected.iter()) {
            assert!((p.x - ex).abs() < TEST_TOL);
            assert!((p.y - ey).abs() < TEST_TOL);
            assert!((p.z - ez).abs() < TEST_TOL);
        }
        for w in weights.iter() {
            assert!((*w - 1.0).abs() < TEST_TOL);
        }
    }

    #[test]
    fn shape_function_derivatives_match_reference() {
        let (xi, eta, zeta) = (0.2, -0.4, 0.6);
        let d = shape_function_derivatives_hex8(xi, eta, zeta);
        // Expected derivatives from reference/details.md section 4 evaluated at (xi, eta, zeta) = (0.2, -0.4, 0.6)
        let expected = Matrix3x8::from_row_slice(&[
            -0.07, 0.07, 0.03, -0.03, -0.28, 0.28, 0.12, -0.12, // d/dxi
            -0.04, -0.06, 0.06, 0.04, -0.16, -0.24, 0.24, 0.16, // d/deta
            -0.14, -0.21, -0.09, -0.06, 0.14, 0.21, 0.09, 0.06, // d/dzeta
        ]);

        for i in 0..3 {
            for j in 0..8 {
                assert!((d[(i, j)] - expected[(i, j)]).abs() < TEST_TOL);
            }
        }
    }

    #[test]
    fn elasticity_matrix_matches_reference_definition() {
        let e_pa = 200e9;
        let nu = 0.3;
        let d = elasticity_matrix_3d(e_pa, nu);

        let factor = e_pa / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let normal = factor * (1.0 - nu);
        let coupling = factor * nu;
        let shear = factor * (1.0 - 2.0 * nu) / 2.0;

        assert!((d[(0, 0)] - normal).abs() < ELASTICITY_TOL);
        assert!((d[(1, 1)] - normal).abs() < ELASTICITY_TOL);
        assert!((d[(2, 2)] - normal).abs() < ELASTICITY_TOL);

        assert!((d[(0, 1)] - coupling).abs() < ELASTICITY_TOL);
        assert!((d[(1, 0)] - coupling).abs() < ELASTICITY_TOL);
        assert!((d[(0, 2)] - coupling).abs() < ELASTICITY_TOL);
        assert!((d[(2, 0)] - coupling).abs() < ELASTICITY_TOL);
        assert!((d[(1, 2)] - coupling).abs() < ELASTICITY_TOL);
        assert!((d[(2, 1)] - coupling).abs() < ELASTICITY_TOL);

        assert!((d[(3, 3)] - shear).abs() < ELASTICITY_TOL);
        assert!((d[(4, 4)] - shear).abs() < ELASTICITY_TOL);
        assert!((d[(5, 5)] - shear).abs() < ELASTICITY_TOL);
    }

    #[test]
    fn stiffness_and_mass_are_symmetric() {
        use nalgebra::RowVector3;

        let coords = NodeCoords::from_rows(&[
            RowVector3::new(0.0, 0.0, 0.0),
            RowVector3::new(1.0, 0.0, 0.0),
            RowVector3::new(1.0, 1.0, 0.0),
            RowVector3::new(0.0, 1.0, 0.0),
            RowVector3::new(0.0, 0.0, 1.0),
            RowVector3::new(1.0, 0.0, 1.0),
            RowVector3::new(1.0, 1.0, 1.0),
            RowVector3::new(0.0, 1.0, 1.0),
        ]);
        let (ke, me) = compute_hex8_matrices(&coords, 200e9, 0.3, 7800.0);

        for i in 0..24 {
            assert!(ke[(i, i)] >= 0.0);
            assert!(me[(i, i)] >= 0.0);
            for j in 0..24 {
                assert!((ke[(i, j)] - ke[(j, i)]).abs() < 1e-6);
                assert!((me[(i, j)] - me[(j, i)]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn mesh_generation_matches_expected_counts() {
        let heights = [1.0, 1.0];
        let mesh = generate_bar_mesh_3d(2.0, 1.0, &heights, 2, 2, 2);
        assert_eq!(mesh.nodes.len(), 27);
        assert_eq!(mesh.elements.len(), 8);
        assert_eq!(mesh.heights_per_element.len(), 8);
        assert_eq!(mesh.elements[0], [0, 9, 12, 3, 1, 10, 13, 4]);
    }

    #[test]
    fn sparse_assembly_builds_matrix() {
        let mut local = SMatrix::<f64, 24, 24>::zeros();
        for i in 0..24 {
            local[(i, i)] = 1.0;
        }
        let elements = [[0, 1, 2, 3, 4, 5, 6, 7]];
        let csr = assemble_global_sparse(24, &elements, &[local]);
        assert_eq!(csr.nrows(), 24);
        assert_eq!(csr.ncols(), 24);
        assert_eq!(csr.nnz(), 24);
    }

    #[test]
    fn dense_modal_analysis_returns_positive_frequencies() {
        let heights = [0.02];
        let mesh = generate_bar_mesh_3d(0.1, 0.02, &heights, 1, 1, 1);

        let freqs = compute_modal_frequencies(&mesh, 70e9, 0.33, 2700.0, 4);
        assert!(!freqs.is_empty());
        for pair in freqs.windows(2) {
            assert!(pair[0] <= pair[1] + LAMBDA_TOL);
        }
        for f in freqs {
            assert!(f.is_finite());
            assert!(f > 0.0);
        }
    }

    #[test]
    fn low_frequency_rigid_body_modes_are_filtered_out() {
        let freqs = vec![1.8, 1.8, 1.8, 357.7];
        let filtered = filter_and_truncate_frequencies(freqs, 4);
        assert_eq!(filtered, vec![357.7]);
    }

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
    fn sparse_solver_returns_positive_frequencies() {
        let heights: Vec<f64> = vec![0.02; 5];
        let mesh = generate_bar_mesh_3d(0.1, 0.02, &heights, 5, 2, 2);

        let freqs = compute_modal_frequencies_sparse(&mesh, 70e9, 0.33, 2700.0, 4);
        assert!(!freqs.is_empty(), "Sparse solver should return frequencies");
        for f in &freqs {
            assert!(f.is_finite(), "Frequencies should be finite");
            assert!(*f > 0.0, "Frequencies should be positive");
        }
    }

    #[test]
    fn sparse_and_dense_solvers_agree() {
        // Use a small mesh where both solvers should work
        let heights: Vec<f64> = vec![0.024; 5];
        let mesh = generate_bar_mesh_3d(0.2, 0.03, &heights, 5, 2, 2);

        let dense_freqs = compute_modal_frequencies(&mesh, 12e9, 0.35, 640.0, 4);
        let sparse_freqs = compute_modal_frequencies_sparse(&mesh, 12e9, 0.35, 640.0, 4);

        assert!(!dense_freqs.is_empty(), "Dense solver should return frequencies");
        assert!(!sparse_freqs.is_empty(), "Sparse solver should return frequencies");

        // Compare first few frequencies (allow 10% tolerance due to different algorithms)
        let num_compare = dense_freqs.len().min(sparse_freqs.len());
        for i in 0..num_compare {
            let rel_diff = (dense_freqs[i] - sparse_freqs[i]).abs() / dense_freqs[i];
            assert!(
                rel_diff < 0.10,
                "Frequency {} differs too much: dense={:.1}, sparse={:.1}, diff={:.1}%",
                i, dense_freqs[i], sparse_freqs[i], rel_diff * 100.0
            );
        }
    }

    #[test]
    fn auto_solver_selection_works() {
        let heights: Vec<f64> = vec![0.02; 3];
        let mesh = generate_bar_mesh_3d(0.1, 0.02, &heights, 3, 1, 1);

        // Small mesh should use dense (< 500 DOF)
        let freqs_auto = compute_modal_frequencies_with_solver(
            &mesh, 70e9, 0.33, 2700.0, 4, EigenSolver::Auto
        );
        let freqs_dense = compute_modal_frequencies_with_solver(
            &mesh, 70e9, 0.33, 2700.0, 4, EigenSolver::Dense
        );

        assert!(!freqs_auto.is_empty());
        assert_eq!(freqs_auto.len(), freqs_dense.len());
        for (a, d) in freqs_auto.iter().zip(freqs_dense.iter()) {
            assert!((a - d).abs() < 1e-6, "Auto should use dense for small mesh");
        }
    }

    #[cfg(feature = "sprs-backend")]
    #[test]
    fn sprs_solver_returns_positive_frequencies() {
        let heights: Vec<f64> = vec![0.02; 5];
        let mesh = generate_bar_mesh_3d(0.1, 0.02, &heights, 5, 2, 2);

        let freqs = compute_modal_frequencies_sprs(&mesh, 70e9, 0.33, 2700.0, 4);
        assert!(!freqs.is_empty(), "Sprs solver should return frequencies");
        for f in &freqs {
            assert!(f.is_finite(), "Frequencies should be finite");
            assert!(*f > 0.0, "Frequencies should be positive");
        }
    }

    #[cfg(feature = "sprs-backend")]
    #[test]
    fn sprs_and_nalgebra_sparse_agree() {
        // Use a small mesh where both backends should work
        let heights: Vec<f64> = vec![0.024; 5];
        let mesh = generate_bar_mesh_3d(0.2, 0.03, &heights, 5, 2, 2);

        let nalgebra_freqs = compute_modal_frequencies_sparse(&mesh, 12e9, 0.35, 640.0, 4);
        let sprs_freqs = compute_modal_frequencies_sprs(&mesh, 12e9, 0.35, 640.0, 4);

        assert!(!nalgebra_freqs.is_empty(), "Nalgebra-sparse solver should return frequencies");
        assert!(!sprs_freqs.is_empty(), "Sprs solver should return frequencies");

        // Compare first few frequencies (should be very close since same algorithm)
        let num_compare = nalgebra_freqs.len().min(sprs_freqs.len());
        for i in 0..num_compare {
            let rel_diff = (nalgebra_freqs[i] - sprs_freqs[i]).abs() / nalgebra_freqs[i];
            assert!(
                rel_diff < 0.01,
                "Frequency {} differs: nalgebra={:.1}, sprs={:.1}, diff={:.1}%",
                i, nalgebra_freqs[i], sprs_freqs[i], rel_diff * 100.0
            );
        }
    }

    #[cfg(feature = "sprs-backend")]
    #[test]
    fn full_api_with_sprs_backend_works() {
        let heights: Vec<f64> = vec![0.02; 5];
        let mesh = generate_bar_mesh_3d(0.1, 0.02, &heights, 5, 2, 2);

        let freqs = compute_modal_frequencies_full(
            &mesh, 70e9, 0.33, 2700.0, 4,
            EigenSolver::Sparse,
            SparseBackend::Sprs
        );
        assert!(!freqs.is_empty(), "Full API with Sprs should return frequencies");
    }

    // Tests for Cut and profile generation
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
        let cuts = vec![
            Cut::new(0.1, 0.015),
            Cut::new(0.2, 0.020),
        ];
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
        // Use cut position that doesn't align with element boundaries
        let cuts = [Cut::new(0.237, 0.012)]; // Offset position to ensure discontinuity is inside elements
        let length = 1.0;
        let h0 = 0.024;
        let num_elements = 50; // More elements for better chance of catching discontinuity
        
        let heights = generate_element_heights(&cuts, length, h0, num_elements);
        
        assert_eq!(heights.len(), num_elements);
        
        // Check that we have both cut height and original height
        let has_cut_height = heights.iter().any(|&h| (h - 0.012).abs() < 1e-3);
        let has_original_height = heights.iter().any(|&h| (h - 0.024).abs() < 1e-3);
        
        assert!(has_cut_height, "Should have elements at cut height");
        assert!(has_original_height, "Should have elements at original height");
        
        // At boundaries, some elements should have intermediate heights from interpolation
        // The quadratic interpolation should create values between the two heights
        let intermediate_heights: Vec<f64> = heights.iter()
            .filter(|&&h| h > 0.013 && h < 0.023)
            .copied()
            .collect();
        
        assert!(
            intermediate_heights.len() > 0,
            "Should find interpolated heights at boundaries. Got heights range: [{:.6}, {:.6}]",
            heights.iter().cloned().fold(f64::INFINITY, f64::min),
            heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
    }

    #[test]
    fn mesh_generation_with_cuts_integration() {
        // Integration test: create a mesh from cuts
        let cuts = [
            Cut::new(0.2, 0.020),
            Cut::new(0.1, 0.015),
        ];
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let num_elements_x = 20;
        
        let heights = generate_element_heights(&cuts, length, h0, num_elements_x);
        let mesh = generate_bar_mesh_3d(length, width, &heights, num_elements_x, 2, 2);
        
        assert_eq!(mesh.elements.len(), num_elements_x * 2 * 2);
        assert!(mesh.nodes.len() > 0);
        
        // Verify that the mesh has varying heights
        let min_height = heights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_height = heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        assert!(min_height < max_height, "Mesh should have varying heights from cuts");
        assert!(min_height >= 0.014, "Minimum height should be around cut height");
        assert!(max_height <= 0.025, "Maximum height should be around h0");
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

    #[test]
    fn adaptive_mesh_1d_integrates_with_3d_mesh() {
        let cuts = [
            Cut::new(0.20, 0.020),
            Cut::new(0.10, 0.015),
        ];
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let base_elements = 15;

        let (x_positions, element_heights) = generate_adaptive_mesh_1d(
            &cuts,
            length,
            h0,
            base_elements,
            4,
            0.02,
        );

        // Use adaptive mesh with generate_bar_mesh_3d_adaptive
        let mesh = generate_bar_mesh_3d_adaptive(
            length,
            width,
            &x_positions,
            &element_heights,
            2,
            2,
        );

        // Mesh should be valid
        assert!(mesh.nodes.len() > 0);
        assert!(mesh.elements.len() > 0);
        assert_eq!(mesh.heights_per_element.len(), mesh.elements.len());

        // Heights should vary
        let min_h = element_heights
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_h = element_heights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(min_h < max_h, "Adaptive mesh should have varying heights");
    }

    #[test]
    fn mesh_serialization_to_json() {
        let cuts = [Cut::new(0.1, 0.015)];
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let num_elements = 10;

        let heights = generate_element_heights(&cuts, length, h0, num_elements);
        let mesh = generate_bar_mesh_3d(length, width, &heights, num_elements, 2, 2);

        // Test serialization
        let serializable = mesh.to_serializable();
        assert_eq!(serializable.nodes.len(), mesh.nodes.len());
        assert_eq!(serializable.elements.len(), mesh.elements.len());
        assert_eq!(serializable.element_heights.len(), mesh.heights_per_element.len());

        // Verify metadata
        assert_eq!(serializable.metadata.num_nodes, mesh.nodes.len());
        assert_eq!(serializable.metadata.num_elements, mesh.elements.len());
        assert_eq!(serializable.metadata.num_dof, mesh.nodes.len() * 3);

        // Test JSON conversion
        let json = mesh.to_json().unwrap();
        assert!(json.contains("nodes"));
        assert!(json.contains("elements"));
        assert!(json.contains("metadata"));

        // Verify it can be deserialized
        let deserialized: SerializableMesh = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nodes.len(), mesh.nodes.len());
    }

    #[test]
    fn mesh_serialization_preserves_node_coordinates() {
        let cuts = [Cut::new(0.08, 0.012)];
        let heights = generate_element_heights(&cuts, 0.3, 0.024, 8);
        let mesh = generate_bar_mesh_3d(0.3, 0.025, &heights, 8, 2, 2);

        let serializable = mesh.to_serializable();

        // Check that node coordinates are preserved
        for (i, (original, serialized)) in mesh.nodes.iter().zip(serializable.nodes.iter()).enumerate() {
            assert!(
                (original.x - serialized.x).abs() < 1e-10,
                "Node {} x coordinate mismatch",
                i
            );
            assert!(
                (original.y - serialized.y).abs() < 1e-10,
                "Node {} y coordinate mismatch",
                i
            );
            assert!(
                (original.z - serialized.z).abs() < 1e-10,
                "Node {} z coordinate mismatch",
                i
            );
        }
    }

    #[test]
    fn mesh_serialization_bounding_box() {
        let cuts = [Cut::new(0.15, 0.018)];
        let heights = generate_element_heights(&cuts, 0.5, 0.024, 15);
        let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 15, 2, 3);

        let serializable = mesh.to_serializable();

        // Bounding box should contain all nodes
        for node in &serializable.nodes {
            assert!(node.x >= serializable.metadata.bbox_min[0]);
            assert!(node.y >= serializable.metadata.bbox_min[1]);
            assert!(node.z >= serializable.metadata.bbox_min[2]);
            assert!(node.x <= serializable.metadata.bbox_max[0]);
            assert!(node.y <= serializable.metadata.bbox_max[1]);
            assert!(node.z <= serializable.metadata.bbox_max[2]);
        }

        // Check bounding box makes sense
        assert!(serializable.metadata.bbox_min[0] < serializable.metadata.bbox_max[0]);
        assert!(serializable.metadata.bbox_min[1] < serializable.metadata.bbox_max[1]);
        assert!(serializable.metadata.bbox_min[2] < serializable.metadata.bbox_max[2]);
    }
}
