use std::collections::HashMap;

use nalgebra::{DMatrix, Matrix3, SMatrix, SVector, Vector3};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

type NodeCoords = SMatrix<f64, 8, 3>;
type Matrix6 = SMatrix<f64, 6, 6>;
type Matrix6x24 = SMatrix<f64, 6, 24>;
type Matrix3x24 = SMatrix<f64, 3, 24>;
type Matrix3x8 = SMatrix<f64, 3, 8>;
const DOF_PER_NODE: usize = 3;
const DEFAULT_CORNER_TOL: f64 = 1e-6;
const Z_DIR_INDEX: usize = 2;
// 1/sqrt(3) Gauss point coordinate for 2×2×2 quadrature
const GAUSS_G: f64 = 0.577_350_269_189_625_8;
const MIN_DET_J: f64 = 1e-12;

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
        if det_j <= MIN_DET_J {
            continue;
        }
        let det_j_abs = det_j.abs();
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

/// Mesh representation for the 3D bar.
#[derive(Debug, Clone)]
pub struct Mesh3d {
    pub nodes: Vec<Vector3<f64>>,
    pub elements: Vec<[usize; 8]>,
    pub heights_per_element: Vec<f64>,
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

    for (idx, freq) in frequencies.iter().copied().enumerate() {
        if idx >= mode_shapes.ncols() {
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
        // Expected derivatives from reference formulas at (xi, eta, zeta) = (0.2, -0.4, 0.6)
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
}
