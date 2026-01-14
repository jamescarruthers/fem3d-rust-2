use std::collections::HashMap;

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

type NodeCoords = SMatrix<f64, 8, 3>;
type Matrix6 = SMatrix<f64, 6, 6>;
type Matrix6x24 = SMatrix<f64, 6, 24>;
type Matrix3x24 = SMatrix<f64, 3, 24>;
type Matrix3x8 = SMatrix<f64, 3, 8>;

/// Return 2×2×2 Gauss points and weights.
pub fn gauss_points_3d() -> ([Vector3<f64>; 8], SVector<f64, 8>) {
    let g = 1.0 / 3.0_f64.sqrt();
    let points = [
        Vector3::new(-g, -g, -g),
        Vector3::new(g, -g, -g),
        Vector3::new(g, g, -g),
        Vector3::new(-g, g, -g),
        Vector3::new(-g, -g, g),
        Vector3::new(g, -g, g),
        Vector3::new(g, g, g),
        Vector3::new(-g, g, g),
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
        if det_j.abs() <= f64::EPSILON {
            continue;
        }
        let Some(j_inv) = j.try_inverse() else {
            continue;
        };

        let d_n_phys = j_inv * d_n_nat;
        let weight = w * det_j.abs();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_functions_sum_to_one() {
        let n = shape_functions_hex8(0.2, -0.3, 0.1);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
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
}
