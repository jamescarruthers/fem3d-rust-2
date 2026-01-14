use std::collections::HashMap;

use nalgebra::linalg::SymmetricEigen;
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, Vector3};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

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
    let mut freqs: Vec<f64> = eig
        .eigenvalues
        .iter()
        .copied()
        .filter(|lambda| *lambda > RIGID_BODY_LAMBDA_THRESHOLD)
        .map(|lambda| lambda.sqrt() / (2.0 * std::f64::consts::PI))
        .collect();

    freqs.sort_by(|a, b| a.total_cmp(b));
    freqs.truncate(num_modes.min(freqs.len()));
    freqs
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

    eigenvalues
        .iter()
        .map(|&lambda| lambda.sqrt() / (2.0 * std::f64::consts::PI))
        .collect()
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

    eigenvalues
        .iter()
        .map(|&lambda| lambda.sqrt() / (2.0 * std::f64::consts::PI))
        .collect()
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
}
