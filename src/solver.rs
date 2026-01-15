//! Eigenvalue solvers for modal analysis.
//!
//! This module provides dense and sparse eigenvalue solvers for computing
//! natural frequencies and mode shapes of structural systems.

use nalgebra::linalg::SymmetricEigen;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;

#[cfg(feature = "sprs-backend")]
use sprs::CsMat;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::assembly::{assemble_global_dense, assemble_global_sparse};
use crate::element::compute_hex8_matrices;
use crate::mesh::Mesh3d;
use crate::types::{
    EigenSolver, Matrix24x24, NodeCoords, SparseBackend, DEFAULT_SHIFT, DOF_PER_NODE, LANCZOS_TOL,
    MAX_LANCZOS_ITER, MIN_FREQUENCY_HZ, RIGID_BODY_LAMBDA_THRESHOLD, SPARSE_DOF_THRESHOLD,
};

#[cfg(feature = "sprs-backend")]
use crate::assembly::assemble_global_sprs;

// ============================================================================
// Helper Functions
// ============================================================================

pub(crate) fn filter_and_truncate_frequencies(mut freqs: Vec<f64>, num_modes: usize) -> Vec<f64> {
    freqs.retain(|f| *f > MIN_FREQUENCY_HZ);
    freqs.sort_by(|a, b| a.total_cmp(b));
    freqs.truncate(num_modes.min(freqs.len()));
    freqs
}

/// Compute element stiffness and mass matrices for a single element.
/// This is extracted to enable parallel computation across elements.
#[inline]
fn compute_element_matrices(
    element_nodes: &[usize; 8],
    mesh_nodes: &[nalgebra::Vector3<f64>],
    e: f64,
    nu: f64,
    rho: f64,
) -> (Matrix24x24, Matrix24x24) {
    let mut coords = NodeCoords::zeros();
    for (i, &node_idx) in element_nodes.iter().enumerate() {
        let node = mesh_nodes[node_idx];
        coords[(i, 0)] = node.x;
        coords[(i, 1)] = node.y;
        coords[(i, 2)] = node.z;
    }
    compute_hex8_matrices(&coords, e, nu, rho)
}

/// Compute all element matrices sequentially.
#[cfg_attr(feature = "parallel", allow(dead_code))]
fn compute_all_element_matrices_sequential(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (Vec<Matrix24x24>, Vec<Matrix24x24>) {
    let mut ke_list = Vec::with_capacity(mesh.elements.len());
    let mut me_list = Vec::with_capacity(mesh.elements.len());

    for nodes in &mesh.elements {
        let (ke, me) = compute_element_matrices(nodes, &mesh.nodes, e, nu, rho);
        ke_list.push(ke);
        me_list.push(me);
    }

    (ke_list, me_list)
}

/// Compute all element matrices in parallel using Rayon.
/// Each element's matrices are computed independently, making this embarrassingly parallel.
#[cfg(feature = "parallel")]
fn compute_all_element_matrices_parallel(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (Vec<Matrix24x24>, Vec<Matrix24x24>) {
    let results: Vec<(Matrix24x24, Matrix24x24)> = mesh
        .elements
        .par_iter()
        .map(|nodes| compute_element_matrices(nodes, &mesh.nodes, e, nu, rho))
        .collect();

    results.into_iter().unzip()
}

/// Compute all element matrices, using parallel computation when available.
#[inline]
fn compute_all_element_matrices(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (Vec<Matrix24x24>, Vec<Matrix24x24>) {
    #[cfg(feature = "parallel")]
    {
        compute_all_element_matrices_parallel(mesh, e, nu, rho)
    }
    #[cfg(not(feature = "parallel"))]
    {
        compute_all_element_matrices_sequential(mesh, e, nu, rho)
    }
}

/// Sparse matrix-vector product: y = A * x
/// Uses nalgebra-sparse's optimized implementation.
#[inline(always)]
fn spmv(a: &CsrMatrix<f64>, x: &DVector<f64>) -> DVector<f64> {
    a * x
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

// ============================================================================
// Dense Solver
// ============================================================================

/// Build dense global stiffness and mass matrices for a mesh (intended for small test cases).
/// Uses parallel element matrix computation when the "parallel" feature is enabled.
pub fn compute_global_matrices_dense(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let (ke_list, me_list) = compute_all_element_matrices(mesh, e, nu, rho);

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
/// Uses parallel element matrix computation when the "parallel" feature is enabled.
pub fn compute_global_matrices_sparse(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (CsrMatrix<f64>, CsrMatrix<f64>) {
    let (ke_list, me_list) = compute_all_element_matrices(mesh, e, nu, rho);

    let num_dofs = mesh.nodes.len() * DOF_PER_NODE;
    (
        assemble_global_sparse(num_dofs, &mesh.elements, &ke_list),
        assemble_global_sparse(num_dofs, &mesh.elements, &me_list),
    )
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
    // Use LU decomposition since the shifted matrix may be indefinite
    // (rigid body modes have λ ≈ 0, so K - σM has negative eigenvalues for σ > 0)
    let a_shifted = build_shifted_matrix_dense(k, m, sigma);
    let lu = a_shifted.clone().lu();

    // Check if LU decomposition succeeded (matrix is non-singular)
    let lu_factor = if lu.is_invertible() {
        lu
    } else {
        // Try with regularization
        let mut a_reg = a_shifted;
        for i in 0..n {
            a_reg[(i, i)] += 1e-8 * a_reg[(i, i)].abs().max(1e-8);
        }
        let lu_reg = a_reg.lu();
        if !lu_reg.is_invertible() {
            return (Vec::new(), DMatrix::zeros(n, 0));
        }
        lu_reg
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
    // Cache M*v products to avoid recomputing during reorthogonalization
    // This reduces O(m²) SPMV calls to O(m), a significant optimization
    let mut mv_matrix = DMatrix::zeros(n, num_lanczos);
    let mut alpha = Vec::with_capacity(num_lanczos);
    let mut beta = Vec::with_capacity(num_lanczos);

    for j in 0..num_lanczos {
        v_matrix.set_column(j, &v_curr);

        // w = (K - sigma*M)^(-1) * M * v_curr
        let mv_curr = spmv(m, &v_curr);
        // Cache M*v_curr for reorthogonalization
        mv_matrix.set_column(j, &mv_curr);

        let w = lu_factor
            .solve(&mv_curr)
            .unwrap_or_else(|| DVector::zeros(n));

        // alpha_j = w^T * M * v_curr
        let mw = spmv(m, &w);
        let alpha_j = v_curr.dot(&mw);
        alpha.push(alpha_j);

        // Orthogonalize: w = w - alpha_j * v_curr - beta_{j-1} * v_prev
        let mut w_orth = w - alpha_j * &v_curr;
        if j > 0 {
            w_orth -= beta[j - 1] * &v_prev;
        }

        // Full reorthogonalization using cached M*v products
        // Previously this was O(j) SPMV calls per iteration = O(m²) total
        // Now it's O(1) lookups per iteration = O(m) total
        for k in 0..=j {
            let v_k = v_matrix.column(k);
            let mv_k = mv_matrix.column(k);
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
/// Uses parallel element matrix computation when the "parallel" feature is enabled.
#[cfg(feature = "sprs-backend")]
pub fn compute_global_matrices_sprs(
    mesh: &Mesh3d,
    e: f64,
    nu: f64,
    rho: f64,
) -> (CsMat<f64>, CsMat<f64>) {
    let (ke_list, me_list) = compute_all_element_matrices(mesh, e, nu, rho);

    let num_dofs = mesh.nodes.len() * DOF_PER_NODE;
    (
        assemble_global_sprs(num_dofs, &mesh.elements, &ke_list),
        assemble_global_sprs(num_dofs, &mesh.elements, &me_list),
    )
}

/// Sparse matrix-vector product using sprs: y = A * x
/// Uses sprs's optimized mul_acc_mat_vec_csr for better performance.
#[cfg(feature = "sprs-backend")]
#[inline(always)]
fn spmv_sprs(a: &CsMat<f64>, x: &DVector<f64>) -> DVector<f64> {
    let n = a.rows();
    let mut y = DVector::zeros(n);
    // Use sprs's optimized sparse matrix-vector multiplication
    sprs::prod::mul_acc_mat_vec_csr(
        a.view(),
        x.as_slice(),
        y.as_mut_slice(),
    );
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
    // Use LU decomposition since the shifted matrix may be indefinite
    let a_shifted = build_shifted_matrix_dense_sprs(k, m, sigma);
    let lu = a_shifted.clone().lu();

    let lu_factor = if lu.is_invertible() {
        lu
    } else {
        let mut a_reg = a_shifted;
        for i in 0..n {
            a_reg[(i, i)] += 1e-8 * a_reg[(i, i)].abs().max(1e-8);
        }
        let lu_reg = a_reg.lu();
        if !lu_reg.is_invertible() {
            return (Vec::new(), DMatrix::zeros(n, 0));
        }
        lu_reg
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
    // Cache M*v products to avoid recomputing during reorthogonalization
    // This reduces O(m²) SPMV calls to O(m), a significant optimization
    let mut mv_matrix = DMatrix::zeros(n, num_lanczos);
    let mut alpha = Vec::with_capacity(num_lanczos);
    let mut beta = Vec::with_capacity(num_lanczos);

    for j in 0..num_lanczos {
        v_matrix.set_column(j, &v_curr);

        // w = (K - sigma*M)^(-1) * M * v_curr
        let mv_curr = spmv_sprs(m, &v_curr);
        // Cache M*v_curr for reorthogonalization
        mv_matrix.set_column(j, &mv_curr);

        let w = lu_factor
            .solve(&mv_curr)
            .unwrap_or_else(|| DVector::zeros(n));

        // alpha_j = w^T * M * v_curr
        let mw = spmv_sprs(m, &w);
        let alpha_j = v_curr.dot(&mw);
        alpha.push(alpha_j);

        // Orthogonalize
        let mut w_orth = w - alpha_j * &v_curr;
        if j > 0 {
            w_orth -= beta[j - 1] * &v_prev;
        }

        // Full reorthogonalization using cached M*v products
        // Previously this was O(j) SPMV calls per iteration = O(m²) total
        // Now it's O(1) lookups per iteration = O(m) total
        for k in 0..=j {
            let v_k = v_matrix.column(k);
            let mv_k = mv_matrix.column(k);
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
        SparseBackend::Sprs => compute_modal_frequencies_sprs(mesh, e, nu, rho, num_modes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::generate_bar_mesh_3d;
    use crate::types::LAMBDA_TOL;

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

        assert!(
            !dense_freqs.is_empty(),
            "Dense solver should return frequencies"
        );
        assert!(
            !sparse_freqs.is_empty(),
            "Sparse solver should return frequencies"
        );

        // Compare first few frequencies (allow 10% tolerance due to different algorithms)
        let num_compare = dense_freqs.len().min(sparse_freqs.len());
        for i in 0..num_compare {
            let rel_diff = (dense_freqs[i] - sparse_freqs[i]).abs() / dense_freqs[i];
            assert!(
                rel_diff < 0.10,
                "Frequency {} differs too much: dense={:.1}, sparse={:.1}, diff={:.1}%",
                i,
                dense_freqs[i],
                sparse_freqs[i],
                rel_diff * 100.0
            );
        }
    }

    #[test]
    fn auto_solver_selection_works() {
        let heights: Vec<f64> = vec![0.02; 3];
        let mesh = generate_bar_mesh_3d(0.1, 0.02, &heights, 3, 1, 1);

        // Small mesh should use dense (< 500 DOF)
        let freqs_auto =
            compute_modal_frequencies_with_solver(&mesh, 70e9, 0.33, 2700.0, 4, EigenSolver::Auto);
        let freqs_dense =
            compute_modal_frequencies_with_solver(&mesh, 70e9, 0.33, 2700.0, 4, EigenSolver::Dense);

        assert!(!freqs_auto.is_empty());
        assert_eq!(freqs_auto.len(), freqs_dense.len());
        for (a, d) in freqs_auto.iter().zip(freqs_dense.iter()) {
            assert!(
                (a - d).abs() < 1e-6,
                "Auto should use dense for small mesh"
            );
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

        assert!(
            !nalgebra_freqs.is_empty(),
            "Nalgebra-sparse solver should return frequencies"
        );
        assert!(
            !sprs_freqs.is_empty(),
            "Sprs solver should return frequencies"
        );

        // Compare first few frequencies (should be very close since same algorithm)
        let num_compare = nalgebra_freqs.len().min(sprs_freqs.len());
        for i in 0..num_compare {
            let rel_diff = (nalgebra_freqs[i] - sprs_freqs[i]).abs() / nalgebra_freqs[i];
            assert!(
                rel_diff < 0.01,
                "Frequency {} differs: nalgebra={:.1}, sprs={:.1}, diff={:.1}%",
                i,
                nalgebra_freqs[i],
                sprs_freqs[i],
                rel_diff * 100.0
            );
        }
    }

    #[cfg(feature = "sprs-backend")]
    #[test]
    fn full_api_with_sprs_backend_works() {
        let heights: Vec<f64> = vec![0.02; 5];
        let mesh = generate_bar_mesh_3d(0.1, 0.02, &heights, 5, 2, 2);

        let freqs = compute_modal_frequencies_full(
            &mesh,
            70e9,
            0.33,
            2700.0,
            4,
            EigenSolver::Sparse,
            SparseBackend::Sprs,
        );
        assert!(
            !freqs.is_empty(),
            "Full API with Sprs should return frequencies"
        );
    }
}
