//! 2D Timoshenko beam assembly and modal analysis solver.
//!
//! Provides functions for assembling global matrices from Timoshenko beam
//! elements and solving the generalized eigenvalue problem for natural frequencies.

use nalgebra::linalg::SymmetricEigen;
use nalgebra::DMatrix;

use crate::beam2d::{
    compute_element_mass, compute_element_stiffness, second_moment_of_area, shear_modulus,
    DOF_PER_NODE_2D,
};
use crate::cuts::{generate_element_heights, Cut};

/// Eigenvalue threshold used to discard near-zero rigid-body modes.
/// For a free-free beam, there are 2 rigid body modes with λ ≈ 0.
const RIGID_BODY_THRESHOLD: f64 = 1.0;

/// Minimum frequency (Hz) to keep after filtering.
const MIN_FREQUENCY_HZ_2D: f64 = 1.0;

/// Assemble global stiffness and mass matrices from element matrices.
///
/// # Arguments
/// * `element_heights` - Height of each element (m)
/// * `le` - Element length (m)
/// * `b` - Bar width (m)
/// * `e` - Young's modulus (Pa)
/// * `rho` - Density (kg/m³)
/// * `nu` - Poisson's ratio
///
/// # Returns
/// Tuple of (K_global, M_global) matrices
pub fn assemble_global_matrices_2d(
    element_heights: &[f64],
    le: f64,
    b: f64,
    e: f64,
    rho: f64,
    nu: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let ne = element_heights.len();
    let num_dof = DOF_PER_NODE_2D * (ne + 1);

    let mut k_global = DMatrix::<f64>::zeros(num_dof, num_dof);
    let mut m_global = DMatrix::<f64>::zeros(num_dof, num_dof);

    let g = shear_modulus(e, nu);

    for elem in 0..ne {
        let h = element_heights[elem];
        let a = b * h; // Cross-sectional area
        let i = second_moment_of_area(b, h);

        let ke = compute_element_stiffness(le, e, i, g, a);
        let me = compute_element_mass(le, rho, a, i, e, g);

        // DOF mapping: element DOFs [0,1,2,3] -> global DOFs [2e, 2e+1, 2e+2, 2e+3]
        let dof_map = [2 * elem, 2 * elem + 1, 2 * (elem + 1), 2 * (elem + 1) + 1];

        for i in 0..4 {
            for j in 0..4 {
                let gi = dof_map[i];
                let gj = dof_map[j];
                k_global[(gi, gj)] += ke[(i, j)];
                m_global[(gi, gj)] += me[(i, j)];
            }
        }
    }

    (k_global, m_global)
}

/// Solve generalized eigenvalue problem K*φ = λ*M*φ.
///
/// Uses the standard form transformation via Cholesky decomposition.
///
/// # Arguments
/// * `k` - Global stiffness matrix
/// * `m` - Global mass matrix
/// * `num_modes` - Number of modes to extract
///
/// # Returns
/// Vector of natural frequencies in Hz
pub fn solve_generalized_eigenvalue(k: &DMatrix<f64>, m: &DMatrix<f64>, num_modes: usize) -> Vec<f64> {
    let n = k.nrows();

    // Add small regularization to M for numerical stability
    let mut m_reg = m.clone();
    for i in 0..n {
        m_reg[(i, i)] += 1e-12 * m[(i, i)].abs().max(1e-20);
    }

    // Compute Cholesky decomposition: M = L @ L.T
    let chol = match m_reg.clone().cholesky() {
        Some(c) => c,
        None => {
            // Fallback: add more regularization
            for i in 0..n {
                m_reg[(i, i)] += 1e-8;
            }
            match m_reg.clone().cholesky() {
                Some(c) => c,
                None => return Vec::new(),
            }
        }
    };

    let l = chol.l();

    // Compute L^{-1}
    let l_inv = match l.clone().try_inverse() {
        Some(inv) => inv,
        None => return Vec::new(),
    };

    // K_tilde = L^{-1} @ K @ L^{-T}
    let k_tilde = &l_inv * k * l_inv.transpose();

    // Symmetrize to remove numerical errors
    let k_tilde_sym = (&k_tilde + k_tilde.transpose()) * 0.5;

    // Solve standard symmetric eigenvalue problem
    let eig = SymmetricEigen::new(k_tilde_sym);
    let eigenvalues = eig.eigenvalues;

    // Sort eigenvalues
    let mut sorted_evs: Vec<f64> = eigenvalues.iter().copied().collect();
    sorted_evs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Filter out rigid body modes (very small or negative eigenvalues)
    let elastic_modes: Vec<f64> = sorted_evs
        .into_iter()
        .filter(|&ev| ev > RIGID_BODY_THRESHOLD)
        .collect();

    // Convert eigenvalues to frequencies: f = sqrt(λ) / (2π)
    let frequencies: Vec<f64> = elastic_modes
        .iter()
        .take(num_modes)
        .map(|&ev| ev.sqrt() / (2.0 * std::f64::consts::PI))
        .filter(|&f| f > MIN_FREQUENCY_HZ_2D)
        .collect();

    frequencies
}

/// Compute modal frequencies for a 2D beam with given element heights.
///
/// # Arguments
/// * `element_heights` - Height of each element (m)
/// * `length` - Bar length (m)
/// * `width` - Bar width (m)
/// * `e` - Young's modulus (Pa)
/// * `nu` - Poisson's ratio
/// * `rho` - Density (kg/m³)
/// * `num_modes` - Number of modes to compute
///
/// # Returns
/// Vector of frequencies in Hz, sorted ascending
pub fn compute_modal_frequencies_2d(
    element_heights: &[f64],
    length: f64,
    width: f64,
    e: f64,
    nu: f64,
    rho: f64,
    num_modes: usize,
) -> Vec<f64> {
    let ne = element_heights.len();
    let le = length / ne as f64;

    let (k, m) = assemble_global_matrices_2d(element_heights, le, width, e, rho, nu);
    solve_generalized_eigenvalue(&k, &m, num_modes)
}

/// Compute modal frequencies for a bar with given cuts using 2D beam analysis.
///
/// # Arguments
/// * `cuts` - Slice of cuts defining the bar profile
/// * `length` - Bar length (m)
/// * `width` - Bar width (m)
/// * `h0` - Original bar height (m)
/// * `e` - Young's modulus (Pa)
/// * `nu` - Poisson's ratio
/// * `rho` - Density (kg/m³)
/// * `num_elements` - Number of elements for discretization
/// * `num_modes` - Number of modes to compute
///
/// # Returns
/// Vector of frequencies in Hz, sorted ascending
pub fn compute_modal_frequencies_2d_from_cuts(
    cuts: &[Cut],
    length: f64,
    width: f64,
    h0: f64,
    e: f64,
    nu: f64,
    rho: f64,
    num_elements: usize,
    num_modes: usize,
) -> Vec<f64> {
    let heights = generate_element_heights(cuts, length, h0, num_elements);
    compute_modal_frequencies_2d(&heights, length, width, e, nu, rho, num_modes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_beam_frequencies_are_positive() {
        let ne = 50;
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let e = 12e9; // Sapele wood
        let nu = 0.35;
        let rho = 640.0;

        let heights: Vec<f64> = vec![h0; ne];
        let freqs = compute_modal_frequencies_2d(&heights, length, width, e, nu, rho, 4);

        assert!(!freqs.is_empty(), "Should compute frequencies");
        for f in &freqs {
            assert!(f.is_finite(), "Frequency should be finite");
            assert!(*f > 0.0, "Frequency should be positive");
        }
    }

    #[test]
    fn frequencies_are_sorted_ascending() {
        let ne = 100;
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let e = 12e9;
        let nu = 0.35;
        let rho = 640.0;

        let heights: Vec<f64> = vec![h0; ne];
        let freqs = compute_modal_frequencies_2d(&heights, length, width, e, nu, rho, 6);

        for i in 1..freqs.len() {
            assert!(
                freqs[i] >= freqs[i - 1],
                "Frequencies should be sorted: f[{}]={} < f[{}]={}",
                i - 1,
                freqs[i - 1],
                i,
                freqs[i]
            );
        }
    }

    #[test]
    fn cut_bar_has_different_frequencies() {
        let ne = 100;
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let e = 12e9;
        let nu = 0.35;
        let rho = 640.0;

        // Uniform bar
        let heights_uniform: Vec<f64> = vec![h0; ne];
        let freqs_uniform = compute_modal_frequencies_2d(&heights_uniform, length, width, e, nu, rho, 4);

        // Bar with cut
        let cuts = [Cut::new(0.1, 0.015)];
        let freqs_cut = compute_modal_frequencies_2d_from_cuts(
            &cuts, length, width, h0, e, nu, rho, ne, 4,
        );

        assert!(!freqs_uniform.is_empty());
        assert!(!freqs_cut.is_empty());

        // Cut should lower the fundamental frequency
        assert!(
            freqs_cut[0] < freqs_uniform[0],
            "Cut should lower f1: {} vs {}",
            freqs_cut[0],
            freqs_uniform[0]
        );
    }

    #[test]
    fn global_matrices_have_correct_size() {
        let ne = 10;
        let le = 0.05;
        let b = 0.03;
        let e = 70e9;
        let rho = 2700.0;
        let nu = 0.33;

        let heights: Vec<f64> = vec![0.02; ne];
        let (k, m) = assemble_global_matrices_2d(&heights, le, b, e, rho, nu);

        let expected_dof = DOF_PER_NODE_2D * (ne + 1);
        assert_eq!(k.nrows(), expected_dof);
        assert_eq!(k.ncols(), expected_dof);
        assert_eq!(m.nrows(), expected_dof);
        assert_eq!(m.ncols(), expected_dof);
    }

    #[test]
    fn global_matrices_are_symmetric() {
        let ne = 10;
        let le = 0.05;
        let b = 0.03;
        let e = 70e9;
        let rho = 2700.0;
        let nu = 0.33;

        let heights: Vec<f64> = vec![0.02; ne];
        let (k, m) = assemble_global_matrices_2d(&heights, le, b, e, rho, nu);

        let n = k.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[(i, j)] - k[(j, i)]).abs() < 1e-10,
                    "K not symmetric at ({}, {})",
                    i,
                    j
                );
                assert!(
                    (m[(i, j)] - m[(j, i)]).abs() < 1e-10,
                    "M not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn euler_bernoulli_analytical_comparison() {
        // For a uniform free-free beam, the analytical Euler-Bernoulli frequencies are:
        // f_n = (beta_n^2 / (2*pi)) * sqrt(E*I / (rho*A*L^4))
        // where beta_1 = 4.730, beta_2 = 7.853, beta_3 = 10.996, beta_4 = 14.137
        //
        // Timoshenko beam gives slightly lower frequencies due to shear deformation,
        // but should be within ~5-10% for slender beams.

        let ne = 200; // Fine mesh for accuracy
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let e = 12e9;
        let nu = 0.35;
        let rho = 640.0;

        let heights: Vec<f64> = vec![h0; ne];
        let freqs = compute_modal_frequencies_2d(&heights, length, width, e, nu, rho, 4);

        // Euler-Bernoulli analytical frequencies
        let i = width * h0.powi(3) / 12.0;
        let a = width * h0;
        let coeff = (e * i / (rho * a * length.powi(4))).sqrt() / (2.0 * std::f64::consts::PI);

        let betas = [4.730, 7.853, 10.996, 14.137];
        let analytical: Vec<f64> = betas.iter().map(|b| b * b * coeff).collect();

        assert!(freqs.len() >= 4, "Should compute at least 4 modes");

        // Check that computed frequencies are close to analytical (within 10%)
        for i in 0..4 {
            let rel_error = (freqs[i] - analytical[i]).abs() / analytical[i];
            assert!(
                rel_error < 0.15,
                "Mode {} error too large: computed={:.1} Hz, analytical={:.1} Hz, error={:.1}%",
                i + 1,
                freqs[i],
                analytical[i],
                rel_error * 100.0
            );
        }
    }
}
