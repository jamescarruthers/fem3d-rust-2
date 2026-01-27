//! 1D Timoshenko frame element assembly and modal analysis solver.
//!
//! Provides functions for:
//! - Assembling global stiffness and mass matrices from frame elements
//! - Applying boundary conditions
//! - Solving the generalized eigenvalue problem for natural frequencies
//! - Computing mode shapes

use nalgebra::linalg::SymmetricEigen;
use nalgebra::DMatrix;

use crate::beam1d::{
    compute_element_mass_consistent, compute_element_mass_lumped, compute_element_stiffness,
    shear_modulus, CrossSection, DOF_PER_ELEMENT_1D, DOF_PER_NODE_1D,
};

/// Eigenvalue threshold used to discard near-zero rigid-body modes.
/// For a free-free beam, there are 3 rigid body modes (2 translation + 1 rotation).
const RIGID_BODY_THRESHOLD: f64 = 1.0;

/// Minimum frequency (Hz) to keep after filtering.
const MIN_FREQUENCY_HZ: f64 = 1.0;

/// Boundary condition types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// No constraints (free)
    Free,
    /// Fixed (clamped) - all DOFs constrained
    Fixed,
    /// Pinned - displacements constrained, rotation free
    Pinned,
    /// Roller - transverse displacement constrained only
    Roller,
}

/// Element connectivity and properties for 1D mesh.
#[derive(Debug, Clone)]
pub struct Element1D {
    /// Node indices [node1, node2]
    pub nodes: [usize; 2],
    /// Element length (m)
    pub length: f64,
    /// Cross-section properties
    pub cross_section: CrossSection,
}

/// 1D frame mesh structure.
#[derive(Debug, Clone)]
pub struct Mesh1D {
    /// Number of nodes
    pub num_nodes: usize,
    /// Node coordinates (x position along beam axis)
    pub node_coords: Vec<f64>,
    /// Element connectivity and properties
    pub elements: Vec<Element1D>,
}

impl Mesh1D {
    /// Create a uniform mesh for a straight beam.
    ///
    /// # Arguments
    /// * `length` - Total beam length (m)
    /// * `num_elements` - Number of elements
    /// * `cross_section` - Cross-section properties (same for all elements)
    pub fn uniform_beam(length: f64, num_elements: usize, cross_section: CrossSection) -> Self {
        let num_nodes = num_elements + 1;
        let le = length / num_elements as f64;

        let node_coords: Vec<f64> = (0..num_nodes).map(|i| i as f64 * le).collect();

        let elements: Vec<Element1D> = (0..num_elements)
            .map(|i| Element1D {
                nodes: [i, i + 1],
                length: le,
                cross_section,
            })
            .collect();

        Self {
            num_nodes,
            node_coords,
            elements,
        }
    }

    /// Create a mesh with varying cross-sections.
    ///
    /// # Arguments
    /// * `length` - Total beam length (m)
    /// * `cross_sections` - Cross-section for each element
    pub fn variable_section_beam(length: f64, cross_sections: &[CrossSection]) -> Self {
        let num_elements = cross_sections.len();
        let num_nodes = num_elements + 1;
        let le = length / num_elements as f64;

        let node_coords: Vec<f64> = (0..num_nodes).map(|i| i as f64 * le).collect();

        let elements: Vec<Element1D> = cross_sections
            .iter()
            .enumerate()
            .map(|(i, cs)| Element1D {
                nodes: [i, i + 1],
                length: le,
                cross_section: *cs,
            })
            .collect();

        Self {
            num_nodes,
            node_coords,
            elements,
        }
    }

    /// Create a mesh with specified element lengths and cross-sections.
    pub fn custom(element_lengths: &[f64], cross_sections: &[CrossSection]) -> Self {
        assert_eq!(
            element_lengths.len(),
            cross_sections.len(),
            "Element lengths and cross-sections must have same length"
        );

        let num_elements = element_lengths.len();
        let num_nodes = num_elements + 1;

        let mut node_coords = vec![0.0];
        let mut x = 0.0;
        for &le in element_lengths {
            x += le;
            node_coords.push(x);
        }

        let elements: Vec<Element1D> = element_lengths
            .iter()
            .zip(cross_sections.iter())
            .enumerate()
            .map(|(i, (&le, cs))| Element1D {
                nodes: [i, i + 1],
                length: le,
                cross_section: *cs,
            })
            .collect();

        Self {
            num_nodes,
            node_coords,
            elements,
        }
    }

    /// Get total number of DOFs.
    pub fn num_dofs(&self) -> usize {
        self.num_nodes * DOF_PER_NODE_1D
    }
}

/// Assemble global stiffness and mass matrices.
///
/// # Arguments
/// * `mesh` - 1D mesh structure
/// * `e` - Young's modulus (Pa)
/// * `rho` - Density (kg/m³)
/// * `nu` - Poisson's ratio
/// * `use_consistent_mass` - If true, use consistent mass; otherwise lumped mass
///
/// # Returns
/// Tuple of (K_global, M_global) matrices
pub fn assemble_global_matrices(
    mesh: &Mesh1D,
    e: f64,
    rho: f64,
    nu: f64,
    use_consistent_mass: bool,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let num_dof = mesh.num_dofs();
    let g = shear_modulus(e, nu);

    let mut k_global = DMatrix::<f64>::zeros(num_dof, num_dof);
    let mut m_global = DMatrix::<f64>::zeros(num_dof, num_dof);

    for elem in &mesh.elements {
        let le = elem.length;
        let cs = &elem.cross_section;

        let ke = compute_element_stiffness(le, e, g, cs);
        let me = if use_consistent_mass {
            compute_element_mass_consistent(le, rho, e, g, cs)
        } else {
            compute_element_mass_lumped(le, rho, cs)
        };

        // DOF mapping: element DOFs [0..5] -> global DOFs
        // Node 1: [u1, w1, θ1] at indices [3*n1, 3*n1+1, 3*n1+2]
        // Node 2: [u2, w2, θ2] at indices [3*n2, 3*n2+1, 3*n2+2]
        let n1 = elem.nodes[0];
        let n2 = elem.nodes[1];
        let dof_map = [
            3 * n1,
            3 * n1 + 1,
            3 * n1 + 2,
            3 * n2,
            3 * n2 + 1,
            3 * n2 + 2,
        ];

        for i in 0..DOF_PER_ELEMENT_1D {
            for j in 0..DOF_PER_ELEMENT_1D {
                let gi = dof_map[i];
                let gj = dof_map[j];
                k_global[(gi, gj)] += ke[(i, j)];
                m_global[(gi, gj)] += me[(i, j)];
            }
        }
    }

    (k_global, m_global)
}

/// Apply boundary conditions by modifying the matrices.
///
/// Uses the penalty method with large stiffness values.
///
/// # Arguments
/// * `k` - Global stiffness matrix (modified in place)
/// * `m` - Global mass matrix (modified in place)
/// * `bc_start` - Boundary condition at start (x=0)
/// * `bc_end` - Boundary condition at end (x=L)
/// * `num_nodes` - Total number of nodes
pub fn apply_boundary_conditions(
    k: &mut DMatrix<f64>,
    m: &mut DMatrix<f64>,
    bc_start: BoundaryCondition,
    bc_end: BoundaryCondition,
    num_nodes: usize,
) {
    let penalty = 1e20;

    // Helper to apply constraint to a DOF
    let apply_constraint = |k: &mut DMatrix<f64>, m: &mut DMatrix<f64>, dof: usize| {
        k[(dof, dof)] += penalty;
        // Zero out mass for constrained DOF to avoid numerical issues
        for j in 0..k.ncols() {
            if j != dof {
                m[(dof, j)] = 0.0;
                m[(j, dof)] = 0.0;
            }
        }
    };

    // Apply start boundary condition (node 0)
    match bc_start {
        BoundaryCondition::Free => {}
        BoundaryCondition::Fixed => {
            // Constrain u, w, θ
            apply_constraint(k, m, 0);
            apply_constraint(k, m, 1);
            apply_constraint(k, m, 2);
        }
        BoundaryCondition::Pinned => {
            // Constrain u, w (rotation free)
            apply_constraint(k, m, 0);
            apply_constraint(k, m, 1);
        }
        BoundaryCondition::Roller => {
            // Constrain w only
            apply_constraint(k, m, 1);
        }
    }

    // Apply end boundary condition (last node)
    let last_node = num_nodes - 1;
    let base_dof = 3 * last_node;

    match bc_end {
        BoundaryCondition::Free => {}
        BoundaryCondition::Fixed => {
            apply_constraint(k, m, base_dof);
            apply_constraint(k, m, base_dof + 1);
            apply_constraint(k, m, base_dof + 2);
        }
        BoundaryCondition::Pinned => {
            apply_constraint(k, m, base_dof);
            apply_constraint(k, m, base_dof + 1);
        }
        BoundaryCondition::Roller => {
            apply_constraint(k, m, base_dof + 1);
        }
    }
}

/// Get the constrained DOF indices for given boundary conditions.
pub fn get_constrained_dofs(
    bc_start: BoundaryCondition,
    bc_end: BoundaryCondition,
    num_nodes: usize,
) -> Vec<usize> {
    let mut constrained = Vec::new();

    // Start node (node 0)
    match bc_start {
        BoundaryCondition::Free => {}
        BoundaryCondition::Fixed => {
            constrained.extend_from_slice(&[0, 1, 2]);
        }
        BoundaryCondition::Pinned => {
            constrained.extend_from_slice(&[0, 1]);
        }
        BoundaryCondition::Roller => {
            constrained.push(1);
        }
    }

    // End node
    let last_node = num_nodes - 1;
    let base_dof = 3 * last_node;

    match bc_end {
        BoundaryCondition::Free => {}
        BoundaryCondition::Fixed => {
            constrained.extend_from_slice(&[base_dof, base_dof + 1, base_dof + 2]);
        }
        BoundaryCondition::Pinned => {
            constrained.extend_from_slice(&[base_dof, base_dof + 1]);
        }
        BoundaryCondition::Roller => {
            constrained.push(base_dof + 1);
        }
    }

    constrained
}

/// Reduce matrices by removing constrained DOFs.
///
/// Alternative to penalty method - physically removes constrained DOFs.
pub fn reduce_matrices(
    k: &DMatrix<f64>,
    m: &DMatrix<f64>,
    constrained_dofs: &[usize],
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>) {
    let n = k.nrows();
    let free_dofs: Vec<usize> = (0..n)
        .filter(|dof| !constrained_dofs.contains(dof))
        .collect();

    let n_free = free_dofs.len();
    let mut k_reduced = DMatrix::<f64>::zeros(n_free, n_free);
    let mut m_reduced = DMatrix::<f64>::zeros(n_free, n_free);

    for (i, &gi) in free_dofs.iter().enumerate() {
        for (j, &gj) in free_dofs.iter().enumerate() {
            k_reduced[(i, j)] = k[(gi, gj)];
            m_reduced[(i, j)] = m[(gi, gj)];
        }
    }

    (k_reduced, m_reduced, free_dofs)
}

/// Solve generalized eigenvalue problem K*φ = λ*M*φ.
///
/// Uses the standard form transformation via Cholesky decomposition.
///
/// # Arguments
/// * `k` - Global stiffness matrix
/// * `m` - Global mass matrix
/// * `num_modes` - Maximum number of modes to extract
///
/// # Returns
/// Vector of natural frequencies in Hz, sorted ascending
pub fn solve_eigenvalue_problem(k: &DMatrix<f64>, m: &DMatrix<f64>, num_modes: usize) -> Vec<f64> {
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

    // Filter out rigid body modes
    let elastic_modes: Vec<f64> = sorted_evs
        .into_iter()
        .filter(|&ev| ev > RIGID_BODY_THRESHOLD)
        .collect();

    // Convert eigenvalues to frequencies: f = sqrt(λ) / (2π)
    let frequencies: Vec<f64> = elastic_modes
        .iter()
        .take(num_modes)
        .map(|&ev| ev.sqrt() / (2.0 * std::f64::consts::PI))
        .filter(|&f| f > MIN_FREQUENCY_HZ)
        .collect();

    frequencies
}

/// Solve eigenvalue problem and return both frequencies and mode shapes.
///
/// # Arguments
/// * `k` - Global stiffness matrix
/// * `m` - Global mass matrix
/// * `num_modes` - Maximum number of modes to extract
///
/// # Returns
/// Tuple of (frequencies in Hz, mode shapes as column vectors)
pub fn solve_eigenvalue_problem_with_shapes(
    k: &DMatrix<f64>,
    m: &DMatrix<f64>,
    num_modes: usize,
) -> (Vec<f64>, DMatrix<f64>) {
    let n = k.nrows();

    // Add small regularization
    let mut m_reg = m.clone();
    for i in 0..n {
        m_reg[(i, i)] += 1e-12 * m[(i, i)].abs().max(1e-20);
    }

    let chol = match m_reg.clone().cholesky() {
        Some(c) => c,
        None => {
            for i in 0..n {
                m_reg[(i, i)] += 1e-8;
            }
            match m_reg.clone().cholesky() {
                Some(c) => c,
                None => return (Vec::new(), DMatrix::<f64>::zeros(0, 0)),
            }
        }
    };

    let l = chol.l();
    let l_inv = match l.clone().try_inverse() {
        Some(inv) => inv,
        None => return (Vec::new(), DMatrix::<f64>::zeros(0, 0)),
    };

    let k_tilde = &l_inv * k * l_inv.transpose();
    let k_tilde_sym = (&k_tilde + k_tilde.transpose()) * 0.5;

    let eig = SymmetricEigen::new(k_tilde_sym);
    let eigenvalues = eig.eigenvalues;
    let eigenvectors = eig.eigenvectors;

    // Get sorted indices
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Filter and collect
    let mut frequencies = Vec::new();
    let mut shapes = Vec::new();

    for &idx in &indices {
        let ev = eigenvalues[idx];
        if ev > RIGID_BODY_THRESHOLD {
            let freq = ev.sqrt() / (2.0 * std::f64::consts::PI);
            if freq > MIN_FREQUENCY_HZ {
                frequencies.push(freq);
                // Transform eigenvector back: φ = L^{-T} * y
                let y = eigenvectors.column(idx);
                let phi = l_inv.transpose() * y;
                shapes.push(phi);

                if frequencies.len() >= num_modes {
                    break;
                }
            }
        }
    }

    // Assemble mode shapes into matrix
    let num_found = frequencies.len();
    let mut shape_matrix = DMatrix::<f64>::zeros(n, num_found);
    for (i, shape) in shapes.iter().enumerate() {
        shape_matrix.set_column(i, shape);
    }

    (frequencies, shape_matrix)
}

/// Compute modal frequencies for a uniform beam with given boundary conditions.
///
/// # Arguments
/// * `length` - Beam length (m)
/// * `width` - Cross-section width (m)
/// * `height` - Cross-section height (m)
/// * `e` - Young's modulus (Pa)
/// * `nu` - Poisson's ratio
/// * `rho` - Density (kg/m³)
/// * `num_elements` - Number of elements for discretization
/// * `num_modes` - Number of modes to compute
/// * `bc_start` - Boundary condition at start
/// * `bc_end` - Boundary condition at end
///
/// # Returns
/// Vector of frequencies in Hz
pub fn compute_frequencies_uniform_beam(
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    nu: f64,
    rho: f64,
    num_elements: usize,
    num_modes: usize,
    bc_start: BoundaryCondition,
    bc_end: BoundaryCondition,
) -> Vec<f64> {
    let cs = CrossSection::rectangular(width, height);
    let mesh = Mesh1D::uniform_beam(length, num_elements, cs);

    let (mut k, mut m) = assemble_global_matrices(&mesh, e, rho, nu, true);
    apply_boundary_conditions(&mut k, &mut m, bc_start, bc_end, mesh.num_nodes);

    solve_eigenvalue_problem(&k, &m, num_modes)
}

/// Compute modal frequencies for a free-free beam (no constraints).
///
/// Convenience function for percussion instrument analysis.
pub fn compute_frequencies_free_free(
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    nu: f64,
    rho: f64,
    num_elements: usize,
    num_modes: usize,
) -> Vec<f64> {
    compute_frequencies_uniform_beam(
        length,
        width,
        height,
        e,
        nu,
        rho,
        num_elements,
        num_modes,
        BoundaryCondition::Free,
        BoundaryCondition::Free,
    )
}

/// Compute modal frequencies for a cantilever beam (fixed-free).
pub fn compute_frequencies_cantilever(
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    nu: f64,
    rho: f64,
    num_elements: usize,
    num_modes: usize,
) -> Vec<f64> {
    compute_frequencies_uniform_beam(
        length,
        width,
        height,
        e,
        nu,
        rho,
        num_elements,
        num_modes,
        BoundaryCondition::Fixed,
        BoundaryCondition::Free,
    )
}

/// Compute modal frequencies for a simply-supported beam (pinned-pinned).
pub fn compute_frequencies_simply_supported(
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    nu: f64,
    rho: f64,
    num_elements: usize,
    num_modes: usize,
) -> Vec<f64> {
    compute_frequencies_uniform_beam(
        length,
        width,
        height,
        e,
        nu,
        rho,
        num_elements,
        num_modes,
        BoundaryCondition::Pinned,
        BoundaryCondition::Pinned,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn get_test_beam_params() -> (f64, f64, f64, f64, f64, f64) {
        let length = 1.0; // 1m beam
        let width = 0.02; // 20mm width
        let height = 0.03; // 30mm height
        let e = 200e9; // Steel
        let nu = 0.3;
        let rho = 7850.0;
        (length, width, height, e, nu, rho)
    }

    #[test]
    fn uniform_mesh_creation() {
        let cs = CrossSection::rectangular(0.02, 0.03);
        let mesh = Mesh1D::uniform_beam(1.0, 10, cs);

        assert_eq!(mesh.num_nodes, 11);
        assert_eq!(mesh.elements.len(), 10);
        assert_eq!(mesh.num_dofs(), 33);

        // Check node coordinates
        assert!((mesh.node_coords[0] - 0.0).abs() < TOL);
        assert!((mesh.node_coords[10] - 1.0).abs() < TOL);
    }

    #[test]
    fn global_matrices_have_correct_size() {
        let (length, width, height, e, nu, rho) = get_test_beam_params();
        let cs = CrossSection::rectangular(width, height);
        let mesh = Mesh1D::uniform_beam(length, 10, cs);

        let (k, m) = assemble_global_matrices(&mesh, e, rho, nu, true);

        let expected_dof = 11 * 3; // 11 nodes × 3 DOF/node = 33
        assert_eq!(k.nrows(), expected_dof);
        assert_eq!(k.ncols(), expected_dof);
        assert_eq!(m.nrows(), expected_dof);
        assert_eq!(m.ncols(), expected_dof);
    }

    #[test]
    fn global_matrices_are_symmetric() {
        let (length, width, height, e, nu, rho) = get_test_beam_params();
        let cs = CrossSection::rectangular(width, height);
        let mesh = Mesh1D::uniform_beam(length, 10, cs);

        let (k, m) = assemble_global_matrices(&mesh, e, rho, nu, true);

        let n = k.nrows();
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[(i, j)] - k[(j, i)]).abs() < TOL,
                    "K not symmetric at ({}, {})",
                    i,
                    j
                );
                assert!(
                    (m[(i, j)] - m[(j, i)]).abs() < TOL,
                    "M not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn free_free_beam_frequencies_positive() {
        let (length, width, height, e, nu, rho) = get_test_beam_params();

        let freqs = compute_frequencies_free_free(length, width, height, e, nu, rho, 50, 6);

        assert!(!freqs.is_empty(), "Should compute frequencies");
        for f in &freqs {
            assert!(f.is_finite(), "Frequency should be finite");
            assert!(*f > 0.0, "Frequency should be positive: {}", f);
        }
    }

    #[test]
    fn cantilever_frequencies_positive() {
        let (length, width, height, e, nu, rho) = get_test_beam_params();

        let freqs = compute_frequencies_cantilever(length, width, height, e, nu, rho, 50, 6);

        assert!(!freqs.is_empty(), "Should compute frequencies");
        for f in &freqs {
            assert!(f.is_finite(), "Frequency should be finite");
            assert!(*f > 0.0, "Frequency should be positive: {}", f);
        }
    }

    #[test]
    fn simply_supported_frequencies_positive() {
        let (length, width, height, e, nu, rho) = get_test_beam_params();

        let freqs = compute_frequencies_simply_supported(length, width, height, e, nu, rho, 50, 6);

        assert!(!freqs.is_empty(), "Should compute frequencies");
        for f in &freqs {
            assert!(f.is_finite(), "Frequency should be finite");
            assert!(*f > 0.0, "Frequency should be positive: {}", f);
        }
    }

    #[test]
    fn frequencies_are_sorted_ascending() {
        let (length, width, height, e, nu, rho) = get_test_beam_params();

        let freqs = compute_frequencies_free_free(length, width, height, e, nu, rho, 100, 10);

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
    fn cantilever_euler_bernoulli_comparison() {
        // Euler-Bernoulli cantilever beam analytical frequencies:
        // f_n = (beta_n^2 / (2*pi)) * sqrt(E*I / (rho*A*L^4))
        // where beta_1 = 1.875, beta_2 = 4.694, beta_3 = 7.855

        let length = 0.5;
        let width = 0.01;
        let height = 0.02;
        let e = 200e9;
        let nu = 0.3;
        let rho = 7850.0;

        let freqs = compute_frequencies_cantilever(length, width, height, e, nu, rho, 100, 3);

        // Analytical
        let i = width * height.powi(3) / 12.0;
        let a = width * height;
        let coeff = (e * i / (rho * a * length.powi(4))).sqrt() / (2.0 * std::f64::consts::PI);

        let betas = [1.875, 4.694, 7.855];
        let analytical: Vec<f64> = betas.iter().map(|b| b * b * coeff).collect();

        assert!(freqs.len() >= 3, "Should compute at least 3 modes");

        // Timoshenko gives slightly lower frequencies due to shear
        // Check within 15% (thick beam effects)
        for i in 0..3 {
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

    #[test]
    fn simply_supported_euler_bernoulli_comparison() {
        // Euler-Bernoulli simply-supported beam:
        // f_n = (n^2 * pi^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A))

        let length = 0.5;
        let width = 0.01;
        let height = 0.02;
        let e = 200e9;
        let nu = 0.3;
        let rho = 7850.0;

        let freqs = compute_frequencies_simply_supported(length, width, height, e, nu, rho, 100, 3);

        let i = width * height.powi(3) / 12.0;
        let a = width * height;
        let pi = std::f64::consts::PI;
        let coeff = pi / (2.0 * length * length) * (e * i / (rho * a)).sqrt();

        let analytical: Vec<f64> = (1..=3).map(|n| (n * n) as f64 * coeff).collect();

        assert!(freqs.len() >= 3, "Should compute at least 3 modes");

        for i in 0..3 {
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

    #[test]
    fn mode_shapes_have_correct_dimensions() {
        let (length, width, height, e, nu, rho) = get_test_beam_params();
        let cs = CrossSection::rectangular(width, height);
        let mesh = Mesh1D::uniform_beam(length, 20, cs);

        let (mut k, mut m) = assemble_global_matrices(&mesh, e, rho, nu, true);
        apply_boundary_conditions(
            &mut k,
            &mut m,
            BoundaryCondition::Fixed,
            BoundaryCondition::Free,
            mesh.num_nodes,
        );

        let (freqs, shapes) = solve_eigenvalue_problem_with_shapes(&k, &m, 5);

        assert!(!freqs.is_empty(), "Should compute frequencies");
        assert_eq!(shapes.nrows(), mesh.num_dofs());
        assert_eq!(shapes.ncols(), freqs.len());
    }

    #[test]
    fn reduced_matrices_exclude_constrained_dofs() {
        let (length, width, height, e, nu, rho) = get_test_beam_params();
        let cs = CrossSection::rectangular(width, height);
        let mesh = Mesh1D::uniform_beam(length, 5, cs);

        let (k, m) = assemble_global_matrices(&mesh, e, rho, nu, true);

        let constrained = get_constrained_dofs(BoundaryCondition::Fixed, BoundaryCondition::Free, 6);
        assert_eq!(constrained, vec![0, 1, 2]); // First node fixed

        let (k_red, m_red, free_dofs) = reduce_matrices(&k, &m, &constrained);

        assert_eq!(k_red.nrows(), k.nrows() - 3);
        assert_eq!(m_red.nrows(), m.nrows() - 3);
        assert_eq!(free_dofs.len(), k.nrows() - 3);
    }

    #[test]
    fn variable_cross_section_mesh() {
        let sections: Vec<CrossSection> = (0..5)
            .map(|i| CrossSection::rectangular(0.02, 0.02 + 0.002 * i as f64))
            .collect();

        let mesh = Mesh1D::variable_section_beam(1.0, &sections);

        assert_eq!(mesh.num_nodes, 6);
        assert_eq!(mesh.elements.len(), 5);

        // Check varying areas
        for (i, elem) in mesh.elements.iter().enumerate() {
            let expected_h = 0.02 + 0.002 * i as f64;
            let expected_area = 0.02 * expected_h;
            assert!(
                (elem.cross_section.area - expected_area).abs() < TOL,
                "Element {} has incorrect area",
                i
            );
        }
    }
}
