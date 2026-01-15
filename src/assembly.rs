//! Global matrix assembly functions for sparse and dense formats.
//!
//! This module provides functions for assembling element matrices into
//! global stiffness and mass matrices.

use nalgebra::{DMatrix, SMatrix};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

#[cfg(feature = "sprs-backend")]
use sprs::{CsMat, TriMat};

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

/// Assemble global matrix using sprs triplet format.
#[cfg(feature = "sprs-backend")]
pub fn assemble_global_sprs(
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

#[cfg(test)]
mod tests {
    use super::*;

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
