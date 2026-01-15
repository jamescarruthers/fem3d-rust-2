//! 8-node hexahedral (Hex8) finite element functions.
//!
//! This module provides shape functions, material matrices, and element matrix
//! computation for 3D hexahedral elements used in structural analysis.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};

use crate::types::{Matrix3x24, Matrix3x8, Matrix6, Matrix6x24, NodeCoords, GAUSS_G, MIN_DET_J};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GAUSS_G;
    use nalgebra::RowVector3;

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
}
