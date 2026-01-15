//! 2D Timoshenko Beam Element for modal analysis.
//!
//! Implements Timoshenko beam element stiffness and mass matrices
//! for computing natural frequencies of undercut bars. The Timoshenko
//! beam model accounts for shear deformation, which is important for
//! thick bars used in percussion instruments.
//!
//! DOFs per node: 2 (transverse displacement w, rotation θ)
//! Element DOFs: [w1, θ1, w2, θ2]

use nalgebra::SMatrix;

/// Shear correction factor for rectangular cross-section (5/6).
pub const KAPPA: f64 = 5.0 / 6.0;

/// DOFs per node for 2D beam (transverse displacement + rotation).
pub const DOF_PER_NODE_2D: usize = 2;

/// Compute Timoshenko beam element stiffness matrix (4×4).
///
/// DOFs: [w1, θ1, w2, θ2] - transverse displacement and rotation.
///
/// # Arguments
/// * `le` - Element length (m)
/// * `e` - Young's modulus (Pa)
/// * `i` - Second moment of area (m⁴)
/// * `g` - Shear modulus (Pa)
/// * `a` - Cross-sectional area (m²)
///
/// # Returns
/// 4×4 element stiffness matrix
pub fn compute_element_stiffness(le: f64, e: f64, i: f64, g: f64, a: f64) -> SMatrix<f64, 4, 4> {
    let phi = 12.0 * e * i / (KAPPA * g * a * le * le);
    let denom = (1.0 + phi) * le * le * le;

    let k11 = 12.0 * e * i / denom;
    let k12 = 6.0 * e * i * le / denom;
    let k22 = (4.0 + phi) * e * i * le * le / denom;
    let k23 = (2.0 - phi) * e * i * le * le / denom;

    SMatrix::<f64, 4, 4>::from_row_slice(&[
        k11, k12, -k11, k12, // row 0
        k12, k22, -k12, k23, // row 1
        -k11, -k12, k11, -k12, // row 2
        k12, k23, -k12, k22, // row 3
    ])
}

/// Compute Timoshenko beam element mass matrix (4×4).
///
/// Using consistent mass matrix formulation with rotary inertia.
///
/// # Arguments
/// * `le` - Element length (m)
/// * `rho` - Density (kg/m³)
/// * `a` - Cross-sectional area (m²)
/// * `i` - Second moment of area (m⁴)
/// * `e` - Young's modulus (Pa)
/// * `g` - Shear modulus (Pa)
///
/// # Returns
/// 4×4 element mass matrix
pub fn compute_element_mass(
    le: f64,
    rho: f64,
    a: f64,
    i: f64,
    e: f64,
    g: f64,
) -> SMatrix<f64, 4, 4> {
    let phi = 12.0 * e * i / (KAPPA * g * a * le * le);
    let denom = (1.0 + phi) * (1.0 + phi);

    // Translational mass terms
    let m = rho * a * le;

    // Rotary inertia terms
    let r2 = i / a; // radius of gyration squared

    let c1 = (13.0 / 35.0 + 7.0 * phi / 10.0 + phi * phi / 3.0) / denom;
    let c2 = (9.0 / 70.0 + 3.0 * phi / 10.0 + phi * phi / 6.0) / denom;
    let c3 = (11.0 / 210.0 + 11.0 * phi / 120.0 + phi * phi / 24.0) * le / denom;
    let c4 = (13.0 / 420.0 + 3.0 * phi / 40.0 + phi * phi / 24.0) * le / denom;
    let c5 = (1.0 / 105.0 + phi / 60.0 + phi * phi / 120.0) * le * le / denom;
    let c6 = (1.0 / 140.0 + phi / 60.0 + phi * phi / 120.0) * le * le / denom;

    // Add rotary inertia contribution
    let r_scale = r2 / (le * le);
    let r1 = (6.0 / 5.0) / denom * r_scale;
    let r2_term = (2.0 / 15.0 + phi / 6.0 + phi * phi / 3.0) * le * le / denom * r_scale;
    let r3 = (1.0 / 10.0 - phi / 2.0) * le / denom * r_scale;
    let r4 = (-1.0 / 30.0 - phi / 6.0 + phi * phi / 6.0) * le * le / denom * r_scale;

    SMatrix::<f64, 4, 4>::from_row_slice(&[
        m * (c1 + r1),
        m * (c3 + r3),
        m * (c2 - r1),
        m * (-c4 + r3),
        m * (c3 + r3),
        m * (c5 + r2_term),
        m * (c4 - r3),
        m * (-c6 + r4),
        m * (c2 - r1),
        m * (c4 - r3),
        m * (c1 + r1),
        m * (-c3 - r3),
        m * (-c4 + r3),
        m * (-c6 + r4),
        m * (-c3 - r3),
        m * (c5 + r2_term),
    ])
}

/// Compute shear modulus from Young's modulus and Poisson's ratio.
#[inline]
pub fn shear_modulus(e: f64, nu: f64) -> f64 {
    e / (2.0 * (1.0 + nu))
}

/// Compute second moment of area for a rectangular cross-section.
///
/// I = b * h³ / 12
#[inline]
pub fn second_moment_of_area(b: f64, h: f64) -> f64 {
    b * h * h * h / 12.0
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn stiffness_matrix_is_symmetric() {
        let le = 0.025; // 25mm element
        let e = 12e9; // 12 GPa (wood)
        let b = 0.03; // 30mm width
        let h = 0.024; // 24mm height
        let nu = 0.35;

        let a = b * h;
        let i = second_moment_of_area(b, h);
        let g = shear_modulus(e, nu);

        let ke = compute_element_stiffness(le, e, i, g, a);

        for row in 0..4 {
            for col in 0..4 {
                assert!(
                    (ke[(row, col)] - ke[(col, row)]).abs() < TOL,
                    "Stiffness matrix not symmetric at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn mass_matrix_is_symmetric() {
        let le = 0.025;
        let e = 12e9;
        let rho = 640.0; // kg/m³
        let b = 0.03;
        let h = 0.024;
        let nu = 0.35;

        let a = b * h;
        let i = second_moment_of_area(b, h);
        let g = shear_modulus(e, nu);

        let me = compute_element_mass(le, rho, a, i, e, g);

        for row in 0..4 {
            for col in 0..4 {
                assert!(
                    (me[(row, col)] - me[(col, row)]).abs() < TOL,
                    "Mass matrix not symmetric at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn stiffness_diagonal_is_positive() {
        let le = 0.025;
        let e = 70e9; // aluminum
        let b = 0.03;
        let h = 0.02;
        let nu = 0.33;

        let a = b * h;
        let i = second_moment_of_area(b, h);
        let g = shear_modulus(e, nu);

        let ke = compute_element_stiffness(le, e, i, g, a);

        for diag in 0..4 {
            assert!(
                ke[(diag, diag)] > 0.0,
                "Stiffness diagonal {} is not positive: {}",
                diag,
                ke[(diag, diag)]
            );
        }
    }

    #[test]
    fn mass_diagonal_is_positive() {
        let le = 0.025;
        let e = 70e9;
        let rho = 2700.0;
        let b = 0.03;
        let h = 0.02;
        let nu = 0.33;

        let a = b * h;
        let i = second_moment_of_area(b, h);
        let g = shear_modulus(e, nu);

        let me = compute_element_mass(le, rho, a, i, e, g);

        for diag in 0..4 {
            assert!(
                me[(diag, diag)] > 0.0,
                "Mass diagonal {} is not positive: {}",
                diag,
                me[(diag, diag)]
            );
        }
    }

    #[test]
    fn shear_modulus_calculation() {
        let e = 200e9;
        let nu = 0.3;
        let g = shear_modulus(e, nu);

        // G = E / (2 * (1 + nu)) = 200e9 / 2.6 ≈ 76.92 GPa
        let expected = 200e9 / 2.6;
        assert!((g - expected).abs() < 1e3);
    }

    #[test]
    fn second_moment_calculation() {
        let b = 0.03; // 30mm
        let h = 0.024; // 24mm

        let i = second_moment_of_area(b, h);

        // I = b * h³ / 12 = 0.03 * 0.024³ / 12
        let expected = 0.03 * 0.024_f64.powi(3) / 12.0;
        assert!((i - expected).abs() < 1e-15);
    }
}
