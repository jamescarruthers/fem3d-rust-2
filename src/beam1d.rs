//! 1D Timoshenko Frame Element for structural modal analysis.
//!
//! Implements a general-purpose Timoshenko beam/frame element with 3 DOF per node:
//! - u: axial displacement
//! - w: transverse displacement
//! - θ: rotation (bending)
//!
//! This element captures both axial and bending vibration modes, making it suitable
//! for general frame structures. The Timoshenko formulation accounts for shear
//! deformation, which is important for thick/short beams.
//!
//! ## Element DOF ordering
//! ```text
//! Node 1: [u1, w1, θ1]
//! Node 2: [u2, w2, θ2]
//! Element DOFs: [u1, w1, θ1, u2, w2, θ2]
//! ```
//!
//! ## Theory
//! The element stiffness matrix combines:
//! - Axial stiffness (EA/L terms)
//! - Bending stiffness with shear correction (Timoshenko beam theory)
//!
//! The element mass matrix includes:
//! - Translational inertia (axial and transverse)
//! - Rotary inertia

use nalgebra::SMatrix;

/// Shear correction factor for rectangular cross-section (5/6).
/// For circular cross-sections, use κ = 6/7 ≈ 0.857.
/// For I-beams, κ depends on geometry (typically 0.4-0.6 for the web).
pub const KAPPA: f64 = 5.0 / 6.0;

/// DOFs per node for 1D frame element (axial + transverse + rotation).
pub const DOF_PER_NODE_1D: usize = 3;

/// DOFs per element (2 nodes × 3 DOF/node).
pub const DOF_PER_ELEMENT_1D: usize = 6;

/// Compute shear modulus from Young's modulus and Poisson's ratio.
///
/// G = E / (2 * (1 + ν))
#[inline]
pub fn shear_modulus(e: f64, nu: f64) -> f64 {
    e / (2.0 * (1.0 + nu))
}

/// Compute second moment of area for a rectangular cross-section.
///
/// I = b * h³ / 12
///
/// For bending about the neutral axis perpendicular to height h.
#[inline]
pub fn second_moment_of_area_rect(b: f64, h: f64) -> f64 {
    b * h * h * h / 12.0
}

/// Compute second moment of area for a circular cross-section.
///
/// I = π * r⁴ / 4
#[inline]
pub fn second_moment_of_area_circle(r: f64) -> f64 {
    std::f64::consts::PI * r * r * r * r / 4.0
}

/// Cross-section properties for a 1D element.
#[derive(Debug, Clone, Copy)]
pub struct CrossSection {
    /// Cross-sectional area (m²)
    pub area: f64,
    /// Second moment of area about bending axis (m⁴)
    pub i_bend: f64,
    /// Shear correction factor (default: 5/6 for rectangular)
    pub kappa: f64,
}

impl CrossSection {
    /// Create a rectangular cross-section.
    ///
    /// # Arguments
    /// * `b` - Width (m)
    /// * `h` - Height (m)
    pub fn rectangular(b: f64, h: f64) -> Self {
        Self {
            area: b * h,
            i_bend: second_moment_of_area_rect(b, h),
            kappa: 5.0 / 6.0, // Rectangular shear correction
        }
    }

    /// Create a circular cross-section.
    ///
    /// # Arguments
    /// * `r` - Radius (m)
    pub fn circular(r: f64) -> Self {
        Self {
            area: std::f64::consts::PI * r * r,
            i_bend: second_moment_of_area_circle(r),
            kappa: 6.0 / 7.0, // Circular shear correction
        }
    }

    /// Create a custom cross-section with specified properties.
    pub fn custom(area: f64, i_bend: f64, kappa: f64) -> Self {
        Self { area, i_bend, kappa }
    }
}

/// Compute Timoshenko frame element stiffness matrix (6×6).
///
/// Combines axial stiffness with Timoshenko bending stiffness.
///
/// DOF ordering: [u1, w1, θ1, u2, w2, θ2]
///
/// # Arguments
/// * `le` - Element length (m)
/// * `e` - Young's modulus (Pa)
/// * `g` - Shear modulus (Pa)
/// * `cs` - Cross-section properties
///
/// # Returns
/// 6×6 element stiffness matrix
pub fn compute_element_stiffness(
    le: f64,
    e: f64,
    g: f64,
    cs: &CrossSection,
) -> SMatrix<f64, 6, 6> {
    let a = cs.area;
    let i = cs.i_bend;
    let kappa = cs.kappa;

    // Axial stiffness
    let k_axial = e * a / le;

    // Timoshenko bending stiffness with shear correction
    let phi = 12.0 * e * i / (kappa * g * a * le * le);
    let denom = (1.0 + phi) * le * le * le;

    let k11 = 12.0 * e * i / denom;
    let k12 = 6.0 * e * i * le / denom;
    let k22 = (4.0 + phi) * e * i * le * le / denom;
    let k23 = (2.0 - phi) * e * i * le * le / denom;

    // Assemble 6×6 matrix
    // DOF order: [u1, w1, θ1, u2, w2, θ2]
    #[rustfmt::skip]
    let k = SMatrix::<f64, 6, 6>::from_row_slice(&[
        // u1       w1       θ1       u2       w2       θ2
        k_axial,   0.0,     0.0,    -k_axial,  0.0,     0.0,    // u1
        0.0,       k11,     k12,     0.0,     -k11,     k12,    // w1
        0.0,       k12,     k22,     0.0,     -k12,     k23,    // θ1
       -k_axial,   0.0,     0.0,     k_axial,  0.0,     0.0,    // u2
        0.0,      -k11,    -k12,     0.0,      k11,    -k12,    // w2
        0.0,       k12,     k23,     0.0,     -k12,     k22,    // θ2
    ]);

    k
}

/// Compute Timoshenko frame element consistent mass matrix (6×6).
///
/// Includes translational mass (axial and transverse) and rotary inertia.
///
/// DOF ordering: [u1, w1, θ1, u2, w2, θ2]
///
/// # Arguments
/// * `le` - Element length (m)
/// * `rho` - Density (kg/m³)
/// * `e` - Young's modulus (Pa)
/// * `g` - Shear modulus (Pa)
/// * `cs` - Cross-section properties
///
/// # Returns
/// 6×6 element mass matrix
pub fn compute_element_mass(
    le: f64,
    rho: f64,
    e: f64,
    g: f64,
    cs: &CrossSection,
) -> SMatrix<f64, 6, 6> {
    let a = cs.area;
    let i = cs.i_bend;
    let kappa = cs.kappa;

    // Shear parameter
    let phi = 12.0 * e * i / (kappa * g * a * le * le);
    let denom = (1.0 + phi) * (1.0 + phi);

    // Total element mass
    let m = rho * a * le;

    // Axial mass coefficients (consistent mass)
    let ma_11 = m / 3.0;
    let ma_12 = m / 6.0;

    // Bending mass coefficients (Timoshenko consistent mass with rotary inertia)
    let r2 = i / a; // radius of gyration squared

    // Translational mass terms
    let c1 = (13.0 / 35.0 + 7.0 * phi / 10.0 + phi * phi / 3.0) / denom;
    let c2 = (9.0 / 70.0 + 3.0 * phi / 10.0 + phi * phi / 6.0) / denom;
    let c3 = (11.0 / 210.0 + 11.0 * phi / 120.0 + phi * phi / 24.0) * le / denom;
    let c4 = (13.0 / 420.0 + 3.0 * phi / 40.0 + phi * phi / 24.0) * le / denom;
    let c5 = (1.0 / 105.0 + phi / 60.0 + phi * phi / 120.0) * le * le / denom;
    let c6 = (1.0 / 140.0 + phi / 60.0 + phi * phi / 120.0) * le * le / denom;

    // Rotary inertia contribution
    let r_scale = r2 / (le * le);
    let r1 = (6.0 / 5.0) / denom * r_scale;
    let r2_term = (2.0 / 15.0 + phi / 6.0 + phi * phi / 3.0) * le * le / denom * r_scale;
    let r3 = (1.0 / 10.0 - phi / 2.0) * le / denom * r_scale;
    let r4 = (-1.0 / 30.0 - phi / 6.0 + phi * phi / 6.0) * le * le / denom * r_scale;

    // Bending mass matrix entries
    let m_ww_11 = m * (c1 + r1);
    let m_wt_11 = m * (c3 + r3);
    let m_ww_12 = m * (c2 - r1);
    let m_wt_12 = m * (-c4 + r3);
    let m_tt_11 = m * (c5 + r2_term);
    let m_tt_12 = m * (-c6 + r4);

    // Assemble 6×6 matrix
    // DOF order: [u1, w1, θ1, u2, w2, θ2]
    #[rustfmt::skip]
    let mass = SMatrix::<f64, 6, 6>::from_row_slice(&[
        // u1       w1         θ1         u2       w2         θ2
        ma_11,     0.0,       0.0,       ma_12,    0.0,       0.0,       // u1
        0.0,       m_ww_11,   m_wt_11,   0.0,      m_ww_12,   m_wt_12,   // w1
        0.0,       m_wt_11,   m_tt_11,   0.0,      m_wt_12-m_wt_11+m*(c4-r3), m_tt_12, // θ1
        ma_12,     0.0,       0.0,       ma_11,    0.0,       0.0,       // u2
        0.0,       m_ww_12,   m_wt_12-m_wt_11+m*(c4-r3), 0.0, m_ww_11, -m_wt_11, // w2
        0.0,       m_wt_12,   m_tt_12,   0.0,     -m_wt_11,   m_tt_11,   // θ2
    ]);

    mass
}

/// Compute Timoshenko frame element consistent mass matrix (6×6) - simplified version.
///
/// Uses standard Timoshenko consistent mass matrix formulation.
///
/// DOF ordering: [u1, w1, θ1, u2, w2, θ2]
///
/// # Arguments
/// * `le` - Element length (m)
/// * `rho` - Density (kg/m³)
/// * `e` - Young's modulus (Pa)
/// * `g` - Shear modulus (Pa)
/// * `cs` - Cross-section properties
///
/// # Returns
/// 6×6 element mass matrix
pub fn compute_element_mass_consistent(
    le: f64,
    rho: f64,
    e: f64,
    g: f64,
    cs: &CrossSection,
) -> SMatrix<f64, 6, 6> {
    let a = cs.area;
    let i = cs.i_bend;
    let kappa = cs.kappa;

    // Shear parameter
    let phi = 12.0 * e * i / (kappa * g * a * le * le);
    let phi2 = phi * phi;
    let one_plus_phi = 1.0 + phi;
    let one_plus_phi2 = one_plus_phi * one_plus_phi;

    // Element mass
    let m = rho * a * le;

    // Radius of gyration squared (used implicitly in rotary inertia terms)
    let _r2 = i / a;

    // Axial mass coefficients
    let ma_11 = m / 3.0;
    let ma_12 = m / 6.0;

    // Bending mass coefficients (from Timoshenko beam theory)
    // Reference: Przemieniecki, "Theory of Matrix Structural Analysis"
    let coeff = m / (one_plus_phi2 * 420.0);

    // M_ww diagonal
    let m_11 = coeff * (156.0 + 294.0 * phi + 140.0 * phi2);
    // M_ww off-diagonal
    let m_13 = coeff * (54.0 + 126.0 * phi + 70.0 * phi2);
    // M_wθ terms
    let m_12 = coeff * le * (22.0 + 38.5 * phi + 17.5 * phi2);
    let m_14 = -coeff * le * (13.0 + 31.5 * phi + 17.5 * phi2);
    // M_θθ diagonal
    let m_22 = coeff * le * le * (4.0 + 7.0 * phi + 3.5 * phi2);
    // M_θθ off-diagonal
    let m_24 = -coeff * le * le * (3.0 + 7.0 * phi + 3.5 * phi2);

    // Add rotary inertia correction
    let r_coeff = rho * i / (one_plus_phi2 * le * 30.0);
    let r_11 = r_coeff * 36.0;
    let r_12 = r_coeff * le * (3.0 - 15.0 * phi);
    let r_22 = r_coeff * le * le * (4.0 + 5.0 * phi + 10.0 * phi2);
    let r_24 = r_coeff * le * le * (-1.0 - 5.0 * phi + 5.0 * phi2);

    // Combined terms
    let m_ww_11 = m_11 + r_11;
    let m_wt_11 = m_12 + r_12;
    let m_ww_12 = m_13 - r_11;
    let m_wt_12 = m_14 + r_12;
    let m_tt_11 = m_22 + r_22;
    let m_tt_12 = m_24 + r_24;

    // Assemble 6×6 matrix
    #[rustfmt::skip]
    let mass = SMatrix::<f64, 6, 6>::from_row_slice(&[
        // u1       w1         θ1         u2       w2         θ2
        ma_11,     0.0,       0.0,       ma_12,    0.0,       0.0,       // u1
        0.0,       m_ww_11,   m_wt_11,   0.0,      m_ww_12,   m_wt_12,   // w1
        0.0,       m_wt_11,   m_tt_11,   0.0,     -m_wt_12,   m_tt_12,   // θ1
        ma_12,     0.0,       0.0,       ma_11,    0.0,       0.0,       // u2
        0.0,       m_ww_12,  -m_wt_12,   0.0,      m_ww_11,  -m_wt_11,   // w2
        0.0,       m_wt_12,   m_tt_12,   0.0,     -m_wt_11,   m_tt_11,   // θ2
    ]);

    mass
}

/// Compute a lumped mass matrix for the element (6×6).
///
/// Lumped mass places all mass at the nodes with no coupling.
/// Simpler but less accurate than consistent mass.
///
/// # Arguments
/// * `le` - Element length (m)
/// * `rho` - Density (kg/m³)
/// * `cs` - Cross-section properties
///
/// # Returns
/// 6×6 diagonal lumped mass matrix
pub fn compute_element_mass_lumped(le: f64, rho: f64, cs: &CrossSection) -> SMatrix<f64, 6, 6> {
    let a = cs.area;
    let i = cs.i_bend;

    // Total element mass distributed equally to nodes
    let m_node = rho * a * le / 2.0;

    // Rotary inertia at each node
    let j_node = rho * i * le / 2.0;

    SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::<f64, 6>::from_row_slice(&[
        m_node, m_node, j_node, m_node, m_node, j_node,
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn get_test_params() -> (f64, f64, f64, f64, CrossSection) {
        let le = 0.1; // 100mm element
        let e = 200e9; // Steel, 200 GPa
        let nu = 0.3;
        let rho = 7850.0; // Steel density
        let g = shear_modulus(e, nu);
        let cs = CrossSection::rectangular(0.02, 0.03); // 20mm × 30mm
        (le, e, g, rho, cs)
    }

    #[test]
    fn stiffness_matrix_is_symmetric() {
        let (le, e, g, _, cs) = get_test_params();
        let ke = compute_element_stiffness(le, e, g, &cs);

        for row in 0..6 {
            for col in 0..6 {
                assert!(
                    (ke[(row, col)] - ke[(col, row)]).abs() < TOL,
                    "Stiffness matrix not symmetric at ({}, {}): {} vs {}",
                    row,
                    col,
                    ke[(row, col)],
                    ke[(col, row)]
                );
            }
        }
    }

    #[test]
    fn mass_matrix_is_symmetric() {
        let (le, e, g, rho, cs) = get_test_params();
        let me = compute_element_mass_consistent(le, rho, e, g, &cs);

        for row in 0..6 {
            for col in 0..6 {
                assert!(
                    (me[(row, col)] - me[(col, row)]).abs() < TOL,
                    "Mass matrix not symmetric at ({}, {}): {} vs {}",
                    row,
                    col,
                    me[(row, col)],
                    me[(col, row)]
                );
            }
        }
    }

    #[test]
    fn stiffness_diagonal_is_positive() {
        let (le, e, g, _, cs) = get_test_params();
        let ke = compute_element_stiffness(le, e, g, &cs);

        for diag in 0..6 {
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
        let (le, e, g, rho, cs) = get_test_params();
        let me = compute_element_mass_consistent(le, rho, e, g, &cs);

        for diag in 0..6 {
            assert!(
                me[(diag, diag)] > 0.0,
                "Mass diagonal {} is not positive: {}",
                diag,
                me[(diag, diag)]
            );
        }
    }

    #[test]
    fn lumped_mass_is_diagonal() {
        let (le, _, _, rho, cs) = get_test_params();
        let me = compute_element_mass_lumped(le, rho, &cs);

        for row in 0..6 {
            for col in 0..6 {
                if row != col {
                    assert!(
                        me[(row, col)].abs() < TOL,
                        "Lumped mass should be diagonal, but ({}, {}) = {}",
                        row,
                        col,
                        me[(row, col)]
                    );
                }
            }
        }
    }

    #[test]
    fn axial_stiffness_is_correct() {
        let (le, e, g, _, cs) = get_test_params();
        let ke = compute_element_stiffness(le, e, g, &cs);

        let expected_axial = e * cs.area / le;
        assert!(
            (ke[(0, 0)] - expected_axial).abs() / expected_axial < 1e-10,
            "Axial stiffness incorrect: {} vs {}",
            ke[(0, 0)],
            expected_axial
        );
    }

    #[test]
    fn shear_modulus_calculation() {
        let e = 200e9;
        let nu = 0.3;
        let g = shear_modulus(e, nu);

        let expected = 200e9 / 2.6;
        assert!((g - expected).abs() < 1e3);
    }

    #[test]
    fn rectangular_cross_section() {
        let b = 0.02;
        let h = 0.03;
        let cs = CrossSection::rectangular(b, h);

        assert!((cs.area - b * h).abs() < 1e-15);
        assert!((cs.i_bend - b * h * h * h / 12.0).abs() < 1e-20);
        assert!((cs.kappa - 5.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn circular_cross_section() {
        let r = 0.01;
        let cs = CrossSection::circular(r);

        let expected_area = std::f64::consts::PI * r * r;
        let expected_i = std::f64::consts::PI * r.powi(4) / 4.0;

        assert!((cs.area - expected_area).abs() < 1e-15);
        assert!((cs.i_bend - expected_i).abs() < 1e-20);
        assert!((cs.kappa - 6.0 / 7.0).abs() < 1e-15);
    }

    #[test]
    fn total_mass_conserved_consistent() {
        let (le, e, g, rho, cs) = get_test_params();
        let me = compute_element_mass_consistent(le, rho, e, g, &cs);

        // Sum of translational mass coefficients should give total element mass
        // For axial DOFs (indices 0 and 3)
        let axial_mass = me[(0, 0)] + me[(0, 3)] + me[(3, 0)] + me[(3, 3)];
        let expected_mass = rho * cs.area * le;

        // Axial mass should equal element mass
        assert!(
            (axial_mass - expected_mass).abs() / expected_mass < 0.01,
            "Total axial mass mismatch: {} vs expected {}",
            axial_mass,
            expected_mass
        );
    }

    #[test]
    fn total_mass_conserved_lumped() {
        let (le, _, _, rho, cs) = get_test_params();
        let me = compute_element_mass_lumped(le, rho, &cs);

        // For lumped mass, sum of diagonal axial terms equals total mass
        let axial_mass = me[(0, 0)] + me[(3, 3)];
        let expected_mass = rho * cs.area * le;

        assert!(
            (axial_mass - expected_mass).abs() < 1e-10,
            "Lumped mass total incorrect: {} vs {}",
            axial_mass,
            expected_mass
        );
    }
}
