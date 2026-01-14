"""
Timoshenko Beam Element Matrices

Implements Timoshenko beam element stiffness and mass matrices
for computing natural frequencies of undercut bars.

The Timoshenko beam model accounts for shear deformation, which is
important for thick bars used in percussion instruments.
"""

import numpy as np
from ..data.materials import KAPPA


def compute_element_stiffness(
    le: float,
    E: float,
    I: float,
    G: float,
    A: float
) -> np.ndarray:
    """
    Compute Timoshenko beam element stiffness matrix (4x4).
    DOFs: [w1, theta1, w2, theta2] - transverse displacement and rotation.

    Args:
        le: Element length (m)
        E: Young's modulus (Pa)
        I: Second moment of area (m^4)
        G: Shear modulus (Pa)
        A: Cross-sectional area (m^2)

    Returns:
        4x4 element stiffness matrix
    """
    phi = 12.0 * E * I / (KAPPA * G * A * le * le)
    denom = (1.0 + phi) * le * le * le

    k11 = 12.0 * E * I / denom
    k12 = 6.0 * E * I * le / denom
    k22 = (4.0 + phi) * E * I * le * le / denom
    k23 = (2.0 - phi) * E * I * le * le / denom

    return np.array([
        [k11,   k12,  -k11,   k12],
        [k12,   k22,  -k12,   k23],
        [-k11, -k12,   k11,  -k12],
        [k12,   k23,  -k12,   k22],
    ], dtype=np.float64)


def compute_element_mass(
    le: float,
    rho: float,
    A: float,
    I: float,
    E: float,
    G: float
) -> np.ndarray:
    """
    Compute Timoshenko beam element mass matrix (4x4).
    Using consistent mass matrix formulation with rotary inertia.

    Args:
        le: Element length (m)
        rho: Density (kg/m^3)
        A: Cross-sectional area (m^2)
        I: Second moment of area (m^4)
        E: Young's modulus (Pa)
        G: Shear modulus (Pa)

    Returns:
        4x4 element mass matrix
    """
    phi = 12.0 * E * I / (KAPPA * G * A * le * le)
    denom = (1.0 + phi) ** 2

    # Translational mass terms
    m = rho * A * le

    # Rotary inertia terms
    r2 = I / A  # radius of gyration squared

    c1 = (13.0/35.0 + 7.0*phi/10.0 + phi*phi/3.0) / denom
    c2 = (9.0/70.0 + 3.0*phi/10.0 + phi*phi/6.0) / denom
    c3 = (11.0/210.0 + 11.0*phi/120.0 + phi*phi/24.0) * le / denom
    c4 = (13.0/420.0 + 3.0*phi/40.0 + phi*phi/24.0) * le / denom
    c5 = (1.0/105.0 + phi/60.0 + phi*phi/120.0) * le * le / denom
    c6 = (1.0/140.0 + phi/60.0 + phi*phi/120.0) * le * le / denom

    # Add rotary inertia contribution
    r_scale = r2 / (le * le)
    r1 = (6.0/5.0) / denom * r_scale
    r2_term = (2.0/15.0 + phi/6.0 + phi*phi/3.0) * le * le / denom * r_scale
    r3 = (1.0/10.0 - phi/2.0) * le / denom * r_scale
    r4 = (-1.0/30.0 - phi/6.0 + phi*phi/6.0) * le * le / denom * r_scale

    return np.array([
        [m * (c1 + r1),     m * (c3 + r3),      m * (c2 - r1),     m * (-c4 + r3)],
        [m * (c3 + r3),     m * (c5 + r2_term), m * (c4 - r3),     m * (-c6 + r4)],
        [m * (c2 - r1),     m * (c4 - r3),      m * (c1 + r1),     m * (-c3 - r3)],
        [m * (-c4 + r3),    m * (-c6 + r4),     m * (-c3 - r3),    m * (c5 + r2_term)],
    ], dtype=np.float64)
