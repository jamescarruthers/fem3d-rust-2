//! Type aliases, constants, and core enums for the FEM3D library.

use nalgebra::SMatrix;

// Type aliases for common matrix sizes
pub type NodeCoords = SMatrix<f64, 8, 3>;
pub type Matrix6 = SMatrix<f64, 6, 6>;
pub type Matrix6x24 = SMatrix<f64, 6, 24>;
pub type Matrix3x24 = SMatrix<f64, 3, 24>;
pub type Matrix3x8 = SMatrix<f64, 3, 8>;

// Core constants
pub const DOF_PER_NODE: usize = 3;
pub const DEFAULT_CORNER_TOL: f64 = 1e-6;
pub const Z_DIR_INDEX: usize = 2;

// f64 representation of 1/sqrt(3) Gauss point coordinate for 2×2×2 quadrature
pub const GAUSS_G: f64 = 0.577_350_269_189_625_8;
pub const MIN_DET_J: f64 = 1e-12;

/// Tolerance for eigenvalue filtering.
pub const LAMBDA_TOL: f64 = 1e-12;

/// Eigenvalue threshold used to discard the near-zero rigid-body modes of a free-free bar
/// (matches the Python reference's 100.0 cutoff).
pub const RIGID_BODY_LAMBDA_THRESHOLD: f64 = 100.0;

/// Frequency threshold (Hz) to remove any residual near-rigid-body modes that survive the
/// lambda cutoff on large DOF problems.
pub const MIN_FREQUENCY_HZ: f64 = 5.0;

/// Default shift for shift-invert Lanczos (targets eigenvalues near this value).
/// Set to RIGID_BODY_LAMBDA_THRESHOLD so we target the lowest elastic modes
/// rather than rigid body modes (which have λ ≈ 0).
pub const DEFAULT_SHIFT: f64 = RIGID_BODY_LAMBDA_THRESHOLD;

/// DOF threshold above which sparse solver is used automatically.
pub const SPARSE_DOF_THRESHOLD: usize = 500;

/// Maximum Lanczos iterations.
pub const MAX_LANCZOS_ITER: usize = 300;

/// Convergence tolerance for Lanczos.
pub const LANCZOS_TOL: f64 = 1e-10;

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
