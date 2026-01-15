//! FEM3D-Rust: 3D Finite Element Modal Analysis Library
//!
//! This library provides tools for performing modal analysis on 3D bar structures
//! using hexahedral (Hex8) finite elements. It supports:
//!
//! - Mesh generation for uniform and adaptive grids
//! - Cut geometry for undercut bar profiles
//! - Dense and sparse eigenvalue solvers
//! - Mode classification using Soares' method
//!
//! # Example
//!
//! ```
//! use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d, compute_modal_frequencies};
//!
//! // Define bar geometry with a cut
//! let cuts = vec![Cut::new(0.1, 0.015)];
//! let heights = generate_element_heights(&cuts, 0.5, 0.024, 20);
//!
//! // Generate mesh
//! let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 20, 2, 2);
//!
//! // Compute modal frequencies
//! let freqs = compute_modal_frequencies(&mesh, 12e9, 0.35, 640.0, 4);
//! ```

// Module declarations
pub mod assembly;
pub mod cuts;
pub mod element;
pub mod mesh;
pub mod modes;
pub mod solver;
pub mod types;

// Re-export commonly used items at the crate root for convenience

// Types and constants
pub use types::{
    EigenSolver, ModeType, SparseBackend, DEFAULT_SHIFT, DOF_PER_NODE, LAMBDA_TOL,
    MIN_FREQUENCY_HZ, RIGID_BODY_LAMBDA_THRESHOLD, SPARSE_DOF_THRESHOLD,
};

// Cut geometry
pub use cuts::{
    compute_height, cuts_to_genes, generate_adaptive_mesh_1d, generate_element_heights,
    genes_to_cuts, Cut,
};

// Element functions
pub use element::{
    compute_hex8_matrices, elasticity_matrix_3d, gauss_points_3d, shape_function_derivatives_hex8,
    shape_functions_hex8,
};

// Assembly functions
pub use assembly::{assemble_global_dense, assemble_global_sparse};
#[cfg(feature = "sprs-backend")]
pub use assembly::assemble_global_sprs;

// Mesh types and generation
pub use mesh::{
    generate_bar_mesh_3d, generate_bar_mesh_3d_adaptive, Mesh3d, MeshMetadata, SerializableElement,
    SerializableMesh, SerializableNode,
};

// Solvers
pub use solver::{
    compute_global_matrices_dense, compute_global_matrices_sparse, compute_modal_frequencies,
    compute_modal_frequencies_full, compute_modal_frequencies_sparse,
    compute_modal_frequencies_with_solver, lanczos_shift_invert,
};
#[cfg(feature = "sprs-backend")]
pub use solver::{compute_global_matrices_sprs, compute_modal_frequencies_sprs, lanczos_shift_invert_sprs};

// Mode classification
pub use modes::{classify_all_modes, classify_mode_soares, find_corner_nodes};
