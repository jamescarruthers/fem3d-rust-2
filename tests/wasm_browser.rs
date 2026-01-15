//! WASM Browser Tests
//!
//! These tests run in a real browser environment using wasm-pack test.
//! They simulate browser memory constraints and verify WASM compatibility.
//!
//! Run with:
//!   wasm-pack test --headless --chrome
//!   wasm-pack test --headless --firefox
//!   wasm-pack test --node

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use fem3d_rust_2::{
    generate_bar_mesh_3d, compute_modal_frequencies,
    compute_modal_frequencies_2d_from_cuts, Cut,
};

/// Test basic mesh generation in WASM - small mesh for memory constraints.
#[wasm_bindgen_test]
fn test_small_mesh_generation() {
    // Small mesh suitable for browser memory constraints
    let length = 0.3;
    let width = 0.03;
    let heights = vec![0.02; 10]; // Only 10 elements
    let nx = 10;
    let ny = 2;
    let nz = 2;

    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);

    assert!(!mesh.nodes.is_empty(), "Mesh should have nodes");
    assert!(!mesh.elements.is_empty(), "Mesh should have elements");

    // Verify reasonable mesh size
    let expected_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    assert_eq!(mesh.nodes.len(), expected_nodes);
}

/// Test modal frequency computation with minimal mesh.
#[wasm_bindgen_test]
fn test_modal_analysis_minimal() {
    let length = 0.2;
    let width = 0.02;
    let heights = vec![0.015; 8];
    let nx = 8;
    let ny = 2;
    let nz = 2;

    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);

    // Sapele material properties
    let e = 10.0e9;
    let nu = 0.35;
    let rho = 640.0;
    let num_modes = 4;

    let frequencies = compute_modal_frequencies(&mesh, e, nu, rho, num_modes);

    assert_eq!(frequencies.len(), num_modes, "Should return requested number of modes");
    assert!(frequencies[0] > 0.0, "First frequency should be positive");

    // Frequencies should be in ascending order
    for i in 1..frequencies.len() {
        assert!(frequencies[i] >= frequencies[i-1], "Frequencies should be ascending");
    }
}

/// Test 2D beam solver - much faster and lower memory than 3D.
#[wasm_bindgen_test]
fn test_2d_beam_solver() {
    let cuts = vec![
        Cut { lambda: 0.22, h: 0.5 },
        Cut { lambda: 0.78, h: 0.5 },
    ];
    let length = 0.5;
    let width = 0.03;
    let height = 0.024;
    let e = 10.0e9;
    let nu = 0.35;
    let rho = 640.0;
    let elements = 50;
    let modes = 4;

    let frequencies = compute_modal_frequencies_2d_from_cuts(
        &cuts, length, width, height, e, nu, rho, elements, modes,
    );

    assert_eq!(frequencies.len(), modes);
    assert!(frequencies[0] > 100.0 && frequencies[0] < 1000.0,
        "First mode should be in reasonable range: {}", frequencies[0]);
}

/// Test mesh with variable heights (cut bar).
#[wasm_bindgen_test]
fn test_variable_height_mesh() {
    let length = 0.3;
    let width = 0.03;
    // Simulate a cut bar with variable heights
    let heights: Vec<f64> = (0..12)
        .map(|i| {
            let x = i as f64 / 11.0;
            0.02 * (1.0 - 0.3 * (2.0 * std::f64::consts::PI * x).sin().abs())
        })
        .collect();
    let nx = 12;
    let ny = 2;
    let nz = 2;

    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);
    assert!(!mesh.nodes.is_empty());
}

/// Memory stress test - moderate mesh size.
/// This tests whether the WASM memory allocation handles a reasonable workload.
#[wasm_bindgen_test]
fn test_moderate_mesh_memory() {
    let length = 0.4;
    let width = 0.03;
    let heights = vec![0.02; 20];
    let nx = 20;
    let ny = 3;
    let nz = 3;

    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);

    // This creates a mesh with (20+1)*(3+1)*(3+1) = 336 nodes
    // and 20*3*3 = 180 elements - manageable for browser
    assert_eq!(mesh.nodes.len(), 336);
    assert_eq!(mesh.elements.len(), 180);

    // Run modal analysis on moderate mesh
    let e = 10.0e9;
    let nu = 0.35;
    let rho = 640.0;

    let frequencies = compute_modal_frequencies(&mesh, e, nu, rho, 4);
    assert_eq!(frequencies.len(), 4);
}

/// Test that we can create multiple meshes (memory allocation/deallocation).
#[wasm_bindgen_test]
fn test_multiple_mesh_allocations() {
    for i in 0..5 {
        let nx = 8 + i;
        let heights = vec![0.02; nx];
        let mesh = generate_bar_mesh_3d(0.3, 0.03, &heights, nx, 2, 2);
        assert!(!mesh.nodes.is_empty());
        // Mesh goes out of scope and should be deallocated
    }
}

/// Test serialization works in WASM (important for JS interop).
#[wasm_bindgen_test]
fn test_serialization() {
    let cuts = vec![
        Cut { lambda: 0.2, h: 0.6 },
        Cut { lambda: 0.8, h: 0.6 },
    ];

    // Serialize to JSON
    let json = serde_json::to_string(&cuts).expect("Serialization should work");
    assert!(json.contains("lambda"));

    // Deserialize back
    let decoded: Vec<Cut> = serde_json::from_str(&json).expect("Deserialization should work");
    assert_eq!(decoded.len(), 2);
    assert!((decoded[0].lambda - 0.2).abs() < 1e-10);
}
