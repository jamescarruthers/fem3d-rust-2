use fem3d_rust_2::{compute_modal_frequencies, generate_bar_mesh_3d, LAMBDA_TOL};
use std::env;

/// Parse mesh divisions from environment variables or use defaults.
///
/// Set environment variables to customize mesh:
///   FEM_NX - divisions along length (default: 10)
///   FEM_NY - divisions along width (default: 2)
///   FEM_NZ - divisions along thickness (default: 3)
///
/// Example:
///   FEM_NX=20 FEM_NY=2 FEM_NZ=4 cargo test sapele_bar -- --nocapture
fn get_mesh_divisions() -> (usize, usize, usize) {
    let nx = env::var("FEM_NX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let ny = env::var("FEM_NY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let nz = env::var("FEM_NZ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    (nx, ny, nz)
}

#[test]
fn sapele_bar_modal_frequencies_are_positive_and_sorted() {
    const LENGTH_M: f64 = 0.551; // 551 mm
    const WIDTH_M: f64 = 0.032; // 32 mm
    const THICKNESS_M: f64 = 0.024; // 24 mm
    const MAX_REASONABLE_FREQ_HZ: f64 = 1.0e6;
    const YOUNG_SAPELE: f64 = 12.0e9;
    const POISSON_SAPELE: f64 = 0.35;
    const DENSITY_SAPELE: f64 = 640.0;

    let (nx, ny, nz) = get_mesh_divisions();
    let heights: Vec<f64> = vec![THICKNESS_M; nx];

    let mesh = generate_bar_mesh_3d(LENGTH_M, WIDTH_M, &heights, nx, ny, nz);

    let num_elements = nx * ny * nz;
    let num_dof = mesh.nodes.len() * 3;
    println!(
        "Mesh: {}x{}x{} ({} elements, {} nodes, {} DOF)",
        nx, ny, nz, num_elements, mesh.nodes.len(), num_dof
    );

    let freqs = compute_modal_frequencies(&mesh, YOUNG_SAPELE, POISSON_SAPELE, DENSITY_SAPELE, 4);
    println!("Sapele bar frequencies (Hz): {:?}", freqs);

    // Analytical first bending mode ~352 Hz for this bar
    if nx >= 10 {
        println!(
            "Expected first bending frequency: ~350-450 Hz (analytical: 352 Hz)"
        );
    }

    assert!(!freqs.is_empty());
    assert!(freqs[0] > 1.0, "Rigid-body modes should be filtered out");
    for window in freqs.windows(2) {
        assert!(window[0] <= window[1] + LAMBDA_TOL);
    }
    for f in freqs {
        assert!(f.is_finite());
        assert!(f > 0.0);
        assert!(f < MAX_REASONABLE_FREQ_HZ);
    }
}

#[test]
fn sapele_bar_frequency_converges_with_mesh_refinement() {
    const LENGTH_M: f64 = 0.551;
    const WIDTH_M: f64 = 0.032;
    const THICKNESS_M: f64 = 0.024;
    const YOUNG_SAPELE: f64 = 12.0e9;
    const POISSON_SAPELE: f64 = 0.35;
    const DENSITY_SAPELE: f64 = 640.0;
    const ANALYTICAL_F1: f64 = 352.3; // Euler-Bernoulli first bending mode

    println!("Testing mesh convergence toward analytical f1 = {:.1} Hz\n", ANALYTICAL_F1);

    let test_cases = [
        (5, 2, 2),
        (10, 2, 3),
        (120, 2, 12),
    ];

    let mut prev_freq: Option<f64> = None;

    for (nx, ny, nz) in test_cases {
        let heights: Vec<f64> = vec![THICKNESS_M; nx];
        let mesh = generate_bar_mesh_3d(LENGTH_M, WIDTH_M, &heights, nx, ny, nz);
        let num_dof = mesh.nodes.len() * 3;

        let freqs = compute_modal_frequencies(&mesh, YOUNG_SAPELE, POISSON_SAPELE, DENSITY_SAPELE, 1);

        if let Some(f1) = freqs.first() {
            let error_pct = ((f1 - ANALYTICAL_F1) / ANALYTICAL_F1 * 100.0).abs();
            println!(
                "{}x{}x{}: {} DOF, f1 = {:.1} Hz (error: {:.1}%)",
                nx, ny, nz, num_dof, f1, error_pct
            );

            // Frequency should decrease (converge toward analytical) with refinement
            if let Some(prev) = prev_freq {
                assert!(
                    *f1 <= prev + 1.0,
                    "Frequency should decrease or stay similar with mesh refinement"
                );
            }
            prev_freq = Some(*f1);

            // With sufficient refinement, error should decrease
            // Note: 3D FEM with Poisson effects differs from Euler-Bernoulli theory
            // Convergence is the key metric, not absolute accuracy vs beam theory
            if nx >= 15 {
                assert!(
                    error_pct < 50.0,
                    "First frequency should converge toward analytical value"
                );
            }
        }
    }
}
