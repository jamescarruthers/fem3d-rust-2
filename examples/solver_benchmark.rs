use fem3d_rust_2::{
    compute_modal_frequencies, compute_modal_frequencies_sparse, generate_bar_mesh_3d, Material,
};
use std::time::Instant;

#[cfg(feature = "sprs-backend")]
use fem3d_rust_2::compute_modal_frequencies_sprs;

const LENGTH_M: f64 = 0.551;
const WIDTH_M: f64 = 0.032;
const THICKNESS_M: f64 = 0.024;
const NUM_MODES: usize = 4;

struct BenchResult {
    solver: String,
    time_ms: f64,
    freqs: Vec<f64>,
}

fn benchmark_dense(mesh: &fem3d_rust_2::Mesh3d, material: &Material) -> Option<BenchResult> {
    let dof = mesh.nodes.len() * 3;

    // Skip dense for very large problems (would take too long)
    if dof > 2000 {
        return None;
    }

    let start = Instant::now();
    let freqs = compute_modal_frequencies(mesh, material.e, material.nu, material.rho, NUM_MODES);
    let elapsed = start.elapsed();

    Some(BenchResult {
        solver: "Dense".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        freqs,
    })
}

fn benchmark_sparse_nalgebra(mesh: &fem3d_rust_2::Mesh3d, material: &Material) -> BenchResult {
    let start = Instant::now();
    let freqs =
        compute_modal_frequencies_sparse(mesh, material.e, material.nu, material.rho, NUM_MODES);
    let elapsed = start.elapsed();

    BenchResult {
        solver: "Sparse (nalgebra)".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        freqs,
    }
}

#[cfg(feature = "sprs-backend")]
fn benchmark_sparse_sprs(mesh: &fem3d_rust_2::Mesh3d, material: &Material) -> BenchResult {
    let start = Instant::now();
    let freqs =
        compute_modal_frequencies_sprs(mesh, material.e, material.nu, material.rho, NUM_MODES);
    let elapsed = start.elapsed();

    BenchResult {
        solver: "Sparse (sprs)".to_string(),
        time_ms: elapsed.as_secs_f64() * 1000.0,
        freqs,
    }
}

fn format_freq(f: Option<&f64>) -> String {
    match f {
        Some(freq) => format!("{:.1}", freq),
        None => "-".to_string(),
    }
}

fn main() {
    let sapele = Material::sapele();

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    FEM Solver Benchmark Comparison                           ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {} bar: {}m x {}m x {}m                                      ║",
        sapele.name, LENGTH_M, WIDTH_M, THICKNESS_M
    );
    println!(
        "║ E={:.0e} Pa, ν={}, ρ={} kg/m³                                    ║",
        sapele.e, sapele.nu, sapele.rho
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Test configurations: (nx, ny, nz)
    let test_configs = [
        (3, 1, 1),    // Tiny
        (5, 2, 2),    // Small
        (10, 2, 3),   // Medium
        (15, 2, 3),   // Medium-large
        (20, 2, 4),   // Large
        (25, 3, 4),   // Larger
        (30, 3, 5),   // Even larger
    ];

    println!("┌─────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ {:^6} │ {:^6} │ {:^20} │ {:^12} │ {:^40} │", "Mesh", "DOF", "Solver", "Time (ms)", "Frequencies (Hz)");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────────────┤");

    for (nx, ny, nz) in test_configs {
        let heights: Vec<f64> = vec![THICKNESS_M; nx];
        let mesh = generate_bar_mesh_3d(LENGTH_M, WIDTH_M, &heights, nx, ny, nz);
        let mesh_label = format!("{}x{}x{}", nx, ny, nz);
        let dof = mesh.nodes.len() * 3;

        let mut results: Vec<BenchResult> = Vec::new();

        // Dense benchmark
        if let Some(result) = benchmark_dense(&mesh, &sapele) {
            results.push(result);
        }

        // Sparse (nalgebra) benchmark
        results.push(benchmark_sparse_nalgebra(&mesh, &sapele));

        // Sparse (sprs) benchmark
        #[cfg(feature = "sprs-backend")]
        results.push(benchmark_sparse_sprs(&mesh, &sapele));

        // Print results for this mesh size
        for (i, result) in results.iter().enumerate() {
            let mesh_str = if i == 0 { mesh_label.clone() } else { "".to_string() };
            let dof_str = if i == 0 { format!("{}", dof) } else { "".to_string() };

            let freq_str = format!(
                "{}, {}, {}, {}",
                format_freq(result.freqs.get(0)),
                format_freq(result.freqs.get(1)),
                format_freq(result.freqs.get(2)),
                format_freq(result.freqs.get(3))
            );

            println!(
                "│ {:^6} │ {:^6} │ {:^20} │ {:>12.2} │ {:^40} │",
                mesh_str, dof_str, result.solver, result.time_ms, freq_str
            );
        }

        // Add separator between mesh sizes
        if nx != 30 {
            println!("├─────────────────────────────────────────────────────────────────────────────────────────────────┤");
        }
    }

    println!("└─────────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Summary
    println!("Notes:");
    println!("  - Dense solver skipped for DOF > 2000 (would take too long)");
    println!("  - Sparse solvers use shift-invert Lanczos algorithm");
    println!("  - Times include matrix assembly and eigenvalue computation");
    println!("  - Analytical first bending mode (Euler-Bernoulli): ~352 Hz");
    println!();

    // Performance scaling analysis
    println!("Performance Scaling Analysis:");
    println!("─────────────────────────────");

    let scaling_configs = [(5, 2, 2), (10, 2, 3), (15, 2, 3), (20, 2, 4)];
    let mut dense_times: Vec<(usize, f64)> = Vec::new();
    let mut sparse_times: Vec<(usize, f64)> = Vec::new();

    for (nx, ny, nz) in scaling_configs {
        let heights: Vec<f64> = vec![THICKNESS_M; nx];
        let mesh = generate_bar_mesh_3d(LENGTH_M, WIDTH_M, &heights, nx, ny, nz);
        let dof = mesh.nodes.len() * 3;

        if let Some(result) = benchmark_dense(&mesh, &sapele) {
            dense_times.push((dof, result.time_ms));
        }

        let sparse_result = benchmark_sparse_nalgebra(&mesh, &sapele);
        sparse_times.push((dof, sparse_result.time_ms));
    }

    println!();
    println!("Dense solver scaling (expected O(n³)):");
    if dense_times.len() >= 2 {
        for i in 1..dense_times.len() {
            let (dof1, t1) = dense_times[i - 1];
            let (dof2, t2) = dense_times[i];
            let dof_ratio = dof2 as f64 / dof1 as f64;
            let time_ratio = t2 / t1;
            let expected_ratio = dof_ratio.powi(3);
            println!(
                "  {} -> {} DOF: time ratio = {:.2}x (expected ~{:.1}x for O(n³))",
                dof1, dof2, time_ratio, expected_ratio
            );
        }
    }

    println!();
    println!("Sparse solver scaling:");
    for i in 1..sparse_times.len() {
        let (dof1, t1) = sparse_times[i - 1];
        let (dof2, t2) = sparse_times[i];
        let dof_ratio = dof2 as f64 / dof1 as f64;
        let time_ratio = t2 / t1;
        println!(
            "  {} -> {} DOF: time ratio = {:.2}x (DOF ratio = {:.2}x)",
            dof1, dof2, time_ratio, dof_ratio
        );
    }

    println!();
    println!("Crossover Analysis:");
    println!("───────────────────");
    for ((dof_d, time_d), (dof_s, time_s)) in dense_times.iter().zip(sparse_times.iter()) {
        assert_eq!(dof_d, dof_s);
        let speedup = time_d / time_s;
        let faster = if speedup > 1.0 { "Sparse" } else { "Dense" };
        println!(
            "  {} DOF: Dense={:.1}ms, Sparse={:.1}ms -> {} is {:.1}x faster",
            dof_d, time_d, time_s, faster, speedup.max(1.0 / speedup)
        );
    }
}
