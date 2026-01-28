//! Comparison of 1D, 2D, and 3D analysis methods on the same bar geometry.
//!
//! This example demonstrates how the three analysis methods compare in terms of:
//! - Computed frequencies
//! - Computation time
//! - Accuracy vs complexity tradeoffs
//!
//! Run with: cargo run --example analysis_comparison

use fem3d_rust_2::{
    // 1D Timoshenko frame
    compute_frequencies_free_free,
    // 2D Timoshenko beam
    compute_modal_frequencies_2d,
    // 3D Hex8 solid
    compute_modal_frequencies, generate_bar_mesh_3d,
    // Material
    Material,
};
use std::time::Instant;

/// Results from a single analysis method
struct AnalysisResult {
    method: String,
    dof: usize,
    time_ms: f64,
    frequencies: Vec<f64>,
}

/// Run 1D Timoshenko frame analysis (3 DOF/node: axial, transverse, rotation)
fn run_1d_analysis(
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    nu: f64,
    rho: f64,
    num_elements: usize,
    num_modes: usize,
) -> AnalysisResult {
    let start = Instant::now();

    let frequencies =
        compute_frequencies_free_free(length, width, height, e, nu, rho, num_elements, num_modes);

    let elapsed = start.elapsed();

    // DOF = 3 per node * (num_elements + 1) nodes
    let dof = 3 * (num_elements + 1);

    AnalysisResult {
        method: "1D Timoshenko Frame".to_string(),
        dof,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        frequencies,
    }
}

/// Run 2D Timoshenko beam analysis (2 DOF/node: transverse, rotation)
fn run_2d_analysis(
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    nu: f64,
    rho: f64,
    num_elements: usize,
    num_modes: usize,
) -> AnalysisResult {
    let start = Instant::now();

    // Uniform height for all elements
    let element_heights: Vec<f64> = vec![height; num_elements];

    let frequencies =
        compute_modal_frequencies_2d(&element_heights, length, width, e, nu, rho, num_modes);

    let elapsed = start.elapsed();

    // DOF = 2 per node * (num_elements + 1) nodes
    let dof = 2 * (num_elements + 1);

    AnalysisResult {
        method: "2D Timoshenko Beam".to_string(),
        dof,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        frequencies,
    }
}

/// Run 3D Hex8 solid element analysis (3 DOF/node: x, y, z displacement)
fn run_3d_analysis(
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    nu: f64,
    rho: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    num_modes: usize,
) -> AnalysisResult {
    let start = Instant::now();

    // Uniform height for all elements
    let element_heights: Vec<f64> = vec![height; nx];

    let mesh = generate_bar_mesh_3d(length, width, &element_heights, nx, ny, nz);
    let frequencies = compute_modal_frequencies(&mesh, e, nu, rho, num_modes);

    let elapsed = start.elapsed();

    // DOF = 3 per node * total nodes
    let dof = 3 * mesh.nodes.len();

    AnalysisResult {
        method: format!("3D Hex8 ({}x{}x{})", nx, ny, nz),
        dof,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        frequencies,
    }
}

/// Compute Euler-Bernoulli analytical frequencies for free-free beam
fn euler_bernoulli_free_free(
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    rho: f64,
    num_modes: usize,
) -> Vec<f64> {
    let i = width * height.powi(3) / 12.0; // Second moment of area
    let a = width * height; // Cross-sectional area

    // β_n * L values for free-free beam (from transcendental equation)
    let beta_l = [4.730, 7.853, 10.996, 14.137, 17.279, 20.420, 23.562, 26.704];

    let coeff = (e * i / (rho * a)).sqrt() / (length * length * 2.0 * std::f64::consts::PI);

    beta_l
        .iter()
        .take(num_modes)
        .map(|bl| bl * bl * coeff)
        .collect()
}

fn print_separator() {
    println!(
        "├────────────────────────┼────────┼────────────┼─────────────────────────────────────────────────┤"
    );
}

fn print_header() {
    println!();
    println!(
        "┌────────────────────────┬────────┬────────────┬─────────────────────────────────────────────────┐"
    );
    println!(
        "│ {:^22} │ {:^6} │ {:^10} │ {:^47} │",
        "Method", "DOF", "Time (ms)", "Frequencies (Hz)"
    );
    println!(
        "├────────────────────────┼────────┼────────────┼─────────────────────────────────────────────────┤"
    );
}

fn print_result(result: &AnalysisResult) {
    let freq_str: String = result
        .frequencies
        .iter()
        .take(4)
        .map(|f| format!("{:>8.1}", f))
        .collect::<Vec<_>>()
        .join(", ");

    println!(
        "│ {:^22} │ {:>6} │ {:>10.2} │ {:^47} │",
        result.method, result.dof, result.time_ms, freq_str
    );
}

fn print_analytical(frequencies: &[f64]) {
    let freq_str: String = frequencies
        .iter()
        .take(4)
        .map(|f| format!("{:>8.1}", f))
        .collect::<Vec<_>>()
        .join(", ");

    println!(
        "│ {:^22} │ {:^6} │ {:^10} │ {:^47} │",
        "Euler-Bernoulli (ref)", "-", "-", freq_str
    );
}

fn print_footer() {
    println!(
        "└────────────────────────┴────────┴────────────┴─────────────────────────────────────────────────┘"
    );
}

fn compute_error(computed: &[f64], reference: &[f64]) -> Vec<f64> {
    computed
        .iter()
        .zip(reference.iter())
        .map(|(c, r)| ((c - r) / r * 100.0).abs())
        .collect()
}

fn main() {
    // ═══════════════════════════════════════════════════════════════════════════
    // Test Case 1: Typical marimba bar (Sapele wood)
    // ═══════════════════════════════════════════════════════════════════════════

    let sapele = Material::sapele();

    let length = 0.350; // 350mm
    let width = 0.040; // 40mm
    let height = 0.022; // 22mm
    let num_modes = 6;

    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         1D vs 2D vs 3D Analysis Comparison                                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Test Case 1: Marimba Bar ({})                                                                ║",
        sapele.name
    );
    println!(
        "║ Dimensions: {:.0}mm x {:.0}mm x {:.0}mm                                                               ║",
        length * 1000.0,
        width * 1000.0,
        height * 1000.0
    );
    println!(
        "║ E = {:.2}e9 Pa, ν = {:.2}, ρ = {:.0} kg/m³                                                          ║",
        sapele.e / 1e9,
        sapele.nu,
        sapele.rho
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // Analytical reference
    let analytical = euler_bernoulli_free_free(length, width, height, sapele.e, sapele.rho, num_modes);

    print_header();

    // Run analyses
    let result_1d = run_1d_analysis(
        length, width, height, sapele.e, sapele.nu, sapele.rho, 100, num_modes,
    );
    print_result(&result_1d);
    print_separator();

    let result_2d = run_2d_analysis(
        length, width, height, sapele.e, sapele.nu, sapele.rho, 100, num_modes,
    );
    print_result(&result_2d);
    print_separator();

    let result_3d_coarse = run_3d_analysis(
        length, width, height, sapele.e, sapele.nu, sapele.rho, 10, 2, 2, num_modes,
    );
    print_result(&result_3d_coarse);
    print_separator();

    let result_3d_medium = run_3d_analysis(
        length, width, height, sapele.e, sapele.nu, sapele.rho, 20, 2, 3, num_modes,
    );
    print_result(&result_3d_medium);
    print_separator();

    let result_3d_fine = run_3d_analysis(
        length, width, height, sapele.e, sapele.nu, sapele.rho, 30, 3, 4, num_modes,
    );
    print_result(&result_3d_fine);
    print_separator();

    print_analytical(&analytical);
    print_footer();

    // Error analysis
    println!();
    println!("Error vs Euler-Bernoulli Reference (%):");
    println!("────────────────────────────────────────");

    let errors_1d = compute_error(&result_1d.frequencies, &analytical);
    let errors_2d = compute_error(&result_2d.frequencies, &analytical);
    let errors_3d = compute_error(&result_3d_fine.frequencies, &analytical);

    println!(
        "  1D Timoshenko: {:>5.1}%, {:>5.1}%, {:>5.1}%, {:>5.1}%",
        errors_1d.get(0).unwrap_or(&0.0),
        errors_1d.get(1).unwrap_or(&0.0),
        errors_1d.get(2).unwrap_or(&0.0),
        errors_1d.get(3).unwrap_or(&0.0)
    );
    println!(
        "  2D Timoshenko: {:>5.1}%, {:>5.1}%, {:>5.1}%, {:>5.1}%",
        errors_2d.get(0).unwrap_or(&0.0),
        errors_2d.get(1).unwrap_or(&0.0),
        errors_2d.get(2).unwrap_or(&0.0),
        errors_2d.get(3).unwrap_or(&0.0)
    );
    println!(
        "  3D Hex8 (fine): {:>5.1}%, {:>5.1}%, {:>5.1}%, {:>5.1}%",
        errors_3d.get(0).unwrap_or(&0.0),
        errors_3d.get(1).unwrap_or(&0.0),
        errors_3d.get(2).unwrap_or(&0.0),
        errors_3d.get(3).unwrap_or(&0.0)
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Case 2: Steel beam (thicker, where shear effects matter more)
    // ═══════════════════════════════════════════════════════════════════════════

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ Test Case 2: Thick Steel Beam (L/h = 5, strong shear effects)                                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝");

    let length2 = 0.250; // 250mm
    let width2 = 0.050; // 50mm
    let height2 = 0.050; // 50mm (L/h = 5, very thick beam)
    let e_steel = 200e9;
    let nu_steel = 0.3;
    let rho_steel = 7850.0;

    let analytical2 =
        euler_bernoulli_free_free(length2, width2, height2, e_steel, rho_steel, num_modes);

    print_header();

    let result_1d_thick = run_1d_analysis(
        length2, width2, height2, e_steel, nu_steel, rho_steel, 100, num_modes,
    );
    print_result(&result_1d_thick);
    print_separator();

    let result_2d_thick = run_2d_analysis(
        length2, width2, height2, e_steel, nu_steel, rho_steel, 100, num_modes,
    );
    print_result(&result_2d_thick);
    print_separator();

    let result_3d_thick = run_3d_analysis(
        length2, width2, height2, e_steel, nu_steel, rho_steel, 15, 3, 3, num_modes,
    );
    print_result(&result_3d_thick);
    print_separator();

    print_analytical(&analytical2);
    print_footer();

    println!();
    println!("Note: For thick beams (L/h = 5), Timoshenko theory gives LOWER frequencies than");
    println!("      Euler-Bernoulli due to shear deformation. This is physically correct!");
    println!();
    println!("Error vs Euler-Bernoulli (negative = softer due to shear):");
    println!("──────────────────────────────────────────────────────────");

    let errors_1d_thick = result_1d_thick
        .frequencies
        .iter()
        .zip(analytical2.iter())
        .map(|(c, r)| (c - r) / r * 100.0)
        .collect::<Vec<_>>();
    let errors_2d_thick = result_2d_thick
        .frequencies
        .iter()
        .zip(analytical2.iter())
        .map(|(c, r)| (c - r) / r * 100.0)
        .collect::<Vec<_>>();

    println!(
        "  1D Timoshenko: {:>+5.1}%, {:>+5.1}%, {:>+5.1}%, {:>+5.1}%",
        errors_1d_thick.get(0).unwrap_or(&0.0),
        errors_1d_thick.get(1).unwrap_or(&0.0),
        errors_1d_thick.get(2).unwrap_or(&0.0),
        errors_1d_thick.get(3).unwrap_or(&0.0)
    );
    println!(
        "  2D Timoshenko: {:>+5.1}%, {:>+5.1}%, {:>+5.1}%, {:>+5.1}%",
        errors_2d_thick.get(0).unwrap_or(&0.0),
        errors_2d_thick.get(1).unwrap_or(&0.0),
        errors_2d_thick.get(2).unwrap_or(&0.0),
        errors_2d_thick.get(3).unwrap_or(&0.0)
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // Performance comparison
    // ═══════════════════════════════════════════════════════════════════════════

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║ Performance Scaling Comparison                                                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Running multiple mesh refinements...");
    println!();

    let ne_values = [10, 25, 50, 100, 200];

    println!(
        "┌─────────────┬──────────────────────────┬──────────────────────────┬─────────────────────────────────┐"
    );
    println!(
        "│ {:^11} │ {:^24} │ {:^24} │ {:^31} │",
        "Elements", "1D Frame", "2D Beam", "3D Solid"
    );
    println!(
        "│             │ {:^10} {:^11} │ {:^10} {:^11} │ {:^13} {:^15} │",
        "DOF", "Time (ms)", "DOF", "Time (ms)", "DOF", "Time (ms)"
    );
    println!(
        "├─────────────┼──────────────────────────┼──────────────────────────┼─────────────────────────────────┤"
    );

    for ne in ne_values {
        let r1d = run_1d_analysis(length, width, height, sapele.e, sapele.nu, sapele.rho, ne, 4);
        let r2d = run_2d_analysis(length, width, height, sapele.e, sapele.nu, sapele.rho, ne, 4);

        // Scale 3D mesh proportionally (but keep reasonable ny, nz)
        let nx = ne.min(30);
        let ny = 2;
        let nz = 2;
        let r3d =
            run_3d_analysis(length, width, height, sapele.e, sapele.nu, sapele.rho, nx, ny, nz, 4);

        println!(
            "│ {:>11} │ {:>10} {:>11.2} │ {:>10} {:>11.2} │ {:>13} {:>15.2} │",
            ne, r1d.dof, r1d.time_ms, r2d.dof, r2d.time_ms, r3d.dof, r3d.time_ms
        );
    }

    println!(
        "└─────────────┴──────────────────────────┴──────────────────────────┴─────────────────────────────────┘"
    );

    // ═══════════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════════

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("1D TIMOSHENKO FRAME (beam1d):");
    println!("  - DOF per node: 3 (axial u, transverse w, rotation θ)");
    println!("  - Captures: Bending modes + axial modes");
    println!("  - Speed: Very fast (analytical element matrices)");
    println!("  - Best for: Quick estimates, parametric studies, optimization");
    println!();
    println!("2D TIMOSHENKO BEAM (beam2d):");
    println!("  - DOF per node: 2 (transverse w, rotation θ)");
    println!("  - Captures: Bending modes only");
    println!("  - Speed: Very fast (analytical element matrices)");
    println!("  - Best for: Bars with varying cross-section (cuts), fast tuning");
    println!();
    println!("3D HEX8 SOLID (element):");
    println!("  - DOF per node: 3 (displacements x, y, z)");
    println!("  - Captures: All modes (bending, torsional, axial, lateral)");
    println!("  - Speed: Slower (numerical integration, larger matrices)");
    println!("  - Best for: Final verification, complex geometries, mode shapes");
    println!();
    println!("ACCURACY NOTES:");
    println!("  - Timoshenko models account for shear deformation (important for L/h < 10)");
    println!("  - 3D model captures 3D effects (lateral modes, torsion, Poisson coupling)");
    println!("  - All models converge to similar bending frequencies for slender beams");
    println!("  - For percussion bars, 2D beam is typically sufficient for tuning");
    println!();
}
