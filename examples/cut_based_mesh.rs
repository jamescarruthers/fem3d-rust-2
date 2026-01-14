/// Example demonstrating mesh generation with cut-based undercut profile.
///
/// This example shows how to:
/// 1. Define cuts that create an undercut bar profile
/// 2. Generate element heights from cuts with quadratic interpolation
/// 3. Create a 3D mesh from the height profile
/// 4. Compute modal frequencies of the undercut bar
///
/// The undercut profile is symmetric about the bar center, with nested cuts
/// creating a stepped profile commonly seen in percussion instruments like
/// xylophones and marimbas.

use fem3d_rust_2::{
    compute_height, compute_modal_frequencies_with_solver, cuts_to_genes, genes_to_cuts,
    generate_bar_mesh_3d, generate_element_heights, Cut, EigenSolver,
};

fn main() {
    println!("=== Cut-Based Mesh Generation Example ===\n");

    // Bar dimensions (similar to a marimba bar)
    let length = 0.551; // 551 mm
    let width = 0.032; // 32 mm
    let h0 = 0.024; // 24 mm original thickness

    // Material properties (Sapele wood)
    let e = 12.0e9; // Young's modulus (Pa)
    let rho = 640.0; // Density (kg/m³)
    let nu = 0.35; // Poisson's ratio

    println!("Bar dimensions:");
    println!("  Length: {:.0} mm", length * 1000.0);
    println!("  Width: {:.0} mm", width * 1000.0);
    println!("  Original thickness: {:.1} mm\n", h0 * 1000.0);

    // Example 1: Simple single cut
    println!("Example 1: Single symmetric cut");
    println!("================================");
    let cuts_simple = vec![Cut::new(0.15, 0.015)];
    
    println!("Cut definition:");
    println!("  Lambda (from center): {:.0} mm", cuts_simple[0].lambda * 1000.0);
    println!("  Height at cut: {:.1} mm", cuts_simple[0].h * 1000.0);
    
    // Generate element heights
    let num_elements = 30;
    let heights_simple = generate_element_heights(&cuts_simple, length, h0, num_elements);
    
    println!("\nElement statistics:");
    let min_h = heights_simple
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_h = heights_simple
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("  Min height: {:.2} mm", min_h * 1000.0);
    println!("  Max height: {:.2} mm", max_h * 1000.0);
    println!("  Height variation: {:.2} mm", (max_h - min_h) * 1000.0);

    // Generate mesh and compute frequencies
    let mesh_simple = generate_bar_mesh_3d(length, width, &heights_simple, num_elements, 2, 2);
    println!("\nMesh statistics:");
    println!("  Elements: {}", mesh_simple.elements.len());
    println!("  Nodes: {}", mesh_simple.nodes.len());
    println!("  DOF: {}", mesh_simple.nodes.len() * 3);

    let freqs_simple = compute_modal_frequencies_with_solver(
        &mesh_simple,
        e,
        nu,
        rho,
        4,
        EigenSolver::Auto,
    );
    println!("\nModal frequencies:");
    for (i, freq) in freqs_simple.iter().enumerate() {
        println!("  Mode {}: {:.1} Hz", i + 1, freq);
    }

    // Example 2: Nested cuts (typical marimba bar)
    println!("\n\nExample 2: Nested cuts (marimba-style)");
    println!("======================================");
    let cuts_nested = vec![
        Cut::new(0.20, 0.020), // Outer cut
        Cut::new(0.12, 0.015), // Middle cut
        Cut::new(0.06, 0.012), // Inner cut (deepest)
    ];

    println!("Cut definitions:");
    for (i, cut) in cuts_nested.iter().enumerate() {
        println!(
            "  Cut {}: lambda={:.0} mm, h={:.1} mm",
            i + 1,
            cut.lambda * 1000.0,
            cut.h * 1000.0
        );
    }

    // Generate element heights
    let heights_nested = generate_element_heights(&cuts_nested, length, h0, num_elements);
    
    println!("\nElement statistics:");
    let min_h = heights_nested
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_h = heights_nested
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("  Min height: {:.2} mm", min_h * 1000.0);
    println!("  Max height: {:.2} mm", max_h * 1000.0);
    println!("  Height variation: {:.2} mm", (max_h - min_h) * 1000.0);

    let mesh_nested = generate_bar_mesh_3d(length, width, &heights_nested, num_elements, 2, 3);
    println!("\nMesh statistics:");
    println!("  Elements: {}", mesh_nested.elements.len());
    println!("  Nodes: {}", mesh_nested.nodes.len());
    println!("  DOF: {}", mesh_nested.nodes.len() * 3);

    let freqs_nested = compute_modal_frequencies_with_solver(
        &mesh_nested,
        e,
        nu,
        rho,
        4,
        EigenSolver::Auto,
    );
    println!("\nModal frequencies:");
    for (i, freq) in freqs_nested.iter().enumerate() {
        println!("  Mode {}: {:.1} Hz", i + 1, freq);
    }

    // Example 3: Demonstrating genes/cuts conversion
    println!("\n\nExample 3: Genes/Cuts conversion");
    println!("=================================");
    
    // Convert cuts to genes format (used in optimization)
    let genes = cuts_to_genes(&cuts_nested);
    println!("Cuts as genes: {:?}", genes);
    
    // Convert back to cuts
    let cuts_from_genes = genes_to_cuts(&genes);
    println!("Genes back to cuts:");
    for (i, cut) in cuts_from_genes.iter().enumerate() {
        println!(
            "  Cut {}: lambda={:.3}, h={:.3}",
            i + 1,
            cut.lambda,
            cut.h
        );
    }

    // Example 4: Profile visualization (print heights along bar)
    println!("\n\nExample 4: Profile visualization");
    println!("=================================");
    let num_points = 40;
    println!("Height profile (from left to right):");
    print!("Position (mm): ");
    for i in 0..=num_points {
        let x = (i as f64 / num_points as f64) * length;
        if i % 5 == 0 {
            print!("{:5.0} ", x * 1000.0);
        }
    }
    println!();
    print!("Height (mm):   ");
    for i in 0..=num_points {
        let x = (i as f64 / num_points as f64) * length;
        let h = compute_height(x, &cuts_nested, length, h0);
        if i % 5 == 0 {
            print!("{:5.1} ", h * 1000.0);
        }
    }
    println!("\n");

    // ASCII art visualization
    println!("Visual profile (side view, '#' = material):");
    let viz_width = 60;
    let viz_height = 12;
    
    for row in 0..viz_height {
        let threshold_height = h0 * (1.0 - row as f64 / viz_height as f64);
        for col in 0..viz_width {
            let x = (col as f64 / viz_width as f64) * length;
            let h = compute_height(x, &cuts_nested, length, h0);
            if h >= threshold_height {
                print!("#");
            } else {
                print!(" ");
            }
        }
        if row == 0 {
            println!("  <- {:.1} mm (top)", h0 * 1000.0);
        } else if row == viz_height - 1 {
            let bottom_h = heights_nested
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            println!("  <- {:.1} mm (bottom)", bottom_h * 1000.0);
        } else {
            println!();
        }
    }

    println!("\n=== Example Complete ===");
}
