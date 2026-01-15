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
    classify_all_modes, compute_height, compute_modal_frequencies_with_shapes_solver,
    cuts_to_genes, genes_to_cuts, generate_adaptive_mesh_1d, generate_bar_mesh_3d,
    generate_bar_mesh_3d_adaptive, generate_element_heights, Cut, EigenSolver, Material, ModeType,
};

fn mode_type_prefix(mode_type: &ModeType) -> &'static str {
    match mode_type {
        ModeType::VerticalBending => "V",
        ModeType::Torsional => "T",
        ModeType::Lateral => "L",
        ModeType::Axial => "A",
        ModeType::Unknown => "?",
    }
}

fn print_classified_modes(
    freqs: &[f64],
    mode_shapes: &nalgebra::DMatrix<f64>,
    nodes: &[nalgebra::Vector3<f64>],
) {
    let classified = classify_all_modes(freqs, mode_shapes, nodes);
    println!("\nClassified modes:");
    for (mode_type, modes) in &classified {
        if !modes.is_empty() {
            let mode_strs: Vec<String> = modes
                .iter()
                .map(|(f, _, rank)| format!("{}{}: {:.1} Hz", mode_type_prefix(mode_type), rank, f))
                .collect();
            println!("  {:?}: {}", mode_type, mode_strs.join(", "));
        }
    }
}

fn main() {
    println!("=== Cut-Based Mesh Generation Example ===\n");

    // Bar dimensions (similar to a marimba bar)
    let length = 0.551; // 551 mm
    let width = 0.032; // 32 mm
    let h0 = 0.024; // 24 mm original thickness

    // Material properties (Sapele wood)
    let sapele = Material::sapele();

    println!("Bar dimensions:");
    println!("  Length: {:.0} mm", length * 1000.0);
    println!("  Width: {:.0} mm", width * 1000.0);
    println!("  Original thickness: {:.1} mm", h0 * 1000.0);
    println!("  Material: {}\n", sapele.name);

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

    let (freqs_simple, shapes_simple) = compute_modal_frequencies_with_shapes_solver(
        &mesh_simple,
        sapele.e,
        sapele.nu,
        sapele.rho,
        4,
        EigenSolver::Auto,
    );
    println!("\nModal frequencies:");
    for (i, freq) in freqs_simple.iter().enumerate() {
        println!("  Mode {}: {:.1} Hz", i + 1, freq);
    }
    print_classified_modes(&freqs_simple, &shapes_simple, &mesh_simple.nodes);

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

    let (freqs_nested, shapes_nested) = compute_modal_frequencies_with_shapes_solver(
        &mesh_nested,
        sapele.e,
        sapele.nu,
        sapele.rho,
        4,
        EigenSolver::Auto,
    );
    println!("\nModal frequencies:");
    for (i, freq) in freqs_nested.iter().enumerate() {
        println!("  Mode {}: {:.1} Hz", i + 1, freq);
    }
    print_classified_modes(&freqs_nested, &shapes_nested, &mesh_nested.nodes);

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

    // Example 5: Adaptive mesh refinement
    println!("\n\nExample 5: Adaptive mesh refinement");
    println!("====================================");
    
    println!("Comparing uniform vs adaptive meshing:");
    
    // Uniform mesh
    let uniform_heights = generate_element_heights(&cuts_nested, length, h0, 30);
    println!("\nUniform mesh:");
    println!("  Elements: {}", uniform_heights.len());
    
    // Adaptive mesh - refines near cut boundaries
    let (x_positions, adaptive_heights) = generate_adaptive_mesh_1d(
        &cuts_nested,
        length,
        h0,
        30,     // base elements
        4,      // refinement factor
        0.02,   // transition width (2% of length)
    );
    
    println!("\nAdaptive mesh:");
    println!("  Elements: {}", adaptive_heights.len());
    println!("  X positions: {} boundary points", x_positions.len());
    
    // Compute element sizes for adaptive mesh
    let mut element_sizes: Vec<f64> = Vec::new();
    for i in 0..x_positions.len() - 1 {
        element_sizes.push((x_positions[i + 1] - x_positions[i]) * 1000.0);
    }
    
    let min_size = element_sizes
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_size = element_sizes
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    
    println!("  Element size range: {:.2} - {:.2} mm", min_size, max_size);
    println!("  Refinement ratio: {:.1}x", max_size / min_size);
    
    // Create 3D mesh with adaptive spacing
    let adaptive_mesh = generate_bar_mesh_3d_adaptive(
        length,
        width,
        &x_positions,
        &adaptive_heights,
        2,
        2,
    );
    
    println!("\nAdaptive 3D mesh:");
    println!("  Elements: {}", adaptive_mesh.elements.len());
    println!("  Nodes: {}", adaptive_mesh.nodes.len());
    println!("  DOF: {}", adaptive_mesh.nodes.len() * 3);
    
    // Compute frequencies with adaptive mesh
    let (freqs_adaptive, shapes_adaptive) = compute_modal_frequencies_with_shapes_solver(
        &adaptive_mesh,
        sapele.e,
        sapele.nu,
        sapele.rho,
        4,
        EigenSolver::Auto,
    );

    println!("\nModal frequencies (adaptive mesh):");
    for (i, freq) in freqs_adaptive.iter().enumerate() {
        println!("  Mode {}: {:.1} Hz", i + 1, freq);
    }
    print_classified_modes(&freqs_adaptive, &shapes_adaptive, &adaptive_mesh.nodes);
    
    println!("\nBenefit of adaptive meshing:");
    println!("  - Finer elements near cut boundaries capture discontinuities");
    println!("  - Coarser elements in uniform regions reduce computational cost");
    println!("  - Better accuracy per DOF compared to uniform mesh");

    println!("\n=== Example Complete ===");
}
