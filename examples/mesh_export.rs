/// Example demonstrating mesh export for frontend visualization.
///
/// This example shows how to:
/// 1. Generate a mesh with cuts
/// 2. Export the mesh to JSON for frontend consumption
/// 3. Demonstrate the structure of the exported data
///
/// The exported mesh can be used in web-based 3D visualization libraries
/// like Three.js, Babylon.js, or other WebGL frameworks.

use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d};

fn main() {
    println!("=== Mesh Export for Frontend Visualization ===\n");

    // Define a bar with nested cuts (marimba-style)
    let cuts = vec![
        Cut::new(0.20, 0.020), // Outer cut
        Cut::new(0.12, 0.015), // Middle cut
        Cut::new(0.06, 0.012), // Inner cut (deepest)
    ];

    let length = 0.551; // 551 mm
    let width = 0.032; // 32 mm
    let h0 = 0.024; // 24 mm original thickness
    let num_elements = 30;

    println!("Bar configuration:");
    println!("  Length: {:.0} mm", length * 1000.0);
    println!("  Width: {:.0} mm", width * 1000.0);
    println!("  Original thickness: {:.1} mm", h0 * 1000.0);
    println!("  Number of cuts: {}", cuts.len());
    println!();

    // Generate element heights from cuts
    let heights = generate_element_heights(&cuts, length, h0, num_elements);

    // Generate 3D mesh
    let mesh = generate_bar_mesh_3d(length, width, &heights, num_elements, 2, 3);

    println!("Generated mesh:");
    println!("  Nodes: {}", mesh.nodes.len());
    println!("  Elements: {}", mesh.elements.len());
    println!("  DOF: {}", mesh.nodes.len() * 3);
    println!();

    // Convert to serializable format
    let serializable_mesh = mesh.to_serializable();

    println!("Mesh metadata:");
    println!("  Bounding box min: [{:.3}, {:.3}, {:.3}] m",
        serializable_mesh.metadata.bbox_min[0],
        serializable_mesh.metadata.bbox_min[1],
        serializable_mesh.metadata.bbox_min[2]
    );
    println!("  Bounding box max: [{:.3}, {:.3}, {:.3}] m",
        serializable_mesh.metadata.bbox_max[0],
        serializable_mesh.metadata.bbox_max[1],
        serializable_mesh.metadata.bbox_max[2]
    );
    println!();

    // Export to JSON (compact)
    let json_compact = mesh.to_json().unwrap();
    println!("Compact JSON size: {} bytes", json_compact.len());
    println!();

    // Export to pretty-printed JSON
    let json_pretty = mesh.to_json_pretty().unwrap();
    println!("Pretty JSON (first 50 lines):");
    println!("{}", "-".repeat(60));
    for line in json_pretty.lines().take(50) {
        println!("{}", line);
    }
    if json_pretty.lines().count() > 50 {
        println!("... ({} more lines)", json_pretty.lines().count() - 50);
    }
    println!("{}", "-".repeat(60));
    println!();

    // Show a sample element
    if let Some(element) = serializable_mesh.elements.first() {
        println!("Sample element structure:");
        println!("  Hexahedral element with 8 nodes:");
        println!("  Node indices: {:?}", element.nodes);
        println!("  Node 0 position: ({:.3}, {:.3}, {:.3})",
            serializable_mesh.nodes[element.nodes[0]].x,
            serializable_mesh.nodes[element.nodes[0]].y,
            serializable_mesh.nodes[element.nodes[0]].z
        );
        println!("  Node 6 position: ({:.3}, {:.3}, {:.3})",
            serializable_mesh.nodes[element.nodes[6]].x,
            serializable_mesh.nodes[element.nodes[6]].y,
            serializable_mesh.nodes[element.nodes[6]].z
        );
        println!();
    }

    // Show how to save to file
    println!("To save to file:");
    println!("  use std::fs::File;");
    println!("  use std::io::Write;");
    println!();
    println!("  let json = mesh.to_json_pretty().unwrap();");
    println!("  let mut file = File::create(\"mesh.json\").unwrap();");
    println!("  file.write_all(json.as_bytes()).unwrap();");
    println!();

    // Frontend integration hints
    println!("Frontend Integration:");
    println!("====================");
    println!();
    println!("The exported JSON contains:");
    println!("  • nodes: Array of {{x, y, z}} positions");
    println!("  • elements: Array of hexahedral elements (8 node indices each)");
    println!("  • element_heights: Height value for each element (for coloring)");
    println!("  • metadata: Bounding box and mesh statistics");
    println!();
    println!("Example Three.js usage:");
    println!("  // Load mesh data");
    println!("  const meshData = await fetch('mesh.json').then(r => r.json());");
    println!();
    println!("  // Create geometry from nodes and elements");
    println!("  const geometry = new THREE.BufferGeometry();");
    println!("  const vertices = new Float32Array(");
    println!("    meshData.nodes.flatMap(n => [n.x, n.y, n.z])");
    println!("  );");
    println!("  geometry.setAttribute('position',");
    println!("    new THREE.BufferAttribute(vertices, 3)");
    println!("  );");
    println!();
    println!("  // Add indices for rendering");
    println!("  // Convert hex8 elements to triangles for Three.js");
    println!("  const indices = convertHex8ToTriangles(meshData.elements);");
    println!("  geometry.setIndex(indices);");
    println!();
    println!("  // Use element_heights for color mapping");
    println!("  const colors = meshData.element_heights.map(h => ");
    println!("    mapHeightToColor(h, minHeight, maxHeight)");
    println!("  );");
    println!();

    println!("=== Export Complete ===");
}
