use fem3d_rust_2::{
    classify_all_modes, compute_modal_frequencies_with_shapes, generate_bar_mesh_3d, Material,
    ModeType,
};
use std::time::Instant;

fn main() {
    const LENGTH_M: f64 = 0.551;
    const WIDTH_M: f64 = 0.032;
    const THICKNESS_M: f64 = 0.024;

    let sapele = Material::sapele();

    println!(
        "{} bar: {}m x {}m x {}m",
        sapele.name, LENGTH_M, WIDTH_M, THICKNESS_M
    );
    println!(
        "E={:.0e} Pa, nu={}, rho={} kg/m³\n",
        sapele.e, sapele.nu, sapele.rho
    );

    // Test with 1x1x1 mesh (same as current test)
    test_mesh(
        "1x1x1",
        1,
        1,
        1,
        LENGTH_M,
        WIDTH_M,
        THICKNESS_M,
        &sapele,
    );

    // Test with 5x2x2 mesh
    test_mesh(
        "5x2x2",
        5,
        2,
        2,
        LENGTH_M,
        WIDTH_M,
        THICKNESS_M,
        &sapele,
    );

    // Test with 10x2x3 mesh
    test_mesh(
        "10x2x3",
        10,
        2,
        3,
        LENGTH_M,
        WIDTH_M,
        THICKNESS_M,
        &sapele,
    );

    // Test with 20x2x4 mesh
    test_mesh(
        "20x2x4",
        20,
        2,
        4,
        LENGTH_M,
        WIDTH_M,
        THICKNESS_M,
        &sapele,
    );

    // Test with larger mesh (if time permits)
    test_mesh(
        "30x2x5",
        30,
        2,
        5,
        LENGTH_M,
        WIDTH_M,
        THICKNESS_M,
        &sapele,
    );

    println!("\nAnalytical first bending mode (Euler-Bernoulli): ~352 Hz");
}

fn mode_type_prefix(mode_type: &ModeType) -> &'static str {
    match mode_type {
        ModeType::VerticalBending => "V",
        ModeType::Torsional => "T",
        ModeType::Lateral => "L",
        ModeType::Axial => "A",
        ModeType::Unknown => "?",
    }
}

fn test_mesh(
    name: &str,
    nx: usize,
    ny: usize,
    nz: usize,
    length: f64,
    width: f64,
    thickness: f64,
    material: &Material,
) {
    let heights: Vec<f64> = vec![thickness; nx];
    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);
    let num_elements = nx * ny * nz;
    let num_dof = mesh.nodes.len() * 3;

    print!(
        "{} mesh: {} elements, {} nodes, {} DOF... ",
        name, num_elements, mesh.nodes.len(), num_dof
    );

    let start = Instant::now();
    let (freqs, mode_shapes) =
        compute_modal_frequencies_with_shapes(&mesh, material.e, material.nu, material.rho, 4);
    let elapsed = start.elapsed();

    if freqs.is_empty() {
        println!("FAILED (no frequencies)");
    } else {
        println!("done in {:?}", elapsed);
        println!(
            "  Frequencies: [{:.1}, {:.1}, {:.1}, {:.1}] Hz",
            freqs[0],
            freqs.get(1).unwrap_or(&0.0),
            freqs.get(2).unwrap_or(&0.0),
            freqs.get(3).unwrap_or(&0.0)
        );

        // Classify and display modes
        let classified = classify_all_modes(&freqs, &mode_shapes, &mesh.nodes);
        print!("  Classified: ");
        let mut mode_strs: Vec<String> = Vec::new();
        for (mode_type, modes) in &classified {
            for (freq, _, rank) in modes {
                mode_strs.push(format!(
                    "{}{}: {:.1}",
                    mode_type_prefix(mode_type),
                    rank,
                    freq
                ));
            }
        }
        // Sort by frequency for display
        mode_strs.sort_by(|a, b| {
            let freq_a: f64 = a.split(": ").nth(1).unwrap_or("0").parse().unwrap_or(0.0);
            let freq_b: f64 = b.split(": ").nth(1).unwrap_or("0").parse().unwrap_or(0.0);
            freq_a.partial_cmp(&freq_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        println!("{}\n", mode_strs.join(", "));
    }
}
