use fem3d_rust_2::{compute_modal_frequencies, generate_bar_mesh_3d};
use std::time::Instant;

fn main() {
    const LENGTH_M: f64 = 0.551;
    const WIDTH_M: f64 = 0.032;
    const THICKNESS_M: f64 = 0.024;
    const YOUNG: f64 = 12.0e9;
    const POISSON: f64 = 0.35;
    const DENSITY: f64 = 640.0;

    println!("Sapele bar: {}m x {}m x {}m", LENGTH_M, WIDTH_M, THICKNESS_M);
    println!("E={:.0e} Pa, nu={}, rho={} kg/m³\n", YOUNG, POISSON, DENSITY);

    // Test with 1x1x1 mesh (same as current test)
    test_mesh("1x1x1", 1, 1, 1, LENGTH_M, WIDTH_M, THICKNESS_M, YOUNG, POISSON, DENSITY);

    // Test with 5x2x2 mesh
    test_mesh("5x2x2", 5, 2, 2, LENGTH_M, WIDTH_M, THICKNESS_M, YOUNG, POISSON, DENSITY);

    // Test with 10x2x3 mesh
    test_mesh("10x2x3", 10, 2, 3, LENGTH_M, WIDTH_M, THICKNESS_M, YOUNG, POISSON, DENSITY);

    // Test with 20x2x4 mesh
    test_mesh("20x2x4", 20, 2, 4, LENGTH_M, WIDTH_M, THICKNESS_M, YOUNG, POISSON, DENSITY);

    // Test with larger mesh (if time permits)
    test_mesh("30x2x5", 30, 2, 5, LENGTH_M, WIDTH_M, THICKNESS_M, YOUNG, POISSON, DENSITY);

    println!("\nAnalytical first bending mode (Euler-Bernoulli): ~352 Hz");
}

fn test_mesh(
    name: &str,
    nx: usize,
    ny: usize,
    nz: usize,
    length: f64,
    width: f64,
    thickness: f64,
    e: f64,
    nu: f64,
    rho: f64,
) {
    let heights: Vec<f64> = vec![thickness; nx];
    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);
    let num_elements = nx * ny * nz;
    let num_dof = mesh.nodes.len() * 3;

    print!("{} mesh: {} elements, {} nodes, {} DOF... ", name, num_elements, mesh.nodes.len(), num_dof);

    let start = Instant::now();
    let freqs = compute_modal_frequencies(&mesh, e, nu, rho, 4);
    let elapsed = start.elapsed();

    if freqs.is_empty() {
        println!("FAILED (no frequencies)");
    } else {
        println!("done in {:?}", elapsed);
        println!("  Frequencies: [{:.1}, {:.1}, {:.1}, {:.1}] Hz\n",
            freqs[0], freqs.get(1).unwrap_or(&0.0),
            freqs.get(2).unwrap_or(&0.0), freqs.get(3).unwrap_or(&0.0));
    }
}
