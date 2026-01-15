use fem3d_rust_2::{compute_global_matrices_dense, generate_bar_mesh_3d, Material};
use nalgebra::{Matrix3, SMatrix};

type Matrix3x8 = SMatrix<f64, 3, 8>;
type NodeCoords = SMatrix<f64, 8, 3>;

const GAUSS_G: f64 = 0.577_350_269_189_625_8;

fn shape_function_derivatives_hex8(xi: f64, eta: f64, zeta: f64) -> Matrix3x8 {
    let mut d_n = Matrix3x8::zeros();

    d_n[(0, 0)] = -0.125 * (1.0 - eta) * (1.0 - zeta);
    d_n[(0, 1)] = 0.125 * (1.0 - eta) * (1.0 - zeta);
    d_n[(0, 2)] = 0.125 * (1.0 + eta) * (1.0 - zeta);
    d_n[(0, 3)] = -0.125 * (1.0 + eta) * (1.0 - zeta);
    d_n[(0, 4)] = -0.125 * (1.0 - eta) * (1.0 + zeta);
    d_n[(0, 5)] = 0.125 * (1.0 - eta) * (1.0 + zeta);
    d_n[(0, 6)] = 0.125 * (1.0 + eta) * (1.0 + zeta);
    d_n[(0, 7)] = -0.125 * (1.0 + eta) * (1.0 + zeta);

    d_n[(1, 0)] = -0.125 * (1.0 - xi) * (1.0 - zeta);
    d_n[(1, 1)] = -0.125 * (1.0 + xi) * (1.0 - zeta);
    d_n[(1, 2)] = 0.125 * (1.0 + xi) * (1.0 - zeta);
    d_n[(1, 3)] = 0.125 * (1.0 - xi) * (1.0 - zeta);
    d_n[(1, 4)] = -0.125 * (1.0 - xi) * (1.0 + zeta);
    d_n[(1, 5)] = -0.125 * (1.0 + xi) * (1.0 + zeta);
    d_n[(1, 6)] = 0.125 * (1.0 + xi) * (1.0 + zeta);
    d_n[(1, 7)] = 0.125 * (1.0 - xi) * (1.0 + zeta);

    d_n[(2, 0)] = -0.125 * (1.0 - xi) * (1.0 - eta);
    d_n[(2, 1)] = -0.125 * (1.0 + xi) * (1.0 - eta);
    d_n[(2, 2)] = -0.125 * (1.0 + xi) * (1.0 + eta);
    d_n[(2, 3)] = -0.125 * (1.0 - xi) * (1.0 + eta);
    d_n[(2, 4)] = 0.125 * (1.0 - xi) * (1.0 - eta);
    d_n[(2, 5)] = 0.125 * (1.0 + xi) * (1.0 - eta);
    d_n[(2, 6)] = 0.125 * (1.0 + xi) * (1.0 + eta);
    d_n[(2, 7)] = 0.125 * (1.0 - xi) * (1.0 + eta);

    d_n
}

fn check_element_jacobians(mesh: &fem3d_rust_2::Mesh3d) -> (usize, usize, usize) {
    let gauss_points = [
        (-GAUSS_G, -GAUSS_G, -GAUSS_G),
        (GAUSS_G, -GAUSS_G, -GAUSS_G),
        (GAUSS_G, GAUSS_G, -GAUSS_G),
        (-GAUSS_G, GAUSS_G, -GAUSS_G),
        (-GAUSS_G, -GAUSS_G, GAUSS_G),
        (GAUSS_G, -GAUSS_G, GAUSS_G),
        (GAUSS_G, GAUSS_G, GAUSS_G),
        (-GAUSS_G, GAUSS_G, GAUSS_G),
    ];

    let mut total_positive = 0;
    let mut total_negative = 0;
    let mut total_zero = 0;

    for (elem_idx, nodes) in mesh.elements.iter().enumerate() {
        let mut coords = NodeCoords::zeros();
        for (i, &node_idx) in nodes.iter().enumerate() {
            let node = mesh.nodes[node_idx];
            coords[(i, 0)] = node.x;
            coords[(i, 1)] = node.y;
            coords[(i, 2)] = node.z;
        }

        for (gp_idx, &(xi, eta, zeta)) in gauss_points.iter().enumerate() {
            let d_n_nat = shape_function_derivatives_hex8(xi, eta, zeta);
            let j: Matrix3<f64> = d_n_nat * coords;
            let det_j = j.determinant();

            if det_j > 1e-12 {
                total_positive += 1;
            } else if det_j < -1e-12 {
                total_negative += 1;
                println!(
                    "Element {} GP {}: NEGATIVE det_j = {:.6e}",
                    elem_idx, gp_idx, det_j
                );
            } else {
                total_zero += 1;
                println!(
                    "Element {} GP {}: NEAR-ZERO det_j = {:.6e}",
                    elem_idx, gp_idx, det_j
                );
            }
        }
    }

    (total_positive, total_negative, total_zero)
}

fn main() {
    println!("=== Undercut Mesh Jacobian Diagnostic ===\n");

    // Test 1: Uniform bar
    println!("Test 1: Uniform bar (no undercut)");
    let heights: Vec<f64> = vec![0.024; 10];
    let mesh = generate_bar_mesh_3d(0.551, 0.032, &heights, 10, 2, 2);
    let (pos, neg, zero) = check_element_jacobians(&mesh);
    println!(
        "  {} elements, {} Gauss points total",
        mesh.elements.len(),
        mesh.elements.len() * 8
    );
    println!(
        "  Positive: {}, Negative: {}, Zero: {}\n",
        pos, neg, zero
    );

    // Test 2: Linear taper undercut
    println!("Test 2: Linear taper (simulating undercut)");
    let heights: Vec<f64> = (0..10)
        .map(|i| 0.024 - (i as f64) * 0.001)
        .collect();
    println!("  Heights: {:?}", heights);
    let mesh = generate_bar_mesh_3d(0.551, 0.032, &heights, 10, 2, 2);
    let (pos, neg, zero) = check_element_jacobians(&mesh);
    println!(
        "  {} elements, {} Gauss points total",
        mesh.elements.len(),
        mesh.elements.len() * 8
    );
    println!(
        "  Positive: {}, Negative: {}, Zero: {}\n",
        pos, neg, zero
    );

    // Test 3: Symmetric undercut (like a marimba bar)
    println!("Test 3: Symmetric undercut (marimba bar style)");
    let heights: Vec<f64> = (0..20)
        .map(|i| {
            let center_dist = (i as f64 - 9.5).abs() / 9.5;
            let undercut_depth = 0.012 * (1.0 - center_dist.powi(2)); // Parabolic undercut
            0.024 - undercut_depth
        })
        .collect();
    println!("  Heights: {:?}", &heights[..5]);
    println!("  ... (middle) ...");
    println!("  Heights: {:?}", &heights[15..]);
    let mesh = generate_bar_mesh_3d(0.551, 0.032, &heights, 20, 2, 3);
    let (pos, neg, zero) = check_element_jacobians(&mesh);
    println!(
        "  {} elements, {} Gauss points total",
        mesh.elements.len(),
        mesh.elements.len() * 8
    );
    println!(
        "  Positive: {}, Negative: {}, Zero: {}\n",
        pos, neg, zero
    );

    // Test 4: Abrupt height change
    println!("Test 4: Abrupt height change (stress test)");
    let mut heights: Vec<f64> = vec![0.024; 10];
    heights[5] = 0.012; // Sudden drop
    heights[6] = 0.012;
    println!("  Heights: {:?}", heights);
    let mesh = generate_bar_mesh_3d(0.551, 0.032, &heights, 10, 2, 2);
    let (pos, neg, zero) = check_element_jacobians(&mesh);
    println!(
        "  {} elements, {} Gauss points total",
        mesh.elements.len(),
        mesh.elements.len() * 8
    );
    println!(
        "  Positive: {}, Negative: {}, Zero: {}\n",
        pos, neg, zero
    );

    // Test 5: Check matrix conditioning
    println!("Test 5: Check matrix properties for abrupt undercut");
    let sapele = Material::sapele();
    let (k, m) = compute_global_matrices_dense(&mesh, sapele.e, sapele.nu, sapele.rho);

    let k_diag_sum: f64 = (0..k.nrows()).map(|i| k[(i, i)]).sum();
    let m_diag_sum: f64 = (0..m.nrows()).map(|i| m[(i, i)]).sum();
    let k_diag_neg: usize = (0..k.nrows()).filter(|&i| k[(i, i)] < 0.0).count();
    let m_diag_neg: usize = (0..m.nrows()).filter(|&i| m[(i, i)] < 0.0).count();

    println!("  K diagonal sum: {:.6e}", k_diag_sum);
    println!("  M diagonal sum: {:.6e}", m_diag_sum);
    println!("  K negative diagonal entries: {}", k_diag_neg);
    println!("  M negative diagonal entries: {}", m_diag_neg);

    if k_diag_neg > 0 || m_diag_neg > 0 {
        println!("  WARNING: Negative diagonal entries detected!");
    } else {
        println!("  OK: All diagonal entries non-negative");
    }
}
