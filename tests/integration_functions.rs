use fem3d_rust_2::{
    compute_global_matrices_dense, generate_bar_mesh_3d, generate_bar_mesh_3d_adaptive,
    find_corner_nodes,
};

const TOL: f64 = 1e-12;
const SYMMETRY_TOL: f64 = 1e-9;

#[test]
fn adaptive_mesh_respects_supplied_x_positions() {
    const LENGTH: f64 = 2.0;
    const WIDTH: f64 = 1.0;
    const NY: usize = 1;
    const NZ: usize = 1;
    const X_POSITIONS: [f64; 3] = [0.0, 0.5, 2.0];
    const HEIGHTS: [f64; 2] = [0.2, 0.3];
    let mesh = generate_bar_mesh_3d_adaptive(LENGTH, WIDTH, &X_POSITIONS, &HEIGHTS, NY, NZ);

    let mut xs: Vec<f64> = mesh.nodes.iter().map(|n| n.x).collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    xs.dedup();

    assert_eq!(xs.len(), X_POSITIONS.len());
    for (got, expected) in xs.iter().zip(X_POSITIONS.iter()) {
        assert!((*got - *expected).abs() < TOL);
    }
}

#[test]
fn find_corner_nodes_returns_top_end_pair() {
    let mesh = generate_bar_mesh_3d(1.0, 1.0, &[1.0], 1, 1, 1);
    let corners = find_corner_nodes(&mesh.nodes, 1e-9).expect("corner nodes should exist");
    let (s1, s2) = corners;
    let n1 = mesh.nodes[s1];
    let n2 = mesh.nodes[s2];

    assert!((n1.x).abs() < TOL);
    assert!((n2.x).abs() < TOL);
    assert!((n1.z - 1.0).abs() < TOL);
    assert!((n2.z - 1.0).abs() < TOL);
    let ys = [n1.y, n2.y];
    assert!(ys.iter().any(|y| (*y - 1.0).abs() < TOL));
    assert!(ys.iter().any(|y| (*y).abs() < TOL));
}

#[test]
fn global_matrices_are_symmetric_after_dense_assembly() {
    const LENGTH: f64 = 0.1;
    const WIDTH: f64 = 0.05;
    const HEIGHT: f64 = 0.02;
    const YOUNG: f64 = 2.0e9;
    const POISSON: f64 = 0.3;
    const DENSITY: f64 = 1200.0;

    let mesh = generate_bar_mesh_3d(LENGTH, WIDTH, &[HEIGHT], 1, 1, 1);
    let (k, m) = compute_global_matrices_dense(&mesh, YOUNG, POISSON, DENSITY);

    let dof = mesh.nodes.len() * 3;
    assert_eq!(k.nrows(), dof);
    assert_eq!(k.ncols(), dof);
    assert_eq!(m.nrows(), dof);
    assert_eq!(m.ncols(), dof);

    for i in 0..k.nrows() {
        assert!(m[(i, i)] >= 0.0);
    }

    for i in 0..k.nrows() {
        for j in 0..k.ncols() {
            let kd = (k[(i, j)] - k[(j, i)]).abs();
            let md = (m[(i, j)] - m[(j, i)]).abs();
            assert!(kd < SYMMETRY_TOL);
            assert!(md < SYMMETRY_TOL);
        }
    }
}
