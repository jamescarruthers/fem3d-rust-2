use fem3d_rust_2::{compute_modal_frequencies, generate_bar_mesh_3d, LAMBDA_TOL};

#[test]
fn sapele_bar_modal_frequencies_are_positive_and_sorted() {
    const LENGTH_M: f64 = 0.551; // 551 mm
    const WIDTH_M: f64 = 0.032; // 32 mm
    const THICKNESS_M: f64 = 0.024; // 24 mm
    const DIVISIONS: (usize, usize, usize) = (1, 1, 1);
    const MAX_REASONABLE_FREQ_HZ: f64 = 1.0e6;
    const YOUNG_SAPELE: f64 = 12.0e9;
    const POISSON_SAPELE: f64 = 0.35;
    const DENSITY_SAPELE: f64 = 640.0;
    const HEIGHTS: [f64; 1] = [THICKNESS_M];

    let mesh = generate_bar_mesh_3d(
        LENGTH_M,
        WIDTH_M,
        &HEIGHTS,
        DIVISIONS.0,
        DIVISIONS.1,
        DIVISIONS.2,
    );
    let freqs = compute_modal_frequencies(&mesh, YOUNG_SAPELE, POISSON_SAPELE, DENSITY_SAPELE, 4);

    assert!(!freqs.is_empty());
    for window in freqs.windows(2) {
        assert!(window[0] <= window[1] + LAMBDA_TOL);
    }
    for f in freqs {
        assert!(f.is_finite());
        assert!(f > 0.0);
        assert!(f < MAX_REASONABLE_FREQ_HZ);
    }
}
