//! Objective function and penalties for optimization.

use crate::cuts::{genes_to_cuts, Cut};

use super::types::{DetailedEvaluation, PenaltyType};

/// Compute the weighted squared tuning error (modified Eq. 7 from paper).
///
/// epsilon = 100 * sum(w_m * ((omega_bar_m - omega_star_m) / omega_star_m)²) / sum(w_m)
///
/// With f1_priority > 1, f1 is weighted more heavily than higher modes.
pub fn compute_tuning_error(
    computed_freq: &[f64],
    target_freq: &[f64],
    f1_priority: f64,
) -> f64 {
    let m = computed_freq.len().min(target_freq.len());
    if m == 0 {
        return f64::INFINITY;
    }

    let mut weighted_sum_squared_error = 0.0;
    let mut total_weight = 0.0;

    for i in 0..m {
        let computed = computed_freq[i];
        let target = target_freq[i];

        if target == 0.0 {
            continue;
        }

        // Weight: f1 gets f1_priority, other modes get 1
        let weight = if i == 0 { f1_priority } else { 1.0 };

        let relative_error = (computed - target) / target;
        weighted_sum_squared_error += weight * relative_error * relative_error;
        total_weight += weight;
    }

    if total_weight > 0.0 {
        100.0 * (weighted_sum_squared_error / total_weight)
    } else {
        f64::INFINITY
    }
}

/// Compute the maximum squared error (Eq. 8 from paper).
///
/// epsilon = 100 * max((omega_bar_m - omega_star_m) / omega_star_m)²
pub fn compute_max_tuning_error(computed_freq: &[f64], target_freq: &[f64]) -> f64 {
    let m = computed_freq.len().min(target_freq.len());
    if m == 0 {
        return f64::INFINITY;
    }

    let mut max_squared_error = 0.0;

    for i in 0..m {
        let computed = computed_freq[i];
        let target = target_freq[i];

        if target == 0.0 {
            continue;
        }

        let relative_error = (computed - target) / target;
        let squared_error = relative_error * relative_error;

        if squared_error > max_squared_error {
            max_squared_error = squared_error;
        }
    }

    100.0 * max_squared_error
}

/// Compute the volumetric penalty (Eq. 10 from paper).
///
/// V = 100 * (2 / h0*L) * integral(h0 - H(x)) dx
///
/// This represents the percentage of volume extracted from the bar.
pub fn compute_volume_penalty(cuts: &[Cut], length: f64, h0: f64) -> f64 {
    if cuts.is_empty() {
        return 0.0;
    }

    // Sort cuts by lambda (descending - largest first)
    let mut sorted_cuts = cuts.to_vec();
    sorted_cuts.sort_by(|a, b| b.lambda.partial_cmp(&a.lambda).unwrap_or(std::cmp::Ordering::Equal));

    // Original volume (half bar, since symmetric)
    let half_bar_volume = (length / 2.0) * h0;

    // Calculate extracted volume from symmetric undercuts
    let mut extracted_volume = 0.0;
    let mut prev_lambda = 0.0;

    // Process cuts from innermost (smallest lambda) to outermost (largest lambda)
    for cut in sorted_cuts.iter().rev() {
        if cut.lambda > prev_lambda && cut.lambda > 0.0 {
            // Width of this cut region
            let width = cut.lambda - prev_lambda;

            // Height removed in this region
            let height_removed = h0 - cut.h;

            // Volume removed (just one side)
            extracted_volume += width * height_removed;

            prev_lambda = cut.lambda;
        }
    }

    // Calculate percentage
    let volume_percent = 100.0 * (extracted_volume / half_bar_volume);

    volume_percent.clamp(0.0, 100.0)
}

/// Compute the roughness penalty (Eq. 12 from paper).
///
/// S = 100 * (1/N) * sum(|h_{n-1} - h_n| / h0)
///
/// This represents the average height change per discontinuity.
pub fn compute_roughness_penalty(cuts: &[Cut], h0: f64) -> f64 {
    if cuts.is_empty() {
        return 0.0;
    }

    // Sort cuts by lambda (descending - largest first)
    let mut sorted_cuts = cuts.to_vec();
    sorted_cuts.sort_by(|a, b| b.lambda.partial_cmp(&a.lambda).unwrap_or(std::cmp::Ordering::Equal));

    let mut sum_height_changes = 0.0;
    let mut num_discontinuities = 0;

    // First discontinuity: from h0 to first (outermost) cut
    if !sorted_cuts.is_empty() {
        sum_height_changes += (h0 - sorted_cuts[0].h).abs();
        num_discontinuities += 1;
    }

    // Discontinuities between consecutive cuts
    for i in 1..sorted_cuts.len() {
        // Only count if this cut is actually visible
        if sorted_cuts[i].lambda < sorted_cuts[i - 1].lambda {
            sum_height_changes += (sorted_cuts[i - 1].h - sorted_cuts[i].h).abs();
            num_discontinuities += 1;
        }
    }

    if num_discontinuities == 0 {
        return 0.0;
    }

    // Calculate average roughness as percentage
    100.0 * (sum_height_changes / (num_discontinuities as f64 * h0))
}

/// Combined objective function with volumetric penalty (Eq. 11).
///
/// E = (1 - alpha) * epsilon + alpha * V
pub fn combined_objective_volume(tuning_error: f64, volume_penalty: f64, alpha: f64) -> f64 {
    (1.0 - alpha) * tuning_error + alpha * volume_penalty
}

/// Combined objective function with roughness penalty (Eq. 13).
///
/// E = (1 - alpha) * epsilon + alpha * S
pub fn combined_objective_roughness(tuning_error: f64, roughness_penalty: f64, alpha: f64) -> f64 {
    (1.0 - alpha) * tuning_error + alpha * roughness_penalty
}

/// Compute cents error for a single frequency.
pub fn compute_cents_error(computed: f64, target: f64) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    1200.0 * (computed / target).log2()
}

/// Compute cents errors for all frequencies.
pub fn compute_cents_errors(computed_freq: &[f64], target_freq: &[f64]) -> Vec<f64> {
    computed_freq
        .iter()
        .zip(target_freq.iter())
        .map(|(&comp, &target)| compute_cents_error(comp, target))
        .collect()
}

/// Evaluate fitness of an individual.
pub fn evaluate_fitness(
    computed_freq: &[f64],
    target_freq: &[f64],
    genes: &[f64],
    length: f64,
    h0: f64,
    penalty_type: PenaltyType,
    alpha: f64,
    f1_priority: f64,
    num_cuts: usize,
) -> f64 {
    let tuning_error = compute_tuning_error(computed_freq, target_freq, f1_priority);

    if penalty_type == PenaltyType::None || alpha == 0.0 {
        return tuning_error;
    }

    let cuts = genes_to_cuts(&genes[..num_cuts * 2]);

    match penalty_type {
        PenaltyType::Volume => {
            let volume_penalty = compute_volume_penalty(&cuts, length, h0);
            combined_objective_volume(tuning_error, volume_penalty, alpha)
        }
        PenaltyType::Roughness => {
            let roughness_penalty = compute_roughness_penalty(&cuts, h0);
            combined_objective_roughness(tuning_error, roughness_penalty, alpha)
        }
        PenaltyType::None => tuning_error,
    }
}

/// Get detailed evaluation results for an individual.
pub fn evaluate_detailed(
    computed_frequencies: &[f64],
    target_frequencies: &[f64],
    genes: &[f64],
    length: f64,
    h0: f64,
    penalty_type: PenaltyType,
    alpha: f64,
    num_cuts: usize,
) -> DetailedEvaluation {
    let cuts = genes_to_cuts(&genes[..num_cuts * 2]);

    let tuning_error = compute_tuning_error(computed_frequencies, target_frequencies, 1.0);
    let volume_penalty = compute_volume_penalty(&cuts, length, h0);
    let roughness_penalty = compute_roughness_penalty(&cuts, h0);

    let combined_fitness = match penalty_type {
        PenaltyType::None => tuning_error,
        PenaltyType::Volume if alpha > 0.0 => {
            combined_objective_volume(tuning_error, volume_penalty, alpha)
        }
        PenaltyType::Roughness if alpha > 0.0 => {
            combined_objective_roughness(tuning_error, roughness_penalty, alpha)
        }
        _ => tuning_error,
    };

    let cents_errors = compute_cents_errors(computed_frequencies, target_frequencies);
    let max_cents_error = cents_errors
        .iter()
        .map(|e| e.abs())
        .fold(0.0, f64::max);

    DetailedEvaluation {
        computed_frequencies: computed_frequencies.to_vec(),
        target_frequencies: target_frequencies.to_vec(),
        tuning_error,
        volume_penalty,
        roughness_penalty,
        combined_fitness,
        cents_errors,
        max_cents_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning_error_perfect_match() {
        let computed = vec![100.0, 200.0, 300.0];
        let target = vec![100.0, 200.0, 300.0];

        let error = compute_tuning_error(&computed, &target, 1.0);
        assert!(error < 1e-10, "Perfect match should have zero error");
    }

    #[test]
    fn tuning_error_with_mismatch() {
        let computed = vec![105.0, 200.0]; // 5% error on f1
        let target = vec![100.0, 200.0];

        let error = compute_tuning_error(&computed, &target, 1.0);
        // (0.05)² * 100 / 2 = 0.125 for equal weighting
        assert!((error - 0.125).abs() < 0.001);
    }

    #[test]
    fn f1_priority_increases_f1_weight() {
        // Use different error magnitudes: 5% on f1, 10% on f2
        let computed = vec![105.0, 220.0];
        let target = vec![100.0, 200.0];

        let error_normal = compute_tuning_error(&computed, &target, 1.0);
        let error_priority = compute_tuning_error(&computed, &target, 2.0);

        // With f1 priority = 2.0, f1 error is weighted more, so overall error
        // should be different than with equal weighting (priority = 1.0)
        // f1_error = 5%, f2_error = 10%
        // Normal: (0.05² + 0.10²) / 2 * 100 = (0.0025 + 0.01) / 2 * 100 = 0.625
        // Priority: (2*0.05² + 1*0.10²) / 3 * 100 = (0.005 + 0.01) / 3 * 100 = 0.5
        assert!(
            (error_normal - error_priority).abs() > 0.001,
            "Errors should differ: normal={}, priority={}",
            error_normal,
            error_priority
        );
    }

    #[test]
    fn volume_penalty_no_cuts() {
        let cuts: Vec<Cut> = vec![];
        let penalty = compute_volume_penalty(&cuts, 0.5, 0.024);
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn volume_penalty_with_cut() {
        let cuts = vec![Cut::new(0.1, 0.012)]; // Half height cut
        let length = 0.5;
        let h0 = 0.024;

        let penalty = compute_volume_penalty(&cuts, length, h0);

        // Expected: 0.1 * 0.012 / (0.25 * 0.024) * 100 = 20%
        assert!(penalty > 15.0 && penalty < 25.0);
    }

    #[test]
    fn roughness_penalty_single_cut() {
        let cuts = vec![Cut::new(0.1, 0.012)];
        let h0 = 0.024;

        let penalty = compute_roughness_penalty(&cuts, h0);

        // Expected: |0.024 - 0.012| / 0.024 * 100 = 50%
        assert!((penalty - 50.0).abs() < 1.0);
    }

    #[test]
    fn cents_error_calculation() {
        // Perfect octave should be 1200 cents
        let cents = compute_cents_error(200.0, 100.0);
        assert!((cents - 1200.0).abs() < 0.1);

        // Same frequency should be 0 cents
        let cents = compute_cents_error(100.0, 100.0);
        assert!(cents.abs() < 0.1);
    }

    #[test]
    fn combined_objectives() {
        let tuning = 1.0;
        let penalty = 10.0;
        let alpha = 0.5;

        let combined_vol = combined_objective_volume(tuning, penalty, alpha);
        assert!((combined_vol - 5.5).abs() < 0.01);

        let combined_rough = combined_objective_roughness(tuning, penalty, alpha);
        assert!((combined_rough - 5.5).abs() < 0.01);
    }
}
