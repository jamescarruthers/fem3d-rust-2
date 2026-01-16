# Percussion Bar Tuning Guide

A practical guide to tuning ratios, target frequencies, and design considerations for percussion instruments.

## Table of Contents

1. [Tuning Ratios](#tuning-ratios)
2. [Target Frequency Selection](#target-frequency-selection)
3. [Bar Dimensions](#bar-dimensions)
4. [Material Selection](#material-selection)
5. [Practical Examples](#practical-examples)

---

## Tuning Ratios

Percussion bars are tuned so that higher partials (overtones) are harmonically related to the fundamental frequency. This creates a pleasing, musical tone.

### Common Tuning Ratios

| Ratio | Interval | Description | Use Case |
|-------|----------|-------------|----------|
| 1:4 | Double octave | f2 = 4 × f1 | Vibraphones, simple marimbas |
| 1:4:10 | Double octave + Major 10th | f2 = 4×f1, f3 = 10×f1 | Standard marimbas |
| 1:4:9 | Double octave + Major 9th | f2 = 4×f1, f3 = 9×f1 | Alternative marimba tuning |
| 1:3:9 | Octave+5th + Triple octave | f2 = 3×f1, f3 = 9×f1 | Experimental |
| 1:4:12 | Double octave + Octave+5th | f2 = 4×f1, f3 = 12×f1 | Concert xylophones |

### Musical Intervals Reference

| Frequency Ratio | Cents | Interval Name |
|-----------------|-------|---------------|
| 1:1 | 0 | Unison |
| 2:1 | 1200 | Octave |
| 3:1 | 1902 | Octave + Perfect 5th |
| 4:1 | 2400 | Double octave |
| 5:1 | 2786 | Double octave + Major 3rd |
| 6:1 | 3102 | Double octave + Perfect 5th |
| 8:1 | 3600 | Triple octave |
| 9:1 | 3804 | Triple octave + Major 2nd |
| 10:1 | 3986 | Triple octave + Major 3rd |

### Defining Target Frequencies

```rust
use fem3d_rust_2::*;

// Example: A4 = 440 Hz with 1:4:10 tuning
fn marimba_targets(fundamental: f64) -> Vec<f64> {
    vec![
        fundamental,          // f1: fundamental
        fundamental * 4.0,    // f2: double octave
        fundamental * 10.0,   // f3: major 10th above double octave
    ]
}

// Example: 1:4 tuning for vibraphone
fn vibraphone_targets(fundamental: f64) -> Vec<f64> {
    vec![
        fundamental,          // f1: fundamental
        fundamental * 4.0,    // f2: double octave
    ]
}

// Example: 1:4:9 alternative tuning
fn alternative_targets(fundamental: f64) -> Vec<f64> {
    vec![
        fundamental,          // f1
        fundamental * 4.0,    // f2
        fundamental * 9.0,    // f3
    ]
}
```

---

## Target Frequency Selection

### Standard Pitch (A4 = 440 Hz)

| Note | Octave 3 | Octave 4 | Octave 5 | Octave 6 |
|------|----------|----------|----------|----------|
| C | 130.81 | 261.63 | 523.25 | 1046.50 |
| C# | 138.59 | 277.18 | 554.37 | 1108.73 |
| D | 146.83 | 293.66 | 587.33 | 1174.66 |
| D# | 155.56 | 311.13 | 622.25 | 1244.51 |
| E | 164.81 | 329.63 | 659.25 | 1318.51 |
| F | 174.61 | 349.23 | 698.46 | 1396.91 |
| F# | 185.00 | 369.99 | 739.99 | 1479.98 |
| G | 196.00 | 392.00 | 783.99 | 1567.98 |
| G# | 207.65 | 415.30 | 830.61 | 1661.22 |
| A | 220.00 | 440.00 | 880.00 | 1760.00 |
| A# | 233.08 | 466.16 | 932.33 | 1864.66 |
| B | 246.94 | 493.88 | 987.77 | 1975.53 |

### Calculating Any Note

```rust
// Calculate frequency from MIDI note number
fn midi_to_freq(midi_note: u8) -> f64 {
    440.0 * 2.0_f64.powf((midi_note as f64 - 69.0) / 12.0)
}

// Calculate frequency from note name
fn note_to_freq(note: &str, octave: i32) -> f64 {
    let semitones = match note.to_uppercase().as_str() {
        "C" => 0, "C#" | "DB" => 1, "D" => 2, "D#" | "EB" => 3,
        "E" => 4, "F" => 5, "F#" | "GB" => 6, "G" => 7,
        "G#" | "AB" => 8, "A" => 9, "A#" | "BB" => 10, "B" => 11,
        _ => 0,
    };
    let midi = 12 * (octave + 1) + semitones;
    midi_to_freq(midi as u8)
}

// Examples
let a4 = note_to_freq("A", 4);   // 440.0 Hz
let c4 = note_to_freq("C", 4);   // 261.63 Hz
let f3 = note_to_freq("F", 3);   // 174.61 Hz
```

---

## Bar Dimensions

### Fundamental Frequency Relationship

For a uniform rectangular bar:

```
f1 ∝ h / L²
```

Where:
- `f1` = fundamental frequency
- `h` = bar thickness
- `L` = bar length

### Estimating Bar Length

For a given fundamental frequency:

```rust
// Approximate bar length from frequency
// Based on Euler-Bernoulli beam theory
fn estimate_bar_length(
    f1: f64,      // Target fundamental (Hz)
    h: f64,       // Thickness (m)
    e: f64,       // Young's modulus (Pa)
    rho: f64,     // Density (kg/m³)
) -> f64 {
    // λ₁ ≈ 4.73 for free-free beam, first mode
    let lambda1 = 4.730;
    let c = (e / rho).sqrt();

    // f = (λ²h)/(2πL²) × √(E/12ρ)
    // L² = (λ²h)/(2πf) × √(E/12ρ)
    let l_squared = (lambda1 * lambda1 * h) / (2.0 * std::f64::consts::PI * f1)
        * (e / (12.0 * rho)).sqrt();

    l_squared.sqrt()
}
```

### Typical Dimensions by Instrument

| Instrument | Length Range | Width | Thickness | Material |
|------------|--------------|-------|-----------|----------|
| Marimba (bass) | 400-500mm | 60-75mm | 22-28mm | Rosewood, Padauk |
| Marimba (treble) | 150-250mm | 35-50mm | 18-22mm | Rosewood, Padauk |
| Vibraphone | 200-350mm | 40-60mm | 8-12mm | Aluminum |
| Xylophone | 150-300mm | 30-40mm | 15-20mm | Rosewood, Synthetic |
| Glockenspiel | 80-200mm | 20-30mm | 6-10mm | Steel, Bronze |

### Dimension Scaling

When designing a range of bars:

```rust
// Scale bar length for different notes
fn scale_length(base_length: f64, base_freq: f64, target_freq: f64) -> f64 {
    base_length * (base_freq / target_freq).sqrt()
}

// Example: Full octave from C4
let c4_length = 0.35;  // 350mm reference
let c4_freq = 261.63;

let lengths: Vec<f64> = [
    261.63, 293.66, 329.63, 349.23,  // C4, D4, E4, F4
    392.00, 440.00, 493.88, 523.25,  // G4, A4, B4, C5
].iter()
    .map(|&f| scale_length(c4_length, c4_freq, f))
    .collect();
```

---

## Material Selection

### Comparison Table

| Material | E (GPa) | ρ (kg/m³) | E/ρ | Sound Character |
|----------|---------|-----------|-----|-----------------|
| **Honduran Rosewood** | 12.5 | 850 | 14,700 | Warm, rich, complex |
| **Sapele** | 12.0 | 640 | 18,750 | Bright, clear |
| **Padauk** | 11.7 | 750 | 15,600 | Warm, mellow |
| **Bubinga** | 15.8 | 890 | 17,750 | Bright, focused |
| **Aluminum 6061** | 68.9 | 2700 | 25,500 | Bright, sustained |
| **Brass** | 110.0 | 8530 | 12,900 | Warm, bell-like |
| **Steel** | 205.0 | 7870 | 26,100 | Bright, piercing |

### E/ρ Ratio Significance

Higher E/ρ ratio means:
- Higher frequency for same dimensions
- Shorter bars needed for same pitch
- Generally brighter sound

### Material Selection Guide

```rust
use fem3d_rust_2::Material;

// For warm, traditional marimba sound
let premium_marimba = Material::rosewood();

// For bright, affordable marimba
let standard_marimba = Material::padauk();

// For sustained vibraphone tone
let vibraphone = Material::aluminum();

// For bell-like metallophone
let metallophone = Material::bell_bronze();
```

---

## Practical Examples

### Example 1: Marimba Bar (A3 = 220 Hz)

```rust
use fem3d_rust_2::*;

fn optimize_marimba_a3() -> OptimizationResult {
    // Target: A3 with 1:4:10 tuning
    let f1 = 220.0;
    let targets = vec![f1, f1 * 4.0, f1 * 10.0];  // 220, 880, 2200 Hz

    // Estimate initial dimensions
    let bar = BarParameters::new(
        0.42,   // ~420mm length
        0.065,  // 65mm width
        0.024,  // 24mm thickness
        0.006,  // 6mm minimum (for deep undercut)
    );

    let material = Material::rosewood();

    let config = HybridConfig::new(bar, material, targets, 2)
        .with_strategy(OptimizationStrategy::Hybrid2Dto3D {
            ea_params: EAParameters {
                population_size: 40,
                max_generations: 60,
                target_error: 0.005,
                ..Default::default()
            },
            surrogate_config: SurrogateConfig {
                initial_samples: 15,
                max_evaluations: 60,
                convergence_tol: 0.01,
                ..Default::default()
            },
            frequency_correction: 1.0,
        });

    run_hybrid_optimization(&config).into()
}
```

### Example 2: Vibraphone Bar (F4 = 349 Hz)

```rust
use fem3d_rust_2::*;

fn optimize_vibraphone_f4() -> OptimizationResult {
    // Target: F4 with 1:4 tuning
    let f1 = 349.23;
    let targets = vec![f1, f1 * 4.0];  // 349, 1397 Hz

    let bar = BarParameters::new(
        0.28,   // ~280mm length
        0.045,  // 45mm width
        0.010,  // 10mm thickness
        0.003,  // 3mm minimum
    );

    let material = Material::aluminum();

    let mut params = EAParameters::default();
    params.population_size = 30;
    params.max_generations = 80;
    params.target_error = 0.003;  // Tighter tolerance for aluminum
    params.analysis_mode = AnalysisMode::Beam2D;

    let config = EAConfig::new(bar, material, targets, 1)  // Single cut
        .with_params(params);

    run_optimization(&config)
}
```

### Example 3: Full Marimba Octave

```rust
use fem3d_rust_2::*;

fn optimize_marimba_octave() {
    let material = Material::sapele();
    let base_length = 0.35;  // C4 reference length

    // C4 to C5
    let notes = [
        ("C4", 261.63), ("C#4", 277.18), ("D4", 293.66), ("D#4", 311.13),
        ("E4", 329.63), ("F4", 349.23), ("F#4", 369.99), ("G4", 392.00),
        ("G#4", 415.30), ("A4", 440.00), ("A#4", 466.16), ("B4", 493.88),
        ("C5", 523.25),
    ];

    println!("Marimba Octave Optimization Results");
    println!("====================================\n");

    for (name, f1) in &notes {
        // Scale length by frequency
        let length = base_length * (261.63 / f1).sqrt();

        let bar = BarParameters::new(
            length,
            0.05,   // 50mm width
            0.022,  // 22mm thickness
            0.005,
        );

        // 1:4:10 tuning
        let targets = vec![*f1, f1 * 4.0, f1 * 10.0];

        let mut params = EAParameters::default();
        params.population_size = 30;
        params.max_generations = 40;
        params.analysis_mode = AnalysisMode::Beam2D;

        let config = EAConfig::new(bar, material.clone(), targets, 2)
            .with_params(params);

        let result = run_optimization(&config);

        println!("{}: L={:.1}mm, max_err={:.1}ct",
            name,
            length * 1000.0,
            result.max_error_cents
        );

        for (i, cut) in result.cuts.iter().enumerate() {
            println!("  Cut {}: λ={:.1}mm, h={:.1}mm",
                i + 1,
                cut.lambda * 1000.0,
                cut.h * 1000.0
            );
        }
        println!();
    }
}
```

### Example 4: Comparing Number of Cuts

```rust
use fem3d_rust_2::*;

fn compare_cuts() {
    let bar = BarParameters::new(0.35, 0.05, 0.024, 0.005);
    let material = Material::sapele();
    let targets = vec![261.63, 1046.52];  // C4, 1:4 ratio

    println!("Comparing optimization with different numbers of cuts:\n");

    for num_cuts in 1..=3 {
        let mut params = EAParameters::default();
        params.population_size = 30;
        params.max_generations = 50;
        params.analysis_mode = AnalysisMode::Beam2D;

        let config = EAConfig::new(
            bar.clone(),
            material.clone(),
            targets.clone(),
            num_cuts
        ).with_params(params);

        let result = run_optimization(&config);

        println!("{} cut(s):", num_cuts);
        println!("  Max error: {:.1} cents", result.max_error_cents);
        println!("  Volume removed: {:.1}%", result.volume_percent);
        println!("  Generations: {}", result.generations);
        for (i, cut) in result.cuts.iter().enumerate() {
            println!("    Cut {}: λ={:.1}mm, h={:.1}mm",
                i + 1, cut.lambda * 1000.0, cut.h * 1000.0);
        }
        println!();
    }
}
```

---

## Tuning Tolerance Guidelines

| Application | Max Error (cents) | Notes |
|-------------|-------------------|-------|
| Professional concert instrument | < 5 | Hand-tuned, individual verification |
| Educational/practice instrument | < 15 | Machine-tuned, batch production |
| Experimental/prototype | < 30 | Good for design validation |
| Initial optimization target | < 50 | Starting point, refine from here |

### Error Interpretation

```
1 cent = 1/100 of a semitone
12 cents ≈ 1/8 of a semitone (most listeners can hear this)
50 cents = 1/2 semitone (clearly audible)
100 cents = 1 semitone
```

---

## Troubleshooting Tuning Issues

### Problem: Can't achieve target ratio

**Possible causes:**
1. Bar too thick for the target frequency ratio
2. Number of cuts insufficient
3. Constraints too restrictive

**Solutions:**
- Increase `h0` (starting thickness)
- Add more cuts (try 2-3)
- Relax `h_min` constraint
- Consider length adjustment (`max_length_trim`)

### Problem: Torsional modes interfering

**Solution:** Use 3D analysis, which captures torsional modes, then adjust width-to-thickness ratio.

### Problem: 2D and 3D results differ significantly

**Expected behavior:** 2D analysis captures only vertical bending. For stumpy bars (length/thickness < 10), differences can be 5-15%.

**Solution:** Use hybrid optimization with 2D warm-up and 3D refinement, or use pure 3D for final verification.

---

## References

1. Rossing, T.D. (2000). *Science of Percussion Instruments*. World Scientific.
2. Bork, I. (1995). "Practical tuning of xylophone bars and resonators." *Applied Acoustics*, 46(1), 103-127.
3. Soares, F., et al. (2021). "Efficient resonator optimization through metamodel-assisted evolutionary algorithms." *Journal of Sound and Vibration*, 512, 116378.
