# FEM3D-Rust Library Usage Guide

A comprehensive guide to using the FEM3D-Rust library for modal analysis and optimization of percussion instrument bars.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Modal Analysis](#modal-analysis)
5. [Bar Optimization](#bar-optimization)
6. [API Reference](#api-reference)
7. [Common Workflows](#common-workflows)
8. [Advanced Topics](#advanced-topics)

---

## Overview

FEM3D-Rust is a Rust library for **3D Finite Element Modal Analysis** of percussion instrument bars (xylophones, marimbas, vibraphones). It provides:

- **Accurate 3D FEM analysis** using 8-node hexahedral (Hex8) elements
- **Fast 2D beam analysis** using Timoshenko beam elements
- **Three optimization strategies** for bar tuning:
  - Evolutionary Algorithm (EA)
  - Surrogate-assisted optimization (RBF-based)
  - Hybrid 2D→3D optimization (recommended)
- **Mode classification** using Soares' corner displacement method
- **WebAssembly support** for browser-based applications

### Key Features

| Feature | Description |
|---------|-------------|
| 3D Hex8 FEM | Full 3D analysis with trilinear shape functions |
| 2D Timoshenko Beam | Fast analysis accounting for shear deformation |
| Sparse Solvers | Efficient eigenvalue computation for large meshes |
| 19+ Materials | Pre-defined woods and metals with realistic properties |
| Parallel Processing | Rayon-based parallelization (native) |
| WASM Support | Browser-compatible compilation |

---

## Getting Started

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fem3d-rust-2 = { path = "../fem3d-rust-2" }
```

Or with specific features:

```toml
[dependencies]
fem3d-rust-2 = { path = "../fem3d-rust-2", features = ["parallel"] }
```

### Available Features

| Feature | Description | Default |
|---------|-------------|---------|
| `parallel` | Enable Rayon parallelization | Yes |
| `sprs-backend` | Alternative sparse matrix backend | No |
| `parallel-wasm` | Web Worker parallelization for WASM | No |

### Basic Example

```rust
use fem3d_rust_2::{
    Cut, generate_element_heights, generate_bar_mesh_3d,
    compute_modal_frequencies, Material,
};

fn main() {
    // 1. Define material (Sapele wood)
    let material = Material::sapele();

    // 2. Define bar geometry
    let length = 0.35;  // 350mm
    let width = 0.05;   // 50mm
    let h0 = 0.024;     // 24mm thickness

    // 3. Define undercut geometry
    let cuts = vec![
        Cut::new(0.08, 0.012),  // lambda=80mm, h=12mm
        Cut::new(0.12, 0.008),  // lambda=120mm, h=8mm
    ];

    // 4. Generate mesh
    let heights = generate_element_heights(&cuts, length, h0, 30);
    let mesh = generate_bar_mesh_3d(length, width, &heights, 30, 2, 2);

    // 5. Compute modal frequencies
    let frequencies = compute_modal_frequencies(
        &mesh,
        material.e,    // Young's modulus
        material.nu,   // Poisson's ratio
        material.rho,  // Density
        4,             // Number of modes
    );

    println!("Modal frequencies: {:?}", frequencies);
}
```

---

## Core Concepts

### Materials

The library provides pre-defined materials commonly used for percussion instruments:

#### Metals
```rust
use fem3d_rust_2::Material;

let aluminum = Material::aluminum();      // 6061 aluminum (vibraphones)
let aluminum7075 = Material::aluminum7075(); // 7075 aluminum
let brass = Material::brass();            // C260 cartridge brass
let steel = Material::steel();            // 1018 low carbon steel
let stainless = Material::stainless_steel(); // 304 stainless
let bronze = Material::bronze();          // Phosphor bronze
let bell_bronze = Material::bell_bronze(); // B20 bell alloy
```

#### Woods
```rust
let rosewood = Material::rosewood();      // Honduran rosewood (premium marimbas)
let sapele = Material::sapele();          // Sapele (common marimba wood)
let padauk = Material::padauk();          // African padauk
let maple = Material::maple();            // Hard maple
let bubinga = Material::bubinga();        // African bubinga
let ebony = Material::ebony();            // African ebony
let cocobolo = Material::cocobolo();      // Cocobolo
let purpleheart = Material::purpleheart(); // Purpleheart
```

#### Custom Materials
```rust
use fem3d_rust_2::{Material, MaterialCategory};

let custom = Material::new(
    "Custom Alloy",
    75.0e9,              // E: Young's modulus (Pa)
    3000.0,              // rho: Density (kg/m³)
    0.30,                // nu: Poisson's ratio
    MaterialCategory::Metal,
);
```

### Bar Parameters

Define bar geometry using `BarParameters`:

```rust
use fem3d_rust_2::BarParameters;

let bar = BarParameters::new(
    0.35,   // length: 350mm
    0.05,   // width: 50mm
    0.024,  // h0: original thickness 24mm
    0.005,  // h_min: minimum thickness 5mm (for undercuts)
);
```

### Undercut Geometry (Cuts)

Bars are tuned by removing material from the underside. The `Cut` struct defines symmetric undercut regions:

```rust
use fem3d_rust_2::Cut;

// A cut is defined by:
// - lambda: Distance from bar center to cut boundary (m)
// - h: Height (remaining thickness) at the cut center (m)
let cut = Cut::new(0.08, 0.010);  // lambda=80mm, h=10mm
```

**Undercut Profile Visualization:**

```
Top surface (flat)
═══════════════════════════════════════════════

           ┌────────────────────┐
           │                    │  h0 (original thickness)
    ╔══════╧════════════════════╧══════╗
    ║      ╲                    ╱      ║
    ║       ╲__________________╱       ║  h (cut height)
    ║           ← 2*lambda →           ║
    ╚══════════════════════════════════╝
    └─────────────── L ────────────────┘

Center                                  End
```

The profile between cuts uses quadratic interpolation for smooth transitions.

#### Working with Cuts

```rust
use fem3d_rust_2::{Cut, compute_height, generate_element_heights, genes_to_cuts, cuts_to_genes};

// Create cuts
let cuts = vec![
    Cut::new(0.05, 0.015),
    Cut::new(0.10, 0.008),
];

// Get height at any position along the bar
let h_at_center = compute_height(0.0, &cuts, 0.35, 0.024);
let h_at_50mm = compute_height(0.05, &cuts, 0.35, 0.024);

// Discretize profile for mesh generation
let heights = generate_element_heights(&cuts, 0.35, 0.024, 30);

// Convert between cuts and gene representation (for optimization)
let genes = cuts_to_genes(&cuts);  // [lambda1, h1, lambda2, h2, ...]
let cuts_back = genes_to_cuts(&genes);
```

### Mesh Generation

#### 3D Hexahedral Mesh

```rust
use fem3d_rust_2::{generate_bar_mesh_3d, generate_bar_mesh_3d_adaptive};

// Uniform mesh with variable thickness
let mesh = generate_bar_mesh_3d(
    length,     // Bar length (m)
    width,      // Bar width (m)
    &heights,   // Height at each element along length
    30,         // nx: elements along length
    2,          // ny: elements along width
    2,          // nz: elements along thickness
);

// Adaptive mesh (refines near cuts)
let mesh = generate_bar_mesh_3d_adaptive(
    &cuts,
    length,
    width,
    h0,
    30, 2, 2,
);
```

**Mesh Properties:**
- Each element is an 8-node hexahedron (Hex8)
- 3 degrees of freedom per node (ux, uy, uz)
- 2×2×2 Gauss quadrature integration
- Typical mesh: 30×2×2 = 120 elements, ~500 DOF

---

## Modal Analysis

### 2D Beam Analysis (Fast)

2D Timoshenko beam analysis is ~100× faster than 3D FEM but only captures vertical bending modes:

```rust
use fem3d_rust_2::{Cut, compute_modal_frequencies_2d_from_cuts};

let cuts = vec![Cut::new(0.08, 0.012)];

let frequencies = compute_modal_frequencies_2d_from_cuts(
    &cuts,
    0.35,       // length
    0.05,       // width
    0.024,      // h0
    12.0e9,     // E (Pa)
    0.35,       // nu
    640.0,      // rho (kg/m³)
    150,        // num_elements
    4,          // num_modes
);
```

**When to use 2D:**
- Initial exploration and quick iteration
- Slender bars (length/thickness > 15)
- When only vertical bending modes matter
- Optimization warm-up phase

### 3D FEM Analysis (Accurate)

Full 3D analysis captures all mode types:

```rust
use fem3d_rust_2::{
    generate_element_heights, generate_bar_mesh_3d,
    compute_modal_frequencies, compute_modal_frequencies_with_shapes,
    EigenSolver,
};

let heights = generate_element_heights(&cuts, length, h0, 30);
let mesh = generate_bar_mesh_3d(length, width, &heights, 30, 2, 2);

// Just frequencies
let frequencies = compute_modal_frequencies(
    &mesh, E, nu, rho, num_modes
);

// Frequencies and mode shapes
let (frequencies, mode_shapes) = compute_modal_frequencies_with_shapes(
    &mesh, E, nu, rho, num_modes
);
```

**Solver Selection:**

```rust
use fem3d_rust_2::{compute_modal_frequencies_with_solver, EigenSolver};

// Automatic (recommended)
let freqs = compute_modal_frequencies_with_solver(
    &mesh, E, nu, rho, num_modes,
    EigenSolver::Auto,  // Uses sparse for DOF > 500
);

// Force dense solver (small meshes)
let freqs = compute_modal_frequencies_with_solver(
    &mesh, E, nu, rho, num_modes,
    EigenSolver::Dense,
);

// Force sparse solver (large meshes)
let freqs = compute_modal_frequencies_with_solver(
    &mesh, E, nu, rho, num_modes,
    EigenSolver::Sparse,
);
```

### Mode Classification

Classify modes using Soares' corner displacement method:

```rust
use fem3d_rust_2::{classify_all_modes, classify_mode_soares, ModeType};

let (frequencies, mode_shapes) = compute_modal_frequencies_with_shapes(
    &mesh, E, nu, rho, 10
);

// Classify all modes
let classified = classify_all_modes(&frequencies, &mode_shapes, &mesh.nodes);

// Access by type
if let Some(vertical) = classified.get(&ModeType::VerticalBending) {
    println!("Vertical bending modes: {:?}", vertical);
}
```

**Mode Types:**
- `VerticalBending` - Primary tuning modes (f1, f2, f3...)
- `Torsional` - Twisting modes
- `Lateral` - Side-to-side bending
- `Axial` - Longitudinal compression/extension
- `Unknown` - Unclassified modes

---

## Bar Optimization

The library provides three optimization strategies for tuning bars to target frequencies.

### Strategy 1: Evolutionary Algorithm (EA)

Direct genetic algorithm optimization:

```rust
use fem3d_rust_2::{
    run_optimization, EAConfig, EAParameters,
    BarParameters, Material, AnalysisMode, PenaltyType,
};

let bar = BarParameters::new(0.35, 0.05, 0.024, 0.005);
let material = Material::sapele();
let targets = vec![175.0, 700.0, 1575.0];  // 1:4:9 ratio

// Configure EA
let mut params = EAParameters::default();
params.population_size = 50;
params.max_generations = 100;
params.target_error = 0.01;  // 1% error threshold
params.analysis_mode = AnalysisMode::Beam2D;  // Fast 2D analysis

let config = EAConfig::new(bar, material, targets, 2)  // 2 cuts
    .with_params(params)
    .with_penalty(PenaltyType::Volume, 0.1);  // 10% volume weight

let result = run_optimization(&config);

println!("{}", result);
println!("Cuts: {:?}", result.cuts);
println!("Frequencies: {:?}", result.computed_frequencies);
println!("Max error: {:.1} cents", result.max_error_cents);
```

### Strategy 2: Surrogate Optimization

RBF-based surrogate model reduces expensive 3D evaluations:

```rust
use fem3d_rust_2::{
    run_hybrid_optimization, HybridConfig, OptimizationStrategy,
    SurrogateConfig, BarParameters, Material,
};

let bar = BarParameters::new(0.35, 0.05, 0.024, 0.005);
let material = Material::aluminum();
let targets = vec![175.0, 700.0];

let config = HybridConfig::new(bar, material, targets, 2)
    .with_strategy(OptimizationStrategy::Surrogate3D(
        SurrogateConfig {
            initial_samples: 20,    // LHS sampling points
            max_evaluations: 100,   // Total 3D FEM calls
            convergence_tol: 0.01,  // 1% tolerance
            verbose: true,
            ..Default::default()
        }
    ));

let result = run_hybrid_optimization(&config);

println!("Best fitness: {:.6}", result.best_fitness);
println!("3D evaluations: {}", result.total_evaluations_3d);
```

### Strategy 3: Hybrid 2D→3D (Recommended)

Combines fast 2D exploration with accurate 3D refinement:

```rust
use fem3d_rust_2::{
    run_hybrid_optimization, HybridConfig, OptimizationStrategy,
    EAParameters, SurrogateConfig, BarParameters, Material,
};

let bar = BarParameters::new(0.35, 0.05, 0.024, 0.005);
let material = Material::aluminum();
let targets = vec![175.0, 700.0];  // 1:4 ratio

let config = HybridConfig::new(bar, material, targets, 2)
    .with_strategy(OptimizationStrategy::Hybrid2Dto3D {
        // Phase 1: Fast 2D EA exploration
        ea_params: EAParameters {
            population_size: 40,
            max_generations: 50,
            ..Default::default()
        },
        // Phase 2: 3D surrogate refinement
        surrogate_config: SurrogateConfig {
            initial_samples: 15,
            max_evaluations: 50,
            convergence_tol: 0.01,
            ..Default::default()
        },
        frequency_correction: 1.0,  // 2D→3D correction factor
    })
    .with_verbose(true);

let result = run_hybrid_optimization(&config);

println!("Phase 1 fitness: {:?}", result.phase1_fitness);
println!("Final fitness: {:.6}", result.best_fitness);
println!("Max error: {:.1} cents", result.max_error_cents);
println!("2D evals: {}, 3D evals: {}",
    result.total_evaluations_2d,
    result.total_evaluations_3d);
```

### EA Parameters Reference

```rust
use fem3d_rust_2::EAParameters;

let params = EAParameters {
    // Population
    population_size: 50,       // Number of individuals

    // Genetic operators (must sum to 100%)
    elitism_percent: 10.0,     // Top individuals kept unchanged
    crossover_percent: 30.0,   // Pairs for crossover
    mutation_percent: 60.0,    // Individuals mutated

    // Mutation
    mutation_strength: 0.1,    // Sigma for uniform mutation (0.05-0.2)

    // Convergence
    max_generations: 100,
    target_error: 0.01,        // Stop if error < 1%

    // Analysis
    analysis_mode: AnalysisMode::Beam2D,
    num_elements: 150,         // Mesh discretization
    num_elements_y: 2,         // 3D mesh: width elements
    num_elements_z: 2,         // 3D mesh: thickness elements

    // Frequency weights
    f1_priority: 1.0,          // Weight multiplier for f1
    frequency_offset: 0.0,     // 2D/3D calibration offset

    // Constraints
    min_cut_width: 0.0,        // Minimum cut width (m)
    max_cut_width: 0.0,        // Maximum cut width (m), 0=unlimited
    min_cut_depth: 0.0,        // Minimum depth h0-h (m)
    max_cut_depth: 0.0,        // Maximum depth (m)
    max_length_trim: 0.0,      // Max trim from each end (m)
    max_length_extend: 0.0,    // Max extension (m)

    max_workers: 0,            // 0 = auto-detect
};
```

### Selection Methods

```rust
use fem3d_rust_2::SelectionMethod;

SelectionMethod::Roulette   // Fitness-proportional (default)
SelectionMethod::Tournament // Tournament selection
SelectionMethod::Rank       // Rank-based selection
```

### Crossover Methods

```rust
use fem3d_rust_2::CrossoverMethod;

CrossoverMethod::Heuristic   // Paper Eq. 16 (default)
CrossoverMethod::Blend       // BLX-alpha blend
CrossoverMethod::SinglePoint // Single-point crossover
CrossoverMethod::TwoPoint    // Two-point crossover
CrossoverMethod::Uniform     // Uniform crossover
```

### Mutation Methods

```rust
use fem3d_rust_2::MutationMethod;

MutationMethod::Uniform          // Uniform random (default)
MutationMethod::GaussianAdaptive // Self-adaptive Gaussian
MutationMethod::Polynomial       // Polynomial mutation
```

### Penalty Types

Control material removal vs. tuning accuracy trade-off:

```rust
use fem3d_rust_2::PenaltyType;

PenaltyType::None      // Pure tuning error
PenaltyType::Volume    // Penalize material removal
PenaltyType::Roughness // Penalize profile roughness

// Usage
let config = EAConfig::new(bar, material, targets, num_cuts)
    .with_penalty(PenaltyType::Volume, 0.1);  // alpha = 0.1
```

The combined objective is:
```
fitness = (1 - alpha) * tuning_error + alpha * penalty
```

---

## API Reference

### Frequency Computation

```rust
// 3D analysis
pub fn compute_modal_frequencies(mesh, E, nu, rho, num_modes) -> Vec<f64>;
pub fn compute_modal_frequencies_with_shapes(mesh, E, nu, rho, num_modes)
    -> (Vec<f64>, DMatrix<f64>);
pub fn compute_modal_frequencies_with_solver(mesh, E, nu, rho, num_modes, solver)
    -> Vec<f64>;

// 2D beam analysis
pub fn compute_modal_frequencies_2d(heights, le, b, E, rho, nu, num_modes)
    -> Vec<f64>;
pub fn compute_modal_frequencies_2d_from_cuts(cuts, length, width, h0, E, nu, rho,
    num_elements, num_modes) -> Vec<f64>;
```

### Mesh Generation

```rust
pub fn generate_bar_mesh_3d(length, width, heights, nx, ny, nz) -> Mesh3d;
pub fn generate_bar_mesh_3d_adaptive(cuts, length, width, h0, nx, ny, nz) -> Mesh3d;
```

### Cut Geometry

```rust
pub fn compute_height(x, cuts, length, h0) -> f64;
pub fn generate_element_heights(cuts, length, h0, num_elements) -> Vec<f64>;
pub fn genes_to_cuts(genes) -> Vec<Cut>;
pub fn cuts_to_genes(cuts) -> Vec<f64>;
```

### Optimization

```rust
// Evolutionary algorithm
pub fn run_optimization(config: &EAConfig) -> OptimizationResult;
pub fn run_evolutionary_algorithm(config, on_progress, should_stop)
    -> OptimizationResult;

// Surrogate optimization
pub fn run_surrogate_optimization(config, bounds, evaluate_fn) -> SurrogateResult;

// Hybrid strategies
pub fn run_hybrid_optimization(config: &HybridConfig) -> HybridResult;
```

### Objective Functions

```rust
pub fn compute_tuning_error(computed, target, f1_priority) -> f64;
pub fn compute_cents_errors(computed, target) -> Vec<f64>;
pub fn compute_volume_penalty(cuts, length, h0) -> f64;
pub fn compute_roughness_penalty(cuts, h0) -> f64;
```

### Mode Classification

```rust
pub fn classify_all_modes(freqs, mode_shapes, nodes)
    -> HashMap<ModeType, Vec<(usize, f64)>>;
pub fn classify_mode_soares(mode_shape, nodes, corner_indices) -> ModeType;
```

---

## Common Workflows

### Workflow 1: Quick Frequency Check

Compute frequencies for an existing bar design:

```rust
use fem3d_rust_2::*;

fn check_frequencies(cuts: &[Cut], bar: &BarParameters, material: &Material) -> Vec<f64> {
    // Use fast 2D analysis for quick check
    compute_modal_frequencies_2d_from_cuts(
        cuts,
        bar.length,
        bar.width,
        bar.h0,
        material.e,
        material.nu,
        material.rho,
        150,
        4,
    )
}
```

### Workflow 2: Optimize a Single Bar

Find undercut geometry for target frequencies:

```rust
use fem3d_rust_2::*;

fn optimize_bar(
    bar: BarParameters,
    material: Material,
    targets: Vec<f64>,
    num_cuts: usize,
) -> OptimizationResult {
    let config = HybridConfig::new(bar, material, targets, num_cuts)
        .with_strategy(OptimizationStrategy::Hybrid2Dto3D {
            ea_params: EAParameters {
                population_size: 40,
                max_generations: 60,
                target_error: 0.005,  // 0.5% error
                ..Default::default()
            },
            surrogate_config: SurrogateConfig {
                initial_samples: 15,
                max_evaluations: 60,
                ..Default::default()
            },
            frequency_correction: 1.0,
        })
        .with_verbose(true);

    let result = run_hybrid_optimization(&config);

    // Convert to OptimizationResult format
    OptimizationResult {
        best_individual: Individual::with_fitness(result.best_genes.clone(), result.best_fitness),
        cuts: genes_to_cuts(&result.best_genes),
        computed_frequencies: result.computed_frequencies,
        target_frequencies: result.target_frequencies,
        tuning_error: result.best_fitness,
        max_error_cents: result.max_error_cents,
        errors_in_cents: result.errors_cents,
        volume_percent: result.volume_percent,
        roughness_percent: 0.0,
        generations: result.total_evaluations_2d + result.total_evaluations_3d,
        length_trim: 0.0,
        effective_length: bar.length,
    }
}
```

### Workflow 3: Validate 2D vs 3D

Compare 2D and 3D analysis for validation:

```rust
use fem3d_rust_2::*;

fn compare_2d_3d(cuts: &[Cut], bar: &BarParameters, material: &Material) {
    // 2D analysis
    let freqs_2d = compute_modal_frequencies_2d_from_cuts(
        cuts, bar.length, bar.width, bar.h0,
        material.e, material.nu, material.rho, 150, 4,
    );

    // 3D analysis
    let heights = generate_element_heights(cuts, bar.length, bar.h0, 30);
    let mesh = generate_bar_mesh_3d(bar.length, bar.width, &heights, 30, 2, 2);
    let freqs_3d = compute_modal_frequencies(&mesh, material.e, material.nu, material.rho, 4);

    println!("Frequency comparison:");
    println!("{:>8} {:>10} {:>10} {:>10}", "Mode", "2D (Hz)", "3D (Hz)", "Diff (%)");
    for i in 0..freqs_2d.len().min(freqs_3d.len()) {
        let diff = 100.0 * (freqs_2d[i] - freqs_3d[i]) / freqs_3d[i];
        println!("{:>8} {:>10.1} {:>10.1} {:>+10.2}",
            format!("f{}", i+1), freqs_2d[i], freqs_3d[i], diff);
    }
}
```

### Workflow 4: Progress Monitoring

Track optimization progress with callbacks:

```rust
use fem3d_rust_2::*;

fn optimize_with_progress(config: &EAConfig) -> OptimizationResult {
    run_evolutionary_algorithm(
        config,
        Some(|update: ProgressUpdate| {
            println!(
                "Gen {:3}: best={:.6}, avg={:.6}, f1={:.1}Hz, max_err={:.1}ct",
                update.generation,
                update.best_fitness,
                update.average_fitness,
                update.computed_frequencies.as_ref().map(|f| f[0]).unwrap_or(0.0),
                update.errors_in_cents.as_ref()
                    .map(|e| e.iter().map(|x| x.abs()).fold(0.0f64, f64::max))
                    .unwrap_or(0.0),
            );
        }),
        None::<fn() -> bool>,  // No stop condition
    )
}
```

### Workflow 5: Batch Optimization

Optimize multiple bars (e.g., full octave):

```rust
use fem3d_rust_2::*;

fn optimize_octave(base_length: f64, material: Material) -> Vec<OptimizationResult> {
    // C4 to C5 frequencies
    let notes = [
        ("C4", 261.63), ("D4", 293.66), ("E4", 329.63), ("F4", 349.23),
        ("G4", 392.00), ("A4", 440.00), ("B4", 493.88), ("C5", 523.25),
    ];

    notes.iter().enumerate().map(|(i, (name, f1))| {
        // Bar length decreases with pitch
        let length = base_length * (261.63 / f1).sqrt();
        let bar = BarParameters::new(length, 0.05, 0.024, 0.005);

        // Target 1:4:9 ratio
        let targets = vec![*f1, f1 * 4.0, f1 * 9.0];

        let mut params = EAParameters::default();
        params.population_size = 30;
        params.max_generations = 50;

        let config = EAConfig::new(bar, material.clone(), targets, 2)
            .with_params(params);

        println!("Optimizing {} ({:.1} Hz)...", name, f1);
        run_optimization(&config)
    }).collect()
}
```

---

## Advanced Topics

### Custom Surrogate Configuration

Fine-tune the RBF surrogate model:

```rust
use fem3d_rust_2::{SurrogateConfig, RbfKernel, AlphaSchedule};

let config = SurrogateConfig {
    // Sampling
    initial_samples: 25,          // More samples = better initial model
    max_evaluations: 150,         // Total budget

    // RBF model
    kernel: RbfKernel::Cubic,     // Options: Cubic, ThinPlate, Gaussian, Multiquadric
    regularization: 1e-8,         // Ridge regression regularization

    // Search strategy
    convergence_tol: 0.005,       // 0.5% tolerance
    exploration_weight: 0.3,      // Balance exploration vs exploitation
    alpha_schedule: AlphaSchedule::Linear,  // Options: Constant, Linear, Exponential

    // Initial point (warm start)
    initial_point: Some(vec![0.08, 0.012, 0.12, 0.008]),

    verbose: true,
};
```

### Mesh Export for Visualization

Export mesh to JSON for Three.js/Babylon.js visualization:

```rust
use fem3d_rust_2::{generate_bar_mesh_3d, SerializableMesh};

let mesh = generate_bar_mesh_3d(length, width, &heights, 30, 2, 2);
let serializable = SerializableMesh::from_mesh3d(&mesh, &heights);
let json = serde_json::to_string_pretty(&serializable)?;

std::fs::write("bar_mesh.json", json)?;
```

### WebAssembly Usage

The library compiles to WASM for browser use:

```rust
// In wasm.rs, exposed functions:
#[wasm_bindgen]
pub fn compute_frequencies_3d(
    length: f64, width: f64, heights: &[f64],
    nx: usize, ny: usize, nz: usize,
    e: f64, nu: f64, rho: f64, num_modes: usize,
) -> Vec<f64>;

#[wasm_bindgen]
pub fn optimize_bar_2d(
    // ... parameters
) -> JsValue;  // Returns OptimizationResult as JSON
```

### Parallel Processing

Enable parallel evaluation of population fitness:

```toml
# Cargo.toml
[features]
default = ["parallel"]
parallel = ["rayon"]
```

The library automatically parallelizes:
- Element matrix computation
- Population fitness evaluation
- Surrogate model updates

### Tuning Error Metrics

Understand how tuning quality is measured:

```rust
// Weighted sum of squared relative errors (Eq. 7 from paper)
// fitness = sum_i(w_i * ((f_computed - f_target) / f_target)^2)
let tuning_error = compute_tuning_error(&computed, &targets, f1_priority);

// Error in cents (musical unit, 100 cents = 1 semitone)
// cents = 1200 * log2(f_computed / f_target)
let cents_errors = compute_cents_errors(&computed, &targets);

// Typical acceptance criteria:
// < 5 cents: Excellent (professional instrument)
// < 15 cents: Good (acceptable for most musicians)
// < 30 cents: Fair (noticeable but usable)
// > 30 cents: Poor (needs retuning)
```

---

## References

This library is based on the methodology from:

> Soares, F., Debut, V., Antunes, J., & Silva, L. (2021). "Efficient resonator optimization through
> metamodel-assisted evolutionary algorithms." *Journal of Sound and Vibration*, 512, 116378.

Key implementation details:
- 8-node hexahedral finite elements with trilinear shape functions
- 2×2×2 Gauss quadrature integration
- Timoshenko beam theory with shear correction κ = 5/6
- Shift-invert Lanczos eigenvalue solver for sparse systems
- RBF surrogate with cubic kernel (default)

---

## Troubleshooting

### Common Issues

**Problem:** Frequencies are all zero or very small
**Solution:** Check units - E should be in Pa (not GPa), dimensions in meters

**Problem:** Optimization doesn't converge
**Solution:**
- Increase `max_generations` or `max_evaluations`
- Try different number of cuts (1-3 typically sufficient)
- Verify target frequencies are physically achievable

**Problem:** 2D and 3D frequencies differ significantly
**Solution:**
- 2D is less accurate for stumpy bars (length/thickness < 10)
- 3D captures torsional modes that 2D misses
- Use `frequency_correction` factor in hybrid optimization

**Problem:** Memory usage is high
**Solution:**
- Reduce mesh density (nx, ny, nz)
- Use sparse solver: `EigenSolver::Sparse`
- Consider 2D analysis for exploration phase

---

## License

MIT License - see LICENSE file for details.
