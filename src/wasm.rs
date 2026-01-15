//! WASM bindings for browser integration.
//!
//! This module provides JavaScript-friendly wrappers around the core FEM functionality.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use crate::beam2d_solver::compute_modal_frequencies_2d_from_cuts;
use crate::cuts::{genes_to_cuts, generate_element_heights};
use crate::mesh::generate_bar_mesh_3d;
use crate::modes::classify_all_modes;
use crate::optimization::materials::{get_material, get_materials_by_category, MATERIAL_KEYS};
use crate::solver::compute_modal_frequencies_with_shapes;
use crate::types::ModeType;
use crate::Cut;

/// Initialize panic hook for better error messages in browser console.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Compute modal frequencies using 2D Timoshenko beam theory.
/// This is fast and suitable for optimization loops.
///
/// # Arguments
/// * `cuts_json` - JSON array of cuts: `[{"lambda": 0.2, "h": 0.5}, ...]`
/// * `length` - Bar length in meters
/// * `width` - Bar width in meters
/// * `height` - Bar height (thickness) in meters
/// * `e` - Young's modulus in Pa
/// * `nu` - Poisson's ratio
/// * `rho` - Density in kg/m³
/// * `elements` - Number of beam elements
/// * `modes` - Number of modes to compute
///
/// # Returns
/// JSON array of frequencies in Hz
#[wasm_bindgen]
pub fn compute_frequencies_2d(
    cuts_json: &str,
    length: f64,
    width: f64,
    height: f64,
    e: f64,
    nu: f64,
    rho: f64,
    elements: usize,
    modes: usize,
) -> Result<String, JsValue> {
    let cuts: Vec<Cut> = serde_json::from_str(cuts_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid cuts JSON: {}", e)))?;

    let frequencies = compute_modal_frequencies_2d_from_cuts(
        &cuts, length, width, height, e, nu, rho, elements, modes,
    );

    serde_json::to_string(&frequencies)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Compute modal frequencies using full 3D FEM analysis with Soares mode classification.
/// More accurate but slower and more memory-intensive.
///
/// # Arguments
/// * `length` - Bar length in meters
/// * `width` - Bar width in meters
/// * `heights_json` - JSON array of element heights
/// * `nx`, `ny`, `nz` - Mesh divisions
/// * `e` - Young's modulus in Pa
/// * `nu` - Poisson's ratio
/// * `rho` - Density in kg/m³
/// * `modes` - Number of modes to compute
///
/// # Returns
/// JSON object with classified modes:
/// ```json
/// {
///   "frequencies": [f1, f2, ...],
///   "classified": {
///     "vertical_bending": [{"frequency": f, "mode_index": i, "family_rank": r}, ...],
///     "torsional": [...],
///     "lateral": [...],
///     "axial": [...]
///   }
/// }
/// ```
#[wasm_bindgen]
pub fn compute_frequencies_3d(
    length: f64,
    width: f64,
    heights_json: &str,
    nx: usize,
    ny: usize,
    nz: usize,
    e: f64,
    nu: f64,
    rho: f64,
    modes: usize,
) -> Result<String, JsValue> {
    use serde_json::json;

    let heights: Vec<f64> = serde_json::from_str(heights_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid heights JSON: {}", e)))?;

    if heights.len() != nx {
        return Err(JsValue::from_str(&format!(
            "Heights length ({}) must equal nx ({})",
            heights.len(),
            nx
        )));
    }

    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);
    let (frequencies, mode_shapes) = compute_modal_frequencies_with_shapes(&mesh, e, nu, rho, modes);

    // Classify modes using Soares method
    let classified = classify_all_modes(&frequencies, &mode_shapes, &mesh.nodes);

    // Convert to JSON-friendly format
    let format_modes = |modes: &[(f64, usize, usize)]| -> Vec<serde_json::Value> {
        modes
            .iter()
            .map(|(freq, mode_idx, family_rank)| {
                json!({
                    "frequency": freq,
                    "mode_index": mode_idx,
                    "family_rank": family_rank
                })
            })
            .collect()
    };

    let result = json!({
        "frequencies": frequencies,
        "classified": {
            "vertical_bending": format_modes(classified.get(&ModeType::VerticalBending).unwrap_or(&vec![])),
            "torsional": format_modes(classified.get(&ModeType::Torsional).unwrap_or(&vec![])),
            "lateral": format_modes(classified.get(&ModeType::Lateral).unwrap_or(&vec![])),
            "axial": format_modes(classified.get(&ModeType::Axial).unwrap_or(&vec![]))
        }
    });

    serde_json::to_string(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Convert optimization genes to cuts.
///
/// # Arguments
/// * `genes_json` - JSON array of gene values (pairs of lambda, h for each cut)
///
/// # Returns
/// JSON array of Cut objects
#[wasm_bindgen]
pub fn genes_to_cuts_json(genes_json: &str) -> Result<String, JsValue> {
    let genes: Vec<f64> = serde_json::from_str(genes_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid genes JSON: {}", e)))?;

    let cuts = genes_to_cuts(&genes);

    serde_json::to_string(&cuts)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Generate element heights from cuts.
///
/// # Arguments
/// * `cuts_json` - JSON array of cuts
/// * `length` - Bar length in meters
/// * `h0` - Base height in meters
/// * `num_elements` - Number of elements
///
/// # Returns
/// JSON array of heights for each element
#[wasm_bindgen]
pub fn cuts_to_heights(
    cuts_json: &str,
    length: f64,
    h0: f64,
    num_elements: usize,
) -> Result<String, JsValue> {
    let cuts: Vec<Cut> = serde_json::from_str(cuts_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid cuts JSON: {}", e)))?;

    let heights = generate_element_heights(&cuts, length, h0, num_elements);

    serde_json::to_string(&heights)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Get memory usage estimate for a 3D mesh.
/// Useful for checking if a mesh will fit in browser memory.
///
/// # Returns
/// Estimated memory usage in bytes
#[wasm_bindgen]
pub fn estimate_memory_usage(nx: usize, ny: usize, nz: usize) -> usize {
    let num_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    let num_elements = nx * ny * nz;
    let dof = num_nodes * 3;

    // Rough estimate:
    // - Nodes: 3 f64 per node = 24 bytes
    // - Elements: 8 usize per element = 64 bytes
    // - Stiffness matrix (dense): dof * dof * 8 bytes
    // - Mass matrix (dense): dof * dof * 8 bytes
    let node_mem = num_nodes * 24;
    let element_mem = num_elements * 64;
    let matrix_mem = 2 * dof * dof * 8;

    node_mem + element_mem + matrix_mem
}

/// Check if a mesh size is safe for browser memory limits.
///
/// # Returns
/// true if the estimated memory usage is under 512MB
#[wasm_bindgen]
pub fn is_mesh_size_safe(nx: usize, ny: usize, nz: usize) -> bool {
    let mem = estimate_memory_usage(nx, ny, nz);
    mem < 512 * 1024 * 1024 // 512 MB limit
}

/// Get list of all available material keys.
///
/// # Returns
/// JSON array of material key strings
#[wasm_bindgen]
pub fn get_material_keys() -> String {
    serde_json::to_string(&MATERIAL_KEYS).unwrap_or_else(|_| "[]".to_string())
}

/// Get materials grouped by category.
///
/// # Returns
/// JSON object with "metals" and "woods" arrays, each containing objects with
/// key, name, e (Young's modulus), rho (density), nu (Poisson's ratio)
#[wasm_bindgen]
pub fn get_materials_json() -> String {
    use serde_json::{json, Value};

    let by_category = get_materials_by_category();

    let format_materials = |materials: &[(&str, crate::optimization::types::Material)]| -> Vec<Value> {
        materials.iter().map(|(key, mat)| {
            json!({
                "key": key,
                "name": mat.name,
                "e": mat.e,
                "rho": mat.rho,
                "nu": mat.nu
            })
        }).collect()
    };

    let result = json!({
        "metals": format_materials(by_category.get("metals").unwrap_or(&vec![])),
        "woods": format_materials(by_category.get("woods").unwrap_or(&vec![]))
    });

    serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
}

/// Get properties for a specific material.
///
/// # Arguments
/// * `key` - Material key (e.g., "sapele", "aluminum")
///
/// # Returns
/// JSON object with name, e, rho, nu or null if not found
#[wasm_bindgen]
pub fn get_material_json(key: &str) -> String {
    use serde_json::json;

    match get_material(key) {
        Some(mat) => {
            let result = json!({
                "name": mat.name,
                "e": mat.e,
                "rho": mat.rho,
                "nu": mat.nu
            });
            serde_json::to_string(&result).unwrap_or_else(|_| "null".to_string())
        }
        None => "null".to_string()
    }
}

/// Generate 3D mesh data for visualization.
/// Returns mesh in a format suitable for Three.js rendering.
///
/// # Arguments
/// * `length` - Bar length in meters
/// * `width` - Bar width in meters
/// * `heights_json` - JSON array of element heights
/// * `nx`, `ny`, `nz` - Mesh divisions
///
/// # Returns
/// JSON object with nodes, elements, element_heights, and metadata
#[wasm_bindgen]
pub fn generate_mesh_json(
    length: f64,
    width: f64,
    heights_json: &str,
    nx: usize,
    ny: usize,
    nz: usize,
) -> Result<String, JsValue> {
    let heights: Vec<f64> = serde_json::from_str(heights_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid heights JSON: {}", e)))?;

    if heights.len() != nx {
        return Err(JsValue::from_str(&format!(
            "Heights length ({}) must equal nx ({})", heights.len(), nx
        )));
    }

    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);

    mesh.to_json()
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Compute 3D analysis and return frequencies, classified modes, and mesh data.
/// Combines frequency computation with Soares mode classification and mesh generation for visualization.
///
/// # Arguments
/// * `length` - Bar length in meters
/// * `width` - Bar width in meters
/// * `heights_json` - JSON array of element heights
/// * `nx`, `ny`, `nz` - Mesh divisions
/// * `e` - Young's modulus in Pa
/// * `nu` - Poisson's ratio
/// * `rho` - Density in kg/m³
/// * `modes` - Number of modes to compute
///
/// # Returns
/// JSON object with frequencies, classified modes, and mesh data:
/// ```json
/// {
///   "frequencies": [f1, f2, ...],
///   "classified": {
///     "vertical_bending": [{"frequency": f, "mode_index": i, "family_rank": r}, ...],
///     "torsional": [...],
///     "lateral": [...],
///     "axial": [...]
///   },
///   "mesh": { ... }
/// }
/// ```
#[wasm_bindgen]
pub fn compute_3d_with_mesh(
    length: f64,
    width: f64,
    heights_json: &str,
    nx: usize,
    ny: usize,
    nz: usize,
    e: f64,
    nu: f64,
    rho: f64,
    modes: usize,
) -> Result<String, JsValue> {
    use serde_json::json;

    let heights: Vec<f64> = serde_json::from_str(heights_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid heights JSON: {}", e)))?;

    if heights.len() != nx {
        return Err(JsValue::from_str(&format!(
            "Heights length ({}) must equal nx ({})",
            heights.len(),
            nx
        )));
    }

    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);
    let (frequencies, mode_shapes) =
        compute_modal_frequencies_with_shapes(&mesh, e, nu, rho, modes);
    let serializable_mesh = mesh.to_serializable();

    // Classify modes using Soares method
    let classified = classify_all_modes(&frequencies, &mode_shapes, &mesh.nodes);

    // Convert to JSON-friendly format
    let format_modes = |modes: &[(f64, usize, usize)]| -> Vec<serde_json::Value> {
        modes
            .iter()
            .map(|(freq, mode_idx, family_rank)| {
                json!({
                    "frequency": freq,
                    "mode_index": mode_idx,
                    "family_rank": family_rank
                })
            })
            .collect()
    };

    let result = json!({
        "frequencies": frequencies,
        "classified": {
            "vertical_bending": format_modes(classified.get(&ModeType::VerticalBending).unwrap_or(&vec![])),
            "torsional": format_modes(classified.get(&ModeType::Torsional).unwrap_or(&vec![])),
            "lateral": format_modes(classified.get(&ModeType::Lateral).unwrap_or(&vec![])),
            "axial": format_modes(classified.get(&ModeType::Axial).unwrap_or(&vec![]))
        },
        "mesh": serializable_mesh
    });

    serde_json::to_string(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
