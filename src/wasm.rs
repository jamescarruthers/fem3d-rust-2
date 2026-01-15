//! WASM bindings for browser integration.
//!
//! This module provides JavaScript-friendly wrappers around the core FEM functionality.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

use crate::cuts::{genes_to_cuts, generate_element_heights};
use crate::mesh::generate_bar_mesh_3d;
use crate::solver::compute_modal_frequencies;
use crate::beam2d_solver::compute_modal_frequencies_2d_from_cuts;
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

/// Compute modal frequencies using full 3D FEM analysis.
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
/// JSON array of frequencies in Hz
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
    let heights: Vec<f64> = serde_json::from_str(heights_json)
        .map_err(|e| JsValue::from_str(&format!("Invalid heights JSON: {}", e)))?;

    if heights.len() != nx {
        return Err(JsValue::from_str(&format!(
            "Heights length ({}) must equal nx ({})", heights.len(), nx
        )));
    }

    let mesh = generate_bar_mesh_3d(length, width, &heights, nx, ny, nz);
    let frequencies = compute_modal_frequencies(&mesh, e, nu, rho, modes);

    serde_json::to_string(&frequencies)
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
