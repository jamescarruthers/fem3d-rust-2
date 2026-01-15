//! Material properties for bar tuning optimization.
//!
//! Contains properties for common metals and woods used in percussion instruments.

use super::types::{Material, MaterialCategory};
use std::collections::HashMap;

/// Shear correction factor for rectangular cross-section (Timoshenko beam theory).
pub const KAPPA: f64 = 5.0 / 6.0;

/// All available material keys.
pub const MATERIAL_KEYS: &[&str] = &[
    // Metals
    "aluminum",
    "aluminum7075",
    "brass",
    "steel",
    "stainless_steel",
    "bronze",
    "bell_bronze",
    "fiberglass",
    // Woods
    "rosewood",
    "african_rosewood",
    "padauk",
    "sapele",
    "bubinga",
    "maple",
    "purpleheart",
    "wenge",
    "bocote",
    "zebrawood",
    "cocobolo",
    "ebony",
    "teak",
];

/// Get material by key.
///
/// # Arguments
/// * `key` - Material key (e.g., "sapele", "aluminum", "rosewood")
///
/// # Returns
/// `Some(Material)` if the key is valid, `None` otherwise.
///
/// # Example
/// ```
/// use fem3d_rust_2::optimization::materials::get_material;
///
/// let sapele = get_material("sapele").unwrap();
/// assert_eq!(sapele.name, "Sapele");
/// assert!((sapele.e - 12.0e9).abs() < 1.0);
/// ```
pub fn get_material(key: &str) -> Option<Material> {
    match key {
        // Metals
        "aluminum" => Some(Material::aluminum()),
        "aluminum7075" => Some(Material::aluminum7075()),
        "brass" => Some(Material::brass()),
        "steel" => Some(Material::steel()),
        "stainless_steel" => Some(Material::stainless_steel()),
        "bronze" => Some(Material::bronze()),
        "bell_bronze" => Some(Material::bell_bronze()),
        "fiberglass" => Some(Material::fiberglass()),
        // Woods
        "rosewood" => Some(Material::rosewood()),
        "african_rosewood" => Some(Material::african_rosewood()),
        "padauk" => Some(Material::padauk()),
        "sapele" => Some(Material::sapele()),
        "bubinga" => Some(Material::bubinga()),
        "maple" => Some(Material::maple()),
        "purpleheart" => Some(Material::purpleheart()),
        "wenge" => Some(Material::wenge()),
        "bocote" => Some(Material::bocote()),
        "zebrawood" => Some(Material::zebrawood()),
        "cocobolo" => Some(Material::cocobolo()),
        "ebony" => Some(Material::ebony()),
        "teak" => Some(Material::teak()),
        _ => None,
    }
}

/// Get all materials grouped by category.
///
/// # Returns
/// A HashMap with "metals" and "woods" keys, each containing a vector of (key, Material) tuples.
///
/// # Example
/// ```
/// use fem3d_rust_2::optimization::materials::get_materials_by_category;
///
/// let by_category = get_materials_by_category();
/// assert!(by_category.get("metals").unwrap().len() >= 8);
/// assert!(by_category.get("woods").unwrap().len() >= 13);
/// ```
pub fn get_materials_by_category() -> HashMap<&'static str, Vec<(&'static str, Material)>> {
    let mut metals = Vec::new();
    let mut woods = Vec::new();

    for &key in MATERIAL_KEYS {
        if let Some(material) = get_material(key) {
            match material.category {
                MaterialCategory::Metal => metals.push((key, material)),
                MaterialCategory::Wood => woods.push((key, material)),
            }
        }
    }

    let mut result = HashMap::new();
    result.insert("metals", metals);
    result.insert("woods", woods);
    result
}

/// Get all available materials as a vector of (key, Material) tuples.
///
/// # Example
/// ```
/// use fem3d_rust_2::optimization::materials::get_all_materials;
///
/// let materials = get_all_materials();
/// assert_eq!(materials.len(), 21);
/// ```
pub fn get_all_materials() -> Vec<(&'static str, Material)> {
    MATERIAL_KEYS
        .iter()
        .filter_map(|&key| get_material(key).map(|m| (key, m)))
        .collect()
}

/// Calculate shear modulus from Young's modulus and Poisson's ratio.
///
/// G = E / (2 * (1 + nu))
///
/// # Arguments
/// * `e` - Young's modulus (Pa)
/// * `nu` - Poisson's ratio
///
/// # Returns
/// Shear modulus G (Pa)
///
/// # Example
/// ```
/// use fem3d_rust_2::optimization::materials::calculate_shear_modulus;
///
/// // For Sapele: E = 12.0e9, nu = 0.35
/// let g = calculate_shear_modulus(12.0e9, 0.35);
/// assert!((g - 4.444e9).abs() < 1e7);
/// ```
pub fn calculate_shear_modulus(e: f64, nu: f64) -> f64 {
    e / (2.0 * (1.0 + nu))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_material_sapele() {
        let material = get_material("sapele").unwrap();
        assert_eq!(material.name, "Sapele");
        assert!((material.e - 12.0e9).abs() < 1.0);
        assert!((material.rho - 640.0).abs() < 0.1);
        assert!((material.nu - 0.35).abs() < 0.001);
        assert_eq!(material.category, MaterialCategory::Wood);
    }

    #[test]
    fn test_get_material_aluminum() {
        let material = get_material("aluminum").unwrap();
        assert_eq!(material.name, "Aluminum 6061");
        assert!((material.e - 68.9e9).abs() < 1.0);
        assert!((material.rho - 2700.0).abs() < 0.1);
        assert!((material.nu - 0.33).abs() < 0.001);
        assert_eq!(material.category, MaterialCategory::Metal);
    }

    #[test]
    fn test_get_material_invalid() {
        assert!(get_material("invalid_material").is_none());
        assert!(get_material("").is_none());
    }

    #[test]
    fn test_get_materials_by_category() {
        let by_category = get_materials_by_category();

        let metals = by_category.get("metals").unwrap();
        let woods = by_category.get("woods").unwrap();

        // Check counts
        assert_eq!(metals.len(), 8);
        assert_eq!(woods.len(), 13);

        // Check all metals have Metal category
        for (_, material) in metals {
            assert_eq!(material.category, MaterialCategory::Metal);
        }

        // Check all woods have Wood category
        for (_, material) in woods {
            assert_eq!(material.category, MaterialCategory::Wood);
        }
    }

    #[test]
    fn test_get_all_materials() {
        let materials = get_all_materials();
        assert_eq!(materials.len(), 21);
    }

    #[test]
    fn test_calculate_shear_modulus() {
        // G = E / (2 * (1 + nu))
        // For E = 12.0e9, nu = 0.35:
        // G = 12.0e9 / (2 * 1.35) = 12.0e9 / 2.7 ≈ 4.444e9
        let g = calculate_shear_modulus(12.0e9, 0.35);
        let expected = 12.0e9 / 2.7;
        assert!((g - expected).abs() < 1.0);
    }

    #[test]
    fn test_kappa_constant() {
        assert!((KAPPA - 5.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_all_material_keys_valid() {
        for &key in MATERIAL_KEYS {
            assert!(
                get_material(key).is_some(),
                "Material key '{}' should be valid",
                key
            );
        }
    }
}
