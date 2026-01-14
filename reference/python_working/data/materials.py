"""
Material properties for bar tuning optimization.

Contains properties for common metals and woods used in percussion instruments.
"""

from typing import Dict, Tuple, List, Optional
from ..types import Material


# Shear correction factor for rectangular cross-section
KAPPA = 5.0 / 6.0


# Material database
MATERIALS: Dict[str, Material] = {
    # Metals commonly used in vibraphones, glockenspiels, metallophones
    "aluminum": Material(
        name="Aluminum 6061",
        E=68.9e9,      # Pa
        rho=2700,      # kg/m^3
        nu=0.33,
        category="metal"
    ),
    "aluminum7075": Material(
        name="Aluminum 7075",
        E=71.7e9,
        rho=2810,
        nu=0.33,
        category="metal"
    ),
    "brass": Material(
        name="Brass C260",
        E=110e9,
        rho=8530,
        nu=0.35,
        category="metal"
    ),
    "steel": Material(
        name="Steel 1018",
        E=205e9,
        rho=7870,
        nu=0.29,
        category="metal"
    ),
    "stainless_steel": Material(
        name="Stainless Steel 304",
        E=193e9,
        rho=8000,
        nu=0.29,
        category="metal"
    ),
    "bronze": Material(
        name="Phosphor Bronze",
        E=110e9,
        rho=8800,
        nu=0.34,
        category="metal"
    ),
    "bell_bronze": Material(
        name="Bell Bronze (B20)",
        E=100e9,
        rho=8600,
        nu=0.34,
        category="metal"
    ),

    # Premium tonewoods for marimbas/xylophones
    "rosewood": Material(
        name="Honduran Rosewood",
        E=12.5e9,
        rho=850,
        nu=0.37,
        category="wood"
    ),
    "african_rosewood": Material(
        name="African Rosewood (Bubinga)",
        E=15.8e9,
        rho=890,
        nu=0.36,
        category="wood"
    ),
    "padauk": Material(
        name="African Padauk",
        E=11.7e9,
        rho=750,
        nu=0.35,
        category="wood"
    ),
    "sapele": Material(
        name="Sapele",
        E=12.0e9,
        rho=640,
        nu=0.35,
        category="wood"
    ),
    "bubinga": Material(
        name="Bubinga",
        E=15.8e9,
        rho=890,
        nu=0.36,
        category="wood"
    ),

    # Other tonewoods
    "maple": Material(
        name="Hard Maple",
        E=12.6e9,
        rho=705,
        nu=0.35,
        category="wood"
    ),
    "purpleheart": Material(
        name="Purpleheart",
        E=17.0e9,
        rho=880,
        nu=0.35,
        category="wood"
    ),
    "wenge": Material(
        name="Wenge",
        E=14.0e9,
        rho=870,
        nu=0.35,
        category="wood"
    ),
    "bocote": Material(
        name="Bocote",
        E=14.1e9,
        rho=930,
        nu=0.36,
        category="wood"
    ),
    "zebrawood": Material(
        name="Zebrawood",
        E=15.2e9,
        rho=780,
        nu=0.35,
        category="wood"
    ),
    "cocobolo": Material(
        name="Cocobolo",
        E=14.1e9,
        rho=1100,
        nu=0.36,
        category="wood"
    ),
    "ebony": Material(
        name="African Ebony",
        E=17.4e9,
        rho=1050,
        nu=0.38,
        category="wood"
    ),
    "teak": Material(
        name="Teak",
        E=12.3e9,
        rho=630,
        nu=0.35,
        category="wood"
    ),

    # Synthetic alternatives
    "fiberglass": Material(
        name="Fiberglass Composite",
        E=17.0e9,
        rho=1800,
        nu=0.30,
        category="metal"  # grouped with metals for categorization
    ),
}


def get_material(key: str) -> Optional[Material]:
    """Get material by key."""
    return MATERIALS.get(key)


def get_materials_by_category() -> Dict[str, List[Tuple[str, Material]]]:
    """Get all materials grouped by category."""
    metals = [(k, m) for k, m in MATERIALS.items() if m.category == "metal"]
    woods = [(k, m) for k, m in MATERIALS.items() if m.category == "wood"]
    return {"metals": metals, "woods": woods}


def calculate_shear_modulus(E: float, nu: float) -> float:
    """Calculate shear modulus from Young's modulus and Poisson's ratio."""
    return E / (2 * (1 + nu))
