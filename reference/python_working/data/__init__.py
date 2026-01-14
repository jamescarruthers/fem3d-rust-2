"""Data module containing materials and tuning presets."""

from .materials import MATERIALS, get_material, get_materials_by_category, calculate_shear_modulus, KAPPA
from .presets import (
    TUNING_PRESETS,
    get_preset,
    calculate_target_frequencies,
    frequency_to_cents,
    UNIFORM_BAR_BETAS,
    UNIFORM_BAR_RATIOS,
)

__all__ = [
    "MATERIALS",
    "get_material",
    "get_materials_by_category",
    "calculate_shear_modulus",
    "KAPPA",
    "TUNING_PRESETS",
    "get_preset",
    "calculate_target_frequencies",
    "frequency_to_cents",
    "UNIFORM_BAR_BETAS",
    "UNIFORM_BAR_RATIOS",
]
