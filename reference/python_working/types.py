"""
Type definitions for the multi-modal bar tuning optimization.

Uses dataclasses for clean, typed structures that mirror the TypeScript interfaces.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum
import math


class AnalysisMode(Enum):
    """FEM analysis mode selection."""
    BEAM_2D = "2d"      # Timoshenko beam elements (fast, good for slender bars)
    SOLID_3D = "3d"     # 3D hexahedral elements (accurate, slower)


@dataclass
class Material:
    """Material properties for bar tuning."""
    name: str
    E: float        # Young's modulus (Pa)
    rho: float      # Density (kg/m^3)
    nu: float       # Poisson's ratio
    category: Literal['metal', 'wood']


@dataclass
class BarParameters:
    """Bar geometry parameters."""
    L: float        # Length (m)
    b: float        # Width (m)
    h0: float       # Original thickness (m)
    hMin: float     # Minimum thickness (m) - typically 10% of h0


@dataclass
class Cut:
    """Single rectangular cut."""
    lambda_: float  # Length from center (m) - using lambda_ to avoid Python keyword
    h: float        # Height at cut (m)


@dataclass
class TuningPreset:
    """Tuning preset with frequency ratios."""
    name: str
    ratios: List[float]
    description: str
    instrument: str


@dataclass
class EAParameters:
    """Evolutionary algorithm parameters."""
    population_size: int = 50         # Npop
    elitism_percent: float = 10.0     # Pe (0-100)
    crossover_percent: float = 30.0   # Pc (0-100)
    mutation_percent: float = 60.0    # Pm (0-100, Pe + Pc + Pm = 100)
    mutation_strength: float = 0.1    # sigma for uniform mutation (0.05-0.2)
    max_generations: int = 100
    target_error: float = 0.01        # Stopping criterion (percentage)
    num_elements: int = 150           # Ne for FEM discretization (x-direction for 3D)
    f1_priority: float = 1.0          # Weight multiplier for f1
    min_cut_width: float = 0.0        # Minimum width between cut boundaries (m)
    max_cut_width: float = 0.0        # Maximum cut width (2*lambda) (m), 0 = no limit
    min_cut_depth: float = 0.0        # Minimum cut depth (h0 - h) (m), 0 = no limit
    max_cut_depth: float = 0.0        # Maximum cut depth (h0 - h) (m), 0 = no limit
    max_length_trim: float = 0.0      # Max trim from each end (m), 0 = no trimming
    max_length_extend: float = 0.0    # Max extension from each end (m), 0 = no extension
    max_workers: int = 0              # Max worker threads (0 = auto)
    # Analysis mode selection
    analysis_mode: AnalysisMode = AnalysisMode.BEAM_2D
    # 3D mesh parameters (only used when analysis_mode is SOLID_3D)
    num_elements_y: int = 2           # Elements in width direction
    num_elements_z: int = 2           # Elements in thickness direction
    # Frequency offset for 2D/3D calibration (e.g., 0.05 = target 5% higher)
    # Applied as: effective_target = target * (1 + offset)
    frequency_offset: float = 0.0


@dataclass
class Individual:
    """Individual in EA population."""
    genes: List[float]                # [lambda_1, h_1, lambda_2, h_2, ..., length_adjust?]
    fitness: float = math.inf
    sigmas: Optional[List[float]] = None  # For self-adaptive Gaussian mutation


@dataclass
class OptimizationResult:
    """Result of optimization."""
    best_individual: Individual
    cuts: List[Cut]
    computed_frequencies: List[float]
    target_frequencies: List[float]
    tuning_error: float               # epsilon (%)
    max_error_cents: float
    errors_in_cents: List[float]
    volume_percent: float
    roughness_percent: float
    generations: int
    length_trim: float = 0.0          # How much trimmed from each end (m)
    effective_length: float = 0.0     # L - 2*length_trim (m)


@dataclass
class ProgressUpdate:
    """Progress update during optimization."""
    generation: int
    best_fitness: float
    best_individual: Individual
    average_fitness: float
    computed_frequencies: Optional[List[float]] = None
    errors_in_cents: Optional[List[float]] = None
    length_trim: float = 0.0


@dataclass
class VariableBounds:
    """Bounds for optimization variables."""
    lambda_min: float
    lambda_max: float
    h_min: float
    h_max: float
    min_cut_width: float = 0.0        # Minimum distance between cut boundaries
    max_cut_width: float = 0.0        # Maximum cut width (2*lambda), 0 = use lambda_max
    min_cut_depth: float = 0.0        # Minimum depth (constrains h_max = h0 - min_depth)
    max_cut_depth: float = 0.0        # Maximum depth (constrains h_min = h0 - max_depth)
    max_length_trim: float = 0.0      # Maximum trim from each end, 0 = no trimming
    max_length_extend: float = 0.0    # Maximum extension from each end, 0 = no extension


@dataclass
class DetailedEvaluation:
    """Detailed evaluation results for an individual."""
    computed_frequencies: List[float]
    target_frequencies: List[float]
    tuning_error: float
    volume_penalty: float
    roughness_penalty: float
    combined_fitness: float
    cents_errors: List[float]
    max_cents_error: float
