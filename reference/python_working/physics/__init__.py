"""Physics module for FEM and frequency computation.

Supports both 2D Timoshenko beam elements and 3D solid hexahedral elements.
"""

from .bar_profile import (
    compute_height,
    generate_element_heights,
    genes_to_cuts,
    cuts_to_genes,
    count_effective_cuts,
    validate_cuts,
    generate_profile_points,
)

from .timoshenko import (
    compute_element_stiffness,
    compute_element_mass,
)

from .fem_assembly import (
    assemble_global_matrices,
    solve_generalized_eigenvalue,
)

from .fem_3d import (
    compute_frequencies_3d,
    generate_bar_mesh_3d,
    assemble_global_matrices_3d,
    solve_eigenvalue_3d,
)

from .frequencies import (
    compute_frequencies,
    compute_frequencies_from_genes,
    batch_compute_fitness,
)

__all__ = [
    # Bar profile
    "compute_height",
    "generate_element_heights",
    "genes_to_cuts",
    "cuts_to_genes",
    "count_effective_cuts",
    "validate_cuts",
    "generate_profile_points",
    # Timoshenko (2D)
    "compute_element_stiffness",
    "compute_element_mass",
    # FEM assembly (2D)
    "assemble_global_matrices",
    "solve_generalized_eigenvalue",
    # FEM 3D
    "compute_frequencies_3d",
    "generate_bar_mesh_3d",
    "assemble_global_matrices_3d",
    "solve_eigenvalue_3d",
    # Frequencies (unified interface)
    "compute_frequencies",
    "compute_frequencies_from_genes",
    "batch_compute_fitness",
]
