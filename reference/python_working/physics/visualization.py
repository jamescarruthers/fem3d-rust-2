"""
3D Mesh Visualization for Undercut Bars

Provides functions to visualize the 3D hexahedral mesh generated
for FEM analysis of undercut percussion bars.
"""

from typing import List, Optional, Tuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def visualize_bar_mesh(
    nodes: np.ndarray,
    elements: np.ndarray,
    title: str = "3D Bar Mesh",
    show_edges: bool = True,
    alpha: float = 0.3,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Visualize the 3D hexahedral mesh.

    Args:
        nodes: (num_nodes, 3) array of node coordinates
        elements: (num_elements, 8) array of node indices
        title: Plot title
        show_edges: Whether to show element edges
        alpha: Face transparency (0-1)
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available for visualization")
        return None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Define the 6 faces of a hexahedron using local node indices
    # Each face is defined by 4 corner nodes
    hex_faces = [
        [0, 1, 2, 3],  # bottom (z-)
        [4, 5, 6, 7],  # top (z+)
        [0, 1, 5, 4],  # front (y-)
        [2, 3, 7, 6],  # back (y+)
        [0, 3, 7, 4],  # left (x-)
        [1, 2, 6, 5],  # right (x+)
    ]

    faces = []
    for elem in elements:
        elem_nodes = nodes[elem]
        for face_idx in hex_faces:
            face_coords = elem_nodes[face_idx]
            faces.append(face_coords)

    # Create 3D polygon collection
    mesh = Poly3DCollection(
        faces,
        alpha=alpha,
        facecolor='steelblue',
        edgecolor='darkblue' if show_edges else None,
        linewidth=0.5 if show_edges else 0
    )
    ax.add_collection3d(mesh)

    # Set axis limits based on node coordinates
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    z_min, z_max = nodes[:, 2].min(), nodes[:, 2].max()

    # Add some padding
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    z_pad = (z_max - z_min) * 0.1

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_zlim(z_min - z_pad, z_max + z_pad)

    ax.set_xlabel('X (Length) [m]')
    ax.set_ylabel('Y (Width) [m]')
    ax.set_zlabel('Z (Height) [m]')
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    ax.set_box_aspect([
        (x_max - x_min) / max_range,
        (y_max - y_min) / max_range,
        (z_max - z_min) / max_range
    ])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_bar_profile(
    element_heights: List[float],
    length: float,
    h0: float,
    title: str = "Bar Profile (Side View)",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Visualize the bar profile as a 2D side view.

    Args:
        element_heights: Height at each element position
        length: Bar length (m)
        h0: Original bar height (m)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available for visualization")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    n = len(element_heights)
    dx = length / n

    # Create x coordinates for element boundaries
    x = np.linspace(0, length, n + 1)

    # Create the profile polygon
    # Bottom edge (undercut profile)
    bottom_x = []
    bottom_y = []
    for i in range(n):
        h = element_heights[i]
        bottom_x.extend([x[i], x[i + 1]])
        bottom_y.extend([h0 - h, h0 - h])

    # Top edge (flat at h0)
    top_x = [length, 0]
    top_y = [h0, h0]

    # Combine for filled polygon
    poly_x = [0] + bottom_x + top_x
    poly_y = [h0] + bottom_y + top_y

    ax.fill(poly_x, poly_y, color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)

    # Add reference lines
    ax.axhline(y=h0, color='gray', linestyle='--', linewidth=0.5, label=f'Original height: {h0*1000:.1f} mm')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Convert to mm for display
    ax.set_xlabel('Length [m]')
    ax.set_ylabel('Height [m]')
    ax.set_title(title)
    ax.set_xlim(-length * 0.05, length * 1.05)
    ax.set_ylim(-h0 * 0.1, h0 * 1.2)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_mesh_cross_section(
    nodes: np.ndarray,
    elements: np.ndarray,
    x_position: float,
    title: str = "Cross Section",
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """
    Visualize a cross-section of the mesh at a given x position.

    Args:
        nodes: (num_nodes, 3) array of node coordinates
        elements: (num_elements, 8) array of node indices
        x_position: X coordinate for the cross-section
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        matplotlib Figure object, or None if matplotlib unavailable
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available for visualization")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Find elements that span the x_position
    for elem in elements:
        elem_nodes = nodes[elem]
        x_coords = elem_nodes[:, 0]

        if x_coords.min() <= x_position <= x_coords.max():
            # This element spans the cross-section
            # Project the y-z face
            y_coords = elem_nodes[:, 1]
            z_coords = elem_nodes[:, 2]

            # Draw the element outline (simplified - just the bounding box in y-z)
            y_min, y_max = y_coords.min(), y_coords.max()
            z_min, z_max = z_coords.min(), z_coords.max()

            rect = plt.Rectangle(
                (y_min, z_min), y_max - y_min, z_max - z_min,
                fill=True, facecolor='steelblue', edgecolor='darkblue',
                alpha=0.5, linewidth=0.5
            )
            ax.add_patch(rect)

    ax.set_xlabel('Y (Width) [m]')
    ax.set_ylabel('Z (Height) [m]')
    ax.set_title(f"{title} at x = {x_position:.3f} m")
    ax.set_aspect('equal')
    ax.autoscale()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig
