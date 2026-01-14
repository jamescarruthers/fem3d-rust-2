"""
Bar Diagram Visualization

Generates 2D profile and 3D isometric diagrams for tuned percussion bars.
Uses matplotlib for rendering and exports to PNG.
"""

from typing import List, Tuple, Optional
import os
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from ..types import BarParameters, Cut
from ..physics.bar_profile import generate_profile_points


def generate_2d_profile_diagram(
    bar: BarParameters,
    cuts: List[Cut],
    note_name: str,
    frequencies: List[float],
    target_frequencies: List[float],
    output_path: str,
    title: Optional[str] = None
) -> str:
    """
    Generate a 2D side profile diagram of the bar showing the undercut.

    Args:
        bar: Bar parameters (L, b, h0, hMin)
        cuts: List of cuts defining the undercut profile
        note_name: Note name for labeling (e.g., "F4")
        frequencies: Computed frequencies [f1, f2, f3, ...]
        target_frequencies: Target frequencies for comparison
        output_path: Path to save the PNG file
        title: Optional custom title

    Returns:
        Path to the saved image
    """
    # Convert to mm for display
    L_mm = bar.L * 1000
    h0_mm = bar.h0 * 1000

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate profile points
    profile_points = generate_profile_points(cuts, bar.L, bar.h0, num_points=500)

    # Convert to mm and create arrays
    x_profile = np.array([p[0] * 1000 for p in profile_points])
    h_profile = np.array([p[1] * 1000 for p in profile_points])

    # Draw the bar outline (full rectangle at h0)
    ax.fill_between([0, L_mm], [0, 0], [h0_mm, h0_mm],
                    color='lightgray', alpha=0.3, label='Material removed')

    # Draw the undercut profile (filled area representing remaining material)
    ax.fill_between(x_profile, np.zeros_like(h_profile), h_profile,
                    color='#8B4513', alpha=0.8, label='Bar profile')

    # Draw outline
    ax.plot(x_profile, h_profile, 'k-', linewidth=1.5)
    ax.plot([0, 0], [0, h0_mm], 'k-', linewidth=1.5)
    ax.plot([L_mm, L_mm], [0, h0_mm], 'k-', linewidth=1.5)
    ax.plot([0, L_mm], [h0_mm, h0_mm], 'k-', linewidth=1.5)
    ax.plot([0, L_mm], [0, 0], 'k-', linewidth=1.5)

    # Add dimension lines
    _add_dimension_line(ax, 0, -3, L_mm, -3, f'{L_mm:.1f} mm', 'below')
    _add_dimension_line(ax, L_mm + 5, 0, L_mm + 5, h0_mm, f'{h0_mm:.1f} mm', 'right')

    # Add cut dimensions
    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)
    y_offset = h0_mm + 3
    for i, cut in enumerate(sorted_cuts):
        lambda_mm = cut.lambda_ * 1000
        h_mm = cut.h * 1000
        depth_mm = h0_mm - h_mm
        width_mm = lambda_mm * 2

        # Draw dimension for cut width
        center = L_mm / 2
        left_edge = center - lambda_mm
        right_edge = center + lambda_mm

        color = f'C{i}'
        ax.plot([left_edge, left_edge], [h_mm, h0_mm + 2], '--', color=color, linewidth=0.8, alpha=0.7)
        ax.plot([right_edge, right_edge], [h_mm, h0_mm + 2], '--', color=color, linewidth=0.8, alpha=0.7)

        # Add cut info text
        ax.annotate(f'Cut {i+1}: {width_mm:.1f}mm wide, {depth_mm:.1f}mm deep',
                   xy=(center, y_offset), fontsize=8, ha='center', color=color)
        y_offset += 2.5

    # Add centerline
    ax.axvline(x=L_mm/2, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.annotate('CL', xy=(L_mm/2, h0_mm + 1), fontsize=8, ha='center', color='gray')

    # Add frequency information box
    freq_text = f'{note_name}\n'
    for i, (f, ft) in enumerate(zip(frequencies, target_frequencies)):
        error_cents = 1200 * math.log2(f / ft) if ft > 0 else 0
        sign = '+' if error_cents >= 0 else ''
        freq_text += f'f{i+1}: {f:.1f} Hz ({sign}{error_cents:.1f}¢)\n'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, freq_text.strip(), transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    # Set title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{note_name} Bar - 2D Profile', fontsize=12, fontweight='bold')

    ax.set_xlabel('Length (mm)')
    ax.set_ylabel('Height (mm)')
    ax.set_xlim(-15, L_mm + 25)
    ax.set_ylim(-10, h0_mm + 15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return output_path


def generate_3d_isometric_diagram(
    bar: BarParameters,
    cuts: List[Cut],
    note_name: str,
    frequencies: List[float],
    target_frequencies: List[float],
    output_path: str,
    title: Optional[str] = None
) -> str:
    """
    Generate a 3D isometric diagram of the bar showing the undercut.

    Uses pyvista for proper 3D solid rendering.

    Args:
        bar: Bar parameters (L, b, h0, hMin)
        cuts: List of cuts defining the undercut profile
        note_name: Note name for labeling (e.g., "F4")
        frequencies: Computed frequencies [f1, f2, f3, ...]
        target_frequencies: Target frequencies for comparison
        output_path: Path to save the PNG file
        title: Optional custom title

    Returns:
        Path to the saved image
    """
    import pyvista as pv

    # Convert to mm for display
    L_mm = bar.L * 1000
    b_mm = bar.b * 1000
    h0_mm = bar.h0 * 1000

    # Generate profile for the undercut
    profile_points = generate_profile_points(cuts, bar.L, bar.h0, num_points=100)

    # Convert profile to mm
    x_profile = np.array([p[0] * 1000 for p in profile_points])
    z_profile = np.array([p[1] * 1000 for p in profile_points])

    # Create the bar mesh by extruding the profile along the width (y-axis)
    # Build vertices and faces for the solid bar

    n = len(x_profile)

    # Create vertices: front face (y=0) and back face (y=b_mm)
    # For each x position, we have 3 z levels: 0, profile height, h0
    vertices = []

    # Front face vertices (y=0)
    for i in range(n):
        vertices.append([x_profile[i], 0, 0])           # bottom
        vertices.append([x_profile[i], 0, z_profile[i]]) # profile
        vertices.append([x_profile[i], 0, h0_mm])        # top

    # Back face vertices (y=b_mm)
    for i in range(n):
        vertices.append([x_profile[i], b_mm, 0])           # bottom
        vertices.append([x_profile[i], b_mm, z_profile[i]]) # profile
        vertices.append([x_profile[i], b_mm, h0_mm])        # top

    vertices = np.array(vertices)

    # Build faces
    faces = []

    # Front face (y=0) - quads between profile points
    for i in range(n - 1):
        # Bottom section: from z=0 to profile
        v0 = i * 3          # current bottom
        v1 = (i + 1) * 3    # next bottom
        v2 = (i + 1) * 3 + 1  # next profile
        v3 = i * 3 + 1      # current profile
        faces.append([4, v0, v1, v2, v3])

        # Top section: from profile to h0
        v0 = i * 3 + 1      # current profile
        v1 = (i + 1) * 3 + 1  # next profile
        v2 = (i + 1) * 3 + 2  # next top
        v3 = i * 3 + 2      # current top
        faces.append([4, v0, v1, v2, v3])

    # Back face (y=b_mm) - quads between profile points
    back_offset = n * 3
    for i in range(n - 1):
        # Bottom section
        v0 = back_offset + (i + 1) * 3
        v1 = back_offset + i * 3
        v2 = back_offset + i * 3 + 1
        v3 = back_offset + (i + 1) * 3 + 1
        faces.append([4, v0, v1, v2, v3])

        # Top section
        v0 = back_offset + (i + 1) * 3 + 1
        v1 = back_offset + i * 3 + 1
        v2 = back_offset + i * 3 + 2
        v3 = back_offset + (i + 1) * 3 + 2
        faces.append([4, v0, v1, v2, v3])

    # Top face (z=h0) - single quad
    v0 = 2  # front left top
    v1 = (n - 1) * 3 + 2  # front right top
    v2 = back_offset + (n - 1) * 3 + 2  # back right top
    v3 = back_offset + 2  # back left top
    faces.append([4, v0, v1, v2, v3])

    # Bottom face (z=0) - single quad
    v0 = (n - 1) * 3  # front right bottom
    v1 = 0  # front left bottom
    v2 = back_offset  # back left bottom
    v3 = back_offset + (n - 1) * 3  # back right bottom
    faces.append([4, v0, v1, v2, v3])

    # Undercut surface (connects front and back profile lines)
    for i in range(n - 1):
        v0 = i * 3 + 1  # front current profile
        v1 = (i + 1) * 3 + 1  # front next profile
        v2 = back_offset + (i + 1) * 3 + 1  # back next profile
        v3 = back_offset + i * 3 + 1  # back current profile
        faces.append([4, v0, v1, v2, v3])

    # Left end face (x=0)
    v0 = 0  # front bottom
    v1 = 1  # front profile
    v2 = 2  # front top
    v3 = back_offset + 2  # back top
    v4 = back_offset + 1  # back profile
    v5 = back_offset  # back bottom
    # Split into two quads
    faces.append([4, v0, v1, v4, v5])  # bottom part
    faces.append([4, v1, v2, v3, v4])  # top part

    # Right end face (x=L)
    v0 = (n - 1) * 3  # front bottom
    v1 = (n - 1) * 3 + 1  # front profile
    v2 = (n - 1) * 3 + 2  # front top
    v3 = back_offset + (n - 1) * 3 + 2  # back top
    v4 = back_offset + (n - 1) * 3 + 1  # back profile
    v5 = back_offset + (n - 1) * 3  # back bottom
    faces.append([4, v5, v4, v1, v0])  # bottom part
    faces.append([4, v4, v3, v2, v1])  # top part

    # Create the mesh
    faces_flat = np.hstack(faces)
    mesh = pv.PolyData(vertices, faces_flat)

    # Set up the plotter
    pv.global_theme.background = 'white'
    plotter = pv.Plotter(off_screen=True, window_size=[1400, 1000])

    # Add the mesh with wood color
    plotter.add_mesh(mesh, color='#A0522D', show_edges=True, edge_color='#4a2511',
                     lighting=True, smooth_shading=False)

    # Set camera for isometric-ish view from below to see undercut
    # Position camera below and to the side for good view of undercut
    plotter.camera_position = [
        (L_mm * 0.7, -b_mm * 8, -h0_mm * 15),  # camera position (below bar)
        (L_mm / 2, b_mm / 2, h0_mm / 3),        # focal point (slightly below center)
        (0, 0, 1)                                # up vector
    ]

    # Add axes
    plotter.add_axes()

    # Build info text
    freq_text = f'{note_name}\n'
    freq_text += f'L={L_mm:.1f}  W={b_mm:.1f}  H={h0_mm:.1f} mm\n\n'
    for i, (f, ft) in enumerate(zip(frequencies, target_frequencies)):
        error_cents = 1200 * math.log2(f / ft) if ft > 0 else 0
        sign = '+' if error_cents >= 0 else ''
        freq_text += f'f{i+1}: {f:.1f} Hz ({sign}{error_cents:.1f} cents)\n'

    sorted_cuts = sorted(cuts, key=lambda c: c.lambda_, reverse=True)
    freq_text += '\nCuts:\n'
    for i, cut in enumerate(sorted_cuts):
        lambda_mm = cut.lambda_ * 1000
        depth_mm = h0_mm - cut.h * 1000
        freq_text += f'  {i+1}: {lambda_mm*2:.1f}mm wide, {depth_mm:.1f}mm deep\n'

    # Add text annotation
    plotter.add_text(freq_text, position='upper_left', font_size=10,
                     color='black', font='courier')

    # Add title
    plot_title = title if title else f'{note_name} Bar - 3D View'
    plotter.add_text(plot_title, position='upper_edge', font_size=14,
                     color='black')

    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plotter.screenshot(output_path)
    plotter.close()

    return output_path


def generate_bar_diagrams(
    bar: BarParameters,
    cuts: List[Cut],
    note_name: str,
    frequencies: List[float],
    target_frequencies: List[float],
    output_dir: str
) -> Tuple[str, str]:
    """
    Generate both 2D and 3D diagrams for a bar.

    Args:
        bar: Bar parameters
        cuts: List of cuts
        note_name: Note name (e.g., "F4")
        frequencies: Computed frequencies
        target_frequencies: Target frequencies
        output_dir: Directory to save diagrams

    Returns:
        Tuple of (2d_path, 3d_path)
    """
    # Clean note name for filename (replace # with 's' for sharp)
    safe_name = note_name.replace('#', 's').replace('b', 'b')

    path_2d = os.path.join(output_dir, f'{safe_name}_2d_profile.png')
    path_3d = os.path.join(output_dir, f'{safe_name}_3d_isometric.png')

    generate_2d_profile_diagram(bar, cuts, note_name, frequencies, target_frequencies, path_2d)
    generate_3d_isometric_diagram(bar, cuts, note_name, frequencies, target_frequencies, path_3d)

    return path_2d, path_3d


def _add_dimension_line(ax, x1, y1, x2, y2, text, position='below'):
    """Add a dimension line with text."""
    arrow_props = dict(arrowstyle='<->', color='black', lw=0.8)

    if position == 'below':
        ax.annotate('', xy=(x2, y1), xytext=(x1, y1), arrowprops=arrow_props)
        ax.text((x1 + x2) / 2, y1 - 2, text, ha='center', va='top', fontsize=8)
    elif position == 'right':
        ax.annotate('', xy=(x1, y2), xytext=(x1, y1), arrowprops=arrow_props)
        ax.text(x1 + 2, (y1 + y2) / 2, text, ha='left', va='center', fontsize=8, rotation=90)
    elif position == 'above':
        ax.annotate('', xy=(x2, y1), xytext=(x1, y1), arrowprops=arrow_props)
        ax.text((x1 + x2) / 2, y1 + 2, text, ha='center', va='bottom', fontsize=8)


