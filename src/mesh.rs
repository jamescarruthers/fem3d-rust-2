//! 3D mesh representation and generation for bar structures.
//!
//! This module provides the `Mesh3d` struct and functions for generating
//! uniform and adaptive meshes for bar geometries.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// Mesh representation for the 3D bar.
#[derive(Debug, Clone)]
pub struct Mesh3d {
    pub nodes: Vec<Vector3<f64>>,
    pub elements: Vec<[usize; 8]>,
    pub heights_per_element: Vec<f64>,
}

/// Serializable 3D node position for frontend visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableNode {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Serializable hexahedral element for frontend visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableElement {
    /// Indices of the 8 nodes that form this hexahedral element.
    /// Node ordering: bottom face (z-), then top face (z+), counterclockwise.
    pub nodes: [usize; 8],
}

/// Serializable mesh data for frontend visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableMesh {
    /// Node positions in 3D space.
    pub nodes: Vec<SerializableNode>,
    /// Hexahedral elements defined by node indices.
    pub elements: Vec<SerializableElement>,
    /// Height value for each element (useful for coloring/visualization).
    pub element_heights: Vec<f64>,
    /// Mesh metadata.
    pub metadata: MeshMetadata,
}

/// Metadata about the mesh for frontend display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshMetadata {
    /// Total number of nodes.
    pub num_nodes: usize,
    /// Total number of elements.
    pub num_elements: usize,
    /// Total degrees of freedom (3 * num_nodes).
    pub num_dof: usize,
    /// Bounding box minimum coordinates.
    pub bbox_min: [f64; 3],
    /// Bounding box maximum coordinates.
    pub bbox_max: [f64; 3],
}

impl Mesh3d {
    /// Convert the mesh to a serializable format for frontend visualization.
    ///
    /// # Returns
    /// A `SerializableMesh` that can be serialized to JSON for frontend consumption.
    ///
    /// # Example
    /// ```
    /// use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d};
    ///
    /// let cuts = vec![Cut::new(0.1, 0.015)];
    /// let heights = generate_element_heights(&cuts, 0.5, 0.024, 20);
    /// let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 20, 2, 2);
    ///
    /// let serializable = mesh.to_serializable();
    /// let json = serde_json::to_string(&serializable).unwrap();
    /// ```
    pub fn to_serializable(&self) -> SerializableMesh {
        // Convert nodes
        let nodes: Vec<SerializableNode> = self
            .nodes
            .iter()
            .map(|v| SerializableNode {
                x: v.x,
                y: v.y,
                z: v.z,
            })
            .collect();

        // Convert elements
        let elements: Vec<SerializableElement> = self
            .elements
            .iter()
            .map(|&elem| SerializableElement { nodes: elem })
            .collect();

        // Compute bounding box
        let mut bbox_min = [f64::INFINITY; 3];
        let mut bbox_max = [f64::NEG_INFINITY; 3];

        for node in &self.nodes {
            bbox_min[0] = bbox_min[0].min(node.x);
            bbox_min[1] = bbox_min[1].min(node.y);
            bbox_min[2] = bbox_min[2].min(node.z);
            bbox_max[0] = bbox_max[0].max(node.x);
            bbox_max[1] = bbox_max[1].max(node.y);
            bbox_max[2] = bbox_max[2].max(node.z);
        }

        let metadata = MeshMetadata {
            num_nodes: self.nodes.len(),
            num_elements: self.elements.len(),
            num_dof: self.nodes.len() * 3,
            bbox_min,
            bbox_max,
        };

        SerializableMesh {
            nodes,
            elements,
            element_heights: self.heights_per_element.clone(),
            metadata,
        }
    }

    /// Export mesh to JSON string for frontend visualization.
    ///
    /// # Returns
    /// A JSON string representation of the mesh.
    ///
    /// # Example
    /// ```
    /// use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d};
    ///
    /// let cuts = vec![Cut::new(0.1, 0.015)];
    /// let heights = generate_element_heights(&cuts, 0.5, 0.024, 20);
    /// let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 20, 2, 2);
    ///
    /// let json = mesh.to_json().unwrap();
    /// println!("{}", json);
    /// ```
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let serializable = self.to_serializable();
        serde_json::to_string(&serializable)
    }

    /// Export mesh to pretty-printed JSON string for frontend visualization.
    ///
    /// # Returns
    /// A pretty-printed JSON string representation of the mesh.
    ///
    /// # Example
    /// ```
    /// use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d};
    ///
    /// let cuts = vec![Cut::new(0.1, 0.015)];
    /// let heights = generate_element_heights(&cuts, 0.5, 0.024, 20);
    /// let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 20, 2, 2);
    ///
    /// let json = mesh.to_json_pretty().unwrap();
    /// println!("{}", json);
    /// ```
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        let serializable = self.to_serializable();
        serde_json::to_string_pretty(&serializable)
    }
}

/// Generate uniform 3D mesh following the Python reference.
/// Uses direct indexing instead of HashMap for better performance.
pub fn generate_bar_mesh_3d(
    length: f64,
    width: f64,
    element_heights: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> Mesh3d {
    let expected_nx = element_heights.len();
    assert_eq!(nx, expected_nx, "nx must match element_heights length");
    let dx = length / expected_nx as f64;
    let dy = width / ny as f64;

    let nnx = expected_nx + 1;
    let nny = ny + 1;
    let nnz = nz + 1;

    // Direct indexing function - O(1) instead of HashMap lookup
    // Node index = ix * (nny * nnz) + iy * nnz + iz
    let node_idx = |ix: usize, iy: usize, iz: usize| -> usize {
        ix * (nny * nnz) + iy * nnz + iz
    };

    let mut nodes = Vec::with_capacity(nnx * nny * nnz);

    for ix in 0..nnx {
        let x = ix as f64 * dx;
        let h = if ix == 0 {
            element_heights[0]
        } else if ix == expected_nx {
            element_heights[element_heights.len() - 1]
        } else {
            (element_heights[ix - 1] + element_heights[ix]) / 2.0
        };
        let dz = h / nz as f64;

        for iy in 0..nny {
            let y = iy as f64 * dy;
            for iz in 0..nnz {
                let z = iz as f64 * dz;
                nodes.push(Vector3::new(x, y, z));
            }
        }
    }

    let mut elements = Vec::with_capacity(nx * ny * nz);
    let mut heights = Vec::with_capacity(nx * ny * nz);

    for ix in 0..expected_nx {
        let h = element_heights[ix];
        for iy in 0..ny {
            for iz in 0..nz {
                let n0 = node_idx(ix, iy, iz);
                let n1 = node_idx(ix + 1, iy, iz);
                let n2 = node_idx(ix + 1, iy + 1, iz);
                let n3 = node_idx(ix, iy + 1, iz);
                let n4 = node_idx(ix, iy, iz + 1);
                let n5 = node_idx(ix + 1, iy, iz + 1);
                let n6 = node_idx(ix + 1, iy + 1, iz + 1);
                let n7 = node_idx(ix, iy + 1, iz + 1);

                elements.push([n0, n1, n2, n3, n4, n5, n6, n7]);
                heights.push(h);
            }
        }
    }

    Mesh3d {
        nodes,
        elements,
        heights_per_element: heights,
    }
}

/// Generate adaptive mesh using supplied x-positions.
/// Uses direct indexing instead of HashMap for better performance.
pub fn generate_bar_mesh_3d_adaptive(
    length: f64,
    width: f64,
    x_positions: &[f64],
    element_heights: &[f64],
    ny: usize,
    nz: usize,
) -> Mesh3d {
    let nx = element_heights.len();
    assert_eq!(
        x_positions.len(),
        nx + 1,
        "x_positions must have length len(element_heights)+1"
    );
    let end = *x_positions
        .last()
        .expect("x_positions must contain at least one position");
    assert!(
        (end - length).abs() < 1e-9,
        "x_positions must span the provided length"
    );

    let dy = width / ny as f64;
    let nnx = nx + 1;
    let nny = ny + 1;
    let nnz = nz + 1;

    // Direct indexing function - O(1) instead of HashMap lookup
    let node_idx = |ix: usize, iy: usize, iz: usize| -> usize {
        ix * (nny * nnz) + iy * nnz + iz
    };

    let mut nodes = Vec::with_capacity(nnx * nny * nnz);

    for ix in 0..nnx {
        let x = x_positions[ix];
        let h = if ix == 0 {
            element_heights[0]
        } else if ix == nx {
            element_heights[element_heights.len() - 1]
        } else {
            (element_heights[ix - 1] + element_heights[ix]) / 2.0
        };
        let dz = h / nz as f64;

        for iy in 0..nny {
            let y = iy as f64 * dy;
            for iz in 0..nnz {
                let z = iz as f64 * dz;
                nodes.push(Vector3::new(x, y, z));
            }
        }
    }

    let mut elements = Vec::with_capacity(nx * ny * nz);
    let mut heights = Vec::with_capacity(nx * ny * nz);
    for ix in 0..nx {
        let h = element_heights[ix];
        for iy in 0..ny {
            for iz in 0..nz {
                let n0 = node_idx(ix, iy, iz);
                let n1 = node_idx(ix + 1, iy, iz);
                let n2 = node_idx(ix + 1, iy + 1, iz);
                let n3 = node_idx(ix, iy + 1, iz);
                let n4 = node_idx(ix, iy, iz + 1);
                let n5 = node_idx(ix + 1, iy, iz + 1);
                let n6 = node_idx(ix + 1, iy + 1, iz + 1);
                let n7 = node_idx(ix, iy + 1, iz + 1);

                elements.push([n0, n1, n2, n3, n4, n5, n6, n7]);
                heights.push(h);
            }
        }
    }

    Mesh3d {
        nodes,
        elements,
        heights_per_element: heights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuts::{generate_adaptive_mesh_1d, generate_element_heights, Cut};

    #[test]
    fn mesh_generation_matches_expected_counts() {
        let heights = [1.0, 1.0];
        let mesh = generate_bar_mesh_3d(2.0, 1.0, &heights, 2, 2, 2);
        assert_eq!(mesh.nodes.len(), 27);
        assert_eq!(mesh.elements.len(), 8);
        assert_eq!(mesh.heights_per_element.len(), 8);
        assert_eq!(mesh.elements[0], [0, 9, 12, 3, 1, 10, 13, 4]);
    }

    #[test]
    fn mesh_generation_with_cuts_integration() {
        // Integration test: create a mesh from cuts
        let cuts = [Cut::new(0.2, 0.020), Cut::new(0.1, 0.015)];
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let num_elements_x = 20;

        let heights = generate_element_heights(&cuts, length, h0, num_elements_x);
        let mesh = generate_bar_mesh_3d(length, width, &heights, num_elements_x, 2, 2);

        assert_eq!(mesh.elements.len(), num_elements_x * 2 * 2);
        assert!(!mesh.nodes.is_empty());

        // Verify that the mesh has varying heights
        let min_height = heights.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_height = heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(
            min_height < max_height,
            "Mesh should have varying heights from cuts"
        );
        assert!(
            min_height >= 0.014,
            "Minimum height should be around cut height"
        );
        assert!(max_height <= 0.025, "Maximum height should be around h0");
    }

    #[test]
    fn adaptive_mesh_1d_integrates_with_3d_mesh() {
        let cuts = [Cut::new(0.20, 0.020), Cut::new(0.10, 0.015)];
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let base_elements = 15;

        let (x_positions, element_heights) =
            generate_adaptive_mesh_1d(&cuts, length, h0, base_elements, 4, 0.02);

        // Use adaptive mesh with generate_bar_mesh_3d_adaptive
        let mesh =
            generate_bar_mesh_3d_adaptive(length, width, &x_positions, &element_heights, 2, 2);

        // Mesh should be valid
        assert!(!mesh.nodes.is_empty());
        assert!(!mesh.elements.is_empty());
        assert_eq!(mesh.heights_per_element.len(), mesh.elements.len());

        // Heights should vary
        let min_h = element_heights
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_h = element_heights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(min_h < max_h, "Adaptive mesh should have varying heights");
    }

    #[test]
    fn mesh_serialization_to_json() {
        let cuts = [Cut::new(0.1, 0.015)];
        let length = 0.5;
        let width = 0.03;
        let h0 = 0.024;
        let num_elements = 10;

        let heights = generate_element_heights(&cuts, length, h0, num_elements);
        let mesh = generate_bar_mesh_3d(length, width, &heights, num_elements, 2, 2);

        // Test serialization
        let serializable = mesh.to_serializable();
        assert_eq!(serializable.nodes.len(), mesh.nodes.len());
        assert_eq!(serializable.elements.len(), mesh.elements.len());
        assert_eq!(
            serializable.element_heights.len(),
            mesh.heights_per_element.len()
        );

        // Verify metadata
        assert_eq!(serializable.metadata.num_nodes, mesh.nodes.len());
        assert_eq!(serializable.metadata.num_elements, mesh.elements.len());
        assert_eq!(serializable.metadata.num_dof, mesh.nodes.len() * 3);

        // Test JSON conversion
        let json = mesh.to_json().unwrap();
        assert!(json.contains("nodes"));
        assert!(json.contains("elements"));
        assert!(json.contains("metadata"));

        // Verify it can be deserialized
        let deserialized: SerializableMesh = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nodes.len(), mesh.nodes.len());
    }

    #[test]
    fn mesh_serialization_preserves_node_coordinates() {
        let cuts = [Cut::new(0.08, 0.012)];
        let heights = generate_element_heights(&cuts, 0.3, 0.024, 8);
        let mesh = generate_bar_mesh_3d(0.3, 0.025, &heights, 8, 2, 2);

        let serializable = mesh.to_serializable();

        // Check that node coordinates are preserved
        for (i, (original, serialized)) in mesh
            .nodes
            .iter()
            .zip(serializable.nodes.iter())
            .enumerate()
        {
            assert!(
                (original.x - serialized.x).abs() < 1e-10,
                "Node {} x coordinate mismatch",
                i
            );
            assert!(
                (original.y - serialized.y).abs() < 1e-10,
                "Node {} y coordinate mismatch",
                i
            );
            assert!(
                (original.z - serialized.z).abs() < 1e-10,
                "Node {} z coordinate mismatch",
                i
            );
        }
    }

    #[test]
    fn mesh_serialization_bounding_box() {
        let cuts = [Cut::new(0.15, 0.018)];
        let heights = generate_element_heights(&cuts, 0.5, 0.024, 15);
        let mesh = generate_bar_mesh_3d(0.5, 0.03, &heights, 15, 2, 3);

        let serializable = mesh.to_serializable();

        // Bounding box should contain all nodes
        for node in &serializable.nodes {
            assert!(node.x >= serializable.metadata.bbox_min[0]);
            assert!(node.y >= serializable.metadata.bbox_min[1]);
            assert!(node.z >= serializable.metadata.bbox_min[2]);
            assert!(node.x <= serializable.metadata.bbox_max[0]);
            assert!(node.y <= serializable.metadata.bbox_max[1]);
            assert!(node.z <= serializable.metadata.bbox_max[2]);
        }

        // Check bounding box makes sense
        assert!(serializable.metadata.bbox_min[0] < serializable.metadata.bbox_max[0]);
        assert!(serializable.metadata.bbox_min[1] < serializable.metadata.bbox_max[1]);
        assert!(serializable.metadata.bbox_min[2] < serializable.metadata.bbox_max[2]);
    }
}
