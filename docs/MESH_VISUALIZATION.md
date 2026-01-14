# Mesh Visualization for Frontend

## Overview

The `fem3d-rust-2` library now supports exporting meshes in JSON format for visualization in web frontends using WebGL frameworks like Three.js, Babylon.js, or similar.

## JSON Structure

The exported mesh has the following structure:

```json
{
  "nodes": [
    { "x": 0.0, "y": 0.0, "z": 0.0 },
    { "x": 0.1, "y": 0.0, "z": 0.0 },
    ...
  ],
  "elements": [
    { "nodes": [0, 1, 2, 3, 4, 5, 6, 7] },
    ...
  ],
  "element_heights": [0.024, 0.023, 0.015, ...],
  "metadata": {
    "num_nodes": 372,
    "num_elements": 180,
    "num_dof": 1116,
    "bbox_min": [0.0, 0.0, 0.0],
    "bbox_max": [0.551, 0.032, 0.024]
  }
}
```

### Fields

- **`nodes`**: Array of 3D positions for each node in the mesh
  - Each node has `x`, `y`, `z` coordinates in meters
  
- **`elements`**: Array of hexahedral (8-node brick) elements
  - Each element contains indices of 8 nodes
  - Node ordering: bottom face (4 nodes), then top face (4 nodes), counterclockwise
  
- **`element_heights`**: Array of height values for each element
  - Useful for color-coding elements by thickness
  - Can be used to visualize the undercut profile
  
- **`metadata`**: Mesh statistics and bounding box
  - `num_nodes`: Total number of nodes
  - `num_elements`: Total number of elements
  - `num_dof`: Total degrees of freedom (3 per node)
  - `bbox_min`: Minimum coordinates [x, y, z]
  - `bbox_max`: Maximum coordinates [x, y, z]

## Usage in Rust

```rust
use fem3d_rust_2::{Cut, generate_element_heights, generate_bar_mesh_3d};

// Create mesh with cuts
let cuts = vec![
    Cut::new(0.20, 0.020),
    Cut::new(0.12, 0.015),
];
let heights = generate_element_heights(&cuts, 0.551, 0.024, 30);
let mesh = generate_bar_mesh_3d(0.551, 0.032, &heights, 30, 2, 3);

// Export to JSON
let json = mesh.to_json_pretty().unwrap();

// Save to file
std::fs::write("mesh.json", json).unwrap();
```

## Frontend Integration

### Three.js Example

```javascript
// Load mesh data
const response = await fetch('mesh.json');
const meshData = await response.json();

// Create Three.js geometry
const geometry = new THREE.BufferGeometry();

// Add vertices
const vertices = new Float32Array(
  meshData.nodes.flatMap(node => [node.x, node.y, node.z])
);
geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

// Convert hexahedral elements to triangles
// Each hex8 element has 6 faces, each face needs 2 triangles
const indices = [];
for (const element of meshData.elements) {
  const n = element.nodes;
  
  // Bottom face (z-)
  indices.push(n[0], n[1], n[2]);
  indices.push(n[0], n[2], n[3]);
  
  // Top face (z+)
  indices.push(n[4], n[6], n[5]);
  indices.push(n[4], n[7], n[6]);
  
  // Front face (y-)
  indices.push(n[0], n[5], n[1]);
  indices.push(n[0], n[4], n[5]);
  
  // Back face (y+)
  indices.push(n[2], n[7], n[3]);
  indices.push(n[2], n[6], n[7]);
  
  // Left face (x-)
  indices.push(n[0], n[3], n[7]);
  indices.push(n[0], n[7], n[4]);
  
  // Right face (x+)
  indices.push(n[1], n[5], n[6]);
  indices.push(n[1], n[6], n[2]);
}

geometry.setIndex(indices);
geometry.computeVertexNormals();

// Create material with color based on height
const colors = new Float32Array(meshData.nodes.length * 3);
for (let i = 0; i < meshData.element_heights.length; i++) {
  const height = meshData.element_heights[i];
  const color = mapHeightToColor(height, 0.012, 0.024); // min/max heights
  
  // Apply color to all 8 nodes of this element
  const element = meshData.elements[i];
  for (const nodeIdx of element.nodes) {
    colors[nodeIdx * 3] = color.r;
    colors[nodeIdx * 3 + 1] = color.g;
    colors[nodeIdx * 3 + 2] = color.b;
  }
}
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

// Create mesh with vertex colors
const material = new THREE.MeshPhongMaterial({ 
  vertexColors: true,
  side: THREE.DoubleSide 
});
const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// Position camera using bounding box
const center = [
  (meshData.metadata.bbox_min[0] + meshData.metadata.bbox_max[0]) / 2,
  (meshData.metadata.bbox_min[1] + meshData.metadata.bbox_max[1]) / 2,
  (meshData.metadata.bbox_min[2] + meshData.metadata.bbox_max[2]) / 2,
];
camera.lookAt(center[0], center[1], center[2]);

function mapHeightToColor(height, minHeight, maxHeight) {
  const t = (height - minHeight) / (maxHeight - minHeight);
  // Blue (low) to red (high)
  return {
    r: t,
    g: 0.2,
    b: 1 - t
  };
}
```

## Performance Considerations

- For large meshes (>100k nodes), consider using compact JSON format with `to_json()` instead of `to_json_pretty()`
- The JSON file can be compressed (gzip) for transfer over network
- Consider implementing progressive loading for very large meshes
- Use instancing or level-of-detail (LOD) techniques for rendering many elements

## Example Output

Run the example to see the mesh export in action:

```bash
cargo run --example mesh_export
```

This will generate and display:
- Mesh statistics
- JSON structure preview
- Integration examples for Three.js
- Sample element and node data
