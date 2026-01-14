FEM Implementation Technical Overview
1. Element Type: 8-Node Hexahedral (Hex8)
Characteristics:

3D brick element with 8 corner nodes
Trilinear shape functions (linear in each direction)
3 DOF per node (ux, uy, uz) = 24 DOF per element
Full integration: 2×2×2 Gauss quadrature (8 integration points)
2. Node Numbering Convention
Natural coordinates (ξ, η, ζ) range from -1 to +1:


Node 0: (-1, -1, -1)    Node 4: (-1, -1, +1)
Node 1: (+1, -1, -1)    Node 5: (+1, -1, +1)
Node 2: (+1, +1, -1)    Node 6: (+1, +1, +1)
Node 3: (-1, +1, -1)    Node 7: (-1, +1, +1)
Physical orientation:

X = length (along bar)
Y = width
Z = height/thickness
3. Shape Functions

N[i] = (1/8) × (1 ± ξ) × (1 ± η) × (1 ± ζ)
Explicitly:


N0 = 0.125 × (1-ξ)(1-η)(1-ζ)
N1 = 0.125 × (1+ξ)(1-η)(1-ζ)
N2 = 0.125 × (1+ξ)(1+η)(1-ζ)
N3 = 0.125 × (1-ξ)(1+η)(1-ζ)
N4 = 0.125 × (1-ξ)(1-η)(1+ζ)
N5 = 0.125 × (1+ξ)(1-η)(1+ζ)
N6 = 0.125 × (1+ξ)(1+η)(1+ζ)
N7 = 0.125 × (1-ξ)(1+η)(1+ζ)
4. Shape Function Derivatives
∂N/∂ξ, ∂N/∂η, ∂N/∂ζ follow the pattern with appropriate sign changes. For example:


∂N0/∂ξ  = -0.125 × (1-η)(1-ζ)
∂N0/∂η  = -0.125 × (1-ξ)(1-ζ)
∂N0/∂ζ  = -0.125 × (1-ξ)(1-η)
5. Gauss Quadrature (2×2×2)
Points: g = 1/√3 ≈ 0.577350269


Point 0: (-g, -g, -g)    Point 4: (-g, -g, +g)
Point 1: (+g, -g, -g)    Point 5: (+g, -g, +g)
Point 2: (+g, +g, -g)    Point 6: (+g, +g, +g)
Point 3: (-g, +g, -g)    Point 7: (-g, +g, +g)
Weights: All 8 weights = 1.0

6. Jacobian Computation

J = [∂N/∂ξ; ∂N/∂η; ∂N/∂ζ] × [node_coordinates]  (3×3 matrix)
detJ = determinant(J)
J_inv = inverse(J)
Physical derivatives:


[∂N/∂x; ∂N/∂y; ∂N/∂z] = J_inv × [∂N/∂ξ; ∂N/∂η; ∂N/∂ζ]
7. Strain-Displacement Matrix B (6×24)
Voigt notation strain order: [εxx, εyy, εzz, γxy, γyz, γxz]

For each node i (column offset = 3×i):


B[0, col+0] = ∂Ni/∂x     (εxx = ∂u/∂x)
B[1, col+1] = ∂Ni/∂y     (εyy = ∂v/∂y)
B[2, col+2] = ∂Ni/∂z     (εzz = ∂w/∂z)
B[3, col+0] = ∂Ni/∂y     (γxy = ∂u/∂y + ∂v/∂x)
B[3, col+1] = ∂Ni/∂x
B[4, col+1] = ∂Ni/∂z     (γyz = ∂v/∂z + ∂w/∂y)
B[4, col+2] = ∂Ni/∂y
B[5, col+0] = ∂Ni/∂z     (γxz = ∂u/∂z + ∂w/∂x)
B[5, col+2] = ∂Ni/∂x
8. Elasticity Matrix D (6×6) - Isotropic

factor = E / ((1+ν)(1-2ν))

D = factor × [
  (1-ν)    ν       ν       0           0           0
  ν        (1-ν)   ν       0           0           0
  ν        ν       (1-ν)   0           0           0
  0        0       0       (1-2ν)/2    0           0
  0        0       0       0           (1-2ν)/2    0
  0        0       0       0           0           (1-2ν)/2
]
9. Element Matrix Integration
At each Gauss point:


Ke += weight × detJ × (Bᵀ × D × B)
Me += weight × detJ × ρ × (Nmatᵀ × Nmat)
Where Nmat (3×24) maps nodal displacements to physical displacement:


Nmat[0, 3i]   = Ni    (ux interpolation)
Nmat[1, 3i+1] = Ni    (uy interpolation)
Nmat[2, 3i+2] = Ni    (uz interpolation)
10. Global Assembly
DOF mapping for element with nodes [n0, n1, ..., n7]:


global_dof = [3×n0, 3×n0+1, 3×n0+2, 3×n1, 3×n1+1, 3×n1+2, ..., 3×n7+2]
Assembly:


K_global[gi, gj] += Ke[i, j]
M_global[gi, gj] += Me[i, j]
Use sparse matrices when DOF > 1000.

11. Eigenvalue Solution
Problem: Kφ = λMφ (generalized eigenvalue)

Method: Shift-invert with σ = 1.0


eigenvalues, eigenvectors = eigsh(K, M=M, sigma=1.0, which='LM')
Alternative (dense): Cholesky transformation


M = LLᵀ
K̃ = L⁻¹ K L⁻ᵀ
Solve standard: K̃ψ = λψ
Rigid body filtering: ω² > 100 rad²/s² threshold (filters 6 rigid body modes)

Frequency conversion:


f = √λ / (2π)  [Hz]
12. Mode Classification (Soares Method)
Find two corner nodes at x=0 end (top surface, max/min y)
Extract displacement at each corner: ψ = [ψx, ψy, ψz]
Find dominant direction at corner s1:
Y dominant → lateral
X dominant → axial
Z dominant + same sign at both corners → vertical bending
Z dominant + opposite sign → torsional
13. Mesh Generation
Uniform mesh:

nx elements along length, ny along width, nz along height
dx = L/nx, dy = W/ny
dz varies per x-position based on undercut profile
Node generation:


for ix in 0..nx:
    h = height at position ix
    dz = h / nz
    for iy in 0..ny:
        for iz in 0..nz:
            node = [ix×dx, iy×dy, iz×dz]
Element connectivity:


n0 = node_index(ix, iy, iz)
n1 = node_index(ix+1, iy, iz)
... (8 nodes per hex following the numbering convention)
14. Key Constants
Constant	Value	Description
KAPPA	5/6	Timoshenko shear correction (2D only)
Rigid body threshold	100 rad²/s²	Filter ω² below this
Sparse threshold	1000 DOF	Switch to sparse matrices
Shift σ	1.0	Eigenvalue shift-invert
Mass regularization	1e-12	Added to M diagonal
15. Typical Mesh Settings
Your current configuration:

NX = 120 elements along length
NY = 2 elements along width
NZ = 24 elements along thickness
Total elements = 120 × 2 × 24 = 5,760
Total nodes = 121 × 3 × 25 = 9,075
Total DOF = 27,225
This should give you everything needed to reimplement the 3D FEM in another language. The critical pieces are the shape functions, B matrix construction, and the eigenvalue solution approach.
