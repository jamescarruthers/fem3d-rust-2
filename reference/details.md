# FEM Implementation Technical Specification

Complete specification for reimplementing the 3D FEM analysis for undercut percussion bar vibration.

## 1. Element Type: 8-Node Hexahedral (Hex8)

- 3D brick element with 8 corner nodes
- Trilinear shape functions
- 3 DOF per node (ux, uy, uz) = 24 DOF per element
- Full integration: 2×2×2 Gauss quadrature (8 points)

## 2. Node Numbering Convention

Natural coordinates (ξ, η, ζ) range from -1 to +1:

```
Node 0: (-1, -1, -1)    Node 4: (-1, -1, +1)
Node 1: (+1, -1, -1)    Node 5: (+1, -1, +1)
Node 2: (+1, +1, -1)    Node 6: (+1, +1, +1)
Node 3: (-1, +1, -1)    Node 7: (-1, +1, +1)
```

Physical axes: X=length, Y=width, Z=height

## 3. Shape Functions

```
N0 = 0.125 × (1-ξ)(1-η)(1-ζ)
N1 = 0.125 × (1+ξ)(1-η)(1-ζ)
N2 = 0.125 × (1+ξ)(1+η)(1-ζ)
N3 = 0.125 × (1-ξ)(1+η)(1-ζ)
N4 = 0.125 × (1-ξ)(1-η)(1+ζ)
N5 = 0.125 × (1+ξ)(1-η)(1+ζ)
N6 = 0.125 × (1+ξ)(1+η)(1+ζ)
N7 = 0.125 × (1-ξ)(1+η)(1+ζ)
```

## 4. Shape Function Derivatives

∂N/∂ξ:
```
∂N0/∂ξ = -0.125(1-η)(1-ζ)    ∂N4/∂ξ = -0.125(1-η)(1+ζ)
∂N1/∂ξ = +0.125(1-η)(1-ζ)    ∂N5/∂ξ = +0.125(1-η)(1+ζ)
∂N2/∂ξ = +0.125(1+η)(1-ζ)    ∂N6/∂ξ = +0.125(1+η)(1+ζ)
∂N3/∂ξ = -0.125(1+η)(1-ζ)    ∂N7/∂ξ = -0.125(1+η)(1+ζ)
```

∂N/∂η:
```
∂N0/∂η = -0.125(1-ξ)(1-ζ)    ∂N4/∂η = -0.125(1-ξ)(1+ζ)
∂N1/∂η = -0.125(1+ξ)(1-ζ)    ∂N5/∂η = -0.125(1+ξ)(1+ζ)
∂N2/∂η = +0.125(1+ξ)(1-ζ)    ∂N6/∂η = +0.125(1+ξ)(1+ζ)
∂N3/∂η = +0.125(1-ξ)(1-ζ)    ∂N7/∂η = +0.125(1-ξ)(1+ζ)
```

∂N/∂ζ:
```
∂N0/∂ζ = -0.125(1-ξ)(1-η)    ∂N4/∂ζ = +0.125(1-ξ)(1-η)
∂N1/∂ζ = -0.125(1+ξ)(1-η)    ∂N5/∂ζ = +0.125(1+ξ)(1-η)
∂N2/∂ζ = -0.125(1+ξ)(1+η)    ∂N6/∂ζ = +0.125(1+ξ)(1+η)
∂N3/∂ζ = -0.125(1-ξ)(1+η)    ∂N7/∂ζ = +0.125(1-ξ)(1+η)
```

## 5. Gauss Quadrature (2×2×2)

g = 1/√3 ≈ 0.577350269

```
Point 0: (-g,-g,-g)  w=1    Point 4: (-g,-g,+g)  w=1
Point 1: (+g,-g,-g)  w=1    Point 5: (+g,-g,+g)  w=1
Point 2: (+g,+g,-g)  w=1    Point 6: (+g,+g,+g)  w=1
Point 3: (-g,+g,-g)  w=1    Point 7: (-g,+g,+g)  w=1
```

## 6. Jacobian

```
J = dN_natural × node_coords    (3×3)
detJ = det(J)
dN_physical = inv(J) × dN_natural
```

## 7. Strain-Displacement Matrix B (6×24)

Strain order: [εxx, εyy, εzz, γxy, γyz, γxz]

For node i (column = 3×i):
```
B[0, 3i+0] = ∂Ni/∂x     (εxx)
B[1, 3i+1] = ∂Ni/∂y     (εyy)
B[2, 3i+2] = ∂Ni/∂z     (εzz)
B[3, 3i+0] = ∂Ni/∂y     (γxy)
B[3, 3i+1] = ∂Ni/∂x
B[4, 3i+1] = ∂Ni/∂z     (γyz)
B[4, 3i+2] = ∂Ni/∂y
B[5, 3i+0] = ∂Ni/∂z     (γxz)
B[5, 3i+2] = ∂Ni/∂x
```

## 8. Elasticity Matrix D (6×6)

```
factor = E / ((1+ν)(1-2ν))

D = factor × | (1-ν)   ν      ν      0          0          0         |
             |   ν   (1-ν)    ν      0          0          0         |
             |   ν     ν    (1-ν)    0          0          0         |
             |   0     0      0    (1-2ν)/2     0          0         |
             |   0     0      0      0       (1-2ν)/2      0         |
             |   0     0      0      0          0       (1-2ν)/2    |
```

## 9. Element Matrix Integration

At each Gauss point:
```
Ke += weight × detJ × (Bᵀ × D × B)
Me += weight × detJ × ρ × (Nmatᵀ × Nmat)
```

Nmat (3×24): `Nmat[0,3i]=Ni`, `Nmat[1,3i+1]=Ni`, `Nmat[2,3i+2]=Ni`

## 10. Global Assembly

DOF mapping: `[3×n0, 3×n0+1, 3×n0+2, 3×n1, ...]`

```
K_global[gi,gj] += Ke[i,j]
M_global[gi,gj] += Me[i,j]
```

Use sparse matrices when DOF > 1000.

## 11. Eigenvalue Solution

Problem: Kφ = λMφ

Shift-invert method (σ=1.0):
```
eigsh(K, M=M, sigma=1.0, which='LM')
```

Rigid body filtering: ω² > 100 rad²/s²

Frequency: f = √λ / (2π) Hz

## 12. Mode Classification (Soares Method)

1. Find two corner nodes at x=0 (top surface, max/min y)
2. Extract displacement ψ = [ψx, ψy, ψz] at each corner
3. Find dominant direction at s1:
   - Y dominant → lateral
   - X dominant → axial
   - Z dominant + same sign → vertical_bending
   - Z dominant + opposite sign → torsional

## 13. Mesh Generation

```
For ix in 0..nx:
    h = element_heights[ix]
    dz = h / nz
    For iy in 0..ny:
        For iz in 0..nz:
            node = [ix×dx, iy×dy, iz×dz]
```

Element connectivity:
```
[n0,n1,n2,n3,n4,n5,n6,n7] at grid (ix,iy,iz)
```

## 14. 2D Timoshenko Beam (Fast)

DOFs: [w1, θ1, w2, θ2]

κ = 5/6 (shear correction)
G = E/(2(1+ν))
φ = 12EI/(κGA L²)

## 15. Key Constants

| Constant | Value |
|----------|-------|
| KAPPA | 5/6 |
| Rigid body threshold (3D) | 100 rad²/s² |
| Rigid body threshold (2D) | 1 rad²/s² |
| Sparse threshold | 1000 DOF |
| Shift σ | 1.0 |
| Mass regularization | 1e-12 |

## 16. Current Configuration

```
NUM_ELEMENTS_2D = 120
NUM_ELEMENTS_3D_X = 120
NY = 2
NZ = 24
```

Total: 5,760 elements, 9,075 nodes, 27,225 DOF
