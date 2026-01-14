# Rust FEM (3D) modal analysis for idiophone bars (WASM-friendly)

This note compiles a practical plan for writing a **3D FEM vibration / modal** module in **Rust** that can run in **WebAssembly (WASM)** with **low runtime memory use**.

---

## Problem shape

For linear elastic vibration you typically solve the **symmetric generalized eigenproblem**:

\[
K\,\phi = \lambda\, M\,\phi,\quad \lambda = \omega^2,\quad f = \frac{\sqrt{\lambda}}{2\pi}
\]

- **K**: stiffness matrix (sparse, symmetric)
- **M**: mass matrix (sparse symmetric; often approximated by a diagonal “lumped” mass for simplicity/memory)
- You generally want the **lowest N modes** (e.g., 10–50), not a full decomposition.

At **5,000–10,000 DOF**, this is very feasible in WASM with **sparse storage** and an **iterative eigensolver**.

---

## Key constraints for WASM

- Prefer **pure-Rust** math stacks to avoid external BLAS/LAPACK dependencies.
- Avoid dense eigensolvers on the full system (memory/time blow up quickly).
- Minimize allocations by:
  - assembling from **COO (triplets) → CSR/CSC** once,
  - reusing work buffers inside iterative loops.

---

## Recommended Rust crate stacks

### Default (simple, ergonomic, good for 5k–10k DOF)

**Use:**
- `nalgebra` (core vectors/matrices; pure-Rust decompositions can work on wasm targets)
- `nalgebra-sparse` (COO/CSR/CSC sparse matrices for assembly & matvec)
- `lanczos` (iterative eigenpairs for symmetric/Hermitian matrices; works with `nalgebra_sparse`)
- optional: `nalgebra-sparse-linalg` (iterative sparse linear solvers)

This stack stays in a single ecosystem and is straightforward to prototype.

### If you need sparse direct solves / shift-invert

If you decide to implement **shift-invert** (common for extracting smallest modes robustly) you’ll want a sparse factorization backend:

- `faer-sparse` (sparse Cholesky variants)

You can still keep your FEM assembly design similar (sparse K/M), just change the solver backend.

---

## Solver strategy that works well in WASM

### Approach A: Lumped mass + scaled operator (very WASM-friendly)

If you lump mass so **M is diagonal** (store as `Vec<f64>`), you can turn the generalized eigenproblem into a standard symmetric one without explicitly forming dense matrices:

Let \(D = \mathrm{diag}(M)\). Define the operator:

\[
B = D^{-1/2} K D^{-1/2}
\]

Then run Lanczos on matvecs:

1. \(u = D^{-1/2} x\)
2. \(v = K u\)  (sparse matvec)
3. \(y = D^{-1/2} v\)

This stores **only**:
- sparse **K**
- diagonal mass **D**

…and gives you \(\lambda\) directly from \(B\), which are still \(\omega^2\).

### Approach B: Consistent mass + generalized eigensolve

You keep sparse **K** and sparse **M** and use a generalized iterative method. This can be more accurate but increases storage/complexity.

### Free-free rigid modes

A free 3D body has **6 rigid-body modes** with ~0 Hz. If your bar is unconstrained, you’ll see these as the smallest eigenvalues. In reporting “audible” modes, skip those first ~6.

---

## Meshing pipeline (recommended for WASM)

Generating tetra meshes **inside** the browser is rarely worth it.

A practical flow:
1. Mesh offline (e.g., with Gmsh)
2. Ship `.msh` (MSH 4.1) with your app
3. Parse in WASM using `mshio`

This keeps the WASM module lighter and more deterministic.

---

## Assembly notes for 3D elasticity (high-level)

For each element:
- compute element stiffness \(K_e\) using shape function gradients and material matrix \(C\)
- compute element mass \(M_e\) (consistent or lumped)
- scatter-add into global sparse structures (COO triplets are easiest, then compress to CSR/CSC)

Implementation tips:
- use **block structure** (3 DOF per node) to keep indexing clear
- consider **row-wise assembly** into triplets to avoid repeated reallocations

---

## Memory expectations at 5k–10k DOF

Sparse memory depends on connectivity and element type, but at this size:
- assembled sparse **K** is typically manageable in **single-digit to low–tens of MB**
- keeping **M** as diagonal helps a lot

If you ever outgrow this:
- move to **matrix-free** matvecs (store connectivity + coords, apply Kx by looping elements)
- Lanczos can still work as long as you can implement the operator matvec

---

## What to avoid (for wasm32-unknown-unknown)

- Crates that require external BLAS/LAPACK at runtime (compilation/portability pain in browsers).
- “Maximum-performance” stacks that depend on SuiteSparse/MUMPS/OpenBLAS unless you *explicitly* control your WASM toolchain and are willing to do extra integration work.

---

## Suggested project layout

- `mesh/`  
  - `.msh` parsing, topology, boundary sets
- `fem/`  
  - element kernels (tet4 / tet10 / hex8), quadrature, material laws
- `assembly/`  
  - COO triplets → CSR/CSC build, boundary condition application
- `modal/`  
  - Lanczos driver, mode filtering (skip rigid modes), frequency formatting
- `wasm/`  
  - `wasm-bindgen` interface, serialization, memory reuse hooks

---

## References (docs & crate pages)

- nalgebra: WASM & embedded targets notes (incl. `nalgebra-lapack` caveats)  
  https://www.nalgebra.rs/docs/user_guide/wasm_and_embedded_targets/
- nalgebra-sparse docs  
  https://docs.rs/nalgebra-sparse
- lanczos crate (iterative eigen for symmetric/Hermitian; sparse via nalgebra-sparse)  
  https://crates.io/crates/lanczos
- nalgebra-sparse-linalg crate (iterative sparse solvers)  
  https://crates.io/crates/nalgebra-sparse-linalg
- faer-sparse docs (sparse Cholesky module overview)  
  https://docs.rs/faer-sparse
- faer-cholesky (LLT/LDLT/Bunch–Kaufman)  
  https://crates.io/crates/faer-cholesky
- mshio (Gmsh MSH 4.1 parser)  
  https://github.com/w1th0utnam3/mshio
- fenris FEM crate warning about unstable API (useful reference, not production-stable)  
  https://docs.rs/fenris
- russell notes on external deps (OpenBLAS/MKL/MUMPS/SuiteSparse)  
  https://github.com/cpmech/russell
- tritet (tet mesh generation via Tetgen wrapper; typically not ideal for in-browser use)  
  https://docs.rs/tritet

