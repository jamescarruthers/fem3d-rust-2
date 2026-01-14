Idiophone optimisation and FEM in Rust

A Rust implementation of the Python code in /reference/python_working

It should follow the methods, techniques as closely as possible.

It must be compilable into WASM.

The FEM 3D code must be lightweight and designed to use as little memory as possible.

You should use existing Rust crates as much as possible, such as nalgebra + nalgebra-sparse - assuming that they implement methods the same as the Python code.

If you get stuck, there are some thoughts on implementation in /reference/thoughts/ - however only use these if the Python code is not understood.

