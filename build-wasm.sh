#!/bin/bash
#
# Build script for WASM target with optional multithreading support.
#
# Usage:
#   ./build-wasm.sh              # Single-threaded build (stable Rust)
#   ./build-wasm.sh --threads    # Multi-threaded build (nightly Rust required)
#
# Requirements for multi-threaded build:
#   - Nightly Rust: rustup install nightly
#   - rust-src component: rustup component add rust-src --toolchain nightly
#   - wasm-bindgen-cli: cargo install wasm-bindgen-cli
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

if [ "$1" == "--threads" ] || [ "$1" == "-t" ]; then
    echo_info "Building WASM with multithreading support (requires nightly Rust)"
    echo ""

    # Check for nightly toolchain
    if ! rustup run nightly rustc --version > /dev/null 2>&1; then
        echo_error "Nightly Rust not found. Install with: rustup install nightly"
        exit 1
    fi

    # Check for rust-src component
    if ! rustup run nightly rustc --print sysroot | xargs -I {} test -d "{}/lib/rustlib/src/rust"; then
        echo_warn "rust-src component may not be installed."
        echo_info "Installing rust-src for nightly..."
        rustup component add rust-src --toolchain nightly
    fi

    # Check for wasm-bindgen-cli
    if ! command -v wasm-bindgen > /dev/null 2>&1; then
        echo_error "wasm-bindgen-cli not found. Install with: cargo install wasm-bindgen-cli"
        exit 1
    fi

    echo_info "Building with cargo +nightly..."

    # Build with nightly, atomics, and build-std
    RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
    cargo +nightly build \
        --target wasm32-unknown-unknown \
        --release \
        --features parallel-wasm \
        -Z build-std=std,panic_abort

    echo_info "Running wasm-bindgen..."

    # Run wasm-bindgen to generate JS bindings
    # --reference-types and --weak-refs are required for proper thread pool support
    mkdir -p pkg
    wasm-bindgen \
        --target web \
        --out-dir pkg \
        --reference-types \
        --weak-refs \
        target/wasm32-unknown-unknown/release/fem3d_rust_2.wasm

    echo ""
    echo_info "Multi-threaded WASM build complete!"
    echo_info "Output in: pkg/"
    echo ""
    echo_warn "Remember: Server must send COOP/COEP headers for SharedArrayBuffer"
    echo_info "Use: python web/server.py"

else
    echo_info "Building WASM (single-threaded, stable Rust)"
    echo ""

    # Check for wasm-pack
    if ! command -v wasm-pack > /dev/null 2>&1; then
        echo_error "wasm-pack not found. Install with: cargo install wasm-pack"
        exit 1
    fi

    # Standard single-threaded build with wasm-pack
    wasm-pack build --target web --release --no-default-features --features sprs-backend

    echo ""
    echo_info "Single-threaded WASM build complete!"
    echo_info "Output in: pkg/"
    echo ""
    echo_info "For multi-threaded build, run: ./build-wasm.sh --threads"
fi
