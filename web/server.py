#!/usr/bin/env python3
"""
Simple HTTP server with COOP/COEP headers for WASM multithreading.

These headers enable SharedArrayBuffer which is required for Web Workers
to share memory with the main thread (needed for wasm-bindgen-rayon).

Usage:
    python server.py [port]

    Default port is 8080. Access at http://localhost:8080
"""

import http.server
import socketserver
import sys
import os

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that adds COOP/COEP headers for SharedArrayBuffer support."""

    def end_headers(self):
        # Required headers for SharedArrayBuffer (WASM multithreading)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")

        # CORS headers for local development
        self.send_header("Access-Control-Allow-Origin", "*")

        # Proper MIME types for WASM
        super().end_headers()

    def do_GET(self):
        # Handle requests for pkg/ directory by redirecting to the main JS file
        # This fixes the issue where web workers try to import from the directory URL
        if self.path.endswith('/pkg/') or self.path.endswith('/pkg'):
            # Redirect to the main entry point
            self.send_response(302)
            self.send_header('Location', self.path.rstrip('/') + '/fem3d_rust_2.js')
            self.end_headers()
            return
        return super().do_GET()

    def guess_type(self, path):
        """Add proper MIME type for .wasm and .js files."""
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".js"):
            return "application/javascript"
        return super().guess_type(path)


def main():
    # Change to the project root so we can serve both web/ and pkg/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    with socketserver.TCPServer(("", PORT), COOPCOEPHandler) as httpd:
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  FEM3D WASM Development Server                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:{PORT}                        ║
║  Web interface at:  http://localhost:{PORT}/web/                   ║
╠══════════════════════════════════════════════════════════════════╣
║  COOP/COEP headers enabled for SharedArrayBuffer support         ║
║  (Required for WASM multithreading with Web Workers)             ║
╠══════════════════════════════════════════════════════════════════╣
║  Build commands:                                                 ║
║    Single-threaded: wasm-pack build --target web                 ║
║    Multi-threaded:  RUSTUP_TOOLCHAIN=nightly wasm-pack build \\   ║
║                     --target web --features parallel-wasm        ║
╠══════════════════════════════════════════════════════════════════╣
║  Press Ctrl+C to stop                                            ║
╚══════════════════════════════════════════════════════════════════╝
""")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
