#!/usr/bin/env python3
"""
openGP Local LAN Bundle Server
==============================
Runs a lightweight HTTP server on the master laptop to allow university lab PCs
on the same local WiFi/Ethernet network to download the offline worker bundle
directly via web browser or curl/wget, bypassing terminal proxies!
"""

import http.server
import socketserver
import socket
from pathlib import Path

PORT = 8000
SERVE_DIR = Path(__file__).resolve().parent / "dist"

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_DIR), **kwargs)

if __name__ == "__main__":
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    local_ip = get_local_ip()
    print("=" * 70)
    print("  OPENGP LOCAL LAN WORKER BUNDLE SERVER")
    print("=" * 70)
    print(f"  Serving directory: {SERVE_DIR}")
    print(f"\n  On University Lab PCs, open in Browser or run:")
    print(f"  --> http://{local_ip}:{PORT}/")
    print(f"  --> wget http://{local_ip}:{PORT}/openGP_worker_linux_x86_64.tar.gz")
    print(f"  --> curl -O http://{local_ip}:{PORT}/openGP_worker_linux_x86_64.tar.gz")
    print("=" * 70)
    print("  Press Ctrl+C to stop server.\n")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
