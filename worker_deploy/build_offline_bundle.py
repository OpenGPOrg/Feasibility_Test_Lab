#!/usr/bin/env python3
"""
openGP Offline Worker Packaging Tool
====================================
Builds a 100% self-contained, zero-install, zero-admin, zero-docker,
fully offline worker bundle for Linux (and Windows) university lab PCs.

The bundle includes:
  - Portable standalone Python runtime (no root / no apt required)
  - Pre-installed PyTorch (CPU or CUDA), Transformers, Psutil, Tabulate
  - openGP worker_node.py + start_worker.sh
  - 1-click startup without internet or pip access

Usage:
  python3 build_offline_bundle.py [--cuda] [--output-dir dist]
"""

import os
import sys
import shutil
import urllib.request
import tarfile
import subprocess
import argparse
from pathlib import Path

# Astral python-build-standalone tested release
PYTHON_STANDALONE_URL_LINUX = (
    "https://github.com/indygreg/python-build-standalone/releases/download/"
    "20240224/cpython-3.11.8+20240224-x86_64-unknown-linux-gnu-install_only.tar.gz"
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BUILD_DIR = SCRIPT_DIR / "build"
DIST_DIR = SCRIPT_DIR / "dist"

def download_file(url: str, dest: Path):
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")
    def reporthook(blocknum, blocksize, totalsize):
        read = blocknum * blocksize
        if totalsize > 0:
            pct = read / totalsize * 100
            mb_read = read / (1024 * 1024)
            mb_tot = totalsize / (1024 * 1024)
            print(f"\r  Progress: {mb_read:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook)
    print("\n  ✓ Download complete.")

def build_bundle(use_cuda: bool = False, bundle_name: str = "openGP_worker_linux_x86_64"):
    print("=" * 70)
    print("  BUILDING OPENGP ZERO-INSTALL OFFLINE WORKER BUNDLE")
    print(f"  Target: Linux x86_64 (Zero Admin / No Docker / No Pip Required)")
    print(f"  PyTorch Mode: {'CUDA (GPU Accelerated)' if use_cuda else 'CPU Only (Lightweight Uni Lab)'}")
    print("=" * 70)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    staging_dir = BUILD_DIR / "openGP_worker"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download & Extract Portable Standalone Python
    py_tar = BUILD_DIR / "python_standalone_linux.tar.gz"
    if not py_tar.exists() or py_tar.stat().st_size < 1024 * 1024:
        print("\n[1/5] Downloading standalone portable Python runtime...")
        download_file(PYTHON_STANDALONE_URL_LINUX, py_tar)
    else:
        print("\n[1/5] Using cached standalone portable Python archive.")

    print("  Extracting portable Python to staging directory...")
    with tarfile.open(py_tar, "r:gz") as tar:
        tar.extractall(path=staging_dir)

    py_bin = staging_dir / "python" / "bin" / "python3"
    if not py_bin.exists():
        raise RuntimeError(f"Python binary not found at {py_bin}")
    py_bin.chmod(0o755)

    # 2. Install Dependencies into Portable Python
    print("\n[2/5] Installing core dependencies into portable Python bundle...")
    pip_bin = staging_dir / "python" / "bin" / "pip3"
    
    # Upgrade pip / wheel in bundle
    subprocess.run([str(pip_bin), "install", "--no-warn-script-location", "--upgrade", "pip", "setuptools", "wheel"], check=True)

    # Install PyTorch
    if use_cuda:
        print("  Installing PyTorch with CUDA support...")
        subprocess.run([str(pip_bin), "install", "--no-warn-script-location", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
    else:
        print("  Installing lightweight PyTorch CPU...")
        subprocess.run([str(pip_bin), "install", "--no-warn-script-location", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)

    # Install other required libraries (including bitsandbytes for 4bit/8bit quantization)
    print("  Installing transformers, bitsandbytes, psutil, tabulate, safetensors, accelerate, matplotlib...")
    subprocess.run([str(pip_bin), "install", "--no-warn-script-location", "transformers", "bitsandbytes", "psutil", "tabulate", "safetensors", "accelerate", "matplotlib"], check=True)

    # Clean pip caches inside bundle to reduce size
    subprocess.run([str(pip_bin), "cache", "purge"], check=False)

    # 3. Copy Worker Node Code and Launchers
    print("\n[3/5] Copying openGP worker node and launcher scripts...")
    shutil.copy2(ROOT_DIR / "worker_node.py", staging_dir / "worker_node.py")
    shutil.copy2(SCRIPT_DIR / "start_worker.sh", staging_dir / "start_worker.sh")
    (staging_dir / "start_worker.sh").chmod(0o755)
    shutil.copy2(SCRIPT_DIR / "start_worker.bat", staging_dir / "start_worker.bat")

    # Create README in bundle
    readme_text = f"""# openGP Volunteer Worker Node (Zero-Admin Offline Edition)

## Quick Start on Lab PC (No Admin / No Internet / No Pip Needed):

1. Open a terminal in this folder:
   ```bash
   ./start_worker.sh --port 9900
   ```

2. Run in the background (survives terminal close):
   ```bash
   nohup ./start_worker.sh --port 9900 > worker.log 2>&1 &
   ```

3. Check worker output:
   ```bash
   tail -f worker.log
   ```

4. To stop:
   ```bash
   pkill -f worker_node.py
   ```

## LAN Master Connection:
Find your lab PC local IP with `hostname -I` or `ip addr`.
From the master machine, point to: `<lab_pc_ip>:9900`.
"""
    (staging_dir / "README.txt").write_text(readme_text)

    # 4. Package into Compressed Archive (.tar.gz)
    print("\n[4/5] Packaging into self-contained distribution archive...")
    tar_out = DIST_DIR / f"{bundle_name}.tar.gz"
    if tar_out.exists():
        tar_out.unlink()

    with tarfile.open(tar_out, "w:gz") as tar:
        tar.add(staging_dir, arcname="openGP_worker")

    size_mb = tar_out.stat().st_size / (1024 * 1024)
    print(f"  ✓ Output bundle: {tar_out} ({size_mb:.1f} MB)")

    # 5. Summary & Transfer Instructions
    print("\n" + "=" * 70)
    print("  BUNDLE BUILD COMPLETE! 🚀")
    print("=" * 70)
    print(f"  Archive: {tar_out}")
    print(f"  Size:    {size_mb:.1f} MB")
    print("\n  Deployment to University Lab PCs:")
    print("  1. Transfer via USB flash drive OR local LAN:")
    print(f"     python3 serve_to_lan.py")
    print("  2. On Lab PC, extract and run:")
    print("     tar -xzf openGP_worker_linux_x86_64.tar.gz")
    print("     cd openGP_worker")
    print("     ./start_worker.sh --port 9900")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline openGP worker bundle")
    parser.add_argument("--cuda", action="store_true", help="Include CUDA PyTorch support (larger bundle)")
    parser.add_argument("--name", default="openGP_worker_linux_x86_64", help="Bundle filename prefix")
    args = parser.parse_args()
    build_bundle(use_cuda=args.cuda, bundle_name=args.name)
