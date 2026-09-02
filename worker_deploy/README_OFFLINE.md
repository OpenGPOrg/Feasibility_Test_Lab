# University Lab PC Deployment Guide (Zero-Admin / Fully Offline)

This guide explains how to deploy and run **openGP Worker Nodes** on university lab PCs where:
* ❌ **No `sudo` / No Admin rights** (Normal user access only).
* ❌ **No Docker** installed.
* ❌ **No `pip`** or Python package installation tools.
* ❌ **No Internet in terminal** (Proxied/Air-gapped network — LAN only).

---

## Architecture of the Zero-Admin Offline Solution

Instead of installing packages or containers on the lab PCs:
1. **On your Laptop (Master with Internet):** You run `build_offline_bundle.py`. This downloads a portable, standalone Python 3.11 runtime and pre-installs PyTorch, Transformers, Psutil, Tabulate, and openGP worker scripts directly into an isolated folder.
2. **Transfer:** You copy the single compressed file (`openGP_worker_linux_x86_64.tar.gz`) to the lab PCs via:
   * **USB Flash Drive**, OR
   * **Local LAN HTTP Server** (using `python3 serve_to_lan.py` on your laptop).
3. **On the Lab PC:** Extract to your home directory (`~/openGP_worker`) and run `./start_worker.sh --port 9900`.

Everything runs self-contained inside the user's home directory with **zero dependencies**, **zero system installations**, and **zero internet**.

---

## Step 1: Build the Offline Bundle (On Master Laptop)

Run the automated bundler on your laptop:

```bash
cd /media/vithurshan/vithu/llm/openGP/LoadTime/worker_deploy

# Build lightweight CPU bundle (~350 MB archive):
/media/vithurshan/vithu/llm/.venv/bin/python3 build_offline_bundle.py

# (Optional) If lab PCs have NVIDIA GPUs, build CUDA bundle:
# /media/vithurshan/vithu/llm/.venv/bin/python3 build_offline_bundle.py --cuda
```

This generates: `worker_deploy/dist/openGP_worker_linux_x86_64.tar.gz`.

---

## Step 2: Transfer to Lab PCs

Choose whichever transfer method is easiest in your lab:

### Method A: Local LAN Transfer (No USB Needed)
1. On your laptop (connected to the same lab WiFi/Ethernet as the lab PCs):
   ```bash
   python3 serve_to_lan.py
   ```
2. Note your laptop's IP displayed on screen (e.g. `192.168.1.50`).
3. On each Lab PC:
   * **Via Browser:** Open Chrome/Firefox and go to `http://192.168.1.50:8000/`. Click `openGP_worker_linux_x86_64.tar.gz` to download. *(Browsers typically bypass terminal proxies for LAN IPs).*
   * **Via Terminal:**
     ```bash
     wget http://192.168.1.50:8000/openGP_worker_linux_x86_64.tar.gz
     # OR
     curl -O http://192.168.1.50:8000/openGP_worker_linux_x86_64.tar.gz
     ```

### Method B: USB Flash Drive
* Copy `openGP_worker_linux_x86_64.tar.gz` onto a USB drive.
* Plug into the lab PC and copy the file to `~/`.

---

## Step 3: Run Worker on Lab PCs (Zero Sudo Required)

### Standard Mode (If inbound port 9900 is open):
```bash
tar -xzf openGP_worker_linux_x86_64.tar.gz
cd openGP_worker

# Start worker listening on port 9900:
./start_worker.sh --port 9900
```

### ⚡ Reverse Connection Mode (If Lab PC has UFW Firewall active / blocks incoming ports):
If the Lab PC blocks incoming connections with `ufw`, connect **outbound** directly to your Master Laptop:

```bash
# 1. On your Master Laptop:
#    Start distributed_inference.py (it automatically listens for reverse workers on port 9900)
#    (Optional on your laptop: sudo ufw allow 9900)

# 2. On each Lab PC (no sudo needed!):
cd ~/openGP_worker
./start_worker.sh --connect <YOUR_LAPTOP_IP>:9900

# 3. Or launch as a detached background daemon:
nohup ./start_worker.sh --connect <YOUR_LAPTOP_IP>:9900 > worker.log 2>&1 &
```
* The Lab PC initiates an **outbound** connection to your laptop.
* UFW firewall automatically permits all outbound connections.
* The master registers the worker as a connected reverse node and displays:
  `⚡ [NEW REVERSE WORKER CONNECTED] Lab-PC (192.168.1.105:reverse)`.

---

## Step 4: Check Status and Run Inference

### Checking Worker Status on Lab PC:
```bash
# View live logs
tail -f worker.log

# Check if process is running
ps aux | grep worker_node.py

# Stop the worker when done
pkill -f worker_node.py
```

---

## Step 5: Distributed Inference Execution

1. In `distributed_inference.py` on your Master Laptop, go to:
   * `[Manage Remote Nodes]` -> all standard and reverse-connected lab PCs appear in the table with `Status: Connected`.
2. Go to `[Configure Distribution]` -> assign transformer layers to your lab PC workers.
3. Go to `[Run Inference]` -> start generating tokens across your distributed cluster!
