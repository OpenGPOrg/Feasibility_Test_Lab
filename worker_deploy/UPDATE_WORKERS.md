# Pushing Code Updates to Lab PC Workers

Whenever you modify `worker_node.py` or adjust the project's dependencies on your Master Laptop, you must push the updated code to all your University Lab PCs. 

Because the Lab PCs have no internet access, no pip, and no sudo rights, you update them by generating a **new portable bundle** and downloading it over the Local Area Network (LAN).

---

## 1. Build the Updated Bundle (On Master Laptop)

After saving your changes to `worker_node.py`, open a terminal on your Master Laptop:

```bash
cd /media/vithurshan/vithu/llm/openGP/LoadTime/worker_deploy
python3 build_offline_bundle.py
```

*This automatically packages your new code and portable Python environment into a fresh `openGP_worker_linux_x86_64.tar.gz`.*

---

## 2. Start the LAN Distribution Server (On Master Laptop)

Next, host the new file on your local network:

```bash
python3 serve_to_lan.py
```
*Note the local IP address it displays (e.g., `192.168.1.50:8000`). Leave this terminal open.*

---

## 3. Pull the Update and Restart (On Lab PCs)

Walk over to the Lab PCs (or SSH into them). You can download the update directly via the terminal using either `wget` or `curl`.

### Using `wget`:
```bash
# 1. Stop any currently running worker processes
pkill -f start_worker.sh
pkill -f worker_node.py

# 2. Download the new bundle from your Master Laptop (Replace IP with your actual IP)
wget http://192.168.1.50:8000/openGP_worker_linux_x86_64.tar.gz -O ~/openGP_worker_linux_x86_64.tar.gz

# 3. Extract the new bundle (this safely overwrites the old code)
tar -xzf ~/openGP_worker_linux_x86_64.tar.gz -C ~/

# 4. Restart the worker!
cd ~/openGP_worker
nohup ./start_worker.sh --connect 192.168.1.50:9900 > worker.log 2>&1 &
```

### Alternatively, Using `curl`:
```bash
# 1. Stop any currently running worker processes
pkill -f start_worker.sh
pkill -f worker_node.py

# 2. Download the new bundle from your Master Laptop
curl -o ~/openGP_worker_linux_x86_64.tar.gz http://192.168.1.50:8000/openGP_worker_linux_x86_64.tar.gz

# 3. Extract the new bundle (this safely overwrites the old code)
tar -xzf ~/openGP_worker_linux_x86_64.tar.gz -C ~/

# 4. Restart the worker!
cd ~/openGP_worker
nohup ./start_worker.sh --connect 192.168.1.50:9900 > worker.log 2>&1 &
```

> **Tip:** By running the `pkill` and `tar` extraction steps, your local cached `.pt` model weights inside `~/.cache/openGP_components` remain entirely untouched. You do not need to re-transfer model weights after updating the Python code!
