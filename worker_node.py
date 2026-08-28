#!/usr/bin/env python3
"""
Worker Node — Run on remote machines to participate in distributed model inference.

Usage:
    python3 worker_node.py                 # default port 9900
    python3 worker_node.py --port 9901     # custom port

Requirements (same versions as controller):
    pip install torch transformers psutil
"""

import argparse
import gc
import io
import pickle
import platform
import socket
import struct
import sys
import threading
import time
import hashlib
from pathlib import Path

import torch
import psutil

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
DISC_PORT = 9899
DEFAULT_PORT = 9900
MAGIC = b"LDIST_V1"
RECV_BUF = 1 << 16  # 64 KB
COMPONENT_CACHE_DIR = Path.home() / ".dist_inference_cache"


# ═══════════════════════════════════════════════════════════════════════════════
# Network Protocol
# ═══════════════════════════════════════════════════════════════════════════════
def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), RECV_BUF))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock, obj):
    """Send a Python object (including tensors/modules) over a socket."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)
    return 8 + len(data)


class CPU_Unpickler(pickle.Unpickler):
    """Custom unpickler that automatically maps any CUDA tensors to CPU on CPU-only workers."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device("cpu"), weights_only=False)
        return super().find_class(module, name)


def recv_msg(sock):
    """Receive a Python object from a socket. Returns (obj, bytes_received)."""
    raw = _recv_exact(sock, 8)
    length = struct.unpack("!Q", raw)[0]
    data = _recv_exact(sock, length)
    if not torch.cuda.is_available():
        try:
            obj = CPU_Unpickler(io.BytesIO(data)).load()
        except Exception:
            obj = pickle.loads(data)
    else:
        obj = pickle.loads(data)
    return obj, 8 + length


# ═══════════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════════
class Worker:
    def __init__(self, port):
        self.port = port
        self.hostname = platform.node()
        self.components = {}     # comp_id → nn.Module
        self.kv_caches = {}      # comp_id → DynamicCache or None
        self.comp_types = {}     # comp_id → str (layer, attention, ffn, expert, embedding, lm_head)
        self.running = True
        self.lock = threading.Lock()
        self.stats = {"bytes_sent": 0, "bytes_received": 0, "forward_calls": 0}
        self.broadcast_addr = "255.255.255.255"  # works on any network

        self.loaded_model_id = None

        # Component disk cache
        self.cache_dir = COMPONENT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, comp_id, model_id=""):
        """Get the disk cache path for a component scoped by model_id."""
        safe_model = (model_id or "default").replace("/", "_").replace("\\", "_")
        safe_comp = comp_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_model}_{safe_comp}.pt"

    def _save_component(self, comp_id, module, model_id=""):
        """Save a component to disk cache scoped by model_id."""
        try:
            path = self._cache_path(comp_id, model_id or "default")
            torch.save({
                "module": module,
                "comp_id": comp_id,
                "model_id": model_id or "default",
                "comp_type": self.comp_types.get(comp_id, "unknown"),
            }, path)
        except Exception as e:
            print(f"  ⚠ Cache save failed for {comp_id}: {e}")

    def _load_cached_for_model(self, model_id):
        """Load cached components for a specific model from disk."""
        safe_model = (model_id or "default").replace("/", "_").replace("\\", "_")
        loaded_count = 0
        for f in self.cache_dir.glob(f"{safe_model}_*.pt"):
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
                if isinstance(data, dict) and "module" in data:
                    comp_id = data.get("comp_id")
                    if not comp_id:
                        name = f.stem
                        parts = name.split("_", 1)
                        comp_id = parts[1] if len(parts) == 2 else name
                    module = data["module"].cpu().eval()
                    self.components[comp_id] = module
                    self.comp_types[comp_id] = data.get("comp_type", "unknown")
                    self.kv_caches[comp_id] = None
                    loaded_count += 1
            except Exception as e:
                print(f"  ⚠ Failed to load cache file {f.name}: {e}")
        if loaded_count:
            self.loaded_model_id = model_id
            print(f"  ✓ Loaded {loaded_count} cached component(s) from disk for model '{model_id}'")

    # ── Startup ───────────────────────────────────────────────────────────
    def start(self):
        print(f"\033[1;96m{'═' * 55}")
        print(f"  Worker Node: {self.hostname}")
        print(f"  Port:        {self.port}")
        print(f"  CPU:         {platform.processor() or 'Unknown'}")
        print(f"  Cores:       {psutil.cpu_count(logical=True)}")
        print(f"  RAM:         {psutil.virtual_memory().total / (1024**3):.1f} GB")
        gpu = "None"
        if torch.cuda.is_available():
            gpu = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)"
        print(f"  GPU:         {gpu}")
        print(f"{'═' * 55}\033[0m")

        # Start UDP discovery broadcast
        t = threading.Thread(target=self._discovery_loop, daemon=True)
        t.start()
        print(f"  ✓ Discovery broadcast on UDP port {DISC_PORT}")

        # Start TCP server
        self._serve()

    def _discovery_loop(self):
        """Broadcast presence via UDP every 3 seconds."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(1)

        while self.running:
            try:
                info = {
                    "hostname": self.hostname,
                    "port": self.port,
                    "ram_total": psutil.virtual_memory().total,
                    "ram_free": psutil.virtual_memory().available,
                    "cpu": platform.processor() or "Unknown",
                    "cores": psutil.cpu_count(logical=True),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                }
                msg = MAGIC + pickle.dumps(info)
                sock.sendto(msg, (self.broadcast_addr, DISC_PORT))
            except Exception:
                pass
            time.sleep(3)

    def _serve(self):
        """TCP server — handles one client at a time for simplicity."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(5)  # allow more pending connections for VPN peers
        print(f"  ✓ Listening on TCP port {self.port}")
        print(f"  Waiting for controller connection...\n")

        while self.running:
            try:
                conn, addr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"  ← Connected: {addr[0]}:{addr[1]}")
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
            except Exception as e:
                print(f"  ✗ Accept error: {e}")

    def _handle(self, conn, addr):
        try:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            conn.settimeout(120)  # generous timeout for VPN
            while True:
                msg, nbytes = recv_msg(conn)
                self.stats["bytes_received"] += nbytes

                with self.lock:
                    response = self._dispatch(msg)

                sent = send_msg(conn, response)
                self.stats["bytes_sent"] += sent
        except (ConnectionError, EOFError, OSError):
            pass
        finally:
            conn.close()
            print(f"  ✗ Disconnected: {addr[0]}:{addr[1]}")

    # ── Command Dispatch ──────────────────────────────────────────────────
    def _dispatch(self, msg):
        cmd = msg.get("cmd", "")
        try:
            handler = {
                "ping":              self._cmd_ping,
                "info":              self._cmd_info,
                "status":            self._cmd_status,
                "check_components":  self._cmd_check_components,
                "load":              self._cmd_load,
                "load_cached":       self._cmd_load_cached,
                "unload":            self._cmd_unload,
                "clear_kv":          self._cmd_clear_kv,
                "forward_layer":     self._cmd_forward_layer,
                "forward_attention": self._cmd_forward_attention,
                "forward_ffn":       self._cmd_forward_ffn,
                "forward_expert":    self._cmd_forward_expert,
                "forward_embedding": self._cmd_forward_embedding,
                "forward_lm_head":   self._cmd_forward_lm_head,
            }.get(cmd)
            if handler:
                return handler(msg)
            return {"status": "error", "msg": f"Unknown command: {cmd}"}
        except Exception as e:
            return {"status": "error", "msg": str(e)}

    # ── Commands ──────────────────────────────────────────────────────────
    def _cmd_ping(self, msg):
        return {"status": "ok", "hostname": self.hostname}

    def _cmd_info(self, msg):
        return {
            "status": "ok",
            "hostname": self.hostname,
            "ram_total": psutil.virtual_memory().total,
            "ram_free": psutil.virtual_memory().available,
            "cpu": platform.processor() or "Unknown",
            "cores": psutil.cpu_count(logical=True),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "components": list(self.components.keys()),
        }

    def _cmd_status(self, msg):
        comp_info = {}
        for cid, module in self.components.items():
            size = sum(p.numel() * p.element_size() for p in module.parameters())
            comp_info[cid] = {
                "type": self.comp_types.get(cid, "unknown"),
                "class": type(module).__name__,
                "size_bytes": size,
                "has_kv": cid in self.kv_caches and self.kv_caches[cid] is not None,
            }
        return {
            "status": "ok",
            "components": comp_info,
            "ram_free": psutil.virtual_memory().available,
            "stats": dict(self.stats),
        }

    def _cmd_load(self, msg):
        comp_id = msg["component_id"]
        comp_type = msg.get("component_type", "unknown")
        model_id = msg.get("model_id", "default")
        module = msg["module"]

        # If switching models, unload old model's components
        if self.loaded_model_id and self.loaded_model_id != model_id:
            self.components.clear()
            self.comp_types.clear()
            self.kv_caches.clear()
            gc.collect()

        self.loaded_model_id = model_id
        module = module.cpu().eval()
        self.components[comp_id] = module
        self.comp_types[comp_id] = comp_type
        self.kv_caches[comp_id] = None

        size = sum(p.numel() * p.element_size() for p in module.parameters())
        self._save_component(comp_id, module, model_id)
        print(f"  ✓ Loaded: {comp_id} [{model_id}] ({comp_type}, {type(module).__name__}, "
              f"{size / 1024**2:.1f} MB)")
        return {"status": "ok", "size_bytes": size}

    def _cmd_unload(self, msg):
        comp_id = msg.get("component_id")
        if comp_id and comp_id in self.components:
            del self.components[comp_id]
            self.kv_caches.pop(comp_id, None)
            self.comp_types.pop(comp_id, None)
            gc.collect()
            print(f"  ✓ Unloaded: {comp_id}")
        return {"status": "ok"}

    def _cmd_clear_kv(self, msg):
        comp_id = msg.get("component_id")
        if comp_id:
            self.kv_caches[comp_id] = None
        else:
            self.kv_caches = {k: None for k in self.kv_caches}
        return {"status": "ok"}

    def _cmd_check_components(self, msg):
        """Report components loaded in memory for the requested model_id."""
        req_model = msg.get("model_id")

        # If model is different, clear currently loaded memory components
        if req_model and self.loaded_model_id and self.loaded_model_id != req_model:
            self.components.clear()
            self.comp_types.clear()
            self.kv_caches.clear()
            self.loaded_model_id = None
            gc.collect()

        # If memory is empty, try loading cached components for this model from disk
        if req_model and not self.components:
            self._load_cached_for_model(req_model)

        loaded = list(self.components.keys()) if (not req_model or self.loaded_model_id == req_model) else []
        return {
            "status": "ok",
            "model_id": self.loaded_model_id,
            "loaded": loaded,
            "hostname": self.hostname,
        }

    def _cmd_load_cached(self, msg):
        """Load a component from disk cache into memory using provided module as template."""
        comp_id = msg["component_id"]
        comp_type = msg.get("component_type", "unknown")
        model_hash = msg.get("model_hash", "default")
        module = msg["module"]  # template module with correct architecture

        cache_path = self._cache_path(comp_id, model_hash)
        if cache_path.exists():
            try:
                cached_data = torch.load(cache_path, map_location="cpu", weights_only=True)
                module.load_state_dict(cached_data["state_dict"])
                module = module.cpu().eval()
                self.components[comp_id] = module
                self.comp_types[comp_id] = comp_type
                self.kv_caches[comp_id] = None
                size = sum(p.numel() * p.element_size() for p in module.parameters())
                print(f"  ✓ Loaded from cache: {comp_id} ({size / 1024**2:.1f} MB)")
                return {"status": "ok", "size_bytes": size, "from_cache": True}
            except Exception as e:
                print(f"  ⚠ Cache load failed for {comp_id}: {e}, loading fresh")

        # Fall through to normal load
        return self._cmd_load(msg)

def _get_module_floating_dtype(module):
    """Find the floating point dtype of a module's parameters or buffers."""
    for p in module.parameters():
        if p.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return p.dtype
    for b in module.buffers():
        if b.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return b.dtype
    return None

def _match_dtype(tensor, module):
    """Ensure tensor dtype matches the module's active floating point dtype."""
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return tensor
    dt = _get_module_floating_dtype(module)
    if dt is not None and tensor.dtype != dt:
        return tensor.to(dt)
    return tensor

def _match_tuple_dtype(tup, module):
    if tup is None:
        return tup
    dt = _get_module_floating_dtype(module)
    if dt is not None:
        return tuple(t.to(dt) if isinstance(t, torch.Tensor) and t.dtype != dt else t for t in tup)
    return tup


    # ── Forward Passes ────────────────────────────────────────────────────
    def _get_cache(self, comp_id):
        """Get or create a DynamicCache for a component."""
        cache = self.kv_caches.get(comp_id)
        if cache is None:
            from transformers import DynamicCache
            cache = DynamicCache()
            self.kv_caches[comp_id] = cache
        return cache

    def _run_module_safe(self, module, hs, pos_emb, kwargs):
        """Execute a module with multi-stage signature handling and automatic dtype mismatch recovery."""
        try:
            return module(hs, **kwargs)
        except TypeError as e:
            if "past_key_values" in str(e) or "unexpected keyword argument" in str(e):
                kwargs["past_key_value"] = kwargs.pop("past_key_values", None)
                try:
                    return module(hs, **kwargs)
                except TypeError:
                    kwargs.pop("position_embeddings", None)
                    try:
                        return module(hs, **kwargs)
                    except TypeError:
                        kwargs.pop("attention_mask", None)
                        return module(hs, **kwargs)
            else:
                kwargs.pop("position_embeddings", None)
                try:
                    return module(hs, **kwargs)
                except TypeError:
                    kwargs.pop("attention_mask", None)
                    return module(hs, **kwargs)
        except RuntimeError as e:
            if "must have the same dtype" in str(e) or "expected dtype" in str(e):
                for dt in (torch.bfloat16, torch.float16, torch.float32):
                    if hs.dtype == dt:
                        continue
                    try:
                        fallback_hs = hs.to(dt)
                        if pos_emb is not None:
                            kwargs["position_embeddings"] = tuple(p.to(dt) if isinstance(p, torch.Tensor) else p for p in pos_emb)
                        return module(fallback_hs, **kwargs)
                    except Exception:
                        continue
            raise

    def _cmd_forward_layer(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        hs = _match_dtype(msg["hidden_states"], module)
        pos_ids = msg.get("position_ids")
        pos_emb = _match_tuple_dtype(msg.get("position_embeddings"), module)
        cache = self._get_cache(comp_id)

        with torch.no_grad():
            kwargs = dict(position_ids=pos_ids, past_key_values=cache, use_cache=True, attention_mask=None)
            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb
            outputs = self._run_module_safe(module, hs, pos_emb, kwargs)

        hidden_out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        # Update cache reference (DynamicCache is mutated in-place, but just in case)
        if not isinstance(outputs, torch.Tensor) and len(outputs) > 1 and outputs[1] is not None:
            self.kv_caches[comp_id] = outputs[1]

        self.stats["forward_calls"] += 1
        return {"status": "ok", "hidden_states": hidden_out}

    def _cmd_forward_attention(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        hs = _match_dtype(msg["hidden_states"], module)
        pos_ids = msg.get("position_ids")
        pos_emb = _match_tuple_dtype(msg.get("position_embeddings"), module)
        cache = self._get_cache(comp_id)

        with torch.no_grad():
            kwargs = dict(position_ids=pos_ids, past_key_values=cache, use_cache=True, attention_mask=None)
            if pos_emb is not None:
                kwargs["position_embeddings"] = pos_emb
            outputs = self._run_module_safe(module, hs, pos_emb, kwargs)

        attn_out = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        if not isinstance(outputs, torch.Tensor) and len(outputs) > 2 and outputs[2] is not None:
            self.kv_caches[comp_id] = outputs[2]

        self.stats["forward_calls"] += 1
        return {"status": "ok", "hidden_states": attn_out}

    def _cmd_forward_ffn(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        hs = _match_dtype(msg["hidden_states"], module)
        with torch.no_grad():
            out = module(hs)
        # MoE blocks return (output, router_logits); dense MLPs return tensor
        hidden_out = out[0] if isinstance(out, tuple) else out

        self.stats["forward_calls"] += 1
        return {"status": "ok", "hidden_states": hidden_out}

    def _cmd_forward_expert(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        hs = _match_dtype(msg["hidden_states"], module)
        with torch.no_grad():
            out = module(hs)

        self.stats["forward_calls"] += 1
        return {"status": "ok", "hidden_states": out}

    def _cmd_forward_embedding(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        input_ids = msg["input_ids"]
        with torch.no_grad():
            out = module(input_ids)

        self.stats["forward_calls"] += 1
        return {"status": "ok", "hidden_states": out}

    def _cmd_forward_lm_head(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        hs = _match_dtype(msg["hidden_states"], module)
        with torch.no_grad():
            logits = module(hs)

        self.stats["forward_calls"] += 1
        return {"status": "ok", "logits": logits}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Worker node for distributed inference")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port (default: 9900)")
    parser.add_argument("--broadcast", type=str, default="255.255.255.255",
                        help="Broadcast address for UDP discovery (default: 255.255.255.255)")
    args = parser.parse_args()

    worker = Worker(args.port)
    worker.broadcast_addr = args.broadcast
    try:
        worker.start()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        worker.running = False
        sys.exit(0)


if __name__ == "__main__":
    main()
