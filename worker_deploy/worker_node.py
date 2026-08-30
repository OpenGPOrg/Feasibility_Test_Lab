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


def _get_gpu_info():
    """Detect NVIDIA CUDA or Apple Silicon Metal (MPS) GPU device."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"{torch.cuda.get_device_name(0)} ({props.total_memory / (1024**3):.1f} GB)"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple Silicon GPU (Metal Performance Shaders / MPS)"
    return None


def _detect_device(override=None):
    if override and override != "auto":
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _to_cpu_recursive(obj):
    """Recursively detach and convert all tensors to device-agnostic CPU storage before network serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: _to_cpu_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [_to_cpu_recursive(v) for v in obj]
        return tuple(converted) if isinstance(obj, tuple) else converted
    return obj


def send_msg(sock, obj):
    """Send a Python object (including tensors/modules) over a socket."""
    obj = _to_cpu_recursive(obj)
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)
    return 8 + len(data)


class Universal_Unpickler(pickle.Unpickler):
    """Cross-platform unpickler mapping any remote device storage (CUDA, MPS, XPU, NPU) to CPU."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def recv_msg(sock):
    """Receive a Python object from a socket. Returns (obj, bytes_received)."""
    raw = _recv_exact(sock, 8)
    length = struct.unpack("!Q", raw)[0]
    data = _recv_exact(sock, length)
    try:
        obj = Universal_Unpickler(io.BytesIO(data)).load()
    except Exception:
        obj = pickle.loads(data)
    return obj, 8 + length


def get_kv_cache_size_bytes(cache):
    """Calculate exact memory occupied by a DynamicCache or tuple KV cache."""
    if cache is None:
        return 0
    total_bytes = 0
    try:
        if hasattr(cache, "layers"):
            for layer in cache.layers:
                if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
                    total_bytes += layer.keys.numel() * layer.keys.element_size()
                if hasattr(layer, "values") and isinstance(layer.values, torch.Tensor):
                    total_bytes += layer.values.numel() * layer.values.element_size()
            if total_bytes > 0:
                return total_bytes
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            for k in cache.key_cache:
                if isinstance(k, torch.Tensor):
                    total_bytes += k.numel() * k.element_size()
            for v in cache.value_cache:
                if isinstance(v, torch.Tensor):
                    total_bytes += v.numel() * v.element_size()
            if total_bytes > 0:
                return total_bytes
        if isinstance(cache, (list, tuple)):
            for item in cache:
                if isinstance(item, torch.Tensor):
                    total_bytes += item.numel() * item.element_size()
                elif isinstance(item, (list, tuple)):
                    total_bytes += get_kv_cache_size_bytes(item)
            return total_bytes
    except Exception:
        pass
    return total_bytes


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


# ═══════════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════════
class Worker:
    def __init__(self, port, device="auto"):
        self.port = port
        self.hostname = platform.node()
        self.device = _detect_device(device)
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
                "module": module.cpu(),
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
                    module = data["module"].to(self.device).eval()
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
        print(f"  Device:      {self.device}")
        print(f"  CPU:         {platform.processor() or 'Unknown'}")
        print(f"  Cores:       {psutil.cpu_count(logical=True)}")
        print(f"  RAM:         {psutil.virtual_memory().total / (1024**3):.1f} GB")
        gpu = _get_gpu_info() or "None"
        print(f"  GPU/Accel:   {gpu}")
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
                "get_memory":        self._cmd_get_memory,
                "forward_layer":     self._cmd_forward_layer,
                "forward_stage":     self._cmd_forward_stage,
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
            "device": str(self.device),
            "gpu": _get_gpu_info(),
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
            "device": str(self.device),
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
        module = module.to(self.device).eval()
        self.components[comp_id] = module
        self.comp_types[comp_id] = comp_type
        self.kv_caches[comp_id] = None

        size = sum(p.numel() * p.element_size() for p in module.parameters())
        self._save_component(comp_id, module, model_id)
        print(f"  ✓ Loaded on {self.device}: {comp_id} [{model_id}] ({comp_type}, {type(module).__name__}, "
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

    def _cmd_get_memory(self, msg):
        """Return memory footprint of loaded layers, active KV caches, and process RSS/VRAM."""
        layer_bytes = 0
        for m in self.components.values():
            try:
                layer_bytes += sum(p.numel() * p.element_size() for p in m.parameters())
            except Exception:
                pass

        kv_bytes = sum(get_kv_cache_size_bytes(c) for c in self.kv_caches.values())

        ram_rss = 0
        try:
            import os
            proc = psutil.Process(os.getpid())
            ram_rss = proc.memory_info().rss
        except Exception:
            pass

        vram_alloc = 0
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                vram_alloc = torch.mps.current_allocated_memory()
            except Exception:
                pass
        return {
            "status": "ok",
            "layer_bytes": layer_bytes,
            "kv_bytes": kv_bytes,
            "ram_rss": ram_rss,
            "vram_alloc": vram_alloc,
        }

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
                module = module.to(self.device).eval()
                self.components[comp_id] = module
                self.comp_types[comp_id] = comp_type
                self.kv_caches[comp_id] = None
                size = sum(p.numel() * p.element_size() for p in module.parameters())
                print(f"  ✓ Loaded from cache onto {self.device}: {comp_id} ({size / 1024**2:.1f} MB)")
                return {"status": "ok", "size_bytes": size, "from_cache": True}
            except Exception as e:
                print(f"  ⚠ Cache load failed for {comp_id}: {e}, loading fresh")

        # Fall through to normal load
        return self._cmd_load(msg)

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
        # Ensure module and inputs are strictly on the active worker device (CUDA, Apple Silicon MPS, or CPU)
        module = module.to(self.device)
        hs = hs.to(self.device)
        if pos_emb is not None:
            pos_emb = tuple(p.to(self.device) if isinstance(p, torch.Tensor) else p for p in pos_emb)
        if kwargs.get("position_ids") is not None and isinstance(kwargs["position_ids"], torch.Tensor):
            kwargs["position_ids"] = kwargs["position_ids"].to(self.device)

        def _try_call(curr_hs, curr_pos_emb, curr_kwargs):
            k = dict(curr_kwargs)
            # Strategy 1: Standard full keyword args
            try:
                return module(curr_hs, **k)
            except (TypeError, ValueError):
                pass

            # Strategy 2: Qwen2 / Llama 3.2 with explicit position_embeddings & attention_mask
            try:
                return module(
                    curr_hs,
                    position_embeddings=curr_pos_emb,
                    attention_mask=None,
                    past_key_value=k.get("past_key_values") or k.get("past_key_value"),
                    position_ids=k.get("position_ids")
                )
            except (TypeError, ValueError):
                pass

            # Strategy 3: Positional parameters (hidden_states, pos_emb, mask, past_kv, cache_pos, pos_ids)
            try:
                return module(
                    curr_hs,
                    curr_pos_emb,
                    None,
                    k.get("past_key_values") or k.get("past_key_value"),
                    None,
                    k.get("position_ids")
                )
            except (TypeError, ValueError):
                pass

            # Strategy 4: Legacy past_key_value
            try:
                k_legacy = dict(k)
                if "past_key_values" in k_legacy:
                    k_legacy["past_key_value"] = k_legacy.pop("past_key_values")
                k_legacy.pop("position_embeddings", None)
                return module(curr_hs, **k_legacy)
            except (TypeError, ValueError):
                pass

            # Strategy 5: Without position embeddings
            try:
                k_nopos = dict(k)
                k_nopos.pop("position_embeddings", None)
                k_nopos.pop("attention_mask", None)
                return module(curr_hs, **k_nopos)
            except (TypeError, ValueError):
                pass

            # Strategy 6: Position IDs only
            try:
                return module(curr_hs, position_ids=k.get("position_ids"))
            except (TypeError, ValueError):
                pass

            # Strategy 7: Pure tensor call
            return module(curr_hs)

        try:
            return _try_call(hs, pos_emb, kwargs)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ("dtype", "mat1", "m1 and m2", "same type", "half", "bfloat16", "float")):
                for dt in (torch.bfloat16, torch.float16, torch.float32):
                    if hs.dtype == dt:
                        continue
                    try:
                        fallback_hs = hs.to(dt)
                        fallback_pos_emb = tuple(p.to(dt) if isinstance(p, torch.Tensor) else p for p in pos_emb) if pos_emb is not None else None
                        return _try_call(fallback_hs, fallback_pos_emb, kwargs)
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
        return {"status": "ok", "hidden_states": hidden_out.cpu() if isinstance(hidden_out, torch.Tensor) else hidden_out}

    def _cmd_forward_stage(self, msg):
        """Execute a contiguous sequence of layers locally on the worker without intermediate network hops."""
        component_ids = msg.get("component_ids")
        if not component_ids:
            start_l = msg.get("start_layer")
            end_l = msg.get("end_layer")
            if start_l is not None and end_l is not None:
                component_ids = [f"layer_{i}" for i in range(start_l, end_l + 1)]
            else:
                return {"status": "error", "msg": "Missing component_ids or start/end layer"}

        hs = msg["hidden_states"]
        pos_ids = msg.get("position_ids")
        raw_pos_emb = msg.get("position_embeddings")

        with torch.no_grad():
            for comp_id in component_ids:
                module = self.components.get(comp_id)
                if module is None:
                    return {"status": "error", "msg": f"Layer not found on worker: {comp_id}"}

                hs = _match_dtype(hs, module)
                pos_emb = _match_tuple_dtype(raw_pos_emb, module)
                cache = self._get_cache(comp_id)

                kwargs = dict(position_ids=pos_ids, past_key_values=cache, use_cache=True, attention_mask=None)
                if pos_emb is not None:
                    kwargs["position_embeddings"] = pos_emb

                outputs = self._run_module_safe(module, hs, pos_emb, kwargs)
                hs = outputs if isinstance(outputs, torch.Tensor) else outputs[0]

                if not isinstance(outputs, torch.Tensor) and len(outputs) > 1 and outputs[1] is not None:
                    self.kv_caches[comp_id] = outputs[1]

                self.stats["forward_calls"] += 1

        return {"status": "ok", "hidden_states": hs.cpu() if isinstance(hs, torch.Tensor) else hs}

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
        return {"status": "ok", "hidden_states": attn_out.cpu() if isinstance(attn_out, torch.Tensor) else attn_out}

    def _cmd_forward_ffn(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        hs = _match_dtype(msg["hidden_states"], module)
        with torch.no_grad():
            out = module(hs.to(self.device))
        # MoE blocks return (output, router_logits); dense MLPs return tensor
        hidden_out = out[0] if isinstance(out, tuple) else out

        self.stats["forward_calls"] += 1
        return {"status": "ok", "hidden_states": hidden_out.cpu() if isinstance(hidden_out, torch.Tensor) else hidden_out}

    def _cmd_forward_expert(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        hs = _match_dtype(msg["hidden_states"], module)
        with torch.no_grad():
            out = module(hs.to(self.device))

        self.stats["forward_calls"] += 1
        return {"status": "ok", "hidden_states": out.cpu() if isinstance(out, torch.Tensor) else out}

    def _cmd_forward_embedding(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        input_ids = msg["input_ids"]
        with torch.no_grad():
            out = module(input_ids.to(self.device) if isinstance(input_ids, torch.Tensor) else input_ids)

        self.stats["forward_calls"] += 1
        return {"status": "ok", "hidden_states": out.cpu() if isinstance(out, torch.Tensor) else out}

    def _cmd_forward_lm_head(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        hs = _match_dtype(msg["hidden_states"], module)
        with torch.no_grad():
            logits = module(hs.to(self.device))

        self.stats["forward_calls"] += 1
        return {"status": "ok", "logits": logits.cpu() if isinstance(logits, torch.Tensor) else logits}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Worker node for distributed inference")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port (default: 9900)")
    parser.add_argument("--broadcast", type=str, default="255.255.255.255",
                        help="Broadcast address for UDP discovery (default: 255.255.255.255)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute accelerator device (auto, cpu, cuda, mps) (default: auto)")
    args = parser.parse_args()

    worker = Worker(args.port, device=args.device)
    worker.broadcast_addr = args.broadcast
    try:
        worker.start()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        worker.running = False
        sys.exit(0)


if __name__ == "__main__":
    main()
