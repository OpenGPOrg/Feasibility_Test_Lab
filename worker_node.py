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
import json
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


def _move_module_to_device(module, device):
    """Move module to device and ensure bitsandbytes quant_state tensors are also moved to the device."""
    if module is None or device is None:
        return module
    if hasattr(module, "to"):
        module = module.to(device)
    for p in module.parameters():
        if hasattr(p, "quant_state") and p.quant_state is not None:
            try:
                p.quant_state.to(device)
            except Exception:
                pass
    return module



def _patch_transformers_qwen3_5():
    """Auto-patch Qwen3.5 linear attention to be robust across all transformers versions (5.5.x through 5.16+)."""
    try:
        from transformers.cache_utils import DynamicCache, LinearAttentionLayer
        if not getattr(DynamicCache, "_opengp_patched", False):
            orig_has_prev = getattr(DynamicCache, "has_previous_state", None)
            if orig_has_prev is not None:
                def patched_has_prev(self, layer_idx=None):
                    res = orig_has_prev(self, layer_idx=layer_idx)
                    if isinstance(res, dict):
                        return res.get(0, False)
                    return bool(res)
                DynamicCache.has_previous_state = patched_has_prev
            DynamicCache._opengp_patched = True

        if not getattr(LinearAttentionLayer, "_opengp_patched", False):
            orig_urs = getattr(LinearAttentionLayer, "update_recurrent_state", None)
            if orig_urs is not None:
                def safe_update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
                    if hasattr(self, 'is_recurrent_states_initialized'):
                        is_init = self.is_recurrent_states_initialized.get(state_idx, False) if isinstance(self.is_recurrent_states_initialized, dict) else self.is_recurrent_states_initialized
                        if not is_init:
                            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
                    target = self.recurrent_states[state_idx] if isinstance(self.recurrent_states, dict) else self.recurrent_states
                    if target is not None and hasattr(target, 'shape') and target.shape != recurrent_states.shape:
                        if recurrent_states.ndim == target.ndim + 1 and recurrent_states.shape[0] == 1:
                            target.copy_(recurrent_states.squeeze(0))
                        elif recurrent_states.ndim + 1 == target.ndim and target.shape[0] == 1:
                            target.squeeze(0).copy_(recurrent_states)
                        else:
                            target.copy_(recurrent_states)
                    elif target is not None:
                        target.copy_(recurrent_states)
                    return target
                LinearAttentionLayer.update_recurrent_state = safe_update_recurrent_state

            orig_ucs = getattr(LinearAttentionLayer, "update_conv_state", None)
            if orig_ucs is not None:
                def safe_update_conv_state(self, conv_states, state_idx=0, **kwargs):
                    if hasattr(self, 'is_conv_states_initialized'):
                        is_init = self.is_conv_states_initialized.get(state_idx, False) if isinstance(self.is_conv_states_initialized, dict) else self.is_conv_states_initialized
                        if not is_init:
                            self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, **kwargs)
                    return orig_ucs(self, conv_states, state_idx=state_idx, **kwargs)
                LinearAttentionLayer.update_conv_state = safe_update_conv_state

            LinearAttentionLayer._opengp_patched = True

        from transformers.models.qwen3_5 import modeling_qwen3_5
        if hasattr(modeling_qwen3_5, "Qwen3_5GatedDeltaNet"):
            target_cls = modeling_qwen3_5.Qwen3_5GatedDeltaNet
            if not getattr(target_cls, "_opengp_patched_v2", False):
                orig_fwd = getattr(target_cls, "_orig_fwd", target_cls.forward)
                target_cls._orig_fwd = orig_fwd
                def patched_fwd(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
                    if not getattr(self, "_opengp_wrapped", False):
                        orig_conv_up = self.causal_conv1d_update
                        def safe_conv_up(mixed_qkv, conv_state, *c_args, **c_kwargs):
                            if isinstance(conv_state, dict) and 0 in conv_state:
                                conv_state = conv_state[0]
                            return orig_conv_up(mixed_qkv, conv_state, *c_args, **c_kwargs)
                        self.causal_conv1d_update = safe_conv_up

                        orig_rec_rule = self.recurrent_gated_delta_rule
                        def safe_rec_rule(*r_args, **r_kwargs):
                            if "initial_state" in r_kwargs and isinstance(r_kwargs["initial_state"], dict):
                                r_kwargs["initial_state"] = r_kwargs["initial_state"].get(0, None)
                            return orig_rec_rule(*r_args, **r_kwargs)
                        self.recurrent_gated_delta_rule = safe_rec_rule
                        self._opengp_wrapped = True

                    return orig_fwd(self, hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
                target_cls.forward = patched_fwd
                target_cls._opengp_patched_v2 = True
    except Exception:
        pass

_patch_transformers_qwen3_5()


# ═══════════════════════════════════════════════════════════════════════════════
# Worker
# ═══════════════════════════════════════════════════════════════════════════════
class Worker:
    def __init__(self, port=DEFAULT_PORT, device="auto", connect_target=None):
        _patch_transformers_qwen3_5()
        self.port = port
        self.connect_target = connect_target
        self.hostname = platform.node()
        self.device = _detect_device(device)
        self.components = {}     # comp_id → nn.Module
        self.kv_caches = {}      # comp_id → DynamicCache or None
        self.comp_types = {}     # comp_id → str (layer, attention, ffn, expert, embedding, lm_head)
        self.running = True
        self.lock = threading.Lock()
        self.stats = {"bytes_sent": 0, "bytes_received": 0, "forward_calls": 0}
        self.broadcast_addr = "255.255.255.255"  # works on any network
        self.peer_sockets = {}  # "host:port" -> socket for direct peer-to-peer ring pipeline

        self.loaded_model_id = None

        self.profiler_running = False
        self.profiler_thread = None
        self.memory_log = []

        # Component disk cache
        self.cache_dir = COMPONENT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_peer_socket(self, host, port):
        """Get or establish a persistent direct TCP socket to a peer worker node over LAN."""
        key = f"{host}:{port}"
        sock = self.peer_sockets.get(key)
        if sock is not None:
            try:
                sock.getpeername()
                return sock
            except Exception:
                try: sock.close()
                except Exception: pass
                self.peer_sockets.pop(key, None)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.settimeout(60.0)
        sock.connect((host, port))
        self.peer_sockets[key] = sock
        return sock

    def _cache_path(self, comp_id, model_id=""):
        """Get the disk cache path for a component scoped by model_id."""
        safe_model = (model_id or "default").replace("/", "_").replace("\\", "_")
        safe_comp = comp_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_model}_{safe_comp}.pt"

    def _save_component(self, comp_id, module, model_id=""):
        """Save a component to disk cache scoped by model_id."""
        try:
            path = self._cache_path(comp_id, model_id or "default")
            is_cuda = any(p.is_cuda for p in module.parameters())
            if is_cuda:
                import copy
                save_mod = copy.deepcopy(module).cpu()
            else:
                save_mod = module
            torch.save({
                "module": save_mod,
                "comp_id": comp_id,
                "model_id": model_id or "default",
                "comp_type": self.comp_types.get(comp_id, "unknown"),
            }, path)
        except Exception as e:
            print(f"  ⚠ Cache save failed for {comp_id}: {e}")

    def _load_cached_for_model(self, model_id, target_components=None):
        """Load cached components for a specific model from disk into VRAM."""
        if not self.cache_dir.exists():
            return
        safe_model = (model_id or "default").replace("/", "_").replace("\\", "_")
        prefix = f"{safe_model}_"
        loaded_count = 0

        # If specific target components are specified, only load those to prevent VRAM overflow
        if target_components is not None:
            paths_to_check = []
            for cid in target_components:
                if cid in self.components:
                    continue
                safe_comp = cid.replace("/", "_").replace("\\", "_")
                p = self.cache_dir / f"{prefix}{safe_comp}.pt"
                if p.exists():
                    paths_to_check.append((p, cid))
        else:
            paths_to_check = []
            for path in sorted(self.cache_dir.glob(f"{prefix}*.pt")):
                comp_id_from_stem = path.stem[len(prefix):]
                if comp_id_from_stem not in self.components:
                    paths_to_check.append((path, comp_id_from_stem))

        for path, comp_id_from_stem in paths_to_check:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                comp_id = data.get("comp_id", comp_id_from_stem)
                if comp_id in self.components:
                    continue

                module = data["module"]
                comp_type = data.get("comp_type", "unknown")
                if hasattr(module, "to"):
                    module = module.to(self.device).eval()
                self.components[comp_id] = module
                self.comp_types[comp_id] = comp_type
                if comp_type in ("layer", "attention"):
                    self.kv_caches[comp_id] = None
                loaded_count += 1
            except Exception as e:
                print(f"  ⚠ Failed to load cached component from {path}: {e}")
        if loaded_count > 0:
            self.loaded_model_id = model_id
            print(f"  ✓ Loaded {loaded_count} cached component(s) from disk for model '{model_id}'")

    # ── Startup ───────────────────────────────────────────────────────────
    def start(self):
        print(f"\033[1;96m{'═' * 55}")
        print(f"  Worker Node: {self.hostname}")
        if self.connect_target:
            print(f"  Mode:        Reverse Connection (Outbound to Master)")
            print(f"  Master Target: {self.connect_target}")
        else:
            print(f"  Mode:        Direct Server (Listening)")
            print(f"  Port:        {self.port}")
        print(f"  Device:      {self.device}")
        print(f"  CPU:         {platform.processor() or 'Unknown'}")
        print(f"  Cores:       {psutil.cpu_count(logical=True)}")
        print(f"  RAM:         {psutil.virtual_memory().total / (1024**3):.1f} GB")
        gpu = _get_gpu_info() or "None"
        print(f"  GPU/Accel:   {gpu}")
        print(f"{'═' * 55}\033[0m")

        if self.connect_target:
            if ":" in self.connect_target:
                m_host, m_port_str = self.connect_target.split(":", 1)
                m_port = int(m_port_str)
            else:
                m_host = self.connect_target
                m_port = DEFAULT_PORT
            self._connect_to_master(m_host, m_port)
        else:
            # Start UDP discovery broadcast
            t = threading.Thread(target=self._discovery_loop, daemon=True)
            t.start()
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

    def _connect_to_master(self, host, port):
        """Reverse connection: connect to Master's ReverseCoordinatorListener."""
        print(f"  Initiating reverse connection to Master at {host}:{port}...")
        while self.running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                sock.connect((host, port))
                print(f"  ✓ Connected to Master at {host}:{port}!")

                # Handshake
                handshake = {
                    "type": "openGP_worker_register",
                    "hostname": self.hostname,
                    "device": str(self.device),
                    "cores": psutil.cpu_count(logical=True),
                    "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                    "gpu": _get_gpu_info() or "None",
                    "components": list(self.components.keys()),
                }
                send_msg(sock, handshake)

                self._handle(sock, (host, port))
            except Exception as e:
                print(f"  ✗ Connection to Master failed: {e}. Retrying in 3s...")
                time.sleep(3)

    # ── TCP Server (Inbound Mode) ─────────────────────────────────────────
    def _serve(self):
        """TCP server — handles one client at a time for simplicity."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(5)
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
                "load_disk_component": self._cmd_load_disk_component,
                "unload":            self._cmd_unload,
                "clear_kv":          self._cmd_clear_kv,
                "get_memory":        self._cmd_get_memory,
                "forward_layer":           self._cmd_forward_layer,
                "forward_stage":           self._cmd_forward_stage,
                "pipeline_forward_chain":  self._cmd_pipeline_forward_chain,
                "forward_attention":       self._cmd_forward_attention,
                "forward_ffn":       self._cmd_forward_ffn,
                "forward_expert":    self._cmd_forward_expert,
                "forward_embedding": self._cmd_forward_embedding,
                "forward_lm_head":   self._cmd_forward_lm_head,
                "start_profiling":   self._cmd_start_profiling,
                "stop_profiling":    self._cmd_stop_profiling,
            }.get(cmd)
            if handler:
                return handler(msg)
            return {"status": "error", "msg": f"Unknown command: {cmd}"}
        except Exception as e:
            return {"status": "error", "msg": str(e)}

    # ── Commands ──────────────────────────────────────────────────────────
    def _cmd_start_profiling(self, msg):
        """Start recording VRAM and RAM at high frequency."""
        if self.profiler_running:
            return {"status": "ok", "msg": "Profiler already running"}
        self.profiler_running = True
        self.memory_log = []
        self.profiler_start_time = time.time()
        self.profiler_thread = threading.Thread(target=self._profiler_loop, daemon=True)
        self.profiler_thread.start()
        print(f"  📊 Background memory profiler started on {self.hostname}")
        return {"status": "ok"}

    def _cmd_stop_profiling(self, msg):
        """Stop profiling, generate plot, and return saved path."""
        if not self.profiler_running:
            return {"status": "ok", "msg": "Profiler was not running"}
        self.profiler_running = False
        if self.profiler_thread:
            self.profiler_thread.join(timeout=2.0)
        saved_path = self._plot_memory_profile()
        return {"status": "ok", "saved_path": saved_path}

    def _profiler_loop(self):
        proc = psutil.Process(os.getpid())
        while self.profiler_running:
            now = time.time() - self.profiler_start_time
            vram_mb = 0.0
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024**2)
            ram_mb = proc.memory_info().rss / (1024**2)
            self.memory_log.append((now, vram_mb, ram_mb))
            time.sleep(0.01)

    def _plot_memory_profile(self):
        if not self.memory_log:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ts, vram, ram = zip(*self.memory_log)
            plt.figure(figsize=(10, 5))
            if torch.cuda.is_available():
                plt.plot(ts, vram, label="GPU VRAM Allocated (MB)", color="red", linewidth=1.5)
            plt.plot(ts, ram, label="CPU RAM (RSS) (MB)", color="blue", linewidth=1.5)
            plt.title(f"Worker Memory Footprint During Inference ({self.hostname})")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Memory (MB)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            out_file = f"worker_memory_profile_{int(time.time())}.png"
            plt.savefig(out_file)
            plt.close()
            return out_file
        except Exception as e:
            print(f"  ⚠ Failed to plot worker profile: {e}")
            return None

    def _cmd_ping(self, msg):
        return {"status": "pong", "hostname": self.hostname}

    def _cmd_info(self, msg):
        gpu = _get_gpu_info()
        return {
            "status": "ok",
            "hostname": self.hostname,
            "device": str(self.device),
            "cpu": platform.processor() or "Unknown",
            "cores": psutil.cpu_count(logical=True),
            "ram_total": psutil.virtual_memory().total,
            "ram_avail": psutil.virtual_memory().available,
            "gpu": gpu,
            "torch_version": torch.__version__,
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
        # Save to disk while on CPU before moving to GPU
        self._save_component(comp_id, module, model_id)

        module = _move_module_to_device(module, self.device).eval()
        self.components[comp_id] = module
        self.comp_types[comp_id] = comp_type
        self.kv_caches[comp_id] = None

        size = sum(p.numel() * p.element_size() for p in module.parameters())
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
        required_comps = msg.get("required_components")

        # If model is different, clear currently loaded memory components
        if req_model and self.loaded_model_id and self.loaded_model_id != req_model:
            self.components.clear()
            self.comp_types.clear()
            self.kv_caches.clear()
            self.loaded_model_id = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # If required_components is specified, evict unassigned components from VRAM to prevent OOM
        if required_comps is not None:
            req_set = set(required_comps)
            to_evict = [cid for cid in list(self.components.keys()) if cid not in req_set]
            if to_evict:
                for cid in to_evict:
                    del self.components[cid]
                    self.comp_types.pop(cid, None)
                    self.kv_caches.pop(cid, None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print(f"  ✓ Evicted {len(to_evict)} unassigned component(s) from VRAM to prevent OOM")

            # Load only the missing required components from disk cache
            self._load_cached_for_model(req_model, target_components=req_set)

        # Collect list of available disk cached component IDs for this model
        disk_cached = []
        if req_model and self.cache_dir.exists():
            safe_model = (req_model or "default").replace("/", "_").replace("\\", "_")
            prefix = f"{safe_model}_"
            for p in self.cache_dir.glob(f"{prefix}*.pt"):
                disk_cached.append(p.stem[len(prefix):])

        loaded = list(self.components.keys()) if (not req_model or self.loaded_model_id == req_model) else []
        return {
            "status": "ok",
            "model_id": self.loaded_model_id,
            "loaded": loaded,
            "disk_cached": disk_cached,
            "hostname": self.hostname,
        }

    def _cmd_load_disk_component(self, msg):
        """Load a single component from disk cache into VRAM on demand."""
        comp_id = msg.get("component_id")
        model_id = msg.get("model_id", self.loaded_model_id or "default")
        if not comp_id or not self.cache_dir.exists():
            return {"status": "error", "msg": "Missing component_id or cache_dir"}

        safe_model = (model_id or "default").replace("/", "_").replace("\\", "_")
        safe_comp = comp_id.replace("/", "_").replace("\\", "_")
        path = self.cache_dir / f"{safe_model}_{safe_comp}.pt"
        if not path.exists():
            return {"status": "error", "msg": f"Not found on disk: {path.name}"}

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            module = data["module"]
            comp_type = data.get("comp_type", "unknown")
            if hasattr(module, "to"):
                module = _move_module_to_device(module, self.device).eval()
            self.components[comp_id] = module
            self.comp_types[comp_id] = comp_type
            if comp_type in ("layer", "attention"):
                self.kv_caches[comp_id] = None
            self.loaded_model_id = model_id
            size = sum(p.numel() * p.element_size() for p in module.parameters())
            print(f"  ✓ Loaded from disk onto {self.device}: {comp_id} ({size / 1024**2:.1f} MB)")
            return {"status": "ok", "size_bytes": size, "from_cache": True}
        except Exception as e:
            return {"status": "error", "msg": str(e)}

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
                module = _move_module_to_device(module, self.device).eval()
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
            module = self.components.get(comp_id)
            cfg = getattr(module, 'config', getattr(getattr(module, 'mlp', None), 'config', getattr(getattr(module, 'self_attn', None), 'config', getattr(getattr(module, 'linear_attn', None), 'config', None)))) if module is not None else None
            if cfg is None:
                for m in self.components.values():
                    c = getattr(m, 'config', getattr(getattr(m, 'mlp', None), 'config', None))
                    if c is not None:
                        cfg = c
                        break
            text_cfg = getattr(cfg, 'text_config', cfg) if cfg is not None else None
            try:
                cache = DynamicCache(config=text_cfg) if text_cfg is not None else DynamicCache()
            except Exception:
                cache = DynamicCache()
            self.kv_caches[comp_id] = cache
        return cache

    def _run_module_safe(self, module, hs, pos_emb, kwargs):
        """Execute a module with multi-stage signature handling and automatic dtype mismatch recovery."""
        # Ensure module and inputs are strictly on the active worker device (CUDA, Apple Silicon MPS, or CPU)
        module = _move_module_to_device(module, self.device)
        hs = hs.to(self.device)
        if pos_emb is not None:
            pos_emb = tuple(p.to(self.device) if isinstance(p, torch.Tensor) else p for p in pos_emb)

        clean_kwargs = {}
        for kw, val in kwargs.items():
            if kw == "position_embeddings" and pos_emb is not None:
                clean_kwargs[kw] = pos_emb
            elif isinstance(val, torch.Tensor):
                clean_kwargs[kw] = val.to(self.device)
            elif isinstance(val, tuple):
                clean_kwargs[kw] = tuple(v.to(self.device) if isinstance(v, torch.Tensor) else v for v in val)
            elif isinstance(val, list):
                clean_kwargs[kw] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in val]
            else:
                clean_kwargs[kw] = val
        if pos_emb is not None and "position_embeddings" not in clean_kwargs:
            clean_kwargs["position_embeddings"] = pos_emb

        def _try_call(curr_hs, curr_pos_emb, curr_kwargs):
            k = dict(curr_kwargs)
            if curr_pos_emb is not None:
                k["position_embeddings"] = curr_pos_emb
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
            return _try_call(hs, pos_emb, clean_kwargs)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if any(k in err_msg for k in ("dtype", "mat1", "m1 and m2", "same type", "half", "bfloat16", "float")):
                for dt in (torch.bfloat16, torch.float16, torch.float32):
                    if hs.dtype == dt:
                        continue
                    try:
                        fallback_hs = hs.to(dt)
                        fallback_pos_emb = tuple(p.to(dt) if isinstance(p, torch.Tensor) else p for p in pos_emb) if pos_emb is not None else None
                        return _try_call(fallback_hs, fallback_pos_emb, clean_kwargs)
                    except Exception:
                        continue
            raise

    def _cmd_forward_layer(self, msg):
        comp_id = msg["component_id"]
        module = self.components.get(comp_id)
        if module is None:
            return {"status": "error", "msg": f"Not found: {comp_id}"}

        if comp_id.startswith("layer_"):
            try:
                lidx = int(comp_id.split("_")[1])
                if hasattr(module, "self_attn") and hasattr(module.self_attn, "layer_idx"):
                    module.self_attn.layer_idx = lidx
                if hasattr(module, "linear_attn") and hasattr(module.linear_attn, "layer_idx"):
                    module.linear_attn.layer_idx = lidx
            except Exception:
                pass

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
        if hidden_out.ndim == 2:
            hidden_out = hidden_out.unsqueeze(0)
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

                if comp_id.startswith("layer_"):
                    try:
                        lidx = int(comp_id.split("_")[1])
                        if hasattr(module, "self_attn") and hasattr(module.self_attn, "layer_idx"):
                            module.self_attn.layer_idx = lidx
                        if hasattr(module, "linear_attn") and hasattr(module.linear_attn, "layer_idx"):
                            module.linear_attn.layer_idx = lidx
                    except Exception:
                        pass

                hs = _match_dtype(hs, module)
                pos_emb = _match_tuple_dtype(raw_pos_emb, module)
                cache = self._get_cache(comp_id)

                kwargs = dict(position_ids=pos_ids, past_key_values=cache, use_cache=True, attention_mask=None)
                if pos_emb is not None:
                    kwargs["position_embeddings"] = pos_emb

                outputs = self._run_module_safe(module, hs, pos_emb, kwargs)
                hs = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                if hs.ndim == 2:
                    hs = hs.unsqueeze(0)

                if not isinstance(outputs, torch.Tensor) and len(outputs) > 1 and outputs[1] is not None:
                    self.kv_caches[comp_id] = outputs[1]

                self.stats["forward_calls"] += 1

        return {"status": "ok", "hidden_states": hs.cpu() if isinstance(hs, torch.Tensor) else hs}

    def _cmd_pipeline_forward_chain(self, msg):
        """Execute one stage in a direct peer-to-peer ring pipeline and forward directly to the next peer node over LAN."""
        stages = msg.get("stages", [])
        stage_idx = msg.get("stage_idx", 0)

        if stage_idx >= len(stages):
            return {"status": "error", "msg": f"stage_idx {stage_idx} exceeds total stages count {len(stages)}"}

        curr_stage = stages[stage_idx]
        start_l = curr_stage["start_layer"]
        end_l = curr_stage["end_layer"]
        component_ids = [f"layer_{i}" for i in range(start_l, end_l + 1)]

        hs = msg["hidden_states"]
        pos_ids = msg.get("position_ids")
        raw_pos_emb = msg.get("position_embeddings")

        # 1. Execute current node's layers locally in RAM/GPU
        with torch.no_grad():
            for comp_id in component_ids:
                module = self.components.get(comp_id)
                if module is None:
                    return {"status": "error", "msg": f"Layer not found on worker {self.hostname}: {comp_id}"}

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

        # 2. If this is the last stage in the pipeline chain, return final hidden states to caller
        if stage_idx == len(stages) - 1:
            return {"status": "ok", "hidden_states": hs.cpu() if isinstance(hs, torch.Tensor) else hs}

        # 3. Forward DIRECTLY to the NEXT peer worker node over LAN (bypassing Master Laptop)
        next_stage = stages[stage_idx + 1]
        next_host = next_stage["node_host"]
        next_port = next_stage["node_port"]

        next_msg = {
            "cmd": "pipeline_forward_chain",
            "stages": stages,
            "stage_idx": stage_idx + 1,
            "hidden_states": hs.cpu() if isinstance(hs, torch.Tensor) else hs,
            "position_ids": pos_ids,
            "position_embeddings": raw_pos_emb,
        }

        try:
            peer_sock = self._get_peer_socket(next_host, next_port)
            send_msg(peer_sock, next_msg)
            resp, _ = recv_msg(peer_sock)
            return resp
        except Exception as e:
            # If peer connection failed, invalidate cached peer socket and retry once
            self.peer_sockets.pop(f"{next_host}:{next_port}", None)
            try:
                peer_sock = self._get_peer_socket(next_host, next_port)
                send_msg(peer_sock, next_msg)
                resp, _ = recv_msg(peer_sock)
                return resp
            except Exception as e2:
                return {"status": "error", "msg": f"P2P direct forward from {self.hostname} to {next_host}:{next_port} failed: {e2}"}

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

        module = _move_module_to_device(module, self.device)
        self.components[comp_id] = module

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

        module = _move_module_to_device(module, self.device)
        self.components[comp_id] = module

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

        module = _move_module_to_device(module, self.device)
        self.components[comp_id] = module

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

        module = _move_module_to_device(module, self.device)
        self.components[comp_id] = module

        hs = _match_dtype(msg["hidden_states"], module)
        if isinstance(hs, torch.Tensor) and hs.ndim == 3 and hs.shape[1] > 1:
            hs = hs[:, -1:, :]
        with torch.no_grad():
            logits = module(hs.to(self.device))

        self.stats["forward_calls"] += 1
        return {"status": "ok", "logits": logits.cpu() if isinstance(logits, torch.Tensor) else logits}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Worker node for distributed inference")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port for listening server (default: 9900)")
    parser.add_argument("--connect", "-c", type=str, default=None,
                        help="Connect outbound to Master coordinator (e.g. 192.168.1.50:9900) to bypass UFW/firewalls")
    parser.add_argument("--broadcast", type=str, default="255.255.255.255",
                        help="Broadcast address for UDP discovery (default: 255.255.255.255)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute accelerator device (auto, cpu, cuda, mps) (default: auto)")
    args = parser.parse_args()

    worker = Worker(port=args.port, device=args.device, connect_target=args.connect)
    worker.broadcast_addr = args.broadcast
    try:
        worker.start()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        worker.running = False
        sys.exit(0)


if __name__ == "__main__":
    main()
