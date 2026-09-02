#!/usr/bin/env python3
"""
Distributed Inference Controller — Interactive CLI.

Split a model across multiple machines, assign components to nodes,
and run distributed inference with stats tracking.

Usage:
    python3 distributed_inference.py
"""

import gc
import io
import json
import os
import pickle
import platform
import re
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import psutil
from huggingface_hub import snapshot_download, scan_cache_dir
from tabulate import tabulate

# Disable hf_transfer to prevent CAS client crashes on packet drops and enable resilient HTTP range resumption
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ═══════════════════════════════════════════════════════════════════════════════
# Constants & Colors
# ═══════════════════════════════════════════════════════════════════════════════
DISC_PORT = 9899
DEFAULT_PORT = 9900
MAGIC = b"LDIST_V1"
RECV_BUF = 1 << 16
CACHE_DIR = Path("/media/vithurshan/vithu/llm/openGP/LoadTime/model_cache")
DOWNLOAD_PATTERNS = ["*.safetensors", "*.json", "*.model", "*.tiktoken",
                     "*.txt", "tokenizer*"]

class C:
    B = "\033[1m"; D = "\033[2m"; G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    CY = "\033[96m"; MG = "\033[95m"; BL = "\033[94m"; W = "\033[97m"
    RS = "\033[0m"; H = "\033[1;96m"
    OK = "\033[92m✓\033[0m"; FL = "\033[91m✗\033[0m"; WN = "\033[93m⚠\033[0m"

MODELS = [
    {"id": "HuggingFaceTB/SmolLM2-135M", "name": "SmolLM2 135M", "type": "Dense", "gb": 0.27},
    {"id": "HuggingFaceTB/SmolLM2-360M", "name": "SmolLM2 360M", "type": "Dense", "gb": 0.72},
    {"id": "PrimeIntellect/qwen3-moe-tiny", "name": "Qwen3-MoE-Tiny 670M", "type": "MoE", "gb": 1.34},
    {"id": "Qwen/Qwen2.5-0.5B", "name": "Qwen2.5 0.5B", "type": "Dense", "gb": 0.99},
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "name": "TinyLlama 1.1B", "type": "Dense", "gb": 2.20},
    {"id": "Qwen/Qwen2.5-1.5B", "name": "Qwen2.5 1.5B", "type": "Dense", "gb": 3.09},
    {"id": "microsoft/phi-2", "name": "Phi-2 2.7B", "type": "Dense", "gb": 5.56},
    {"id": "Qwen/Qwen2.5-3B", "name": "Qwen2.5 3B", "type": "Dense", "gb": 6.39},
    {"id": "allenai/OLMoE-1B-7B-0924", "name": "OLMoE 1B/7B", "type": "MoE", "gb": 12.89},
    {"id": "Qwen/Qwen2.5-7B", "name": "Qwen2.5 7B", "type": "Dense", "gb": 15.23},
    {"id": "meta-llama/Llama-3.1-8B", "name": "Llama 3.1 8B", "type": "Dense", "gb": 16.07, "gated": True},
]


def fmt_bytes(n):
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024: return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} TB"

def fmt_time(s):
    if s < 1e-3: return f"{s*1e6:.0f}µs"
    if s < 1: return f"{s*1e3:.1f}ms"
    return f"{s:.3f}s"

def fmt_params(n):
    if n >= 1e9: return f"{n/1e9:.2f} B"
    if n >= 1e3: return f"{n/1e3:.2f} K"
    return str(n)


class AttentionRemoteAdapter(torch.nn.Module):
    """Universal callable wrapper for remote attention modules handling modern Transformers signatures."""
    def __init__(self, attn_module, rotary_emb_module=None):
        super().__init__()
        self.attn = attn_module
        self.rotary_emb = rotary_emb_module

    def forward(self, hidden_states, *args, **kwargs):
        pos_ids = kwargs.get("position_ids")
        pos_emb = kwargs.get("position_embeddings")
        past_kv = kwargs.get("past_key_values") or kwargs.get("past_key_value")
        mask = kwargs.get("attention_mask")

        if pos_emb is None and self.rotary_emb is not None and pos_ids is not None:
            try:
                pos_emb = self.rotary_emb(hidden_states, pos_ids)
            except Exception:
                pass

        try:
            return self.attn(hidden_states, position_embeddings=pos_emb, attention_mask=mask, past_key_value=past_kv, position_ids=pos_ids)
        except Exception:
            try:
                return self.attn(hidden_states, pos_emb, mask, past_kv, None, pos_ids)
            except Exception:
                return self.attn(hidden_states)


def compress_comp_list(comps):
    """Compress list of components into readable ranges (e.g. 'layers 0-23, attn 24-30')."""
    if not comps: return "(none)"
    import re

    groups = {
        "layer": [],
        "attn": [],
        "ffn": [],
        "ffn_only": [],
        "attn_only": [],
        "other": []
    }

    for c in comps:
        m = re.match(r"^layer_(\d+)$", c)
        if m:
            groups["layer"].append(int(m.group(1)))
            continue
        m = re.match(r"^attn_(\d+)$", c)
        if m:
            groups["attn"].append(int(m.group(1)))
            continue
        m = re.match(r"^ffn_(\d+)$", c)
        if m:
            groups["ffn"].append(int(m.group(1)))
            continue
        m = re.match(r"^layer_(\d+)\(ffn only\)$", c)
        if m:
            groups["ffn_only"].append(int(m.group(1)))
            continue
        m = re.match(r"^layer_(\d+)\(attn only\)$", c)
        if m:
            groups["attn_only"].append(int(m.group(1)))
            continue
        groups["other"].append(c)

    parts = []

    def format_runs(nums, prefix, suffix=""):
        if not nums: return []
        nums = sorted(nums)
        runs = []
        start = end = nums[0]
        for n in nums[1:]:
            if n == end + 1:
                end = n
            else:
                runs.append((start, end))
                start = end = n
        runs.append((start, end))
        res = []
        for s, e in runs:
            if s == e:
                res.append(f"{prefix}{s}{suffix}")
            else:
                res.append(f"{prefix}{s}-{e}{suffix}")
        return res

    parts.extend(format_runs(groups["layer"], "layers "))
    parts.extend(format_runs(groups["attn"], "attn "))
    parts.extend(format_runs(groups["ffn"], "ffn "))
    parts.extend(format_runs(groups["ffn_only"], "layers ", "(ffn only)"))
    parts.extend(format_runs(groups["attn_only"], "layers ", "(attn only)"))
    parts.extend(groups["other"])
    return ", ".join(parts)

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
    if module is None:
        return None
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

def hdr(t):
    print(f"\n{C.H}{'═'*58}\n  {t}\n{'═'*58}{C.RS}")

def shdr(t):
    print(f"\n  {C.CY}── {t} ──{C.RS}")

def ask(prompt, rng):
    while True:
        try:
            r = input(f"\n  {C.B}{prompt}{C.RS} ").strip()
            if r == "": return -1
            v = int(r)
            if v in rng: return v
            print(f"  {C.R}Enter {rng.start}-{rng.stop-1}{C.RS}")
        except ValueError: print(f"  {C.R}Number please{C.RS}")
        except (EOFError, KeyboardInterrupt): print(); return 0

def ask_int(prompt, default):
    try:
        r = input(f"  {prompt} [{default}]: ").strip()
        return int(r) if r else default
    except: return default

def ask_str(prompt):
    try: return input(f"  {prompt}").strip()
    except: return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Network Protocol (same as worker)
# ═══════════════════════════════════════════════════════════════════════════════
def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), RECV_BUF))
        if not chunk: raise ConnectionError("closed")
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
    raw = _recv_exact(sock, 8)
    length = struct.unpack("!Q", raw)[0]
    data = _recv_exact(sock, length)
    try:
        obj = Universal_Unpickler(io.BytesIO(data)).load()
    except Exception:
        obj = pickle.loads(data)
    return obj, 8 + length


# ═══════════════════════════════════════════════════════════════════════════════
# Node Connection
# ═══════════════════════════════════════════════════════════════════════════════
class NodeConnection:
    """Persistent TCP connection to a worker node with auto-reconnect."""

    def __init__(self, host, port, hostname="unknown", existing_sock=None, is_reverse=False):
        self.host = host
        self.port = port
        self.hostname = hostname
        self.sock = existing_sock
        self.is_reverse = is_reverse
        self.bytes_sent = 0
        self.bytes_received = 0
        self.label = f"{hostname} ({host}:{port} [reverse])" if is_reverse else f"{hostname} ({host}:{port})"
        self._max_retries = 3
        self._retry_delay = 2  # seconds

    def connect(self, timeout=None):
        if self.is_reverse:
            if self.is_connected():
                return
            raise ConnectionError(f"Reverse worker {self.label} is disconnected. Waiting for worker to reconnect...")
        self.close()  # close any stale socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # TCP keepalive for VPN connections that may drop
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.sock.settimeout(timeout if timeout is not None else 300)
        self.sock.connect((self.host, self.port))
        if timeout is not None:
            self.sock.settimeout(300)

    def _reconnect(self):
        """Attempt to reconnect to the worker node."""
        if self.is_reverse:
            return self.is_connected()
        for attempt in range(self._max_retries):
            try:
                self.connect()
                return True
            except Exception as e:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        return False

    def send_cmd(self, msg):
        """Send a command and return the response, with auto-reconnect."""
        last_error = None
        is_load = isinstance(msg, dict) and msg.get("cmd") == "load"
        for attempt in range(self._max_retries):
            try:
                if not self.is_connected():
                    self.connect()
                if self.sock:
                    self.sock.settimeout(600 if is_load else 180)
                sent = send_msg(self.sock, msg)
                self.bytes_sent += sent
                resp, recvd = recv_msg(self.sock)
                self.bytes_received += recvd
                if resp.get("status") == "error":
                    raise RuntimeError(f"Worker error: {resp.get('msg', '?')}")
                return resp
            except (ConnectionError, OSError, EOFError, BrokenPipeError) as e:
                last_error = e
                self.sock = None  # mark as disconnected
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        raise ConnectionError(
            f"Failed to reach {self.label} after {self._max_retries} attempts: {last_error}"
        )

    def close(self):
        if self.sock:
            try: self.sock.close()
            except: pass
            self.sock = None

    def is_connected(self):
        if self.sock is None:
            return False
        # Quick liveness check
        try:
            self.sock.getpeername()
            return True
        except:
            self.sock = None
            return False


class ReverseCoordinatorListener:
    """Background TCP listener on Master that accepts inbound reverse connections from workers (bypassing UFW)."""

    def __init__(self, port=DEFAULT_PORT, on_worker_connected=None):
        self.port = port
        self.on_worker_connected = on_worker_connected
        self.server_sock = None
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def _listen_loop(self):
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.server_sock.bind(("0.0.0.0", self.port))
            self.server_sock.listen(10)
        except Exception:
            self.running = False
            return

        while self.running:
            try:
                conn, addr = self.server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(target=self._handle_reverse_worker, args=(conn, addr), daemon=True).start()
            except Exception:
                break

    def _handle_reverse_worker(self, conn, addr):
        try:
            msg, _ = recv_msg(conn)
            if isinstance(msg, dict) and msg.get("cmd") == "register_reverse_worker":
                hostname = msg.get("hostname", f"lab-worker-{addr[0]}")
                send_msg(conn, {"status": "ok", "msg": f"Registered reverse worker {hostname}"})
                nc = NodeConnection(
                    host=addr[0],
                    port=addr[1],
                    hostname=hostname,
                    existing_sock=conn,
                    is_reverse=True
                )
                if self.on_worker_connected:
                    self.on_worker_connected(nc, msg)
            else:
                send_msg(conn, {"status": "error", "msg": "Expected register_reverse_worker handshake"})
                conn.close()
        except Exception:
            try: conn.close()
            except: pass

    def stop(self):
        self.running = False
        if self.server_sock:
            try: self.server_sock.close()
            except: pass


def discover_nodes(timeout=5.0):
    """Listen for UDP discovery broadcasts from workers."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    sock.bind(("", DISC_PORT))

    found = {}
    end_time = time.time() + timeout
    print(f"  Scanning for {timeout:.0f}s...", end="", flush=True)

    while time.time() < end_time:
        try:
            data, addr = sock.recvfrom(4096)
            if data.startswith(MAGIC):
                info = pickle.loads(data[len(MAGIC):])
                key = f"{addr[0]}:{info['port']}"
                if key not in found:
                    found[key] = {**info, "ip": addr[0]}
                    print(f"\n    Found: {info['hostname']} ({addr[0]}:{info['port']})", end="", flush=True)
        except socket.timeout:
            print(".", end="", flush=True)
    sock.close()
    print()
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# Distributed Model
# ═══════════════════════════════════════════════════════════════════════════════
class DistributedModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.model_id = None

        # Architecture info
        self.num_layers = 0
        self.num_experts = 1
        self.num_experts_per_tok = 1
        self.is_moe = False
        self.hidden_size = 0
        self.num_kv_heads = 0
        self.head_dim = 0
        self.dtype_size = 2

        # References to model sub-modules (set during load)
        self.embedding = None
        self.layers = []
        self.norm = None
        self.lm_head = None
        self.rotary_emb = None

        # Distribution assignments
        # comp_id → {"node": NodeConnection, "type": str} or None (local)
        self.assignments = {}
        self.is_distributed = False
        self.local_device = torch.device("cpu")

        # KV caches for local layers
        self.local_kv = {}   # layer_idx → (key, value) tensors
        self.total_past_len = 0

        # Stats
        self.net_bytes_sent = 0
        self.net_bytes_received = 0

    # ── Load Model ────────────────────────────────────────────────────────
    def load_model(self, model_id, model_dir, quant="none"):
        """Load the model with optional quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            BitsAndBytesConfig = None

        self.model_id = model_id
        self.quant = quant
        print(f"  {C.CY}Loading tokenizer...{C.RS}", end="", flush=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(CACHE_DIR))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"\r  {C.OK} Tokenizer loaded")

        config_path = model_dir / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        dtype_str = self.config.get("torch_dtype", "float16")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}.get(dtype_str, torch.float16)

        if quant == "4bit" and BitsAndBytesConfig is not None:
            print(f"  {C.CY}Loading model in 4-bit NF4 quantization (BitsAndBytes)...{C.RS}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                quantization_config=bnb_config,
                device_map="auto"
            )
        elif quant == "8bit" and BitsAndBytesConfig is not None:
            print(f"  {C.CY}Loading model in 8-bit quantization (BitsAndBytes)...{C.RS}")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif quant in ("int8", "int8-dynamic"):
            print(f"  {C.CY}Loading model and applying PyTorch dynamic int8 quantization...{C.RS}")
            self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32)
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            print(f"  {C.CY}Loading model to CPU ({dtype_str})...{C.RS}")
            self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype)
        self.model.eval()
        try:
            p = next(self.model.parameters())
            self.local_device = p.device
        except Exception:
            self.local_device = torch.device("cpu")

        size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**2)
        print(f"  {C.OK} Model loaded ({size_mb:.1f} MB in memory on {self.local_device})")

        # Extract references to sub-modules
        inner = self.model.model if hasattr(self.model, "model") else self.model.transformer
        self.embedding = inner.embed_tokens if hasattr(inner, "embed_tokens") else inner.wte
        self.layers = list(inner.layers if hasattr(inner, "layers") else inner.h)
        self.norm = inner.norm if hasattr(inner, "norm") else inner.ln_f
        self.lm_head = self.model.lm_head
        self.rotary_emb = getattr(inner, "rotary_emb", None)

        # Architecture info
        self.num_layers = len(self.layers)
        self.hidden_size = self.config.get("hidden_size", self.config.get("n_embd", 0))
        nh = self.config.get("num_attention_heads", self.config.get("n_head", 1))
        self.num_kv_heads = self.config.get("num_key_value_heads", nh)
        self.head_dim = self.hidden_size // nh if nh else 0
        self.dtype_size = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype_str, 2)
        self.num_experts = self.config.get("num_experts",
                                           self.config.get("num_local_experts", 1))
        self.num_experts_per_tok = self.config.get("num_experts_per_tok", 1)
        self.is_moe = self.num_experts > 1

        # Initialize assignments — everything local
        self.assignments = {}
        self.is_distributed = False

    # ── Model Inspection ──────────────────────────────────────────────────
    def inspect_model(self):
        """Display an extensive parameter and size breakdown of the loaded model."""
        if self.model is None:
            print(f"\n  {C.Y}No model loaded yet. Load a model first.{C.RS}")
            return

        hdr(f"MODEL INSPECTION: {self.model_id}")

        total_params = sum(p.numel() for p in self.model.parameters())
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        dtype_str = getattr(next(self.model.parameters()), "dtype", "unknown")
        elem_size = getattr(next(self.model.parameters()), "element_size", lambda: self.dtype_size)()

        # Summary Info Card
        shdr("ARCHITECTURE OVERVIEW")
        moe_desc = f"MoE ({self.num_experts} experts, top-{self.num_experts_per_tok} active/tok)" if self.is_moe else "Dense Transformer"
        overview_rows = [
            ["Model Identifier", self.model_id],
            ["Architecture Type", moe_desc],
            ["Precision / Dtype", f"{dtype_str} ({elem_size} bytes/param)"],
            ["Total Parameters", f"{total_params:,} ({fmt_params(total_params)})"],
            ["Total Model Memory", f"{C.G}{fmt_bytes(total_bytes)}{C.RS}"],
            ["", ""],
            ["Hidden Dimension (d_model)", f"{self.hidden_size:,}"],
            ["Transformer Layers", f"{self.num_layers}"],
            ["Attention Heads (Q / KV)", f"{self.config.get('num_attention_heads', '?')} Q / {self.num_kv_heads} KV (Head dim: {self.head_dim})"],
            ["Vocabulary Size", f"{self.config.get('vocab_size', '?'):,}"],
        ]
        print(tabulate(overview_rows, tablefmt="rounded_outline", colalign=("left", "right")))

        # Component Breakdown
        shdr("TOP-LEVEL COMPONENT BREAKDOWN")
        emb_p = sum(p.numel() for p in self.embedding.parameters()) if self.embedding else 0
        emb_b = sum(p.numel() * p.element_size() for p in self.embedding.parameters()) if self.embedding else 0

        layers_p = sum(sum(p.numel() for p in l.parameters()) for l in self.layers)
        layers_b = sum(sum(p.numel() * p.element_size() for p in l.parameters()) for l in self.layers)

        norm_p = sum(p.numel() for p in self.norm.parameters()) if self.norm else 0
        norm_b = sum(p.numel() * p.element_size() for p in self.norm.parameters()) if self.norm else 0

        head_p = sum(p.numel() for p in self.lm_head.parameters()) if self.lm_head else 0
        head_b = sum(p.numel() * p.element_size() for p in self.lm_head.parameters()) if self.lm_head else 0
        is_tied = (self.lm_head.weight.data_ptr() == self.embedding.weight.data_ptr()) if (hasattr(self.lm_head, "weight") and hasattr(self.embedding, "weight")) else False

        comp_rows = [
            ["Token Embedding (embed_tokens)", f"{emb_p:,}", fmt_params(emb_p), f"{emb_p/total_params*100:.1f}%", fmt_bytes(emb_b)],
            [f"Transformer Layers ({self.num_layers} layers)", f"{layers_p:,}", fmt_params(layers_p), f"{layers_p/total_params*100:.1f}%", fmt_bytes(layers_b)],
            ["Final LayerNorm (norm)", f"{norm_p:,}", fmt_params(norm_p), f"{norm_p/total_params*100:.2f}%", fmt_bytes(norm_b)],
            ["LM Head (lm_head)" + (" [tied with embed]" if is_tied else ""), f"{head_p:,}", fmt_params(head_p), f"{head_p/total_params*100:.1f}%", fmt_bytes(head_b)],
            ["", "", "", "", ""],
            [f"{C.B}TOTAL MODEL{C.RS}", f"{C.B}{total_params:,}{C.RS}", f"{C.B}{fmt_params(total_params)}{C.RS}", "100.0%", f"{C.G}{fmt_bytes(total_bytes)}{C.RS}"],
        ]
        print(tabulate(comp_rows, headers=["Component", "Param Count", "Params", "Share", "Size"], tablefmt="rounded_outline", colalign=("left", "right", "right", "right", "right")))

        # Single Layer Anatomy
        shdr("LAYER ANATOMY & SUB-COMPONENTS (Single Layer Breakdown)")
        sample_layer = self.layers[1] if (len(self.layers) > 1 and self.is_moe) else self.layers[0]
        sub_rows = []

        # Attention sub-block
        attn = sample_layer.self_attn
        attn_p = sum(p.numel() for p in attn.parameters())
        attn_b = sum(p.numel() * p.element_size() for p in attn.parameters())
        for name, p in attn.named_parameters():
            sub_rows.append([f"  • Attention: {name}", list(p.shape), f"{p.numel():,}", fmt_params(p.numel()), fmt_bytes(p.numel() * p.element_size())])
        sub_rows.append([f"{C.CY}► Total Self-Attention{C.RS}", "", f"{attn_p:,}", fmt_params(attn_p), fmt_bytes(attn_b)])
        sub_rows.append(["", "", "", "", ""])

        # FFN / MoE sub-block
        mlp = sample_layer.mlp
        mlp_p = sum(p.numel() for p in mlp.parameters())
        mlp_b = sum(p.numel() * p.element_size() for p in mlp.parameters())
        for name, p in mlp.named_parameters():
            sub_rows.append([f"  • MLP/MoE: {name}", list(p.shape), f"{p.numel():,}", fmt_params(p.numel()), fmt_bytes(p.numel() * p.element_size())])
        sub_rows.append([f"{C.CY}► Total FFN/MoE Block{C.RS}", "", f"{mlp_p:,}", fmt_params(mlp_p), fmt_bytes(mlp_b)])
        sub_rows.append(["", "", "", "", ""])

        # Norms
        in_norm_p = sum(p.numel() for p in sample_layer.input_layernorm.parameters())
        in_norm_b = sum(p.numel() * p.element_size() for p in sample_layer.input_layernorm.parameters())
        post_norm_p = sum(p.numel() for p in sample_layer.post_attention_layernorm.parameters())
        post_norm_b = sum(p.numel() * p.element_size() for p in sample_layer.post_attention_layernorm.parameters())
        sub_rows.append(["  • input_layernorm", list(sample_layer.input_layernorm.weight.shape), f"{in_norm_p:,}", fmt_params(in_norm_p), fmt_bytes(in_norm_b)])
        sub_rows.append(["  • post_attention_layernorm", list(sample_layer.post_attention_layernorm.weight.shape), f"{post_norm_p:,}", fmt_params(post_norm_p), fmt_bytes(post_norm_b)])

        layer_tot_p = sum(p.numel() for p in sample_layer.parameters())
        layer_tot_b = sum(p.numel() * p.element_size() for p in sample_layer.parameters())
        sub_rows.append(["", "", "", "", ""])
        sub_rows.append([f"{C.B}► TOTAL PER LAYER{C.RS}", "", f"{C.B}{layer_tot_p:,}{C.RS}", f"{C.B}{fmt_params(layer_tot_p)}{C.RS}", f"{C.G}{fmt_bytes(layer_tot_b)}{C.RS}"])

        print(tabulate(sub_rows, headers=["Sub-Component", "Weight Shape", "Param Count", "Params", "Size"], tablefmt="rounded_outline", colalign=("left", "left", "right", "right", "right")))

        # Layer Sizing & Distribution Matrix
        shdr("LAYER SIZING & ALLOCATION MATRIX")
        layer_rows = []
        cum_bytes = emb_b
        for idx, lyr in enumerate(self.layers):
            l_p = sum(p.numel() for p in lyr.parameters())
            l_b = sum(p.numel() * p.element_size() for p in lyr.parameters())
            cum_bytes += l_b
            
            # Check assignment
            lid = f"layer_{idx}"
            if lid in self.assignments:
                target = f"{C.Y}{self.assignments[lid]['node'].label}{C.RS}"
            else:
                target = f"{C.G}Local GPU/CPU{C.RS}"

            l_type = "MoE" if (hasattr(lyr, "mlp") and (hasattr(lyr.mlp, "experts") or "Moe" in type(lyr.mlp).__name__)) else "Dense"
            layer_rows.append([f"Layer {idx}", l_type, fmt_params(l_p), fmt_bytes(l_b), fmt_bytes(cum_bytes), target])

        print(tabulate(layer_rows, headers=["Layer", "Type", "Params", "Layer Size", "Cumulative (incl. Emb)", "Target Allocation"], tablefmt="rounded_outline", colalign=("left", "center", "right", "right", "right", "left")))

    # ── Component Assignment ──────────────────────────────────────────────
    def assign_to_remote(self, comp_id, comp_type, node, layer_idx=None, expert_idx=None):
        """Mark a component for remote execution."""
        self.assignments[comp_id] = {
            "node": node, "type": comp_type,
            "layer_idx": layer_idx, "expert_idx": expert_idx,
        }

    def check_worker_memory_admission(self, node, assigned_components=None, max_sequence_len=512, gate_ratio=0.75):
        """
        Pre-flight Predictive Memory Admission Gate:
        Checks whether expected layer weights + peak KV cache + activation buffer footprint
        exceeds 75% of the worker's FULL (total) RAM capacity.

        - If Total Required > 75% of Worker Total RAM -> Gate fails (SKIP).
        - If Total Required <= 75% of Worker Total RAM -> Gate passes (ADMIT).
          (If a runtime physical OOM occurs, it will be caught and recorded as failed).

        Returns (can_admit: bool, required_bytes: int, total_ram_bytes: int, reason: str)
        """
        if assigned_components is None:
            assigned_components = [
                (v["type"], v.get("layer_idx", 0))
                for v in self.assignments.values() if v.get("node") == node
            ]

        # 1. Query full (total) RAM from worker or default to 8.0 GB
        total_ram = 8.0 * (1024**3)
        try:
            info = node.send_cmd({"cmd": "info"})
            if "ram_total" in info and info["ram_total"] > 0:
                total_ram = info["ram_total"]
        except Exception:
            pass

        # 2. Calculate layer weight memory
        weight_bytes = 0
        kv_layers = 0
        for comp_type, l_idx in assigned_components:
            if l_idx is not None and l_idx < len(self.layers):
                lyr = self.layers[l_idx]
                if comp_type == "layer":
                    weight_bytes += sum(p.numel() * self.dtype_size for p in lyr.parameters())
                    kv_layers += 1
                elif comp_type == "attention" and hasattr(lyr, "self_attn"):
                    weight_bytes += sum(p.numel() * self.dtype_size for p in lyr.self_attn.parameters())
                    kv_layers += 1
                elif comp_type == "ffn" and hasattr(lyr, "mlp"):
                    weight_bytes += sum(p.numel() * self.dtype_size for p in lyr.mlp.parameters())

        # 3. Calculate peak theoretical KV cache
        num_kv_heads = getattr(getattr(self, "model", None).config, "num_key_value_heads", getattr(getattr(self, "model", None).config, "num_attention_heads", 16)) if hasattr(self, "model") and self.model else 16
        head_dim = getattr(getattr(self, "model", None).config, "head_dim", 64) if hasattr(self, "model") and self.model else 64
        kv_cache_bytes = 2 * kv_layers * num_kv_heads * head_dim * self.dtype_size * max_sequence_len

        # 4. Activation buffer overhead
        hidden_size = getattr(getattr(self, "model", None).config, "hidden_size", 2048) if hasattr(self, "model") and self.model else 2048
        act_overhead = 2 * max_sequence_len * hidden_size * self.dtype_size + (400 * 1024 * 1024)

        total_required = weight_bytes + kv_cache_bytes + act_overhead
        gate_limit = total_ram * gate_ratio

        if total_required > gate_limit:
            reason = (
                f"Predicted memory ({total_required / (1024**3):.2f} GB = "
                f"{weight_bytes/(1024**3):.2f}GB weights + {kv_cache_bytes/(1024**2):.1f}MB KV + {act_overhead/(1024**2):.1f}MB buffer) "
                f"exceeds 75% full RAM gate ({gate_limit / (1024**3):.2f} GB of {total_ram / (1024**3):.2f} GB total RAM)"
            )
            return False, total_required, total_ram, reason

        return True, total_required, total_ram, "Admitted safely within 75% full RAM gate"

    def unassign(self, comp_id):
        self.assignments.pop(comp_id, None)

    def export_assignments(self):
        """Export current distribution assignments to a JSON-serializable list."""
        exported = []
        for comp_id, info in self.assignments.items():
            node = info["node"]
            exported.append({
                "comp_id": comp_id,
                "type": info["type"],
                "layer_idx": info.get("layer_idx"),
                "expert_idx": info.get("expert_idx"),
                "node_host": node.host,
                "node_port": node.port,
                "node_hostname": getattr(node, "hostname", "unknown"),
                "node_label": node.label,
            })
        return exported

    def import_assignments(self, cached_entries, node_matcher_fn):
        """Import assignments from cache using a node matcher callback."""
        self.assignments.clear()
        restored_count = 0
        for entry in cached_entries:
            node = node_matcher_fn(entry)
            if node is not None:
                self.assign_to_remote(
                    entry["comp_id"],
                    entry["type"],
                    node,
                    layer_idx=entry.get("layer_idx"),
                    expert_idx=entry.get("expert_idx")
                )
                restored_count += 1
        return restored_count

    def get_assignment_summary(self):
        """Return a dict: node_label → list of component descriptions."""
        summary = defaultdict(list)
        assigned_ids = set(self.assignments.keys())

        # Remote assignments
        for cid, info in self.assignments.items():
            label = info["node"].label
            summary[label].append(cid)

        # Local components
        local = []
        if "embedding" not in assigned_ids:
            local.append("embedding")
        if "lm_head" not in assigned_ids:
            local.append("lm_head + norm")

        for i in range(self.num_layers):
            lid = f"layer_{i}"
            aid = f"attn_{i}"
            fid = f"ffn_{i}"
            if lid in assigned_ids:
                continue  # whole layer remote
            if aid in assigned_ids and fid in assigned_ids:
                continue  # both parts remote
            if aid in assigned_ids:
                local.append(f"layer_{i}(ffn only)")
            elif fid in assigned_ids:
                local.append(f"layer_{i}(attn only)")
            elif any(f"expert_{i}_{e}" in assigned_ids for e in range(self.num_experts)):
                remote_experts = [e for e in range(self.num_experts) if f"expert_{i}_{e}" in assigned_ids]
                local_experts = self.num_experts - len(remote_experts)
                local.append(f"layer_{i}(attn+router+{local_experts}experts)")
            else:
                local.append(f"layer_{i}")

        my_hostname = platform.node()
        gpu = ""
        if torch.cuda.is_available():
            gpu = f" + {torch.cuda.get_device_name(0)}"
        summary[f"Local: {my_hostname}{gpu}"] = local
        return dict(summary)

    def start_profiling(self):
        """Start memory profiling on master and all assigned workers."""
        self.profiler_running = True
        self.memory_log = []
        
        # Start remote profilers
        notified_nodes = set()
        for info in self.assignments.values():
            node = info["node"]
            if id(node) not in notified_nodes:
                try:
                    node.send_cmd({"cmd": "start_profiling"})
                    notified_nodes.add(id(node))
                except Exception as e:
                    print(f"  ⚠ Failed to start profiling on {node.label}: {e}")
                    
        # Start local profiler
        def _profiler():
            import os
            import psutil
            proc = psutil.Process(os.getpid())
            start_time = time.time()
            while self.profiler_running:
                rss = proc.memory_info().rss / (1024**2)
                vram = 0
                if torch.cuda.is_available():
                    vram = torch.cuda.memory_allocated() / (1024**2)
                elif hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                    try: vram = torch.mps.current_allocated_memory() / (1024**2)
                    except: pass
                self.memory_log.append((time.time() - start_time, rss, vram))
                time.sleep(0.1)
        import threading
        self.profiler_thread = threading.Thread(target=_profiler, daemon=True)
        self.profiler_thread.start()
        
    def stop_profiling(self):
        """Stop memory profiling, save local graph, and stop remote profilers."""
        self.profiler_running = False
        if hasattr(self, "profiler_thread") and self.profiler_thread:
            self.profiler_thread.join(timeout=1.0)
            self.profiler_thread = None
            
        # Plot local memory
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import platform
            
            plt.figure(figsize=(10, 5))
            times = [x[0] for x in self.memory_log]
            rss = [x[1] for x in self.memory_log]
            vram = [x[2] for x in self.memory_log]
            
            plt.plot(times, rss, label="RAM (RSS) MB", color='blue')
            if any(v > 0 for v in vram):
                plt.plot(times, vram, label="VRAM Allocated MB", color='green')
            
            plt.title(f"Master Memory Consumption - {platform.node()}")
            plt.xlabel("Time (s)")
            plt.ylabel("Memory (MB)")
            plt.legend()
            plt.grid(True)
            
            out_name = f"master_memory_profile_{int(time.time())}.png"
            from pathlib import Path
            out_path = Path.cwd() / out_name
            plt.savefig(out_path)
            plt.close()
            print(f"  📊 Master memory profile saved to {out_path}")
        except Exception as e:
            print(f"  ⚠ Failed to plot master memory profile: {e}")
            
        # Stop remote profilers
        notified_nodes = set()
        for info in self.assignments.values():
            node = info["node"]
            if id(node) not in notified_nodes:
                try:
                    resp = node.send_cmd({"cmd": "stop_profiling"})
                    if resp.get("status") == "ok":
                        print(f"  📊 Worker {node.label} profile saved to {resp.get('saved_path')}")
                    notified_nodes.add(id(node))
                except Exception as e:
                    print(f"  ⚠ Failed to stop profiling on {node.label}: {e}")

    # ── Apply Distribution ────────────────────────────────────────────────
    def apply_distribution(self):
        """Send assigned components to remote workers, move local to GPU."""
        if not self.assignments:
            print(f"  {C.Y}No remote assignments. Everything stays local.{C.RS}")

        # Query remote workers for already-cached components
        node_loaded_comps = {}
        for info in self.assignments.values():
            node = info["node"]
            if id(node) not in node_loaded_comps:
                try:
                    if not node.is_connected():
                        node.connect()
                    chk = node.send_cmd({"cmd": "check_components", "model_id": self.model_id})
                    node_loaded_comps[id(node)] = set(chk.get("loaded", []))
                except Exception:
                    node_loaded_comps[id(node)] = set()

        if not hasattr(self, "sent_components"):
            self.sent_components = set()

        # Send components to remote workers
        all_success = True
        for comp_id, info in self.assignments.items():
            node = info["node"]
            comp_type = info["type"]
            layer_idx = info.get("layer_idx")
            expert_idx = info.get("expert_idx")

            # Check if worker already has it in memory or auto-loaded from disk cache
            if comp_id in node_loaded_comps.get(id(node), set()):
                print(f"  {C.OK} {comp_id} already cached on {node.label} for {self.model_id}, skipping transfer!")
                continue

            # Extract the sub-module
            module = self._extract_module(comp_type, layer_idx, expert_idx)
            if module is None:
                print(f"  {C.FL} Cannot extract: {comp_id}")
                all_success = False
                continue

            # Remap layer_idx for KV cache (always use 0 on worker)
            if comp_type in ("layer", "attention") and hasattr(module, "self_attn"):
                attn = module.self_attn if comp_type == "layer" else module
                if hasattr(attn, "layer_idx"):
                    attn.layer_idx = 0

            # Send to worker
            print(f"  {C.CY}Sending {comp_id} to {node.label}...{C.RS}", end="", flush=True)
            try:
                if not node.is_connected():
                    node.connect()
                t0 = time.perf_counter()
                resp = node.send_cmd({
                    "cmd": "load",
                    "component_id": comp_id,
                    "component_type": comp_type,
                    "module": module,
                    "model_id": self.model_id,
                })
                elapsed = time.perf_counter() - t0
                size = resp.get("size_bytes", 0)
                print(f"\r  {C.OK} Sent {comp_id} ({fmt_bytes(size)}) in {fmt_time(elapsed)}")
                node_loaded_comps.setdefault(id(node), set()).add(comp_id)
            except Exception as e:
                print(f"\r  {C.FL} Failed to send {comp_id}: {e}")
                all_success = False
                continue

        if not all_success:
            print(f"\n  {C.FL} Distribution incomplete: one or more components failed to transfer. Please retry Apply & Load.{C.RS}")
            self.is_distributed = False
            return False

        gc.collect()

        # Dynamically calculate remaining local component sizes and offload to GPU VRAM if they fit
        local_modules = []
        if "embedding" not in self.assignments and self.embedding is not None:
            local_modules.append(("embedding", self.embedding))
        if "lm_head" not in self.assignments:
            if self.norm is not None:
                local_modules.append(("norm", self.norm))
            if self.lm_head is not None:
                local_modules.append(("lm_head", self.lm_head))

        for i in range(self.num_layers):
            lid = f"layer_{i}"
            aid = f"attn_{i}"
            fid = f"ffn_{i}"
            if lid in self.assignments:
                continue
            layer = self.layers[i]
            if aid in self.assignments and hasattr(layer, "mlp"):
                local_modules.append((f"layer_{i}.mlp", layer.mlp))
                if hasattr(layer, "input_layernorm"):
                    local_modules.append((f"layer_{i}.input_layernorm", layer.input_layernorm))
                if hasattr(layer, "post_attention_layernorm"):
                    local_modules.append((f"layer_{i}.post_attention_layernorm", layer.post_attention_layernorm))
            elif fid in self.assignments and hasattr(layer, "self_attn"):
                local_modules.append((f"layer_{i}.self_attn", layer.self_attn))
                if hasattr(layer, "input_layernorm"):
                    local_modules.append((f"layer_{i}.input_layernorm", layer.input_layernorm))
                if hasattr(layer, "post_attention_layernorm"):
                    local_modules.append((f"layer_{i}.post_attention_layernorm", layer.post_attention_layernorm))
            else:
                local_modules.append((f"layer_{i}", layer))

        local_bytes = sum(
            sum(p.numel() * p.element_size() for p in mod.parameters()) +
            sum(b.numel() * b.element_size() for b in mod.buffers())
            for _, mod in local_modules
        )

        has_gpu = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        target_gpu_dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))

        if has_gpu and target_gpu_dev.type != "cpu":
            try:
                if target_gpu_dev.type == "cuda":
                    torch.cuda.empty_cache()
                    free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    # Reserve 200 MB (or 4% of free VRAM) for KV cache & activations headroom
                    safety_headroom = max(150 * 1024 * 1024, min(300 * 1024 * 1024, int(free_vram * 0.04)))
                    safe_vram_limit = max(0, free_vram - safety_headroom)
                else:
                    # Apple Silicon MPS has unified memory
                    free_vram = psutil.virtual_memory().available
                    safety_headroom = 200 * 1024 * 1024
                    safe_vram_limit = free_vram

                if local_bytes <= safe_vram_limit:
                    print(f"  {C.CY}Offloading all {len(local_modules)} local component(s) ({fmt_bytes(local_bytes)}) to {target_gpu_dev}...{C.RS}", end="", flush=True)
                    for name, mod in local_modules:
                        mod.to(target_gpu_dev)
                    if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
                        try: self.rotary_emb.to(target_gpu_dev)
                        except Exception: pass
                    self.local_device = target_gpu_dev
                    print(f"\r  {C.OK} All local components ({fmt_bytes(local_bytes)}) accelerated on {target_gpu_dev} ({fmt_bytes(free_vram)} VRAM total, {fmt_bytes(safety_headroom)} headroom reserved)")
                else:
                    # Partial Greedy GPU Offloading: Fit as many local components on GPU as possible!
                    print(f"  {C.CY}Packing local components into GPU VRAM ({fmt_bytes(safe_vram_limit)} budget)...{C.RS}", end="", flush=True)
                    gpu_bytes = 0
                    gpu_mods = []
                    cpu_mods = []
                    for name, mod in local_modules:
                        mod_bytes = sum(p.numel() * p.element_size() for p in mod.parameters()) + sum(b.numel() * b.element_size() for b in mod.buffers())
                        if gpu_bytes + mod_bytes <= safe_vram_limit:
                            mod.to(target_gpu_dev)
                            gpu_bytes += mod_bytes
                            gpu_mods.append(name)
                        else:
                            mod.to("cpu")
                            cpu_mods.append(name)

                    if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
                        try: self.rotary_emb.to(target_gpu_dev if gpu_mods else "cpu")
                        except Exception: pass

                    self.local_device = target_gpu_dev if gpu_mods else torch.device("cpu")
                    print(f"\r  {C.OK} Hybrid Local Placement: {len(gpu_mods)} component(s) ({fmt_bytes(gpu_bytes)}) on {target_gpu_dev} | {len(cpu_mods)} on CPU")
            except Exception as e:
                print(f"\r  {C.WN} GPU placement warning ({e}). Falling back to CPU.")
                for name, mod in local_modules:
                    try: mod.to("cpu")
                    except Exception: pass
                self.local_device = torch.device("cpu")
        else:
            self.local_device = torch.device("cpu")

        self.is_distributed = True
        print(f"  {C.OK} Distribution applied!")
        return True

    def _extract_module(self, comp_type, layer_idx, expert_idx):
        """Extract a sub-module from the model (returns a copy on CPU)."""
        import copy
        try:
            if comp_type == "embedding":
                mod = copy.deepcopy(self.embedding)
            elif comp_type == "lm_head":
                # Bundle norm + lm_head together
                class NormAndHead(torch.nn.Module):
                    def __init__(self, norm, head):
                        super().__init__()
                        self.norm = norm
                        self.head = head
                    def forward(self, x):
                        return self.head(self.norm(x))
                mod = NormAndHead(
                    copy.deepcopy(self.norm),
                    copy.deepcopy(self.lm_head)
                )
            elif comp_type == "layer":
                mod = copy.deepcopy(self.layers[layer_idx])
            elif comp_type == "attention":
                layer = self.layers[layer_idx]
                mod = copy.deepcopy(layer.self_attn)
            elif comp_type == "ffn":
                layer = self.layers[layer_idx]
                mod = copy.deepcopy(layer.mlp)
            elif comp_type == "expert":
                layer = self.layers[layer_idx]
                mod = copy.deepcopy(layer.mlp.experts[expert_idx])
            else:
                mod = None

            if mod is not None:
                mod = mod.to("cpu")
                for p in mod.parameters():
                    p.data = p.data.cpu()
                for b in mod.buffers():
                    b.data = b.data.cpu()
            return mod
        except Exception as e:
            print(f"  {C.R}Extract error: {e}{C.RS}")
        return None

    def _remove_local_module(self, comp_type, layer_idx, expert_idx):
        """Keep local layers intact in memory to allow dynamic re-splitting and clearing assignments without reloading."""
        pass

    # ── Distributed Forward Pass ──────────────────────────────────────────
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=128, temperature=0.0):
        """Run distributed autoregressive generation."""
        device = self.local_device

        # Verify that all assigned remote components are currently loaded on workers (auto-heal if worker restarted)
        for cid, info in self.assignments.items():
            node = info["node"]
            try:
                if not node.is_connected():
                    node.connect()
                chk = node.send_cmd({"cmd": "check_components", "model_id": self.model_id})
                loaded_set = set(chk.get("loaded", []))
                if cid not in loaded_set:
                    comp_type = info["type"]
                    layer_idx = info.get("layer_idx")
                    expert_idx = info.get("expert_idx")
                    module = self._extract_module(comp_type, layer_idx, expert_idx)
                    if module is not None:
                        if comp_type in ("layer", "attention") and hasattr(module, "self_attn"):
                            attn = module.self_attn if comp_type == "layer" else module
                            if hasattr(attn, "layer_idx"):
                                attn.layer_idx = 0
                        print(f"  {C.CY}Auto-loading missing {cid} to {node.label}...{C.RS}", end="", flush=True)
                        node.send_cmd({
                            "cmd": "load",
                            "component_id": cid,
                            "component_type": comp_type,
                            "module": module,
                            "model_id": self.model_id,
                        })
                        print(f"\r  {C.OK} Auto-loaded {cid} to {node.label}           ")
            except Exception as e:
                print(f"  {C.WN} Warning: could not verify {cid} on {node.label}: {e}")

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        num_input = input_ids.shape[1]

        # Reset state
        self.local_kv = {}
        self.total_past_len = 0
        self.net_bytes_sent = 0
        self.net_bytes_received = 0
        # Clear remote KV caches
        for info in self.assignments.values():
            try:
                info["node"].send_cmd({"cmd": "clear_kv"})
            except Exception:
                pass

        # Track stats
        token_times = []
        generated_ids = []

        # Reset byte counters on nodes
        for info in self.assignments.values():
            info["node"].bytes_sent = 0
            info["node"].bytes_received = 0

        print(f"\n  {C.D}Input: {num_input} tokens{C.RS}")
        print(f"  {C.B}Response:{C.RS} {C.G}", end="", flush=True)

        # ── Prefill ───────────────────────────────────────────────────────
        vram_before = torch.cuda.memory_allocated() if device.type == "cuda" else 0
        t_prefill_start = time.perf_counter()

        logits = self._forward_pass(input_ids, is_prefill=True)
        self.total_past_len = num_input

        if device.type == "cuda":
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t_prefill_start

        # Sample first token
        first_token = self._sample(logits, temperature)
        generated_ids.append(first_token.item())
        tok_str = self.tokenizer.decode([first_token.item()], skip_special_tokens=True)
        print(tok_str, end="", flush=True)

        # ── Decode Loop ───────────────────────────────────────────────────
        next_token = first_token
        for step in range(max_new_tokens - 1):
            t0 = time.perf_counter()

            tok_input = next_token.view(1, 1).to(device)
            logits = self._forward_pass(tok_input, is_prefill=False)
            self.total_past_len += 1

            if device.type == "cuda":
                torch.cuda.synchronize()

            next_token = self._sample(logits, temperature)
            token_times.append(time.perf_counter() - t0)

            tok_id = next_token.item()
            generated_ids.append(tok_id)
            tok_str = self.tokenizer.decode([tok_id], skip_special_tokens=True)
            print(tok_str, end="", flush=True)

            if tok_id == self.tokenizer.eos_token_id:
                break

        print(f"{C.RS}\n")
        vram_after = torch.cuda.memory_allocated() if device.type == "cuda" else 0

        # ── Collect stats ─────────────────────────────────────────────────
        num_output = len(generated_ids)
        decode_time = sum(token_times)
        total_time = ttft + decode_time

        # Network stats
        # Deduplicate nodes to avoid counting same node's bytes multiple times
        unique_nodes = {id(info["node"]): info["node"] for info in self.assignments.values()}
        net_sent = sum(n.bytes_sent for n in unique_nodes.values())
        net_recv = sum(n.bytes_received for n in unique_nodes.values())

        # KV cache estimation
        total_seq = num_input + num_output
        kv_per_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_size
        kv_total = kv_per_token * total_seq

        # Collect per-node memory stats (layers + active KV cache + process footprint)
        node_mem_stats = {}

        # 1. Local Master Node
        local_layer_bytes = 0
        assigned_ids = set(self.assignments.keys())
        if "embedding" not in assigned_ids and self.embedding is not None:
            local_layer_bytes += sum(p.numel() * p.element_size() for p in self.embedding.parameters())
        if "lm_head" not in assigned_ids and self.lm_head is not None:
            local_layer_bytes += sum(p.numel() * p.element_size() for p in self.lm_head.parameters())
        if "lm_head" not in assigned_ids and self.norm is not None:
            local_layer_bytes += sum(p.numel() * p.element_size() for p in self.norm.parameters())

        for idx, layer in enumerate(self.layers):
            if layer is None:
                continue
            lid = f"layer_{idx}"
            aid = f"attn_{idx}"
            fid = f"ffn_{idx}"
            if lid in assigned_ids or (aid in assigned_ids and fid in assigned_ids):
                continue
            if aid in assigned_ids:
                if hasattr(layer, "mlp") and layer.mlp is not None:
                    local_layer_bytes += sum(p.numel() * p.element_size() for p in layer.mlp.parameters())
                if hasattr(layer, "post_attention_layernorm") and layer.post_attention_layernorm is not None:
                    local_layer_bytes += sum(p.numel() * p.element_size() for p in layer.post_attention_layernorm.parameters())
            elif fid in assigned_ids:
                if hasattr(layer, "self_attn") and layer.self_attn is not None:
                    local_layer_bytes += sum(p.numel() * p.element_size() for p in layer.self_attn.parameters())
                if hasattr(layer, "input_layernorm") and layer.input_layernorm is not None:
                    local_layer_bytes += sum(p.numel() * p.element_size() for p in layer.input_layernorm.parameters())
            else:
                local_layer_bytes += sum(p.numel() * p.element_size() for p in layer.parameters())

        local_kv_bytes = sum(get_kv_cache_size_bytes(c) for c in self.local_kv.values())
        my_hostname = platform.node()
        gpu_name = f" + {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""
        local_label = f"Local: {my_hostname}{gpu_name}"

        node_mem_stats[local_label] = {
            "is_local": True,
            "layer_bytes": local_layer_bytes,
            "kv_bytes": local_kv_bytes,
            "vram_delta": max(vram_after - vram_before, 0),
            "vram_after": vram_after,
        }

        # 2. Remote Worker Nodes
        for node in unique_nodes.values():
            try:
                # Calculate assigned layer bytes and KV layer count for this node from assignments
                w_layer_bytes = 0
                w_assigned_kv_layers = 0
                for cid, ainfo in self.assignments.items():
                    if ainfo["node"] == node:
                        ctype = ainfo["type"]
                        lidx = ainfo.get("layer_idx")
                        if ctype == "layer" and lidx is not None and lidx < len(self.layers) and self.layers[lidx] is not None:
                            w_layer_bytes += sum(p.numel() * p.element_size() for p in self.layers[lidx].parameters())
                            w_assigned_kv_layers += 1
                        elif ctype == "attention" and lidx is not None and lidx < len(self.layers) and self.layers[lidx] is not None:
                            if hasattr(self.layers[lidx], "self_attn"):
                                w_layer_bytes += sum(p.numel() * p.element_size() for p in self.layers[lidx].self_attn.parameters())
                            w_assigned_kv_layers += 1
                        elif ctype == "ffn" and lidx is not None and lidx < len(self.layers) and self.layers[lidx] is not None:
                            if hasattr(self.layers[lidx], "mlp"):
                                w_layer_bytes += sum(p.numel() * p.element_size() for p in self.layers[lidx].mlp.parameters())

                w_kv_bytes = 2 * w_assigned_kv_layers * self.num_kv_heads * self.head_dim * self.dtype_size * total_seq
                w_ram = 0
                w_vram = 0

                # Query live worker for memory info if available
                try:
                    resp = node.send_cmd({"cmd": "get_memory"})
                    if resp and resp.get("status") == "ok":
                        w_layer_bytes = resp.get("layer_bytes", w_layer_bytes)
                        w_kv_bytes = resp.get("kv_bytes", w_kv_bytes)
                        w_ram = resp.get("ram_rss", 0)
                        w_vram = resp.get("vram_alloc", 0)
                except Exception:
                    try:
                        resp_st = node.send_cmd({"cmd": "status"})
                        if resp_st and resp_st.get("status") == "ok":
                            comps_dict = resp_st.get("components", {})
                            if comps_dict:
                                w_layer_bytes = sum(c.get("size_bytes", 0) for c in comps_dict.values())
                    except Exception:
                        pass

                node_mem_stats[node.label] = {
                    "is_local": False,
                    "layer_bytes": w_layer_bytes,
                    "kv_bytes": w_kv_bytes,
                    "ram_rss": w_ram,
                    "vram_alloc": w_vram,
                }
            except Exception:
                pass

        stats_dict = {
            "num_input": num_input, "num_output": num_output,
            "total_seq": total_seq, "ttft": ttft,
            "decode_time": decode_time, "total_time": total_time,
            "token_times": token_times,
            "net_sent": net_sent, "net_recv": net_recv,
            "kv_total": kv_total, "kv_per_token": kv_per_token,
            "vram_before": vram_before, "vram_after": vram_after,
            "node_mem_stats": node_mem_stats,
        }

        # Display stats
        self._display_stats(stats_dict)

        # Append detailed audit log report (without prompt/response text)
        self._audit_log_report(stats_dict, temperature)
        self.last_stats = stats_dict
        return stats_dict

    def _forward_pass(self, input_ids_or_embeds, is_prefill):
        """One forward pass through all components."""
        device = self.local_device

        # 1. Embedding
        if input_ids_or_embeds.dtype in (torch.long, torch.int, torch.int64):
            if "embedding" in self.assignments:
                node = self.assignments["embedding"]["node"]
                resp = node.send_cmd({
                    "cmd": "forward_embedding",
                    "component_id": "embedding",
                    "input_ids": input_ids_or_embeds.cpu(),
                })
                hidden = resp["hidden_states"].to(device)
            else:
                emb_dev = next(self.embedding.parameters()).device
                hidden = self.embedding(input_ids_or_embeds.to(emb_dev))
        else:
            hidden = input_ids_or_embeds.to(device)

        # 2. Position info
        seq_len = hidden.shape[1]
        start_pos = self.total_past_len
        position_ids = torch.arange(start_pos, start_pos + seq_len,
                                    device=hidden.device).unsqueeze(0)

        # 3. Position embeddings (RoPE)
        pos_emb = None
        if hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            rope_dev = hidden.device
            if hasattr(self.rotary_emb, "inv_freq"):
                rope_dev = self.rotary_emb.inv_freq.device
            try:
                pos_emb = self.rotary_emb(hidden.to(rope_dev), position_ids.to(rope_dev))
            except Exception:
                pass

        # 4. Run layers (with Direct P2P Ring Pipeline for multi-node contiguous stages)
        i = 0
        while i < self.num_layers:
            lid = f"layer_{i}"
            if lid in self.assignments:
                # Find all consecutive remote layers across possibly multiple remote nodes
                chain_stages = []
                curr_i = i
                while curr_i < self.num_layers and f"layer_{curr_i}" in self.assignments:
                    curr_node = self.assignments[f"layer_{curr_i}"]["node"]
                    next_i = curr_i + 1
                    while next_i < self.num_layers and f"layer_{next_i}" in self.assignments and self.assignments[f"layer_{next_i}"]["node"] == curr_node:
                        next_i += 1
                    chain_stages.append({
                        "start_layer": curr_i,
                        "end_layer": next_i - 1,
                        "node_host": curr_node.host,
                        "node_port": curr_node.port,
                        "node": curr_node,
                    })
                    curr_i = next_i

                # Multi-node Direct P2P Pipeline Chain
                if len(chain_stages) > 1:
                    first_node = chain_stages[0]["node"]
                    payload_stages = [
                        {"start_layer": s["start_layer"], "end_layer": s["end_layer"],
                         "node_host": s["node_host"], "node_port": s["node_port"]}
                        for s in chain_stages
                    ]
                    resp = first_node.send_cmd({
                        "cmd": "pipeline_forward_chain",
                        "stages": payload_stages,
                        "stage_idx": 0,
                        "hidden_states": hidden.cpu(),
                        "position_ids": position_ids.cpu(),
                        "position_embeddings": _to_cpu_recursive(pos_emb),
                    })
                    hidden = resp["hidden_states"].to(device)
                    i = chain_stages[-1]["end_layer"] + 1
                    continue
                elif len(chain_stages) == 1 and (chain_stages[0]["end_layer"] - chain_stages[0]["start_layer"]) > 0:
                    # Single-node multi-layer stage
                    hidden = self._remote_stage(chain_stages[0]["start_layer"], chain_stages[0]["end_layer"], chain_stages[0]["node"], hidden, position_ids, pos_emb)
                    i = chain_stages[0]["end_layer"] + 1
                    continue

            hidden = self._run_layer(i, hidden, position_ids, pos_emb)
            i += 1

        # 5. Norm + LM head
        if "lm_head" in self.assignments:
            node = self.assignments["lm_head"]["node"]
            resp = node.send_cmd({
                "cmd": "forward_lm_head",
                "component_id": "lm_head",
                "hidden_states": hidden.cpu(),
            })
            logits = resp["logits"].to(device)
        else:
            if hasattr(self, "norm") and self.norm is not None:
                norm_dev = next(self.norm.parameters()).device if list(self.norm.parameters()) else device
                hidden = self.norm(hidden.to(norm_dev))
            if hasattr(self, "lm_head") and self.lm_head is not None:
                head_dev = next(self.lm_head.parameters()).device if list(self.lm_head.parameters()) else device
                logits = self.lm_head(hidden.to(head_dev))
            else:
                logits = hidden

        return logits[:, -1, :]

    def _run_layer(self, idx, hidden, position_ids, pos_emb):
        """Run a single layer — local, remote, or hybrid."""
        device = self.local_device
        lid = f"layer_{idx}"
        aid = f"attn_{idx}"
        fid = f"ffn_{idx}"

        # Case 1: Whole layer is remote
        if lid in self.assignments:
            return self._remote_layer(idx, hidden, position_ids, pos_emb)

        layer = self.layers[idx]

        # Case 2: Attention is remote, FFN local
        if aid in self.assignments:
            return self._hybrid_attn_remote(idx, layer, hidden, position_ids, pos_emb)

        # Case 3: FFN is remote, attention local
        if fid in self.assignments:
            return self._hybrid_ffn_remote(idx, layer, hidden, position_ids, pos_emb)

        # Case 4: Some experts are remote
        remote_experts = {int(cid.split("_")[2]): info
                          for cid, info in self.assignments.items()
                          if cid.startswith(f"expert_{idx}_")}
        if remote_experts:
            return self._hybrid_experts_remote(idx, layer, hidden, position_ids, pos_emb, remote_experts)

        # Case 5: Fully local
        return self._local_layer(idx, layer, hidden, position_ids, pos_emb)

    def _local_layer(self, idx, layer, hidden, position_ids, pos_emb):
        """Run a layer fully locally with per-layer KV cache."""
        from transformers import DynamicCache

        layer_dev = next(layer.parameters()).device
        hidden = _match_dtype(hidden, layer).to(layer_dev)
        if position_ids is not None:
            position_ids = position_ids.to(layer_dev)
        pos_emb = _match_tuple_dtype(pos_emb, layer)
        if pos_emb is not None:
            pos_emb = tuple(p.to(layer_dev) if isinstance(p, torch.Tensor) else p for p in pos_emb)

        if idx not in self.local_kv:
            self.local_kv[idx] = DynamicCache()
        cache = self.local_kv[idx]

        orig_idx = getattr(layer.self_attn, "layer_idx", 0) if hasattr(layer, "self_attn") else 0
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = 0

        kwargs = dict(position_ids=position_ids, past_key_values=cache, use_cache=True, attention_mask=None)
        if pos_emb is not None:
            kwargs["position_embeddings"] = pos_emb
        try:
            outputs = layer(hidden, **kwargs)
        except TypeError as e:
            if "past_key_values" in str(e) or "unexpected keyword argument" in str(e):
                kwargs["past_key_value"] = kwargs.pop("past_key_values", None)
                try:
                    outputs = layer(hidden, **kwargs)
                except TypeError:
                    kwargs.pop("position_embeddings", None)
                    try:
                        outputs = layer(hidden, **kwargs)
                    except TypeError:
                        kwargs.pop("attention_mask", None)
                        outputs = layer(hidden, **kwargs)
            else:
                kwargs.pop("position_embeddings", None)
                try:
                    outputs = layer(hidden, **kwargs)
                except TypeError:
                    kwargs.pop("attention_mask", None)
                    outputs = layer(hidden, **kwargs)

        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = orig_idx

        return outputs if isinstance(outputs, torch.Tensor) else outputs[0]

    def _send_cmd_with_dtype_recovery(self, node, msg):
        """Send command to worker, auto-recovering from any remote PyTorch CPU dtype mismatch."""
        try:
            return node.send_cmd(msg)
        except RuntimeError as e:
            err_str = str(e).lower()
            if any(k in err_str for k in ("dtype", "mat1", "m1 and m2", "same type", "half", "bfloat16", "float")):
                orig_hidden = msg.get("hidden_states")
                if orig_hidden is not None and isinstance(orig_hidden, torch.Tensor):
                    candidates = [torch.bfloat16, torch.float16, torch.float32]
                    for target_dt in candidates:
                        if orig_hidden.dtype == target_dt:
                            continue
                        try:
                            msg_retry = dict(msg)
                            msg_retry["hidden_states"] = orig_hidden.to(dtype=target_dt)
                            if "position_embeddings" in msg_retry and msg_retry["position_embeddings"] is not None:
                                msg_retry["position_embeddings"] = tuple(
                                    p.to(dtype=target_dt) if isinstance(p, torch.Tensor) else p
                                    for p in msg_retry["position_embeddings"]
                                )
                            return node.send_cmd(msg_retry)
                        except Exception:
                            continue
            raise e

    def _remote_stage(self, start_idx, end_idx, node, hidden, position_ids, pos_emb):
        """Send hidden_states to remote worker for a contiguous sequence of layers in 1 network round-trip."""
        target_dt = torch.bfloat16 if getattr(self, "config", {}).get("torch_dtype") == "bfloat16" else hidden.dtype
        hs_cpu = hidden.to(dtype=target_dt).cpu()
        comp_ids = [f"layer_{k}" for k in range(start_idx, end_idx + 1)]
        msg = {
            "cmd": "forward_stage",
            "start_layer": start_idx,
            "end_layer": end_idx,
            "component_ids": comp_ids,
            "hidden_states": hs_cpu,
            "position_ids": position_ids.cpu(),
        }
        if pos_emb is not None:
            msg["position_embeddings"] = tuple(p.to(dtype=target_dt).cpu() if isinstance(p, torch.Tensor) else p for p in pos_emb)

        try:
            resp = self._send_cmd_with_dtype_recovery(node, msg)
            if isinstance(resp, dict) and resp.get("status") == "ok" and "hidden_states" in resp:
                return resp["hidden_states"].to(self.local_device, dtype=hidden.dtype)
        except Exception:
            pass

        # Graceful fallback to per-layer forward if remote worker lacks forward_stage
        curr_hidden = hidden
        for k in range(start_idx, end_idx + 1):
            curr_hidden = self._remote_layer(k, curr_hidden, position_ids, pos_emb)
        return curr_hidden

    def _remote_layer(self, idx, hidden, position_ids, pos_emb):
        """Send hidden_states to remote worker for full layer forward."""
        node = self.assignments[f"layer_{idx}"]["node"]
        target_dt = torch.bfloat16 if getattr(self, "config", {}).get("torch_dtype") == "bfloat16" else hidden.dtype
        hs_cpu = hidden.to(dtype=target_dt).cpu()
        msg = {
            "cmd": "forward_layer",
            "component_id": f"layer_{idx}",
            "hidden_states": hs_cpu,
            "position_ids": position_ids.cpu(),
        }
        if pos_emb is not None:
            msg["position_embeddings"] = tuple(p.to(dtype=target_dt).cpu() if isinstance(p, torch.Tensor) else p for p in pos_emb)
        resp = self._send_cmd_with_dtype_recovery(node, msg)
        return resp["hidden_states"].to(self.local_device, dtype=hidden.dtype)

    def _hybrid_attn_remote(self, idx, layer, hidden, position_ids, pos_emb):
        """Attention on remote, FFN locally."""
        # 1. Input layernorm
        in_norm_dev = next(layer.input_layernorm.parameters()).device if hasattr(layer, "input_layernorm") else self.local_device
        hidden = hidden.to(in_norm_dev)
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        # 2. Remote attention
        node = self.assignments[f"attn_{idx}"]["node"]
        target_dt = torch.bfloat16 if getattr(self, "config", {}).get("torch_dtype") == "bfloat16" else hidden.dtype
        hs_cpu = hidden.to(dtype=target_dt).cpu()
        msg = {
            "cmd": "forward_attention",
            "component_id": f"attn_{idx}",
            "hidden_states": hs_cpu,
            "position_ids": position_ids.cpu() if position_ids is not None else None,
        }
        if pos_emb is not None:
            msg["position_embeddings"] = tuple(p.to(dtype=target_dt).cpu() if isinstance(p, torch.Tensor) else p for p in pos_emb)
        resp = self._send_cmd_with_dtype_recovery(node, msg)
        
        post_norm_dev = next(layer.post_attention_layernorm.parameters()).device if hasattr(layer, "post_attention_layernorm") else self.local_device
        attn_out = resp["hidden_states"].to(post_norm_dev, dtype=hidden.dtype)
        
        hidden = residual.to(post_norm_dev) + attn_out

        # 3. Local FFN
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        
        mlp_dev = next(layer.mlp.parameters()).device if hasattr(layer, "mlp") else self.local_device
        hidden = hidden.to(mlp_dev)
        residual = residual.to(mlp_dev)
        
        mlp_out = layer.mlp(hidden)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        hidden = residual + mlp_out
        return hidden

    def _hybrid_ffn_remote(self, idx, layer, hidden, position_ids, pos_emb):
        """Attention locally, FFN on remote."""
        from transformers import DynamicCache

        # 1. Input layernorm + local attention
        in_norm_dev = next(layer.input_layernorm.parameters()).device if hasattr(layer, "input_layernorm") else self.local_device
        hidden = hidden.to(in_norm_dev)
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        
        attn_dev = next(layer.self_attn.parameters()).device if hasattr(layer, "self_attn") else self.local_device
        hidden = hidden.to(attn_dev)
        if position_ids is not None:
            position_ids = position_ids.to(attn_dev)
        if pos_emb is not None:
            pos_emb = tuple(p.to(attn_dev) if isinstance(p, torch.Tensor) else p for p in pos_emb)

        if idx not in self.local_kv:
            self.local_kv[idx] = DynamicCache()
        cache = self.local_kv[idx]

        orig_idx = getattr(layer.self_attn, "layer_idx", 0)
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = 0

        attn_kwargs = dict(position_ids=position_ids, past_key_values=cache, use_cache=True, attention_mask=None)
        if pos_emb is not None:
            attn_kwargs["position_embeddings"] = pos_emb
        try:
            attn_out = layer.self_attn(hidden, **attn_kwargs)
        except TypeError as e:
            if "past_key_values" in str(e) or "unexpected keyword argument" in str(e):
                attn_kwargs["past_key_value"] = attn_kwargs.pop("past_key_values", None)
                try:
                    attn_out = layer.self_attn(hidden, **attn_kwargs)
                except TypeError:
                    attn_kwargs.pop("position_embeddings", None)
                    try:
                        attn_out = layer.self_attn(hidden, **attn_kwargs)
                    except TypeError:
                        attn_kwargs.pop("attention_mask", None)
                        attn_out = layer.self_attn(hidden, **attn_kwargs)
            else:
                attn_kwargs.pop("position_embeddings", None)
                try:
                    attn_out = layer.self_attn(hidden, **attn_kwargs)
                except TypeError:
                    attn_kwargs.pop("attention_mask", None)
                    attn_out = layer.self_attn(hidden, **attn_kwargs)

        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = orig_idx

        hidden = residual.to(attn_dev) + attn_out[0]

        # 2. Post-norm + remote FFN
        post_norm_dev = next(layer.post_attention_layernorm.parameters()).device if hasattr(layer, "post_attention_layernorm") else self.local_device
        hidden = hidden.to(post_norm_dev)
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)

        node = self.assignments[f"ffn_{idx}"]["node"]
        target_dt = torch.bfloat16 if getattr(self, "config", {}).get("torch_dtype") == "bfloat16" else hidden.dtype
        resp = self._send_cmd_with_dtype_recovery(node, {
            "cmd": "forward_ffn",
            "component_id": f"ffn_{idx}",
            "hidden_states": hidden.to(dtype=target_dt).cpu(),
        })
        ffn_out = resp["hidden_states"].to(post_norm_dev, dtype=hidden.dtype)
        hidden = residual + ffn_out
        return hidden

    def _hybrid_experts_remote(self, idx, layer, hidden, position_ids, pos_emb, remote_experts):
        """Layer with some experts remote, others local. Router always local."""
        from transformers import DynamicCache

        # 1. Attention (local)
        in_norm_dev = next(layer.input_layernorm.parameters()).device if hasattr(layer, "input_layernorm") else self.local_device
        hidden = hidden.to(in_norm_dev)
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        
        attn_dev = next(layer.self_attn.parameters()).device if hasattr(layer, "self_attn") else self.local_device
        hidden = hidden.to(attn_dev)
        if position_ids is not None:
            position_ids = position_ids.to(attn_dev)
        if pos_emb is not None:
            pos_emb = tuple(p.to(attn_dev) if isinstance(p, torch.Tensor) else p for p in pos_emb)

        if idx not in self.local_kv:
            self.local_kv[idx] = DynamicCache()
        cache = self.local_kv[idx]

        orig_idx = getattr(layer.self_attn, "layer_idx", 0)
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = 0

        attn_kwargs = dict(position_ids=position_ids, past_key_values=cache, use_cache=True)
        if pos_emb is not None:
            attn_kwargs["position_embeddings"] = pos_emb
        try:
            attn_out = layer.self_attn(hidden, **attn_kwargs)
        except TypeError as e:
            if "past_key_values" in str(e) or "unexpected keyword argument" in str(e):
                attn_kwargs["past_key_value"] = attn_kwargs.pop("past_key_values")
                try:
                    attn_out = layer.self_attn(hidden, **attn_kwargs)
                except TypeError:
                    attn_kwargs.pop("position_embeddings", None)
                    attn_out = layer.self_attn(hidden, **attn_kwargs)
            else:
                attn_kwargs.pop("position_embeddings", None)
                attn_out = layer.self_attn(hidden, **attn_kwargs)

        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = orig_idx

        hidden = residual.to(attn_dev) + attn_out[0]

        # 2. Post-norm
        post_norm_dev = next(layer.post_attention_layernorm.parameters()).device if hasattr(layer, "post_attention_layernorm") else self.local_device
        hidden = hidden.to(post_norm_dev)
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        
        router_dev = next(layer.mlp.gate.parameters()).device if hasattr(layer.mlp, "gate") else self.local_device
        hidden = hidden.to(router_dev)

        batch_size, seq_len, hidden_dim = hidden.shape
        hidden_flat = hidden.view(-1, hidden_dim)

        # 3. Router (local)
        router_logits = layer.mlp.gate(hidden_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.mlp.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim), dtype=hidden.dtype, device=hidden.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        target_dt = torch.bfloat16 if getattr(self, "config", {}).get("torch_dtype") == "bfloat16" else hidden.dtype

        for expert_idx in range(self.num_experts):
            idx_list, top_x = torch.where(expert_mask[expert_idx])
            if idx_list.shape[0] == 0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx_list.tolist()
            current_state = hidden_flat[None, top_x_list].reshape(-1, hidden_dim)
            
            if expert_idx in remote_experts:
                node = remote_experts[expert_idx]["node"]
                resp = self._send_cmd_with_dtype_recovery(node, {
                    "cmd": "forward_expert",
                    "component_id": f"expert_{idx}_{expert_idx}",
                    "hidden_states": current_state.to(dtype=target_dt).cpu(),
                })
                current_hidden_states = resp["hidden_states"].to(hidden.device, dtype=hidden.dtype)
            else:
                expert_layer = layer.mlp.experts[expert_idx]
                if expert_layer is None:
                    continue
                exp_dev = next(expert_layer.parameters()).device
                current_hidden_states = expert_layer(current_state.to(exp_dev)).to(hidden.device)
                
            current_hidden_states = current_hidden_states * routing_weights[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x.to(hidden.device), current_hidden_states)

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        hidden = residual.to(hidden.device) + final_hidden_states
        return hidden

    def _sample(self, logits, temperature):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ── Stats Display ─────────────────────────────────────────────────────
    def _display_stats(self, s):
        hdr("DISTRIBUTED INFERENCE STATISTICS")

        decode_tps = ((s["num_output"] - 1) / s["decode_time"]
                      if s["decode_time"] > 0 and s["num_output"] > 1 else 0)
        prefill_tps = s["num_input"] / s["ttft"] if s["ttft"] > 0 else 0

        shdr("TOKENS & TIMING")
        rows = [
            ["Input tokens", f"{s['num_input']:,}"],
            ["Output tokens", f"{s['num_output']:,}"],
            ["Total sequence", f"{s['total_seq']:,}"],
            ["", ""],
            [f"{C.B}Time to First Token (TTFT){C.RS}", f"{C.G}{fmt_time(s['ttft'])}{C.RS}"],
            ["Decode time", fmt_time(s["decode_time"])],
            [f"{C.B}Total time{C.RS}", f"{C.B}{fmt_time(s['total_time'])}{C.RS}"],
            ["", ""],
            ["Prefill speed", f"{prefill_tps:.1f} tok/s"],
            [f"{C.B}Decode speed{C.RS}", f"{C.G}{decode_tps:.1f} tok/s{C.RS}"],
        ]
        if s["token_times"]:
            import statistics
            rows += [
                ["", ""],
                ["Per-token avg", fmt_time(statistics.mean(s["token_times"]))],
                ["Per-token min", fmt_time(min(s["token_times"]))],
                ["Per-token max", fmt_time(max(s["token_times"]))],
            ]
        print(tabulate(rows, tablefmt="rounded_outline", colalign=("left", "right")))

        shdr("MEMORY")
        mem_rows = [
            ["KV cache/token (theoretical)", fmt_bytes(s["kv_per_token"])],
            [f"KV cache total ({s['total_seq']} tokens)", fmt_bytes(s["kv_total"])],
        ]
        if s["vram_before"] or s["vram_after"]:
            mem_rows += [
                ["", ""],
                ["VRAM before generation", fmt_bytes(s["vram_before"])],
                ["VRAM after generation", fmt_bytes(s["vram_after"])],
                ["VRAM delta (KV+activations)", fmt_bytes(s["vram_after"] - s["vram_before"])],
            ]
        print(tabulate(mem_rows, tablefmt="rounded_outline", colalign=("left", "right")))

        if s.get("node_mem_stats"):
            shdr("MEMORY CONSUMPTION PER NODE (LAYERS + ACTIVE KV CACHE)")
            mem_headers = ["Node", "Components", "Layer Memory", "KV Cache", "Memory Footprint"]
            mem_table_rows = []
            summary = self.get_assignment_summary()
            for nlabel, mstats in s["node_mem_stats"].items():
                comps = summary.get(nlabel, [])
                comp_str = compress_comp_list(comps)
                if mstats.get("is_local"):
                    mem_footprint = f"{fmt_bytes(mstats.get('vram_after', 0))} (VRAM)" if mstats.get("vram_after", 0) > 0 else "Local RAM"
                else:
                    if mstats.get("vram_alloc", 0) > 0:
                        mem_footprint = f"{fmt_bytes(mstats.get('vram_alloc', 0))} (VRAM)"
                    else:
                        mem_footprint = f"{fmt_bytes(mstats.get('ram_rss', 0))} (Process RAM)"
                mem_table_rows.append([
                    nlabel,
                    comp_str,
                    fmt_bytes(mstats.get("layer_bytes", 0)),
                    fmt_bytes(mstats.get("kv_bytes", 0)),
                    mem_footprint
                ])
            print(tabulate(mem_table_rows, headers=mem_headers, tablefmt="rounded_outline"))

        shdr("NETWORK TRANSFER")
        net_rows = [
            [f"{C.B}Data sent to remote nodes{C.RS}", f"{C.Y}{fmt_bytes(s['net_sent'])}{C.RS}"],
            [f"{C.B}Data received from remote nodes{C.RS}", f"{C.Y}{fmt_bytes(s['net_recv'])}{C.RS}"],
            ["Total network I/O", fmt_bytes(s["net_sent"] + s["net_recv"])],
        ]
        if s["total_time"] > 0:
            net_rows.append(["Avg network I/O rate (over total time)",
                             f"{(s['net_sent'] + s['net_recv']) / s['total_time'] / (1024**2):.1f} MB/s"])
        # Per-node breakdown
        node_stats = defaultdict(lambda: {"sent": 0, "recv": 0})
        for info in self.assignments.values():
            n = info["node"]
            node_stats[n.label]["sent"] = n.bytes_sent   # already includes this session
            node_stats[n.label]["recv"] = n.bytes_received
        if node_stats:
            net_rows.append(["", ""])
            for nlabel, ns in node_stats.items():
                net_rows.append([f"  {nlabel} ↑sent", fmt_bytes(ns["sent"])])
                net_rows.append([f"  {nlabel} ↓recv", fmt_bytes(ns["recv"])])
        print(tabulate(net_rows, tablefmt="rounded_outline", colalign=("left", "right")))

    def _audit_log_report(self, s, temperature):
        """Append a full audit report of this inference run to inference_audit.log."""
        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters()) if self.model else 0
        dtype_str = getattr(next(self.model.parameters()), "dtype", "unknown") if self.model else "?"

        quant_mode = getattr(self, "quant", "none")
        quant_str = {
            "4bit": "4-bit NF4 Quantization (BitsAndBytes)",
            "8bit": "8-bit Quantization (BitsAndBytes)",
            "int8": "8-bit Dynamic Quantization (PyTorch native int8)",
            "none": f"Full Precision ({dtype_str})",
        }.get(quant_mode, quant_mode)

        arch_desc = f"MoE ({self.num_experts} experts, top-{self.num_experts_per_tok} active/tok)" if self.is_moe else "Dense Transformer"

        prefill_tps = s["num_input"] / s["ttft"] if s["ttft"] > 0 else 0
        decode_tps = ((s["num_output"] - 1) / s["decode_time"]
                      if s["decode_time"] > 0 and s["num_output"] > 1 else 0)

        # Build Node / Split topology
        summary = self.get_assignment_summary()
        split_lines = []
        for nlabel, comps in summary.items():
            comp_str = compress_comp_list(comps)
            split_lines.append(f"  • {nlabel}:\n    Allocated: {comp_str}")
        split_block = "\n".join(split_lines) if split_lines else "  • Fully Local Execution"

        # Per-node network transfer breakdown
        node_stats = defaultdict(lambda: {"sent": 0, "recv": 0})
        for info in self.assignments.values():
            n = info["node"]
            node_stats[n.label]["sent"] = n.bytes_sent
            node_stats[n.label]["recv"] = n.bytes_received
        net_lines = []
        for nlabel, ns in node_stats.items():
            net_lines.append(f"  • {nlabel}: Sent: {fmt_bytes(ns['sent'])}, Received: {fmt_bytes(ns['recv'])}")
        net_block = "\n".join(net_lines) if net_lines else "  • Fully Local Execution (0 network bytes)"

        # Per-node memory breakdown (Layers + Active KV cache + footprint)
        node_mem_lines = []
        for nlabel, mstats in s.get("node_mem_stats", {}).items():
            comps = summary.get(nlabel, [])
            comp_str = compress_comp_list(comps)
            if mstats.get("is_local"):
                footprint = f"{fmt_bytes(mstats.get('vram_after', 0))} (VRAM)" if mstats.get("vram_after", 0) > 0 else "Local RAM"
            else:
                footprint = f"{fmt_bytes(mstats.get('vram_alloc', 0))} (VRAM)" if mstats.get("vram_alloc", 0) > 0 else f"{fmt_bytes(mstats.get('ram_rss', 0))} (Process RAM)"
            node_mem_lines.append(
                f"  • {nlabel}:\n"
                f"    - Assigned:      {comp_str}\n"
                f"    - Layer Memory:  {fmt_bytes(mstats.get('layer_bytes', 0))}\n"
                f"    - Active KV:     {fmt_bytes(mstats.get('kv_bytes', 0))}\n"
                f"    - Total Footprint: {footprint}"
            )
        node_mem_block = "\n".join(node_mem_lines) if node_mem_lines else "  • N/A"

        # Token latency distribution
        lat_lines = []
        if s.get("token_times"):
            import statistics
            lat_lines.append(f"Per-token Latency (Avg):    {fmt_time(statistics.mean(s['token_times']))}")
            lat_lines.append(f"Per-token Latency (Min):    {fmt_time(min(s['token_times']))}")
            lat_lines.append(f"Per-token Latency (Max):    {fmt_time(max(s['token_times']))}")
            if len(s['token_times']) > 4:
                sorted_t = sorted(s['token_times'])
                p50 = sorted_t[int(len(sorted_t) * 0.50)]
                p90 = sorted_t[int(len(sorted_t) * 0.90)]
                p99 = sorted_t[int(len(sorted_t) * 0.99)]
                lat_lines.append(f"Per-token Latency (P50):    {fmt_time(p50)}")
                lat_lines.append(f"Per-token Latency (P90):    {fmt_time(p90)}")
                lat_lines.append(f"Per-token Latency (P99):    {fmt_time(p99)}")
        lat_block = "\n".join(lat_lines) if lat_lines else "N/A"

        report = f"""================================================================================
INFERENCE AUDIT REPORT — {now_str}
================================================================================

[1] MODEL INFORMATION
--------------------------------------------------------------------------------
Model ID / Name:     {self.model_id}
Architecture:        {arch_desc}
Quantization Mode:   {quant_str}
Precision Base:      {dtype_str}
Total Parameters:    {total_params:,} ({fmt_params(total_params)})
Total Memory in RAM: {fmt_bytes(total_bytes)}
Dimensions:          Layers: {self.num_layers}, Hidden Size: {self.hidden_size}, KV Heads: {self.num_kv_heads}, Head Dim: {self.head_dim}

[2] TOPOLOGY & COMPONENT DISTRIBUTION
--------------------------------------------------------------------------------
{split_block}

[3] GENERATION METADATA
--------------------------------------------------------------------------------
Temperature:         {temperature}
Prompt Tokens:       {s['num_input']:,} tokens
Generated Tokens:    {s['num_output']:,} tokens
Total Sequence:      {s['total_seq']:,} tokens

[4] LATENCY & THROUGHPUT METRICS
--------------------------------------------------------------------------------
Time to First Token (TTFT): {fmt_time(s['ttft'])} (Prefill: {prefill_tps:.1f} tok/s)
Decode Time:                {fmt_time(s['decode_time'])} (Decode: {decode_tps:.1f} tok/s)
Total End-to-End Latency:   {fmt_time(s['total_time'])}
{lat_block}

[5] NETWORK PAYLOAD & DATA TRANSFER
--------------------------------------------------------------------------------
Total Data Sent:            {fmt_bytes(s['net_sent'])}
Total Data Received:        {fmt_bytes(s['net_recv'])}
Total Network I/O:          {fmt_bytes(s['net_sent'] + s['net_recv'])}
Avg Network Rate:           {(s['net_sent'] + s['net_recv']) / max(s['total_time'], 0.001) / (1024**2):.2f} MB/s

Node Breakdown:
{net_block}

[6] HARDWARE & MEMORY FOOTPRINT (PER NODE)
--------------------------------------------------------------------------------
Local Device:               {self.local_device}
GPU VRAM Before:            {fmt_bytes(s['vram_before'])}
GPU VRAM After:             {fmt_bytes(s['vram_after'])}
GPU VRAM Delta:             {fmt_bytes(max(s['vram_after'] - s['vram_before'], 0))}
KV Cache Total (est):       {fmt_bytes(s['kv_total'])}

Node Memory Breakdown (Layer Weights + Active KV Cache):
{node_mem_block}
================================================================================
"""
        log_file = Path("inference_audit.log")
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(report + "\n")
            print(f"  {C.OK} Inference audit report saved to {C.B}{log_file}{C.RS}")
        except Exception as e:
            print(f"  {C.WN} Failed to write audit log: {e}")

        # --- CSV LOGGING ---
        csv_file = Path("inference_audit.csv")
        try:
            import csv
            file_exists = csv_file.exists()
            with open(csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "Timestamp", "Model", "Quantization", "Total Params",
                        "Input Tokens", "Output Tokens", "TTFT (s)", "Decode Time (s)", "Total Time (s)",
                        "Prefill TPS", "Decode TPS", "Net Sent (Bytes)", "Net Recv (Bytes)",
                        "VRAM Delta (Bytes)", "KV Total (Bytes)"
                    ])
                writer.writerow([
                    now_str, self.model_id, quant_mode, total_params,
                    s["num_input"], s["num_output"], s["ttft"], s["decode_time"], s["total_time"],
                    prefill_tps, decode_tps, s["net_sent"], s["net_recv"],
                    max(s["vram_after"] - s["vram_before"], 0), s["kv_total"]
                ])
            print(f"  {C.OK} Inference audit stats appended to {C.B}{csv_file}{C.RS}")
        except Exception as e:
            print(f"  {C.WN} Failed to write CSV log: {e}")

    def cleanup(self):
        """Unload all remote components and close connections."""
        for comp_id, info in self.assignments.items():
            try:
                info["node"].send_cmd({"cmd": "unload", "component_id": comp_id})
            except Exception:
                pass
        for info in self.assignments.values():
            info["node"].close()


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive CLI
# ═══════════════════════════════════════════════════════════════════════════════
class CLI:
    def __init__(self):
        self.nodes: dict[str, NodeConnection] = {}  # key → NodeConnection
        self.selected_model = None
        self.model_dir = None
        self.dist_model = DistributedModel()
        self.config_path = Path("dist_config.json")
        self.assignment_cache_path = Path("assignment_cache.json")
        self.reverse_listener = ReverseCoordinatorListener(
            port=DEFAULT_PORT,
            on_worker_connected=self._on_reverse_worker_connected
        )
        self.reverse_listener.start()
        self.memory_profiling_enabled = False
        self._load_last()

    def _on_reverse_worker_connected(self, nc, info):
        key = f"{nc.host}:{nc.port}"
        self.nodes[key] = nc
        ram_free_gb = info.get("ram_free", 0) / (1024**3)
        dev = info.get("device", "cpu")
        print(f"\n  {C.G}⚡ [NEW REVERSE WORKER CONNECTED]{C.RS} {C.B}{nc.label}{C.RS} (RAM Free: {ram_free_gb:.1f} GB, Device: {dev})")

    def _save_assignment_cache(self, preset_name="default"):
        """Save the current model's distribution assignment topology to cache."""
        if not self.selected_model or not self.dist_model.assignments:
            return False
        model_id = self.selected_model["id"]
        cache_data = {}
        if self.assignment_cache_path.exists():
            try:
                with open(self.assignment_cache_path) as f:
                    cache_data = json.load(f)
            except Exception:
                cache_data = {}

        exported = self.dist_model.export_assignments()
        summary = self.dist_model.get_assignment_summary()
        summary_rows = [f"{k}: {self._compress_comp_list(v)}" for k, v in summary.items()]

        model_entry = cache_data.setdefault(model_id, {})
        model_entry[preset_name] = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_layers": self.dist_model.num_layers,
            "quant": getattr(self.dist_model, "quant", "none"),
            "assignments": exported,
            "summary": summary_rows,
        }

        try:
            with open(self.assignment_cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
            return True
        except Exception:
            return False

    def _get_cached_assignment(self, preset_name="default"):
        """Retrieve cached assignment dictionary for the currently selected model."""
        if not self.selected_model or not self.assignment_cache_path.exists():
            return None
        model_id = self.selected_model["id"]
        try:
            with open(self.assignment_cache_path) as f:
                cache_data = json.load(f)
            return cache_data.get(model_id, {}).get(preset_name)
        except Exception:
            return None

    def _match_cached_node(self, entry):
        """Find the matching active NodeConnection for a cached assignment entry."""
        target_host = entry.get("node_host")
        target_port = entry.get("node_port")
        target_hostname = entry.get("node_hostname")

        # 1. Exact host:port match
        for key, nc in self.nodes.items():
            if nc.host == target_host and nc.port == target_port:
                return nc

        # 2. Hostname match (handles IP address changes over DHCP/LAN)
        for key, nc in self.nodes.items():
            if target_hostname and target_hostname != "unknown" and getattr(nc, "hostname", "") == target_hostname:
                return nc

        # 3. Host only match
        for key, nc in self.nodes.items():
            if nc.host == target_host:
                return nc

        # 4. Fallback: if only 1 node connected, match it
        if len(self.nodes) == 1:
            return list(self.nodes.values())[0]

        return None

    def _restore_cached_assignment(self, preset_name="default"):
        """Restore model distribution from cached preset."""
        cached = self._get_cached_assignment(preset_name)
        if not cached:
            return 0

        entries = cached.get("assignments", [])
        if not entries:
            return 0

        restored_count = self.dist_model.import_assignments(entries, self._match_cached_node)
        return restored_count

    def _save_config(self):
        try:
            nodes_info = {}
            for key, nc in self.nodes.items():
                if not getattr(nc, "is_reverse", False):
                    nodes_info[key] = {"host": nc.host, "port": nc.port, "hostname": nc.hostname}
            data = {
                "model": self.selected_model,
                "model_dir": str(self.model_dir) if self.model_dir else None,
                "quant": getattr(self.dist_model, "quant", "none"),
                "nodes": nodes_info,
            }
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_last(self):
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                    if data.get("model") and data.get("model_dir"):
                        self.selected_model = data["model"]
                        self.model_dir = Path(data["model_dir"])
                        quant = data.get("quant", "none")
                        self.dist_model.load_model(self.selected_model["id"], self.model_dir, quant=quant)
                    # Reconnect saved nodes (fast check, immediately drop if unreachable/ping fails)
                    saved_nodes = data.get("nodes", {})
                    for key, ninfo in saved_nodes.items():
                        try:
                            nc = NodeConnection(ninfo["host"], ninfo["port"], ninfo.get("hostname", "unknown"))
                            nc.connect(timeout=1.0)
                            resp = nc.send_cmd({"cmd": "ping"})
                            if resp and resp.get("status") == "ok":
                                info_resp = nc.send_cmd({"cmd": "info"})
                                nc.hostname = info_resp.get("hostname", nc.hostname)
                                nc.label = f"{nc.hostname} ({nc.host}:{nc.port})"
                                self.nodes[key] = nc
                                print(f"  {C.OK} Connected node: {nc.label}")
                            else:
                                nc.close()
                        except Exception:
                            # Node is offline/unreachable — immediately leave it
                            pass
            except Exception:
                pass

    def run(self):
        self._banner()
        while True:
            c = self._main_menu()
            if   c == 1: self._node_menu()
            elif c == 2: self._model_menu()
            elif c == 3: self._inspect_menu()
            elif c == 4: self._distribution_menu()
            elif c == 5: self._inference_menu()
            elif c == 0:
                self.reverse_listener.stop()
                self.dist_model.cleanup()
                print(f"\n  {C.G}Goodbye!{C.RS}\n")
                sys.exit(0)

    def _banner(self):
        print(f"""
{C.H}╔════════════════════════════════════════════════════════════╗
║          Distributed Model Inference Controller            ║
║    Split models across machines · Interactive setup         ║
╚════════════════════════════════════════════════════════════╝{C.RS}""")
        # Local system info
        print(f"\n  {C.B}Local node:{C.RS} {platform.node()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"    GPU:  {C.G}{p.name}{C.RS} ({p.total_memory/(1024**3):.1f} GB)")
        print(f"    RAM:  {psutil.virtual_memory().total/(1024**3):.1f} GB")

    def _main_menu(self):
        q_tag = f" ({self.dist_model.quant})" if (self.dist_model.model and getattr(self.dist_model, "quant", "none") != "none") else ""
        model_lbl = f"{C.G}{self.selected_model['name']}{q_tag}{C.RS}" if self.selected_model else f"{C.D}None{C.RS}"
        nodes_lbl = f"{C.G}{len(self.nodes)} connected{C.RS}" if self.nodes else f"{C.D}0{C.RS}"
        ready = self.dist_model.is_distributed

        print(f"""
{C.H}{'─'*58}
  MAIN MENU          Model: {model_lbl}  Nodes: {nodes_lbl}
{'─'*58}{C.RS}

  {C.B}1.{C.RS} Manage Remote Nodes
  {C.B}2.{C.RS} Select & Download Model
  {C.B}3.{C.RS} Inspect Model Parameters & Sizes  {"" if self.dist_model.model else f"{C.D}(load model first){C.RS}"}
  {C.B}4.{C.RS} Configure Distribution            {"" if self.dist_model.model else f"{C.D}(load model first){C.RS}"}
  {C.B}5.{C.RS} Run Inference                     {f"{C.G}Ready{C.RS}" if ready else f"{C.D}(configure first){C.RS}"}
  {C.B}0.{C.RS} Exit""")
        return ask("Select [0-5]:", range(0, 6))

    def _inspect_menu(self):
        if self.dist_model.model is None:
            print(f"\n  {C.Y}Load a model first (option 2).{C.RS}")
            return
        self.dist_model.inspect_model()
        input(f"\n  {C.D}Press Enter to continue...{C.RS}")

    # ── Node Management ──────────────────────────────────────────────────
    def _node_menu(self):
        while True:
            hdr("REMOTE NODES")

            # Detect Master local LAN IP
            my_ip = "127.0.0.1"
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 80))
                my_ip = s.getsockname()[0]
                s.close()
            except Exception:
                pass

            print(f"  {C.CY}⚡ Reverse Coordinator:{C.RS} Listening on {C.B}0.0.0.0:{self.reverse_listener.port}{C.RS}")
            print(f"  {C.D}   (Lab PCs behind UFW can run: ./start_worker.sh --connect {my_ip}:{self.reverse_listener.port}){C.RS}\n")

            if self.nodes:
                rows = []
                for key, n in self.nodes.items():
                    status = f"{C.G}Connected{C.RS}" if n.is_connected() else f"{C.R}Disconnected{C.RS}"
                    mode = f"{C.CY}Reverse (Outbound){C.RS}" if getattr(n, "is_reverse", False) else "Standard (Inbound)"
                    rows.append([key, n.hostname, mode, status])
                print(tabulate(rows, headers=["Address", "Hostname", "Mode", "Status"],
                               tablefmt="rounded_outline"))
            else:
                print(f"  {C.D}No remote nodes connected yet.{C.RS}")

            print(f"""
  {C.B}1.{C.RS} Scan network (UDP discovery)
  {C.B}2.{C.RS} Add node manually (IP:port)
  {C.B}3.{C.RS} Test connection
  {C.B}4.{C.RS} Remove node
  {C.B}0.{C.RS} Back""")

            c = ask("Select [0-4]:", range(0, 5))
            if c <= 0: return

            if c == 1:
                found = discover_nodes(timeout=5)
                for key, info in found.items():
                    if key not in self.nodes:
                        nc = NodeConnection(info["ip"], info["port"], info["hostname"])
                        try:
                            nc.connect()
                            resp = nc.send_cmd({"cmd": "info"})
                            self.nodes[key] = nc
                            print(f"  {C.OK} Added: {nc.label}")
                            print(f"       CPU: {resp.get('cpu', '?')}, "
                                  f"RAM: {resp.get('ram_total', 0)/(1024**3):.1f} GB, "
                                  f"GPU: {resp.get('gpu') or 'None'}")
                        except Exception as e:
                            print(f"  {C.FL} Cannot connect to {key}: {e}")
                if not found:
                    print(f"  {C.Y}No nodes found. Make sure worker_node.py is running.{C.RS}")
                else:
                    self._save_config()

            elif c == 2:
                addr = ask_str(f"  Enter IP:port (e.g. 192.168.8.50:{DEFAULT_PORT}): ")
                if ":" in addr:
                    ip, port = addr.rsplit(":", 1)
                    port = int(port)
                else:
                    ip = addr
                    port = DEFAULT_PORT
                key = f"{ip}:{port}"
                nc = NodeConnection(ip, port)
                try:
                    nc.connect()
                    resp = nc.send_cmd({"cmd": "info"})
                    nc.hostname = resp.get("hostname", ip)
                    nc.label = f"{nc.hostname} ({ip}:{port})"
                    self.nodes[key] = nc
                    print(f"  {C.OK} Connected: {nc.label}")
                    self._save_config()
                except Exception as e:
                    print(f"  {C.FL} Connection failed: {e}")

            elif c == 3:
                for key, n in self.nodes.items():
                    try:
                        if not n.is_connected():
                            n.connect()
                        resp = n.send_cmd({"cmd": "ping"})
                        print(f"  {C.OK} {n.label}: OK")
                    except Exception as e:
                        print(f"  {C.FL} {n.label}: {e}")

            elif c == 4:
                keys = list(self.nodes.keys())
                for i, k in enumerate(keys, 1):
                    print(f"  {i}. {self.nodes[k].label}")
                ci = ask(f"Remove [1-{len(keys)}]:", range(1, len(keys)+1))
                if 1 <= ci <= len(keys):
                    self.nodes[keys[ci-1]].close()
                    del self.nodes[keys[ci-1]]
                    self._save_config()
                    print(f"  {C.OK} Removed")

    # ── Model Selection ──────────────────────────────────────────────────
    def _model_menu(self):
        downloaded = set()
        try:
            ci = scan_cache_dir(str(CACHE_DIR))
            for r in ci.repos:
                for rev in r.revisions:
                    p = Path(rev.snapshot_path)
                    if list(p.glob("*.safetensors")) or list(p.glob("*.bin")):
                        downloaded.add(r.repo_id)
                        break
        except: pass

        hdr("SELECT A MODEL")
        rows = []
        for i, m in enumerate(MODELS, 1):
            st = f"{C.G}✓{C.RS}" if m["id"] in downloaded else f"{C.D}✗{C.RS}"
            g = f" {C.Y}🔒{C.RS}" if m.get("gated") else ""
            tc = C.MG if m["type"] == "MoE" else C.BL
            rows.append([f"{C.B}{i}{C.RS}", m["name"]+g, f"{tc}{m['type']}{C.RS}",
                         f"~{m['gb']:.1f} GB", st])
        print(tabulate(rows, headers=["#", "Model", "Type", "Size", "DL"],
                       tablefmt="rounded_outline"))
        print(f"  {C.B}{len(MODELS)+1}{C.RS} Load from local folder...")

        c = ask(f"Select [1-{len(MODELS)+1}] (0=back):", range(0, len(MODELS)+2))
        if c <= 0: return

        if c == len(MODELS) + 1:
            folder_path = ask_str("  Enter absolute path to model folder: ")
            folder = Path(folder_path)
            if not folder.exists() or not (folder / "config.json").exists():
                print(f"  {C.FL} Invalid folder or missing config.json")
                return
            
            with open(folder / "config.json") as f:
                config = json.load(f)
            
            # Guess architecture
            arch = config.get("architectures", [""])[0]
            is_moe = "MoE" in arch or config.get("num_experts", 1) > 1 or config.get("num_local_experts", 1) > 1
            m_type = "MoE" if is_moe else "Dense"
            
            self.selected_model = {"id": str(folder), "name": folder.name, "type": m_type}
            self.model_dir = folder
            print(f"  {C.OK} Selected local model: {folder.name}")
        else:
            model = MODELS[c-1]
            self.selected_model = model

            has_weights = False
            if model["id"] in downloaded:
                try:
                    ci = scan_cache_dir(str(CACHE_DIR))
                    for r in ci.repos:
                        if r.repo_id == model["id"]:
                            for rev in sorted(r.revisions, key=lambda x: x.last_modified, reverse=True):
                                p = Path(rev.snapshot_path)
                                if list(p.glob("*.safetensors")) or list(p.glob("*.bin")):
                                    self.model_dir = p
                                    has_weights = True
                                    break
                except: pass

            if has_weights:
                print(f"  {C.OK} Already downloaded.")
            else:
                print(f"  {C.CY}Downloading {model['name']} (resumable)...{C.RS}")
                max_retries = 10
                download_success = False
                for attempt in range(1, max_retries + 1):
                    try:
                        self.model_dir = Path(snapshot_download(
                            model["id"],
                            cache_dir=str(CACHE_DIR),
                            allow_patterns=DOWNLOAD_PATTERNS,
                            max_workers=2,
                            resume_download=True
                        ))
                        print(f"  {C.OK} Download complete!")
                        download_success = True
                        break
                    except Exception as e:
                        if attempt < max_retries:
                            print(f"\n  {C.WN} Network drop ({e}). Automatically resuming download in 3s (attempt {attempt+1}/{max_retries})...")
                            time.sleep(3)
                        else:
                            print(f"\n  {C.FL} Download failed after {max_retries} attempts: {e}")
                            return

        # Select Quantization
        shdr("SELECT QUANTIZATION PRECISION")
        print(f"""
  {C.B}1.{C.RS} Full Precision (bfloat16/float16 — Original weights)
  {C.B}2.{C.RS} 4-bit NF4 Quantization (BitsAndBytes — 75% memory reduction, fast network!)
  {C.B}3.{C.RS} 8-bit Quantization (BitsAndBytes — 50% memory reduction)
  {C.B}4.{C.RS} 8-bit Dynamic Quantization (PyTorch native int8 — CPU friendly)
""")
        qc = ask("Select Quantization [1-4] (default 1):", range(1, 5))
        quant_mode = {1: "none", 2: "4bit", 3: "8bit", 4: "int8"}.get(qc, "none")

        # Load model
        self.dist_model = DistributedModel()
        self.dist_model.load_model(self.selected_model["id"], self.model_dir, quant=quant_mode)
        print(f"  {C.OK} Model ready: {self.dist_model.num_layers} layers, "
              f"{self.dist_model.num_experts} experts/layer (Precision: {quant_mode})")
        
        # Save config
        self._save_config()

    # ── Distribution Config ──────────────────────────────────────────────
    def _distribution_menu(self):
        dm = self.dist_model
        if dm.model is None:
            print(f"\n  {C.Y}Load a model first (option 2).{C.RS}"); return
        if not self.nodes:
            print(f"\n  {C.Y}Add remote nodes first (option 1). "
                  f"Or choose 'Apply' to run fully local.{C.RS}")

        while True:
            hdr(f"CONFIGURE DISTRIBUTION: {self.selected_model['name']}")

            # Show current assignments
            summary = dm.get_assignment_summary()
            rows = []
            for node_label, comps in summary.items():
                comp_str = self._compress_comp_list(comps)
                rows.append([node_label, comp_str])
            print(tabulate(rows, headers=["Node", "Components"],
                           tablefmt="rounded_outline", colalign=("left", "left"),
                           maxcolwidths=[30, 50]))

            # Show cached assignment status if available
            cached_preset = self._get_cached_assignment()
            if cached_preset:
                saved_time = cached_preset.get("saved_at", "")
                summary_lines = cached_preset.get("summary", [])
                cached_desc = " | ".join(summary_lines) if summary_lines else "Saved distribution topology"
                print(f"  {C.CY}💾 Cached Assignment Available ({saved_time}):{C.RS}\n     {C.D}{cached_desc}{C.RS}\n")

            el = "Expert" if dm.is_moe else "FFN/MLP"
            if dm.is_moe:
                print(f"""
  {C.B}1.{C.RS} Assign full layer(s) to remote
  {C.B}2.{C.RS} Assign attention of layer(s) to remote
  {C.B}3.{C.RS} Assign {el} of layer(s) to remote
  {C.B}4.{C.RS} Assign embedding to remote
  {C.B}5.{C.RS} Assign lm_head to remote
  {C.B}6.{C.RS} Assign expert(s) to remote
  {C.B}7.{C.RS} Inspect model parameters & layer sizes
  {C.B}8.{C.RS} Clear all assignments
  {C.B}9.{C.RS} {C.G}Apply & Load{C.RS}  (send components to workers)
  {C.B}L.{C.RS} {C.CY}Restore Cached Assignment{C.RS}
  {C.B}S.{C.RS} Save Current Assignment as Preset
  {C.B}0.{C.RS} Back""")
            else:
                print(f"""
  {C.B}1.{C.RS} Assign full layer(s) to remote
  {C.B}2.{C.RS} Assign attention of layer(s) to remote
  {C.B}3.{C.RS} Assign {el} of layer(s) to remote
  {C.B}4.{C.RS} Assign embedding to remote
  {C.B}5.{C.RS} Assign lm_head to remote
  {C.B}6.{C.RS} Inspect model parameters & layer sizes
  {C.B}7.{C.RS} Clear all assignments
  {C.B}8.{C.RS} {C.G}Apply & Load{C.RS}  (send components to workers)
  {C.B}L.{C.RS} {C.CY}Restore Cached Assignment{C.RS}
  {C.B}S.{C.RS} Save Current Assignment as Preset
  {C.B}0.{C.RS} Back""")

            raw_choice = ask_str("Select option: ").strip().upper()
            if not raw_choice or raw_choice == "0": return

            if raw_choice == "L":
                restored = self._restore_cached_assignment()
                if restored > 0:
                    print(f"  {C.OK} Restored {restored} component assignments from cache!")
                else:
                    print(f"  {C.WN} Could not restore assignments (no matching connected nodes or empty cache).")
                continue

            if raw_choice == "S":
                if not dm.assignments:
                    print(f"  {C.WN} No assignments to save.")
                else:
                    self._save_assignment_cache()
                    print(f"  {C.OK} Current distribution assignment saved to cache preset!")
                continue

            try:
                c = int(raw_choice)
            except ValueError:
                print(f"  {C.R}Invalid choice.{C.RS}")
                continue

            # Handle non-target commands (Apply, Clear, Inspect)
            apply_opt = 9 if dm.is_moe else 8
            clear_opt = 8 if dm.is_moe else 7
            inspect_opt = 7 if dm.is_moe else 6

            if c == apply_opt:
                success = dm.apply_distribution()
                if success and dm.assignments:
                    self._save_assignment_cache()
                    print(f"  {C.OK} Distribution configuration automatically saved to cache.")
                input(f"\n  {C.D}Press Enter to continue...{C.RS}")
                return

            if c == clear_opt:
                dm.assignments.clear()
                print(f"  {C.OK} All assignments cleared.")
                continue

            if c == inspect_opt:
                dm.inspect_model()
                input(f"\n  {C.D}Press Enter to continue...{C.RS}")
                continue

            # Select target node
            node = self._pick_node()
            if node is None: continue

            if c == 1:
                layers = self._parse_layer_range(f"Layer range (0-{dm.num_layers-1}), e.g. '25-29' or '0,5,10': ")
                for li in layers:
                    dm.assign_to_remote(f"layer_{li}", "layer", node, layer_idx=li)
                print(f"  {C.OK} Assigned layers {layers} to {node.label}")

            elif c == 2:
                layers = self._parse_layer_range(f"Layer range (0-{dm.num_layers-1}): ")
                for li in layers:
                    dm.assign_to_remote(f"attn_{li}", "attention", node, layer_idx=li)
                print(f"  {C.OK} Assigned attention of layers {layers} to {node.label}")

            elif c == 3:
                layers = self._parse_layer_range(f"Layer range (0-{dm.num_layers-1}): ")
                for li in layers:
                    dm.assign_to_remote(f"ffn_{li}", "ffn", node, layer_idx=li)
                print(f"  {C.OK} Assigned FFN of layers {layers} to {node.label}")

            elif c == 4:
                dm.assign_to_remote("embedding", "embedding", node)
                print(f"  {C.OK} Assigned embedding to {node.label}")

            elif c == 5:
                dm.assign_to_remote("lm_head", "lm_head", node)
                print(f"  {C.OK} Assigned lm_head to {node.label}")

            elif c == 6 and dm.is_moe:
                li = ask_int(f"Layer index (0-{dm.num_layers-1})", 0)
                experts = self._parse_layer_range(
                    f"Expert indices (0-{dm.num_experts-1}), e.g. '0-7' or '0,1,2': ",
                    max_val=dm.num_experts-1)
                for ei in experts:
                    dm.assign_to_remote(f"expert_{li}_{ei}", "expert", node,
                                        layer_idx=li, expert_idx=ei)
                print(f"  {C.OK} Assigned {len(experts)} experts of layer {li} to {node.label}")

    def _pick_node(self):
        if not self.nodes:
            print(f"  {C.R}No remote nodes. Add one first (Main Menu → option 1).{C.RS}")
            return None
        keys = list(self.nodes.keys())
        print(f"\n  Select target node:")
        for i, k in enumerate(keys, 1):
            print(f"  {C.B}{i}.{C.RS} {self.nodes[k].label}")
        c = ask(f"Node [1-{len(keys)}]:", range(1, len(keys)+1))
        if c < 1: return None
        return self.nodes[keys[c-1]]

    def _parse_layer_range(self, prompt, max_val=None):
        if max_val is None:
            max_val = self.dist_model.num_layers - 1
        raw = ask_str(prompt)
        if not raw: return []
        indices = set()
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                for i in range(int(a), int(b)+1):
                    if 0 <= i <= max_val:
                        indices.add(i)
            else:
                i = int(part)
                if 0 <= i <= max_val:
                    indices.add(i)
        return sorted(indices)

    def _compress_comp_list(self, comps):
        return compress_comp_list(comps)

    # ── Inference ─────────────────────────────────────────────────────────
    def _inference_menu(self):
        dm = self.dist_model
        if not dm.is_distributed and dm.model is None:
            print(f"\n  {C.Y}Configure distribution first (option 3).{C.RS}")
            return
        if not dm.is_distributed:
            # No remote assignments — just apply locally
            dm.is_distributed = True
            if torch.cuda.is_available():
                try:
                    dm.model.to("cuda")
                    dm.local_device = torch.device("cuda")
                except:
                    dm.local_device = torch.device("cpu")

        while True:
            hdr("DISTRIBUTED INFERENCE")
            # Show setup summary
            summary = dm.get_assignment_summary()
            for node_label, comps in summary.items():
                comp_str = self._compress_comp_list(comps)
                print(f"  {C.CY}{node_label}:{C.RS} {comp_str}")

            prof_str = f"{C.G}Enabled{C.RS}" if self.memory_profiling_enabled else f"{C.D}Disabled{C.RS}"
            print(f"""
  {C.B}1.{C.RS} Send prompt
  {C.B}2.{C.RS} Toggle Memory Profiling Graph ({prof_str})
  {C.B}0.{C.RS} Back""")

            c = ask("Select [0-2]:", range(0, 3))
            if c <= 0: return

            if c == 2:
                self.memory_profiling_enabled = not self.memory_profiling_enabled
                continue

            prompt = ask_str(f"\n  {C.B}Enter prompt:{C.RS} ")
            if not prompt: continue

            max_tok = ask_int("Max new tokens", 128)
            temp = 0.0
            try:
                raw = input(f"  Temperature [{0.0}]: ").strip()
                if raw: temp = float(raw)
            except: pass

            try:
                if self.memory_profiling_enabled:
                    dm.start_profiling()
                dm.generate(prompt, max_tok, temp)
            except Exception as e:
                print(f"\n  {C.FL} Inference error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if self.memory_profiling_enabled:
                    dm.stop_profiling()

            input(f"\n  {C.D}Press Enter to continue...{C.RS}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        cli = CLI()
        cli.run()
    except KeyboardInterrupt:
        print(f"\n\n  {C.Y}Interrupted. Goodbye!{C.RS}\n")
        sys.exit(0)
