#!/usr/bin/env python3
"""
LoadTime Benchmark — Interactive CLI for measuring model loading & inference performance.

Supports multiple models from 135M to 8B (Dense & MoE).
Measures:
  - Load times: Disk→RAM, Disk→VRAM, RAM→VRAM (whole model / layer / expert)
  - Inference:  Prefill, decode, TTFT, throughput, KV cache memory
"""

import os
import sys
import gc
import json
import time
import statistics
import re
from pathlib import Path
from collections import defaultdict

import torch
import psutil
from safetensors import safe_open
from huggingface_hub import snapshot_download, scan_cache_dir
from tabulate import tabulate

# ═══════════════════════════════════════════════════════════════════════════════
# ANSI Colors
# ═══════════════════════════════════════════════════════════════════════════════
class C:
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    WHITE   = "\033[97m"
    RESET   = "\033[0m"
    HEADER  = "\033[1;96m"
    OK      = "\033[92m✓\033[0m"
    FAIL    = "\033[91m✗\033[0m"
    WARN    = "\033[93m⚠\033[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# Model Catalog
# ═══════════════════════════════════════════════════════════════════════════════
MODELS = [
    {
        "id": "HuggingFaceTB/SmolLM2-135M",
        "name": "SmolLM2 135M",
        "type": "Dense",
        "approx_size_gb": 0.27,
    },
    {
        "id": "HuggingFaceTB/SmolLM2-360M",
        "name": "SmolLM2 360M",
        "type": "Dense",
        "approx_size_gb": 0.72,
    },
    {
        "id": "Qwen/Qwen2.5-0.5B",
        "name": "Qwen2.5 0.5B",
        "type": "Dense",
        "approx_size_gb": 0.99,
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama 1.1B",
        "type": "Dense",
        "approx_size_gb": 2.20,
    },
    {
        "id": "Qwen/Qwen2.5-1.5B",
        "name": "Qwen2.5 1.5B",
        "type": "Dense",
        "approx_size_gb": 3.09,
    },
    {
        "id": "microsoft/phi-2",
        "name": "Phi-2 2.7B",
        "type": "Dense",
        "approx_size_gb": 5.56,
    },
    {
        "id": "Qwen/Qwen2.5-3B",
        "name": "Qwen2.5 3B",
        "type": "Dense",
        "approx_size_gb": 6.39,
    },
    {
        "id": "allenai/OLMoE-1B-7B-0924",
        "name": "OLMoE 1B/7B",
        "type": "MoE",
        "approx_size_gb": 12.89,
    },
    {
        "id": "Qwen/Qwen2.5-7B",
        "name": "Qwen2.5 7B",
        "type": "Dense",
        "approx_size_gb": 15.23,
    },
    {
        "id": "meta-llama/Llama-3.1-8B",
        "name": "Llama 3.1 8B",
        "type": "Dense",
        "approx_size_gb": 16.07,
        "gated": True,
    },
]

CACHE_DIR = Path("/media/vithurshan/vithu/llm/openGP/LoadTime/model_cache")

# Download patterns — includes tokenizer files for inference
DOWNLOAD_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.model",          # sentencepiece
    "*.tiktoken",       # tiktoken
    "*.txt",            # vocab/merges
    "tokenizer*",       # catch-all for tokenizer files
]


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════
def fmt_bytes(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def fmt_bw(size_bytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "∞"
    return f"{(size_bytes / (1024**3)) / seconds:.2f} GB/s"


def print_header(text: str):
    width = 60
    print(f"\n{C.HEADER}{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}{C.RESET}")


def print_subheader(text: str):
    print(f"\n  {C.CYAN}── {text} ──{C.RESET}")


def prompt_choice(prompt: str, valid_range: range) -> int:
    while True:
        try:
            raw = input(f"\n  {C.BOLD}{prompt}{C.RESET} ").strip()
            if raw == "":
                return -1
            val = int(raw)
            if val in valid_range:
                return val
            print(f"  {C.RED}Invalid. Enter {valid_range.start}-{valid_range.stop - 1}.{C.RESET}")
        except ValueError:
            print(f"  {C.RED}Please enter a number.{C.RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0


def prompt_int(prompt: str, default: int) -> int:
    try:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return int(raw) if raw else default
    except (ValueError, EOFError, KeyboardInterrupt):
        return default


def prompt_float(prompt: str, default: float) -> float:
    try:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return float(raw) if raw else default
    except (ValueError, EOFError, KeyboardInterrupt):
        return default


def prompt_string(prompt: str, default: str = "") -> str:
    try:
        raw = input(f"  {prompt}").strip()
        return raw if raw else default
    except (EOFError, KeyboardInterrupt):
        return default


def drop_caches():
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except PermissionError:
        os.system("sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null")


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_downloaded_models() -> set[str]:
    try:
        cache_info = scan_cache_dir(str(CACHE_DIR))
        return {repo.repo_id for repo in cache_info.repos}
    except Exception:
        return set()


def find_model_dir(model_id: str) -> Path | None:
    try:
        cache_info = scan_cache_dir(str(CACHE_DIR))
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                for rev in sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True):
                    return rev.snapshot_path
    except Exception:
        pass
    return None


def get_gpu_vram_free() -> float:
    """Get free VRAM in bytes."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Model Analyzer
# ═══════════════════════════════════════════════════════════════════════════════
class ModelAnalyzer:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.config = self._load_config()
        self.weight_map = self._load_weight_map()
        self.tensor_sizes = self._compute_tensor_sizes()
        self.is_moe = self._detect_moe()
        self.num_layers = self._get_num_layers()
        self.num_experts = self._get_num_experts()

    def _load_config(self) -> dict:
        with open(self.model_dir / "config.json") as f:
            return json.load(f)

    def _load_weight_map(self) -> dict[str, str]:
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)["weight_map"]
        single = self.model_dir / "model.safetensors"
        if single.exists():
            with safe_open(str(single), framework="pt", device="cpu") as f:
                return {key: "model.safetensors" for key in f.keys()}
        raise FileNotFoundError("No safetensors files found!")

    def _compute_tensor_sizes(self) -> dict[str, int]:
        sizes = {}
        for shard_file in sorted(set(self.weight_map.values())):
            shard_path = self.model_dir / shard_file
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    t = f.get_tensor(key)
                    sizes[key] = t.numel() * t.element_size()
        return sizes

    def _detect_moe(self) -> bool:
        for key in ("num_experts", "num_local_experts"):
            if self.config.get(key, 0) > 1:
                return True
        return any("experts." in k for k in self.weight_map)

    def _get_num_layers(self) -> int:
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            if key in self.config:
                return self.config[key]
        layer_indices = set()
        for name in self.weight_map:
            m = re.search(r"layers\.(\d+)\.", name)
            if m:
                layer_indices.add(int(m.group(1)))
        return max(layer_indices) + 1 if layer_indices else 0

    def _get_num_experts(self) -> int:
        if not self.is_moe:
            return 1
        for key in ("num_experts", "num_local_experts"):
            if key in self.config:
                return self.config[key]
        expert_indices = set()
        for name in self.weight_map:
            m = re.search(r"layers\.0\..*experts\.(\d+)\.", name)
            if m:
                expert_indices.add(int(m.group(1)))
        return len(expert_indices) or 1

    def get_shard_files(self) -> list[str]:
        return sorted(set(self.weight_map.values()))

    def get_layer_keys(self, layer_idx: int) -> list[str]:
        pattern = f"layers.{layer_idx}."
        return [k for k in self.weight_map if pattern in k]

    def get_expert_keys(self, layer_idx: int, expert_idx: int) -> list[str]:
        if self.is_moe:
            pattern = f"layers.{layer_idx}.mlp.experts.{expert_idx}."
            return [k for k in self.weight_map if pattern in k]
        return [k for k in self.weight_map
                if f"layers.{layer_idx}." in k and "mlp" in k.lower()]

    def get_shard_files_for_keys(self, keys: list[str]) -> list[str]:
        return sorted(set(self.weight_map[k] for k in keys if k in self.weight_map))

    def get_total_size(self) -> int:
        return sum(self.tensor_sizes.values())

    def get_keys_size(self, keys: list[str]) -> int:
        return sum(self.tensor_sizes.get(k, 0) for k in keys)

    def get_hidden_size(self) -> int:
        return self.config.get("hidden_size", self.config.get("n_embd", 0))

    def get_num_kv_heads(self) -> int:
        return self.config.get("num_key_value_heads",
                               self.config.get("num_attention_heads",
                                               self.config.get("n_head", 0)))

    def get_head_dim(self) -> int:
        hs = self.get_hidden_size()
        nh = self.config.get("num_attention_heads", self.config.get("n_head", 1))
        return hs // nh if nh else 0

    def get_dtype_size(self) -> int:
        """Bytes per element for model's storage dtype."""
        dtype_str = self.config.get("torch_dtype", "float16")
        return {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype_str, 2)

    def compute_kv_cache_bytes(self, seq_len: int) -> int:
        """Theoretical KV cache size for a given sequence length."""
        # KV cache = 2 (K,V) × num_layers × num_kv_heads × head_dim × seq_len × dtype_size
        return (2 * self.num_layers * self.get_num_kv_heads()
                * self.get_head_dim() * seq_len * self.get_dtype_size())

    def compute_kv_cache_per_token(self) -> int:
        """KV cache bytes per token."""
        return self.compute_kv_cache_bytes(1)

    # ── Display ───────────────────────────────────────────────────────────
    def display_config(self):
        config = self.config
        hidden_size   = config.get("hidden_size", config.get("n_embd", "?"))
        inter_size    = config.get("intermediate_size", config.get("n_inner", "?"))
        num_heads     = config.get("num_attention_heads", config.get("n_head", "?"))
        num_kv_heads  = config.get("num_key_value_heads", config.get("n_head", num_heads))
        head_dim      = (hidden_size // num_heads) if isinstance(hidden_size, int) and isinstance(num_heads, int) else "?"
        vocab_size    = config.get("vocab_size", "?")
        max_pos       = config.get("max_position_embeddings", config.get("n_positions", "?"))
        act_fn        = config.get("hidden_act", config.get("activation_function", "?"))
        dtype         = config.get("torch_dtype", "?")
        arch          = config.get("architectures", ["?"])[0] if "architectures" in config else config.get("model_type", "?")
        num_active    = config.get("num_experts_per_tok", 1)

        print_header("MODEL CONFIGURATION")
        rows = [
            ["Architecture", arch],
            ["Model Type", config.get("model_type", "?")],
            ["Data Type", dtype],
            ["", ""],
            ["Vocabulary Size", f"{vocab_size:,}" if isinstance(vocab_size, int) else vocab_size],
            ["Max Position Embeddings", f"{max_pos:,}" if isinstance(max_pos, int) else max_pos],
            ["Hidden Size (d_model)", f"{hidden_size:,}" if isinstance(hidden_size, int) else hidden_size],
            ["Intermediate Size (d_ff)", f"{inter_size:,}" if isinstance(inter_size, int) else inter_size],
            ["Activation Function", act_fn],
            ["", ""],
            ["Num Hidden Layers", self.num_layers],
            ["Num Attention Heads (Q)", num_heads],
            ["Num KV Heads", num_kv_heads],
            ["Head Dimension", head_dim],
            ["", ""],
        ]
        if self.is_moe:
            rows += [
                [f"{C.MAGENTA}MoE: Experts/Layer{C.RESET}", self.num_experts],
                [f"{C.MAGENTA}MoE: Active Experts/Token{C.RESET}", num_active],
                ["", ""],
            ]
        else:
            rows.append(["Model Type", "Dense (no MoE)"])
            rows.append(["", ""])

        total_params = sum(self.tensor_sizes.values()) // self.get_dtype_size()
        rows.append([f"{C.BOLD}Total Parameters{C.RESET}", f"{total_params:,}"])
        kv_per_token = self.compute_kv_cache_per_token()
        rows.append([f"{C.BOLD}KV Cache / Token{C.RESET}", fmt_bytes(kv_per_token)])

        print(tabulate(rows, headers=["Property", "Value"],
                       tablefmt="rounded_outline", colalign=("left", "right")))

        self._display_size_breakdown()
        self._display_component_details()

    def _display_size_breakdown(self):
        print_header("SIZE BREAKDOWN")

        categories = defaultdict(int)
        layer_sizes = defaultdict(lambda: defaultdict(int))
        expert_detail = defaultdict(lambda: defaultdict(int))

        for name, size in self.tensor_sizes.items():
            if "embed" in name.lower():
                categories["Embedding"] += size
            elif "lm_head" in name.lower():
                categories["LM Head"] += size
            elif re.search(r"^model\.norm", name) or (name.endswith(".norm.weight") and "layers" not in name):
                categories["Final Norm"] += size
            elif "layers." in name:
                m = re.search(r"layers\.(\d+)\.", name)
                if not m:
                    categories["Other"] += size
                    continue
                li = int(m.group(1))
                if "self_attn" in name or "input_layernorm" in name or "q_norm" in name or "k_norm" in name:
                    layer_sizes[li]["Attention"] += size
                elif "experts." in name:
                    layer_sizes[li]["Experts (All)"] += size
                    em = re.search(r"experts\.(\d+)\.", name)
                    if em:
                        expert_detail[li][int(em.group(1))] += size
                elif "mlp.gate." in name or "block_sparse_moe.gate" in name:
                    layer_sizes[li]["Router"] += size
                elif "mlp" in name.lower():
                    layer_sizes[li]["FFN/MLP"] += size
                elif "norm" in name.lower():
                    layer_sizes[li]["LayerNorm"] += size
                else:
                    layer_sizes[li]["Other"] += size
            else:
                categories["Other"] += size

        total_size = self.get_total_size()
        l0 = layer_sizes.get(0, {})
        l0_total = sum(l0.values())

        rows = [
            [f"{C.CYAN}Embedding{C.RESET}", fmt_bytes(categories.get("Embedding", 0)),
             f"{categories.get('Embedding', 0) / total_size * 100:.1f}%"],
            [f"{C.CYAN}LM Head{C.RESET}", fmt_bytes(categories.get("LM Head", 0)),
             f"{categories.get('LM Head', 0) / total_size * 100:.1f}%"],
            [f"{C.CYAN}Final Norm{C.RESET}", fmt_bytes(categories.get("Final Norm", 0)),
             f"{categories.get('Final Norm', 0) / total_size * 100:.1f}%"],
            ["", "", ""],
            [f"{C.BOLD}── Per Layer (×{self.num_layers}) ──{C.RESET}", "", ""],
            [f"  Attention (Q+K+V+O + norms)", fmt_bytes(l0.get("Attention", 0)),
             f"{l0.get('Attention', 0) / l0_total * 100:.1f}% of layer" if l0_total else ""],
        ]

        if self.is_moe:
            single_expert_size = list(expert_detail.get(0, {0: 0}).values())[0] if expert_detail.get(0) else 0
            num_active = self.config.get("num_experts_per_tok", 1)
            rows += [
                [f"  Router/Gate", fmt_bytes(l0.get("Router", 0)),
                 f"{l0.get('Router', 0) / l0_total * 100:.1f}% of layer" if l0_total else ""],
                [f"  All {self.num_experts} Experts", fmt_bytes(l0.get("Experts (All)", 0)),
                 f"{l0.get('Experts (All)', 0) / l0_total * 100:.1f}% of layer" if l0_total else ""],
                [f"    └─ {C.MAGENTA}Single Expert{C.RESET}", fmt_bytes(single_expert_size), ""],
                [f"    └─ {C.MAGENTA}{num_active} Active Experts{C.RESET}",
                 fmt_bytes(single_expert_size * num_active), ""],
            ]
        else:
            rows.append([f"  FFN/MLP", fmt_bytes(l0.get("FFN/MLP", 0)),
                         f"{l0.get('FFN/MLP', 0) / l0_total * 100:.1f}% of layer" if l0_total else ""])

        if l0.get("LayerNorm", 0):
            rows.append([f"  LayerNorm", fmt_bytes(l0.get("LayerNorm", 0)),
                         f"{l0.get('LayerNorm', 0) / l0_total * 100:.1f}% of layer" if l0_total else ""])

        rows += [
            [f"  {C.BOLD}Layer Total{C.RESET}", f"{C.BOLD}{fmt_bytes(l0_total)}{C.RESET}", ""],
            ["", "", ""],
            [f"{C.BOLD}All {self.num_layers} Layers{C.RESET}",
             fmt_bytes(l0_total * self.num_layers),
             f"{l0_total * self.num_layers / total_size * 100:.1f}%"],
            ["", "", ""],
            [f"{C.GREEN}{C.BOLD}═══ TOTAL MODEL ═══{C.RESET}",
             f"{C.GREEN}{C.BOLD}{fmt_bytes(total_size)}{C.RESET}", "100%"],
        ]
        print(tabulate(rows, headers=["Component", "Size", "Fraction"],
                       tablefmt="rounded_outline", colalign=("left", "right", "right")))

    def _display_component_details(self):
        if self.is_moe:
            label = f"EXPERT WEIGHTS (each of {self.num_experts} experts)"
            detail_keys = self.get_expert_keys(0, 0)
        else:
            label = "FFN / MLP WEIGHTS (per layer)"
            detail_keys = [k for k in self.weight_map if "layers.0." in k and "mlp" in k.lower()]

        print_subheader(label)
        rows = []
        for name in sorted(detail_keys):
            short = re.sub(r"model\.layers\.0\.(mlp\.)?(experts\.0\.)?", "", name)
            size = self.tensor_sizes.get(name, 0)
            rows.append([short, f"{size // self.get_dtype_size():,}", fmt_bytes(size)])
        if rows:
            print(tabulate(rows, headers=["Weight", "Parameters", "Size"],
                           tablefmt="rounded_outline", colalign=("left", "right", "right")))

        print_subheader("ATTENTION WEIGHTS (per layer)")
        attn_keys = [k for k in self.weight_map
                     if "layers.0." in k and ("self_attn" in k or "input_layernorm" in k
                                               or "q_norm" in k or "k_norm" in k)]
        rows = []
        for name in sorted(attn_keys):
            short = re.sub(r"model\.layers\.0\.", "", name)
            size = self.tensor_sizes.get(name, 0)
            rows.append([short, f"{size // self.get_dtype_size():,}", fmt_bytes(size)])
        if rows:
            print(tabulate(rows, headers=["Weight", "Parameters", "Size"],
                           tablefmt="rounded_outline", colalign=("left", "right", "right")))


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Engine (Load Time)
# ═══════════════════════════════════════════════════════════════════════════════
class BenchmarkEngine:
    def __init__(self, analyzer: ModelAnalyzer):
        self.analyzer = analyzer
        self.model_dir = analyzer.model_dir
        self.results: list[dict] = []

    def run(self, component: str, paths: list[str],
            num_warmup: int, num_runs: int,
            layer_idx: int = 0, expert_idx: int = 0) -> list[dict]:
        if component == "expert":
            keys = self.analyzer.get_expert_keys(layer_idx, expert_idx)
            label = (f"Expert [{layer_idx}][{expert_idx}]" if self.analyzer.is_moe
                     else f"FFN/MLP [layer {layer_idx}]")
        elif component == "layer":
            keys = self.analyzer.get_layer_keys(layer_idx)
            label = f"Layer [{layer_idx}]"
        elif component == "model":
            keys = None
            label = "Whole Model"
        else:
            raise ValueError(f"Unknown component: {component}")

        shard_files = (self.analyzer.get_shard_files_for_keys(keys)
                       if keys else self.analyzer.get_shard_files())
        data_size = self.analyzer.get_keys_size(keys) if keys else self.analyzer.get_total_size()

        print(f"\n  {C.CYAN}Component:{C.RESET} {label}  |  "
              f"{C.CYAN}Size:{C.RESET} {fmt_bytes(data_size)}  |  "
              f"{C.CYAN}Tensors:{C.RESET} {len(keys) if keys else len(self.analyzer.weight_map)}  |  "
              f"{C.CYAN}Runs:{C.RESET} {num_warmup}w+{num_runs}t")

        results = []
        for path_name in paths:
            print(f"  {C.YELLOW}▶ {path_name} ...{C.RESET}", end="", flush=True)
            try:
                fn = {"Disk → RAM": self._d2r, "Disk → VRAM": self._d2v, "RAM → VRAM": self._r2v}[path_name]
                r = fn(shard_files, keys, label, num_warmup, num_runs)
                r["size_bytes"] = data_size
                results.append(r)
                print(f"\r  {C.OK} {path_name:14s} → {C.GREEN}{fmt_time(r['mean']):>10s}{C.RESET}  "
                      f"({fmt_bw(data_size, r['mean'])})")
            except torch.cuda.OutOfMemoryError:
                print(f"\r  {C.WARN} {path_name:14s} → {C.RED}OUT OF MEMORY{C.RESET}")
                results.append({"label": label, "path": path_name, "size_bytes": data_size,
                                "times": [], "mean": float("nan"), "std": 0})
                clear_gpu()

        self.results.extend(results)
        return results

    def _d2r(self, shard_files, tensor_keys, label, nw, nr):
        times, total_bytes = [], 0
        for i in range(nw + nr):
            drop_caches(); gc.collect()
            t0 = time.perf_counter()
            tensors = {}
            for sf in shard_files:
                with safe_open(str(self.model_dir / sf), framework="pt", device="cpu") as f:
                    lk = f.keys() if tensor_keys is None else [k for k in tensor_keys if k in f.keys()]
                    for key in lk:
                        tensors[key] = f.get_tensor(key)
            elapsed = time.perf_counter() - t0
            if i >= nw: times.append(elapsed)
            total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            del tensors; gc.collect()
        return {"label": label, "path": "Disk → RAM", "times": times,
                "mean": statistics.mean(times), "std": statistics.stdev(times) if len(times) > 1 else 0}

    def _d2v(self, shard_files, tensor_keys, label, nw, nr):
        times = []
        for i in range(nw + nr):
            drop_caches(); clear_gpu()
            t0 = time.perf_counter()
            tensors = {}
            for sf in shard_files:
                with safe_open(str(self.model_dir / sf), framework="pt", device="cuda:0") as f:
                    lk = f.keys() if tensor_keys is None else [k for k in tensor_keys if k in f.keys()]
                    for key in lk:
                        tensors[key] = f.get_tensor(key)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if i >= nw: times.append(elapsed)
            del tensors; clear_gpu()
        return {"label": label, "path": "Disk → VRAM", "times": times,
                "mean": statistics.mean(times), "std": statistics.stdev(times) if len(times) > 1 else 0}

    def _r2v(self, shard_files, tensor_keys, label, nw, nr):
        cpu = {}
        for sf in shard_files:
            with safe_open(str(self.model_dir / sf), framework="pt", device="cpu") as f:
                lk = f.keys() if tensor_keys is None else [k for k in tensor_keys if k in f.keys()]
                for key in lk:
                    cpu[key] = f.get_tensor(key)
        times = []
        for i in range(nw + nr):
            clear_gpu()
            t0 = time.perf_counter()
            gpu = {k: v.to("cuda:0", non_blocking=False) for k, v in cpu.items()}
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if i >= nw: times.append(elapsed)
            del gpu; clear_gpu()
        del cpu; gc.collect()
        return {"label": label, "path": "RAM → VRAM", "times": times,
                "mean": statistics.mean(times), "std": statistics.stdev(times) if len(times) > 1 else 0}

    def display_results(self, results: list[dict] | None = None):
        results = results or self.results
        if not results:
            print(f"\n  {C.YELLOW}No results.{C.RESET}")
            return
        print_header("BENCHMARK RESULTS")
        rows = []
        for r in results:
            if r["mean"] != r["mean"]:
                rows.append([r["label"], r["path"], fmt_bytes(r["size_bytes"]),
                             f"{C.RED}OOM{C.RESET}", "—", "—"])
            else:
                rows.append([r["label"], r["path"], fmt_bytes(r["size_bytes"]),
                             f"{C.GREEN}{fmt_time(r['mean'])}{C.RESET}",
                             fmt_time(r["std"]), fmt_bw(r["size_bytes"], r["mean"])])
        print(tabulate(rows, headers=["Component", "Path", "Size", "Mean", "StdDev", "Bandwidth"],
                       tablefmt="rounded_outline", colalign=("left", "center", "right", "right", "right", "right")))

        valid = [r for r in results if r["mean"] == r["mean"]]
        if len(valid) > 1:
            print_subheader("VISUAL COMPARISON")
            mx = max(r["mean"] for r in valid)
            for r in valid:
                bar = "█" * int((r["mean"] / mx) * 35)
                print(f"  {r['label']:25s} {r['path']:14s} {fmt_time(r['mean']):>10s}  {C.CYAN}{bar}{C.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════
class InferenceEngine:
    def __init__(self, model_id: str, model_dir: Path, analyzer: ModelAnalyzer):
        self.model_id = model_id
        self.model_dir = model_dir
        self.analyzer = analyzer
        self.model = None
        self.tokenizer = None
        self.device_label = None   # "cpu", "cuda", "auto"
        self.model_mem = 0         # VRAM/RAM used by model after loading

    def is_loaded(self) -> bool:
        return self.model is not None

    def status_str(self) -> str:
        if not self.is_loaded():
            return f"{C.DIM}Not loaded{C.RESET}"
        return f"{C.GREEN}Loaded on {self.device_label}{C.RESET} ({fmt_bytes(self.model_mem)})"

    def load_model(self, device: str):
        """Load model & tokenizer. device = 'cpu', 'cuda', or 'auto'."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.is_loaded():
            print(f"  {C.YELLOW}Model already loaded. Unload first.{C.RESET}")
            return

        self.device_label = device

        # ── Load tokenizer ────────────────────────────────────────────────
        print(f"  {C.CYAN}Loading tokenizer...{C.RESET}", end="", flush=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        except Exception:
            print(f" (fetching from hub)", end="", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=str(CACHE_DIR))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"\r  {C.OK} Tokenizer loaded (vocab: {len(self.tokenizer):,})")

        # ── Determine dtype ───────────────────────────────────────────────
        dtype_str = self.analyzer.config.get("torch_dtype", "float16")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}.get(dtype_str, torch.float16)

        # ── Load model ────────────────────────────────────────────────────
        model_size = self.analyzer.get_total_size()
        print(f"  {C.CYAN}Loading model ({fmt_bytes(model_size)}) to {device}...{C.RESET}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_before = torch.cuda.memory_allocated()

        t_start = time.perf_counter()
        ram_before = psutil.Process().memory_info().rss

        try:
            if device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_dir), torch_dtype=dtype, device_map="auto")
            elif device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_dir), torch_dtype=dtype).to("cuda:0")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_dir), torch_dtype=dtype)

            self.model.eval()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        except torch.cuda.OutOfMemoryError:
            print(f"  {C.FAIL} {C.RED}OUT OF MEMORY! Model too large for GPU.{C.RESET}")
            print(f"  {C.YELLOW}Try 'auto' (splits across GPU+CPU) or 'cpu'.{C.RESET}")
            self.model = None
            self.tokenizer = None
            clear_gpu()
            return
        except Exception as e:
            print(f"  {C.FAIL} {C.RED}Load failed: {e}{C.RESET}")
            self.model = None
            self.tokenizer = None
            return

        t_load = time.perf_counter() - t_start
        ram_after = psutil.Process().memory_info().rss
        ram_used = ram_after - ram_before

        if torch.cuda.is_available():
            vram_after = torch.cuda.memory_allocated()
            vram_used = vram_after - vram_before
            self.model_mem = vram_used if device != "cpu" else ram_used
        else:
            self.model_mem = ram_used

        print(f"  {C.OK} Model loaded in {C.GREEN}{fmt_time(t_load)}{C.RESET}")
        rows = [
            ["Load time", fmt_time(t_load)],
            ["RAM consumed", fmt_bytes(ram_used)],
        ]
        if torch.cuda.is_available() and device != "cpu":
            rows.append(["VRAM consumed", fmt_bytes(vram_used)])
            rows.append(["VRAM free", fmt_bytes(get_gpu_vram_free())])
        print(tabulate(rows, tablefmt="rounded_outline", colalign=("left", "right")))

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.model_mem = 0
        self.device_label = None
        gc.collect()
        clear_gpu()
        print(f"  {C.OK} Model unloaded.")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> dict:
        """
        Manual autoregressive generation with detailed statistics.
        Returns a stats dict.
        """
        if not self.is_loaded():
            print(f"  {C.RED}No model loaded!{C.RESET}")
            return {}

        model = self.model
        tokenizer = self.tokenizer

        # Determine device for input tensors
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        on_cuda = device.type == "cuda"

        # ── Tokenize ──────────────────────────────────────────────────────
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        num_input_tokens = input_ids.shape[1]

        print(f"\n  {C.DIM}Input tokens: {num_input_tokens}{C.RESET}")
        print(f"\n  {C.BOLD}Prompt:{C.RESET} {C.CYAN}{prompt}{C.RESET}")
        print(f"  {C.BOLD}Response:{C.RESET} {C.GREEN}", end="", flush=True)

        # ── Memory baseline ───────────────────────────────────────────────
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            vram_before_prefill = torch.cuda.memory_allocated()
        ram_before = psutil.Process().memory_info().rss

        # ── Prefill (forward pass on full input, produces first token) ────
        t_prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
        if on_cuda:
            torch.cuda.synchronize()
        t_prefill_end = time.perf_counter()
        ttft = t_prefill_end - t_prefill_start

        if on_cuda:
            vram_after_prefill = torch.cuda.memory_allocated()

        # Sample first token
        next_logits = outputs.logits[:, -1, :]
        next_token = self._sample(next_logits, temperature)
        generated_ids = [next_token.item()]

        # Print first token
        first_tok_str = tokenizer.decode([next_token.item()], skip_special_tokens=True)
        print(first_tok_str, end="", flush=True)

        # ── Decode loop ───────────────────────────────────────────────────
        token_times = []
        eos_id = tokenizer.eos_token_id

        for _ in range(max_new_tokens - 1):
            t_tok_start = time.perf_counter()
            with torch.no_grad():
                tok_input = next_token.view(1, 1) if next_token.dim() == 0 else next_token
                if tok_input.dim() == 1:
                    tok_input = tok_input.unsqueeze(0)
                outputs = model(tok_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            if on_cuda:
                torch.cuda.synchronize()

            next_logits = outputs.logits[:, -1, :]
            next_token = self._sample(next_logits, temperature)
            t_tok_end = time.perf_counter()

            token_id = next_token.item()
            generated_ids.append(token_id)
            token_times.append(t_tok_end - t_tok_start)

            # Stream token
            tok_str = tokenizer.decode([token_id], skip_special_tokens=True)
            print(tok_str, end="", flush=True)

            if token_id == eos_id:
                break

        print(f"{C.RESET}\n")  # End green text

        # ── Memory after generation ───────────────────────────────────────
        if on_cuda:
            torch.cuda.synchronize()
            vram_after_gen = torch.cuda.memory_allocated()
            vram_peak = torch.cuda.max_memory_allocated()
        ram_after = psutil.Process().memory_info().rss

        # ── Compute statistics ────────────────────────────────────────────
        num_output_tokens = len(generated_ids)
        total_seq_len = num_input_tokens + num_output_tokens
        decode_time = sum(token_times)
        total_time = ttft + decode_time

        # KV cache
        kv_theoretical = self.analyzer.compute_kv_cache_bytes(total_seq_len)
        kv_measured = (vram_after_gen - vram_before_prefill) if on_cuda else 0

        # Throughput
        prefill_tps = num_input_tokens / ttft if ttft > 0 else 0
        decode_tps = (num_output_tokens - 1) / decode_time if decode_time > 0 and num_output_tokens > 1 else 0
        overall_tps = num_output_tokens / total_time if total_time > 0 else 0

        # Per-token decode stats
        if token_times:
            avg_tok = statistics.mean(token_times)
            min_tok = min(token_times)
            max_tok = max(token_times)
        else:
            avg_tok = min_tok = max_tok = 0

        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        stats = {
            "prompt": prompt,
            "response": response_text,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "total_seq_len": total_seq_len,
            "ttft": ttft,
            "decode_time": decode_time,
            "total_time": total_time,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "overall_tps": overall_tps,
            "avg_token_time": avg_tok,
            "min_token_time": min_tok,
            "max_token_time": max_tok,
            "token_times": token_times,
            "kv_cache_theoretical": kv_theoretical,
            "kv_cache_measured": kv_measured,
            "ram_delta": ram_after - ram_before,
        }

        if on_cuda:
            stats.update({
                "vram_before": vram_before_prefill,
                "vram_after_prefill": vram_after_prefill,
                "vram_after_gen": vram_after_gen,
                "vram_peak": vram_peak,
                "vram_kv_prefill": vram_after_prefill - vram_before_prefill,
            })

        # ── Display stats ─────────────────────────────────────────────────
        self._display_stats(stats, on_cuda)

        # Clean up KV cache
        del past_key_values, outputs
        gc.collect()
        if on_cuda:
            clear_gpu()

        return stats

    def _sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0 or temperature < 1e-7:
            return torch.argmax(logits, dim=-1)
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _display_stats(self, s: dict, on_cuda: bool):
        print_header("INFERENCE STATISTICS")

        # ── Token Counts ──────────────────────────────────────────────────
        print_subheader("TOKENS")
        tok_rows = [
            ["Input tokens (prompt)", f"{s['num_input_tokens']:,}"],
            ["Output tokens (generated)", f"{s['num_output_tokens']:,}"],
            ["Total sequence length", f"{s['total_seq_len']:,}"],
        ]
        print(tabulate(tok_rows, tablefmt="rounded_outline", colalign=("left", "right")))

        # ── Timing ────────────────────────────────────────────────────────
        print_subheader("TIMING")
        time_rows = [
            [f"{C.BOLD}Time to First Token (TTFT){C.RESET}", f"{C.GREEN}{fmt_time(s['ttft'])}{C.RESET}"],
            ["Decode time (remaining tokens)", fmt_time(s["decode_time"])],
            [f"{C.BOLD}Total generation time{C.RESET}", f"{C.BOLD}{fmt_time(s['total_time'])}{C.RESET}"],
            ["", ""],
            ["Prefill speed", f"{s['prefill_tps']:.1f} tok/s"],
            [f"{C.BOLD}Decode speed{C.RESET}", f"{C.GREEN}{s['decode_tps']:.1f} tok/s{C.RESET}"],
            ["Overall throughput", f"{s['overall_tps']:.1f} tok/s"],
            ["", ""],
            ["Per-token decode (avg)", fmt_time(s["avg_token_time"])],
            ["Per-token decode (min)", fmt_time(s["min_token_time"])],
            ["Per-token decode (max)", fmt_time(s["max_token_time"])],
        ]
        print(tabulate(time_rows, tablefmt="rounded_outline", colalign=("left", "right")))

        # ── Memory ────────────────────────────────────────────────────────
        print_subheader("MEMORY")
        mem_rows = [
            ["KV cache per token (theoretical)", fmt_bytes(self.analyzer.compute_kv_cache_per_token())],
            [f"KV cache total (theoretical, {s['total_seq_len']} tokens)", fmt_bytes(s["kv_cache_theoretical"])],
        ]

        if on_cuda:
            mem_rows += [
                ["", ""],
                [f"{C.BOLD}KV cache (measured VRAM delta){C.RESET}",
                 f"{C.GREEN}{fmt_bytes(s['kv_cache_measured'])}{C.RESET}"],
                ["VRAM after prefill (KV for input)", fmt_bytes(s["vram_kv_prefill"])],
                ["VRAM before generation", fmt_bytes(s["vram_before"])],
                ["VRAM after generation", fmt_bytes(s["vram_after_gen"])],
                [f"{C.BOLD}VRAM peak{C.RESET}", f"{C.YELLOW}{fmt_bytes(s['vram_peak'])}{C.RESET}"],
                ["VRAM overhead (peak - baseline)", fmt_bytes(s["vram_peak"] - s["vram_before"])],
            ]
        else:
            mem_rows.append(["RAM delta", fmt_bytes(s["ram_delta"])])

        print(tabulate(mem_rows, tablefmt="rounded_outline", colalign=("left", "right")))




# ═══════════════════════════════════════════════════════════════════════════════
# Interactive CLI
# ═══════════════════════════════════════════════════════════════════════════════
class InteractiveCLI:
    def __init__(self):
        self.selected_model: dict | None = None
        self.model_dir: Path | None = None
        self.analyzer: ModelAnalyzer | None = None
        self.bench_engine: BenchmarkEngine | None = None
        self.infer_engine: InferenceEngine | None = None

    def run(self):
        self._print_banner()
        self._print_system_info()
        while True:
            choice = self._main_menu()
            if   choice == 1: self._model_selection_menu()
            elif choice == 2: self._view_config()
            elif choice == 3: self._benchmark_menu()
            elif choice == 4: self._view_all_results()
            elif choice == 5: self._inference_menu()
            elif choice == 0:
                if self.infer_engine and self.infer_engine.is_loaded():
                    self.infer_engine.unload_model()
                print(f"\n  {C.GREEN}Goodbye!{C.RESET}\n")
                sys.exit(0)

    def _print_banner(self):
        print(f"""
{C.HEADER}╔════════════════════════════════════════════════════════════════╗
║             LoadTime Benchmark — Interactive CLI               ║
║   Load times · Model config · Inference stats · KV cache       ║
╚════════════════════════════════════════════════════════════════╝{C.RESET}""")

    def _print_system_info(self):
        print(f"\n  {C.BOLD}System Info:{C.RESET}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"    GPU:   {C.GREEN}{props.name}{C.RESET}  "
                  f"({props.total_memory / (1024**3):.1f} GB VRAM)")
        else:
            print(f"    GPU:   {C.RED}Not available{C.RESET}")
        ram = psutil.virtual_memory()
        print(f"    RAM:   {ram.total / (1024**3):.1f} GB total, "
              f"{ram.available / (1024**3):.1f} GB free")
        import shutil
        disk = shutil.disk_usage(str(CACHE_DIR.parent))
        print(f"    Disk:  {disk.free / (1024**3):.1f} GB free on {CACHE_DIR.parent}")

    # ── Main Menu ─────────────────────────────────────────────────────────
    def _main_menu(self) -> int:
        model_label = (f"{C.GREEN}{self.selected_model['name']}{C.RESET}"
                       if self.selected_model else f"{C.DIM}None{C.RESET}")
        infer_label = (self.infer_engine.status_str()
                       if self.infer_engine else f"{C.DIM}—{C.RESET}")
        need_model = f"{C.DIM}(select model first){C.RESET}" if not self.analyzer else ""

        print(f"""
{C.HEADER}{'─' * 58}
  MAIN MENU               Model: {model_label}
{'─' * 58}{C.RESET}

  {C.BOLD}1.{C.RESET} Select & Download Model
  {C.BOLD}2.{C.RESET} View Model Configuration        {need_model}
  {C.BOLD}3.{C.RESET} Run Load Time Benchmark         {need_model}
  {C.BOLD}4.{C.RESET} View All Benchmark Results
  {C.BOLD}5.{C.RESET} Run Inference & Stats           {infer_label if self.analyzer else need_model}
  {C.BOLD}0.{C.RESET} Exit""")

        return prompt_choice("Select [0-5]:", range(0, 6))

    # ── Model Selection ──────────────────────────────────────────────────
    def _model_selection_menu(self):
        # Unload inference model if switching
        if self.infer_engine and self.infer_engine.is_loaded():
            print(f"\n  {C.YELLOW}Unloading current inference model...{C.RESET}")
            self.infer_engine.unload_model()

        downloaded = get_downloaded_models()
        print_header("SELECT A MODEL")
        rows = []
        for i, m in enumerate(MODELS, 1):
            status = (f"{C.GREEN}✓ Downloaded{C.RESET}" if m["id"] in downloaded
                      else f"{C.DIM}Not downloaded{C.RESET}")
            gated = f" {C.YELLOW}🔒{C.RESET}" if m.get("gated") else ""
            tc = C.MAGENTA if m["type"] == "MoE" else C.BLUE
            rows.append([f"  {C.BOLD}{i}{C.RESET}", m["name"] + gated,
                         f"{tc}{m['type']}{C.RESET}",
                         f"~{m['approx_size_gb']:.1f} GB", status])
        print(tabulate(rows, headers=["#", "Model", "Type", "Size", "Status"],
                       tablefmt="rounded_outline", colalign=("center", "left", "center", "right", "left")))
        print(f"\n  {C.DIM}🔒 = Requires `huggingface-cli login`{C.RESET}")

        choice = prompt_choice(f"Select [1-{len(MODELS)}] (0=back):", range(0, len(MODELS) + 1))
        if choice <= 0:
            return

        model = MODELS[choice - 1]
        self.selected_model = model

        if model["id"] in downloaded:
            print(f"\n  {C.OK} {C.GREEN}{model['name']}{C.RESET} already downloaded.")
            self.model_dir = find_model_dir(model["id"])
        else:
            print(f"\n  {C.CYAN}Downloading {model['name']} (~{model['approx_size_gb']:.1f} GB)...{C.RESET}")
            try:
                self.model_dir = Path(snapshot_download(
                    model["id"], cache_dir=str(CACHE_DIR),
                    allow_patterns=DOWNLOAD_PATTERNS))
                print(f"  {C.OK} Download complete!")
            except Exception as e:
                print(f"  {C.FAIL} Download failed: {e}")
                self.selected_model = None
                return

        print(f"  {C.CYAN}Analyzing model...{C.RESET}", end="", flush=True)
        try:
            self.analyzer = ModelAnalyzer(self.model_dir)
            self.bench_engine = BenchmarkEngine(self.analyzer)
            self.infer_engine = InferenceEngine(model["id"], self.model_dir, self.analyzer)
            print(f"\r  {C.OK} Ready: {self.analyzer.num_layers} layers, "
                  f"{self.analyzer.num_experts} expert(s)/layer, "
                  f"{'MoE' if self.analyzer.is_moe else 'Dense'}, "
                  f"{fmt_bytes(self.analyzer.get_total_size())}")
        except Exception as e:
            print(f"\r  {C.FAIL} Analysis failed: {e}")
            self.analyzer = None

    # ── View Config ───────────────────────────────────────────────────────
    def _view_config(self):
        if not self.analyzer:
            print(f"\n  {C.YELLOW}Select a model first (option 1).{C.RESET}")
            return
        self.analyzer.display_config()
        input(f"\n  {C.DIM}Press Enter to go back...{C.RESET}")

    # ── Benchmark Menu ────────────────────────────────────────────────────
    def _benchmark_menu(self):
        if not self.analyzer or not self.bench_engine:
            print(f"\n  {C.YELLOW}Select a model first (option 1).{C.RESET}")
            return
        while True:
            el = "Expert" if self.analyzer.is_moe else "FFN/MLP"
            print(f"""
{C.HEADER}{'─' * 58}
  BENCHMARK MENU          Model: {C.GREEN}{self.selected_model['name']}{C.RESET}
{'─' * 58}{C.RESET}

  {C.BOLD}1.{C.RESET} Single {el}
  {C.BOLD}2.{C.RESET} Single Layer
  {C.BOLD}3.{C.RESET} Whole Model
  {C.BOLD}4.{C.RESET} All (Expert + Layer + Model)
  {C.BOLD}0.{C.RESET} Back""")

            cc = prompt_choice("Select [0-4]:", range(0, 5))
            if cc <= 0:
                return

            paths = self._select_paths()
            if paths is None:
                continue

            print()
            nw = prompt_int("Warmup runs", 1)
            nr = prompt_int("Timed runs", 3)

            li = 0
            ei = 0
            if cc in (1, 2, 4):
                li = prompt_int(f"Layer index (0-{self.analyzer.num_layers - 1})", 0)
                li = max(0, min(li, self.analyzer.num_layers - 1))
            if cc in (1, 4) and self.analyzer.is_moe:
                ei = prompt_int(f"Expert index (0-{self.analyzer.num_experts - 1})", 0)
                ei = max(0, min(ei, self.analyzer.num_experts - 1))

            print_header("RUNNING BENCHMARKS")
            comps = {1: ["expert"], 2: ["layer"], 3: ["model"], 4: ["expert", "layer", "model"]}[cc]
            all_r = []
            for comp in comps:
                all_r.extend(self.bench_engine.run(comp, paths, nw, nr, li, ei))
            self.bench_engine.display_results(all_r)
            input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")

    def _select_paths(self) -> list[str] | None:
        gpu = torch.cuda.is_available()
        ng = "" if gpu else f" {C.RED}(no GPU){C.RESET}"
        print(f"""
  Transfer path:
  {C.BOLD}1.{C.RESET} Disk → RAM
  {C.BOLD}2.{C.RESET} Disk → VRAM{ng}
  {C.BOLD}3.{C.RESET} RAM → VRAM{ng}
  {C.BOLD}4.{C.RESET} All paths
  {C.BOLD}0.{C.RESET} Back""")
        c = prompt_choice("Select [0-4]:", range(0, 5))
        return {1: ["Disk → RAM"], 2: ["Disk → VRAM"], 3: ["RAM → VRAM"],
                4: ["Disk → RAM", "Disk → VRAM", "RAM → VRAM"]}.get(c)

    # ── View All Results ──────────────────────────────────────────────────
    def _view_all_results(self):
        if not self.bench_engine or not self.bench_engine.results:
            print(f"\n  {C.YELLOW}No benchmark results yet.{C.RESET}")
            return
        self.bench_engine.display_results()
        input(f"\n  {C.DIM}Press Enter to go back...{C.RESET}")

    # ── Inference Menu ────────────────────────────────────────────────────
    def _inference_menu(self):
        if not self.analyzer or not self.infer_engine:
            print(f"\n  {C.YELLOW}Select a model first (option 1).{C.RESET}")
            return

        while True:
            loaded = self.infer_engine.is_loaded()
            status = self.infer_engine.status_str()

            print(f"""
{C.HEADER}{'─' * 58}
  INFERENCE MENU          Model: {C.GREEN}{self.selected_model['name']}{C.RESET}
  Status: {status}
{'─' * 58}{C.RESET}
""")

            if not loaded:
                vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)
                           if torch.cuda.is_available() else 0)
                model_gb = self.selected_model["approx_size_gb"]
                fits = model_gb < vram_gb * 0.9

                print(f"  {C.BOLD}Load model first:{C.RESET}")
                if torch.cuda.is_available():
                    fit_label = f"{C.GREEN}fits{C.RESET}" if fits else f"{C.RED}may OOM{C.RESET}"
                    print(f"  {C.BOLD}1.{C.RESET} Load to GPU          "
                          f"({model_gb:.1f} GB model vs {vram_gb:.1f} GB VRAM — {fit_label})")
                print(f"  {C.BOLD}2.{C.RESET} Load to CPU          (slower inference, always works)")
                if torch.cuda.is_available():
                    print(f"  {C.BOLD}3.{C.RESET} Load auto (GPU+CPU)  "
                          f"{C.GREEN}(recommended if model > VRAM){C.RESET}")
                print(f"  {C.BOLD}0.{C.RESET} Back")

                c = prompt_choice("Select:", range(0, 4))
                if c <= 0:
                    return
                device = {1: "cuda", 2: "cpu", 3: "auto"}.get(c, "cpu")
                self.infer_engine.load_model(device)
                continue

            else:
                print(f"  {C.BOLD}1.{C.RESET} Send prompt (generate response)")
                print(f"  {C.BOLD}2.{C.RESET} Unload model")
                print(f"  {C.BOLD}0.{C.RESET} Back to main menu")

                c = prompt_choice("Select [0-2]:", range(0, 3))
                if c <= 0:
                    return
                elif c == 1:
                    self._run_inference()
                elif c == 2:
                    self.infer_engine.unload_model()

    def _run_inference(self):
        print(f"\n  {C.BOLD}Enter your prompt{C.RESET} (or 'back' to cancel):")
        prompt = prompt_string(f"  {C.CYAN}>{C.RESET} ")
        if not prompt or prompt.lower() == "back":
            return

        max_tokens = prompt_int("Max new tokens", 128)
        temperature = prompt_float("Temperature (0=greedy)", 0.0)

        self.infer_engine.generate(prompt, max_tokens, temperature)
        input(f"\n  {C.DIM}Press Enter to continue...{C.RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        cli = InteractiveCLI()
        cli.run()
    except KeyboardInterrupt:
        print(f"\n\n  {C.YELLOW}Interrupted. Goodbye!{C.RESET}\n")
        sys.exit(0)
