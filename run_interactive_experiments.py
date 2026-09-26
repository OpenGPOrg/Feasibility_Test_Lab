#!/usr/bin/env python3
"""
Interactive LLM Inference Feasibility Experiment Suite.

Orchestrates continuous, empirical benchmarking of decentralized LLM inference
across network topologies (T1-T5) and workloads (W1-W7) using openGP architecture.

Features:
- Dual-node 10Hz memory telemetry (Master RTX 3060 6GB + Worker RTX 4070 Ti Super 16GB)
- Per-trial publication-grade memory profile plots (PNG)
- Fine-grained ITL (P50, P90, P99), TTFT, and prefill/decode throughput logging
- Precise inference-window network I/O accounting
- Autonomous fault tolerance and hardware capacity OOM failure classification
- Live CSV metric streaming to `interactive_decentralized_experiments.csv`
"""

import argparse
import csv
import gc
import json
import os
import pickle
import platform
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

# Ensure LoadTime directory is in path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from distributed_inference import DistributedModel, NodeConnection

# ── Global Workload Definitions ────────────────────────────────────────────────
WORKLOADS = {
    "W1": {"name": "Short Chat", "input_tokens": 50, "output_tokens": 32},
    "W2": {"name": "Standard Chat", "input_tokens": 150, "output_tokens": 64},
    "W3": {"name": "Detailed Assistant", "input_tokens": 500, "output_tokens": 128},
    "W4": {"name": "Agentic Context", "input_tokens": 1000, "output_tokens": 128},
    "W5": {"name": "RAG Document Retrieval", "input_tokens": 2000, "output_tokens": 256},
    "W6": {"name": "Heavy Codebase Stress", "input_tokens": 3000, "output_tokens": 256},
    "W7": {"name": "Long Generation Output", "input_tokens": 200, "output_tokens": 512},
}

# ── Topology Layer Partitioning Definitions ──────────────────────────────────
# 28 Layers in Qwen2.5-7B (0 to 27)
def get_topology_assignments(topology_id: str, num_layers: int = 28) -> Tuple[str, int, List[int], List[int]]:
    """
    Returns (topology_name, hops_per_token, master_layers, worker_layers).
    All assignments are full-layer only (no separate attention/FFN splitting).
    """
    if topology_id == "T1":
        # Coarse 2-Stage Pipeline: Master 0-3 (4 layers), Worker 4-27 (24 layers)
        master_layers = list(range(0, 4))
        worker_layers = list(range(4, num_layers))
        return "Coarse 2-Stage", 1, master_layers, worker_layers

    elif topology_id == "T2":
        # 4-Stage Interleaved Pipeline:
        # Master: 0-3, 14-17 (8 layers)
        # Worker: 4-13, 18-27 (20 layers)
        # Transitions: M(0-3) -> W(4-13) -> M(14-17) -> W(18-27) -> M(head) -> 3 boundary crossings
        master_layers = list(range(0, 4)) + list(range(14, 18))
        worker_layers = list(range(4, 14)) + list(range(18, num_layers))
        return "Interleaved 4-Stage", 3, master_layers, worker_layers

    elif topology_id == "T3":
        # 8-Stage Interleaved Pipeline:
        # M: 0-3, 8-10, 15-17, 22-24 (13 layers)
        # W: 4-7, 11-14, 18-21, 25-27 (15 layers)
        master_layers = list(range(0, 4)) + list(range(8, 11)) + list(range(15, 18)) + list(range(22, 25))
        worker_layers = list(range(4, 8)) + list(range(11, 15)) + list(range(18, 22)) + list(range(25, num_layers))
        return "Interleaved 8-Stage", 7, master_layers, worker_layers

    elif topology_id == "T4":
        # 14-Stage Interleaved Pipeline (2-layer chunks):
        # M: 0-1, 4-5, 8-9, 12-13, 16-17, 20-21, 24-25 (14 layers)
        # W: 2-3, 6-7, 10-11, 14-15, 18-19, 22-23, 26-27 (14 layers)
        master_layers = []
        worker_layers = []
        for blk in range(0, num_layers, 4):
            master_layers.extend([blk, blk + 1])
            if blk + 2 < num_layers:
                worker_layers.append(blk + 2)
            if blk + 3 < num_layers:
                worker_layers.append(blk + 3)
        return "Interleaved 14-Stage", 13, master_layers, worker_layers

    elif topology_id == "T5":
        # Fully Interleaved Single-Layer Pipeline:
        # M: Even layers (0, 2, 4, ..., 26) (14 layers)
        # W: Odd layers (1, 3, 5, ..., 27) (14 layers)
        master_layers = [i for i in range(num_layers) if i % 2 == 0]
        worker_layers = [i for i in range(num_layers) if i % 2 != 0]
        return "Extreme Alternating", 27, master_layers, worker_layers

    else:
        raise ValueError(f"Unknown topology: {topology_id}")


def format_layer_ranges(layers):
    """Format a list of layer numbers into contiguous start-end ranges (e.g. '0-3, 14-17')."""
    if not layers:
        return ""
    if isinstance(layers, str):
        import ast
        try:
            parsed = ast.literal_eval(layers)
            if isinstance(parsed, list):
                layers = parsed
            else:
                return layers
        except Exception:
            return layers
    layers = sorted(list(set(layers)))
    ranges = []
    start = layers[0]
    end = layers[0]
    for x in layers[1:]:
        if x == end + 1:
            end = x
        else:
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = x
    ranges.append(f"{start}-{end}" if start != end else f"{start}")
    return ", ".join(ranges)


# ── Dual-Node High-Frequency Telemetry Profiler ──────────────────────────────
class DualNodeMemoryProfiler:
    """
    Background 10Hz memory profiler sampling both local (Master) and remote (Worker)
    VRAM and RAM footprints during the active inference execution window.
    """
    def __init__(self, worker_host: str, worker_port: int, interval_sec: float = 0.1):
        self.worker_host = worker_host
        self.worker_port = worker_port
        self.interval_sec = interval_sec
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: float = 0.0
        self.telemetry_data: List[Dict[str, float]] = []
        self._worker_sock: Optional[socket.socket] = None

    def _connect_worker_socket(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.5)
            sock.connect((self.worker_host, self.worker_port))
            self._worker_sock = sock
        except Exception:
            self._worker_sock = None

    def _query_worker_memory(self) -> Tuple[float, float]:
        """Returns (vram_mb, ram_mb) from remote worker."""
        if not self._worker_sock:
            self._connect_worker_socket()
        if not self._worker_sock:
            return 0.0, 0.0

        try:
            payload = pickle.dumps({"cmd": "get_memory"})
            self._worker_sock.sendall(struct.pack(">Q", len(payload)) + payload)
            raw_len = self._worker_sock.recv(8)
            if not raw_len or len(raw_len) < 8:
                raise ConnectionResetError("Truncated response header")
            size = struct.unpack(">Q", raw_len)[0]
            data = b""
            while len(data) < size:
                chunk = self._worker_sock.recv(min(4096, size - len(data)))
                if not chunk:
                    break
                data += chunk
            resp = pickle.loads(data)
            if resp.get("status") == "ok":
                vram_mb = resp.get("vram_alloc", 0) / (1024**2)
                ram_mb = resp.get("ram_rss", 0) / (1024**2)
                return vram_mb, ram_mb
        except Exception:
            try:
                if self._worker_sock:
                    self._worker_sock.close()
            except Exception:
                pass
            self._worker_sock = None
        return 0.0, 0.0

    def start(self):
        self.running = True
        self.telemetry_data = []
        self.start_time = time.time()
        self._connect_worker_socket()
        self.thread = threading.Thread(target=self._profiler_loop, daemon=True)
        self.thread.start()

    def _profiler_loop(self):
        proc = psutil.Process(os.getpid())
        while self.running:
            t = time.time() - self.start_time
            # Local Master
            m_vram_mb = 0.0
            if torch.cuda.is_available():
                m_vram_mb = torch.cuda.memory_allocated() / (1024**2)
            m_ram_mb = proc.memory_info().rss / (1024**2)

            # Remote Worker
            w_vram_mb, w_ram_mb = self._query_worker_memory()

            self.telemetry_data.append({
                "time": t,
                "master_vram_mb": m_vram_mb,
                "master_ram_mb": m_ram_mb,
                "worker_vram_mb": w_vram_mb,
                "worker_ram_mb": w_ram_mb,
            })
            time.sleep(self.interval_sec)

    def stop(self) -> List[Dict[str, float]]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.5)
            self.thread = None
        if self._worker_sock:
            try:
                self._worker_sock.close()
            except Exception:
                pass
            self._worker_sock = None
        return self.telemetry_data

    def save_plot(self, plot_path: Path, title_info: Dict[str, any]) -> Optional[Path]:
        """Generate dual-panel visualization for Master and Worker memory footprints."""
        if not self.telemetry_data:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            times = [d["time"] for d in self.telemetry_data]
            m_vram = [d["master_vram_mb"] for d in self.telemetry_data]
            m_ram = [d["master_ram_mb"] for d in self.telemetry_data]
            w_vram = [d["worker_vram_mb"] for d in self.telemetry_data]
            w_ram = [d["worker_ram_mb"] for d in self.telemetry_data]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

            topo = title_info.get("topology", "Unknown")
            wl = title_info.get("workload", "Unknown")
            status = title_info.get("status", "SUCCESS")
            ttft = title_info.get("ttft", 0.0)

            fig.suptitle(f"Memory Telemetry — {topo} | {wl} | Status: {status}", fontsize=13, fontweight="bold")

            # Master Panel
            ax1.plot(times, m_vram, label="GPU VRAM Allocated (MB)", color="#1f77b4", linewidth=2.0)
            ax1.plot(times, m_ram, label="CPU RAM RSS (MB)", color="#aec7e8", linewidth=1.5, linestyle="--")
            if ttft > 0 and ttft <= max(times):
                ax1.axvline(ttft, color="gray", linestyle=":", label=f"TTFT ({ttft:.2f}s)")
            ax1.set_title("Master Laptop (RTX 3060 6GB VRAM, 16GB RAM)", fontsize=10, loc="left", fontweight="bold")
            ax1.set_ylabel("Memory (MB)", fontsize=9)
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(loc="upper left", fontsize=8)

            # Worker Panel
            ax2.plot(times, w_vram, label="GPU VRAM Allocated (MB)", color="#d62728", linewidth=2.0)
            ax2.plot(times, w_ram, label="CPU RAM RSS (MB)", color="#ff9896", linewidth=1.5, linestyle="--")
            if ttft > 0 and ttft <= max(times):
                ax2.axvline(ttft, color="gray", linestyle=":", label=f"TTFT ({ttft:.2f}s)")
            ax2.set_title(f"Worker Server {self.worker_host} (RTX 4070 Ti Super 16GB VRAM, 32GB RAM)", fontsize=10, loc="left", fontweight="bold")
            ax2.set_xlabel("Time (seconds)", fontsize=9)
            ax2.set_ylabel("Memory (MB)", fontsize=9)
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="upper left", fontsize=8)

            plt.tight_layout()
            plot_path = plot_path.resolve()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=120)
            plt.close(fig)
            return plot_path
        except Exception as e:
            print(f"  ⚠ Failed to render plot: {e}")
            return None


# ── Calibrated Synthetic Prompt Generator ────────────────────────────────────
class CalibratedPromptGenerator:
    """Generates synthetic prompts matching exact calibrated token lengths."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.base_text = (
            "Distributed computing systems divide computational workloads across multiple networked nodes. "
            "In decentralized large language model inference, transformer layers and attention operations are "
            "partitioned over volunteer nodes connected by local or wide area networks. However, interactive "
            "chat sessions require low latency, high throughput, and swift time-to-first-token. "
            "As sequence lengths scale, activation tensors transferred across network boundaries impose severe "
            "bandwidth bottlenecks. Prefilling long contexts creates substantial transmission delays, while autoregressive "
            "token generation compounds round-trip latencies for every generated token. Network latency, packet jitter, "
            "and serial tensor dependencies degrade conversational fluidity. "
            "Below is a technical evaluation of distributed LLM systems under varying context windows. "
        )
        self.base_tokens = self.tokenizer.encode(self.base_text, add_special_tokens=False)

    def generate_prompt(self, target_tokens: int) -> str:
        """Produce a coherent prompt containing precisely `target_tokens` tokens."""
        reps = (target_tokens // len(self.base_tokens)) + 2
        repeated_tokens = (self.base_tokens * reps)[:target_tokens]
        prompt_str = self.tokenizer.decode(repeated_tokens)
        final_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        if len(final_ids) > target_tokens:
            prompt_str = self.tokenizer.decode(final_ids[:target_tokens])
        return prompt_str


# ── CSV Logging Schema ────────────────────────────────────────────────────────
CSV_FIELDS = [
    "timestamp",
    "model_id",
    "topology_id",
    "topology_name",
    "master_layers",
    "worker_layers",
    "num_hops_per_token",
    "workload_id",
    "workload_name",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "ttft_sec",
    "prefill_tps",
    "decode_sec",
    "decode_tps",
    "avg_itl_ms",
    "min_itl_ms",
    "max_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "p99_itl_ms",
    "master_net_sent_mb",
    "master_net_recv_mb",
    "net_rate_mbps",
    "master_vram_start_gb",
    "master_vram_peak_gb",
    "worker_vram_start_gb",
    "worker_vram_peak_gb",
    "kv_cache_total_mb",
    "profile_plot_path",
    "status",
    "error_notes",
]

def init_csv(file_path: Path):
    if not file_path.exists():
        with open(file_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


# ── Experiment Runner ─────────────────────────────────────────────────────────
class InteractiveExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.output_csv = (CURRENT_DIR / args.output_csv).resolve()
        self.profile_dir = (CURRENT_DIR / args.profile_dir).resolve()
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        init_csv(self.output_csv)

        print("════════════════════════════════════════════════════════════════════")
        print("  🚀 openGP Interactive LLM Decentralized Feasibility Study")
        print("════════════════════════════════════════════════════════════════════")
        print(f"  Model ID:     {args.model_id}")
        print(f"  Worker Node:  {args.worker_host}:{args.worker_port}")
        print(f"  Topologies:   {args.topologies}")
        print(f"  Workloads:    {args.workloads}")
        print(f"  Output CSV:   {self.output_csv}")
        print(f"  Profile Dir:  {self.profile_dir}")
        print("════════════════════════════════════════════════════════════════════\n")

        self.dist_model = DistributedModel()
        self.node = NodeConnection(host=args.worker_host, port=args.worker_port, hostname=f"worker-{args.worker_host}")

        print("  Connecting to worker node...", end="", flush=True)
        try:
            self.node.connect(timeout=8.0)
            print(f" Connected! ({self.node.hostname})\n")
        except Exception as e:
            print(f"\n  ❌ Failed to connect to worker node at {args.worker_host}:{args.worker_port}: {e}")
            sys.exit(1)

        print("  Loading model and tokenizer...", flush=True)
        t0 = time.time()
        self.dist_model.load_model(args.model_id, Path(args.model_dir), quant="none")
        print(f"  ✓ Model ready in {time.time()-t0:.2f}s ({self.dist_model.num_layers} layers)\n")

        self.prompt_generator = CalibratedPromptGenerator(self.dist_model.tokenizer)

    def ensure_network_liveness(self, max_wait_sec: int = 90) -> bool:
        """Polls worker node and waits for local Wi-Fi / network interface to recover if dropped."""
        t0 = time.time()
        while time.time() - t0 < max_wait_sec:
            try:
                if not self.node.is_connected():
                    self.node.connect(timeout=4.0)
                resp = self.node.send_cmd({"cmd": "ping"})
                if resp and resp.get("status") == "pong":
                    return True
            except Exception:
                pass
            time.sleep(2.0)
        return False

    def setup_topology(self, topology_id: str) -> bool:
        """Partition model layers according to the topology configuration."""
        name, hops, master_layers, worker_layers = get_topology_assignments(topology_id, self.dist_model.num_layers)
        print(f"\n  🔄 Configuring Topology {topology_id} ({name}):")
        print(f"     Master Layers ({len(master_layers)}): {master_layers[:8]}{'...' if len(master_layers) > 8 else ''}")
        print(f"     Worker Layers ({len(worker_layers)}): {worker_layers[:8]}{'...' if len(worker_layers) > 8 else ''}")
        print(f"     Network Hops/Token: {hops}")

        # Clear existing assignments
        self.dist_model.assignments.clear()

        # Assign worker layers
        for l_idx in worker_layers:
            self.dist_model.assign_to_remote(f"layer_{l_idx}", "layer", self.node, layer_idx=l_idx)

        # Apply distribution across cluster
        success = self.dist_model.apply_distribution()
        if not success:
            print(f"  ⚠ Failed to apply distribution for {topology_id}")
            return False
        return True

    def run_single_trial(self, topology_id: str, workload_id: str, trial_idx: int) -> Dict[str, any]:
        """Execute a single calibrated inference benchmark trial with full telemetry and stall recovery."""
        wl = WORKLOADS[workload_id]
        target_in = wl["input_tokens"]
        target_out = wl["output_tokens"]
        topo_name, hops, master_layers, worker_layers = get_topology_assignments(topology_id, self.dist_model.num_layers)

        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = self.profile_dir / f"run_{topology_id}_{workload_id}_t{trial_idx}_{ts_str}.png"

        print(f"\n────────────────────────────────────────────────────────────────────")
        print(f"  ▶ Trial {trial_idx} | {topology_id} ({topo_name}) | {workload_id} ({wl['name']})")
        print(f"    Target Tokens: {target_in} in, {target_out} out | Network Hops/Tok: {hops}")
        print(f"────────────────────────────────────────────────────────────────────")

        # 1. Pre-flight Network Liveness Check
        if not self.ensure_network_liveness(max_wait_sec=30):
            print(f"  ⚠ Network unreachable! Waiting up to 60s for connection to recover...")
            if not self.ensure_network_liveness(max_wait_sec=60):
                print(f"  ❌ Network outage persisted. Skipping trial {trial_idx} as FAILED_NETWORK.")
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "model_id": self.args.model_id,
                    "topology_id": topology_id,
                    "topology_name": topo_name,
                    "master_layers": format_layer_ranges(master_layers),
                    "worker_layers": format_layer_ranges(worker_layers),
                    "num_hops_per_token": hops,
                    "workload_id": workload_id,
                    "workload_name": wl["name"],
                    "input_tokens": target_in,
                    "output_tokens": 0,
                    "total_tokens": target_in,
                    "ttft_sec": 0.0, "prefill_tps": 0.0, "decode_sec": 0.0, "decode_tps": 0.0,
                    "avg_itl_ms": 0.0, "min_itl_ms": 0.0, "max_itl_ms": 0.0,
                    "p50_itl_ms": 0.0, "p90_itl_ms": 0.0, "p99_itl_ms": 0.0,
                    "master_net_sent_mb": 0.0, "master_net_recv_mb": 0.0, "net_rate_mbps": 0.0,
                    "master_vram_start_gb": 0.0, "master_vram_peak_gb": 0.0,
                    "worker_vram_start_gb": 0.0, "worker_vram_peak_gb": 0.0,
                    "kv_cache_total_mb": 0.0, "profile_plot_path": "",
                    "status": "FAILED_NETWORK",
                    "error_notes": "Network interface unreachable before trial start (Wi-Fi disconnect)",
                }
                with open(self.output_csv, mode="a", newline="") as f:
                    csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
                return row

        # Prepare synthetic prompt
        prompt_text = self.prompt_generator.generate_prompt(target_in)

        # Pre-run memory recovery & baseline check
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        try:
            self.node.send_cmd({"cmd": "clear_kv"})
        except Exception:
            pass

        # Baseline VRAM
        m_vram_start = (torch.cuda.memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
        w_mem_resp = {}
        try:
            w_mem_resp = self.node.send_cmd({"cmd": "get_memory"})
        except Exception:
            pass
        w_vram_start = (w_mem_resp.get("vram_alloc", 0) / (1024**3)) if w_mem_resp else 0.0

        # Start Telemetry Profiler
        profiler = DualNodeMemoryProfiler(self.args.worker_host, self.args.worker_port)
        profiler.start()

        status = "SUCCESS"
        error_notes = ""
        stats_dict = None
        t_start = time.perf_counter()

        # 2. Watchdog Timer to abort frozen sockets if Wi-Fi disconnects
        # Scale generously for multi-hop network round trips and CPU offload phases
        stall_limit = max(180.0, (target_in / 1.5) + (target_out * max(3.0, hops * 0.5)) + (hops * 30.0) + 120.0)
        watchdog_triggered = False

        def _watchdog_interrupt():
            nonlocal watchdog_triggered
            watchdog_triggered = True
            print(f"\n  ⚠ [WATCHDOG] Inference stalled for >{stall_limit:.0f}s (Wi-Fi flap or dead link)! Forcing reconnect...")
            self.node.close()

        watchdog = threading.Timer(stall_limit, _watchdog_interrupt)
        watchdog.start()

        try:
            stats_dict = self.dist_model.generate(
                prompt=prompt_text,
                max_new_tokens=target_out,
                temperature=0.0,
            )
        except torch.cuda.OutOfMemoryError as oom:
            status = "FAILED_OOM"
            error_notes = (
                f"Hardware VRAM capacity exceeded (Local CUDA OOM): "
                f"Master RTX 3060 Laptop (6GB limit) ran out of memory attempting {target_in} context + "
                f"{target_out} generation in topology {topology_id}. Error: {str(oom)[:160]}"
            )
            print(f"\n  ❌ {status}: {error_notes}")
        except Exception as e:
            err_str = str(e)
            if "out of memory" in err_str.lower() or "CUDA out of memory" in err_str:
                status = "FAILED_OOM"
                error_notes = (
                    f"Hardware VRAM capacity exceeded (Worker CUDA OOM): "
                    f"Worker node {self.node.hostname} ran out of memory under {target_in} context + "
                    f"{target_out} generation in topology {topology_id}. Error: {err_str[:160]}"
                )
            elif watchdog_triggered:
                status = "FAILED_NETWORK"
                error_notes = f"Network stall detected: generation exceeded {stall_limit:.0f}s watchdog without completion (Wi-Fi disconnect/reconnect)"
            elif "Connection" in err_str or "Broken pipe" in err_str or "timeout" in err_str.lower() or "unreachable" in err_str.lower():
                status = "FAILED_NETWORK"
                error_notes = f"Network socket connection dropped during inference: {err_str[:160]}"
            else:
                status = "FAILED_OTHER"
                error_notes = f"Inference execution error: {err_str[:160]}"
            print(f"\n  ❌ {status}: {error_notes}")
        finally:
            watchdog.cancel()

        total_elapsed = time.perf_counter() - t_start

        # Post-trial recovery if network dropped
        if status == "FAILED_NETWORK":
            print(f"  🔄 Recovering from network event... Waiting for network interface to reconnect...")
            if self.ensure_network_liveness(max_wait_sec=90):
                print(f"  ✓ Network restored! Resyncing worker state...")
                try: self.node.send_cmd({"cmd": "clear_kv"})
                except Exception: pass
                self.setup_topology(topology_id)

        # Stop Telemetry Profiler
        telemetry = profiler.stop()

        # Compute peak memory from telemetry
        m_vram_peak = max([d["master_vram_mb"] / 1024 for d in telemetry] + [m_vram_start]) if telemetry else m_vram_start
        w_vram_peak = max([d["worker_vram_mb"] / 1024 for d in telemetry] + [w_vram_start]) if telemetry else w_vram_start

        # Generate Memory Profile Plot
        plot_meta = {
            "topology": f"{topology_id} ({topo_name})",
            "workload": f"{workload_id} ({wl['name']})",
            "status": status,
            "ttft": stats_dict.get("ttft", 0.0) if stats_dict else 0.0,
        }
        saved_plot = profiler.save_plot(plot_filename, plot_meta)
        rel_plot_path = str(saved_plot.resolve().relative_to(CURRENT_DIR.resolve())) if saved_plot else ""

        # Aggregate Metrics
        if status == "SUCCESS" and stats_dict:
            actual_in = stats_dict.get("num_input", target_in)
            actual_out = stats_dict.get("num_output", 0)
            total_tok = actual_in + actual_out
            ttft = stats_dict.get("ttft", 0.0)
            prefill_tps = actual_in / ttft if ttft > 0 else 0.0
            decode_sec = stats_dict.get("decode_time", 0.0)
            decode_tps = (actual_out - 1) / decode_sec if (decode_sec > 0 and actual_out > 1) else 0.0
            token_times = stats_dict.get("token_times", [])

            avg_itl = (np.mean(token_times) * 1000) if token_times else 0.0
            min_itl = (np.min(token_times) * 1000) if token_times else 0.0
            max_itl = (np.max(token_times) * 1000) if token_times else 0.0
            p50_itl = (np.percentile(token_times, 50) * 1000) if token_times else 0.0
            p90_itl = (np.percentile(token_times, 90) * 1000) if token_times else 0.0
            p99_itl = (np.percentile(token_times, 99) * 1000) if token_times else 0.0

            net_sent_mb = stats_dict.get("net_sent", 0) / (1024**2)
            net_recv_mb = stats_dict.get("net_recv", 0) / (1024**2)
            total_net_mb = net_sent_mb + net_recv_mb
            net_rate_mbps = (total_net_mb * 8) / total_elapsed if total_elapsed > 0 else 0.0
            kv_total_mb = stats_dict.get("kv_total", 0) / (1024**2)
        else:
            actual_in = target_in
            actual_out = 0
            total_tok = target_in
            ttft = 0.0
            prefill_tps = 0.0
            decode_sec = total_elapsed
            decode_tps = 0.0
            avg_itl = min_itl = max_itl = p50_itl = p90_itl = p99_itl = 0.0
            net_sent_mb = self.node.bytes_sent / (1024**2)
            net_recv_mb = self.node.bytes_received / (1024**2)
            net_rate_mbps = 0.0
            kv_total_mb = 0.0

        row = {
            "timestamp": datetime.now().isoformat(),
            "model_id": self.args.model_id,
            "topology_id": topology_id,
            "topology_name": topo_name,
            "master_layers": format_layer_ranges(master_layers),
            "worker_layers": format_layer_ranges(worker_layers),
            "num_hops_per_token": hops,
            "workload_id": workload_id,
            "workload_name": wl["name"],
            "input_tokens": actual_in,
            "output_tokens": actual_out,
            "total_tokens": total_tok,
            "ttft_sec": round(ttft, 4),
            "prefill_tps": round(prefill_tps, 2),
            "decode_sec": round(decode_sec, 4),
            "decode_tps": round(decode_tps, 2),
            "avg_itl_ms": round(avg_itl, 2),
            "min_itl_ms": round(min_itl, 2),
            "max_itl_ms": round(max_itl, 2),
            "p50_itl_ms": round(p50_itl, 2),
            "p90_itl_ms": round(p90_itl, 2),
            "p99_itl_ms": round(p99_itl, 2),
            "master_net_sent_mb": round(net_sent_mb, 3),
            "master_net_recv_mb": round(net_recv_mb, 3),
            "net_rate_mbps": round(net_rate_mbps, 2),
            "master_vram_start_gb": round(m_vram_start, 3),
            "master_vram_peak_gb": round(m_vram_peak, 3),
            "worker_vram_start_gb": round(w_vram_start, 3),
            "worker_vram_peak_gb": round(w_vram_peak, 3),
            "kv_cache_total_mb": round(kv_total_mb, 2),
            "profile_plot_path": rel_plot_path,
            "status": status,
            "error_notes": error_notes,
        }

        # Real-time CSV append:
        # Only record valid trials and legitimate physical hardware OOM boundaries.
        # Discard transient network dropouts from the primary dataset per user instruction.
        if status in ("SUCCESS", "FAILED_OOM"):
            with open(self.output_csv, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(row)
        else:
            print(f"  ⚠ Disconnection occurred! Discarding trial from results CSV.")
            with open("network_disconnections.log", mode="a") as f:
                f.write(f"[{row['timestamp']}] Discarded {topology_id}/{workload_id}: {row['error_notes']}\n")

        # Display Summary
        print(f"  ✓ Result: {status}")
        if status == "SUCCESS":
            print(f"    TTFT: {ttft:.3f}s ({prefill_tps:.1f} tok/s) | Decode: {decode_sec:.2f}s ({decode_tps:.1f} tok/s)")
            print(f"    Avg ITL: {avg_itl:.1f} ms (P50: {p50_itl:.1f}ms, P90: {p90_itl:.1f}ms, P99: {p99_itl:.1f}ms)")
            print(f"    Net I/O: Sent {net_sent_mb:.2f} MB, Recv {net_recv_mb:.2f} MB ({net_rate_mbps:.1f} Mbps)")
            print(f"    VRAM Peak: Master {m_vram_peak:.2f} GB | Worker {w_vram_peak:.2f} GB")
        if rel_plot_path and status in ("SUCCESS", "FAILED_OOM"):
            print(f"    Profile Graph: {rel_plot_path}")

        # Post-run cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return row

    def run_suite(self):
        """Execute the configured experimental suite over all topologies and workloads."""
        topologies = [t.strip() for t in self.args.topologies.split(",") if t.strip()]
        workloads = [w.strip() for w in self.args.workloads.split(",") if w.strip()]

        start_time = time.time()
        max_duration_sec = self.args.duration_hours * 3600
        iteration = 1

        print(f"🚀 Launching experiment suite across {len(topologies)} topologies and {len(workloads)} workloads.")
        print(f"   Max Duration: {self.args.duration_hours} hours | Target CSV: {self.output_csv}\n")

        while True:
            for topo_id in topologies:
                if (time.time() - start_time) > max_duration_sec:
                    print(f"\n⏱ Experiment duration limit ({self.args.duration_hours}h) reached.")
                    return

                success = self.setup_topology(topo_id)
                if not success:
                    print(f"  ⚠ Skipping workloads for topology {topo_id} due to setup failure.")
                    continue

                for wl_id in workloads:
                    if (time.time() - start_time) > max_duration_sec:
                        print(f"\n⏱ Experiment duration limit reached.")
                        return

                    # Automatically redo trial if any network disconnection occurred
                    max_net_retries = 3
                    for net_attempt in range(max_net_retries):
                        row = self.run_single_trial(topo_id, wl_id, trial_idx=iteration)
                        if row["status"] in ("SUCCESS", "FAILED_OOM"):
                            break
                        print(f"  🔄 Redoing {topo_id}/{wl_id} due to network interruption (attempt {net_attempt+1}/{max_net_retries})...")
                        time.sleep(3.0)

                    # Inter-trial pause for thermal stability and network socket drain
                    time.sleep(1.0)

                    if Path(".stop_experiments").exists():
                        print(f"\n🛑 Graceful stop flag detected (.stop_experiments)! Stopping suite as requested...")
                        try: Path(".stop_experiments").unlink()
                        except Exception: pass
                        return

            iteration += 1
            if iteration > self.args.repeats and self.args.repeats > 0:
                print(f"\n🎉 Completed requested {self.args.repeats} repetition(s) of test suite.")
                break


# ── CLI Entrypoint ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive LLM Feasibility Experiment Runner")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_dir", type=str,
                        default="model_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--worker_host", type=str, default="10.8.100.23")
    parser.add_argument("--worker_port", type=int, default=9900)
    parser.add_argument("--topologies", type=str, default="T1,T2,T3,T4,T5")
    parser.add_argument("--workloads", type=str, default="W1,W2,W3,W4,W5,W6,W7")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions across full suite (0 for continuous until time limit)")
    parser.add_argument("--duration_hours", type=float, default=12.0, help="Maximum execution time in hours")
    parser.add_argument("--output_csv", type=str, default="interactive_decentralized_experiments.csv")
    parser.add_argument("--profile_dir", type=str, default="experiment_profiles")
    return parser.parse_args()


def main():
    args = parse_args()
    runner = InteractiveExperimentRunner(args)
    runner.run_suite()


if __name__ == "__main__":
    main()
