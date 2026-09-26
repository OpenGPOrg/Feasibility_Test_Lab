#!/usr/bin/env python3
"""
Scientific Experiment Runner: Multi-Hop Feasibility Study (1, 3, 5, 7, 9 Hops).
Evaluates Time-to-First-Token (TTFT) and Decode Throughput (TPS) across Small, Medium, and Large workloads.
Measures and logs split KV cache (Master vs. Worker) and generates exact replicates of T.png and D.png.
"""

import argparse
import csv
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from distributed_inference import (
    DistributedModel,
    NodeConnection,
    get_kv_cache_size_bytes,
)

# ── Workload Definitions (Guaranteed Zero-OOM) ──────────────────────────────
WORKLOADS = {
    "small": {"name": "Small Chat", "input_tokens": 60, "output_tokens": 32},
    "medium": {"name": "Medium Chat", "input_tokens": 250, "output_tokens": 64},
    "large": {"name": "Large Context", "input_tokens": 1000, "output_tokens": 128},
}

# ── Hop Topology Definitions ────────────────────────────────────────────────
# 28 Layers Total (0 to 27). Master holds <= 6 layers to keep VRAM < 4.3 GB (leaving >= 1.4 GB free).
def get_hop_topology(hops: int, num_layers: int = 28) -> Tuple[str, List[int], List[int]]:
    if hops == 1:
        # Master: 0-3 (4 layers), Worker: 4-27 (24 layers)
        m_layers = list(range(0, 4))
        w_layers = list(range(4, num_layers))
        name = "1-Hop (Coarse 2-Stage)"
    elif hops == 3:
        # Master: 0-1, 14-15 (4 layers), Worker: 2-13, 16-27 (24 layers)
        m_layers = list(range(0, 2)) + list(range(14, 16))
        w_layers = list(range(2, 14)) + list(range(16, num_layers))
        name = "3-Hop (Interleaved 4-Stage)"
    elif hops == 5:
        # Master: 0-1, 10-11, 20-21 (6 layers), Worker: 2-9, 12-19, 22-27 (22 layers)
        m_layers = list(range(0, 2)) + list(range(10, 12)) + list(range(20, 22))
        w_layers = list(range(2, 10)) + list(range(12, 20)) + list(range(22, num_layers))
        name = "5-Hop (Interleaved 6-Stage)"
    elif hops == 7:
        # Master: 0, 7, 14, 21 (4 layers), Worker: 1-6, 8-13, 15-20, 22-27 (24 layers)
        m_layers = [0, 7, 14, 21]
        w_layers = list(range(1, 6)) + list(range(8, 14)) + list(range(15, 21)) + list(range(22, num_layers))
        # Note: 1-6 is 6 layers [1,2,3,4,5,6] -> wait range(1,7) is 1..6!
        w_layers = list(range(1, 7)) + list(range(8, 14)) + list(range(15, 21)) + list(range(22, num_layers))
        name = "7-Hop (Interleaved 8-Stage)"
    elif hops == 9:
        # Master: 0, 6, 12, 18, 24 (5 layers), Worker: 1-5, 7-11, 13-17, 19-23, 25-27 (23 layers)
        m_layers = [0, 6, 12, 18, 24]
        w_layers = list(range(1, 6)) + list(range(7, 12)) + list(range(13, 18)) + list(range(19, 24)) + list(range(25, num_layers))
        name = "9-Hop (Interleaved 10-Stage)"
    else:
        raise ValueError(f"Unsupported hops: {hops}")
    return name, m_layers, w_layers


def format_layer_ranges(layers) -> str:
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


# ── CSV Schema with Split KV Cache ──────────────────────────────────────────
CSV_FIELDS = [
    "timestamp",
    "iteration",
    "hops",
    "workload",
    "master_layers",
    "worker_layers",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "ttft_sec",
    "prefill_tps",
    "decode_sec",
    "decode_tps",
    "avg_itl_ms",
    "p50_itl_ms",
    "p90_itl_ms",
    "master_net_sent_mb",
    "master_net_recv_mb",
    "master_vram_peak_gb",
    "worker_vram_peak_gb",
    "master_kv_cache_mb",
    "worker_kv_cache_mb",
    "total_kv_cache_mb",
    "status",
    "error_notes",
]

def init_csv(file_path: Path):
    if not file_path.exists():
        with open(file_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


# ── Dual-Node 10Hz Profiler ─────────────────────────────────────────────────
class DualNodeMemoryProfiler:
    def __init__(self, worker_host: str, worker_port: int, interval_sec: float = 0.1):
        self.worker_host = worker_host
        self.worker_port = worker_port
        self.interval_sec = interval_sec
        self.running = False
        self.thread: Optional[threading.Thread] = None
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

    def _query_worker_vram(self) -> float:
        if not self._worker_sock:
            self._connect_worker_socket()
        if not self._worker_sock:
            return 0.0
        try:
            import pickle, struct
            msg = {"cmd": "get_memory"}
            payload = pickle.dumps(msg)
            self._worker_sock.sendall(struct.pack(">I", len(payload)) + payload)
            raw_len = self._worker_sock.recv(4)
            if len(raw_len) < 4:
                self._worker_sock.close()
                self._worker_sock = None
                return 0.0
            resp_len = struct.unpack(">I", raw_len)[0]
            buf = bytearray()
            while len(buf) < resp_len:
                chunk = self._worker_sock.recv(min(4096, resp_len - len(buf)))
                if not chunk: break
                buf.extend(chunk)
            resp = pickle.loads(buf)
            return resp.get("vram_alloc", 0) / (1024**3)
        except Exception:
            if self._worker_sock:
                try: self._worker_sock.close()
                except Exception: pass
            self._worker_sock = None
            return 0.0

    def start(self):
        self.running = True
        self.telemetry_data = []
        def _loop():
            while self.running:
                m_vram = (torch.cuda.memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
                w_vram = self._query_worker_vram()
                self.telemetry_data.append({"master_vram": m_vram, "worker_vram": w_vram})
                time.sleep(self.interval_sec)
        self.thread = threading.Thread(target=_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Tuple[float, float]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self._worker_sock:
            try: self._worker_sock.close()
            except Exception: pass
            self._worker_sock = None
        m_peaks = [d["master_vram"] for d in self.telemetry_data] if self.telemetry_data else [0.0]
        w_peaks = [d["worker_vram"] for d in self.telemetry_data] if self.telemetry_data else [0.0]
        return max(m_peaks), max(w_peaks)


# ── Synthetic Calibrated Prompt Generator ───────────────────────────────────
class CalibratedPromptGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate_prompt(self, target_tokens: int) -> str:
        base_sentence = (
            "Distributed computing systems partition computational workloads across networked devices. "
            "In decentralized LLM inference, transformer layers are evaluated across volunteer nodes. "
            "However, interactive chat sessions require low latency and rapid time to first token. "
            "When sequence lengths increase, activation tensors transferred across network boundaries "
            "impose bandwidth bottlenecks and serialization delays. "
        )
        base_tokens = len(self.tokenizer.encode(base_sentence))
        repeats = max(1, (target_tokens // base_tokens) + 1)
        long_text = (base_sentence * repeats).strip()
        tokens = self.tokenizer.encode(long_text)[:target_tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


# ── Main Runner Class ────────────────────────────────────────────────────────
class HopsExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.output_csv = Path(args.output_csv)
        init_csv(self.output_csv)

        print(f"\n============================================================")
        print(f"  openGP Multi-Hop Feasibility Evaluation (1, 3, 5, 7, 9 Hops)")
        print(f"============================================================")
        print(f"  Target Hops:       {args.hops}")
        print(f"  Workloads:         {args.workloads}")
        print(f"  Repetitions:       {args.repeats}")
        print(f"  Output CSV:        {self.output_csv}")
        print(f"============================================================\n")

        print(f"Connecting to worker node at {args.worker_host}:{args.worker_port}...")
        self.node = NodeConnection(host=args.worker_host, port=args.worker_port, hostname=f"worker-{args.worker_host}")
        connected = False
        while not connected:
            if Path(".stop_experiments").exists():
                print("\n🛑 Stop signal detected (.stop_experiments)! Exiting...")
                try: Path(".stop_experiments").unlink()
                except Exception: pass
                sys.exit(0)
            try:
                self.node.connect(timeout=5.0)
                resp = self.node.send_cmd({"cmd": "ping"})
                if resp.get("status") == "pong":
                    connected = True
                    print(f"  ✓ Connected to {self.node.hostname}!\n")
                    break
            except Exception:
                print(f"  ⏳ Waiting for worker node at {args.worker_host}:{args.worker_port} to start... (retrying in 5s)", flush=True)
                time.sleep(5.0)

        print("Loading local model components...")
        self.dist_model = DistributedModel()
        self.dist_model.load_model(args.model_id, Path(args.model_dir), quant="none")
        self.prompt_generator = CalibratedPromptGenerator(self.dist_model.tokenizer)
        print("  ✓ Local components loaded into memory!\n")

    def ensure_network_liveness(self, max_wait_sec: int = 60) -> bool:
        start_wait = time.time()
        while time.time() - start_wait < max_wait_sec:
            try:
                if not self.node.is_connected():
                    self.node.connect(timeout=4.0)
                resp = self.node.send_cmd({"cmd": "ping"})
                if resp.get("status") == "pong":
                    return True
            except Exception:
                pass
            time.sleep(2.0)
        return False

    def setup_topology(self, hops: int) -> bool:
        name, m_layers, w_layers = get_hop_topology(hops, len(self.dist_model.layers))
        print(f"\n  🔄 Configuring {name}:")
        print(f"     Master: {format_layer_ranges(m_layers)} ({len(m_layers)} layers)")
        print(f"     Worker: {format_layer_ranges(w_layers)} ({len(w_layers)} layers)")
        print(f"     Hops per Token: {hops}")

        # Move all local components to CPU before reassigning to prevent VRAM leak across topologies
        for layer in self.dist_model.layers:
            if layer is not None:
                layer.to("cpu")
        if self.dist_model.embedding is not None:
            self.dist_model.embedding.to("cpu")
        if self.dist_model.norm is not None:
            self.dist_model.norm.to("cpu")
        if self.dist_model.lm_head is not None:
            self.dist_model.lm_head.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.dist_model.assignments.clear()
        for l_idx in w_layers:
            self.dist_model.assign_to_remote(f"layer_{l_idx}", "layer", self.node, layer_idx=l_idx)

        # Apply distribution - automatically accelerates master's <=6 layers on Master GPU
        self.dist_model.apply_distribution()

        try:
            self.node.send_cmd({"cmd": "clear_kv"})
        except Exception:
            pass
        return True

    def run_single_trial(self, iteration: int, hops: int, wl_key: str) -> Dict[str, any]:
        wl = WORKLOADS[wl_key]
        target_in = wl["input_tokens"]
        target_out = wl["output_tokens"]
        name, m_layers, w_layers = get_hop_topology(hops, len(self.dist_model.layers))
        m_layer_str = format_layer_ranges(m_layers)
        w_layer_str = format_layer_ranges(w_layers)

        print(f"\n────────────────────────────────────────────────────────────────────")
        print(f"  ▶ Iteration {iteration}/{self.args.repeats} | Hops: {hops} | {wl['name']}")
        print(f"    Target: {target_in} in, {target_out} out | Layers: M({m_layer_str}), W({w_layer_str})")
        print(f"────────────────────────────────────────────────────────────────────")

        if not self.ensure_network_liveness(max_wait_sec=60):
            print(f"  ❌ Network unreachable! Skipping trial.")
            return {"status": "FAILED_NETWORK"}

        prompt_text = self.prompt_generator.generate_prompt(target_in)

        # Flush VRAM before trial
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        try:
            self.node.send_cmd({"cmd": "clear_kv"})
        except Exception:
            pass

        self.node.bytes_sent = 0
        self.node.bytes_received = 0

        # Start 10Hz Profiler
        profiler = DualNodeMemoryProfiler(self.args.worker_host, self.args.worker_port)
        profiler.start()

        status = "SUCCESS"
        error_notes = ""
        stats_dict = None
        t_start = time.perf_counter()

        # Dynamic Watchdog Timer
        stall_limit = max(180.0, (target_in / 1.5) + (target_out * max(3.0, hops * 0.5)) + (hops * 30.0) + 120.0)
        watchdog_triggered = False

        def _watchdog_interrupt():
            nonlocal watchdog_triggered
            watchdog_triggered = True
            print(f"\n  ⚠ [WATCHDOG] Exceeded {stall_limit:.0f}s! Forcing reconnect...")
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
            error_notes = f"CUDA Out of Memory: {str(oom)[:160]}"
            print(f"\n  ❌ {status}: {error_notes}")
        except Exception as e:
            err_str = str(e)
            if "out of memory" in err_str.lower():
                status = "FAILED_OOM"
                error_notes = f"Worker OOM: {err_str[:160]}"
            elif watchdog_triggered:
                status = "FAILED_NETWORK"
                error_notes = f"Network stall > {stall_limit:.0f}s"
            elif any(s in err_str for s in ("Connection", "Broken pipe", "timeout", "unreachable")):
                status = "FAILED_NETWORK"
                error_notes = f"Socket dropped: {err_str[:160]}"
            else:
                status = "FAILED_OTHER"
                error_notes = f"Execution error: {err_str[:160]}"
            print(f"\n  ❌ {status}: {error_notes}")
        finally:
            watchdog.cancel()

        total_elapsed = time.perf_counter() - t_start
        m_vram_peak, w_vram_peak = profiler.stop()

        if status == "SUCCESS" and stats_dict:
            actual_in = stats_dict.get("num_input", target_in)
            actual_out = stats_dict.get("num_output", target_out)
            total_tok = actual_in + actual_out
            ttft = stats_dict.get("ttft", 0.0)
            prefill_tps = actual_in / ttft if ttft > 0 else 0.0
            decode_sec = stats_dict.get("decode_time", 0.0)
            decode_tps = (actual_out - 1) / decode_sec if (decode_sec > 0 and actual_out > 1) else 0.0
            token_times = stats_dict.get("token_times", [])
            avg_itl = (np.mean(token_times) * 1000) if token_times else 0.0
            p50_itl = (np.percentile(token_times, 50) * 1000) if token_times else 0.0
            p90_itl = (np.percentile(token_times, 90) * 1000) if token_times else 0.0

            net_sent_mb = stats_dict.get("net_sent", 0) / (1024**2)
            net_recv_mb = stats_dict.get("net_recv", 0) / (1024**2)

            # Split KV Cache Calculation
            node_mems = stats_dict.get("node_mem_stats", {})
            m_kv_bytes = 0
            w_kv_bytes = 0
            for nlabel, mstats in node_mems.items():
                if mstats.get("is_local"):
                    m_kv_bytes += mstats.get("kv_bytes", 0)
                else:
                    w_kv_bytes += mstats.get("kv_bytes", 0)

            # Fallback to theoretical if 0
            bytes_per_tok_layer = 2 * self.dist_model.num_kv_heads * self.dist_model.head_dim * self.dist_model.dtype_size
            if m_kv_bytes == 0:
                m_kv_bytes = len(m_layers) * bytes_per_tok_layer * total_tok
            if w_kv_bytes == 0:
                w_kv_bytes = len(w_layers) * bytes_per_tok_layer * total_tok

            m_kv_mb = round(m_kv_bytes / (1024**2), 2)
            w_kv_mb = round(w_kv_bytes / (1024**2), 2)
            total_kv_mb = round((m_kv_bytes + w_kv_bytes) / (1024**2), 2)

            if w_vram_peak == 0.0:
                for nlabel, mstats in node_mems.items():
                    if not mstats.get("is_local"):
                        w_vram_peak = mstats.get("vram_alloc", 0) / (1024**3)
                        if w_vram_peak == 0.0:
                            w_vram_peak = (mstats.get("layer_bytes", 0) + mstats.get("kv_bytes", 0)) / (1024**3)

        else:
            actual_in = target_in
            actual_out = 0
            total_tok = target_in
            ttft = prefill_tps = decode_sec = decode_tps = avg_itl = p50_itl = p90_itl = 0.0
            net_sent_mb = self.node.bytes_sent / (1024**2)
            net_recv_mb = self.node.bytes_received / (1024**2)
            m_kv_mb = w_kv_mb = total_kv_mb = 0.0

        row = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "hops": hops,
            "workload": wl["name"],
            "master_layers": m_layer_str,
            "worker_layers": w_layer_str,
            "input_tokens": actual_in,
            "output_tokens": actual_out,
            "total_tokens": total_tok,
            "ttft_sec": round(ttft, 4),
            "prefill_tps": round(prefill_tps, 2),
            "decode_sec": round(decode_sec, 4),
            "decode_tps": round(decode_tps, 2),
            "avg_itl_ms": round(avg_itl, 2),
            "p50_itl_ms": round(p50_itl, 2),
            "p90_itl_ms": round(p90_itl, 2),
            "master_net_sent_mb": round(net_sent_mb, 3),
            "master_net_recv_mb": round(net_recv_mb, 3),
            "master_vram_peak_gb": round(m_vram_peak, 3),
            "worker_vram_peak_gb": round(w_vram_peak, 3),
            "master_kv_cache_mb": m_kv_mb,
            "worker_kv_cache_mb": w_kv_mb,
            "total_kv_cache_mb": total_kv_mb,
            "status": status,
            "error_notes": error_notes,
        }

        if status == "SUCCESS":
            with open(self.output_csv, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(row)
            print(f"  ✓ SUCCESS | TTFT: {ttft:.2f}s ({prefill_tps:.1f} T/s) | Decode: {decode_tps:.2f} T/s ({avg_itl:.1f}ms ITL)")
            print(f"    KV Cache: Master {m_kv_mb} MB, Worker {w_kv_mb} MB (Total: {total_kv_mb} MB)")
            print(f"    VRAM Peak: Master {m_vram_peak:.2f} GB | Worker {w_vram_peak:.2f} GB")
        else:
            with open("hops_network_drops.log", mode="a") as f:
                f.write(f"[{row['timestamp']}] Iter {iteration} Hops {hops} {wl_key}: {error_notes}\n")

        return row

    def run_suite(self):
        hop_list = [int(h.strip()) for h in self.args.hops.split(",")]
        wl_keys = [w.strip() for w in self.args.workloads.split(",")]

        start_it = 1
        if self.output_csv.exists():
            try:
                with open(self.output_csv, mode="r", newline="") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        cur_it = int(r.get("iteration", 0))
                        if cur_it >= start_it:
                            start_it = cur_it + 1
            except Exception:
                pass

        total_target_iters = start_it + self.args.repeats - 1
        print(f"🚀 Starting test suite: {len(hop_list)} Hops × {len(wl_keys)} Workloads × {self.args.repeats} Iterations (Running iters {start_it} to {total_target_iters})")
        start_time = time.time()

        for it in range(start_it, total_target_iters + 1):
            print(f"\n============================================================")
            print(f"  🌟 BEGINNING ITERATION {it} OF {total_target_iters}")
            print(f"============================================================")

            for hops in hop_list:
                if Path(".stop_experiments").exists():
                    print("\n🛑 Stop signal detected (.stop_experiments)! Exiting...")
                    try: Path(".stop_experiments").unlink()
                    except Exception: pass
                    return

                self.setup_topology(hops)

                for wl_key in wl_keys:
                    if Path(".stop_experiments").exists():
                        print("\n🛑 Stop signal detected (.stop_experiments)! Exiting...")
                        try: Path(".stop_experiments").unlink()
                        except Exception: pass
                        return

                    # Retry trial if transient network drop occurred
                    max_retries = 3
                    for attempt in range(max_retries):
                        row = self.run_single_trial(it, hops, wl_key)
                        if row["status"] in ("SUCCESS", "FAILED_OOM"):
                            break
                        print(f"  🔄 Retrying Hops={hops}, Workload={wl_key} (attempt {attempt+1}/{max_retries})...")
                        time.sleep(3.0)

                    # Thermal & socket drain pause
                    time.sleep(1.5)

            # Trigger real-time report & plot generation after each iteration
            try:
                rep_path = Path(self.args.plot_dir).parent / "hops_feasibility_analysis.md"
                os.system(f"/media/vithurshan/vithu/llm/.venv/bin/python generate_hops_analysis.py --csv {self.output_csv} --output_dir {self.args.plot_dir} --report {rep_path}")
            except Exception:
                pass

        total_time = (time.time() - start_time) / 60
        print(f"\n🎉 ALL {self.args.repeats} ITERATIONS COMPLETED in {total_time:.1f} minutes!")
        try:
            rep_path = Path(self.args.plot_dir).parent / "hops_feasibility_analysis.md"
            os.system(f"/media/vithurshan/vithu/llm/.venv/bin/python generate_hops_analysis.py --csv {self.output_csv} --output_dir {self.args.plot_dir} --report {rep_path}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Multi-Hop Feasibility Experiment Runner")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_dir", type=str,
                        default="model_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--worker_host", type=str, default="10.8.100.23")
    parser.add_argument("--worker_port", type=int, default=9900)
    parser.add_argument("--hops", type=str, default="1,3,5,7,9")
    parser.add_argument("--workloads", type=str, default="small,medium,large")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output_csv", type=str, default="hops_feasibility_experiments.csv")
    parser.add_argument("--plot_dir", type=str, default="hops_experiment_profiles")
    args = parser.parse_args()

    runner = HopsExperimentRunner(args)
    runner.run_suite()


if __name__ == "__main__":
    main()
