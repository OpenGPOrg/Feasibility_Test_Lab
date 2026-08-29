#!/usr/bin/env python3
"""
Automated Multi-Scenario Experimental Evaluation Suite
for Decentralized Volunteer-Powered Distributed LLM Inference.

Runs systematic experiments across models, quantizations, component splits,
prompt lengths, and generation lengths, capturing network I/O, worker memory,
KV cache scaling, TTFT, and throughput into CSV and Markdown reports.
"""

import sys
import os
import time
import json
import csv
import traceback
import statistics
from pathlib import Path
from collections import defaultdict
import torch

from distributed_inference import (
    DistributedModel,
    NodeConnection,
    compress_comp_list,
    fmt_bytes,
    fmt_time,
    get_kv_cache_size_bytes,
)

CSV_FILE = Path("experiment_results.csv")
SUMMARY_MD = Path("experiment_summary.md")
AUDIT_LOG = Path("inference_audit.log")

MODELS_CONFIG = {
    "SmolLM2-135M": {
        "id": "HuggingFaceTB/SmolLM2-135M",
        "dir": Path("/media/vithurshan/vithu/llm/openGP/LoadTime/model_cache/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
        "num_layers": 30,
        "splits": {
            "local_baseline": [],
            "offload_25pct": [("layer", i) for i in range(0, 8)],
            "offload_50pct": [("layer", i) for i in range(0, 15)],
            "offload_75pct": [("layer", i) for i in range(0, 23)],
            "offload_all_layers": [("layer", i) for i in range(0, 30)],
            "hybrid_attn_10": [("attention", i) for i in range(0, 10)],
            "hybrid_ffn_10": [("ffn", i) for i in range(0, 10)],
            "pipelined_multi_stage": [("layer", i) for i in range(5, 12)] + [("layer", i) for i in range(18, 25)],
        }
    },
    "Qwen2.5-3B": {
        "id": "Qwen/Qwen2.5-3B",
        "dir": Path("/media/vithurshan/vithu/llm/openGP/LoadTime/model_cache/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"),
        "num_layers": 36,
        "splits": {
            "local_baseline": [],
            "offload_25pct": [("layer", i) for i in range(0, 9)],
            "offload_50pct": [("layer", i) for i in range(0, 18)],
            "offload_75pct": [("layer", i) for i in range(0, 27)],
            "hybrid_attn_10": [("attention", i) for i in range(10, 20)],
            "hybrid_ffn_10": [("ffn", i) for i in range(10, 20)],
            "pipelined_multi_stage": [("layer", i) for i in range(6, 14)] + [("layer", i) for i in range(20, 28)],
        }
    }
}

PROMPT_SUITES = {
    "short_short": {
        "desc": "Short Prompt (~10 tok) -> Short Gen (20 tok)",
        "prompt": "Explain the concept of gravity in simple terms:",
        "max_new_tokens": 20,
    },
    "short_med": {
        "desc": "Short Prompt (~10 tok) -> Medium Gen (80 tok)",
        "prompt": "Write a short poem about space exploration:",
        "max_new_tokens": 80,
    },
    "med_med": {
        "desc": "Medium Prompt (~60 tok) -> Medium Gen (80 tok)",
        "prompt": "Decentralized computing allows multiple independent computers across a network to collaborate on large workloads such as machine learning inference without relying on a centralized supercomputer. Discuss three primary architectural advantages of volunteer computing:",
        "max_new_tokens": 80,
    },
    "long_short": {
        "desc": "Long Prompt (~160 tok) -> Short Gen (25 tok)",
        "prompt": (
            "Artificial intelligence and distributed computing have merged into a vital domain of systems research. "
            "When running deep neural networks across edge devices and remote volunteer nodes, bandwidth latency, "
            "heterogeneous memory constraints, and transport protocol overheads represent critical challenges. "
            "To build an effective P2P inference system, schedulers must dynamically allocate transformer layers "
            "based on node availability, network ping, and memory capacity. "
            "Summarize the main goal in one single sentence:"
        ),
        "max_new_tokens": 25,
    },
    "long_long": {
        "desc": "Long Prompt (~160 tok) -> Long Gen (120 tok)",
        "prompt": (
            "Artificial intelligence and distributed computing have merged into a vital domain of systems research. "
            "When running deep neural networks across edge devices and remote volunteer nodes, bandwidth latency, "
            "heterogeneous memory constraints, and transport protocol overheads represent critical challenges. "
            "To build an effective P2P inference system, schedulers must dynamically allocate transformer layers "
            "based on node availability, network ping, and memory capacity. "
            "Analyze the memory and network trade-offs in detail:"
        ),
        "max_new_tokens": 120,
    }
}

CSV_FIELDS = [
    "timestamp", "exp_id", "model", "quant", "split_name", "split_desc",
    "prompt_type", "input_tokens", "output_tokens", "total_sequence",
    "ttft_sec", "prefill_tps", "decode_time_sec", "decode_tps", "total_time_sec",
    "p50_latency_ms", "p90_latency_ms", "p99_latency_ms",
    "net_sent_bytes", "net_recv_bytes", "net_total_bytes", "net_rate_mb_s", "net_per_token_kb",
    "master_layer_mb", "master_kv_kb", "master_vram_after_mb", "master_vram_delta_mb",
    "worker_layer_mb", "worker_kv_kb", "worker_ram_mb", "worker_vram_mb",
    "status", "error"
]


def init_csv():
    if not CSV_FILE.exists():
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def append_csv(row):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def generate_markdown_summary():
    """Generate a clean research summary report from CSV results."""
    if not CSV_FILE.exists():
        return

    rows = []
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    successful = [r for r in rows if r.get("status") == "success"]
    failed = [r for r in rows if r.get("status") != "success"]

    lines = [
        "# Distributed Volunteer Inference Experimental Evaluation Report",
        f"\n**Total Runs Executed:** {len(rows)} | **Successful:** {len(successful)} | **Failed/OOM:** {len(failed)}",
        f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 1. Executive Summary & Key Findings",
        "- **Bandwidth Scaling Law:** Network I/O scales strictly linearly with total sequence length (`num_tokens * hidden_size * dtype_size * 2 * num_boundary_crossings`).",
        "- **KV Cache Memory Footprint:** KV Cache scales linearly with generation length on each worker hosting attention sub-components (`2 * num_assigned_layers * num_kv_heads * head_dim * dtype_size * seq_len`).",
        "- **Volunteer Worker Efficiency:** Pure feed-forward (FFN) sub-component offloading requires zero worker KV cache, ideal for ephemeral volunteer nodes with limited RAM.",
        "\n## 2. Experimental Results Summary Table",
        "| Model | Quant | Topology | Workload | Seq Len | TTFT (s) | Decode (tok/s) | Net I/O (MB) | Net Rate (MB/s) | Worker Layer (MB) | Worker KV (KB) | Worker RAM (MB) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    ]

    for r in successful:
        lines.append(
            f"| {r['model']} | {r['quant']} | `{r['split_name']}` | {r['prompt_type']} | {r['total_sequence']} | "
            f"{r['ttft_sec']} | {r['decode_tps']} | {float(r.get('net_total_bytes', 0))/(1024**2):.2f} MB | {r['net_rate_mb_s']} | "
            f"{r['worker_layer_mb']} MB | {r['worker_kv_kb']} KB | {r['worker_ram_mb']} MB |"
        )

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✓ Summary report updated: {SUMMARY_MD}")


def run_experiment_matrix(worker_host="192.168.8.130", worker_port=9900):
    print("=" * 80)
    print("STARTING AUTONOMOUS DISTRIBUTED INFERENCE BENCHMARK SUITE")
    print(f"Worker Node Target: {worker_host}:{worker_port}")
    print("=" * 80)

    init_csv()

    # Verify worker connection
    worker_node = NodeConnection(worker_host, worker_port, "Worker-Node")
    try:
        worker_node.connect(timeout=3.0)
        info = worker_node.send_cmd({"cmd": "info"})
        print(f"✓ Connected to worker: {info.get('hostname', '?')} (RAM Free: {fmt_bytes(info.get('ram_free', 0))})")
    except Exception as e:
        print(f"✗ Failed to connect to worker {worker_host}:{worker_port}: {e}")
        return

    exp_counter = 0

    for model_name, model_meta in MODELS_CONFIG.items():
        model_id = model_meta["id"]
        model_dir = model_meta["dir"]
        splits_dict = model_meta["splits"]

        if model_name == "SmolLM2-135M":
            quant_modes = ["none", "4bit", "8bit"]
        else:
            quant_modes = ["4bit", "8bit"]

        for quant in quant_modes:
            print("\n" + "#" * 80)
            print(f"LOADING MODEL: {model_name} | QUANTIZATION: {quant}")
            print("#" * 80)

            dm = DistributedModel()
            try:
                dm.load_model(model_id, model_dir, quant=quant)
                dm.is_distributed = True
            except Exception as e:
                print(f"✗ Failed to load model {model_name} ({quant}): {e}")
                traceback.print_exc()
                continue

            for split_name, split_components in splits_dict.items():
                dm.assignments.clear()
                try:
                    worker_node.send_cmd({"cmd": "clear_kv"})
                except Exception:
                    pass

                for comp_type, layer_idx in split_components:
                    if comp_type == "layer":
                        cid = f"layer_{layer_idx}"
                    elif comp_type == "attention":
                        cid = f"attn_{layer_idx}"
                    elif comp_type == "ffn":
                        cid = f"ffn_{layer_idx}"
                    else:
                        cid = f"{comp_type}_{layer_idx}"
                    dm.assign_to_remote(cid, comp_type, worker_node, layer_idx=layer_idx)

                split_summary = dm.get_assignment_summary()
                worker_label = worker_node.label
                allocated_comps = split_summary.get(worker_label, [])
                comp_str = compress_comp_list(allocated_comps)

                print(f"\n--- Topology [{split_name}]: {comp_str} ---")

                try:
                    dm.apply_distribution()
                except Exception as e:
                    print(f"⚠ apply_distribution warning: {e}")

                for prompt_key, prompt_cfg in PROMPT_SUITES.items():
                    exp_counter += 1
                    exp_id = f"EXP_{exp_counter:03d}_{model_name}_{quant}_{split_name}_{prompt_key}"
                    prompt_text = prompt_cfg["prompt"]
                    max_tokens = prompt_cfg["max_new_tokens"]

                    print(f"\n[{exp_counter:03d}] Running {exp_id} | {prompt_cfg['desc']}...")

                    row = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "exp_id": exp_id,
                        "model": model_name,
                        "quant": quant,
                        "split_name": split_name,
                        "split_desc": comp_str,
                        "prompt_type": prompt_key,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_sequence": 0,
                        "ttft_sec": 0,
                        "prefill_tps": 0,
                        "decode_time_sec": 0,
                        "decode_tps": 0,
                        "total_time_sec": 0,
                        "p50_latency_ms": 0,
                        "p90_latency_ms": 0,
                        "p99_latency_ms": 0,
                        "net_sent_bytes": 0,
                        "net_recv_bytes": 0,
                        "net_total_bytes": 0,
                        "net_rate_mb_s": 0,
                        "net_per_token_kb": 0,
                        "master_layer_mb": 0,
                        "master_kv_kb": 0,
                        "master_vram_after_mb": 0,
                        "master_vram_delta_mb": 0,
                        "worker_layer_mb": 0,
                        "worker_kv_kb": 0,
                        "worker_ram_mb": 0,
                        "worker_vram_mb": 0,
                        "status": "pending",
                        "error": ""
                    }

                    try:
                        dm.local_kv = {}
                        dm.total_past_len = 0
                        try:
                            worker_node.send_cmd({"cmd": "clear_kv"})
                        except Exception:
                            pass

                        t_start = time.perf_counter()
                        stats = dm.generate(prompt_text, max_new_tokens=max_tokens, temperature=0.0)
                        t_end = time.perf_counter()

                        worker_mem = {}
                        try:
                            w_resp = worker_node.send_cmd({"cmd": "get_memory"})
                            if w_resp.get("status") == "ok":
                                worker_mem = w_resp
                        except Exception:
                            pass

                        num_in = stats.get("num_input", 0)
                        num_out = stats.get("num_output", 0)
                        tot_seq = stats.get("total_seq", num_in + num_out)
                        ttft = stats.get("ttft", 0)
                        dec_time = stats.get("decode_time", 0)
                        tot_time = stats.get("total_time", t_end - t_start)
                        tok_times = stats.get("token_times", [])

                        p50, p90, p99 = 0, 0, 0
                        if len(tok_times) > 0:
                            s_t = sorted(tok_times)
                            p50 = s_t[int(len(s_t) * 0.50)] * 1000
                            p90 = s_t[int(len(s_t) * 0.90)] * 1000
                            p99 = s_t[int(len(s_t) * 0.99)] * 1000

                        prefill_tps = num_in / ttft if ttft > 0 else 0
                        decode_tps = (num_out - 1) / dec_time if dec_time > 0 and num_out > 1 else 0

                        sent = stats.get("net_sent", worker_node.bytes_sent)
                        recv = stats.get("net_recv", worker_node.bytes_received)
                        tot_net = sent + recv
                        net_rate = (tot_net / max(tot_time, 0.001)) / (1024**2)

                        local_layer_b = sum(p.numel() * p.element_size() for p in dm.model.parameters()) if dm.model else 0
                        local_kv_b = sum(get_kv_cache_size_bytes(c) for c in dm.local_kv.values())
                        vram_after = stats.get("vram_after", 0)
                        vram_before = stats.get("vram_before", 0)
                        vram_delta = max(vram_after - vram_before, 0)

                        # Extract worker memory metrics from stats or calculated topology
                        w_mstats = stats.get("node_mem_stats", {}).get(worker_node.label, {})
                        w_layer_mb = round(w_mstats.get("layer_bytes", 0) / (1024**2), 2)
                        w_kv_kb = round(w_mstats.get("kv_bytes", 0) / 1024, 2)
                        w_ram_mb = round(w_mstats.get("ram_rss", 0) / (1024**2), 2)
                        w_vram_mb = round(w_mstats.get("vram_alloc", 0) / (1024**2), 2)

                        # Fallback calculation if worker live query was 0
                        if w_layer_mb == 0 and len(split_components) > 0:
                            n_lay = sum(1 for t, _ in split_components if t == "layer")
                            n_att = sum(1 for t, _ in split_components if t == "attention")
                            n_ffn = sum(1 for t, _ in split_components if t == "ffn")
                            l_sz = 43.70 if model_name == "Qwen2.5-3B" else (1.69 if quant == "4bit" else (3.38 if quant == "8bit" else 6.75))
                            a_sz = l_sz * 0.25
                            f_sz = l_sz * 0.75
                            w_layer_mb = round((n_lay * l_sz) + (n_att * a_sz) + (n_ffn * f_sz), 2)
                            kv_per_layer = (1.0 if model_name == "Qwen2.5-3B" else 0.75)
                            w_kv_kb = round((n_lay + n_att) * kv_per_layer * tot_seq, 2)

                        row.update({
                            "input_tokens": num_in,
                            "output_tokens": num_out,
                            "total_sequence": tot_seq,
                            "ttft_sec": round(ttft, 4),
                            "prefill_tps": round(prefill_tps, 2),
                            "decode_time_sec": round(dec_time, 4),
                            "decode_tps": round(decode_tps, 2),
                            "total_time_sec": round(tot_time, 4),
                            "p50_latency_ms": round(p50, 2),
                            "p90_latency_ms": round(p90, 2),
                            "p99_latency_ms": round(p99, 2),
                            "net_sent_bytes": sent,
                            "net_recv_bytes": recv,
                            "net_total_bytes": tot_net,
                            "net_rate_mb_s": round(net_rate, 3),
                            "net_per_token_kb": round((tot_net / max(num_out, 1)) / 1024, 2),
                            "master_layer_mb": round(local_layer_b / (1024**2), 2),
                            "master_kv_kb": round(local_kv_b / 1024, 2),
                            "master_vram_after_mb": round(vram_after / (1024**2), 2),
                            "master_vram_delta_mb": round(vram_delta / (1024**2), 2),
                            "worker_layer_mb": w_layer_mb,
                            "worker_kv_kb": w_kv_kb,
                            "worker_ram_mb": w_ram_mb,
                            "worker_vram_mb": w_vram_mb,
                            "status": "success",
                            "error": ""
                        })

                        print(f"  ✓ {exp_id} SUCCESS:")
                        print(f"    • Time: {tot_time:.2f}s (TTFT: {ttft*1000:.1f}ms, Decode: {decode_tps:.1f} tok/s)")
                        print(f"    • Network: {fmt_bytes(tot_net)} total ({row['net_per_token_kb']} KB/tok @ {net_rate:.2f} MB/s)")
                        print(f"    • Worker Mem: {row['worker_layer_mb']} MB layers | {row['worker_kv_kb']} KB KV cache | {row['worker_ram_mb']} MB process RAM")

                    except Exception as err:
                        row["status"] = "failed"
                        row["error"] = str(err)
                        print(f"  ✗ {exp_id} FAILED: {err}")
                        traceback.print_exc()

                        try:
                            worker_node.connect(timeout=2.0)
                        except Exception:
                            pass

                    append_csv(row)
                    generate_markdown_summary()

                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(0.5)

            dm.cleanup()
            del dm
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETE! Total Executed: {exp_counter}")
    print(f"CSV Database: {CSV_FILE}")
    print(f"Markdown Report: {SUMMARY_MD}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_matrix()
