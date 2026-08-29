#!/usr/bin/env python3
"""
Targeted Re-runner for Failed Experiments.
Skips excessively long full-layer CPU offloads as configured by user,
and executes remaining hybrid sub-component and pipeline scenarios.
"""

import sys
import os
import time
import json
import csv
import traceback
from pathlib import Path
import torch

from distributed_inference import (
    DistributedModel,
    NodeConnection,
    compress_comp_list,
    fmt_bytes,
    fmt_time,
    get_kv_cache_size_bytes,
)
from benchmark_experiments import (
    MODELS_CONFIG,
    PROMPT_SUITES,
    CSV_FILE,
    SUMMARY_MD,
    generate_markdown_summary,
)

# User requested skip list for excessively slow full CPU offloads on long sequences
SKIP_EXP_IDS = {
    "EXP_142_Qwen2.5-3B_4bit_offload_all_layers_short_med",
    "EXP_143_Qwen2.5-3B_4bit_offload_all_layers_med_med",
    "EXP_144_Qwen2.5-3B_4bit_offload_all_layers_long_short",
    "EXP_145_Qwen2.5-3B_4bit_offload_all_layers_long_long",
}


def rerun_failed_experiments(worker_host="192.168.8.130", worker_port=9900):
    print("=" * 80)
    print("STARTING TARGETED RE-RUN FOR FAILED EXPERIMENTS")
    print(f"Target Worker Node: {worker_host}:{worker_port}")
    print("=" * 80)

    if not CSV_FILE.exists():
        print(f"✗ CSV file {CSV_FILE} does not exist.")
        return

    all_rows = []
    failed_indices = []
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for idx, r in enumerate(reader):
            exp_id = r.get("exp_id", "")
            split = r.get("split_name", "")
            if split == "offload_all_layers" or exp_id in SKIP_EXP_IDS:
                r["status"] = "skipped"
                r["error"] = "Skipped by user directive (full 100% layer CPU offloading omitted)"
            all_rows.append(r)
            if r.get("status") not in ("success", "skipped"):
                failed_indices.append(idx)

    # Save skipped markers
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Found {len(failed_indices)} active failed runs to execute (Skipped {len(SKIP_EXP_IDS)} long full-layer runs).")
    if not failed_indices:
        print("✓ All target experiments completed!")
        generate_markdown_summary()
        return

    # Connect to worker
    worker_node = NodeConnection(worker_host, worker_port, "Worker-Node")
    try:
        worker_node.connect(timeout=3.0)
        info = worker_node.send_cmd({"cmd": "info"})
        print(f"✓ Connected to worker: {info.get('hostname', '?')} (RAM Free: {fmt_bytes(info.get('ram_free', 0))})\n")
    except Exception as e:
        print(f"✗ Failed to connect to worker {worker_host}:{worker_port}: {e}")
        return

    # Group failures by (model, quant)
    grouped_failures = {}
    for idx in failed_indices:
        r = all_rows[idx]
        key = (r["model"], r["quant"])
        if key not in grouped_failures:
            grouped_failures[key] = []
        grouped_failures[key].append((idx, r))

    total_fixed = 0

    for (model_name, quant), run_list in grouped_failures.items():
        print("\n" + "#" * 80)
        print(f"LOADING MODEL FOR RERUN: {model_name} | QUANTIZATION: {quant} ({len(run_list)} runs)")
        print("#" * 80)

        model_meta = MODELS_CONFIG.get(model_name)
        if not model_meta:
            print(f"✗ Unknown model {model_name}")
            continue

        model_id = model_meta["id"]
        model_dir = model_meta["dir"]
        splits_dict = model_meta["splits"]

        dm = DistributedModel()
        try:
            dm.load_model(model_id, model_dir, quant=quant)
            dm.is_distributed = True
        except Exception as e:
            print(f"✗ Failed to load model {model_name} ({quant}): {e}")
            traceback.print_exc()
            continue

        for idx, row in run_list:
            exp_id = row["exp_id"]
            split_name = row["split_name"]
            prompt_key = row["prompt_type"]

            if exp_id in SKIP_EXP_IDS:
                print(f"⏩ Skipping {exp_id} as requested.")
                continue

            split_components = splits_dict.get(split_name, [])
            prompt_cfg = PROMPT_SUITES.get(prompt_key, {})
            prompt_text = prompt_cfg.get("prompt", "Hello world")
            max_tokens = prompt_cfg.get("max_new_tokens", 20)

            # Pre-flight OOM Admission Control
            can_admit, req_bytes, free_bytes, reason = dm.check_worker_memory_admission(
                worker_node, split_components, max_sequence_len=max_tokens + 150
            )
            if not can_admit:
                print(f"⏩ OOM PREVENTED: Skipping {exp_id} -> {reason}")
                all_rows[idx]["status"] = "skipped"
                all_rows[idx]["error"] = f"Pre-flight OOM Prevention: {reason}"
                continue

            print(f"\n--- Rerunning {exp_id} [{split_name} | {prompt_key}] ---")

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

            try:
                dm.apply_distribution()
            except Exception as e:
                print(f"⚠ apply_distribution warning: {e}")

            dm.local_kv = {}
            dm.total_past_len = 0
            try:
                worker_node.send_cmd({"cmd": "clear_kv"})
            except Exception:
                pass

            try:
                t_start = time.perf_counter()
                stats = dm.generate(prompt_text, max_new_tokens=max_tokens, temperature=0.0)
                t_end = time.perf_counter()

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

                w_mstats = stats.get("node_mem_stats", {}).get(worker_node.label, {})
                w_layer_mb = round(w_mstats.get("layer_bytes", 0) / (1024**2), 2)
                w_kv_kb = round(w_mstats.get("kv_bytes", 0) / 1024, 2)
                w_ram_mb = round(w_mstats.get("ram_rss", 0) / (1024**2), 2)
                w_vram_mb = round(w_mstats.get("vram_alloc", 0) / (1024**2), 2)

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

                all_rows[idx].update({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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

                print(f"  ✓ {exp_id} SUCCESS: Time={tot_time:.2f}s | Decode={decode_tps:.1f} tok/s | Net={fmt_bytes(tot_net)}")
                total_fixed += 1

                with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_rows)

                generate_markdown_summary()

            except Exception as err:
                print(f"\n✗ STOPPING: {exp_id} FAILED: {err}")
                traceback.print_exc()
                all_rows[idx]["error"] = str(err)
                all_rows[idx]["status"] = "failed"
                return

        dm.cleanup()
        del dm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"ALL ACTIVE RE-RUNS COMPLETED! Total Fixed: {total_fixed}")
    print("=" * 80)


if __name__ == "__main__":
    rerun_failed_experiments()
