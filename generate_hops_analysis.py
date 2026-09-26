#!/usr/bin/env python3
"""
Scientific Analysis & Graph Generator for Multi-Hop Feasibility Experiment (1, 3, 5, 7, 9 Hops).
Directly replicates T.png (Time to First Token Vs Number of Hops) and D.png (Decode Token Per Second Vs Num of Hop).
Generates average value charts across n runs with error bars and writes hops_feasibility_analysis.md.
"""

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
CSV_PATH = CURRENT_DIR / "hops_feasibility_experiments.csv"
REPORT_PATH = CURRENT_DIR / "hops_feasibility_analysis.md"
OUTPUT_DIR = CURRENT_DIR / "hops_experiment_profiles"


def load_data(csv_path: Path) -> List[Dict[str, any]]:
    if not csv_path.exists():
        return []
    records = []
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "SUCCESS":
                records.append(row)
    return records


def generate_plots(records: List[Dict[str, any]], output_dir: Path):
    if not records:
        print("No successful records to plot yet.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group data by (workload, hops)
    data = defaultdict(lambda: defaultdict(lambda: {"ttft": [], "decode_tps": []}))
    workload_names = set()
    all_hops = set()

    for r in records:
        wl = r["workload"]
        h = int(r["hops"])
        ttft = float(r["ttft_sec"])
        tps = float(r["decode_tps"])
        data[wl][h]["ttft"].append(ttft)
        data[wl][h]["decode_tps"].append(tps)
        workload_names.add(wl)
        all_hops.add(h)

    sorted_hops = sorted(list(all_hops))

    # 1. Replicate T.png (Time to First Token Vs Number of Hops)
    # Style matching T.png: light grey gridlines, royal blue bars (#3B82F6 / #4285F4), clean Arial font
    for wl in workload_names:
        hops_list = sorted(data[wl].keys())
        mean_ttft = [np.mean(data[wl][h]["ttft"]) for h in hops_list]
        std_ttft = [np.std(data[wl][h]["ttft"]) for h in hops_list]

        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
        x_positions = np.arange(len(hops_list))
        bars = ax.bar(x_positions, mean_ttft, yerr=std_ttft, capsize=4, width=0.45,
                      color="#3b82f6", edgecolor="#2563eb", alpha=0.9, zorder=3)

        ax.set_title(f"Time to First Token Vs Number of Hops ({wl})", fontsize=13, fontweight="bold", color="#374151", pad=12)
        ax.set_xlabel("Number of Hops", fontsize=10, fontweight="medium", color="#4b5563", labelpad=8)
        ax.set_ylabel("TTFT (seconds)", fontsize=10, fontweight="medium", color="#4b5563", labelpad=8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{h:.2f}" for h in hops_list], fontsize=9)
        ax.grid(axis="y", linestyle="-", color="#e5e7eb", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d1d5db")
        ax.spines["bottom"].set_color("#d1d5db")

        # Value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(mean_ttft)*0.02), f"{yval:.2f}s",
                    ha="center", va="bottom", fontsize=8.5, color="#1f2937", fontweight="bold")

        wl_slug = wl.lower().replace(" ", "_")
        plt.tight_layout()
        plt.savefig(output_dir / f"time_to_first_token_vs_hops_{wl_slug}.png")
        if wl == "Medium Chat" or len(workload_names) == 1:
            plt.savefig(output_dir / "time_to_first_token_vs_hops.png")
            plt.savefig(CURRENT_DIR / "time_to_first_token_vs_hops.png")
        plt.close()

    # 2. Replicate D.png (Decode Token Per Second Vs Num of Hop)
    for wl in workload_names:
        hops_list = sorted(data[wl].keys())
        mean_tps = [np.mean(data[wl][h]["decode_tps"]) for h in hops_list]
        std_tps = [np.std(data[wl][h]["decode_tps"]) for h in hops_list]

        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
        x_positions = np.arange(len(hops_list))
        bars = ax.bar(x_positions, mean_tps, yerr=std_tps, capsize=4, width=0.45,
                      color="#3b82f6", edgecolor="#2563eb", alpha=0.9, zorder=3)

        ax.set_title(f"Decode Token Per Second Vs Num of Hop ({wl})", fontsize=13, fontweight="bold", color="#374151", pad=12)
        ax.set_xlabel("Number of Hops", fontsize=10, fontweight="medium", color="#4b5563", labelpad=8)
        ax.set_ylabel("TPS (Decode)", fontsize=10, fontweight="medium", color="#4b5563", labelpad=8)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{h:.2f}" for h in hops_list], fontsize=9)
        ax.grid(axis="y", linestyle="-", color="#e5e7eb", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d1d5db")
        ax.spines["bottom"].set_color("#d1d5db")

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(mean_tps)*0.02 if max(mean_tps)>0 else 0.1), f"{yval:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color="#1f2937", fontweight="bold")

        wl_slug = wl.lower().replace(" ", "_")
        plt.tight_layout()
        plt.savefig(output_dir / f"decode_tps_vs_hops_{wl_slug}.png")
        if wl == "Medium Chat" or len(workload_names) == 1:
            plt.savefig(output_dir / "decode_tps_vs_hops.png")
            plt.savefig(CURRENT_DIR / "decode_tps_vs_hops.png")
        plt.close()

    # 3. Combined Grouped Bar Chart comparing Small, Medium, Large side-by-side
    if len(workload_names) > 1:
        # Combined TTFT
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
        width = 0.25
        colors = {"Small Chat": "#10b981", "Medium Chat": "#3b82f6", "Large Context": "#8b5cf6"}
        x = np.arange(len(sorted_hops))
        for idx, wl in enumerate(sorted(workload_names)):
            means = [np.mean(data[wl][h]["ttft"]) if h in data[wl] else 0.0 for h in sorted_hops]
            stds = [np.std(data[wl][h]["ttft"]) if h in data[wl] else 0.0 for h in sorted_hops]
            offset = (idx - 1) * width
            ax.bar(x + offset, means, yerr=stds, width=width, label=wl, color=colors.get(wl, "#6b7280"), alpha=0.85, capsize=3)
        ax.set_title("Time to First Token Vs Number of Hops (All Workloads)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Hops", fontsize=10, labelpad=8)
        ax.set_ylabel("Average TTFT (seconds)", fontsize=10, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h} Hops" for h in sorted_hops], fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "combined_ttft_vs_hops.png")
        plt.savefig(CURRENT_DIR / "combined_ttft_vs_hops.png")
        plt.close()

        # Combined Decode TPS
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
        for idx, wl in enumerate(sorted(workload_names)):
            means = [np.mean(data[wl][h]["decode_tps"]) if h in data[wl] else 0.0 for h in sorted_hops]
            stds = [np.std(data[wl][h]["decode_tps"]) if h in data[wl] else 0.0 for h in sorted_hops]
            offset = (idx - 1) * width
            ax.bar(x + offset, means, yerr=stds, width=width, label=wl, color=colors.get(wl, "#6b7280"), alpha=0.85, capsize=3)
        ax.set_title("Decode Token Per Second Vs Number of Hops (All Workloads)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Hops", fontsize=10, labelpad=8)
        ax.set_ylabel("Average Decode TPS (tokens/s)", fontsize=10, labelpad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h} Hops" for h in sorted_hops], fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "combined_decode_tps_vs_hops.png")
        plt.savefig(CURRENT_DIR / "combined_decode_tps_vs_hops.png")
        plt.close()

    print(f"✓ Feasibility plots generated in {output_dir}")


def generate_report(records: List[Dict[str, any]], report_path: Path):
    if not records:
        return

    # Group by (hops, workload)
    grouped = defaultdict(list)
    for r in records:
        key = (int(r["hops"]), r["workload"])
        grouped[key].append(r)

    md = []
    md.append("# Multi-Hop Decentralized Inference Feasibility Report\n")
    md.append(f"**Evaluation of 1, 3, 5, 7, 9 Network Hops across Small, Medium, and Large Workloads**\n")
    md.append(f"- **Hardware Setup**: Master Laptop (RTX 3060 6GB VRAM) + Worker Server `csetuf09` (RTX 4070 Ti Super 16GB VRAM)")
    md.append(f"- **Model**: `Qwen/Qwen2.5-7B-Instruct` (28 Layers, FP16)")
    md.append(f"- **Total Completed Trial Runs**: {len(records)}")
    md.append(f"- **Report Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    md.append("---\n")
    md.append("## 1. Summary Performance Table (Mean ± Standard Deviation)\n")
    md.append("| Hops | Workload | Master Layers | Worker Layers | Tokens (In/Out) | TTFT (s) | Decode TPS | Avg ITL (ms) | Master KV (MB) | Worker KV (MB) | Total KV (MB) | Net I/O (MB) |")
    md.append("| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for (h, wl), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        m_ttft = np.mean([float(i["ttft_sec"]) for i in items])
        s_ttft = np.std([float(i["ttft_sec"]) for i in items])
        m_tps = np.mean([float(i["decode_tps"]) for i in items])
        s_tps = np.std([float(i["decode_tps"]) for i in items])
        m_itl = np.mean([float(i["avg_itl_ms"]) for i in items])
        s_itl = np.std([float(i["avg_itl_ms"]) for i in items])

        first = items[0]
        m_layers = first["master_layers"]
        w_layers = first["worker_layers"]
        in_tok = first["input_tokens"]
        out_tok = first["output_tokens"]

        m_mkv = np.mean([float(i["master_kv_cache_mb"]) for i in items])
        m_wkv = np.mean([float(i["worker_kv_cache_mb"]) for i in items])
        m_tkv = np.mean([float(i["total_kv_cache_mb"]) for i in items])
        m_net = np.mean([float(i["master_net_sent_mb"]) + float(i["master_net_recv_mb"]) for i in items])

        md.append(f"| **{h}** | {wl} | `{m_layers}` | `{w_layers}` | {in_tok} / {out_tok} | {m_ttft:.2f} ± {s_ttft:.2f}s | {m_tps:.2f} ± {s_tps:.2f} | {m_itl:.1f} ± {s_itl:.1f} | {m_mkv:.2f} | {m_wkv:.2f} | {m_tkv:.2f} | {m_net:.2f} |")

    md.append("\n---\n")
    md.append("## 2. Feasibility Curves (Replication of T.png & D.png)\n")
    md.append("### Time to First Token (TTFT) Vs Number of Hops\n")
    md.append("![Time to First Token Vs Number of Hops](time_to_first_token_vs_hops.png)\n\n")
    md.append("### Decode Token Per Second Vs Number of Hops\n")
    md.append("![Decode Token Per Second Vs Number of Hops](decode_tps_vs_hops.png)\n\n")

    with open(report_path, "w") as f:
        f.write("\n".join(md))
    print(f"✓ Analysis report written to {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Hop Feasibility Analysis & Plot Generator")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Path to input experiments CSV")
    parser.add_argument("--report", type=str, default=str(REPORT_PATH), help="Path to output markdown report")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Directory to save generated plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    report_path = Path(args.report)
    output_dir = Path(args.output_dir)

    records = load_data(csv_path)
    generate_plots(records, output_dir)
    generate_report(records, report_path)

    # Sync to final or multi_hop_experiment_results if applicable
    import shutil
    def safe_copy(src: Path, dst: Path):
        try:
            if src.resolve() != dst.resolve():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        except Exception:
            pass

    if "final" in str(output_dir) or "final" in str(csv_path):
        final_dir = CURRENT_DIR / "final"
        if final_dir.exists():
            safe_copy(csv_path, final_dir / "data" / csv_path.name)
            safe_copy(csv_path, final_dir / csv_path.name)
            safe_copy(report_path, final_dir / "reports" / report_path.name)
            safe_copy(report_path, final_dir / report_path.name)
            for p in output_dir.glob("*.png"):
                safe_copy(p, final_dir / "plots" / p.name)
                safe_copy(p, final_dir / p.name)
            print("✓ Synchronized updated results into final/")
    else:
        res_dir = CURRENT_DIR / "multi_hop_experiment_results"
        if res_dir.exists():
            safe_copy(csv_path, res_dir / "data" / csv_path.name)
            safe_copy(csv_path, res_dir / csv_path.name)
            safe_copy(report_path, res_dir / "reports" / report_path.name)
            safe_copy(report_path, res_dir / report_path.name)
            for p in output_dir.glob("*.png"):
                safe_copy(p, res_dir / "plots" / p.name)
                safe_copy(p, res_dir / p.name)
            print("✓ Synchronized updated results into multi_hop_experiment_results/")


if __name__ == "__main__":
    main()
