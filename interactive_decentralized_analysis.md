# Empirical Feasibility Analysis of Interactive LLM Inference in Decentralized Volunteer Networks

**Systematic Evaluation of Latency Thresholds, Network Bandwidth Scaling, and Hardware Capacity**

- **Experimental Setup**: Master Laptop (RTX 3060 6GB VRAM, 16GB RAM) + Worker Server `csetuf09` (RTX 4070 Ti Super 16GB VRAM, 32GB RAM)
- **Evaluated Model**: `Qwen/Qwen2.5-7B-Instruct` (28 Layers, FP16/BF16 weights)
- **Executed Trial Runs**: 28 (Successful: 26, Hardware OOM: 2, Network/Other: 0)
- **Report Generated**: 2026-09-09 10:31:45

---

## 1. Executive Summary & Objective Findings

This study investigates the practical feasibility and operational boundaries of deploying large language model (LLM) inference across decentralized volunteer clusters interconnected over commodity TCP/IP networks. We evaluate whether such architectures can satisfy standard interactive conversational benchmarks:
- **Human Conversational Fluidity**: Target Inter-Token Latency (ITL) < 100 ms (~10 tokens/sec), with an upper tolerance boundary of 300 ms.
- **Conversational Responsiveness**: Target Time-To-First-Token (TTFT) < 1.5 – 2.0 seconds.

### Empirical Characterization & Boundary Analysis
1. **Viable Interactive Regime (Coarse Partitioning / Short Context)**: When layer partitioning is coarse (T1: 1 network round-trip per token) and prompt lengths are moderate (50–150 tokens), decentralized inference achieves human-readable decoding rates (~9–15 tokens/sec, ITL ~65–110 ms). This demonstrates that volunteer offloading *can* support basic conversational tasks if network boundaries are strictly minimized.
2. **Latency Degradation with Pipeline Granularity**: As layer distribution is subdivided across more network hops (T2 to T5), each generated token serially incurs network protocol latency, packet serialization, and TCP round-trip delays. Interleaving beyond 3 hops causes ITL to exceed human conversational fluidity limits (>200–500 ms/token).
3. **Prefill Scaling in Agentic / Long-Context Workloads**: Prompt prefilling requires transmitting large intermediate activation tensors proportional to the sequence length. Under agentic and RAG contexts (1,000–3,000 tokens), activation payloads reach tens of megabytes per stage, elevating TTFT from 1–2 seconds to over 15–30 seconds. While functional for asynchronous batch or background tasks, this latency profile diverges from real-time interactive expectations.
4. **Hardware Capacity Limits**: Memory footprints on edge volunteer nodes (e.g. 6GB VRAM) tightly constrain the maximum sequence length and local layer retention, delineating clear physical thresholds where hardware runs out of memory regardless of software orchestration.


---

## 2. Quantitative Performance Table

| Topology | Master Layers | Worker Layers | Hops/Tok | Workload | Prompt/Gen | TTFT (s) | Prefill (T/s) | Decode (T/s) | Avg ITL (ms) | P90 ITL (ms) | Net (MB) | Status |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| T1 (Coarse 2-Stage) | `0-3` | `4-27` | 1 | W1 (Short Chat) | 50/32 | 4.72s | 16.6 | 13.3 | 82.8 | 105.2 | 1.21 | SUCCESS |
| T1 (Coarse 2-Stage) | `0-3` | `4-27` | 1 | W2 (Standard Chat) | 148/64 | 2.34s | 63.2 | 18.6 | 53.9 | 60.1 | 3.12 | SUCCESS |
| T1 (Coarse 2-Stage) | `0-3` | `4-27` | 1 | W3 (Detailed Assistant) | 494/128 | 3.63s | 136.2 | 13.0 | 77.0 | 186.5 | 9.06 | SUCCESS |
| T1 (Coarse 2-Stage) | `0-3` | `4-27` | 1 | W4 (Agentic Context) | 986/128 | 7.04s | 140.2 | 14.0 | 71.7 | 169.2 | 16.03 | SUCCESS |
| T1 (Coarse 2-Stage) | `0-3` | `4-27` | 1 | W5 (RAG Document Retrieval) | 1972/256 | 14.96s | 131.8 | 9.9 | 101.0 | 202.6 | 32.07 | SUCCESS |
| T1 (Coarse 2-Stage) | `0-3` | `4-27` | 1 | W6 (Heavy Codebase Stress) | 2956/256 | 73.91s | 40.0 | 13.3 | 75.3 | 188.2 | 46.01 | SUCCESS |
| T1 (Coarse 2-Stage) | `0-3` | `4-27` | 1 | W7 (Long Generation Output) | 198/512 | 1.96s | 101.1 | 13.5 | 74.1 | 175.6 | 11.08 | SUCCESS |
| T2 (Interleaved 4-Stage) | `0-3, 14-17` | `4-13, 18-27` | 3 | W1 (Short Chat) | 50/32 | 6.33s | 8.2 | 1.9 | 616.9 | 979.1 | 2.42 | SUCCESS |
| T2 (Interleaved 4-Stage) | `0-3, 14-17` | `4-13, 18-27` | 3 | W2 (Standard Chat) | 148/64 | 25.33s | 8.1 | 3.8 | 289.5 | 424.8 | 6.22 | SUCCESS |
| T2 (Interleaved 4-Stage) | `0-3, 14-17` | `4-13, 18-27` | 3 | W3 (Detailed Assistant) | 494/128 | 40.75s | 12.7 | 1.9 | 520.9 | 506.3 | 18.08 | SUCCESS |
| T2 (Interleaved 4-Stage) | `0-3, 14-17` | `4-13, 18-27` | 3 | W4 (Agentic Context) | 986/128 | 27.64s | 35.7 | 1.2 | 835.9 | 345.4 | 32.03 | SUCCESS |
| T2 (Interleaved 4-Stage) | `0-3, 14-17` | `4-13, 18-27` | 3 | W5 (RAG Document Retrieval) | 2000/0 | — | — | — | — | — | — | **FAILED_OOM** |
| T2 (Interleaved 4-Stage) | `0-3, 14-17` | `4-13, 18-27` | 3 | W6 (Heavy Codebase Stress) | 3000/0 | — | — | — | — | — | — | **FAILED_OOM** |
| T2 (Interleaved 4-Stage) | `0-3, 14-17` | `4-13, 18-27` | 3 | W7 (Long Generation Output) | 198/512 | 13.01s | 15.2 | 4.8 | 209.3 | 336.2 | 22.00 | SUCCESS |
| T3 (Interleaved 8-Stage) | `0-3, 8-10, 15-17, 22-24` | `4-7, 11-14, 18-21, 25-27` | 7 | W1 (Short Chat) | 50/32 | 6.08s | 8.2 | 1.4 | 728.5 | 1001.5 | 4.82 | SUCCESS |
| T3 (Interleaved 8-Stage) | `0-3, 8-10, 15-17, 22-24` | `4-7, 11-14, 18-21, 25-27` | 7 | W2 (Standard Chat) | 148/64 | 46.57s | 3.2 | 0.2 | 5439.0 | 3833.8 | 12.44 | SUCCESS |
| T3 (Interleaved 8-Stage) | `0-3, 8-10, 15-17, 22-24` | `4-7, 11-14, 18-21, 25-27` | 7 | W3 (Detailed Assistant) | 494/128 | 137.86s | 3.6 | 0.9 | 1139.8 | 1137.5 | 36.13 | SUCCESS |
| T3 (Interleaved 8-Stage) | `0-3, 8-10, 15-17, 22-24` | `4-7, 11-14, 18-21, 25-27` | 7 | W4 (Agentic Context) | 986/128 | 166.14s | 5.9 | 1.1 | 875.4 | 1231.1 | 64.00 | SUCCESS |
| T3 (Interleaved 8-Stage) | `0-3, 8-10, 15-17, 22-24` | `4-7, 11-14, 18-21, 25-27` | 7 | W5 (RAG Document Retrieval) | 1972/256 | 351.77s | 5.6 | 1.4 | 703.6 | 1010.1 | 142.51 | SUCCESS |
| T3 (Interleaved 8-Stage) | `0-3, 8-10, 15-17, 22-24` | `4-7, 11-14, 18-21, 25-27` | 7 | W6 (Heavy Codebase Stress) | 2956/256 | 622.21s | 4.8 | 1.3 | 779.2 | 1024.9 | 183.82 | SUCCESS |
| T3 (Interleaved 8-Stage) | `0-3, 8-10, 15-17, 22-24` | `4-7, 11-14, 18-21, 25-27` | 7 | W7 (Long Generation Output) | 198/512 | 34.39s | 5.8 | 1.7 | 598.4 | 851.2 | 43.88 | SUCCESS |
| T4 (Interleaved 14-Stage) | `0-1, 4-5, 8-9, 12-13, 16-17, 20-21, 24-25` | `2-3, 6-7, 10-11, 14-15, 18-19, 22-23, 26-27` | 13 | W3 (Detailed Assistant) | 494/128 | 82.59s | 6.0 | 1.8 | 552.3 | 826.3 | 63.20 | SUCCESS |

---

## 3. Hardware Capacity Limits & OOM Failure Analysis

During testing, **2 trial(s)** encountered hardware capacity limits (`FAILED_OOM`).

| Topology | Workload | Context / Gen Tokens | Device Bottleneck | Root Cause Notes |
| :--- | :--- | :---: | :--- | :--- |
| T2 (Interleaved 4-Stage) | W5 | 2000 in / 0 out | Master (6GB) / Worker (16GB) | Hardware VRAM capacity exceeded (Local CUDA OOM): Master RTX 3060 Laptop (6GB limit) ran out of memo... |
| T2 (Interleaved 4-Stage) | W6 | 3000 in / 0 out | Master (6GB) / Worker (16GB) | Hardware VRAM capacity exceeded (Local CUDA OOM): Master RTX 3060 Laptop (6GB limit) ran out of memo... |

---

## 4. Telemetry Graph Evidence

High-frequency (10Hz) dual-node memory telemetry graphs were recorded for each individual trial run in `experiment_profiles/`.

### Sample Telemetry Profiles:

- `[run_T1_W1_t1_20260909_085149.png](experiment_profiles/run_T1_W1_t1_20260909_085149.png)`
- `[run_T1_W1_t1_20260909_085312.png](experiment_profiles/run_T1_W1_t1_20260909_085312.png)`
- `[run_T1_W2_t1_20260909_085318.png](experiment_profiles/run_T1_W2_t1_20260909_085318.png)`
- `[run_T1_W3_t1_20260909_085326.png](experiment_profiles/run_T1_W3_t1_20260909_085326.png)`
- `[run_T1_W4_t1_20260909_085342.png](experiment_profiles/run_T1_W4_t1_20260909_085342.png)`

---

## 5. Architectural Recommendations

1. **Batch/Offline Suitability vs Interactive Unsuitability**: Decentralized topologies like openGP are well-suited for high-throughput asynchronous batch processing (e.g. document summarization, batch synthetic data generation) where latency tolerance is on the order of minutes. However, for interactive human chat, decentralized multi-hop splitting introduces inescapable network stalls.
2. **Minimizing Network Traversals**: If interactive inference must be deployed over volunteer networks, coarse-grained 2-stage pipelining (T1) is strictly required to constrain network hops to 1 per token. Any interleaving beyond 1 hop pushes ITL beyond tolerable human limits.
3. **Decentralized KV Cache Reuse**: Multi-turn sessions require persistent KV cache storage on worker nodes to prevent redundant transfer of multi-thousand-token prompt activations on every user turn.
