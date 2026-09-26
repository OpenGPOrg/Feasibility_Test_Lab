# Multi-Hop Decentralized Inference Feasibility Report

**Evaluation of 1, 3, 5, 7, 9 Network Hops across Small, Medium, and Large Workloads**

- **Hardware Setup**: Master Laptop (RTX 3060 6GB VRAM) + Worker Server `csetuf09` (RTX 4070 Ti Super 16GB VRAM)
- **Model**: `Qwen/Qwen2.5-7B-Instruct` (28 Layers, FP16)
- **Total Completed Trial Runs**: 90
- **Report Timestamp**: 2026-09-09 20:49:40

---

## 1. Summary Performance Table (Mean ± Standard Deviation)

| Hops | Workload | Master Layers | Worker Layers | Tokens (In/Out) | TTFT (s) | Decode TPS | Avg ITL (ms) | Master KV (MB) | Worker KV (MB) | Total KV (MB) | Net I/O (MB) |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Large Context | `0-3` | `4-27` | 977 / 128 | 2.84 ± 0.47s | 19.25 ± 4.07 | 54.4 ± 11.6 | 8.62 | 51.75 | 60.38 | 15.90 |
| **1** | Medium Chat | `0-3` | `4-27` | 245 / 64 | 0.85 ± 0.56s | 20.83 ± 2.82 | 48.9 ± 6.9 | 2.41 | 14.44 | 16.84 | 4.49 |
| **1** | Small Chat | `0-3` | `4-27` | 60 / 32 | 0.39 ± 0.31s | 20.67 ± 4.92 | 52.8 ± 19.1 | 0.71 | 4.27 | 4.98 | 1.35 |
| **3** | Large Context | `0-1, 14-15` | `2-13, 16-27` | 977 / 128 | 6.47 ± 1.13s | 16.62 ± 3.44 | 63.6 ± 16.5 | 8.62 | 51.75 | 60.38 | 31.77 |
| **3** | Medium Chat | `0-1, 14-15` | `2-13, 16-27` | 245 / 64 | 1.18 ± 0.16s | 14.85 ± 3.27 | 71.1 ± 17.3 | 2.41 | 14.44 | 16.84 | 8.97 |
| **3** | Small Chat | `0-1, 14-15` | `2-13, 16-27` | 60 / 32 | 0.57 ± 0.42s | 11.58 ± 4.35 | 97.5 ± 30.9 | 0.71 | 4.27 | 4.98 | 2.70 |
| **5** | Large Context | `0-1, 10-11, 20-21` | `2-9, 12-19, 22-27` | 977 / 128 | 7.71 ± 2.19s | 10.55 ± 0.91 | 95.5 ± 8.9 | 12.94 | 47.44 | 60.38 | 47.63 |
| **5** | Medium Chat | `0-1, 10-11, 20-21` | `2-9, 12-19, 22-27` | 245 / 64 | 1.60 ± 0.31s | 10.32 ± 2.43 | 102.4 ± 24.1 | 3.61 | 13.23 | 16.84 | 13.44 |
| **5** | Small Chat | `0-1, 10-11, 20-21` | `2-9, 12-19, 22-27` | 60 / 32 | 0.79 ± 0.59s | 12.83 ± 2.60 | 83.2 ± 25.9 | 1.07 | 3.91 | 4.98 | 4.05 |
| **7** | Large Context | `0, 7, 14, 21` | `1-6, 8-13, 15-20, 22-27` | 977 / 128 | 9.19 ± 0.97s | 10.51 ± 1.19 | 96.4 ± 10.4 | 8.62 | 51.75 | 60.38 | 63.50 |
| **7** | Medium Chat | `0, 7, 14, 21` | `1-6, 8-13, 15-20, 22-27` | 245 / 64 | 1.98 ± 0.22s | 10.92 ± 1.83 | 94.3 ± 16.0 | 2.41 | 14.44 | 16.84 | 17.92 |
| **7** | Small Chat | `0, 7, 14, 21` | `1-6, 8-13, 15-20, 22-27` | 60 / 32 | 0.81 ± 0.28s | 9.66 ± 3.69 | 121.4 ± 46.9 | 0.71 | 4.27 | 4.98 | 5.39 |
| **9** | Large Context | `0, 6, 12, 18, 24` | `1-5, 7-11, 13-17, 19-23, 25-27` | 977 / 128 | 11.79 ± 0.70s | 9.10 ± 0.62 | 110.4 ± 7.5 | 10.78 | 49.59 | 60.38 | 79.37 |
| **9** | Medium Chat | `0, 6, 12, 18, 24` | `1-5, 7-11, 13-17, 19-23, 25-27` | 245 / 64 | 3.21 ± 1.38s | 7.64 ± 2.24 | 150.0 ± 68.6 | 3.01 | 13.84 | 16.84 | 22.40 |
| **9** | Small Chat | `0, 6, 12, 18, 24` | `1-5, 7-11, 13-17, 19-23, 25-27` | 60 / 32 | 1.05 ± 0.39s | 9.00 ± 1.82 | 115.7 ± 22.7 | 0.89 | 4.09 | 4.98 | 6.74 |

---

## 2. Feasibility Curves (Replication of T.png & D.png)

### Time to First Token (TTFT) Vs Number of Hops

![Time to First Token Vs Number of Hops](time_to_first_token_vs_hops.png)


### Decode Token Per Second Vs Number of Hops

![Decode Token Per Second Vs Number of Hops](decode_tps_vs_hops.png)

