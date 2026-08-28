# Distributed Inference — Setup Guide

## Architecture

```
┌──────────────────────────┐         TCP/9900         ┌───────────────────────────┐
│   YOUR LAPTOP (Controller)│◄──────────────────────►│  HP ELITEDESK G800 (Worker) │
│   RTX 3060 + 15GB RAM     │    model components     │  i5 7th Gen + 8GB RAM       │
│                            │    + activations        │                             │
│  distributed_inference.py  │                         │  worker_node.py             │
└──────────────────────────┘                          └───────────────────────────┘
```

## Step 1: Setup HP EliteDesk (Worker Node)

On the HP EliteDesk, install dependencies:

```bash
# Create a venv (or use system python)
python3 -m venv .venv
source .venv/bin/activate

# Install CPU-only PyTorch + requirements
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers psutil
```

Copy `worker_node.py` to the HP EliteDesk and run:

```bash
python3 worker_node.py
# or with custom port:
python3 worker_node.py --port 9901
```

You should see:
```
═══════════════════════════════════════════════════════
  Worker Node: HP-EliteDesk-G800
  Port:        9900
  CPU:         Intel i5-7500
  RAM:         8.0 GB
  GPU:         None
═══════════════════════════════════════════════════════
  ✓ Discovery broadcast on UDP port 9899
  ✓ Listening on TCP port 9900
  Waiting for controller connection...
```

## Step 2: Run Controller (Your Laptop)

```bash
cd /media/vithurshan/vithu/llm/openGP/LoadTime
python3 distributed_inference.py
```

## Step 3: Interactive Flow

### 3.1 Add the remote node

```
Main Menu → 1. Manage Remote Nodes → 1. Scan network (auto-discovery)
```

Or manually:
```
Main Menu → 1. Manage Remote Nodes → 2. Add manually → 192.168.x.x:9900
```

### 3.2 Select & download a model

```
Main Menu → 2. Select & Download Model → pick any model
```

### 3.3 Configure distribution

```
Main Menu → 3. Configure Distribution
```

You'll see a table of what's where. Then choose what to offload:

| Option | What it does |
|--------|-------------|
| 1. Full layer(s) | Move entire transformer layer(s) to remote |
| 2. Attention | Move only attention sub-layer(s) to remote |
| 3. FFN/MLP | Move only FFN/MLP sub-layer(s) to remote |
| 4. Embedding | Move token embedding to remote |
| 5. LM Head | Move output head + final norm to remote |
| 6. Expert(s) | Move specific MoE experts to remote (MoE only) |

Example: offload last 5 layers of a 30-layer model:
```
Select: 1 (full layers)
Layer range: 25-29
Target node: 1 (HP-EliteDesk)
→ Apply & Load (option 8)
```

### 3.4 Run inference

```
Main Menu → 4. Run Inference → Enter prompt
```

Shows streaming response + stats (tok/s, TTFT, KV cache, network transfer).

## Offloadable Components

| Component | Description | Network data per forward |
|-----------|-------------|------------------------|
| **Full layer** | Entire transformer layer (attn + FFN/MoE) | ~4KB/token (hidden_states) |
| **Attention** | Q/K/V projections + attention + O projection | ~4KB/token |
| **FFN/MLP** | Feed-forward network (gate+up+down projections) | ~4KB/token |
| **Expert** | Single MoE expert (gate+up+down) | ~4KB/routed token |
| **Embedding** | Token → hidden_states lookup | ~4KB/token |
| **LM Head** | hidden_states → logits projection | ~4KB/token + ~200KB logits |

## Notes

- Both machines must have the **same version** of `transformers` installed
- The worker runs on **CPU** (no GPU needed)
- Network latency adds ~0.5-2ms per remote layer per token
- KV cache for remote layers is stored **on the worker** (uses its RAM)
- Ctrl+C to stop either script gracefully
