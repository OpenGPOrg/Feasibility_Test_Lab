# openGP Developer Documentation

This document provides a technical overview of the openGP distributed inference engine, detailing the architecture and internal mechanics of both the Master (`distributed_inference.py`) and the Worker (`worker_node.py`) scripts.

---

## 1. High-Level Architecture

openGP is a peer-to-peer, layer-wise distributed inference engine for PyTorch-based Large Language Models (LLMs). It splits HuggingFace model architectures at the PyTorch `nn.Module` level and distributes them across a local area network (LAN) using raw TCP sockets. 

* **Master Node (`distributed_inference.py`):** Acts as the coordinator. It parses the model, extracts layers, serializes the weights using `safetensors` or `torch.save`, streams them to workers, manages the KV cache allocation, and orchestrates the autoregressive generation loop.
* **Worker Node (`worker_node.py`):** Acts as a stateless compute engine. It receives serialized PyTorch modules from the Master, caches them to disk, loads them into local RAM/VRAM, and executes forward passes when commanded.

---

## 2. Master Script (`distributed_inference.py`)

The master script is structurally divided into network management, model distribution, and the interactive CLI.

### Core Classes

#### `NodeConnection`
* **Purpose:** Wraps the raw TCP socket connecting to a remote worker.
* **Responsibilities:**
  * Handles custom JSON-header + binary payload framing (`struct.pack("<I")`).
  * Implements `send_cmd()` to transmit commands (like `load`, `forward`, `status`).
  * Tracks network statistics (`bytes_sent`, `bytes_received`).

#### `DistributedModel`
* **Purpose:** The core engine that loads the HuggingFace model and orchestrates the distributed generation loop.
* **Key Methods:**
  * `_extract_module()`: Dynamically extracts specific sub-modules (like `layer.mlp` or `layer.self_attn`) from the main model using `copy.deepcopy` and strips them to CPU before network transmission.
  * `apply_distribution()`: Evaluates the `self.assignments` dictionary. If a layer is assigned to a remote node, it wraps it in a "Hybrid Remote Hook" (e.g., `_hybrid_attn_remote`). If assigned locally, it greedily packs it into local GPU VRAM.
  * `_forward_pass()`: The custom wrapper around the model's sequential pipeline. Iterates through embeddings, transformer blocks, and the LM head.
  * `generate()`: The main autoregressive loop. Calculates Time-to-First-Token (TTFT), tracking token latency, memory consumption, and network I/O.
  * `_post_telemetry_to_web()`: Contains a silent, non-blocking HTTP POST for external telemetry hooks (Non-intrusive design).

#### `CLI`
* **Purpose:** The interactive terminal user interface.
* **Responsibilities:** 
  * Displays the topology menu, node discovery, and component assignment screens using `tabulate`.
  * Manages `AssignmentCaches` (saving/restoring `.json` topology maps).

### Master Data Flow
1. **Load Model:** `transformers.AutoModelForCausalLM.from_pretrained(..., device_map="cpu")`.
2. **Assign:** User maps components (`layer_0`, `layer_1`) to specific `NodeConnection`s.
3. **Stream Weights:** `generate()` checks if remote nodes have the weights. If not, it serializes and sends a `load` command.
4. **Generate:** The master loops `max_new_tokens`. For each token, the hidden states tensor is serialized, sent to the worker via a `forward_stage` command, and the worker returns the processed hidden states.

---

## 3. Worker Script (`worker_node.py`)

The worker script is a lightweight, pure PyTorch listener designed to run without root privileges.

### Core Classes

#### `Worker`
* **Purpose:** The main daemon managing the TCP server and incoming connections.
* **Key Components:**
  * **Network Modes:** Can bind locally (`_serve()`) and wait for the Master, OR can connect outbound to the Master (`_connect_to_master()`) to bypass restrictive UFW/NAT firewalls.
  * **UDP Discovery:** Broadcasts its presence on port 9901 so the Master's UI can auto-detect it.
  * **Memory Profiling:** Spawns a background thread that samples `psutil` and `torch.cuda.memory_allocated()` every 0.1s to generate matplotlib graphs.

#### Command Dispatcher (`_dispatch`)
When the worker receives a TCP message, `_dispatch` routes it based on the `"cmd"` string:

* **`load` (`_cmd_load`)**: 
  * Extracts the binary PyTorch module from the payload.
  * Saves it to local disk cache (`~/.cache/openGP_components/`).
  * Loads it into the `self.components` dictionary and pushes it to the GPU if available.
* **`forward` (`_cmd_forward_layer`)**: 
  * Deserializes the incoming `hidden_states` tensor.
  * Matches dtype/device, executes the local PyTorch module (`out = module(hs)`).
  * Returns the output tensor back across the wire.
* **`forward_stage`**: 
  * An optimized pipeline command. If the worker possesses layers 0, 1, and 2, the Master sends one `forward_stage` command. The worker executes all three layers sequentially in local memory and only returns the final output. This reduces network I/O by 66%.

---

## 4. Network Protocol Details

Because standard HTTP/REST is too slow for real-time tensor streaming, openGP uses a custom binary protocol over TCP:

**Message Structure:**
1. `header_len` (4 bytes, Little Endian Unsigned Int)
2. `header_json` (Variable length UTF-8 encoded JSON metadata)
3. `payload` (Variable length raw binary PyTorch tensors / pickled models)

This structure ensures zero-copy extraction of large multi-gigabyte weight chunks during the model distribution phase.

---

## 5. Development Guidelines

* **Device Mapping:** Always use `next(module.parameters()).device` inside custom forward hooks rather than relying on a global device state. The greedy offloading algorithm may split a layer's components across CPU and GPU on the same machine.
* **Adding New Architectures:** To support a new HuggingFace architecture, update the `_forward_pass` logic in `distributed_inference.py` to map the specific layer attribute names (e.g., `model.layers` vs `model.h`) and ensure the KV cache tuple matches the architecture's format.
