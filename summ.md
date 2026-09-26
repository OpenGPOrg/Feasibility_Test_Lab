# Master Report: KV Cache Affinity — A Deep Technical Exploration

## Executive Summary

**KV Cache Affinity** refers to the degree to which a new inference request can reuse Key-Value (KV) cache tensors already computed and stored from previous requests, rather than recomputing them from scratch. It is the foundational concept behind **prefix caching**, **cache-aware routing**, and **radix-tree-based KV reuse** in modern LLM serving systems (vLLM, SGLang, TensorRT-LLM, TGI). High KV cache affinity means a request "sticks" to cached state — dramatically reducing prefill latency, GPU compute, and time-to-first-token (TTFT). Low affinity means expensive recomputation of attention states for tokens that were already processed before.

This report builds the concept from first principles: what the KV cache is, why it dominates memory and compute economics, what "affinity" formally means, how systems exploit it, worked numerical examples, and how affinity interacts with quantization, pruning, paged memory, and eviction policies described in the research literature.

---

## Part I: Foundations — What Is the KV Cache?

### 1.1 The Attention Recalculation Problem

In a decoder-only transformer, generating token *t* requires attending over all previous tokens 1…t−1. Naively, at every generation step the model would recompute the Key (K) and Value (V) projections for **every prior token**. As the context sources note:

> "Without caching, the attention at step t recomputes keys and values for all prior tokens so work accumulates toward O(n²). With a KV cache, each token's key and value are stored and read later, keeping per-step work closer to O(n) over the whole generation."

The KV cache stores, for every layer *l* and every past token *i*:

- **Kᵢˡ = xᵢ W_Kˡ** (key vector)
- **Vᵢˡ = xᵢ W_Vˡ** (value vector)

During decode, the model computes K/V **only for the newest token** and appends them:

> "The model only computes KV tensors for the newest token and appends them to the cache instead of recalculating attention for the whole sequence every time."

### 1.2 The Two Phases of Inference

**Prefill:** The full prompt is processed in one parallel pass. All K/V vectors for prompt tokens are computed and written into the cache. This phase is compute-bound and highly parallel.

**Decode:** Strictly sequential — "generate one token → append it to the context → attend to all previous tokens → compute attention for the next token → repeat." This phase is memory-bandwidth-bound, because each step must read the entire KV cache.

### 1.3 The Memory Explosion

The KV cache grows **linearly with sequence length** and proportionally to layers × heads × head dimension:

> "Taking OPT-175B as an example, with a total of 96 layers and a hidden size of 12288, its weights occupy 325GB memory, while the KV cache is 3.54× larger, reaching 1152GB under its maximum sequence length."

Per-token KV memory formula:

```
bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_element
```

For Llama-3-70B (80 layers, 8 KV heads via GQA, head_dim 128, FP16):
2 × 80 × 8 × 128 × 2 bytes = **327,680 bytes ≈ 320 KB per token**. A 128K-token context = **~40 GB of KV cache for a single sequence**.

This is precisely why reuse — affinity — is so economically valuable: recomputing or duplicating this state is enormously expensive.

---

## Part II: Defining KV Cache Affinity

### 2.1 Formal Definition

**KV Cache Affinity** between a new request *R* and a stored cache state *C* is the length of the **shared token prefix** whose KV tensors can be reused directly:

```
Affinity(R, C) = |LCP(tokens(R), tokens(C))|
```

where LCP = Longest Common Prefix. Because transformer attention is **causal** (each token's K/V depends only on tokens before it), the KV tensors for a shared prefix are *bit-identical* regardless of what follows. This causality property is the mathematical reason affinity exists at all.

**Affinity ratio** = shared prefix length / total prompt length. An affinity ratio of 0.9 means only 10% of the prompt needs prefill compute.

### 2.2 Why Affinity Arises Naturally in Real Workloads

Real LLM traffic is full of shared prefixes:

1. **System prompts** — Every request to a chatbot may begin with the same 2,000-token system prompt ("You are a helpful assistant…").
2. **Few-shot templates** — RAG pipelines prepend the same instructions and document scaffolding.
3. **Multi-turn conversations** — Turn *N*'s prompt contains turns 1…N−1 verbatim.
4. **Agentic loops / tool use** — Agents re-send the entire history plus one new tool result.
5. **Branching generation** — Beam search, best-of-N sampling, and tree-of-thought share long prefixes.
6. **Long-document QA** — Many questions about the same 100K-token document.

### 2.3 Affinity vs. Related Concepts

| Concept | What it means |
|---|---|
| **KV Cache Affinity** | Reusability of cached KV state by a new request (prefix overlap) |
| **Prefix Caching** | The mechanism that *stores* KV blocks to enable affinity |
| **Cache-aware routing** | Scheduling requests to the GPU replica holding their matching cache |
| **PagedAttention** | Memory layout (paged KV blocks) that makes sharing/eviction tractable |
| **KV quantization/pruning** | Shrinking cache footprint; orthogonal to affinity but interacts with it |

---

## Part III: Mechanisms That Exploit Affinity

### 3.1 Paged Memory as the Enabler

The context describes **PagedAttention** (vLLM):

> "The KV cache is split into small memory 'pages' that can be dynamically allocated across GPU memory… compact KV block tables that can be accessed directly from paged KV cache memory without copying or rearranging tensors."

Paging is what makes affinity *practical*: a shared prefix's pages can be **reference-counted and shared** between sequences (copy-on-write), exactly like OS virtual memory. Without paging, contiguous per-request buffers would force duplication of shared prefixes.

### 3.2 Prefix Caching (Automatic Prefix Reuse)

vLLM's Automatic Prefix Caching (APC) hashes each KV block by (block tokens + hash of previous block). When a new request arrives, matching block hashes are looked up and mapped directly — **zero recompute** for the matched prefix.

### 3.3 Radix-Tree Caching (SGLang's RadixAttention)

SGLang maintains a **radix tree (trie)** of token sequences → KV cache pointers. Incoming requests are matched down the tree to find the longest cached prefix; new tokens extend the tree. Eviction is LRU over tree nodes. This maximizes affinity across *thousands* of concurrent requests with overlapping prompts.

### 3.4 Cache-Aware Routing (Affinity Scheduling)

In multi-replica deployments, affinity extends across machines: the router sends a request to the **worker that already holds its prefix cache**. Metrics like "prefix-match length per worker" drive load balancing — a deliberate trade-off between load evenness and cache affinity. This is sometimes called **locality-aware scheduling**.

---

## Part IV: Worked Example

### 4.1 Scenario: Customer-Support Bot

System prompt: 1,500 tokens (identical for all users).
User A's conversation so far: 800 tokens.
New user message: 50 tokens.

**Without affinity (no prefix caching):**
- Prefill processes 1,500 + 800 + 50 = **2,350 tokens** of compute.
- KV cache written: 2,350 tokens × 320 KB (70B model) ≈ **752 MB**.

**With KV cache affinity:**
- System prompt blocks (1,500 tokens) already cached from prior requests → affinity hit.
- Conversation history (800 tokens) cached from the previous turn → affinity hit.
- **Only 50 new tokens require prefill compute** — a **47× reduction in prefill work**.
- TTFT drops from (say) 900 ms to ~40 ms.

### 4.2 Multi-Turn Compounding

Turn 10 of a conversation has a 20,000-token history. Without affinity, every turn recomputes all 20K tokens (O(n²) cumulative cost across the session). With affinity, each turn prefills only the new ~200 tokens. Over a 10-turn session this is the difference between ~1.1M and ~22K prefill tokens — **~50× compute savings**.

### 4.3 Branching Example (Tree-of-Thought)

A reasoning agent expands 4 candidate continuations from a 5,000-token shared reasoning trace. With paged KV sharing, the 5,000-token prefix is stored **once** and reference-counted by all 4 branches (copy-on-write when branches diverge). Memory: 5,000 + 4×(branch length) tokens instead of 4×(5,000 + branch) — a ~4× memory saving, and the prefix is prefilled once, not four times.

---

## Part V: Affinity Under Memory Pressure — Interactions with Compression

The research context highlights a rich ecosystem of KV cache management strategies, all of which interact with affinity:

### 5.1 The Three Paradigms (from the comparative characterization paper)

1. **Paged memory + continuous batching, no sparsification** (vLLM, ORCA, vTensor) — preserves full fidelity; affinity is exact.
2. **Permanent eviction** (H2O, Scissorhands, StreamingLLM, SnapKV) — attention-score-based token dropping. **This destroys affinity for evicted tokens**: if a shared prefix's "unimportant" tokens were evicted, a new request matching that prefix cannot fully reuse it.
3. **Hierarchical storage / recoverable eviction** (InfiniGen, InfLLM, Quest, RetrievalAttention) — evicted KV goes to CPU/disk and can be re-fetched. Affinity becomes *recoverable* at the cost of transfer latency.

### 5.2 Quantization and Affinity

Quantization (FP16 → FP8/INT8/INT4) gives "2–4× memory reduction at the cost of potential accuracy degradation." Key interactions:

- **More affinity capacity:** halving per-token KV size doubles the number of cached prefixes that fit in GPU memory → higher hit rates.
- **Quantization is memory-bound and cheap:** "KV cache quantization is a strictly element-wise operation… overwhelmingly memory-bound… the naive kernel already saturates memory bandwidth." So quantizing cached blocks adds negligible overhead to an affinity-based system.
- **Dequantization before attention:** "cached KV vectors are dequantized before computation" — affinity reuse is unaffected numerically beyond the fixed quantization error.
- **Quality-adaptive schemes (QAQ, GEAR):** outlier-aware and residual-corrected quantization preserve accuracy, making quantized caches safe to share across requests.

### 5.3 The Risk of Affinity + Pruning

> "Any misjudgment of importance leading to the loss of crucial cache can significantly degrade the performance of the model."

If a cached shared prefix was pruned using one request's attention pattern, reusing it for a *different* request (whose important tokens differ) can silently degrade quality. This is a subtle correctness hazard of combining affinity with sparsification.

---

## Part VI: Systems Design Considerations

1. **Eviction policy:** LRU over prefix blocks; shared (high-affinity) blocks have high reference counts and naturally survive — popularity and affinity align.
2. **Chunked prefill:** New (non-affined) suffix tokens are prefilled in chunks interleaved with decode batches; techniques like CompactAttention accelerate long-context chunked prefill "up to 2.72× attention speedup at 128K context."
3. **Distributed/pipeline deployments:** Each pipeline stage owns only its layers' KV cache ("each stage owns only its own layers' cache"), so affinity decisions must be consistent across all stages — the same worker chain must serve the request.
4. **Cross-layer KV sharing (e.g., Gemma):** shared K/V tensors reduce footprint and change what "a cached prefix" means across layers.
5. **Cache truncation trade-off:** "Truncating the cache or windowing the context reduces memory but sacrifices long-range coherence and forces recomputation for the discarded portion" — i.e., it destroys future affinity.

---

## Part VII: Measuring Affinity

Key production metrics:

- **Prefix cache hit rate** = cached prefix tokens / total prompt tokens
- **TTFT reduction** attributable to cache hits
- **GPU memory saved** via shared blocks (deduplication ratio)
- **Effective batch size** gains (as the TensorRT-LLM analysis shows, memory savings translate to throughput mainly by enabling larger batches)
- **Recompute ratio** = tokens prefilled / tokens that would be prefilled without caching

---

## Part VIII: Summary

| Aspect | Key Insight |
|---|---|
| **Definition** | Affinity = length of reusable shared-prefix KV state between requests |
| **Why it exists** | Causal attention makes prefix KV tensors context-independent |
| **Enablers** | PagedAttention (block sharing), radix trees, prefix hashing, cache-aware routing |
| **Benefits** | 10–100× prefill compute reduction, lower TTFT, higher throughput, memory deduplication |
| **Synergies** | Quantization (2–4× more cache capacity → higher hit rates), hierarchical storage |
| **Risks** | Pruning/eviction can silently break affinity or degrade cross-request accuracy |
| **Where it matters most** | Multi-turn chat, agents, RAG, branching search, shared system prompts |

**In one sentence:** KV Cache Affinity is the exploitation of causal attention's prefix-determinism — because the KV tensors of a shared prefix are identical no matter what follows, serving systems that detect, store, share, and route toward those cached prefixes transform LLM inference from an O(n²) recompute-everything workload into an incremental, amortized one, which is arguably the single most important economic optimization in production LLM serving today.


Sources synthesized via Map-Reduce:
- turingpost.com_How_LLM_Inference_Works_Prefil.md
- arxiv.org_Comparative_Characterization_o.md
- inference.net_KV_Cache_Explained_with_Exampl.md
- arxiv.org_QAQ_Quality_Adaptive_Quantizat.md
- blog.squeezebits.com_vLLM_vs_TensorRT-LLM_8_KV_Cach.md
- arxiv.org_Pre-Compiled_Pipeline_Shards_f.md
- arxiv.org_GPU-Accelerated_INT8_Quantizat.md
