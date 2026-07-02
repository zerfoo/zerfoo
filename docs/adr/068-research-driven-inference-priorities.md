# ADR 068: Research-Driven Inference Optimization Priorities

## Status

Accepted

## Date

2026-03-27

## Context

A survey of 30+ ML research papers published between March 2024 and March 2027
identified techniques that could significantly improve Zerfoo's inference speed,
memory efficiency, and model coverage. The goal is to select the highest-impact
techniques that are practical to implement in a Go-based inference framework
with GPU acceleration via purego (no CGo).

Selection criteria:
- Delivers measurable improvement to tok/s, memory usage, or model coverage.
- Implementable without CGo (purego CUDA kernels, Go standard library).
- Composable with existing infrastructure (Engine[T], GGUF, KV cache).
- Each technique independently valuable (no all-or-nothing dependencies).

## Decision

Adopt five research directions as new epics, ordered by impact:

**E35: QuaRot + KVQuant (Uniform 4-Bit Quantization)**
- QuaRot (ETH Zurich, NeurIPS 2024): Apply Hadamard rotation to weight matrices
  at GGUF load time, eliminating outlier features. This enables uniform 4-bit
  quantization of weights, activations, AND KV cache with zero runtime overhead.
- KVQuant (UC Berkeley, NeurIPS 2024): 3-bit KV cache quantization with
  per-channel keys and non-uniform codebooks. 4.8x KV cache memory reduction.
- Combined: 3.3x prefill speedup, 3.9x memory savings on 70B-class models.

**E36: EAGLE-3 Self-Speculative Decoding**
- EAGLE-3 (March 2025): Lightweight prediction head attached to the target model's
  penultimate layer. Reuses existing features to predict future tokens -- no
  separate draft model needed. 3-6.5x speedup over vanilla autoregressive.
- Replaces Zerfoo's current two-model speculative decoding with a simpler,
  faster single-model approach.

**E37: NSA (Native Sparse Attention)**
- DeepSeek, ACL 2025 Best Paper: Hardware-aligned sparse attention combining
  three paths: coarse-grained token compression, fine-grained token selection,
  and sliding window. Natively trainable.
- Subsumes MSA's sparse attention (E34.4) with better GPU utilization. The
  three-path design maps to GPU warp-level parallelism.

**E38: CPU/GPU Hybrid MoE (KTransformers-style)**
- KTransformers (Tsinghua MADSys, SOSP 2025): Place shared MoE experts on GPU,
  offload routed experts to CPU with AMX/SIMD kernels. Async scheduling hides
  transfer latency. Runs DeepSeek-V3 671B on single GPU + host CPU.
- Go goroutines are well-suited for the async overlap pattern. Leverages
  existing xblas SIMD kernels.

**E39: BitNet b1.58 Ternary Inference**
- BitNet b1.58 (Microsoft, February 2024): Ternary weights {-1, 0, 1} replace
  matrix multiplication with integer additions/subtractions. 1.37-5.07x CPU
  speedup, 55-70% energy reduction.
- Trivially implementable in Go (no cuBLAS needed). Makes CPU inference
  competitive for ternary models.

## Consequences

**Positive:**
- E35 (QuaRot+KVQuant) is the single biggest memory reduction available,
  enabling larger models on the DGX Spark and longer effective contexts.
- E36 (EAGLE-3) is the fastest path to decisively beating Ollama at 7B+.
- E37 (NSA) provides the production-grade sparse attention kernel that E34's
  MSA layers need underneath.
- E38 (hybrid MoE) would be a headline feature: 671B DeepSeek-V3 on one GPU.
- E39 (BitNet) future-proofs for ternary models, where Go CPU inference
  becomes competitive with C++.

**Negative:**
- Five concurrent epics (E34-E39) create breadth risk for a small team.
  Mitigated by each epic being independently valuable and incrementally
  shippable.
- QuaRot requires Hadamard matrices fused at load time -- one-time engineering
  cost per quantization type.
- NSA's three-path kernel is complex CUDA work (~500 lines of kernel code).
- Hybrid MoE async scheduling is architecturally invasive to the inference loop.
- BitNet depends on availability of competitive ternary models in GGUF format.
