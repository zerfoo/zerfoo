# How We Beat Ollama by 18.8%: CUDA Graph Capture in Pure Go

*Performance deep-dive: how CUDA graph capture and fused kernels took Zerfoo from 186 tok/s to 234.30 tok/s on Gemma 3 1B.*

## The Benchmark

On the same hardware -- an NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified LPDDR5x) -- running the same model (Gemma 3 1B Q4_K_M GGUF), Zerfoo decodes at 234.30 tok/s. Ollama, which wraps llama.cpp's C++ inference engine, decodes at approximately 197 tok/s.

That is an 18.8% throughput advantage for a framework written entirely in Go with zero CGo calls.

The difference is not in the binding layer. Whether you call CUDA through CGo or through a dlsym'd function pointer, the GPU executes the same kernel at the same speed. The difference is in two things: CUDA graph capture and fused kernels.

## What CUDA Graphs Do

A normal inference step works like this: the CPU submits GPU operations one at a time to a CUDA stream. Each submission crosses the CPU-GPU boundary, the driver validates parameters, the scheduler finds an SM to run on, and the kernel launches. For a single transformer decode step with dozens of kernels (attention, RMSNorm, RoPE, SwiGLU, matrix multiplications, residual adds), the CPU spends a measurable amount of time just dispatching work.

CUDA graphs eliminate this overhead. The idea is simple:

1. **Capture**: Run the decode step once while CUDA records every operation into a graph data structure. No GPU work actually executes during capture -- CUDA just remembers what was submitted.
2. **Instantiate**: Compile the captured graph into an executable form. CUDA validates the entire sequence once and optimizes the launch schedule.
3. **Replay**: On every subsequent decode step, launch the entire graph with a single API call. The CPU submits one command instead of dozens. The GPU replays the recorded sequence with near-zero launch latency.

The result is that per-token decode overhead drops from "submit N kernels individually" to "submit one graph launch." For small models where individual kernel execution times are short (microseconds), the launch overhead is a significant fraction of total time. Eliminating it produces the 26% speedup we measured over our non-graph baseline (186 tok/s to 234.30 tok/s).

## 99.5% Instruction Coverage

CUDA graph capture has a fundamental constraint: every operation in the captured region must be a pure GPU operation. No CPU-side memory allocations, no device-to-host copies, no host-to-device transfers. If any operation touches the CPU during capture, the capture fails.

This is a hard problem for a real inference pipeline. Token embeddings require looking up token IDs (which are on the CPU). Attention masks are computed on the CPU. Position IDs come from a CPU counter. The KV cache sequence length is tracked by the CPU.

Zerfoo solves this by splitting the execution plan into three regions:

```
[Pre-capture: CPU-touching ops]  [Capture region: GPU-only ops]  [Post-capture: CPU-touching ops]
```

The `CUDAGraphExecutor` in `ztensor/graph/cuda_graph.go` scans the instruction list and finds the longest contiguous run of capturable instructions. Non-capturable operations are explicitly listed:

- **EmbeddingLookup**: reads token IDs via a device-to-host copy
- **Gather**: uses CPU index tensors
- **AutoAttentionMask / AutoPositionIds**: allocate CPU tensors
- **Slice**: reads indices from GPU via device-to-host copy
- **ConstantOfShape / Shape**: produce CPU-resident tensors

Everything else -- the transformer layers, attention, normalization, FFN, residual connections -- runs entirely on GPU and is captured into the graph. On Gemma 3 1B, the pre-capture region is a handful of embedding and position operations. The capture region covers all 18 transformer layers. The result: 99.5% of instructions execute inside the CUDA graph.

The key engineering work that made this possible was moving position-dependent state onto the GPU. GroupedQueryAttention was previously non-capturable because it read `cache.SeqLen()` on the CPU for RoPE positions and used CPU-computed offsets for KV cache appends. We added a GPU-resident sequence counter (via an `offset_memcpy` kernel) and a `rope_select` kernel that reads positions from GPU memory at replay time. This made GQA fully capturable, which was the difference between capturing a few percent of instructions and capturing 99.5%.

## Fused Kernels

CUDA graph capture eliminates launch overhead, but fused kernels reduce the number of launches in the first place. Each fused kernel replaces a sequence of smaller operations with a single GPU kernel that does the same work in one pass.

Zerfoo uses four primary fused operations in the decode path:

**FusedAddRMSNorm** -- In a transformer layer, the output of attention is added to the residual stream and then normalized. Without fusion, this is two kernels: an element-wise add and an RMSNorm (which itself would be 6 kernels if decomposed: Pow, ReduceMean, Add, Sqrt, Div, Mul). The fused kernel does the residual add and the full RMSNorm normalization in a single pass, reading from global memory once and writing once.

**FusedSwiGLU** -- The feed-forward network uses SwiGLU activation: `SiLU(gate) * up`. Without fusion, this requires a Concat, Split, sigmoid, two Muls. The fused kernel computes `x * sigmoid(x) * y` in one pass over the gate and up projections.

**FusedRoPE** -- Rotary positional embeddings apply sin/cos rotations to query and key vectors. The fused kernel computes the rotation in-place without materializing intermediate sin/cos tensors.

**FusedQKNormRoPE** -- For architectures like Gemma 3 that apply QK normalization before RoPE, this kernel combines the normalization and rotation into a single pass.

The impact is multiplicative with CUDA graph capture. Fewer kernels in the graph means less work for the graph executor, smaller graph instantiation time, and tighter replay scheduling.

## The Measurement

All benchmark numbers follow the methodology documented in `docs/benchmarking-methodology.md`:

| Property | Value |
|----------|-------|
| Hardware | NVIDIA DGX Spark (GB10 Grace Blackwell) |
| Memory | 128 GB unified LPDDR5x |
| GPU SM | sm_121 |
| Model | Gemma 3 1B Q4_K_M (GGUF) |
| Go | 1.25.0 |
| CUDA | 13.0 |
| Measurement | Decode-only throughput (tok/s) |
| Token count | 256 tokens minimum |
| Warmup | 32-token generation discarded |
| Repetitions | 3 runs, median reported |

The benchmark command:

```bash
go run ./cmd/bench_tps -model ~/models/gemma-3-1b-q4_k_m.gguf -tokens 256
```

Both Zerfoo and Ollama were benchmarked on the same machine with the same model file. Ollama was given the same warmup opportunity.

| Runtime | Decode tok/s | Notes |
|---------|-------------|-------|
| **Zerfoo** | **234.30** | CUDA graph capture, fused kernels, zero CGo |
| Ollama | ~197 | llama.cpp backend, same model |

The 26% improvement over our own non-graph baseline (186 tok/s) confirms that the speedup comes from CUDA graph capture specifically, not from other optimizations that happened concurrently.

## What We Learned

CUDA graph capture is powerful but unforgiving. Here are the practical lessons from making it work in a real inference pipeline:

**Any CPU touch kills the capture.** A single `cudaMemcpy` (host-to-device or device-to-host) on the capturing stream causes the entire capture to fail with error 901. This includes implicit copies -- calling `.Data()` on a GPU tensor triggers a device-to-host copy. We spent significant time tracing down operations that looked GPU-only but had hidden CPU interactions.

**Debug logging is essential.** Setting `ZERFOO_DEBUG_GPU=1` enables per-instruction capture logging. When capture fails, the log shows exactly which instruction caused the failure. Without this, you are left with an opaque CUDA error and a list of hundreds of instructions.

**Frozen weights must be pre-uploaded.** Model weights that are initially CPU-resident (loaded from the GGUF file) must be transferred to GPU before capture begins. If a weight is lazily uploaded during capture, the host-to-device copy breaks the capture. `PreUploadFrozenWeights()` handles this by walking all instruction inputs and ensuring they are GPU-resident.

**KV cache state needs snapshotting.** If capture fails partway through, the KV cache has already been mutated by GroupedQueryAttention layers that ran before the failure. Without snapshotting and restoring the cache state, the fallback (running instructions normally) would double-update the cache. The `snapshotCache` callback handles this rollback.

**Dynamic shapes break graphs.** A captured CUDA graph records fixed tensor shapes. If the input shape changes between replay calls, the graph is invalid. For autoregressive decode, this is fine -- every decode step processes exactly one token, so shapes are constant. For prefill (variable-length prompts), CUDA graph capture is not applicable.

## Conclusion

The 18.8% speedup over Ollama is not a single trick. It is the combination of CUDA graph capture (eliminating per-kernel launch overhead), fused kernels (reducing kernel count), and engineering work to make the decode path GPU-only (moving position state to GPU memory).

All of this runs in pure Go with zero CGo. The performance comes from the GPU kernels and the graph capture strategy, not from the language the host code is written in.

The benchmark is reproducible. The methodology is documented. The code is open. See `docs/benchmarking-methodology.md` for the full measurement procedure.
