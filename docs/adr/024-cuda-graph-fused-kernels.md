# ADR 024: CUDA Graph Capture and Fused Kernels for Inference Acceleration

## Status
Accepted

## Date
2026-03-06

## Context
Phase 33 achieved 10.32 tok/s peak (7.78 median) for Gemma 3 2B Q4 on DGX Spark
GB10. llama.cpp achieves 24-38 tok/s for comparable models on the same hardware.
The gap is primarily caused by:

1. Per-op CGo kernel launch overhead: each of 25+ GPU operations per forward pass
   incurs ~100ns CGo call overhead plus CUDA kernel launch latency. For small
   tensors in autoregressive decode (batch=1), launch overhead dominates compute.

2. No kernel fusion: operations like Scale+Softmax, Gate*SiLU(Up), and
   RMSNorm+Weight*Add are separate kernel launches with intermediate GPU memory
   round-trips. llama.cpp fuses these into single kernels.

3. No CUDA graph capture: llama.cpp records the decode forward pass as a CUDA
   graph and replays it per token, eliminating all per-op launch overhead.

The DGX Spark GB10 has 273 GB/s LPDDR5x bandwidth. For a 1.5GB Q4 model, the
theoretical max is ~182 tok/s. Zerfoo at 7.78 median is 4.3% of theoretical;
llama.cpp at 24-38 tok/s is 13-21%.

## Decision
Phase 34 will implement three optimizations in priority order:

1. **Pre-allocated buffer pool**: At ExecutionPlan compile time, compute the shape
   of every intermediate tensor and pre-allocate a fixed GPU buffer layout. This
   eliminates per-op pool.Alloc/Free calls and enables CUDA graph capture (which
   requires fixed memory addresses).

2. **CUDA graph capture and replay**: Add CGo wrappers for cudaStreamBeginCapture,
   cudaStreamEndCapture, cudaGraphInstantiate, cudaGraphLaunch, and
   cudaGraphExecUpdate. Record the decode forward pass on the first token, then
   replay the captured graph for subsequent tokens. Re-capture only when the graph
   topology changes (e.g., context length crosses a threshold).

3. **Fused CUDA kernels**: Implement three high-value fused kernels:
   - Fused SwiGLU: gate * silu(up) in a single kernel (saves 2 kernel launches
     and 1 intermediate buffer per FFN block)
   - Fused Scale+Softmax: scale attention scores by 1/sqrt(d) and compute softmax
     in one pass (saves 1 launch + 1 intermediate per attention head)
   - Fused dequant+GEMV for Q4 single-token decode: read Q4 blocks, dequantize
     in registers, multiply by activation vector, accumulate in F32 (saves the
     separate dequantize step that is the largest single kernel)

## Consequences
Positive:
- CUDA graph replay eliminates all per-token CGo overhead for the decode loop
- Pre-allocated buffers reduce memory management overhead to zero during decode
- Fused kernels reduce memory bandwidth by eliminating intermediate writes
- Expected 2-3x throughput improvement (target: 20+ tok/s median)

Negative:
- CUDA graph capture requires fixed memory addresses (no dynamic allocation)
- Graph re-capture needed when sequence length changes significantly
- Fused kernels are specific to Gemma 3 architecture patterns (SwiGLU, GQA)
- Increased complexity in the CUDA kernel codebase
- CGo wrappers for CUDA graph API add to the internal/cuda surface area
