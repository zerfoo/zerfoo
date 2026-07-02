# ADR 026: Megakernel Code Generation for Single-Launch Decode

## Status
Accepted (revised)

## Date
2026-03-06

## Context
Even with CUDA graph capture and fused kernels (ADR 024), the decode loop still
launches multiple CUDA kernels per token (one per fused op group). Each kernel
reads from and writes to global GPU memory for intermediates. For memory-bandwidth
bound workloads (Q4 single-token decode on DGX Spark GB10 with 273 GB/s), the
overhead of multiple global memory round-trips is the dominant bottleneck.

llama.cpp achieves 24-38 tok/s on the same hardware by having hand-tuned kernels
that minimize global memory traffic. The theoretical upper bound is ~182 tok/s
for a 1.5GB Q4 model (273 GB/s / 1.5GB).

Zerfoo's architecture is heavily compositional: all complex layers (TransformerBlock,
SDPA, SwiGLU, etc.) decompose into a small set of primitive Engine ops (Add, Mul,
MatMul, Softmax, Exp, RMSNorm, etc.). The compiled ExecutionPlan in graph/compile.go
captures this as a flat []Instruction array where each instruction has an OpName
(the primitive op type), InputIdx/OutputIdx (slot references), and the tensor
shapes are known from the warmup forward pass.

This means the code generator does not need architecture-specific templates. It
walks the instruction list and emits a CUDA device function for each primitive op.
Any model that compiles into an ExecutionPlan -- transformers, RNNs, CNNs, S4,
HRM -- automatically gets a megakernel.

## Decision
Phase 34 Track B will implement a composition-aware code generator that:

1. Reads the compiled ExecutionPlan []Instruction list. Each instruction has
   OpName (e.g., "Add", "MatMul", "RMSNorm", "Softmax"), InputIdx, OutputIdx,
   and the slot shapes are known from Compile().
2. For each OpName, emits the corresponding CUDA device function that operates
   on data in registers or shared memory (not global memory).
3. Chains the device functions in instruction order, passing intermediates via
   registers/shared memory. Only weight reads and KV cache access touch global
   memory.
4. Wraps everything in a single __global__ kernel with cooperative launch for
   grid-level synchronization between ops that need cross-block communication
   (e.g., reductions in Softmax, RMSNorm).
5. Compiles the generated .cu at model-load time via nvrtc (JIT) or cached nvcc
   (AOT). Loads via cuModuleLoad.

This is architecture-agnostic. The generator maps primitive op names to CUDA
device functions. It does not need to understand what a "transformer" or "RNN" is.
Any model that runs through ExecutionPlan gets a megakernel automatically.

The primitive op -> device function mapping:
- Add, Sub, Mul, Div -> register elementwise
- MatMul (Q4 GEMV) -> register dequant + dot product
- MatMul (F32 GEMV) -> register dot product
- RMSNorm -> shared memory reduction + register normalize
- Softmax -> shared memory max/sum reduction + register exp/div
- Exp, Log, Sqrt, Rsqrt, Tanh -> register unary
- SiLU -> register (x * sigmoid(x))
- Gather -> global memory read to registers
- Concat, Split -> register reindexing
- RoPE -> register rotation

## Consequences
Positive:
- One kernel launch per token (true minimum)
- All intermediates in registers/shared memory (near-zero global memory overhead)
- Approaches theoretical memory bandwidth limit
- Expected 5-10x improvement over current approach (target: 50+ tok/s)
- Architecture-agnostic: any model compiled via ExecutionPlan gets a megakernel
- Leverages Zerfoo's compositional design (complex layers = primitive ops)
- No architecture-specific templates needed

Negative:
- High implementation complexity (code generator + CUDA cooperative kernel)
- JIT compilation adds model-load latency (mitigated by caching compiled modules)
- Requires CUDA 12+ cooperative launch (grid sync) -- available on DGX Spark
- Debugging generated CUDA code is harder than hand-written kernels
- Register pressure may limit the hidden dimension sizes that fit
- Different model sizes may need different tiling strategies
- Only works for batch=1 decode (activation tensors must fit in registers/shmem)
- Unsupported primitive ops fall back to ExecutionPlan.Run() for the full model
