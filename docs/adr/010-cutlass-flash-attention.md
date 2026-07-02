# ADR-010: CUTLASS Flash Attention

**Phase:** 13
**Status:** Accepted
**Date:** 2026-03-03

## Context

The attention layer in `layers/attention/multi_head.go` computes attention as
three separate operations:

```
scores = Q * K^T                    // MatMul kernel launch
scores = scores / sqrt(head_dim)    // Scale kernel launch
scores = softmax(scores)            // Softmax kernel launch
output = scores * V                 // MatMul kernel launch
```

Each step materializes intermediate tensors in GPU memory and launches separate
CUDA kernels. For a sequence of length `n` and head dimension `d`:
- Memory: O(n^2) for the scores matrix (e.g., 4096^2 * 4 bytes = 64 MB per
  head per batch element).
- Kernel launches: 4 separate launches with GPU memory round-trips between
  each.

Flash attention (Dao et al., 2022) fuses these operations into a single tiled
kernel that processes Q and K in blocks, maintaining running softmax statistics
in registers. This reduces:
- Memory: O(n) -- no materialized n x n scores matrix.
- Kernel launches: 1 fused launch.

Neither cuDNN nor TensorRT provides a general flash attention implementation
for arbitrary head dimensions and causal masks. CUTLASS provides the building
blocks to write custom fused kernels.

## Decision

### Build Integration

CUTLASS is a header-only C++ template library. The flash attention kernel is
written as a `.cu` file that includes CUTLASS headers. It compiles with nvcc
into the existing `libcudakernels.a` static library alongside the 15 existing
custom CUDA kernels.

No new build system is needed. The existing Makefile/build script that compiles
`.cu` files in `internal/cuda/kernels/` already produces `libcudakernels.a`.
The CUTLASS headers are an additional include path: `-I$(CUTLASS_PATH)/include`.

A new build tag `cutlass` (used alongside `cuda`) gates the flash attention
code path. This allows builds with CUDA but without CUTLASS headers to still
compile.

### Kernel Interface

The flash attention kernel exposes a C function signature:

```c
void flash_attention_forward_f32(
    const float* Q,        // [batch, heads, seq_len, head_dim]
    const float* K,        // [batch, heads, seq_len, head_dim]
    const float* V,        // [batch, heads, seq_len, head_dim]
    float* O,              // [batch, heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int causal,            // 1 for causal mask, 0 for no mask
    cudaStream_t stream
);
```

The Go side calls this via CGo from a build-tag-gated file
(`layers/attention/flash_cuda.go`).

### Tiled Implementation

The kernel divides the sequence dimension into blocks (tiles). Each thread
block processes one tile of Q rows against all K/V tiles:

1. Load Q tile into shared memory.
2. For each K/V tile:
   a. Load K tile into shared memory.
   b. Compute partial scores (Q_tile * K_tile^T).
   c. Update running max and sum for numerically stable softmax.
   d. Load V tile, accumulate weighted output.
3. Write final output tile (rescaled by softmax denominator).

Tile size is chosen to fit in shared memory: typically 64 or 128 rows per
tile, depending on head_dim. For Phase 13, use a fixed tile size optimized
for head_dim = 64 and 128 (covering Gemma, Llama, Mistral, Qwen, Phi).

### Causal Mask

When `causal=1`, the kernel applies an upper-triangular mask: position `i` can
only attend to positions `j <= i`. This is implemented by skipping K/V tiles
that are entirely in the future and masking individual elements within
boundary tiles. No separate mask tensor is needed.

### Dispatch Pattern

Build-tag-gated file pair:
- `layers/attention/flash_cuda.go` (`//go:build cuda && cutlass`): dispatches
  to the CUTLASS flash attention kernel when the input is on GPU.
- `layers/attention/flash_nocuda.go` (`//go:build !(cuda && cutlass)`): falls
  back to the naive three-step attention.

The `MultiHeadAttention` forward method calls a `computeAttention` function
that resolves to the appropriate implementation at compile time.

### Scope Limitations

- Float32 only. FP16/BF16 flash attention deferred.
- Forward pass only. Backward pass (training) deferred.
- Fixed tile sizes for head_dim = 64 and 128. Other head dimensions fall back
  to naive attention.
- No variable-length batching (all sequences in a batch must have the same
  length, with padding).

## Consequences

### Positive

- 2x+ speedup for attention at seq_len >= 512 (the dominant compute for LLM
  inference with long contexts).
- O(n) memory instead of O(n^2) enables much longer sequences without OOM.
- Single kernel launch eliminates inter-kernel GPU memory traffic.
- No new runtime dependencies (CUTLASS is header-only, compiled into the
  binary).
- Backwards compatible: fallback to naive attention when CUTLASS is not
  available.

### Negative

- CUTLASS headers required at build time (not a runtime dependency).
- Template instantiation can increase compile time significantly.
- Fixed tile sizes limit efficiency for non-standard head dimensions.
- CUTLASS API is complex and version-sensitive (target CUTLASS >= 3.0).
- Numerical results differ slightly from naive attention (1e-4 tolerance)
  due to different summation order in the online softmax.

### Files Added

- `internal/cuda/kernels/flash_attention.h` -- C function declaration
- `internal/cuda/kernels/flash_attention.cu` -- tiled flash attention kernel
- `internal/cuda/kernels/flash_attention.go` -- CGo binding (`//go:build cuda && cutlass`)
- `internal/cuda/kernels/flash_attention_test.go` -- kernel-level parity tests
- `layers/attention/flash_cuda.go` -- GPU flash attention dispatch (`//go:build cuda && cutlass`)
- `layers/attention/flash_nocuda.go` -- naive attention fallback (`//go:build !(cuda && cutlass)`)
- `layers/attention/flash_nocuda_test.go` -- fallback dispatch and parity tests
- `tests/parity/flash_attention_test.go` -- GPU parity test and benchmark

### Files Modified

- `layers/attention/scaled_dot_product_attention.go` -- calls `tryFlashForward`
  before naive path when mask is nil
- `internal/cuda/kernels/Makefile` -- added `flash_attention.cu` to SRCS
