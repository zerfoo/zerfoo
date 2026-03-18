# ADR 048: Mamba/SSM Architecture Support

## Status
Accepted

## Date
2026-03-17

## Context
The post-Transformer era is converging on hybrid architectures combining attention,
SSMs (State Space Models), and MoE. Mamba-3 achieves 5x throughput vs Transformers
with linear sequence scaling, making it attractive for Wolf's time-series inference
where sequences can be very long. MoE-Mamba matches Mamba performance in 2.2x fewer
training steps. Jamba combines Transformer + Mamba + MoE for hundreds-of-thousands
token context windows.

Zerfoo must support these architectures to remain competitive as the model landscape
shifts away from pure Transformers in 2026-2027.

## Decision
Implement SSM support in inference/ and layers/ as first-class citizens:

1. Selective Scan Kernel (ztensor/internal/cuda/selective_scan.cu):
   - Implements the SSM parallel scan: h_t = A*h_{t-1} + B*x_t, y_t = C*h_t + D*x_t
   - Batched across D dimensions with CUDA parallel scan (prefix sum pattern)
   - Supports complex-valued A matrices (Mamba-3 complex dynamics)
   - CPU fallback via sequential scan in layers/ssm/selective_scan.go

2. Mamba Block (layers/ssm/mamba_block.go):
   - Implements the full Mamba block: input projection, 1D conv, selective scan,
     output projection
   - Generic over [T tensor.Numeric]
   - Supports both Mamba-1 (real A) and Mamba-3 (complex A, MIMO)

3. GGUF Loader Extension (inference/arch_mamba.go):
   - Maps GGUF tensor names for Mamba-3 checkpoints
   - Builds computation graph mixing Mamba blocks and attention layers (for Jamba)

4. Hybrid Graph Builder (inference/arch_jamba.go):
   - Interleaves Mamba blocks and Transformer layers per architecture config
   - Config driven by GGUF metadata (mamba_layer_indices)

GGUF support for Mamba-3 is contingent on llama.cpp defining the tensor naming
convention; Zerfoo follows llama.cpp GGUF schemas for compatibility.

## Consequences
Positive:
- Unlocks Mamba-3 5x throughput advantage for long-sequence Wolf inference tasks
- Jamba support enables hundreds-of-thousands token context for market data analysis
- Linear scaling with sequence length removes attention's O(n^2) bottleneck

Negative:
- Selective scan is harder to CUDA graph capture than attention (stateful recurrence)
- Complex A matrix support (Mamba-3) requires complex number arithmetic in the engine
- GGUF schema for Mamba-3 may not be standardized until mid-2026; risk of churn
- SSM hidden state must be maintained across decode steps (unlike KV cache which
  is append-only); requires new state management in the session
