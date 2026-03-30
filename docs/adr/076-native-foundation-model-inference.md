# ADR 076: Native Go Foundation Model Inference via GGUF

## Status
Accepted

## Date
2026-03-29

## Context
Foundation models for time series (Chronos-2, TiRex, Moirai-2) forecast unseen series
zero-shot, pre-trained on billions of time points. All three have HuggingFace Transformers
implementations. We need to run these from Go.

A Python bridge was considered and rejected -- it contradicts Zerfoo's core principle of
Go-native, zero-CGo inference. More importantly, Zerfoo already has the infrastructure to
support these architectures natively:

- Engine[T] with 56+ tensor operations (MatMul, LayerNorm, Softmax, GELU, RoPE, etc.)
- Computation graph with compilation and CUDA graph capture
- GGUF model loading with architecture-specific graph builders
- zonnx for converting HuggingFace weights to GGUF format
- Existing transformer graph builders (Llama, Gemma, Mistral, PatchTST, TTM)

Each foundation model is a new graph builder composing existing primitives, not a
reimplementation from scratch.

## Decision
Implement all three foundation models as native Go inference paths:

1. **Weight conversion**: Use zonnx (or model-specific converter scripts) to convert
   HuggingFace SafeTensors/PyTorch weights to GGUF format.

2. **Architecture graph builders**: One file per model following the existing pattern
   (arch_chronos.go, arch_tirex.go, arch_moirai.go) in inference/timeseries/.

3. **New layer primitives** (only where needed):
   - Chronos-2: Value tokenizer (bin continuous values into discrete tokens)
   - TiRex: sLSTM cell (exponential gating), mLSTM cell (matrix memory with covariance)
   - Moirai-2: Any-variate input projection, frequency embedding

4. **Priority order**: TiRex first (35M params, simplest new layers needed, #1 on
   GIFT-Eval), then Chronos-2, then Moirai-2.

## Consequences
Positive:
- Zero Python dependency. Pure Go, consistent with Zerfoo's core principles.
- GPU acceleration via existing Engine[T] and CUDA graph capture.
- GGUF format enables mmap loading, quantization, and sharing with llama.cpp ecosystem.
- New layer primitives (xLSTM cells) are reusable for future architectures.

Negative:
- Must validate numerical parity against HuggingFace reference implementations.
- Weight conversion step required (one-time per model).
- xLSTM cells are new code that needs thorough testing.
