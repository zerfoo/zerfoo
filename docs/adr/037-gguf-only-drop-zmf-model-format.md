# ADR 037: GGUF as Sole Model Format, Drop ZMF for Model Storage

## Status

Accepted

## Date

2026-03-15

## Context

Zerfoo has two execution paths for inference:

1. **GGUF path** (architecture-specific builders): Loads GGUF files via mmap,
   builds the computation graph using hand-tuned fused operations
   (GroupedQueryAttention, FusedAddRMSNorm, FFN). Achieves 241 tok/s with
   99.5% CUDA graph capture on Gemma 3 1B Q4_K.

2. **ZMF/ONNX path** (stored graph): Converts ONNX models to ZMF via zonnx,
   loads the explicit computation graph from protobuf, executes decomposed
   operations. Achieves 4-16 tok/s with 1-4% CUDA graph capture.

The ZMF path will never match the GGUF path because:
- Fused operations require architecture knowledge that a generic stored graph
  cannot provide.
- Decomposed RMSNorm (Pow, ReduceMean, Add, Sqrt, Div, Mul) produces 6
  kernel launches where fused RMSNorm produces 1.
- The graph fusion pass (graph/fusion.go) that attempts to reconstruct fused
  operations from decomposed patterns has been blocked by a runtime slot
  resolution bug since Phase 16 (PR #70).

Every model people want to run (Llama, Gemma, Mistral, Qwen, Phi, DeepSeek)
has GGUF variants on HuggingFace. Adding a new architecture builder is ~200
lines of Go.

ZMF uses protobuf for tensor storage, which is fundamentally wrong for large
binary data: no mmap, 2x memory during deserialization, no random tensor access.

Training checkpoint saving is currently unimplemented (SaveModel returns
"not implemented").

## Decision

1. **GGUF is the sole model loading format.** Remove the ZMF model loading
   and graph construction path from zerfoo.

2. **Delete ZMF-dependent code from zerfoo:**
   - model/builder.go (generic graph-from-ZMF builder, ~824 lines)
   - model/zmf_loader.go, zmf_exporter.go, zmf_mmap.go (~347 lines)
   - model/tensor_encoder.go, tensor_decoder.go (~486 lines)
   - model/adapters.go ZMF-specific adapter code (~487 lines)
   - graph/fusion.go (RMSNorm fusion pass, ~213 lines)
   - Associated tests (~5,124 lines)
   - Total removal: ~7,500 lines

3. **Close PR #70** (RMSNorm fusion bug). The problem is dissolved.

4. **Pivot zonnx from ONNX-to-ZMF to ONNX-to-GGUF.** This makes zonnx the
   only pure-Go ONNX-to-GGUF converter.

5. **Archive the zmf repository.** ZMF is no longer consumed by any project.

6. **Training checkpoints use GGUF.** When checkpoint saving is implemented,
   it writes GGUF files containing model weights plus metadata. GGUF already
   supports arbitrary key-value metadata for storing training state (epoch,
   optimizer name, learning rate). Optimizer state (Adam moments) is stored
   as additional tensors in the same GGUF file with a naming convention
   (e.g., `optimizer.m.layers.0.weight`, `optimizer.v.layers.0.weight`).

## Consequences

**Positive:**
- ~7,500 lines of code removed from zerfoo.
- One inference path instead of two (massive simplification).
- PR #70 (RMSNorm fusion bug) is closed without fixing.
- zerfoo no longer depends on `github.com/zerfoo/zmf` or protobuf for model loading.
- zonnx becomes more useful (only Go ONNX-to-GGUF converter in the ecosystem).
- Training checkpoints use a community-standard format that other tools can read.

**Negative:**
- Cannot run arbitrary ONNX models without an architecture-specific builder.
  (This path produced 4-16 tok/s with semi-coherent output, so the loss is
  minimal.)
- Protobuf dependency remains for gRPC distributed training but is no longer
  on the model-loading critical path.
- GGUF writing is new code that must be implemented in zonnx and eventually
  in zerfoo for training checkpoints.
- zmf repository is archived (sunk cost in protobuf schema design).
