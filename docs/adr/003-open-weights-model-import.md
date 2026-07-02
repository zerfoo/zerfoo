# ADR-003: Open Weights Model Import

**Status:** Accepted
**Phase:** 6
**Date:** 2026-03-02

## Context

Zerfoo could train and run inference on models built with its layer API, but
importing pre-trained open-weights models (Gemma 3, Kimi-VL) required closing
gaps in the ONNX import pipeline (zonnx repo) and layer registry.

Gap analysis identified blockers: zonnx converter missing TENSOR attribute and
UINT8 dtype; MatMulNBits and Constant not registered; vision encoder operators
(Conv2d, Pad, Slice, Resize, BatchNorm, GlobalAveragePool) not implemented;
MoE not implemented.

## Decision

### 4-Bit Weight Packing

MatMulNBits stores 4-bit weights packed two-per-byte in UINT8 tensors. ZMF uses
DataType=UINT8. Dequantization happens in MatMulNBits.Forward() using
numeric.Unpack4BitSlice. Supports symmetric and asymmetric quantization with
per-block scales and optional zero-points.

### Conv2d Strategy

Direct nested-loop convolution (not im2col + MatMul). Simpler, correct for
inference workloads, avoids allocating large intermediate matrices. Deviation
from original plan noted.

### MoE Design

MoEGate routes tokens to top-k experts via softmax + topK selection. Gate weight
passed as runtime Forward input (not from params) to match the ONNX/ZMF pattern.
MixtureOfExperts dispatches to selected experts and aggregates weighted outputs.
Expert loading from ZMF sub-graphs deferred (tech debt).

### Operator Inventory

New operators implemented and registered:

| Operator | File | Category |
|----------|------|----------|
| Softmax | layers/activations/softmax.go | Activation |
| Sigmoid builder | layers/activations/registry.go | Activation |
| Erf | layers/activations/erf.go | Activation |
| LayerNormalization | layers/normalization/registry.go | Normalization |
| BatchNormalization | layers/normalization/batch_norm.go | Normalization |
| Slice | layers/core/slice.go | Core |
| Pad | layers/core/pad.go | Core |
| TopK | layers/core/topk.go | Core |
| Conv2d | layers/core/conv2d.go | Core |
| GlobalAveragePool | layers/core/global_avg_pool.go | Core |
| Resize | layers/core/resize.go | Core |
| MoEGate | layers/core/moe.go | Core |
| MixtureOfExperts | layers/core/moe.go | Core |
| Constant | layers/core/constant.go | Core |

### Multi-Repo Discipline

zonnx and zerfoo are separate repos. Pre-commit hooks reject multi-directory
commits. zonnx converter fixes committed in zonnx; layer/model changes in zerfoo.
ONNX-to-ZMF conversion in zonnx handles special cases (Slice/Pad/TopK input
promotion, Resize scales/sizes, MatMulNBits dequantization).

## Consequences

- Gemma 3 end-to-end import validated (forward pass + greedy decode).
- SigLIP vision encoder and Kimi-VL connector validated.
- 13 new operators added to registry; total 56+ layers.
- MatMulNBits dequantization at converter level (not runtime) for standard
  MatMul path.
- Expert loading from sub-graphs is documented tech debt.
- All parity tests are env-var gated (skip gracefully without model files).

### Key Files

- `layers/core/moe.go` -- MoEGate, MixtureOfExperts
- `layers/core/conv2d.go` -- Conv2d (nested-loop)
- `layers/core/constant.go` -- Constant node
- `tests/parity/gemma3_test.go` -- Gemma 3 parity test
- `tests/parity/siglip_test.go` -- SigLIP/Kimi-VL parity tests
