# ADR-018: Model Parity Testing on DGX Spark

## Status

Accepted (2026-03-04)

## Context

Phase 20 validated all 66 GPU packages on the DGX Spark GB10 (Blackwell sm_121,
CUDA 13.0.2, ARM64). However, 25+ parity tests were skipped because no ZMF
model files existed on the device. The parity test framework in
`tests/parity/helpers_test.go` uses `envOrSkip(t, key)` to gate on environment
variables like `LLAMA3_ZMF_PATH` and `QWEN_ZMF_PATH`.

Phase 21 addressed this by downloading ONNX models from HuggingFace, converting
them to ZMF via the `zonnx` CLI, deploying on DGX Spark, and running the full
parity suite. Ten ONNX-compatibility bugs surfaced and were fixed.

## Decision

Download, convert (ONNX -> ZMF via zonnx), and deploy model files for 7 model
families on DGX Spark. Fix ONNX compatibility issues as they surface. Use
`optimum-cli export onnx` when HuggingFace repos lack pre-built ONNX files.

## Bug Fixes

Ten issues were fixed across the zerfoo and zonnx repositories:

| # | Issue | Fix | Commit |
|---|-------|-----|--------|
| 1 | Reshape: ONNX dynamic 2-input mode unsupported | Support 2-input Reshape with shape tensor | 2a520c4 |
| 2 | Builder: dynamic shape input dropped for Reshape | Keep shape input wiring in builder | f757b29 |
| 3 | Builder: ZMF proto fields (Perm, Epsilon, Axis) not promoted | Promote proto fields during load | b12847a |
| 4 | Where: scalar broadcasting missing | Add scalar broadcasting to Where op | 7f8d73f |
| 5 | Builder: Reshape node not rebuilt after shape extraction (Pass 2) | Rebuild Reshape in Pass 2 | 4644fe3 |
| 6 | MatMul: batched (4D+) matmul unsupported | Support batched matmul dimensions | 5b94cbb |
| 7 | tensor_decoder: empty tensor data not guarded | Return error for empty data | da82e78 |
| 8 | Cos/Sin: elementwise layers missing for RoPE | Add Cos and Sin layers | e119caa |
| 9 | zonnx: external data files not loaded | Load ONNX external data (model.onnx_data) | 8830eb5 |
| 10 | Generator: 3D input shape incompatible with Gather | Use 2D [1, seqLen] input shape | 8f003ad |
| 11 | Tokenizer: array-of-arrays merges format unsupported | Support both string and array merges formats | 830905e |
| 12 | Gather: embedded constant indices not detected | Add NewWithIndices + attribute detection in BuildGather | acae458 |
| 13 | Slice: hybrid mode (dynamic starts, attribute ends) unsupported | Add 2-input hybrid mode to Slice.Forward | 7cafb68 |
| 14 | Slice: out-of-range start for zero-dim tensors | Clamp start to dim size, handle empty ranges | 0937e90 |
| 15 | zonnx: integer initializer inputs promoted to attributes | Keep initializer inputs as graph references | a1dfa04 |
| 16 | LessOrEqual: missing elementwise comparison layer | Add LessOrEqual layer | 8444724 |
| 17 | Or: missing elementwise logical layer | Add Or layer | b08bf51 |
| 18 | Squeeze/Tile/Mod/Gemm: missing ops for SigLIP | Add 4 new layers | db0a2f6 |

## Results

| Test | Status | Notes |
|------|--------|-------|
| FlashAttentionGQA | PASS | |
| Llama3 ForwardPass | PASS | onnx-community/Llama-3.2-1B |
| Llama3 GreedyDecode | PASS | |
| Llama3 Generation | PASS | |
| Qwen25 ForwardPass | PASS | Qwen/Qwen2.5-0.5B |
| Qwen25 GreedyDecode | PASS | |
| Qwen25 Generation | PASS | |
| Gemma3 ForwardPass | PASS | google/gemma-3-1b-it (optimum export) |
| Gemma3 GreedyDecode | PASS | |
| Gemma3 Generation | PASS | |
| Mistral ForwardPass | PASS | mistralai/Mistral-7B-Instruct-v0.3 |
| Mistral GreedyDecode | PASS | |
| Mistral Generation | PASS | |
| Phi3 ForwardPass | PASS | microsoft/Phi-3-mini-4k-instruct |
| Phi3 GreedyDecode | PASS | |
| Phi3 Generation | PASS | |
| MultiGPU DualDevice | SKIP | Single GPU device |
| DeepSeek (3 tests) | SKIP | Model too large (671B) for 128GB |
| SigLIP ForwardPass | PASS | Fixed in Phase 22 (Squeeze scalar + Concat rank alignment) |

**Summary:** 18 PASS, 4 SKIP (DeepSeek too large, 1 test blocked on
single-GPU hardware).

## Infrastructure

- **Test automation:** `scripts/dgx-spark-parity.sh` sets ZMF env vars and runs
  the full parity suite.
- **ONNX export:** Python venv with `optimum-cli export onnx` for models without
  pre-built ONNX files on HuggingFace.
- **ZMF conversion:** `zonnx convert <onnx-dir> <output.zmf>` on DGX Spark.

## Consequences

- 5 of 7 model families pass all parity tests (Llama3, Qwen25, Gemma3, Mistral,
  Phi3). DeepSeek-V3 (671B MoE) exceeds 128GB memory. SigLIP vision model has
  a Concat shape mismatch in its ONNX graph.
- Multi-GPU parity test requires a second DGX Spark unit connected via
  ConnectX-7.
- The 18 bug fixes improved ONNX compatibility for all models, not just the
  tested families.
- The zonnx root-cause fix (commit a1dfa04) eliminates the need for workaround
  code in the builder; dead `materializeConstantAttrs` removed in 671495f.
- 7 new ONNX ops added: LessOrEqual, Or, Squeeze, Tile, Mod, Gemm (for Mistral
  and SigLIP support).
