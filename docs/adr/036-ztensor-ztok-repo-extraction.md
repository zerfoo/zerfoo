# ADR 036: Extract ztensor and ztoken as Independent Repositories

## Status

Accepted

## Date

2026-03-15

## Context

The zerfoo repository contains ~50,000 lines of Go spanning tensor storage,
compute engines (CPU/GPU/ROCm/OpenCL), computation graphs, 25+ CUDA kernels,
ARM NEON SIMD assembly, neural network layers, inference runtimes, training
frameworks, distributed training, an OpenAI-compatible server, and CLI tools.

This creates two problems:

1. **Import weight.** Any Go developer who wants GPU-accelerated tensors or a
   BPE tokenizer must `go get github.com/zerfoo/zerfoo`, pulling in gRPC,
   protobuf, and the entire inference/training/serving stack.

2. **API surface.** The monolithic structure makes it difficult to version the
   tensor/compute layer independently from the inference/serving layer. A
   breaking change to the server API forces a new release even if tensors are
   unchanged.

Dependency analysis confirms clean cut points:

- `tensor/`, `compute/`, `numeric/`, `device/`, `graph/`, `types/`, `log/`,
  `metrics/runtime/`, and all `internal/` GPU packages import only each other,
  `float16`, and `float8`. Zero imports of inference, training, serving, or gRPC.

- `pkg/tokenizer/` has zero internal imports. It is a pure standalone package.

## Decision

Extract two new repositories from the zerfoo monolith:

**github.com/zerfoo/ztensor** -- Tensor, compute, and graph compiler library.
Contains: tensor/, compute/, numeric/, device/, graph/, types/, log/,
metrics/runtime/, internal/cuda/, internal/xblas/, internal/gpuapi/,
internal/cublas/, internal/cudnn/, internal/hip/, internal/miopen/,
internal/opencl/, internal/rocblas/, internal/tensorrt/, internal/nccl/,
internal/clblast/, internal/workerpool/, internal/codegen/, testing/testutils/.

**github.com/zerfoo/ztoken** -- BPE tokenizer library.
Contains: pkg/tokenizer/ (tokenizer.go, bpe.go, loader.go) plus
model/gguf/tokenizer.go extracted as a standalone GGUF tokenizer loader.

After extraction, zerfoo imports ztensor and ztoken as dependencies.

New dependency graph:
```
float16, float8 --> ztensor --> zerfoo
                    ztoken -----> zerfoo
zmf --------------> zerfoo, zonnx
```

## Consequences

**Positive:**
- Go developers can use GPU tensors (`go get github.com/zerfoo/ztensor`) without
  pulling inference, gRPC, or protobuf dependencies.
- Go developers can use BPE tokenizers (`go get github.com/zerfoo/ztoken`) without
  any ML framework dependency.
- Independent versioning: tensor API can stabilize at v1.0 before inference.
- Faster CI: ztensor and ztoken test suites run independently.
- Cleaner contributor experience: tensor contributors do not need gRPC knowledge.

**Negative:**
- Cross-repo refactoring is harder than single-repo refactoring.
- Two additional repositories to maintain (releases, CI, documentation).
- Import path changes require a migration period with type aliases in zerfoo.
- GGUF tokenizer extraction requires splitting model/gguf/ across two repos
  or duplicating the relevant extraction logic.
