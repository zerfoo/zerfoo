# Migrating to Zerfoo v1.0

This guide covers all breaking changes between Zerfoo v0.x and v1.0, with
actionable migration steps for each.

## Overview

Zerfoo v1.0 is the first release with a backwards-compatibility guarantee
(2 years through v1.x, per [ADR-058](adr/058-api-stability-v1-contract.md)).
The major changes are:

1. **ZMF model format removed** -- GGUF is the sole model format.
2. **Repository split** -- tensor/compute/graph packages moved to `ztensor`;
   tokenizer moved to `ztoken`.
3. **`compute.Engine[T]` interface frozen** -- new capabilities use extension
   interfaces.
4. **CGo build tags removed** -- GPU bindings use purego/dlopen exclusively.
5. **High-level API introduced** -- `inference.Load` / `inference.LoadFile`
   replace manual GGUF loading.
6. **KV cache interface unified** -- `CacheProvider[T]` replaces concrete
   `*KVCache[T]` in context helpers.
7. **Sub-package maturity labels** -- packages are labeled Stable, Beta, or
   Alpha with different compatibility guarantees.

---

## Breaking Changes

### 1. ZMF Model Format Removed (ADR-037)

The ZMF/protobuf model loading path has been removed entirely. GGUF is the
sole model format.

| Removed API | Replacement |
|-------------|-------------|
| `model.LoadZMF(path)` | `inference.LoadFile(path)` with a `.gguf` file |
| `model.ExportZMF(path)` | No direct replacement; use GGUF for checkpoints |
| `model.Builder` (generic graph-from-ZMF) | Architecture-specific builders via `inference.RegisterArchitecture` |
| `graph.FuseRMSNorm()` fusion pass | Not needed -- GGUF builders emit fused ops directly |
| `model/tensor_encoder.go`, `tensor_decoder.go` | Removed (protobuf serialization) |
| `model/adapters.go` ZMF adapter code | Removed |

**Migration steps:**

1. Convert any ZMF model files to GGUF using `zonnx` (which now outputs GGUF
   instead of ZMF).
2. Replace `model.LoadZMF(path)` calls with `inference.LoadFile("model.gguf")`.
3. Remove any `github.com/zerfoo/zmf` imports -- the zmf repository is archived.
4. Remove protobuf dependencies that were only used for ZMF model loading.

### 2. Repository Split (ADR-036)

Tensor, compute, graph, and tokenizer packages have been extracted into
independent repositories.

| Old import path | New import path |
|-----------------|-----------------|
| `github.com/zerfoo/zerfoo/tensor` | `github.com/zerfoo/ztensor/tensor` |
| `github.com/zerfoo/zerfoo/compute` | `github.com/zerfoo/ztensor/compute` |
| `github.com/zerfoo/zerfoo/graph` | `github.com/zerfoo/ztensor/graph` |
| `github.com/zerfoo/zerfoo/numeric` | `github.com/zerfoo/ztensor/numeric` |
| `github.com/zerfoo/zerfoo/device` | `github.com/zerfoo/ztensor/device` |
| `github.com/zerfoo/zerfoo/types` | `github.com/zerfoo/ztensor/types` |
| `github.com/zerfoo/zerfoo/log` | `github.com/zerfoo/ztensor/log` |
| `github.com/zerfoo/zerfoo/pkg/tokenizer` | `github.com/zerfoo/ztoken` |

**Migration steps:**

1. Update import paths as shown above.
2. Run `go mod tidy` to add the new `ztensor` and `ztoken` dependencies.
3. The types are identical -- no code changes beyond import paths are needed.

### 3. Engine[T] Interface Frozen (ADR-058)

The `compute.Engine[T]` interface is frozen for v1.x. New GPU capabilities are
exposed via optional extension interfaces checked with type assertions.

| Extension Interface | Purpose |
|---------------------|---------|
| `EngineWithFP8` | FP8 E4M3 compute operations |
| `EngineWithPagedKV` | Paged KV cache memory management |

**Migration steps:**

If you have a custom `Engine[T]` implementation, it will continue to compile
unchanged. To opt into new capabilities, implement the relevant extension
interface:

```go
// Before: would have required adding methods to Engine[T]
// After: implement the extension interface

type MyEngine struct { /* ... */ }

// Existing Engine[T] methods remain unchanged.

// Opt into FP8 by implementing EngineWithFP8.
func (e *MyEngine) FP8MatMul(a, b, out unsafe.Pointer, m, n, k int) error {
    // ...
}
```

### 4. CGo Build Tags Removed

All GPU bindings (CUDA, ROCm, OpenCL, cuBLAS, cuDNN) now use purego/dlopen
exclusively. The `cuda`, `rocm`, and `opencl` build tags are no longer
recognized.

| Old build command | New build command |
|-------------------|-------------------|
| `go build -tags cuda ./...` | `go build ./...` |
| `go test -tags cuda ./...` | `go test ./...` |
| `go build -tags rocm ./...` | `go build ./...` |

**Migration steps:**

1. Remove `-tags cuda`, `-tags rocm`, and `-tags opencl` from all build
   scripts, CI pipelines, Dockerfiles, and Makefiles.
2. GPU acceleration is now detected at runtime. If the CUDA/ROCm/OpenCL shared
   libraries are present on the system, they are loaded automatically via
   `dlopen`.
3. Remove any CGo toolchain dependencies (gcc, nvcc for bindings) from your
   build environment. The CUDA kernel shared library (`libzerfoo_kernels.so`)
   is still compiled separately but is loaded at runtime.

### 5. High-Level API Changes

The top-level `zerfoo` package now provides a stable convenience API.

| Old API (v0.x) | New API (v1.0) |
|----------------|----------------|
| `zerfoo.Load(pathOrID)` returning `(*zerfoo.Model, error)` | Same signature, now marked `Stable` |
| Manual GGUF load + graph build + generator creation | `inference.Load(modelID, opts...)` or `inference.LoadFile(path, opts...)` |

The `inference.Load` function now accepts HuggingFace model IDs in addition
to local file paths. Short aliases like `"gemma-3-1b-q4"` are supported.

**Before (v0.x):**

```go
gguf, err := inference.LoadGGUF("/path/to/model.gguf")
// manually build graph, create engine, create generator...
gen := generate.NewGenerator[float32](g, tok, engine, cfg)
text, _ := gen.Generate(ctx, "Hello", generate.DefaultSamplingConfig())
```

**After (v1.0):**

```go
m, err := inference.Load("gemma-3-1b-q4",
    inference.WithDevice("cuda"),
    inference.WithMaxSeqLen(4096),
)
if err != nil {
    log.Fatal(err)
}
defer m.Close()

text, err := m.Generate(ctx, "Hello",
    inference.WithMaxTokens(256),
    inference.WithTemperature(0.7),
)
```

The low-level `generate.Generator` API remains available for users who need
fine-grained control over the generation loop.

### 6. KV Cache Context Helpers

The concrete-type KV cache context helpers are deprecated in favor of the
interface-based `CacheProvider[T]` versions.

| Deprecated | Replacement |
|------------|-------------|
| `generate.WithKVCache[T](ctx, *KVCache[T])` | `generate.WithCache[T](ctx, CacheProvider[T])` |
| `generate.GetKVCache[T](ctx)` | `generate.GetCache[T](ctx)` |

The deprecated functions still work in v1.x but will be removed in v2.0.

**Migration steps:**

1. Replace `generate.WithKVCache` with `generate.WithCache`.
2. Replace `generate.GetKVCache` with `generate.GetCache`.
3. Both `*KVCache[T]` and `*TensorCache[T]` implement `CacheProvider[T]`, so
   no other changes are needed.

### 7. Package Rename: wolf to timeseries

The internal time-series inference package was renamed.

| Old import path | New import path |
|-----------------|-----------------|
| `inference/wolf` | `inference/timeseries` |
| `cmd/wolf_train` | `cmd/ts_train` |

**Migration steps:**

Update import paths from `wolf` to `timeseries`.

---

## Sub-Package Maturity Labels

Packages are now labeled by maturity level, which determines their
compatibility guarantee:

| Level | Guarantee | Packages |
|-------|-----------|----------|
| **Stable** | Full v1.x backwards compatibility | `inference/`, `generate/`, `serve/`, `model/`, `layers/` |
| **Beta** | Schema preserved, behavior may change | `training/`, `distributed/` |
| **Alpha** | May be restructured | `training/nas/`, `training/automl/` |

---

## Deprecation Policy

Starting with v1.0, all deprecations follow this protocol:

1. A `// Deprecated:` comment is added to the symbol.
2. The deprecated symbol coexists with its replacement for at least 2 minor
   releases.
3. Removal happens only in v2.0.

A deprecation linter (`cmd/deprecation-check`) is available to scan your code
for usage of deprecated symbols.

---

## New Features in v1.0

These are additive and do not require migration, but are worth knowing about:

- **Architecture registry** -- `inference.RegisterArchitecture` / `inference.ListArchitectures` for pluggable model support.
- **12 model architectures** -- Llama 3, Gemma 3, Mistral, Qwen 2, Phi 3/4, DeepSeek V3, Falcon, Command R, Mixtral, RWKV, Jamba, Mamba 3.
- **Speculative decoding** -- `inference.Model.SpeculativeGenerate` and `generate.WithSpeculativeDraft`.
- **Paged KV cache** -- `generate.WithPagedKV` for memory-efficient serving.
- **Prefix caching** -- `generate.WithPrefixCache` for shared system prompt reuse.
- **FP16 KV cache** -- `generate.WithGeneratorKVDtype("fp16")` for 2x bandwidth reduction.
- **Grammar-constrained decoding** -- `inference.WithGrammar` and `serve.ResponseFormat{Type: "json_schema"}`.
- **Tool calling** -- `serve.Tool` / `serve.ToolChoice` in the OpenAI-compatible API.
- **Vision and audio** -- multimodal inference with LLaVA, SigLIP, and Whisper.
- **Batch generation** -- `inference.Model.GenerateBatch` and `serve.BatchScheduler`.
- **Continuous batching** -- `serve.NewBatchScheduler` for high-throughput serving.
- **LoRA/QLoRA fine-tuning** -- `training/lora/` and `cmd/finetune`.
- **FSDP distributed training** -- `distributed/fsdp/` with NCCL AllGather/ReduceScatter.
- **HuggingFace model downloads** -- `zerfoo pull` CLI with resume and SHA256 verification.
- **SSM support** -- Mamba block, SSM state management, Jamba hybrid architecture.

---

## Import Path Reference

The v1.0 import path remains `github.com/zerfoo/zerfoo` (implicit v1 per Go
convention). When v2.0 is released, it will use `github.com/zerfoo/zerfoo/v2`.

```go
import (
    "github.com/zerfoo/zerfoo"           // top-level convenience API
    "github.com/zerfoo/zerfoo/inference"  // model loading and generation
    "github.com/zerfoo/zerfoo/generate"   // low-level generation control
    "github.com/zerfoo/zerfoo/serve"      // OpenAI-compatible server
    "github.com/zerfoo/zerfoo/training"   // training framework (Beta)
    "github.com/zerfoo/ztensor/tensor"    // tensor types
    "github.com/zerfoo/ztensor/compute"   // compute engine interface
    "github.com/zerfoo/ztensor/graph"     // computation graph
    "github.com/zerfoo/ztoken"            // BPE tokenizer
)
```
