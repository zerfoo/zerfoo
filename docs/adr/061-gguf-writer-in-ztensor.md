# ADR 061: Shared GGUF Writer in ztensor

## Status

Accepted

## Date

2026-03-18

## Context

Zerfoo adopted GGUF as its sole model format (ADR-037). Reading GGUF is handled
by `zerfoo/model/gguf` (parser, loader, tokenizer extraction). However, writing
GGUF has no shared implementation. Five independent GGUF writers exist:

1. `zonnx/pkg/gguf/Writer` -- Best implementation. Buffered writer with typed
   metadata helpers, tensor alignment, dimension reversal. Used for ONNX-to-GGUF
   conversion.
2. `zerfoo/training/lora/checkpoint.go` -- Hand-rolled `writeGGUFString` helpers
   for LoRA adapter checkpoints.
3. `zerfoo/training/nas/export.go` -- Hand-rolled GGUF write for NAS architecture
   export with round-trip validation.
4. `zerfoo/distributed/fsdp/checkpoint.go` -- Generic `writeGGUF[T]` for
   distributed training checkpoints.
5. `zerfoo/cmd/ts_train/main.go` -- Copy-pasted `saveModelGGUF` for time-series
   model saving.

Additionally, `training/adapter.go:SaveModel()` returns "not implemented" because
there is no shared writer to build on.

The GGUF writer cannot live in zerfoo because zonnx cannot import zerfoo (wrong
dependency direction). It cannot stay in zonnx because zerfoo cannot import zonnx
either.

## Decision

1. **Create `gguf/` package in ztensor** (`github.com/zerfoo/ztensor/gguf`) with
   a `Writer` type based on zonnx's existing implementation. This package handles
   GGUF v3 binary serialization: header, metadata KV pairs, tensor info, aligned
   tensor data.

2. **The ztensor `gguf/` package is format-only.** It knows how to write GGUF
   binary format but has zero domain knowledge. No model config, no architecture
   names, no tensor name mapping, no tokenizer embedding. Those remain in their
   respective consumers (zerfoo, zonnx).

3. **Both zerfoo and zonnx import `ztensor/gguf`** for writing. zerfoo also keeps
   its own `model/gguf` package for reading (parser, loader, config extraction,
   tensor name mapping) since those are zerfoo-specific concerns.

4. **Consolidate all five hand-rolled writers** in zerfoo to use the shared
   `ztensor/gguf.Writer`. Delete duplicated `writeGGUFString`, `writeGGUF`,
   `saveModelGGUF` functions.

5. **Implement `SaveModel` in `training/adapter.go`** using the shared writer,
   closing the "not implemented" gap.

6. **Update zonnx** to import `ztensor/gguf` instead of its own `pkg/gguf/writer.go`,
   then delete zonnx's writer. zonnx retains `pkg/gguf/metadata.go` and
   `pkg/gguf/tensornames.go` (ONNX-to-GGUF mapping logic specific to zonnx).

## Consequences

**Positive:**
- Single GGUF writer implementation shared across all three consumers (ztensor,
  zerfoo, zonnx).
- `SaveModel` becomes functional, enabling training checkpoint round-trips.
- zonnx drops ~240 lines of writer code and gains bug fixes for free.
- Future GGUF features (v4, new metadata types, mmap-friendly layout) are
  implemented once.
- Dependency direction is clean: ztensor (foundation) -> zerfoo, zonnx (consumers).

**Negative:**
- ztensor gains a new top-level package (`gguf/`). This is a minor scope increase
  but justified because GGUF is the sole model format and serialization is a
  foundational concern.
- Coordinated release across three repos (ztensor first, then zerfoo + zonnx).
- zonnx must update its `go.mod` to import a newer ztensor version.
