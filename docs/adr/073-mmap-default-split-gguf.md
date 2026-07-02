# ADR-073: mmap Default Loading and Split-GGUF Support

**Status:** Accepted
**Date:** 2026-03-28

## Context

Zerfoo's GGUF loader had two limitations:

1. **mmap was opt-in** — heap loading was the default. Users had to pass `WithMmap(true)` or `bench_tps --mmap` to get memory-mapped loading.
2. **No split-GGUF support** — any model packaged as multiple shards (the llama.cpp `-NNNNN-of-NNNNN.gguf` convention) would fail to load. This affects every model larger than ~50 GB, since HuggingFace enforces a ~50 GB per-file limit.

These limitations prevented loading MiniMax-M2 Q4_K_M (229B, 138 GB across 3 shards) on the DGX Spark (128 GB RAM), which was the primary motivating test.

## Decision

### 1. Make mmap the default

Change `loadOptions` in `LoadFile` and `Load` from `mmap: false` to `mmap: true`. `WithMmap(false)` opts out explicitly.

**Rationale:**
- Near-instant startup at all model sizes (mmap returns in microseconds; the OS pages lazily).
- Tensor data stays off the Go heap — zero GC scanning overhead, lower GC pause frequency.
- Models larger than physical RAM work transparently via OS NVMe paging.
- The only downside (first-touch page faults on first inference) is imperceptible in practice: a few milliseconds on a cached SSD for a 1B model.
- The heap path retains one advantage: it eagerly re-quantizes K-quant tensors to Q4_0 for the fused GEMV kernel. The mmap path uses lazy dequantization instead. Both paths produce coherent output; the heap re-quantization is a throughput optimization, not a correctness requirement.

Heap loading (`WithMmap(false)`) remains available and is the better choice when CUDA graph capture is needed for maximum throughput on models that fit in RAM.

### 2. Add split-GGUF loading

Add `model/gguf/split_file.go` implementing:
- `ParseSplit(path)` — detects the `-NNNNN-of-NNNNN.gguf` naming pattern, enumerates all sibling shards, parses each, and returns a merged `SplitFile` with all tensor metadata and a shard index.
- `LoadTensorsMmapSplit(sf, mappedShards)` — creates mmap-backed tensors referencing the correct shard's mapped region. Each shard is independently mmap'd.
- `LoadTensorsSplit(sf, readers)` — heap-loading variant for completeness.

`LoadGGUF` and `LoadGGUFMmap` in `inference/gguf.go` call `ParseSplit` first; if it returns nil (non-split file), the existing single-file path runs unchanged. Callers see no API change.

**Naming convention detected:**
```
Model-Q4_K_M-00001-of-00003.gguf  →  shard 0 (metadata + tensors)
Model-Q4_K_M-00002-of-00003.gguf  →  shard 1 (tensors only)
Model-Q4_K_M-00003-of-00003.gguf  →  shard 2 (tensors only)
```
The pattern is: last `-NNNNN-of-NNNNN` segment in the filename, where N is a zero-padded integer.

## Consequences

- **All GGUF models** now load via mmap by default with no configuration.
- **Split GGUF models** (any 70B+ model from HuggingFace) load transparently.
- **Models larger than RAM** work via OS NVMe paging — proven with MiniMax-M2 Q4_K_M (138 GB on 128 GB DGX Spark).
- **CUDA graph capture** still requires heap loading (`WithMmap(false)`). Mmap + CUDA graph is a future optimization (MM-T8, cudaHostRegister).
- `bench_tps --no-mmap` replaces `--mmap` since mmap is now the default.
