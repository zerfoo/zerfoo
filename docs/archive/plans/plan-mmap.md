# Plan: mmap-Based GGUF Loading

## Problem

Zerfoo currently loads all tensor weights into Go heap memory at startup via
`model/gguf/loader.go:LoadTensors`. For each tensor, it:

1. Seeks to the tensor's offset in the GGUF file
2. Reads raw bytes into a `[]byte` heap allocation (`make([]byte, dataSize)`)
3. Decodes/dequantizes into a Go-managed storage object (Q4Storage, Q8Storage, etc.)
4. For K-quants (Q4_K, Q5_K, Q6_K), additionally dequantizes to float32 and re-quantizes to Q4_0

This approach has three scaling problems:

- **Startup time**: Mistral 7B takes ~65 seconds to load ~8GB of weights. A 72B model
  would take ~8 minutes for ~42GB.
- **Memory pressure**: All weights live on the Go heap simultaneously. GC pauses grow
  with heap size. A 72B Q4_K_M model needs ~42GB of heap just for weights.
- **Model size ceiling**: Models larger than available RAM cannot load at all, even if
  only a fraction of weights are active at any time (e.g., MoE models). DGX Spark can
  handle ~200B parameters at Q4 in its 128GB unified memory, but needs mmap for 400B+
  models like Llama 3.1 405B (~230GB at Q4_K_M) which exceed physical RAM.

## Solution: mmap + Lazy Decode

Memory-map the GGUF file and keep tensor data in mmap'd pages. The OS kernel
manages paging from disk to RAM on demand. Combined with lazy dequantization
(decode quantized blocks only when needed for computation), this gives:

- **Near-instant startup**: `mmap()` returns in microseconds regardless of file size
- **OS-managed memory**: The kernel pages in/out as needed; no GC pressure
- **Models larger than RAM**: With NVMe backing, the OS transparently pages from SSD

DGX Spark has 128GB unified LPDDR5x + NVMe SSD, making it ideal for mmap-based
loading of large models. The 128GB fits ~200B Q4 parameters in RAM, but 400B+
models require mmap to page from NVMe on demand.

## Test Models

### Primary: Qwen 2.5 72B Instruct Q4_K_M (~42GB GGUF)

- Source: [bartowski/Qwen2.5-72B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF) (single-file Q4_K_M variant)
- Architecture: `qwen2` (already supported in `inference/load_gguf.go:buildArchGraph`)
- Parameters: 72.7B (80 layers, 64 heads, 8192 hidden, GQA with 8 KV heads)
- Why this model:
  - Fits in 128GB RAM but at ~42GB creates significant mmap overhead vs heap pressure
  - Tests the mmap vs heap tradeoff: heap loading would need ~42GB and ~8 min
  - Qwen 2 architecture is production-supported with attention bias and RoPE theta=1M
  - Practically useful: strong multilingual, coding, and reasoning performance
  - Single-file GGUF available from bartowski (no split-file handling needed initially)

Download command:
```bash
huggingface-cli download bartowski/Qwen2.5-72B-Instruct-GGUF \
  --include "Qwen2.5-72B-Instruct-Q4_K_M.gguf" \
  --local-dir /models/qwen2.5-72b/
```

### Stretch: Llama 3.1 405B Instruct Q4_K_M (~230GB GGUF)

- Source: [bartowski/Meta-Llama-3.1-405B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-405B-Instruct-GGUF)
- Architecture: `llama` (production-supported)
- Parameters: 405B (126 layers, 128 heads, 16384 hidden, GQA with 8 KV heads)
- Why this model:
  - Does NOT fit in 128GB RAM -- requires mmap to page from NVMe
  - At NVMe Gen4 (~7 GB/s sequential read), expect ~3 tok/s decode throughput
  - This is slow but PROVES the capability: models larger than RAM work at all
  - Validates the OS paging behavior under memory pressure
  - Split-file GGUF (multiple shards); requires split-mmap support or single-file merge

Download command:
```bash
huggingface-cli download bartowski/Meta-Llama-3.1-405B-Instruct-GGUF \
  --include "Meta-Llama-3.1-405B-Instruct-Q4_K_M/*.gguf" \
  --local-dir /models/llama-3.1-405b/
```

## Architecture

### Current Flow (heap)

```
os.Open(path)
  -> gguf.Parse(f)           # reads header + tensor metadata
  -> gguf.LoadTensors(gf, f) # for each tensor: Seek -> ReadFull -> decode -> heap alloc
  -> map GGUF names -> canonical names
  -> re-quantize K-quants to Q4_0
  -> build computation graph
  -> upload weights to GPU
```

### Proposed Flow (mmap)

```
os.Open(path)
  -> gguf.Parse(f)                    # reads header + tensor metadata (unchanged)
  -> syscall.Mmap(fd, 0, fileSize)    # map entire file, returns []byte backed by kernel pages
  -> gguf.LoadTensorsMmap(gf, mapped) # for each tensor: slice into mmap region, wrap in MmapStorage
  -> map GGUF names -> canonical names
  -> on GPU upload: cudaHostRegister(mmap region) for fast DMA, or memcpy per layer
  -> lazy dequantize: decode quantized blocks on first access or during forward pass
```

### Key Types

```go
// In ztensor/tensor package
type MmapStorage struct {
    data       []byte          // slice into mmap'd region (NOT a copy)
    numElems   int
    ggmlType   int             // original GGML type for lazy decode
    shape      []int
    decoded    []float32       // lazily populated on first Dequantize() call
    decodeMu   sync.Once
}

func (s *MmapStorage) Dequantize(dst []float32) { ... }  // decode on demand
func (s *MmapStorage) RawBytes() []byte { return s.data } // for GPU DMA
```

---

## Tasks

### Wave 1: mmap Infrastructure (ztensor + zerfoo)

- [x] MM-T1 Add MmapStorage type to ztensor/tensor  Est: 4h  2026 03 26
  repo: ztensor (v0.8.0)
  Implemented MmapStorage wrapping mmap'd byte slices with lazy dequantization.
  Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, BF16, F32, Q4_K, Q5_K, Q6_K.

- [x] MM-T2 Add mmap file helper to ztensor  Est: 2h  2026 03 26
  repo: ztensor (v0.8.0)
  MmapFile() in tensor package using syscall.Mmap on Linux/Darwin.

- [x] MM-T3 Add LoadTensorsMmap to model/gguf  Est: 3h  2026 03 26
  repo: zerfoo (v1.26.0)
  LoadTensorsMmap creates MmapStorage per tensor from mmap'd region. Zero-copy.
  Fixed missing Q4_1/Q5_0/Q5_1 type mappings in mapGGMLType.

- [x] MM-T4 Add WithMmap option to inference.LoadFile  Est: 2h  2026 03 26
  repo: zerfoo (v1.26.0)
  WithMmap() option wired. bench_tps -mmap flag loads and runs on DGX.
  Note: Output quality degraded vs heap path (mmap keeps K-quant tensors,
  heap re-quantizes to Q4_0). Wave 2 (MM-T5/T6) addresses this.

### Wave 2: Lazy Dequantization

- [x] MM-T5 Implement lazy decode for quantized MmapStorage  Est: 4h  2026 03 27
  repo: ztensor (v0.9.1)
  MmapStorage.Slice() uses sync.Once for lazy dequant. Fixed Q4_K/Q5_K/Q6_K
  dequantizers (delegated to reference DequantizeQ4K/Q5K/Q6K). Coherent output
  confirmed on Gemma 3 1B Q4_K_M (CPU 3.68 tok/s, GPU 25.5 tok/s).

- [x] MM-T6 Skip re-quantization for mmap'd K-quants  Est: 2h  2026 03 27
  repo: ztensor (v0.9.0-v0.9.6) + zerfoo (v1.26.2)
  GPU dispatch added: matMulMmap/matMulMmapB route MmapStorage through
  quantized GEMV/GEMM kernels. LMHead uses MatMulTransposeB for MmapStorage.
  Graph compiler skips all quantized storage (isQuantizedStorage helper).
  UploadWeights skips MmapStorage (per-op dequant+upload instead).
  Coherent output confirmed on Gemma 3 1B Q4_K_M at 25.5 tok/s GPU.
  CUDA graph capture fails (per-op H2D copies during capture), limiting
  throughput vs heap+graph (167 tok/s). Pre-uploading raw quantized bytes
  causes misaligned address on ARM64 — future optimization to investigate.

- [x] MM-T7 Implement madvise hints for sequential/random access  Est: 2h  2026 03 27
  repo: ztensor (v0.10.0-v0.10.1) + zerfoo
  Added MadviseSequential/Random/WillNeed/DontNeed to ztensor tensor pkg.
  MmapFile no longer sets MADV_SEQUENTIAL (caused 7x load time regression
  from 8s to 55s on GB10). LoadGGUFMmap calls MadviseRandom after loading
  to optimize for inference's random layer access pattern. MadviseWillNeed
  and MadviseDontNeed available for future layer prefetch optimization.
  Windows stubs are no-ops.

### Wave 3: GPU Integration

- [ ] MM-T8 cudaHostRegister for mmap'd pages  Est: 3h
  repo: ztensor
  Deps: MM-T1, MM-T2
  When GPU engine detects MmapStorage, call `cudaHostRegister` on the mmap'd
  region to enable fast DMA transfers. This pins the pages in physical memory
  and enables GPU to DMA directly from the mmap'd file without a bounce buffer.
  Acceptance: GPU upload of mmap'd tensors is >= 90% speed of heap-allocated upload.

- [ ] MM-T9 Layer-at-a-time GPU transfer with prefetch  Est: 4h
  repo: zerfoo
  Deps: MM-T8
  Instead of uploading all weights to GPU VRAM at once (which may exceed VRAM
  for 32B models), transfer one transformer layer at a time. While layer N
  runs on GPU, async-copy layer N+1's weights. Double-buffer with two CUDA
  streams.
  Acceptance: Qwen 72B runs inference on DGX Spark with < 8GB VRAM usage.

- [ ] MM-T10 Unified memory fallback for large models  Est: 3h
  repo: ztensor
  Deps: MM-T9
  On DGX Spark (Grace Hopper / GB10 unified memory), detect unified memory
  support and use `cudaMallocManaged` to let the GPU access mmap'd pages
  directly via unified addressing. Skip explicit transfers entirely.
  Acceptance: Qwen 72B inference works with zero explicit GPU memcpy calls.

### Wave 4: Validation and Benchmarking

- [x] MM-T4b Make mmap the default loading strategy  Est: 1h  2026-03-28
  repo: zerfoo (v1.36.0)
  Changed loadOptions to mmap: true in both LoadFile and Load. WithMmap(false)
  to opt out. bench_tps --no-mmap flag replaces --mmap. All existing tests pass;
  test suite now runs via mmap path by default (confirmed by "mmap'd GGUF file"
  in test logs).

- [x] MM-T4c Add split-GGUF (multi-shard) loading support  Est: 4h  2026-03-28
  repo: zerfoo (v1.36.0)
  Added model/gguf/split_file.go: ParseSplit(), LoadTensorsMmapSplit(),
  LoadTensorsSplit(). Auto-detected in LoadGGUF/LoadGGUFMmap — callers see no
  API change. 7 tests: discover shards, missing shard error, merged tensor map,
  mmap split, heap split, non-split passthrough. Required for any 70B+ model
  with split GGUF files (MiniMax-M2, Llama 3.1 405B, etc).

- [ ] MM-T11 Download MiniMax-M2 Q4_K_M to DGX  Est: 1h
  [IN PROGRESS 2026-03-28] Downloading unsloth/MiniMax-M2-GGUF Q4_K_M (3 shards,
  138 GB total) to ~/models/minimax-m2-q4km/ on DGX Spark. ~109 GB downloaded.
  Acceptance: All 3 shards present, hf_download.log contains DOWNLOAD_COMPLETE.

- [ ] MM-T12 End-to-end mmap inference test (MiniMax-M2, 138 GB on 128 GB)  Est: 2h
  Deps: MM-T4c, MM-T11
  Run bench_tps against MiniMax-M2 Q4_K_M on DGX Spark. Model exceeds physical
  RAM (138 GB > 128 GB) — proves over-RAM inference via NVMe paging.
  Acceptance: Coherent output. Load time < 30s. Document tok/s and RSS.

- [ ] MM-T13 Ollama head-to-head: MiniMax-M2 Q4_K_M  Est: 1h
  Deps: MM-T12
  Import same 3-shard GGUF into Ollama via Modelfile. Run same prompt, same
  token count. Record tok/s for both. Ollama also uses mmap (llama.cpp); the
  comparison is pure Go vs C++ on identical hardware and model.
  Acceptance: Both produce coherent output. Record relative tok/s in README.

---

## Dependency Graph

```
MM-T1 (MmapStorage) ──┬── MM-T3 (LoadTensorsMmap) ── MM-T4 (WithMmap option)
                       │                                      │
MM-T2 (mmap helper) ──┤                               MM-T12 (e2e test) ── MM-T13 (benchmark)
                       │                                      │
                       ├── MM-T5 (lazy decode) ── MM-T6 (skip re-quant)    MM-T14 (405B stress)
                       │
                       ├── MM-T7 (madvise)
                       │
                       ├── MM-T8 (cudaHostRegister) ── MM-T9 (layer prefetch)
                       │                                      │
                       └───────────────────────────── MM-T10 (unified memory)

MM-T11 (download model) ── MM-T12
```

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Split GGUF files (Qwen official repo splits into 5 parts) | Medium | Use bartowski single-file variant. Add split-mmap support later. |
| mmap not available on all platforms (Windows, WASM) | Low | Fallback to heap loading by default. mmap is opt-in. |
| Page faults during inference cause latency spikes | Medium | madvise WILLNEED prefetch; cudaHostRegister pins pages; layer prefetch hides latency. |
| Re-quantization removal (MM-T6) changes output quality | Low | Q4_K native is higher quality than Q4_0 re-quant. Net positive. Blocked by KQ-T2 kernel optimization for throughput parity. |
| Go GC interacts poorly with mmap'd memory | Low | mmap'd pages are outside Go heap. GC does not scan them. This is a benefit, not a risk. |

## ADR

This plan will produce ADR-068: mmap-Based Model Loading, documenting the decision
to support memory-mapped GGUF files as an alternative to heap loading.
