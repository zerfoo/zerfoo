# Quality Gates

## Test Coverage (2026-03-05)

Minimum threshold: 75% (enforced by coverage-gate CI)

### 100% Coverage
- config, data, device, internal/xblas, layers/components, layers/registry,
  layers/tokenizers, metrics, shutdown

### 98-99%
- compute (98.0%), distributed/coordinator (98.3%), features (99.0%),
  model/hrm (98.1%), numeric (98.5%), tensor (97.9%),
  testing/testutils (99.3%), tests/internal/testutil (98.5%)

### 96-97%
- graph (97.3%), inference (96.3%), layers/activations (97.4%),
  layers/attention (96.5%), layers/recurrent (97.0%),
  layers/regularization (97.6%), layers/transformer (96.4%),
  layers/transpose (97.6%), log (97.7%), metrics/runtime (96.5%),
  pkg/tokenizer (96.2%), serve (96.4%), training/loss (97.4%),
  training/optimizer (96.6%)

### 95-96%
- distributed (95.8%), generate (95.0%), layers/core (95.9%),
  layers/hrm (95.5%), layers/normalization (95.7%),
  layers/reducesum (95.9%), model (95.1%), training (95.9%)

### 90-94%
- cmd/cli (93.6%), cmd/bench-compare (89.7%), health (90.0%),
  layers/embeddings (92.9%), layers/features (93.8%),
  layers/gather (93.5%), layers/sequence (94.0%), registry (93.2%)

### Below 90%
- cmd/coverage-gate (84.9%), cmd/zerfoo-predict (76.6%), cmd/zerfoo-tokenize (74.1%)

### Known Untestable Gaps
- health: EngineCheck takes concrete *CPUEngine type, preventing mock testing
- Most remaining gaps are tensor.New unreachable error paths or engine error
  paths that require mock infrastructure
- layers/attention: dupl linter blocks MLA Forward engine error test
- cmd/zerfoo-predict: main(), runNewCLI (requires cli framework), os.Exit paths
- cmd/zerfoo-tokenize: main() and os.Exit paths

## Performance Benchmarks (2026-03-13, DGX Spark GB10)

### Inference Throughput — Gemma 3 1B Q4_K_M, 256 tokens, greedy decoding

| Configuration | tok/s | % of Ollama | Notes |
|---------------|-------|-------------|-------|
| Ollama (baseline) | 197.21 | 100% | Measured 2026-03-12 |
| Zerfoo (pre-optimization) | 8.61 | 4.4% | Initial GPU Q4 |
| Zerfoo (arena allocator) | 80.35 | 40.7% | Eliminated cudaMalloc |
| Zerfoo (all fused kernels) | 188.92 | 95.8% | Best achieved 2026-03-12 |
| Zerfoo (post Wave 1-8) | 166.02 | 84.2% | Current with clean defaults |
| Zerfoo (managed memory) | 145.33 | 73.7% | 13% regression from page faults |
| Zerfoo (CUDA graph attempt) | 99.51 | 50.5% | Graph capture fails, D2H in GQA |
| Theoretical max (Q4 on GB10) | ~350-400 | -- | 273 GB/s bandwidth ceiling |

### Performance Gap Analysis

Gap from Ollama: 31 tok/s (15.8%). Root causes:
1. **CUDA graph disabled**: ~338 kernel launches × ~7µs each = ~2.37ms/token overhead
2. **Wave 1-8 code changes**: int64 gather, Q4_K dispatch checks, SubSlice changes
3. **No managed memory**: Page fault overhead prevents unified memory exploitation

### Path to Surpassing Ollama
1. Enable CUDA graph capture (eliminate remaining D2H in GQA/FFN/KV cache)
2. Investigate 188→166 tok/s regression from Wave 1-8 changes
3. Apply kernel register tuning (T209.1) and sm_121 shared memory optimizations

### Output Quality

| Criterion | Zerfoo | Ollama |
|-----------|--------|--------|
| Coherence | Coherent English, Zen philosophy topic | Well-structured conversational |
| Relevance | Relevant — mentions mindfulness | Fully relevant, structured answer |
| Format | Raw completion (no chat template) | Chat template applied |

Both outputs are coherent. Difference is chat template, not model quality.

### Correctness

| Check | Status |
|-------|--------|
| Output coherent English | PASS |
| No gibberish/random tokens | PASS (fixed from earlier sessions) |
| Greedy decoding deterministic | PASS (same output across runs) |
| go build ./... (no build tags) | PASS |
| go vet ./... | PASS (pre-existing purego warnings only) |
| Test suite (DGX Spark) | PARTIAL (pre-existing race in TestBatchGenerate) |

### CUDA Graph Benchmark (2026-03-13)

| Mode | Run 1 | Run 2 | Run 3 | Average |
|------|-------|-------|-------|---------|
| Per-op (baseline) | 183.16 | 183.94 | 184.27 | 183.79 |
| CUDA graph enabled | 183.69 | 184.50 | 184.95 | 184.38 |

Graph capture fails (D2H in GQA). Falls back to per-op. No speedup.

### CUDA Graph Correctness (2026-03-13)

| Mode | Output (50 tokens, temp=0) | Match |
|------|---------------------------|-------|
| Per-op | "not to be to be to be..." | -- |
| CUDA graph | "not to be to be to be..." | IDENTICAL |

### FP16 Inference (2026-03-13)

| Precision | Status | tok/s |
|-----------|--------|-------|
| FP32 | Working | 183.79 avg |
| FP16 | SIGSEGV (null kernel ptr) | N/A |
| BF16 | Not implemented | N/A |

FP16 path crashes due to null function pointer in FP32-to-FP16 conversion
kernel (purego symbol lookup failure). Needs debugging.

### FP8/FP16 Inference Update (2026-03-13)

| Precision | Status | tok/s |
|-----------|--------|-------|
| FP32 | Working | 122.08 |
| FP16 | GQA storage mismatch | N/A |
| FP8 | GQA storage mismatch | N/A |

Root cause of prior SIGSEGV: stale `libkernels.so` in project root missing
FP16 conversion symbols. Fixed by updating the .so. Both FP16 and FP8 now
fail with a GQA tensor storage length mismatch (`storage length (1536) does
not match tensor size (6144)`), indicating a bug in the GQA FP16 compute path.

## Linting

- golangci-lint v2 with project .golangci.yml
- go vet: clean
- gosec: G704/G705 excluded (taint analysis false positives)

## Build

- `go build ./...`: clean
- `-race` flag enabled for all tests
