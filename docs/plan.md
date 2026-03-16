# Zerfoo Phase 20: "Throughput & Release" — Quantization, Batching, DGX, v0.2.0

## 1. Context

### Problem Statement

Phase 19 completed all six GGUF architecture builders, one-line API, HuggingFace download,
structured output, tool calling, and API audits. Three categories of gaps remain:

1. **Quantization accuracy loss**: Q5_K_M and Q6_K weights are silently re-quantized down to
   Q4_0 in `model/gguf/loader.go:170,182` before GEMV. This reduces output quality and breaks
   model parity for users who request these quants explicitly.

2. **Throughput ceiling**: `BatchScheduler` in `serve/batch.go` fans out concurrent requests
   into sequential single-sequence `Generate` calls. `PagedKVCache` in `generate/paged_kv.go`
   exists but is not wired into the `Generator`. True concurrent-sequence batching requires
   multi-sequence decode via paged KV.

3. **DGX verification debt**: T2.6 (all 6 architectures E2E), T1.1/T1.2 DGX FP16/FP8, and
   T1.3 CUDA graph speedup were coded in Phase 19 but never verified on DGX Spark because the
   machine was offline. These remain open until DGX is accessible.

4. **Release gap**: ztensor v0.2.0 and ztoken v0.2.0 shipped in Phase 19. zerfoo itself has
   no published release. Users cannot `go get github.com/zerfoo/zerfoo@v0.2.0`.

### Objectives

- O1: Q5_K_M and Q6_K use native dequant float32 GEMV — no lossy re-quantization.
- O2: Concurrent requests use multi-sequence decode via PagedKV (target: 2× concurrent
  throughput vs serial fan-out at batch size 4).
- O3: DGX E2E verification — all 6 architectures produce coherent text, FP16/FP8 pass,
  CUDA graph delivers 20%+ decode speedup vs non-graph baseline.
- O4: zerfoo v0.2.0 published — CHANGELOG, README, go-release CI, git tag.
- O5: Three runnable example applications: chatbot, RAG, structured output.

### Non-Goals

- LoRA / QLoRA fine-tuning (Phase 3 per VISION.md).
- GPTQ / AWQ quantization formats.
- Metal backend (macOS GPU).
- Prefill/decode phase split (needs multi-node coordination — Phase 21).
- Multimodal (vision-language) inference.

### Constraints

- DGX Spark: 128 GB unified memory, CUDA. Available at ssh ndungu@192.168.86.250.
  E3 tasks (DGX verification) are gated on DGX access. All other tasks run locally.
- Pure Go, zero CGo. GPU via purego/dlopen.
- Go standard library only — no cobra, viper, testify.
- GGUF is the sole model format (ADR-037).

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Q5_K_M accuracy | Lossy (re-quant to Q4_0) | Exact (native dequant) | Perplexity delta < 0.1 vs reference |
| Q6_K accuracy | Lossy (re-quant to Q4_0) | Exact (native dequant) | Perplexity delta < 0.1 vs reference |
| Concurrent throughput (batch=4) | ~234 tok/s (serial) | 300+ tok/s | DGX benchmark |
| DGX architectures verified | 0 (all offline) | 6/6 | E2E coherent text on DGX |
| zerfoo release | None (untagged) | v0.2.0 | `go get github.com/zerfoo/zerfoo@v0.2.0` |
| Example applications | 2 (inference, embedding) | 5 (+ chatbot, rag, json-output) | Run `go run examples/*/main.go` |

---

## 2. Scope and Deliverables

### In Scope

- Native Q5_K and Q6_K dequant GEMV (no Q4_0 re-quantization).
- Multi-sequence decode via PagedKVCache wired into Generator.
- BatchScheduler wired to multi-sequence Generator for concurrent requests.
- DGX E2E verification: 6 architectures, FP16, FP8, CUDA graph speedup.
- zerfoo v0.2.0: CHANGELOG, go-release CI, git tag, README.
- Three new example apps: chatbot CLI, RAG search, structured JSON output.

### Out of Scope

- LoRA / fine-tuning, GPTQ / AWQ, Metal backend, prefill/decode split, multimodal.

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|-------------------|
| D1 | Q5_K/Q6_K native GEMV | Loader no longer re-quantizes; perplexity within 0.1 of reference |
| D2 | Multi-sequence batched decode | PagedKV wired into Generator; 2× throughput at batch=4 on DGX |
| D3 | DGX E2E verification | All 6 architectures coherent, FP16/FP8 pass, CUDA graph 20%+ speedup |
| D4 | zerfoo v0.2.0 release | Tag published, go-release CI green, README updated |
| D5 | Example apps | 3 new runnable examples in examples/ |

---

## 3. Checkable Work Breakdown

### E1: Quantization — Native Q5_K and Q6_K GEMV

Remove the lossy Q4_0 re-quantization fallback added for NEON GEMV and implement
proper float32 dequantization GEMV for both quant types.

Reference: `model/gguf/loader.go:160-185` is where Q5_K and Q6_K currently
re-quantize to Q4_0. `layers/gemv/` contains the GEMV implementations.
The fix: dequant Q5_K/Q6_K blocks to float32 directly in the GEMV path
(or add a Q5_K/Q6_K-native dequant path in `layers/gemv/quantized.go`).

- [x] T1.1 Implement native Q5_K dequant GEMV  Owner: Claude  Done: 2026-03-16
  - Acceptance: `loader.go` no longer calls re-quantize-to-Q4_0 for Q5_K.
    `go test ./layers/gemv/ -run Q5` passes. `go test ./model/gguf/ -run Q5` passes.
    Q5_K_M weights produce identical outputs to reference Python decode on a 16-token prompt.

- [x] T1.2 Implement native Q6_K dequant GEMV  Owner: Claude  Done: 2026-03-16
  - Acceptance: Same as T1.1 but for Q6_K. Both quants tested via table-driven tests.
    `loader.go` Q6_K branch calls native path, not re-quantize.

- [ ] T1.3 go vet/lint clean after E1 changes  Owner:  Done: (deps: T1.1✅ T1.2✅)
  - Acceptance: `go vet ./...` 0 warnings; `golangci-lint run ./...` 0 issues.
    All existing tests pass.

### E2: Batching — Multi-Sequence Decode via PagedKV

Wire `PagedKVCache` (generate/paged_kv.go) into the `Generator` for true
multi-sequence decode. Update `BatchScheduler` (serve/batch.go) to use the
multi-sequence Generator path instead of sequential fan-out.

The Generator currently handles a single sequence. The paged KV API supports
multiple sequences via per-sequence block tables. The work:
1. Add `GenerateBatch(ctx, prompts []string, opts) ([]string, error)` to
   `inference/inference.go` backed by multi-sequence PagedKV decode.
2. Update `BatchHandler` in `serve/batch.go` to call `GenerateBatch` when
   `model.supportsBatch` is true.
3. Integration test: 4 concurrent chat requests in `serve/batch_test.go`.

- [ ] T2.1 Add GenerateBatch to inference.Model via PagedKV  Owner:  Done:
  - Acceptance: `inference.Model.GenerateBatch(ctx, []string, opts) ([]string, error)` added.
    Uses `PagedKVCache` for shared KV across sequences. Unit tests in `inference/batch_test.go`
    cover 1, 2, 4 concurrent sequences. `go test ./inference/ -run Batch -race` passes.

- [ ] T2.2 Wire serve.BatchScheduler to GenerateBatch  Owner:  Done: (deps: T2.1✅)
  - Acceptance: `Server.handleChat` uses `GenerateBatch` when batch size > 1.
    `serve/batch_test.go` integration test: 4 concurrent `/v1/chat/completions` requests
    all complete and return correct responses. No data races.

- [ ] T2.3 go vet/lint clean on generate/ and serve/ after E2 changes  Owner:  Done: (deps: T2.2✅)
  - Acceptance: `go vet ./generate/... ./serve/...` 0 warnings;
    `golangci-lint run ./generate/... ./serve/...` 0 issues.
    Full test suite passes.

### E3: DGX Verification — Architecture, Precision, CUDA Graph

All tasks in E3 are BLOCKED until DGX Spark (ssh ndungu@192.168.86.250) is accessible.

Preflight for each DGX task:
```bash
git pull && GONOSUMDB='github.com/zerfoo/*' GONOPROXY='github.com/zerfoo/*' go build ./...
```

- [ ] T3.1 DGX E2E — all 6 architectures produce coherent text  Owner:  Done: (BLOCKED: DGX offline)
  - Architectures: Llama 3, Gemma 3, Mistral, Qwen 2, Phi 3/4, DeepSeek-V2-Lite.
  - Acceptance: Each model loads GGUF, generates ≥20 coherent tokens on a standard
    prompt. No panics, no "unsupported architecture" errors.
    Record tok/s, model size, quant in docs/benchmarks.md.

- [ ] T3.2 DGX FP16 inference E2E  Owner:  Done: (BLOCKED: DGX offline, deps: T3.1✅)
  - Acceptance: Gemma 3 and Llama 3 FP16 GGUF files load and generate without error.
    Output matches F32 output within float16 precision tolerance.

- [ ] T3.3 DGX FP8 inference E2E  Owner:  Done: (BLOCKED: DGX offline, deps: T3.1✅)
  - Acceptance: Same as T3.2 but for FP8 E4M3FN quants.

- [ ] T3.4 DGX CUDA graph decode speedup ≥ 20%  Owner:  Done: (BLOCKED: DGX offline, deps: T3.1✅)
  - Acceptance: Benchmark `generate/bench_decode_test.go -bench=BenchmarkDecode` with
    and without `WithCUDAGraph`. Graph path must be ≥ 20% faster. Record in benchmarks.md.

- [ ] T3.5 DGX throughput benchmark after batching  Owner:  Done: (BLOCKED: DGX offline, deps: T2.2✅ T3.1✅)
  - Acceptance: `serve/loadtest_test.go` with 4 concurrent clients. Measure tok/s.
    Target ≥ 300 tok/s at batch=4. Record result in benchmarks.md.

### E4: Example Applications

Add three runnable examples to the `examples/` directory. Each must be a
standalone Go program (`go run examples/<name>/main.go`).

- [x] T4.1 Chatbot CLI example  Owner: Claude  Done: 2026-03-16
  - Acceptance: `examples/chat/main.go` implements a readline loop that calls
    `model.Chat(prompt)` and prints the response. Works with any GGUF model path
    passed as `--model` flag. `README.md` in the directory explains usage.
    `go build ./examples/chat/` compiles. `go vet ./examples/chat/` clean.

- [x] T4.2 RAG demo example  Owner: Claude  Done: 2026-03-16
  - Acceptance: `examples/rag/main.go` embeds a small document corpus via
    `model.Embed(texts)`, stores vectors in memory, accepts a query, finds the
    top-3 most similar documents via `CosineSimilarity`, and passes them as context
    to `model.Chat`. Self-contained — corpus is hardcoded (5 short facts).
    `go build ./examples/rag/` compiles. `go vet ./examples/rag/` clean.

- [x] T4.3 Structured JSON output example  Owner: Claude  Done: 2026-03-16
  - Acceptance: `examples/json-output/main.go` calls `model.Generate` with
    `WithSchema(schema)` where the schema describes `{name: string, age: number}`.
    Prints the constrained JSON output. `go build ./examples/json-output/` compiles.
    `go vet ./examples/json-output/` clean.

### E5: Release — zerfoo v0.2.0

- [x] T5.1 Set up go-release CI pipeline  Owner: Claude  Done: 2026-03-16
  - Acceptance: `.github/workflows/release-please.yml` exists in zerfoo repo.
    `release-please` config targets Go library release type.
    `.github/workflows/ci.yml` runs `go test ./... -race` on every PR.
    `go vet ./...` 0 warnings.

- [ ] T5.2 Write CHANGELOG for v0.2.0  Owner:  Done: (deps: T5.1✅)
  - Acceptance: `CHANGELOG.md` created with v0.2.0 section covering:
    Phase 19 + Phase 20 deliverables grouped by category (Features, Bug Fixes,
    API, Performance). Follows Keep a Changelog format.

- [ ] T5.3 Update README for v0.2.0  Owner:  Done: (deps: T5.2✅)
  - Acceptance: `README.md` top section shows one-line inference example
    (`zerfoo.Load` + `model.Chat`), HuggingFace download, structured output,
    and tool calling snippets. Badges: CI, Go version, License. No broken links.

- [ ] T5.4 Tag and publish zerfoo v0.2.0  Owner:  Done: (deps: T5.1✅ T5.2✅ T5.3✅)
  - Acceptance: `git tag v0.2.0 && git push origin v0.2.0`. release-please PR
    auto-created (or manually created if CI not yet live). `go get
    github.com/zerfoo/zerfoo@v0.2.0` resolves successfully (may require GONOSUMDB
    since repo is private until public launch).

---

## 4. Dependency Graph

```
T1.1 ──┐
T1.2 ──┴── T1.3

T2.1 ──── T2.2 ──── T2.3

T3.1 (BLOCKED) ──┬── T3.2
                 ├── T3.3
                 ├── T3.4
                 └── T3.5 (also deps T2.2)

T4.1, T4.2, T4.3  (no deps — fully independent)

T5.1 ──── T5.2 ──── T5.3 ──── T5.4
```

Wave 1 (unblocked now): T1.1, T1.2, T4.1, T4.2, T4.3, T5.1
Wave 2 (after Wave 1): T1.3 (deps T1.1+T1.2), T2.1, T5.2 (deps T5.1)
Wave 3 (after Wave 2): T2.2 (deps T2.1), T5.3 (deps T5.2)
Wave 4 (after Wave 3): T2.3 (deps T2.2), T5.4 (deps T5.1+T5.2+T5.3)
Wave 5 (DGX-gated): T3.1–T3.5

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| DGX remains offline | Medium | High (blocks E3) | E1/E2/E4/E5 proceed; E3 scheduled separately |
| PagedKV multi-seq decode has correctness bugs | Medium | High (blocks T2.2) | Extensive unit tests in T2.1; compare with sequential baseline |
| Q5_K/Q6_K native dequant slower than Q4_0 GEMV | Low | Low | Acceptable — accuracy takes priority; optimize in Phase 21 |
| release-please CI setup takes longer than expected | Low | Low | Manual CHANGELOG + tag if automation stalls |
