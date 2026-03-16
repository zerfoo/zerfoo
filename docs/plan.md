# Zerfoo Phase 19: "Ship-Ready" -- Complete Phase 1 + Developer Experience

## 1. Context

### Problem Statement

Zerfoo has strong inference performance (234 tok/s, 18.8% faster than Ollama) and
production infrastructure (OpenAI API, training, distributed), but Phase 1
(Inference Excellence) has critical gaps that block credible adoption:

- Only 2 of 6 claimed architectures have GGUF graph builders (Llama, Gemma).
  Mistral, Qwen, Phi, and DeepSeek have config parsers and chat templates but
  return "unsupported architecture" when loaded as GGUF.
- FP16 and FP8 inference are broken (GQA tensor storage mismatch).
- CUDA graph capture does not deliver speedup (D2H transfer in GQA prevents
  graph closure).
- ztensor (v0.1.0) and ztoken (v0.1.0) are under-released despite substantial work.

Phase 19 closes these Phase 1 gaps and layers in the highest-impact Phase 2
(Developer Experience) items: one-line inference API, HuggingFace model download,
structured output, and tool calling.

### Objectives

- O1: All 6 architectures produce correct GGUF inference on DGX Spark.
- O2: FP16 and FP8 inference pass end-to-end on at least Gemma 3 and Llama 3.
- O3: CUDA graph capture delivers measurable decode speedup (target 20%+).
- O4: `zerfoo pull` downloads GGUF models from HuggingFace (ADR 039).
- O5: One-line inference API: `zerfoo.Load("model") -> model.Chat("prompt")`.
- O6: Structured output via grammar-guided decoding (ADR 038).
- O7: ztensor v0.2.0 and ztoken v0.2.0 released.

### Non-Goals

- Pre-training at scale.
- Continuous batching (deferred to Phase 20 -- requires PagedAttention).
- Prefill/decode split (deferred to Phase 20).
- Quantization improvements (GPTQ, AWQ, Q5_K/Q6_K native GEMV -- deferred).
- LoRA fine-tuning (Phase 3 per VISION.md).

### Constraints and Assumptions

- DGX Spark: 128 GB unified memory, CUDA capable. Available at ssh ndungu@192.168.86.250.
- DeepSeek V3 (671B) does not fit on DGX Spark. Use DeepSeek-V2-Lite (16B) as the
  test model -- same MLA + MoE architecture, Q8_0 is ~17 GB.
- All code is pure Go, zero CGo. GPU via purego/dlopen.
- GGUF is the sole model format (ADR 037).
- Go standard library only -- no cobra, viper, testify, etc.

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Working GGUF architectures | 2 (Llama, Gemma) | 6 | `buildArchGraph` handles all 6 |
| FP16 inference | Broken | Passing | DGX Spark end-to-end test |
| FP8 inference | Broken | Passing | DGX Spark end-to-end test |
| CUDA graph decode speedup | 0% (fallback) | 20%+ | Benchmark vs non-graph baseline |
| Lines to first inference | ~40 | <10 | `zerfoo.Load` + `model.Chat` |
| ztensor version | v0.1.0 | v0.2.0 | git tag |
| ztoken version | v0.1.0 | v0.2.0 | git tag |

---

## 2. Scope and Deliverables

### In Scope

- GGUF graph builders for Mistral, Qwen, Phi, DeepSeek architectures.
- FP16/FP8 GQA tensor storage mismatch fix.
- CUDA graph D2H transfer elimination in GQA.
- TestBatchGenerate race condition fix.
- ztensor v0.2.0 and ztoken v0.2.0 releases.
- HuggingFace model download CLI and library integration (ADR 039).
- One-line inference API (Load, Chat, Generate, Embed).
- Structured output via grammar-guided decoding (ADR 038).
- Tool/function calling in chat API.
- API stability audit of public interfaces.

### Out of Scope

- Continuous batching / PagedAttention.
- Prefill/decode phase split.
- GPTQ/AWQ/Q5_K/Q6_K native quantization.
- LoRA adapters.
- Model hub registry (beyond HuggingFace download).
- Multimodal inference (vision-language).

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|-------------------|
| D1 | 6 working GGUF architectures | All 6 produce coherent text on DGX Spark |
| D2 | FP16/FP8 inference | Both pass Gemma 3 and Llama 3 end-to-end |
| D3 | CUDA graph speedup | 20%+ decode speedup measured on DGX |
| D4 | ztensor v0.2.0, ztoken v0.2.0 | Git tags, release-please PRs merged |
| D5 | `zerfoo pull` CLI | Downloads GGUF from HuggingFace, caches locally |
| D6 | One-line API | `zerfoo.Load` + `model.Chat` in <10 lines |
| D7 | Structured output | JSON schema-constrained generation works |
| D8 | Tool calling | Function calling via chat completions API |

---

## 3. Checkable Work Breakdown

### E1: Stability -- Fix Broken Inference Paths

- [x] T1.1 Fix FP16 inference -- GQA tensor storage mismatch  Owner: Claude  Done: 2026-03-16
  - Fix: decodeF16Tensor preserves Float16Storage; transposeWeight handles FP16/FP8; NewFloat16StorageFromRaw added to ztensor. DGX verification still needed.

- [ ] T1.2 Fix FP8 inference -- same GQA root cause  Owner: TBD  Est: 2h
  - Deps: T1.1 (shares root cause)
  - AC: FP8 Gemma 3 1B produces coherent text on DGX Spark.

- [x] T1.3 Fix CUDA graph capture -- eliminate D2H transfer in GQA decode  Owner: Claude  Done: 2026-03-16
  - Fix: FullBufferProvider interface + FlashAttentionDecode path eliminates D2H. DGX 20%+ benchmark still needed.

- [x] T1.4 Fix TestBatchGenerate race condition  Owner: Claude  Done: 2026-03-16
  - Fix: sync.Mutex on Generator serializes Generate/GenerateStream. TestGenerate_ConcurrentSafety regression test. go test -race -count=10 passes.

- [x] T1.5 Release ztensor v0.2.0  Owner: Claude  Done: 2026-03-16
  - Tag SHA: c24092c67728b2ef23ba1dcc7fd3daa042e79381. go build/test/vet all clean.

- [x] T1.6 Release ztoken v0.2.0  Owner: Claude  Done: 2026-03-16
  - CHANGELOG created. Tag v0.2.0 created locally. Human must push: cd ztoken && git push origin main && git push origin v0.2.0

- [ ] T1.7 Update zerfoo go.mod to use ztensor v0.2.0 and ztoken v0.2.0  Owner: TBD  Est: 30m
  - Deps: T1.5, T1.6
  - AC: `go mod tidy && go build ./... && go test ./...` passes.

- [ ] T1.8 Run go vet and linter across zerfoo, ztensor, ztoken  Owner: TBD  Est: 30m
  - Deps: T1.7
  - AC: Zero new warnings. Existing purego warnings documented.

### E2: Model Coverage -- GGUF Graph Builders

Each architecture has an existing config parser and chat template. The work is
implementing `buildXxxGraph` functions that construct computation graphs from
GGUF tensor maps, and wiring them into `buildArchGraph` in `load_gguf.go`.

Reference: `buildTransformerGraph` in `arch_common.go` is the shared builder
that Llama and Gemma use. Mistral, Qwen, and Phi are dense transformers that
can extend it via `transformerGraphOpts`. DeepSeek requires MLA + MoE layers
(both already implemented in `layers/attention/` and `layers/core/`).

- [x] T2.1 Implement buildMistralGraph -- sliding window attention  Owner: Claude  Done: 2026-03-16
  - arch_mistral.go + BuildCausalSlidingWindowMask + 5 tests. DGX verification needed.

- [x] T2.2 Implement buildQwenGraph -- attention bias, RoPE theta=1M  Owner: Claude  Done: 2026-03-16
  - arch_qwen.go + attnBias in transformerGraphOpts + 6 tests. DGX verification needed.
  - Implementation: Extend `transformerGraphOpts` with `attnBias bool`. Load `blk.N.attn_q.bias`, `blk.N.attn_k.bias`, `blk.N.attn_v.bias` when present. Wire "qwen2" case.
  - Test: Unit test for bias application. Integration test loading Qwen 2.5 GGUF.

- [x] T2.3 Implement buildPhiGraph -- partial rotary factor  Owner: Claude  Done: 2026-03-16
  - arch_phi.go with partialRotaryFactor in transformerGraphOpts. Merged in Wave 1 (missed in plan update).

- [x] T2.4 Implement buildDeepSeekGraph -- MLA + MoE  Owner: Claude  Done: 2026-03-16
  - arch_deepseek.go + MLA/MoE fields in gguf.ModelConfig + tests. DGX verification needed.

- [x] T2.5 Wire all new architectures into buildArchGraph  Owner: Claude  Done: 2026-03-16
  - All cases added in Wave 1: mistral, qwen2, phi3/phi, deepseek_v3/deepseek2 all wired.

- [ ] T2.6 End-to-end DGX verification for all 6 architectures  Owner: TBD  Est: 4h
  - Deps: T2.5
  - AC: All 6 architectures produce coherent multi-sentence output. Benchmark throughput recorded for each.
  - Models: Gemma 3 1B Q4_K_M, Llama 3 8B Q4_K_M, Mistral 7B Q4_K_M, Qwen 2.5 7B Q4_K_M, Phi 3 mini Q4_K_M, DeepSeek-V2-Lite Q8_0.

- [ ] T2.7 Run go vet and linter on inference package  Owner: TBD  Est: 30m
  - Deps: T2.5
  - AC: Zero new warnings in inference/.

### E3: HuggingFace Model Download (ADR 039)

Decision rationale: docs/adr/039-huggingface-model-download.md

- [x] T3.1 Implement HuggingFace HTTP API client  Owner: Claude  Done: 2026-03-16
  - model/huggingface/ package. NewClient/GetModel/ListGGUFFiles/ResolveGGUF. 11 unit tests + integration tests (//go:build integration).

- [x] T3.2 Implement download with resume and progress  Owner: Claude  Done: 2026-03-16
  - Branch feat/hf-download-resume. Downloader with Range resume, SHA256, progress callback. 7 tests, -race clean.

- [ ] T3.3 Implement cache manifest and management  Owner: TBD  Est: 2h
  - Deps: T3.2
  - AC: JSON manifest tracks cached models. `List()` returns cached models with sizes. `Remove()` deletes model and updates manifest.
  - Test: Unit tests for manifest CRUD operations.

- [ ] T3.4 Implement `zerfoo pull` CLI command  Owner: TBD  Est: 2h
  - Deps: T3.2, T3.3
  - AC: `zerfoo pull google/gemma-3-4b` downloads default Q4_K_M. `--quant Q8_0` selects quant. `zerfoo list` shows cached models. `zerfoo rm` removes.
  - Package: cmd/
  - Test: Integration test for pull/list/rm cycle.

- [ ] T3.5 Integrate cache into zerfoo.Load()  Owner: TBD  Est: 2h
  - Deps: T3.3, T4.1
  - AC: `zerfoo.Load("google/gemma-3-4b")` checks cache, downloads if missing. `zerfoo.Load("/path/to/file.gguf")` still works. Path detection: starts with "/" or ".".
  - Test: Unit test for path vs model-name detection. Integration test for cache hit/miss.

- [ ] T3.6 Run go vet and linter on model/huggingface/ and cmd/  Owner: TBD  Est: 30m
  - Deps: T3.4
  - AC: Zero warnings.

### E4: Developer Experience -- High-Level API

- [x] T4.1 Implement zerfoo.Load() high-level model loader  Owner: Claude  Done: 2026-03-16
  - api.go: Load/Chat/Generate/Embed/Close + GenerateOption (WithGenMaxTokens/WithGenTemperature/WithGenTopP). 5 tests. HF stub returns error.
  - AC: `zerfoo.Load("/path/to/model.gguf")` returns a `*zerfoo.Model` with Chat, Generate, Embed methods. Detects architecture from GGUF metadata.
  - Package: top-level zerfoo/ package
  - Test: Unit test with a small test GGUF fixture.

- [x] T4.2 Implement Model.Chat() and Model.Generate()  Owner: Claude  Done: 2026-03-16
  - Implemented in api.go alongside T4.1. Chat, Generate, GenerateResult, WithGenMaxTokens/Temperature/TopP all present.

- [ ] T4.3 Implement Model.Embed()  Owner: TBD  Est: 3h
  - Deps: T4.1
  - AC: `model.Embed([]string{"hello", "world"})` returns `[]Embedding` where Embedding has `Vector []float32` and `CosineSimilarity(other)` method.
  - Test: Unit test verifying embedding shape and cosine similarity computation.

- [ ] T4.4 Implement Model.ChatStream() for streaming  Owner: TBD  Est: 2h
  - Deps: T4.2
  - AC: `model.ChatStream(ctx, "prompt")` returns a channel or iterator yielding token strings as they are generated.
  - Test: Unit test verifying streaming yields tokens incrementally.

- [ ] T4.5 Run go vet and linter on top-level package  Owner: TBD  Est: 30m
  - Deps: T4.4
  - AC: Zero warnings.

### E5: Structured Output -- Grammar-Guided Decoding (ADR 038)

Decision rationale: docs/adr/038-structured-output-grammar-guided-decoding.md

- [ ] T5.1 Implement JSON Schema to CFG converter  Owner: TBD  Est: 6h
  - Deps: none
  - AC: Converts JSON Schema (object, array, string, number, integer, boolean, null, enum, const, required, nested) to a context-free grammar state machine. Rejects unsupported features ($ref, oneOf, anyOf, pattern) with clear error.
  - Package: generate/grammar/
  - Test: Unit tests for each JSON Schema type. Edge cases: empty schema, deeply nested, enum with special chars.

- [ ] T5.2 Implement token mask computation from CFG state  Owner: TBD  Est: 6h
  - Deps: T5.1
  - AC: Given a CFG state and tokenizer vocabulary, produces a `[]bool` mask of valid next tokens. Mask is correct: sampling only masked tokens produces valid JSON at every prefix.
  - Implementation: Build byte-trie from vocabulary. At each step, intersect CFG valid-byte-set with trie to determine valid tokens.
  - Test: Unit tests: given a schema and partial output, verify mask allows valid continuations and blocks invalid ones.

- [ ] T5.3 Integrate grammar engine into generation pipeline  Owner: TBD  Est: 4h
  - Deps: T5.2, T4.2
  - AC: `model.Generate(ctx, prompt, zerfoo.WithSchema(schema))` produces guaranteed-valid JSON. Token mask applied before sampling at each decode step. Grammar engine runs on CPU in parallel with GPU forward pass.
  - Test: Integration test: generate JSON matching a person schema (name + age), parse result with encoding/json.

- [ ] T5.4 Add response_format support to OpenAI API server  Owner: TBD  Est: 3h
  - Deps: T5.3
  - AC: POST /v1/chat/completions with `response_format.type = "json_schema"` produces schema-constrained output. Matches OpenAI API spec for json_schema response format.
  - Test: Integration test via HTTP client.

- [ ] T5.5 Run go vet and linter on generate/grammar/  Owner: TBD  Est: 30m
  - Deps: T5.4
  - AC: Zero warnings.

### E6: Tool/Function Calling

- [ ] T6.1 Implement tool definition parsing in chat API  Owner: TBD  Est: 3h
  - Deps: none
  - AC: Chat completions accept `tools` array with function definitions (name, description, parameters as JSON Schema). Stored in request context.
  - Test: Unit test for tool definition parsing and validation.

- [ ] T6.2 Implement tool call detection and response formatting  Owner: TBD  Est: 4h
  - Deps: T6.1, T5.3
  - AC: When model output matches tool call pattern, response includes `tool_calls` array with function name and arguments as JSON. Uses grammar-guided decoding to ensure arguments match the tool's parameter schema.
  - Test: Unit test with mock model output. Integration test with real model on DGX.

- [ ] T6.3 Add tool calling to OpenAI API server  Owner: TBD  Est: 3h
  - Deps: T6.2
  - AC: POST /v1/chat/completions with `tools` parameter produces tool_calls in response. Supports `tool_choice: "auto"` and `tool_choice: {"type": "function", "function": {"name": "..."}}`.
  - Test: Integration test via HTTP client.

- [ ] T6.4 Add tool calling to high-level library API  Owner: TBD  Est: 2h
  - Deps: T6.2, T4.2
  - AC: `model.Chat(prompt, zerfoo.WithTools(tools...))` returns result with ToolCalls field.
  - Test: Unit test with tool definitions.

- [ ] T6.5 Run go vet and linter on tool calling code  Owner: TBD  Est: 30m
  - Deps: T6.4
  - AC: Zero warnings.

### E7: API Stability Audit

- [ ] T7.1 Audit public API surface of zerfoo package  Owner: TBD  Est: 3h
  - Deps: T4.5, T5.5, T6.5
  - AC: Every exported type, function, and method in the zerfoo top-level package is documented with godoc. Types marked as stable or experimental. No unexported fields that should be exported, no exported fields that should be private.

- [ ] T7.2 Audit public API surface of ztensor package  Owner: TBD  Est: 2h
  - Deps: T1.5
  - AC: Same criteria as T7.1 for ztensor.

- [ ] T7.3 Audit public API surface of ztoken package  Owner: TBD  Est: 1h
  - Deps: T1.6
  - AC: Same criteria as T7.1 for ztoken.

---

## 4. Parallel Work

### Tracks

| Track | Epics | Description |
|-------|-------|-------------|
| A: Stability | E1 | Fix FP16, FP8, CUDA graph, race condition |
| B: Mistral | T2.1 | Mistral graph builder |
| C: Qwen | T2.2 | Qwen graph builder |
| D: Phi | T2.3 | Phi graph builder |
| E: DeepSeek | T2.4 | DeepSeek MLA+MoE graph builder |
| F: HF Download | E3 (T3.1-T3.4) | HuggingFace API client and CLI |
| G: High-Level API | E4 (T4.1-T4.4) | Load, Chat, Generate, Embed |
| H: Grammar | E5 (T5.1-T5.2) | JSON Schema CFG and token masking |
| I: Tool Calling | E6 (T6.1) | Tool definition parsing |
| J: ztoken release | T1.6 | Independent release |

Sync points:
- After Wave 2: T2.1-T2.4 merge into T2.5 (wire into buildArchGraph).
- After T4.2 + T5.2: T5.3 integrates grammar into generation.
- After T5.3: T6.2 uses grammar for tool argument validation.
- After all epics: T7.1-T7.3 audit APIs.

### Maximum Parallelism

**Wave 1** (10 tasks, no dependencies):
T1.1, T1.3, T1.4, T1.6, T2.1, T2.2, T2.3, T2.4, T3.1, T4.1

**Wave 2** (10 tasks, after Wave 1):
T1.2 (needs T1.1), T1.5 (needs T1.3), T2.5 (needs T2.1-T2.4), T3.2 (needs T3.1), T4.2 (needs T4.1), T4.3 (needs T4.1), T5.1, T6.1, T4.4 (needs T4.2 but can stub), T1.7 (needs T1.5, T1.6)

**Wave 3** (9 tasks, after Wave 2):
T1.8 (needs T1.7), T2.6 (needs T2.5), T2.7 (needs T2.5), T3.3 (needs T3.2), T3.4 (needs T3.2, T3.3), T5.2 (needs T5.1), T6.2 (needs T6.1, T5.3 stubbed), T3.6 (needs T3.4), T4.5 (needs T4.4)

**Wave 4** (8 tasks, after Wave 3):
T3.5 (needs T3.3, T4.1), T5.3 (needs T5.2, T4.2), T5.4 (needs T5.3), T6.3 (needs T6.2), T6.4 (needs T6.2, T4.2), T5.5 (needs T5.4), T6.5 (needs T6.4), T7.2 (needs T1.5)

**Wave 5** (3 tasks, final):
T7.1 (needs T4.5, T5.5, T6.5), T7.3 (needs T1.6)

---

## 5. Timeline and Milestones

| Milestone | Deps | Exit Criteria |
|-----------|------|---------------|
| M1: Stability | T1.1-T1.8 | FP16, FP8, CUDA graph all pass on DGX. ztensor/ztoken released. |
| M2: Full Model Coverage | T2.1-T2.7 | All 6 architectures produce coherent GGUF inference on DGX. |
| M3: Model Download | T3.1-T3.6 | `zerfoo pull` downloads and caches GGUF from HuggingFace. |
| M4: Developer API | T4.1-T4.5, T5.1-T5.5, T6.1-T6.5 | One-line API, structured output, tool calling all working. |
| M5: Ship-Ready | T7.1-T7.3 | API audit complete. All deliverables D1-D8 met. |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | FP16/FP8 GQA root cause spans ztensor+zerfoo boundary | High | Medium | Investigate ztensor tensor allocation first. Both repos in same workspace. |
| R2 | CUDA graph D2H elimination requires GQA restructuring | High | Medium | Profile first. May need GPU-resident position counter (see ADR 032). |
| R3 | DeepSeek MLA+MoE graph builder surfaces bugs in MLA/MoE layers | Medium | Medium | Layers have unit tests. Run layer-level tests before integration. |
| R4 | Grammar-guided decoding token mask is slow for large vocabularies | Medium | Low | Token mask is O(vocab) on CPU, runs in parallel with GPU forward. Profile on 128K vocab models. |
| R5 | HuggingFace API changes or rate limits | Low | Low | Pin to documented API endpoints. Cache aggressively. Retry with backoff. |
| R6 | DeepSeek-V2-Lite GGUF not available or broken | Medium | Low | Multiple providers on HuggingFace (mradermacher, bartowski). Verify before starting T2.4. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. Code compiles: `go build ./...` passes.
2. Tests pass: `go test ./...` passes (including new tests).
3. Race detector clean: `go test -race ./...` passes for modified packages.
4. Linter clean: `go vet ./...` and golangci-lint report zero new warnings.
5. For DGX verification tasks: end-to-end output reviewed and throughput recorded.

### Review and QA

- Every implementation task has a paired test subtask or inline AC requiring tests.
- Run linters after every code change (tasks T1.8, T2.7, T3.6, T4.5, T5.5, T6.5).
- DGX verification (T2.6) is the integration gate for model coverage.
- Never commit files from different repo directories in the same commit.
- Make many small logical commits.

---

## 8. Progress Log

### Change Summary -- 2026-03-15

New plan created for Phase 19: "Ship-Ready". Replaces completed Phase 18.

Phase 18 (Developer Adoption Campaign) was 100% complete (37/37 tasks). All stable
knowledge from Phase 18 was previously preserved in docs/design.md, docs/adr/, and
docs/devlog.md during Phase 18 execution.

New ADRs created:
- docs/adr/038-structured-output-grammar-guided-decoding.md -- Grammar-guided decoding for JSON schema-constrained generation.
- docs/adr/039-huggingface-model-download.md -- HuggingFace model download via zerfoo pull.

Key decisions:
- DeepSeek V3 (671B) does not fit on DGX Spark (128 GB). Using DeepSeek-V2-Lite (16B)
  as the MLA+MoE test model -- same architecture, Q8_0 is ~17 GB.
- HuggingFace API integration approved by founder.
- Structured output / grammar-guided decoding approved by founder.
- Continuous batching, prefill/decode split, and quantization improvements deferred to Phase 20.

---

## 9. Hand-off Notes

- **DGX Spark**: ssh ndungu@192.168.86.250. 128 GB unified memory. CUDA capable.
- **Benchmark baseline**: Gemma 3 1B Q4_K_M at 234.30 tok/s (Phase 11 measurement).
- **GGUF graph builder pattern**: See `inference/arch_common.go` (`buildTransformerGraph`) and `inference/arch_llama.go` / `inference/arch_gemma.go` for the two existing implementations. New builders extend `transformerGraphOpts` or create new builder functions for architectures that differ significantly (DeepSeek).
- **Config parsers**: All 6 architectures already have config parsers in `inference/arch_config.go` with full test coverage.
- **MLA and MoE layers**: Already implemented in `layers/attention/multi_head_latent_attention.go` and `layers/core/moe.go` with tests.
- **Chat templates**: All 6 architectures have chat template formatters.
- **DeepSeek test model**: DeepSeek-V2-Lite Q8_0 (~17 GB) from mradermacher/DeepSeek-V2-Lite-GGUF on HuggingFace.
- **FP16/FP8 bug**: GQA tensor storage mismatch -- storage length 1536 vs tensor size 6144. Documented in QUALITY.md (2026-03-05).
- **CUDA graph bug**: D2H transfer in GQA prevents graph closure. Falls back to per-op execution. See ADR 032 (GPU-resident position counter) for related work.
- **Phase 18 outputs**: 4 blog posts, 4 examples, getting-started guide, GPU setup guide, CONTRIBUTING.md in all repos, issue templates in all repos, distribution drafts. All merged to main.

---

## 10. Appendix

### Architecture Differences from Llama Baseline

| Architecture | Key Differences from Llama | Complexity |
|-------------|---------------------------|------------|
| Mistral | Sliding window attention mask | Low -- add window size to transformerGraphOpts |
| Qwen 2 | Attention bias, RoPE theta=1M | Low -- add bias loading, theta already configurable |
| Phi 3/4 | Partial rotary factor (0.5) | Low -- split head dims, apply RoPE to subset |
| DeepSeek V2 | MLA replaces MHA, MoE replaces FFN | High -- new graph builder, uses existing MLA/MoE layers |

### DeepSeek-V2-Lite Architecture Details

- 16B total parameters, 2.4B active per token (MoE)
- 27 layers, hidden dim 2048, 16 attention heads
- MLA: compressed KV (latent dim), low-rank projections for Q/K/V
- MoE: 2 shared experts + 64 routed experts per layer, 6 activated per token
- GGUF Q8_0: ~17 GB (fits easily in 128 GB DGX Spark)

### JSON Schema Subset for Structured Output (ADR 038)

Supported: object, array, string, number, integer, boolean, null, enum, const,
required, minLength/maxLength, minimum/maximum, nested objects/arrays.

Not supported (deferred): $ref, oneOf, anyOf, allOf, pattern, additionalProperties.
