# Zerfoo Phase 22: "GGUF Compatibility & Structured Output"

## 1. Context

### Problem Statement

Phase 21 ("Community & Polish") completed 36/37 tasks: doc.go for all 8 public packages,
CLI UX (--help, version, progress bars, --system), DeepSeek V3 promoted to Production,
high-level API tests, documentation polish, and DGX verification (T7.1-T7.5). The sole
remaining task (T7.6 DeepSeek V3 DGX E2E) is blocked on model availability.

DGX verification exposed three GGUF loader gaps that prevent Zerfoo from loading
real-world GGUF files from community sources (bartowski, TheBloke, Qwen, Microsoft):

1. **Qwen tokenizer garbled output**: Qwen 2.5 GGUF loads correctly (arch=qwen2, 24
   layers, vocab=151936) but generates garbled BPE bytes. Root cause: GGUF tokenizer
   extraction (`model/gguf/tokenizer.go` line 69) hardcodes `byteLevelBPE=false`. Qwen
   uses GPT-2 style byte-level BPE where token strings contain raw bytes mapped via a
   byte-to-unicode table. The ztoken library already supports byte-level BPE
   (`NewBPETokenizer` accepts a `byteLevelBPE` bool), but the GGUF extractor never
   enables it. Fix: detect `tokenizer.ggml.model == "gpt2"` and pass `byteLevelBPE=true`.

2. **Phi merged QKV tensor not mapped**: Phi 3.5 mini GGUF uses `blk.N.attn_qkv.weight`
   (merged Q/K/V in a single tensor) instead of separate `attn_q`, `attn_k`, `attn_v`.
   The `tensorNameMap` in `model/gguf/arch.go` has no mapping for `attn_qkv`. The loader
   must detect merged QKV tensors, split them into three separate tensors along the
   output dimension, and map them to the canonical names. The split dimensions depend on
   num_heads and num_kv_heads (GQA vs MHA).

3. **Mistral architecture detection**: bartowski's Mistral 7B GGUF reports
   `general.architecture=llama`, so `buildArchGraph` routes to `buildLlamaGraph` instead
   of `buildMistralGraph`. The Mistral-specific sliding window attention is never applied.
   Fix: when architecture is "llama" and `SlidingWindow > 0` in GGUF metadata, route to
   `buildMistralGraph` which already sets `slidingWindowSize` in its graph options.

Additionally, two P1/P2 priorities remain:

4. **Structured output / JSON mode**: The grammar engine exists (`generate/grammar/`)
   with Grammar state machine, TokenMask, and JSON Schema converter. The server
   (`serve/server.go`) wires `response_format.type = "json_schema"` to grammar-constrained
   decoding. The library API has `WithGrammar()` and `WithSchema()`. What is missing is
   end-to-end integration testing, a CLI `--json-schema` flag, and a structured output
   example that verifies the full pipeline from schema to guaranteed-valid JSON output.

5. **Concurrent inference throughput**: DGX verification showed 84.49 tok/s with 4
   concurrent clients (mutex-serialized). The `Generator` holds a `sync.Mutex` that
   serializes all `Generate`/`GenerateStream` calls. The graph is shared and read-only
   after compilation, but the KV cache and position state are per-request. The fix is to
   introduce per-request session state while sharing the compiled graph.

### Research Findings

**Technical landscape (tech-researcher):**
- ztoken already implements byte-level BPE with `buildByteEncoderDecoder()` and
  `byteLevelPreTokenize()`. The GGUF extractor just needs to set `byteLevelBPE=true`
  when `tokenizer.ggml.model == "gpt2"`.
- Phi `attn_qkv.weight` has shape `[(num_heads + 2*num_kv_heads) * head_dim, hidden_size]`
  for GQA models. Split into Q `[num_heads*head_dim, hidden_size]`, K and V each
  `[num_kv_heads*head_dim, hidden_size]`.
- GGUF metadata `attention.sliding_window` is already parsed into `ModelConfig.SlidingWindow`
  in `arch.go:105-106`. Detection: `arch == "llama" && cfg.SlidingWindow > 0` routes to
  `buildMistralGraph`.
- Grammar engine is complete: Grammar, TokenMask, JSONSchema converter, objectNode,
  arrayNode, stringNode, numberNode, integerNode, anyJSONNode. WithGrammar wired in
  generator.go (line 302-304). WithSchema exists in api.go (line 443).
- Generator mutex at `generator.go:95`. Graph is shared (read-only). KV cache is per-request.
  Need per-request Session struct holding KV cache + position + sampling state.

**Risks and pitfalls (risk-researcher):**
- Qwen tokenizer: enabling `byteLevelBPE` for "gpt2" model type must NOT break
  SentencePiece models. The flag is orthogonal to sentencePiece flag. Safe.
- Phi QKV split: incorrect split for GQA (num_heads != num_kv_heads) will corrupt
  attention. Must compute Q/K/V sizes from `num_heads`, `num_kv_heads`, `head_dim`.
  Phi-3 mini uses MHA (32 heads, 32 KV heads), Phi-3 small uses GQA (32 heads, 8 KV heads).
- Mistral detection: Llama 3.1 has `sliding_window` in its config but Gemma 3 also has
  it via `SlidingWindowPattern`. Check: only reroute if `SlidingWindow > 0 AND
  SlidingWindowPattern == 0` (Gemma sets SlidingWindowPattern=6; pure Mistral does not).
- Grammar TokenMask is O(vocab_size * avg_token_length) per decode step. For Qwen
  (151936 tokens), worst case ~1ms on modern CPU. Negligible vs GPU forward pass (~5ms).
- Concurrent inference: removing the global mutex requires per-request KV cache allocation.
  The graph `Forward()` must be stateless (it already is -- parameters are constant after
  compilation). Risk: memory pressure from multiple concurrent KV caches.

**Architecture patterns (arch-researcher):**
- Tokenizer fix: one-line change in `ExtractTokenizer` to check `tokenizer.ggml.model`
  and pass `byteLevelBPE=true`. Defensive: also check for "gpt2" token patterns.
- Merged QKV: add `attn_qkv.weight` to `tensorNameMap` is insufficient -- need a split
  step. Best pattern: handle in the GGUF loader (`LoadGGUF`) after tensor name mapping.
  Add a `splitMergedTensors()` pass that detects `attn_qkv` and splits using config.
  This keeps arch builders unchanged.
- Structured output: WithSchema in api.go already exists. Missing: (a) end-to-end test
  with real grammar-constrained generation, (b) CLI --json-schema flag for `run` command,
  (c) structured output example. The server already handles json_schema response_format.
- Concurrent inference: introduce `InferenceSession` struct with per-request KV cache
  and position. `Generator.NewSession()` creates a session. `Session.Generate()` uses
  the shared graph but private KV state. The global mutex moves to a per-session basis.

### Objectives

- O1: All 6 GGUF architectures load from community GGUF sources (bartowski, TheBloke,
  Qwen, Microsoft) and produce valid output. Currently 3/6 pass.
- O2: Structured output (JSON mode) works end-to-end: library API, CLI, server, example.
- O3: Concurrent inference throughput exceeds the current 84.49 tok/s with 4 clients.
- O4: DGX re-verification confirms all fixes produce coherent text on GPU hardware.

### Non-Goals

- LoRA / QLoRA fine-tuning (P5, 12-18 months).
- ROCm or OpenCL backend work (P4, 6-12 months).
- Metal backend (macOS GPU).
- Absorbed MLA KV cache compression (deferred).
- DeepSeek V3 multi-GPU expert parallelism.
- Full JSON Schema support ($ref, oneOf, anyOf, allOf, pattern).
- Continuous batching (vLLM-style). Phase 22 targets per-request isolation, not dynamic batching.

### Constraints

- Pure Go, zero CGo. GPU via purego/dlopen.
- Go standard library only -- no cobra, viper, testify.
- GGUF is the sole model format (ADR-037).
- DGX Spark: ssh ndungu@192.168.86.250. DGX verification tasks gated on DGX access.
- Each repo has its own git history. Do not cross-commit across repos.
- ztoken changes (if any) must be released as a separate version before zerfoo can use them.

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| GGUF arch load success | 3/6 (Gemma, Llama, Mistral) | 5/6 (+ Qwen, Phi) | DGX test suite |
| Qwen 2.5 output quality | Garbled | Coherent text | DGX test `TestT7_1/qwen2` |
| Phi 3.5 loads | FAIL (missing tensor) | PASS | DGX test `TestT7_1/phi3` |
| Mistral sliding window | Not activated | Activated when detected | Unit test |
| Structured output E2E | Server only | Library + CLI + Server + Example | Integration tests |
| Concurrent throughput | 84.49 tok/s (4 clients) | 200+ tok/s (4 clients) | DGX benchmark |
| DeepSeek V3 DGX E2E | BLOCKED | PASS (if model available) | DGX test |

---

## 2. Scope and Deliverables

### In Scope

- Fix Qwen byte-level BPE tokenizer in GGUF extractor.
- Add merged QKV tensor splitting for Phi models in GGUF loader.
- Add Mistral architecture detection from sliding window metadata.
- End-to-end structured output: CLI flag, library test, example.
- Per-request inference sessions to remove global mutex bottleneck.
- DGX re-verification of all fixes.
- Carry forward T7.6 (DeepSeek V3 DGX E2E, blocked on model).

### Out of Scope

- LoRA/fine-tuning, ROCm/OpenCL/Metal backends, continuous batching.
- Full JSON Schema ($ref, oneOf, anyOf, allOf, pattern).
- DeepSeek V3 backward pass.
- New model architecture support.
- ztoken library changes (byte-level BPE already supported).

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|-------------------|
| D1 | Qwen tokenizer fix | Qwen 2.5 GGUF produces coherent text on DGX |
| D2 | Phi merged QKV support | Phi 3.5 GGUF loads and generates on DGX |
| D3 | Mistral arch detection | Mistral routed to buildMistralGraph when SlidingWindow > 0 |
| D4 | Structured output E2E | CLI --json-schema, library test, server test, example |
| D5 | Concurrent inference | Per-request sessions, 200+ tok/s with 4 clients |
| D6 | DGX re-verification | All 5 non-DeepSeek architectures pass on DGX |
| D7 | DeepSeek V3 DGX E2E (carried) | PASS if model becomes available |

---

## 3. Checkable Work Breakdown

### E1: Qwen Byte-Level BPE Tokenizer Fix

Fix `ExtractTokenizer` in `model/gguf/tokenizer.go` to enable byte-level BPE for
Qwen-family models. The ztoken library already supports this mode.

- [x] T1.1 Enable byte-level BPE in GGUF tokenizer extraction  Owner: Claude  Done: 2026-03-16
  - File: `model/gguf/tokenizer.go`
  - Change: In `ExtractTokenizer`, check `tokenizer.ggml.model` metadata. When the
    value is `"gpt2"`, pass `byteLevelBPE=true` to `tokenizer.NewBPETokenizer` (line 69).
    Currently hardcoded to `false`.
  - Acceptance: `ExtractTokenizer` returns a BPETokenizer with byte-level BPE enabled
    for models with `tokenizer.ggml.model == "gpt2"`. Existing SentencePiece models
    (tokenizer.ggml.model == "llama") are unaffected.

- [x] T1.2 Add unit tests for byte-level BPE tokenizer extraction  Owner: Claude  Done: 2026-03-16
  - Deps: T1.1
  - File: `model/gguf/tokenizer_test.go`
  - Add: `TestExtractTokenizer_ByteLevelBPE` that creates a GGUF File with
    `tokenizer.ggml.model = "gpt2"` and byte-level BPE vocabulary. Verify Encode/Decode
    round-trip produces correct text. Add negative test that "llama" model type does NOT
    enable byte-level BPE.
  - Acceptance: `go test ./model/gguf/ -run TestExtractTokenizer_ByteLevel -race` passes.

- [x] T1.3 go vet/lint clean after Qwen tokenizer fix  Owner: Claude  Done: 2026-03-16
  - Deps: T1.1, T1.2
  - Acceptance: `go vet ./model/...` 0 warnings.

### E2: Phi Merged QKV Tensor Splitting

Add support for merged `attn_qkv.weight` tensors in the GGUF loader. Phi 3/3.5 GGUF
files pack Q, K, V projections into a single tensor that must be split before the
architecture graph builder can use them.

- [x] T2.1 Add attn_qkv tensor name mapping  Owner: Claude  Done: 2026-03-16
  - File: `model/gguf/arch.go`
  - Change: Add `"attn_qkv.weight"` to `tensorNameMap` mapping to a sentinel value
    (e.g., `"self_attn.qkv_proj.weight"`) that the loader can detect for splitting.
    Also add `"attn_qkv.bias"` mapping.
  - Acceptance: `MapTensorName("phi3", "blk.0.attn_qkv.weight")` returns
    `"model.layers.0.self_attn.qkv_proj.weight"`.

- [x] T2.2 Implement QKV tensor split in GGUF loader  Owner: Claude  Done: 2026-03-16
  - Deps: T2.1
  - File: `model/gguf/loader.go` (or new file `model/gguf/split.go`)
  - Add: `splitMergedQKV()` function called after tensor name mapping in `LoadGGUF`.
    For each tensor named `*.self_attn.qkv_proj.weight`, split into three tensors:
    - Q: `*.self_attn.q_proj.weight` shape `[num_heads * head_dim, hidden_size]`
    - K: `*.self_attn.k_proj.weight` shape `[num_kv_heads * head_dim, hidden_size]`
    - V: `*.self_attn.v_proj.weight` shape `[num_kv_heads * head_dim, hidden_size]`
    Compute sizes from `ModelConfig.NumQueryHeads`, `NumKeyValueHeads`, `HiddenSize`.
    Handle both MHA (num_heads == num_kv_heads) and GQA (num_heads != num_kv_heads).
    Also split bias tensors if present.
  - Acceptance: After splitting, the tensor map contains separate q_proj, k_proj, v_proj
    tensors with correct shapes. The original qkv_proj tensor is removed.

- [x] T2.3 Add unit tests for QKV split  Owner: Claude  Done: 2026-03-16
  - Deps: T2.2
  - Files: `model/gguf/split_test.go` or `model/gguf/loader_test.go`
  - Add: Tests for (a) MHA split (32 heads, 32 KV heads, head_dim=96, hidden_size=3072),
    (b) GQA split (32 heads, 8 KV heads, head_dim=96, hidden_size=3072),
    (c) missing config values produce error, (d) bias splitting.
  - Acceptance: `go test ./model/gguf/ -run TestSplitMergedQKV -race` passes.

- [x] T2.4 Add Phi arch builder support for split QKV  Owner: Claude  Done: 2026-03-16
  - Deps: T2.2
  - File: `inference/arch_phi.go`
  - Verify: `buildPhiGraph` already works with separate q_proj, k_proj, v_proj tensors
    (it delegates to `buildTransformerGraph`). If the split is correct, no changes needed.
    Add an integration test that builds a Phi graph from tensors containing split QKV.
  - Acceptance: `go test ./inference/ -run TestBuildPhiGraph -race` passes.

- [x] T2.5 go vet/lint clean after Phi QKV support  Owner: Claude  Done: 2026-03-16
  - Deps: T2.1-T2.4
  - Acceptance: `go vet ./model/... ./inference/...` 0 warnings.

### E3: Mistral Architecture Detection

Route Mistral-architecture models to `buildMistralGraph` when GGUF metadata reports
`general.architecture=llama` but `attention.sliding_window > 0`.

- [x] T3.1 Add Mistral detection in buildArchGraph  Owner: Claude  Done: 2026-03-16
  - File: `inference/load_gguf.go`
  - Change: In `buildArchGraph`, when `arch == "llama"` and `cfg.SlidingWindow > 0` and
    `cfg.SlidingWindowPattern == 0` (to exclude Gemma 3 which uses SlidingWindowPattern=6),
    route to `buildMistralGraph` instead of `buildLlamaGraph`.
  - Acceptance: A GGUF file with arch="llama" and sliding_window=4096 uses
    `buildMistralGraph`. A Gemma 3 file with SlidingWindowPattern=6 still uses
    `buildGemmaGraph`.

- [x] T3.2 Add unit tests for Mistral detection  Owner: Claude  Done: 2026-03-16
  - Deps: T3.1
  - File: `inference/load_gguf_test.go`
  - Add: Tests for (a) arch="llama" + sliding_window=4096 routes to Mistral,
    (b) arch="llama" + no sliding window stays Llama, (c) arch="gemma3" +
    sliding_window + pattern=6 stays Gemma.
  - Acceptance: `go test ./inference/ -run TestBuildArchGraph -race` passes.

- [x] T3.3 go vet/lint clean after Mistral detection  Owner: Claude  Done: 2026-03-16
  - Deps: T3.1, T3.2
  - Acceptance: `go vet ./inference/...` 0 warnings.

### E4: Structured Output End-to-End

Complete the structured output feature: the grammar engine and server integration exist,
but the CLI, library integration tests, and example need work.

- [x] T4.1 Add --json-schema flag to CLI run command  Owner: Claude  Done: 2026-03-16
  - File: `cmd/cli/run.go`
  - Change: Add `--json-schema <schema>` flag that accepts a JSON Schema string. When
    set, parse the schema, convert to grammar via `grammar.Convert`, and pass via
    `inference.WithGrammar` to the generate call. Output the raw JSON result without
    the interactive prompt.
  - Acceptance: `zerfoo run model.gguf --json-schema '{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}' --prompt "Extract the name from: John Smith"`
    produces valid JSON matching the schema.

- [x] T4.2 Add unit tests for CLI --json-schema flag  Owner: Claude  Done: 2026-03-16
  - Deps: T4.1
  - File: `cmd/cli/run_test.go`
  - Add: Tests for (a) --json-schema parses correctly, (b) invalid schema produces
    error, (c) --json-schema without --prompt uses positional arg as prompt.
  - Acceptance: `go test ./cmd/cli/ -run TestRunCommand_JSONSchema -race` passes.

- [x] T4.3 Add structured output integration test  Owner: Claude  Done: 2026-03-16
  - File: `inference/inference_test.go` or new `inference/structured_output_test.go`
  - Add: Integration test that loads a test GGUF fixture, generates with WithGrammar,
    and validates the output is parseable JSON conforming to the schema. Use the minimal
    test GGUF (writeTestGGUF) with a simple schema.
  - Acceptance: `go test ./inference/ -run TestStructuredOutput -race` passes.

- [x] T4.4 Add structured output example  Owner: Claude  Done: 2026-03-16
  - File: `examples/json-output/main.go` (already exists, verify completeness)
  - Verify: The existing json-output example uses `grammar.JSONSchema` and
    `grammar.Convert`. Ensure it compiles, has a README, and demonstrates both
    library-level and explanation of server-level structured output.
  - Acceptance: `go build ./examples/json-output/` succeeds. README.md explains usage.

- [x] T4.5 go vet/lint clean after structured output  Owner: Claude  Done: 2026-03-16
  - Deps: T4.1-T4.4
  - Acceptance: `go vet ./...` 0 warnings.

### E5: Concurrent Inference Sessions

Replace the global Generator mutex with per-request inference sessions. The compiled
graph is shared (read-only), but KV cache and position state are per-session.

- [x] T5.1 Design InferenceSession struct  Owner: Claude  Done: 2026-03-16
  - File: `generate/session.go` (new)
  - Add: `InferenceSession` struct holding per-request state: KV cache, position counter,
    sampling state. `Generator.NewSession()` creates a session that shares the compiled
    graph but has its own KV cache. `Session.Generate()` and `Session.GenerateStream()`
    methods use session-local state.
  - Acceptance: InferenceSession compiles and has method signatures for Generate and
    GenerateStream.

- [x] T5.2 Implement per-session KV cache allocation  Owner: Claude  Done: 2026-03-16
  - Deps: T5.1
  - File: `generate/session.go`, `generate/generator.go`
  - Change: Move KV cache creation from Generator construction to Session creation.
    Each session allocates its own KV cache based on the model config. The Generator
    keeps the graph, tokenizer, engine, and model config. The session gets the KV cache
    and position tracking.
  - Acceptance: Two sessions can exist simultaneously without data races.
    `go test ./generate/ -run TestSession -race` passes.

- [ ] T5.3 Wire sessions into Model.Generate and GenerateStream  Owner:  Est: 60m
  - Deps: T5.2
  - Files: `inference/inference.go`, `generate/generator.go`
  - Change: `Model.Generate` and `Model.GenerateStream` create a per-request session
    (or reuse from a pool). Remove the global `sync.Mutex` from Generator. Add a
    session pool with configurable max concurrent sessions.
  - Acceptance: 4 concurrent `Model.Generate` calls execute in parallel without races.
    `go test ./... -race -count=1` passes.

- [ ] T5.4 Add concurrent inference benchmark test  Owner:  Est: 45m
  - Deps: T5.3
  - File: `generate/session_test.go`
  - Add: Benchmark test running 4 concurrent sessions generating 50 tokens each.
    Verify no races and throughput > 1x single-session baseline.
  - Acceptance: `go test ./generate/ -run TestConcurrentSessions -race -count=3` passes.

- [ ] T5.5 go vet/lint clean after concurrent sessions  Owner:  Est: 15m
  - Deps: T5.1-T5.4
  - Acceptance: `go vet ./...` 0 warnings.

### E6: Integration Gate

Final quality gate after all epics complete.

- [ ] T6.1 Full test suite pass  Owner:  Est: 30m
  - Deps: T1.3, T2.5, T3.3, T4.5, T5.5
  - Acceptance: `go test ./... -race -count=1` passes with 0 failures.
    `go vet ./...` 0 warnings.

### E7: DGX Re-Verification

Re-run the DGX verification test suite with all GGUF loader fixes applied.

- [ ] T7.1 DGX re-verify Qwen 2.5 produces coherent text  Owner:  Est: 30m
  - Deps: T6.1
  - Acceptance: `go test -tags dgx -run TestT7_1/qwen2 -timeout 120s ./tests/dgx/` passes.
    Output is valid UTF-8, >= 5 words.

- [ ] T7.2 DGX re-verify Phi 3.5 loads and generates  Owner:  Est: 30m
  - Deps: T6.1
  - Acceptance: `go test -tags dgx -run TestT7_1/phi3 -timeout 120s ./tests/dgx/` passes.
    Model loads without "missing tensor" error.

- [ ] T7.3 DGX re-verify Mistral uses sliding window  Owner:  Est: 30m
  - Deps: T6.1
  - Acceptance: `go test -tags dgx -run TestT7_1/mistral -timeout 120s ./tests/dgx/` passes.
    Log shows architecture detected as "mistral" (not "llama").

- [ ] T7.4 DGX concurrent throughput benchmark  Owner:  Est: 30m
  - Deps: T6.1
  - Acceptance: `go test -tags dgx -run TestT7_5 -timeout 300s ./tests/dgx/` shows
    throughput > 200 tok/s with 4 concurrent clients.

- [ ] T7.5 DGX structured output verification  Owner:  Est: 30m
  - Deps: T6.1
  - Acceptance: Run structured output with a real model on DGX. Generate JSON conforming
    to a simple schema. Output is valid JSON.

- [ ] T7.6 DGX DeepSeek V3 E2E verification (carried forward)  Owner:  (BLOCKED: no MLA+MoE GGUF model)
  - Deps: T6.1
  - Acceptance: DeepSeek V3 model loads GGUF, generates coherent text on DGX.
    MoE routing activates correctly. Record tok/s.
  - Status: BLOCKED -- DeepSeek V2/V3 GGUF models require HuggingFace authentication.

---

## 4. Parallel Work

### Tracks

| Track | Epics | Description |
|-------|-------|-------------|
| A: Qwen Tokenizer | E1 (T1.1-T1.3) | Byte-level BPE fix in GGUF extractor |
| B: Phi QKV Split | E2 (T2.1-T2.5) | Merged tensor support in GGUF loader |
| C: Mistral Detection | E3 (T3.1-T3.3) | Architecture routing from metadata |
| D: Structured Output | E4 (T4.1-T4.5) | CLI, tests, example for JSON mode |
| E: Concurrent Sessions | E5 (T5.1-T5.5) | Per-request inference isolation |

Sync point: T6.1 (full integration gate) after all tracks complete.
DGX verification (E7) runs after T6.1.

### Maximum Parallelism

**Wave 1** (10 parallel tasks -- all independent, saturates agent slots):
T1.1, T2.1, T2.2, T3.1, T4.1, T4.3, T4.4, T5.1, T5.2, T5.4

Note: T2.2 can start with T2.1 in parallel since each agent works in an isolated
worktree. T2.2 creates the split logic; T2.1 adds the name mapping. Both are needed
but neither blocks the other's code writing. T5.2 and T5.4 can start in parallel with
T5.1 for the same reason.

**Wave 2** (8 parallel tasks -- unblocked after Wave 1):
T1.2, T2.3, T2.4, T3.2, T4.2, T5.3

**Wave 3** (6 parallel tasks -- lint/gate tasks):
T1.3, T2.5, T3.3, T4.5, T5.5

**Wave 4** (1 task):
T6.1

**Wave 5** (6 parallel tasks -- DGX verification):
T7.1, T7.2, T7.3, T7.4, T7.5, T7.6

---

## 5. Dependency Graph

```
T1.1 ──── T1.2 ──── T1.3 ──┐
                             │
T2.1 ──┬── T2.3 ────────────┤
T2.2 ──┤                    │
       ├── T2.4             │
       └── T2.5 ────────────┤
                             │
T3.1 ──── T3.2 ──── T3.3 ──┤
                             │
T4.1 ──── T4.2 ────────────┤
T4.3 (independent) ─────────┤
T4.4 (independent) ── T4.5 ─┤
                             │
T5.1 ──── T5.2 ──── T5.3 ──┤
T5.4 (deps T5.3) ── T5.5 ──┤
                             │
T1.3, T2.5, T3.3, T4.5, T5.5 ──── T6.1
                                      │
T6.1 ──┬── T7.1 (Qwen DGX)
       ├── T7.2 (Phi DGX)
       ├── T7.3 (Mistral DGX)
       ├── T7.4 (Throughput DGX)
       ├── T7.5 (Structured output DGX)
       └── T7.6 (DeepSeek DGX, BLOCKED)
```

---

## 6. Timeline and Milestones

| ID | Milestone | Exit Criteria | Dependencies |
|----|-----------|---------------|--------------|
| M1 | GGUF compatibility fixed | Qwen, Phi, Mistral all pass unit tests | T1.3, T2.5, T3.3 |
| M2 | Structured output E2E | CLI, library, server, example all working | T4.5 |
| M3 | Concurrent sessions | Per-request isolation, no global mutex | T5.5 |
| M4 | Integration gate | Full test suite passes with all changes | T6.1 |
| M5 | DGX verified | All 5 architectures pass on DGX hardware | T7.1-T7.5 |

---

## 7. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | Byte-level BPE fix breaks SentencePiece models | Low | High | Flag is orthogonal. Test both model types. |
| R2 | Phi QKV split incorrect for GQA | Medium | High | Test both MHA (Phi-3 mini) and GQA (Phi-3 small) configurations. |
| R3 | Mistral detection false positives for Llama 3.1 | Medium | Medium | Check SlidingWindowPattern==0 to exclude Gemma. Verify against Llama 3.1 GGUF. |
| R4 | Grammar masking slows CUDA inference | Low | Medium | O(vocab_size) on CPU, ~1ms max. Runs during GPU forward pass. |
| R5 | Concurrent sessions increase memory pressure | Medium | Medium | Add max-sessions limit. Pool and reuse KV cache allocations. |
| R6 | DGX offline during verification | Low | Medium | All CPU tests pass first. DGX verification is a separate wave. |
| R7 | ztoken changes needed for GGUF byte-level BPE | Low | High | ztoken already supports byte-level BPE. No changes needed. |

---

## 8. Operating Procedure

### Definition of Done

A task is done when:
1. Code compiles: `go build ./...` succeeds.
2. Tests pass: `go test ./... -race` in the affected packages.
3. Lint clean: `go vet ./...` 0 warnings in the affected packages.
4. Acceptance criteria from the task description are met.

### Review and QA

- Every code change must have corresponding tests.
- Run `golangci-lint run ./...` before marking lint tasks complete.
- Never commit files from different directories in the same commit.
- Make many small logical commits, not large batches.
- DGX tests require `go test -tags dgx` and DGX Spark access.

---

## 9. Progress Log

### 2026-03-16: Phase 22 plan created

**Change summary:** Created Phase 22 plan. Trimmed completed Phase 21 epics (E1 package
docs, E2 CLI UX, E3 DeepSeek V3 Production, E4 API tests, E5 doc polish, E6 integration
gate, E7 DGX verification) -- operational details preserved in docs/devlog.md entries
dated 2026-03-16. Carried forward T7.6 (DeepSeek V3 DGX E2E, blocked on model).
Added 7 new epics: E1 Qwen tokenizer, E2 Phi QKV split, E3 Mistral detection,
E4 structured output, E5 concurrent sessions, E6 integration gate, E7 DGX re-verification.
Research findings from 3-agent parallel team incorporated.

**Phase 21 trim notes:**
- E1-E6 (36 completed tasks) removed from plan. Stable knowledge already in docs/design.md
  and docs/devlog.md from Phase 21 execution.
- DGX verification findings (Qwen tokenizer, Phi QKV, Mistral arch, CUDA graph 1336%
  speedup, mutex throughput) preserved in docs/devlog.md entry 2026-03-16.
- Phase 21 appendix (DeepSeek V3 notes, Two-API design) already captured in docs/design.md.

---

## 10. Hand-off Notes

### For a new person continuing this work

- **Codebase**: `/Users/dndungu/Code/zerfoo/zerfoo/` is the main repo. 6 active repos
  total but Phase 22 work is entirely in the zerfoo repo.
- **Build**: `go test ./...` for CPU tests. `go test -tags dgx ./tests/dgx/...` on DGX.
- **Key files for Phase 22**:
  - `model/gguf/tokenizer.go` -- Qwen tokenizer fix (line 69, byteLevelBPE flag)
  - `model/gguf/arch.go` -- Tensor name mapping (tensorNameMap, MapTensorName)
  - `model/gguf/loader.go` -- GGUF loading (add QKV split after name mapping)
  - `inference/load_gguf.go` -- Architecture routing (buildArchGraph switch)
  - `inference/arch_mistral.go` -- Mistral graph builder (sliding window)
  - `inference/arch_phi.go` -- Phi graph builder (partial rotary, needs split QKV)
  - `generate/grammar/` -- Grammar engine (complete: Grammar, TokenMask, JSONSchema)
  - `generate/generator.go` -- Generator with global mutex (line 95)
  - `cmd/cli/run.go` -- CLI run command (add --json-schema)
  - `serve/server.go` -- Server (json_schema already wired at line 380)
  - `api.go` -- High-level API (WithSchema at line 443)
  - `tests/dgx/dgx_test.go` -- DGX verification suite
- **DGX Spark**: `ssh ndungu@192.168.86.250`. CUDA kernels at `/tmp/ztensor-kernels/libkernels.so`.
  Copy to `~/Code/zerfoo/libkernels.so` and set `LD_LIBRARY_PATH`.
- **Models on DGX**: Gemma3 (778MB local), TinyLlama (637MB), Qwen 2.5 0.5B (468MB),
  Mistral 7B (4.4GB), Phi 3.5 mini (2.3GB), DeepSeek 7B (4.4GB, llama arch).
- **Git workflow**: Rebase and merge. Each commit scoped to one directory.
- **Prior phases**: Phase 21 delivered docs, CLI UX, DeepSeek V3 Production, DGX verification.
  See docs/devlog.md for full history.

### Links

- DGX Spark: `ssh ndungu@192.168.86.250`
- CI: GitHub Actions (`.github/workflows/ci.yml`)
- ADRs: `docs/adr/` (39 records, 001-039)
- ADR 038: Structured output architecture (docs/adr/038-structured-output-grammar-guided-decoding.md)
- Benchmarks: `docs/benchmarks.md`, `docs/benchmarking-methodology.md`
- Design: `docs/design.md`

---

## 11. Appendix

### Qwen Byte-Level BPE

Qwen uses GPT-2 style byte-level BPE. Each byte (0x00-0xFF) is mapped to a printable
Unicode character via a deterministic mapping. Token strings in the vocabulary contain
these Unicode characters, not raw bytes. The ztoken library implements this mapping in
`buildByteEncoderDecoder()` and applies it during `byteLevelPreTokenize()` and
`decodeByteLevelBPE()`.

The GGUF metadata key `tokenizer.ggml.model` indicates the tokenizer type:
- `"llama"` -- SentencePiece BPE (uses U+2581 space marker)
- `"gpt2"` -- Byte-level BPE (GPT-2 style, Qwen, Phi, many others)

### Phi Merged QKV Tensor Layout

Phi GGUF files store Q, K, V projections in a single tensor:
- MHA: `attn_qkv.weight` shape `[3 * num_heads * head_dim, hidden_size]`
  Split evenly into 3 equal parts.
- GQA: `attn_qkv.weight` shape `[(num_heads + 2*num_kv_heads) * head_dim, hidden_size]`
  Q gets `num_heads * head_dim` rows, K and V each get `num_kv_heads * head_dim` rows.

### Concurrent Inference Session Model

```
Generator (shared, read-only after init)
  +-- graph: compiled computation graph
  +-- tokenizer: BPE tokenizer
  +-- engine: compute engine
  +-- modelConfig: vocab size, max seq len, EOS/BOS

InferenceSession (per-request)
  +-- kvCache: per-request KV cache
  +-- position: current sequence position
  +-- samplingState: temperature, top-k, top-p, grammar
```
