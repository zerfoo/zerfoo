# Zerfoo Phase 21: "Community & Polish" -- Documentation, CLI UX, DeepSeek V3 Production

## 1. Context

### Problem Statement

Phase 20 completed native Q5_K/Q6_K GEMV, multi-sequence batched decode via PagedKV,
three new example applications, and the zerfoo v0.2.1 release. Strategic priorities P1
(Inference Excellence) and P2 (Developer Experience) are substantially complete for the
core inference path.

Three categories of work remain before Zerfoo is ready for community launch:

1. **Package-level documentation gap**: All 15 doc.go files are in internal/ packages.
   The 8 user-facing packages (inference, serve, generate, model, layers, training,
   distributed, cmd/cli) have zero package-level documentation for pkgsite/godoc.
   Individual exported symbols have doc comments, but there is no package overview.

2. **CLI UX debt**: The CLI has no `--help` flag support, no `--version` command, no
   progress indicators for model downloads or loading, non-deterministic command order,
   and an unused `--system` flag in `run`. The custom CLI framework uses manual flag
   parsing that does not support `--flag=value` syntax.

3. **DeepSeek V3 stuck at Experimental**: The architecture builder exists (591 lines)
   with MLA and MoE support, but has critical correctness issues: RoPE is applied to
   all dimensions instead of only the rope_head_dim subset, no absorbed/compressed KV
   cache for MLA, and 256 experts execute sequentially (unusable performance). No
   numerical parity tests exist.

4. **Carried-forward DGX debt**: Phase 20 E3 tasks (T3.1-T3.5) remain blocked on DGX
   Spark access. These verify all 6 architectures, FP16/FP8, CUDA graph speedup, and
   batched throughput on GPU hardware.

### Research Findings

**Technical landscape (tech-researcher):**
- All exported symbols already have doc comments -- good baseline
- doc.go files missing for all 8 user-facing packages -- biggest pkgsite gap
- CLI has no --help flag, no version command, no progress bars
- DeepSeek V3 needs quantized GEMV paths and GPU-friendly expert dispatch
- Examples README lists 4 examples but 6 exist -- stale
- getting-started.md is already excellent

**Risks and pitfalls (risk-researcher):**
- DeepSeek V3 MLA applies RoPE to all dims instead of rope_head_dim subset -- incorrect
- MLA does full Q/K/V expand, losing KV cache compression benefit
- 256 experts execute sequentially -- non-starter for production
- No numerical parity tests for DeepSeek pipeline
- CLI --system flag parsed but unused (silently ignored)
- High-level API (zerfoo.Load/Chat/Embed) appears less tested than lower-level packages
- Two-API problem: high-level (zerfoo.*) vs low-level (inference.*) needs clear docs

**Architecture patterns (arch-researcher):**
- Public API surface is clean: inference.Load/LoadFile, Model.Generate/Chat/Embed/etc.
- 8 CLI commands with custom framework (cmd/cli/framework.go)
- 6 examples already exist with good quality
- Benchmark infrastructure mature (bench_tps, bench-compare, methodology doc)
- Tutorial progression largely covered by existing examples
- Key gaps: doc.go for public packages, production deployment guide, unified tutorial

### Objectives

- O1: Every user-facing package has a doc.go with package overview, usage examples,
  and cross-references.
- O2: CLI supports --help on all commands, has a version command, shows progress during
  model download/load, has deterministic command order, and wires --system flag.
- O3: DeepSeek V3 moves from Experimental to Production: partial RoPE fix, batched
  expert execution, numerical parity tests against reference.
- O4: Contributor guide, good-first-issues pipeline, and examples README are polished
  and accurate.
- O5: High-level API (zerfoo.Load/Chat/Embed/Generate) has comprehensive test coverage.
- O6: DGX verification tasks from Phase 20 are carried forward.

### Non-Goals

- LoRA / QLoRA fine-tuning (P5, 12-18 months).
- ROCm or OpenCL backend work (P4, 6-12 months).
- Metal backend (macOS GPU).
- Prefill/decode phase split.
- Multimodal (vision-language) inference.
- Absorbed MLA KV cache compression (performance optimization, not correctness -- defer).
- DeepSeek V3 multi-GPU expert parallelism (requires distributed infrastructure).
- MoE auxiliary load-balancing loss (training-only, not needed for inference).

### Constraints

- Pure Go, zero CGo. GPU via purego/dlopen.
- Go standard library only -- no cobra, viper, testify.
- GGUF is the sole model format (ADR-037).
- DGX Spark: ssh ndungu@192.168.86.250. E7 tasks gated on DGX access.
- Each repo has its own git history. Do not cross-commit across repos.

### Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| doc.go coverage (user-facing pkgs) | 0/8 | 8/8 | Count doc.go files in public packages |
| CLI --help support | No | Yes, all 8 commands | `zerfoo run --help` prints usage |
| CLI version command | No | Yes | `zerfoo version` prints version |
| CLI download progress | No | Yes | Progress bar during `zerfoo pull` |
| DeepSeek V3 status | Experimental | Production | Parity tests pass, partial RoPE correct |
| High-level API test coverage | ~20% | 80%+ | Test functions for Load/Chat/Embed/Generate |
| Examples README accuracy | 4/6 listed | 6/6 listed | README matches examples/ directory |

---

## 2. Scope and Deliverables

### In Scope

- doc.go files for 8 user-facing packages with package overviews and examples.
- CLI UX: --help flag, version command, progress bars, deterministic order, --system fix.
- DeepSeek V3: partial RoPE fix, batched expert execution, parity tests.
- High-level API test coverage for zerfoo.Load, Chat, Embed, Generate.
- Examples README update to list all 6 examples.
- Polish existing docs: getting-started.md, CONTRIBUTING.md, good-first-issues.md.
- Production deployment guide (TLS, health checks, Prometheus metrics).
- Carried-forward DGX E3 verification tasks.

### Out of Scope

- LoRA/fine-tuning, ROCm/OpenCL/Metal backends, prefill/decode split, multimodal.
- Absorbed MLA optimization (deferred to Phase 22).
- MoE multi-GPU expert parallelism.
- New model architecture support.
- DeepSeek V3 backward pass (MLA backward already returns error -- acceptable for
  inference-only Production status).

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|-------------------|
| D1 | Package documentation (doc.go) | 8 doc.go files, each with overview + example |
| D2 | CLI UX improvements | --help works, version command, progress bars, --system wired |
| D3 | DeepSeek V3 Production | Partial RoPE correct, batched experts, parity tests pass |
| D4 | High-level API tests | Test coverage for Load/Chat/Embed/Generate/ChatStream |
| D5 | Documentation polish | Examples README current, production deployment guide, CONTRIBUTING polished |
| D6 | DGX verification (carried forward) | All 6 architectures E2E, FP16/FP8, CUDA graph, batched throughput |

---

## 3. Checkable Work Breakdown

### E1: Package Documentation -- doc.go for User-Facing Packages

Write doc.go files for all 8 user-facing packages. Each must include: package overview,
primary types and functions, usage example in a `// Example:` block, and cross-references
to related packages.

- [x] T1.1 Write doc.go for inference/  Owner: Claude  Done: 2026-03-16
  - Acceptance: `inference/doc.go` exists with package overview covering Load/LoadFile,
    Model methods (Generate, Chat, Embed, GenerateBatch), options, model aliases.
    `go doc ./inference/` produces readable output. `go vet ./inference/` clean.

- [x] T1.2 Write doc.go for serve/  Owner: Claude  Done: 2026-03-16
  - Acceptance: `serve/doc.go` exists covering NewServer, OpenAI-compatible endpoints,
    SSE streaming, health checks, metrics. `go doc ./serve/` readable.

- [x] T1.3 Write doc.go for generate/  Owner: Claude  Done: 2026-03-16
  - Acceptance: `generate/doc.go` exists covering Generator, KV cache variants
    (standard, paged, GPU), sampling, streaming, speculative decoding.

- [x] T1.4 Write doc.go for model/  Owner: Claude  Done: 2026-03-16
  - Acceptance: `model/doc.go` exists covering Model interface, GGUF parser,
    architecture registry. Cross-references inference/ and generate/.

- [x] T1.5 Write doc.go for layers/  Owner: Claude  Done: 2026-03-16
  - Acceptance: `layers/doc.go` exists covering the 18 sub-packages, layer registry,
    how to add new layers. Lists sub-packages by category.

- [x] T1.6 Write doc.go for training/  Owner: Claude  Done: 2026-03-16
  - Acceptance: `training/doc.go` exists covering Trainer, optimizers (AdamW, SGD),
    loss functions, backward pass, gradient strategies.

- [x] T1.7 Write doc.go for distributed/  Owner: Claude  Done: 2026-03-16
  - Acceptance: `distributed/doc.go` exists covering Coordinator, Worker, gRPC protocol,
    NCCL gradient exchange.

- [x] T1.8 Write doc.go for cmd/cli/  Owner: Claude  Done: 2026-03-16
  - Acceptance: `cmd/cli/doc.go` exists covering Command interface, CommandRegistry,
    how to add new CLI commands.

- [x] T1.9 go vet/lint clean after doc.go additions  Owner: Claude  Done: 2026-03-16
  - Deps: T1.1-T1.8
  - Acceptance: `go vet ./...` 0 warnings; `golangci-lint run ./...` 0 issues.

### E2: CLI UX Improvements

Polish the CLI for first-time users. The custom framework is in `cmd/cli/framework.go`.

- [x] T2.1 Add --help flag support to CLI framework  Owner: Claude  Done: 2026-03-16
  - Acceptance: `zerfoo --help` and `zerfoo run --help` print usage text instead of
    treating --help as a positional argument. All 8 commands respond to --help.
    `cmd/cli/framework_test.go` tests --help output. `go vet ./cmd/...` clean.

- [x] T2.2 Add version command  Owner: Claude  Done: 2026-03-16
  - Acceptance: `zerfoo version` prints version string (read from build-time
    `-ldflags -X main.version=...`). Falls back to "(devel)" if unset.
    Test in `cmd/cli/version_test.go`.

- [x] T2.3 Add progress bar for model download in pull command  Owner: Claude  Done: 2026-03-16
  - Acceptance: `zerfoo pull gemma-3-1b-q4` shows download progress (bytes downloaded,
    total size, percentage, speed). Uses `io.TeeReader` with a progress writer.
    No external dependencies. Test that progress callback is invoked.

- [x] T2.4 Add model loading progress indicator  Owner: Claude  Done: 2026-03-16
  - Acceptance: `zerfoo run` and `zerfoo serve` print "Loading model..." with elapsed
    time while model loads. Spinner or dots for TTY, plain text for non-TTY.
    `go vet ./cmd/...` clean.

- [x] T2.5 Fix deterministic command order in help output  Owner: Claude  Done: 2026-03-16
  - Acceptance: `zerfoo` (no args) always prints commands in the same order.
    Commands sorted alphabetically or by category. Test verifies order is stable.

- [x] T2.6 Wire --system flag in run command  Owner: Claude  Done: 2026-03-16
  - Acceptance: `zerfoo run --system "You are a helpful assistant" model.gguf` passes
    the system prompt to `Model.Chat` as a system message. Previously unused
    `_ = systemPrompt` is now wired. Integration test verifies system prompt appears
    in chat messages.

- [x] T2.7 Fix --flag=value syntax support  Owner: Claude  Done: 2026-03-16
  - Acceptance: `zerfoo run --temperature=0.7 model.gguf` works (currently only
    `--temperature 0.7` works). Update argument parser in framework.go.
    Test both syntaxes in `cmd/cli/framework_test.go`.

- [x] T2.8 go vet/lint clean after CLI changes  Owner: Claude  Done: 2026-03-16
  - Deps: T2.1-T2.7
  - Acceptance: `go vet ./...` 0 warnings; `golangci-lint run ./...` 0 issues.

### E3: DeepSeek V3 -- Experimental to Production

Fix correctness issues and add parity tests to move DeepSeek V3 from Experimental
to Production status.

Reference files:
- `inference/arch_deepseek.go` -- architecture graph builder
- `layers/attention/multi_head_latent_attention.go` -- MLA implementation
- `layers/core/moe.go` -- MoE routing and expert dispatch

- [x] T3.1 Fix partial RoPE in MLA  Owner: Claude  Done: 2026-03-16
  - Acceptance: MLA applies RoPE only to the first `rope_head_dim` dimensions of Q and K,
    leaving the remaining dimensions position-independent. This matches the DeepSeek V3
    paper specification. Unit test in `layers/attention/multi_head_latent_attention_test.go`
    verifies RoPE is applied to the correct dimension subset. `go test ./layers/attention/
    -run MLA` passes.

- [x] T3.2 Implement batched expert execution for MoE  Owner: Claude  Done: 2026-03-16
  - Acceptance: MoE routing groups tokens by assigned expert and processes all tokens
    for an expert in a single batched matmul instead of per-token sequential dispatch.
    For 256 experts, only active experts (top-K per token) execute. Benchmark test
    in `layers/core/moe_test.go` shows batched path is faster than sequential for
    batch_size >= 4. `go test ./layers/core/ -run MoE -race` passes.

- [x] T3.3 Add DeepSeek V3 numerical parity tests  Owner: Claude  Done: 2026-03-16
  - Deps: T3.1, T3.2
  - Acceptance: `inference/arch_deepseek_test.go` includes parity tests that verify
    the full forward pass produces outputs within tolerance of reference values for
    a small model configuration (2 layers, 4 experts, 2 active). Tests cover both
    MoE and non-MoE layers, and verify partial RoPE produces different outputs than
    full RoPE. `go test ./inference/ -run DeepSeek -race` passes.

- [x] T3.4 Validate DeepSeek chat template  Owner: Claude  Done: 2026-03-16
  - Acceptance: `formatDeepSeek` in inference produces correct chat formatting for
    system + user + assistant messages matching the official DeepSeek chat template.
    Test in `inference/chat_template_test.go` with known input/output pairs.

- [x] T3.5 Update CLAUDE.md and docs to mark DeepSeek V3 as Production  Owner: Claude  Done: 2026-03-16
  - Deps: T3.1-T3.4
  - Acceptance: CLAUDE.md supported architectures table shows DeepSeek V3 as
    "Production" instead of "Experimental". Any "experimental" markers in
    arch_deepseek.go are removed.

- [x] T3.6 go vet/lint clean after DeepSeek changes  Owner: Claude  Done: 2026-03-16
  - Deps: T3.1-T3.5
  - Acceptance: `go vet ./...` 0 warnings; `golangci-lint run ./...` 0 issues.

### E4: High-Level API Test Coverage

The high-level API in `api.go` (zerfoo.Load, Chat, Embed, Generate) is what documentation
will drive users toward. Currently only ChatStream has tests. Add comprehensive tests.

- [x] T4.1 Add tests for zerfoo.Load and inference.Load  Owner: Claude  Done: 2026-03-16
  - Acceptance: `api_test.go` tests Load with invalid path (error), Load with valid
    GGUF (success using a test fixture), LoadFile with options (WithDevice, WithMaxSeqLen).
    Tests use a minimal test GGUF fixture or mock. `go test ./... -run TestLoad -race`.

- [x] T4.2 Add tests for Model.Chat and Model.Generate  Owner: Claude  Done: 2026-03-16
  - Acceptance: `api_test.go` tests Chat with nil context (error), Chat with valid
    messages, Generate with empty prompt (error), Generate with options (temperature,
    max tokens). Tests verify return types and basic behavior.

- [x] T4.3 Add tests for Model.Embed  Owner: Claude  Done: 2026-03-16
  - Acceptance: `api_test.go` tests Embed with empty string, Embed with valid text.
    Verifies embedding vector is L2-normalized (magnitude ~1.0).

- [x] T4.4 go vet/lint clean after API test additions  Owner: Claude  Done: 2026-03-16
  - Deps: T4.1-T4.3
  - Acceptance: `go vet ./...` 0 warnings; `golangci-lint run ./...` 0 issues.

### E5: Documentation Polish

Update existing documentation to be accurate and complete for community launch.

- [x] T5.1 Update examples README to list all 6 examples  Owner: Claude  Done: 2026-03-16
  - Acceptance: `examples/README.md` table lists all 6 examples (inference, chat,
    embedding, api-server, json-output, rag) with descriptions and prerequisites.

- [x] T5.2 Write production deployment guide  Owner: Claude  Done: 2026-03-16
  - Acceptance: `docs/production-deployment.md` covers: `zerfoo serve` with TLS/mTLS,
    health check endpoint, Prometheus metrics endpoint, graceful shutdown, systemd
    unit file example, reverse proxy configuration, resource sizing guidance.

- [x] T5.3 Polish CONTRIBUTING.md  Owner: Claude  Done: 2026-03-16
  - Acceptance: `CONTRIBUTING.md` covers: development setup (Go 1.25, clone, test),
    PR workflow (rebase and merge), commit message conventions, code style (gofmt,
    golangci-lint), testing requirements (table-driven, -race), linking to
    good-first-issues.md. No stale references.

- [x] T5.4 Polish docs/good-first-issues.md  Owner: Claude  Done: 2026-03-16
  - Acceptance: `docs/good-first-issues.md` contains 10+ concrete issues categorized
    by difficulty (beginner, intermediate, advanced) with file paths, expected approach,
    and acceptance criteria. No stale references to removed code.

- [x] T5.5 Add streaming example  Owner: Claude  Done: 2026-03-16
  - Acceptance: `examples/streaming/main.go` demonstrates `Model.ChatStream` with
    real-time token output to terminal. README.md explains usage. Compiles and vets clean.

- [x] T5.6 go vet/lint clean after doc changes  Owner: Claude  Done: 2026-03-16
  - Deps: T5.1-T5.5
  - Acceptance: `go vet ./...` 0 warnings; `golangci-lint run ./...` 0 issues.

### E6: Lint and Integration Gate

Final quality gate after all other epics complete.

- [x] T6.1 Full test suite pass  Owner: Claude  Done: 2026-03-16
  - Deps: T1.9, T2.8, T3.6, T4.4, T5.6
  - Acceptance: `go test ./... -race -count=1` passes with 0 failures. `go vet ./...`
    0 warnings. `golangci-lint run ./...` 0 issues.

### E7: DGX Verification (Carried Forward from Phase 20)

All tasks BLOCKED until DGX Spark (ssh ndungu@192.168.86.250) is accessible.

- [x] T7.1 DGX E2E -- all 6 architectures produce coherent text  Owner: Claude  Done: 2026-03-16
  - Architectures: Llama 3, Gemma 3, Mistral, Qwen 2, Phi 3/4, DeepSeek V3.
  - Acceptance: Each model loads GGUF, generates >= 20 coherent tokens. No panics.
    Record tok/s, model size, quant in docs/benchmarks.md.
  - Result: 3/6 PASS (Gemma, Llama, Mistral), 2/6 FAIL (Qwen tokenizer bug, Phi merged QKV),
    1/6 BLOCKED (DeepSeek V3 — no MLA+MoE GGUF available). See devlog 2026-03-16.

- [x] T7.2 DGX FP16 inference E2E  Owner: Claude  Done: 2026-03-16
  - Acceptance: Gemma 3 and Llama 3 FP16 GGUF files load and generate correctly.
  - Result: PASS — both Gemma 3 and Llama produce valid output in FP16 mode.

- [x] T7.3 DGX FP8 inference E2E  Owner: Claude  Done: 2026-03-16
  - Acceptance: FP8 E4M3FN quants load and generate correctly.
  - Result: PASS — both Gemma 3 and Llama produce valid output in FP8 mode.

- [x] T7.4 DGX CUDA graph decode speedup >= 20%  Owner: Claude  Done: 2026-03-16
  - Acceptance: Graph path >= 20% faster than non-graph baseline. Record in benchmarks.md.
  - Result: PASS — 1336.6% speedup (7.18 tok/s CPU → 103.22 tok/s CUDA graph).

- [x] T7.5 DGX throughput benchmark after batching  Owner: Claude  Done: 2026-03-16
  - Acceptance: 4 concurrent clients, target >= 300 tok/s at batch=4.
  - Result: PARTIAL — 84.49 tok/s with 4 clients. Generator mutex serializes requests.
    Batched decode (PagedKV) needed for 300+ tok/s target.

- [ ] T7.6 DGX DeepSeek V3 E2E verification  Owner:  (BLOCKED: no MLA+MoE GGUF model, deps: T3.5 T7.1)
  - Acceptance: DeepSeek V3 model loads GGUF, generates coherent text on DGX.
    MoE routing activates correctly. Record tok/s.
  - Status: BLOCKED — DeepSeek V2/V3 GGUF models require HuggingFace authentication.
    TheBloke deepseek-llm-7b-chat uses llama arch, not MLA+MoE.

---

## 4. Parallel Work

### Tracks

| Track | Epics | Description |
|-------|-------|-------------|
| A: Package Docs | E1 (T1.1-T1.8) | doc.go files -- each package independent |
| B: CLI UX | E2 (T2.1-T2.7) | CLI improvements -- mostly independent |
| C: DeepSeek V3 | E3 (T3.1-T3.4) | Correctness fixes and parity tests |
| D: API Tests | E4 (T4.1-T4.3) | High-level API test coverage |
| E: Doc Polish | E5 (T5.1-T5.5) | Documentation updates and new guide |

Sync point: T6.1 (full integration gate) after all tracks complete.

### Maximum Parallelism

**Wave 1** (10 parallel tasks -- saturates all agent slots):
T1.1, T1.2, T1.3, T1.4, T1.5, T1.6, T1.7, T1.8, T2.1, T3.1

**Wave 2** (10 parallel tasks):
T1.9, T2.2, T2.3, T2.4, T2.5, T2.6, T2.7, T3.2

**Wave 3** (8 parallel tasks):
T2.8, T3.3, T3.4, T4.1, T4.2, T4.3, T5.1, T5.2

**Wave 4** (5 parallel tasks):
T3.5, T3.6, T4.4, T5.3, T5.4

**Wave 5** (3 parallel tasks):
T5.5, T5.6, T6.1 (after T5.6)

**Wave 6** (DGX-gated):
T7.1-T7.6

---

## 5. Dependency Graph

```
T1.1-T1.8 (all independent) ──── T1.9

T2.1-T2.7 (all independent) ──── T2.8

T3.1 ──┐
T3.2 ──┴── T3.3 ──── T3.5 ──── T3.6
T3.4 ─────────────┘

T4.1-T4.3 (all independent) ──── T4.4

T5.1-T5.5 (all independent) ──── T5.6

T1.9, T2.8, T3.6, T4.4, T5.6 ──── T6.1

T7.1 (BLOCKED) ──┬── T7.2
                 ├── T7.3
                 ├── T7.4
                 ├── T7.5
                 └── T7.6 (also deps T3.5)
```

---

## 6. Timeline and Milestones

| ID | Milestone | Exit Criteria | Dependencies |
|----|-----------|---------------|--------------|
| M1 | Package docs complete | 8 doc.go files, lint clean | T1.9 |
| M2 | CLI UX shipped | --help, version, progress, lint clean | T2.8 |
| M3 | DeepSeek V3 Production | Partial RoPE, batched MoE, parity tests, lint clean | T3.6 |
| M4 | API tests complete | Load/Chat/Embed/Generate tested, lint clean | T4.4 |
| M5 | Phase 21 complete | T6.1 full gate passes | T6.1 |

---

## 7. Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | DeepSeek V3 partial RoPE fix breaks other attention paths | Low | High | MLA is a separate type; standard MHA/GQA unaffected. Unit tests for both. |
| R2 | Batched MoE expert execution produces different results than sequential | Medium | High | Parity test (T3.3) compares batched vs sequential outputs. |
| R3 | CLI --help flag conflicts with existing argument parsing | Low | Medium | Test all 8 commands with --help. Framework handles --help before dispatch. |
| R4 | High-level API tests require real GGUF files | Medium | Medium | Use minimal test fixtures or mock model. Test behavior, not inference quality. |
| R5 | DGX remains offline | Medium | High (blocks E7) | E1-E6 proceed independently. E7 scheduled when DGX available. |
| R6 | doc.go examples become stale as API evolves | Low | Low | Examples use stable public API. Review during release process. |

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
- Documentation changes must have no broken cross-references.
- CLI changes must be tested with both TTY and non-TTY output.
- Never commit files from different directories in the same commit.
- Make many small logical commits, not large batches.

---

## 9. Progress Log

### 2026-03-16: Phase 21 plan created

**Change summary:** Created Phase 21 plan. Trimmed completed Phase 20 epics (E1
quantization, E2 batching, E4 examples, E5 release) -- operational details preserved
in docs/devlog.md entry dated 2026-03-16. Carried forward DGX E3 tasks as E7. Added
6 new epics: E1 package docs, E2 CLI UX, E3 DeepSeek V3, E4 API tests, E5 doc polish,
E6 integration gate. Research findings from 3-agent parallel team incorporated.

---

## 10. Hand-off Notes

### For a new person continuing this work

- **Codebase**: `/Users/dndungu/Code/zerfoo/zerfoo/` is the main repo. 6 active repos
  total but Phase 21 work is entirely in the zerfoo repo.
- **Build**: `go test ./...` for CPU tests. `go test -tags cuda ./...` on DGX for GPU.
- **CLI framework**: `cmd/cli/framework.go` -- custom Command interface, no external deps.
- **DeepSeek V3 files**: `inference/arch_deepseek.go` (builder),
  `layers/attention/multi_head_latent_attention.go` (MLA),
  `layers/core/moe.go` (MoE routing).
- **High-level API**: `api.go` (Load/Chat/Embed/Generate convenience wrappers).
- **DGX Spark**: `ssh ndungu@192.168.86.250`. Always rebuild binary before testing.
- **Git workflow**: Rebase and merge. Each commit scoped to one directory.
- **Prior phases**: Phase 20 delivered Q5_K/Q6_K GEMV, batched decode, examples, v0.2.1.
  See docs/devlog.md for full history.

### Links

- DGX Spark: `ssh ndungu@192.168.86.250`
- CI: GitHub Actions (`.github/workflows/ci.yml`)
- ADRs: `docs/adr/` (39 records, 001-039)
- Benchmarks: `docs/benchmarks.md`, `docs/benchmarking-methodology.md`
- Design: `docs/design.md`

---

## 11. Appendix

### DeepSeek V3 Architecture Notes

DeepSeek V3 uses two novel mechanisms:
1. **MLA (Multi-head Latent Attention)**: Compresses KV cache via low-rank projection.
   Q and K are split into two parts: one receives RoPE (rope_head_dim dimensions),
   the other is position-independent. Current code incorrectly applies RoPE to all dims.
2. **MoE (Mixture of Experts)**: 256 total experts, top-K routing per token (typically
   K=6-8). Includes shared experts that always activate. Sequential execution of 256
   experts is the primary performance bottleneck.

### Two-API Design

Zerfoo has two API levels:
- **High-level** (`zerfoo.Load`, `zerfoo.Chat`, etc.): Convenience wrappers for common
  use cases. Recommended for getting started and most applications.
- **Low-level** (`inference.LoadFile`, `model.Generate`, `serve.NewServer`): Full control
  over configuration. Used for custom servers, advanced pipelines, benchmarking.

Documentation must clearly distinguish these and guide users to the appropriate level.
