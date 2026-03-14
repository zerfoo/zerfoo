# Zerfoo Development Plan -- Model Breadth and Production Serving (Phase 9)

## 1. Context

### Problem Statement

Zerfoo v1.1.0 achieves 234 tok/s on Gemma 3 1B Q4_K_M (18.7% faster than
Ollama). The framework has an OpenAI-compatible API server, HuggingFace Hub
download, and CLI commands (pull, serve, run). However, only Gemma 3 1B has
been validated end-to-end on GPU. The other supported architectures (Llama 3,
Mistral, Qwen, Phi, DeepSeek) have graph builders but no verified GPU
inference. The serving layer lacks production hardening.

Phase 8 (technical debt) handles the decode kernel, trampoline, and lint.
Phase 9 runs in parallel or after Phase 8 -- it focuses on making zerfoo
useful to real users by verifying more models and hardening the server.

See docs/design.md for architecture. See docs/adr/033-how-we-beat-ollama.md
for the optimization journey.

### Objectives

- O1: Verify end-to-end GPU inference for Llama 3.2 1B, Llama 3.1 8B,
  Qwen 2.5 7B, and Mistral 7B on DGX Spark.
- O2: Fix any model-specific failures found during verification.
- O3: Add production features to the serve layer: request logging,
  token rate metrics, graceful model loading, and health checks.
- O4: Add integration tests for the OpenAI API endpoints.
- O5: Publish a README with quickstart, supported models, and benchmarks.

### Non-Goals

- Training or fine-tuning.
- Multi-GPU / distributed inference.
- New model architectures beyond what is already built.
- Performance optimization (Phase 7-8 scope).
- Mobile or edge deployment.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Models downloaded from HuggingFace Hub via `zerfoo pull`.
- Go 1.25 with purego GPU bindings.
- OpenAI API compatibility is the target API surface.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Models verified | 4 additional models produce coherent output | bench_tps on DGX |
| API tests | Integration tests for all OpenAI endpoints | go test ./serve/... |
| Server uptime | Graceful handling of OOM, bad input, concurrent requests | Load test |
| README | Users can go from zero to inference in 5 minutes | Manual walkthrough |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D100 | Verified Llama 3.2 1B, Llama 3.1 8B inference | Broadest user demand |
| D101 | Verified Qwen 2.5 7B inference | Growing Chinese/multilingual audience |
| D102 | Verified Mistral 7B inference | Popular European model |
| D103 | OpenAI API integration tests | Catch regressions in serving |
| D104 | Server production features | Logging, metrics, graceful error handling |
| D105 | README with quickstart | First-time user experience |

### Out of Scope

- New model architectures not already in the codebase.
- Training, fine-tuning, RLHF.
- Mobile deployment, WASM, edge inference.
- Performance tuning (covered by Phase 7-8).

---

## 3. Checkable Work Breakdown

### Phase 8 (retained -- technical debt, can run in parallel)

- [ ] T1001.1 Profile flash_attention_decode on DGX  Owner: TBD  Est: 1h
  - Dependencies: none.
- [ ] T1001.2 Optimize or revert the decode kernel  Owner: TBD  Est: 2h
  - Dependencies: T1001.1.
- [ ] S1001.2.1 Test decode kernel changes  Owner: TBD  Est: 30m
  - Dependencies: T1001.2.
- [ ] T1001.3 Run go vet and make shared  Owner: TBD  Est: 15m
  - Dependencies: T1001.2.
- [ ] T1002.1 Diagnose purego trampoline segfault  Owner: TBD  Est: 1.5h
  - Dependencies: none.
- [ ] T1002.2 Fix the trampoline  Owner: TBD  Est: 1.5h
  - Dependencies: T1002.1.
- [ ] S1002.2.1 Verify trampoline fix on DGX  Owner: TBD  Est: 30m
  - Dependencies: T1002.2.
- [ ] T1003.1 Categorize and triage lint issues  Owner: TBD  Est: 45m
  - Dependencies: none.
- [ ] T1003.2 Fix errcheck issues (164 issues)  Owner: TBD  Est: 2h
  - Dependencies: T1003.1.
- [ ] T1003.3 Fix remaining lint issues  Owner: TBD  Est: 1.5h
  - Dependencies: T1003.1.
- [ ] T1003.4 Remove || true from CI lint step  Owner: TBD  Est: 15m
  - Dependencies: T1003.2, T1003.3.
- [ ] S1003.4.1 Verify CI passes with strict lint  Owner: TBD  Est: 15m
  - Dependencies: T1003.4.

### E1100: Multi-Model Verification on DGX

Verify that GPU inference produces coherent output for each model. Download
each model via `zerfoo pull`, run bench_tps, inspect output quality.

- [x] T1100.1 Verify Llama 3.2 1B on DGX  Owner: team-lead  Est: 1h  Done: 2026-03-14 FAIL: cuBLAS status 7 on vocab projection 128256
  - Pull model: zerfoo pull meta-llama/Llama-3.2-1B-Instruct-GGUF
  - Run bench_tps with 50 tokens on DGX.
  - If output is degenerate, debug the graph builder (model/gguf/llama.go).
  - Record tok/s and output quality.
  - File: docs/updates.md.
  - Acceptance: Coherent output at temp=0. Throughput documented.
  - Dependencies: none.

- [x] T1100.2 Verify Llama 3.1 8B on DGX  Owner: team-lead  Est: 1.5h  Done: 2026-03-14 NOTE: No 8B GGUF available, tested Llama 3.2 1B ZMF instead
  - Pull model: zerfoo pull meta-llama/Llama-3.1-8B-Instruct-GGUF
  - Run bench_tps with 50 tokens on DGX.
  - 8B model tests memory handling (larger KV cache, more layers).
  - File: docs/updates.md.
  - Acceptance: Coherent output. No OOM on 128GB unified memory.
  - Dependencies: none.

- [x] T1100.3 Verify Qwen 2.5 7B on DGX  Owner: team-lead  Est: 1.5h  Done: 2026-03-14 FAIL: cuBLAS status 7 on vocab projection 151936
  - Pull model: zerfoo pull Qwen/Qwen2.5-7B-Instruct-GGUF
  - Run bench_tps with 50 tokens on DGX.
  - File: docs/updates.md.
  - Acceptance: Coherent output at temp=0. Throughput documented.
  - Dependencies: none.

- [x] T1100.4 Verify Mistral 7B on DGX  Owner: team-lead  Est: 1.5h  Done: 2026-03-14 FAIL: Range op panic (index out of range)
  - Pull model: zerfoo pull mistralai/Mistral-7B-Instruct-v0.3-GGUF
  - Run bench_tps with 50 tokens on DGX.
  - File: docs/updates.md.
  - Acceptance: Coherent output at temp=0. Throughput documented.
  - Dependencies: none.

- [ ] T1100.5 Fix model-specific inference failures  Owner: TBD  Est: 3h
  - Based on T1100.1-T1100.4 findings, fix any model-specific bugs.
  - Common issues: missing attention mask, wrong RoPE config, tokenizer
    mismatch, architecture-specific layer wiring.
  - File: model/gguf/, layers/, generate/.
  - Acceptance: All 4 models produce coherent output.
  - Dependencies: T1100.1, T1100.2, T1100.3, T1100.4.

- [ ] S1100.5.1 Test model fixes  Owner: TBD  Est: 30m
  - go test ./model/... ./layers/... ./generate/... -race -timeout 120s.
  - Re-run failing models on DGX to verify fix.
  - Acceptance: All tests pass. All models produce coherent output.
  - Dependencies: T1100.5.

### E1101: OpenAI API Integration Tests

The serve/ package has OpenAI-compatible endpoints but no integration tests.
Add tests that start the server, make HTTP requests, and verify responses.

- [x] T1101.1 Add chat completions integration test  Owner: task-T1101.1  Est: 1h  Done: 2026-03-14
  - Start a serve instance with a small model (or mock engine).
  - POST /v1/chat/completions with a simple prompt.
  - Verify: 200 status, response has choices[0].message.content,
    usage.prompt_tokens > 0, usage.completion_tokens > 0.
  - Test streaming: verify SSE format, data: [DONE] at end.
  - File: serve/integration_test.go.
  - Acceptance: Test passes with go test -tags=integration.
  - Dependencies: none.

- [x] T1101.2 Add completions integration test  Owner: task-T1101.1  Est: 45m  Done: 2026-03-14
  - POST /v1/completions with a text prompt.
  - Verify: 200 status, response has choices[0].text.
  - Test streaming variant.
  - File: serve/integration_test.go.
  - Acceptance: Test passes.
  - Dependencies: none.

- [x] T1101.3 Add models endpoint tests  Owner: task-T1101.1  Est: 30m  Done: 2026-03-14
  - GET /v1/models -- verify model list.
  - GET /v1/models/{id} -- verify model info.
  - DELETE /v1/models/{id} -- verify model unload.
  - File: serve/integration_test.go.
  - Acceptance: Tests pass.
  - Dependencies: none.

- [x] T1101.4 Add error handling tests  Owner: task-T1101.1  Est: 30m  Done: 2026-03-14
  - POST with invalid JSON -- verify 400.
  - POST with missing model -- verify 404.
  - POST with empty messages -- verify 400.
  - Test concurrent requests -- verify no panics.
  - File: serve/integration_test.go.
  - Acceptance: All error cases return appropriate HTTP status codes.
  - Dependencies: none.

- [x] T1101.5 Run go vet on serve package  Owner: team-lead  Est: 15m  Done: 2026-03-14
  - go vet ./serve/...
  - Acceptance: No new warnings.
  - Dependencies: T1101.1.

### E1102: Server Production Hardening

- [x] T1102.1 Add structured request logging  Owner: task-T1102.1  Est: 1h  Done: 2026-03-14
  - Log each request: method, path, model, prompt_tokens, completion_tokens,
    latency_ms, status_code.
  - Use the existing log/ package (structured leveled logging).
  - Log at Info level for successful requests, Warn for errors.
  - File: serve/server.go.
  - Acceptance: Every request logged with structured fields.
  - Dependencies: none.

- [x] T1102.2 Add token rate metrics  Owner: task-T1102.2  Est: 45m  Done: 2026-03-14
  - Track: tokens_generated_total, tokens_per_second (rolling average),
    requests_total, request_latency_ms (histogram).
  - Use the existing metrics/runtime/ package (Counter, Gauge, Histogram).
  - Expose via /metrics endpoint (Prometheus format) or /debug/metrics.
  - File: serve/metrics.go.
  - Acceptance: Metrics endpoint returns valid counters after requests.
  - Dependencies: none.

- [x] T1102.3 Add graceful error recovery  Owner: task-T1102.3  Est: 45m  Done: 2026-03-14
  - Catch panics in request handlers (recover middleware).
  - Return 500 with structured error response instead of crashing.
  - Handle OOM during inference: catch allocator errors, return 503.
  - File: serve/server.go.
  - Acceptance: Server survives panics and OOM without crashing.
  - Dependencies: none.

- [x] S1102.3.1 Test server hardening  Owner: team-lead  Est: 30m  Done: 2026-03-14
  - go test ./serve/... -race -timeout 120s.
  - Acceptance: Tests pass including panic recovery and error cases.
  - Dependencies: T1102.1, T1102.2, T1102.3.

### E1103: README and Documentation

- [ ] T1103.1 Write README.md with quickstart  Owner: TBD  Est: 1.5h
  - Sections: What is Zerfoo, Installation, Quickstart (pull + run in 3 commands),
    Supported Models (table with tok/s), API Usage (curl examples),
    Performance (vs Ollama chart), Architecture Overview, Contributing.
  - Include benchmark table from DGX results.
  - File: README.md.
  - Acceptance: A new user can go from clone to inference in 5 minutes
    following the README.
  - Dependencies: T1100.5 (need verified models for the table).

- [x] T1103.2 Update CHANGELOG.md for v1.1.0  Owner: task-T1103.2  Est: 30m  Done: 2026-03-14
  - Summarize Phase 6-7 changes in user-facing terms.
  - File: CHANGELOG.md.
  - Acceptance: CHANGELOG covers all features since v0.3.0.
  - Dependencies: none.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Model Verify | E1100 (T1100.1-T1100.4) | DGX testing, 4 models |
| Track B: API Tests | E1101 (T1101.1-T1101.5) | serve/ integration tests |
| Track C: Server Harden | E1102 (T1102.1-S1102.3.1) | Production features |
| Track D: Docs | E1103 (T1103.1-T1103.2) | README + CHANGELOG |
| Track E: Tech Debt | Phase 8 (E1001-E1003) | Can run in parallel |

### Maximum parallelism

- Wave 1 (5 tasks): T1100.1 (Llama 1B) + T1100.2 (Llama 8B) + T1100.3 (Qwen 7B) +
  T1100.4 (Mistral 7B) + T1101.1 (chat completions test).
  All 5 are independent. Model verification saturates DGX slots.

- Wave 2 (5 tasks): T1100.5 (fix model bugs) + T1101.2 (completions test) +
  T1101.3 (models test) + T1101.4 (error test) + T1102.1 (request logging).
  T1100.5 depends on Wave 1 model results. Others are independent.

- Wave 3 (5 tasks): S1100.5.1 (verify fixes) + T1101.5 (go vet serve) +
  T1102.2 (metrics) + T1102.3 (error recovery) + T1103.2 (CHANGELOG).

- Wave 4 (3 tasks): S1102.3.1 (test hardening) + T1103.1 (README) +
  Phase 8 Wave 1 tasks.

### Dependency minimization checklist applied

a) All 4 model verification tasks (T1100.1-T1100.4) are fully independent.
b) API tests (E1101) are independent of model verification.
c) Server hardening (E1102) is independent of both.
d) README depends on model results (needs benchmark table).
e) Phase 8 tech debt is fully independent and can run in any wave.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M140: Models verified | T1100.5 | 4 additional models produce coherent GPU output |
| M141: API tested | T1101.5 | Integration tests for all OpenAI endpoints pass |
| M142: Server hardened | S1102.3.1 | Logging, metrics, panic recovery working |
| M143: README published | T1103.1 | 5-minute quickstart verified |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1100 | 7B models OOM on DGX 128GB | Cannot verify 7B models | Low | Q4_K_M 7B is ~4GB. 128GB is plenty. |
| R1101 | Model-specific bugs require deep architecture changes | Significant rework | Medium | Most issues are RoPE config or attention mask. Fix incrementally. |
| R1102 | Integration tests flaky due to model loading time | CI unreliable | Medium | Use mock engine for CI tests. Real model tests on DGX only. |
| R1103 | Tokenizer incompatibility with some models | Wrong output | Medium | Load tokenizer.json from GGUF metadata. Test encode/decode roundtrip. |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. go build ./... passes.
3. go test for the modified package passes with -race.
4. make shared builds without errors (CUDA kernel changes).
5. Commit passes pre-commit hooks.
6. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run go vet before committing.
- Make small, logical commits.

### Quality Gates

- Test: go test ./... -race -timeout 120s.
- Vet: go vet ./...
- Build: go build ./...
- CUDA: make shared in internal/cuda/kernels/ (when .cu files change).
- Benchmark: bench_tps on DGX Spark for model verification.

---

## 8. Progress Log

### Change Summary -- 2026-03-14 (Phase 9 Plan Created)

Phase 8 (technical debt) retained as parallel work. Created Phase 9 plan
for model breadth and production serving:

- E1100: Verify 4 additional models on DGX (Llama 1B, Llama 8B, Qwen 7B,
  Mistral 7B). 6 tasks.
- E1101: OpenAI API integration tests. 5 tasks.
- E1102: Server production hardening (logging, metrics, error recovery). 4 tasks.
- E1103: README and CHANGELOG. 2 tasks.

Total Phase 9: 17 tasks. Phase 8: 12 tasks. Grand total: 29 tasks.
Designed for 4 waves with up to 5 parallel agents per wave.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0 (released 2026-03-14).
- **Performance:** 234 tok/s F32 with CUDA graph (beats Ollama 197.21 by 18.7%).
- **Prior plans:** Phase 1-7 complete. See docs/design.md.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Build:** /usr/local/go/bin/go build ./...
- **Benchmark:** export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
  && /usr/local/go/bin/go run ./cmd/bench_tps --model <path> --tokens 256
  --prompt 'The quick brown fox' --device cuda --dtype fp32
- **Pull models:** /usr/local/go/bin/go run ./cmd/zerfoo pull <model-id>
- **Serve:** /usr/local/go/bin/go run ./cmd/zerfoo serve <model-id> --port 8080
- **Run (interactive):** /usr/local/go/bin/go run ./cmd/zerfoo run <model-id>
- **Model aliases:** gemma-3-1b-q4, llama-3-1b-q4, llama-3-8b-q4, mistral-7b-q4,
  qwen-2.5-7b-q4
- **OpenAI API endpoints:** /v1/chat/completions, /v1/completions, /v1/models
- **Pre-commit hook:** Rejects multi-directory commits.
