# Zerfoo Development Plan -- Phase 15: Remaining Model Quality + Performance Regression

## 1. Context

See docs/design.md for full architecture, package layout, and conventions.

### Problem Statement

Phase 14 implemented GPU-native Cos/Sin/Expand/ScatterND ops (eliminating D2H
copies), added repetition penalty to sampling, and fixed the ConstantOfShape
tensor fill bug that caused broken causal masks for Qwen 2.5 and Mistral 7B.

Three issues remain from Phase 14 verification (DGX, 2026-03-15):

1. **Mistral 7B tokenizer space decoding.** Output is "jumpedoverthequickbark..."
   -- words are recognizable English but have no spaces between them. The
   SentencePiece tokenizer uses a special `U+2581` (lower one eighth block,
   displayed as `_`) prefix to mark word boundaries, which must be decoded as
   a space character. The current tokenizer decoding path does not perform this
   substitution.

2. **Phi 4 output regression.** Output degraded from "'s a new and the
   following..." to "jjjjjjjjjj" (single-character repetition). CUDA graph
   capture also fails with "cudaMemcpy failed: operation would make the legacy
   stream depend on a capturing blocking stream" at instruction 75 (Mul).
   The regression may be caused by the ConstantOfShape fix interacting
   differently with Phi 4's graph structure, or by a separate issue in Phi 4's
   execution path. Needs diagnosis.

3. **Gemma 3 GGUF throughput regression.** Performance dropped from 232.86 tok/s
   (Phase 11 baseline) to 122.70 tok/s (Phase 14). This is a 47% regression
   on the same hardware (DGX Spark GB10, sm_121). The GGUF/ZMF codegen pipeline
   uses fused ops and should not be affected by ONNX-path changes. The
   regression may be caused by: code changes affecting the hot path, Go runtime
   changes, CUDA driver updates, or a measurement artifact. Needs profiling.

4. **TestCPUEngine_Exp float precision failure.** Pre-existing test failure in
   compute/cpu_engine_test.go:224. Expected [2.7182817 7.389056 20.085537
   54.59815], got [2.7182794 7.389056 20.085533 54.59815]. This is a float32
   precision issue in the CPU Exp implementation -- likely the expected values
   were computed with a different math library. The fix is to widen the
   tolerance or update the expected values to match Go's math.Exp output.

### Objectives

- O1: Fix Mistral 7B tokenizer to decode SentencePiece word boundaries as spaces.
- O2: Diagnose and fix Phi 4 output regression.
- O3: Diagnose and fix Gemma 3 GGUF throughput regression.
- O4: Fix TestCPUEngine_Exp precision failure.
- O5: All 5 models produce usable output on DGX.

### Non-Goals

- New model architectures.
- Multi-GPU / distributed inference.
- Training, fine-tuning, RLHF.
- FP16 mixed precision (deferred).
- Matching ORT output bit-for-bit.

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- DGX Spark: ssh ndungu@192.168.86.250, project at ~/zerfoo.
- DGX Spark GB10: sm_121, 273 GB/s LPDDR5x, 128GB unified memory.
- Go 1.25 with purego GPU bindings (no CGo for CUDA).
- DGX requires `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- DGX uses `upstream` HTTPS remote for fetch (origin SSH has host key issue).
- All 5 models run without crashes (Phases 11-14 fixes).
- PRs #64 and #65 (Phase 14 code) are open and need merging.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| Mistral 7B | Coherent text with spaces | bench_tps on DGX |
| Phi 4 | No regression, coherent output | bench_tps on DGX |
| Gemma 3 GGUF | 230+ tok/s restored | bench_tps on DGX |
| TestCPUEngine_Exp | PASS | go test ./compute/... -race |
| No regression | All other models still work | bench_tps on DGX |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D800 | Mistral tokenizer space fix | SentencePiece word boundary decoding |
| D801 | Phi 4 regression diagnosis and fix | Output quality restored |
| D802 | Gemma 3 throughput regression diagnosis and fix | 232 tok/s restored |
| D803 | TestCPUEngine_Exp fix | Clean test suite |
| D804 | All-model verification on DGX | Confirm all fixes |

### Out of Scope

- FP16 mixed precision.
- New model architectures.
- CUDA graph capture for ONNX models.
- Training, fine-tuning, RLHF.

---

## 3. Checkable Work Breakdown

### E3600: Mistral Tokenizer Space Fix

Mistral output has no spaces between tokens. The SentencePiece tokenizer uses
U+2581 prefix to mark word boundaries. The tokenizer decoding path must
replace this character with a space.

- [x] T3600.1 Diagnose Mistral tokenizer space issue  Owner: agent  Done: 2026-03-15
  - Read layers/tokenizers/ and any SentencePiece decoding code.
  - Search for U+2581 handling, "sentencepiece", or byte-level BPE decoding.
  - Check if the tokenizer config has a `decoder` field with
    `type: "Replace"` or `type: "Strip"` rules.
  - Read the Mistral tokenizer.json on DGX to understand its decoder config.
  - File: layers/tokenizers/, inference/.
  - Acceptance: Root cause identified. Missing U+2581-to-space replacement located.
  - Dependencies: none.

- [x] T3600.2 Fix Mistral tokenizer space decoding  Owner: agent  Done: 2026-03-15
  - Add U+2581 to space replacement in the token decoding path.
  - This should be a general SentencePiece decoder fix, not Mistral-specific.
  - Verify Llama 3 and Qwen 2.5 (which also use SentencePiece) still work.
  - File: layers/tokenizers/ or inference/.
  - Acceptance: Mistral output has spaces between words.
  - Dependencies: T3600.1.

- [x] S3600.2.1 Test tokenizer space fix  Owner: agent  Done: 2026-03-15
  - Unit test: decode tokens containing U+2581 prefix, verify spaces in output.
  - Test: decode tokens without U+2581, verify no spurious spaces added.
  - go test for affected package -race.
  - Dependencies: T3600.2.

- [ ] S3600.2.2 Test Mistral on DGX  Owner: TBD  Est: 15m
  - bench_tps for Mistral 7B with 20 tokens.
  - Acceptance: Output has spaces between words.
  - Dependencies: T3600.2.

### E3601: Phi 4 Output Regression

Phi 4 output regressed from semi-coherent to "jjjjjjjj". CUDA graph capture
also fails. Need to diagnose whether the ConstantOfShape fix caused the
regression or if there is a separate issue.

- [x] T3601.1 Diagnose Phi 4 output regression  Owner: agent  Done: 2026-03-15
  - Run Phi 4 on DGX with debug logging.
  - Check ConstantOfShape nodes in Phi 4's graph: what fill values are used?
    Did the fix change any values that should have been 0?
  - Check CUDA graph capture failure: instruction 75 (Mul) does cudaMemcpy
    during capture. Is this a new issue or pre-existing?
  - Compare Phi 4 output before and after ConstantOfShape fix by temporarily
    reverting the fix and re-running.
  - Check if Phi 4 uses a different attention pattern than Llama/Qwen/Mistral.
  - File: layers/core/constantofshape.go, compute/gpu_kernels.go.
  - Acceptance: Root cause of regression identified.
  - Dependencies: none.

- [x] T3601.2 Fix Phi 4 CUDA graph capture  Owner: agent  Done: 2026-03-15
  - Apply fix based on diagnosis.
  - If ConstantOfShape fix is the cause, ensure the fix correctly handles
    all Phi 4 ConstantOfShape nodes (some may legitimately need fill=0).
  - Dependencies: T3601.1.

- [ ] S3601.2.1 Test Phi 4 fix on DGX  Owner: TBD  Est: 15m
  - bench_tps for Phi 4 with 20 tokens.
  - Acceptance: Output is at least as good as pre-Phase-14 ("'s a new and...").
  - Dependencies: T3601.2.

### E3602: Gemma 3 GGUF Throughput Regression

Gemma 3 GGUF dropped from 232.86 to 122.70 tok/s. This is the ZMF codegen
pipeline which uses fused ops and should not be affected by ONNX-path changes.

- [x] T3602.1 Profile Gemma 3 throughput regression  Owner: agent  Done: 2026-03-15
  - Run Gemma 3 on DGX with CPU profiling: -cpuprofile=prof.out.
  - Compare hot functions with Phase 11 baseline if available.
  - Check if CUDA graph capture is still working (should capture 184/185 ops).
  - Check if new GPU ops (Cos/Sin/Expand/ScatterND) are being dispatched
    unnecessarily for GGUF models that use fused ops.
  - Verify the Go version and CUDA driver version on DGX match Phase 11.
  - Try running on the Phase 11 commit to establish if the regression is
    code-related or environmental.
  - File: cmd/bench_tps/main.go, graph/cuda_graph.go, compute/gpu_engine.go.
  - Acceptance: Root cause of throughput regression identified.
  - Dependencies: none.

- [x] T3602.2 Fix Gemma 3 throughput regression  Owner: agent  Done: 2026-03-15
  - No fix needed. Regression was measurement artifact (20 tokens vs 256).
  - 235.46 tok/s with 256 tokens (matches Phase 11 baseline).
  - Dependencies: T3602.1.

- [x] S3602.2.1 Test Gemma 3 throughput on DGX  Owner: agent  Done: 2026-03-15
  - bench_tps for Gemma 3 GGUF with 256 tokens.
  - Acceptance: 230+ tok/s restored.
  - Dependencies: T3602.2.

### E3603: Fix TestCPUEngine_Exp Precision

Pre-existing test failure. The expected values do not match Go's math.Exp
output for float32.

- [x] T3603.1 Fix TestCPUEngine_Exp precision failure  Owner: agent  Done: 2026-03-15
  - Read compute/cpu_engine_test.go:224 to understand the test.
  - Determine whether expected values or tolerance needs updating.
  - If the CPU Exp implementation uses a custom fast-math approximation,
    the test should use appropriate tolerance (1e-5 relative error).
  - If the implementation uses math.Exp, update expected values to match.
  - File: compute/cpu_engine_test.go.
  - Acceptance: go test ./compute/... -race passes with zero failures.
  - Dependencies: none.

- [x] S3603.1.1 Run full compute test suite  Owner: agent  Done: 2026-03-15
  - go build ./... && go vet ./... && go test ./compute/... -race -timeout 120s.
  - Acceptance: Zero test failures (including the previously-failing Exp test).
  - Dependencies: T3603.1.

### E3604: All-Model Verification

- [ ] T3604.1 Run all 5 models on DGX and record results  Owner: TBD  Est: 1h
  - bench_tps for Gemma 3 (GGUF), Llama 3, Qwen 2.5, Mistral 7B, Phi 4.
  - Use --repetition-penalty 1.2 for ONNX models.
  - Record tok/s, output quality for each.
  - File: docs/updates.md.
  - Acceptance: Gemma 3 >= 230 tok/s. Mistral has spaces. Phi 4 not regressed.
    All models produce output without crashes.
  - Dependencies: S3600.2.2, S3601.2.1, S3602.2.1, S3603.1.1.

---

## 4. Parallel Work (optimize for up to 5 concurrent agents)

| Track | Tasks | Notes |
|-------|-------|-------|
| Track A: Mistral Tokenizer | T3600.1, T3600.2, S3600.2.1, S3600.2.2 | Local code + DGX test |
| Track B: Phi 4 Regression | T3601.1, T3601.2, S3601.2.1 | DGX diagnosis + fix |
| Track C: Gemma 3 Perf | T3602.1, T3602.2, S3602.2.1 | DGX profiling + fix |
| Track D: Exp Test Fix | T3603.1, S3603.1.1 | Local code only |
| Track E: Final Verify | T3604.1 | DGX, depends on A-D |

### Maximum parallelism

- Wave 1 (4 tasks): T3600.1 (diagnose Mistral tokenizer) + T3601.1 (diagnose
  Phi 4 regression, DGX) + T3602.1 (profile Gemma 3 regression, DGX) +
  T3603.1 (fix Exp test). All independent. 4 teammates.
  Note: T3601.1 and T3602.1 both need DGX. They can share DGX sequentially
  within one agent, or run on separate SSH sessions.

- Wave 2 (3 tasks): T3600.2 (fix Mistral tokenizer) + T3601.2 (fix Phi 4) +
  T3602.2 (fix Gemma 3 perf). Each depends on its diagnosis. 3 teammates.

- Wave 3 (4 tasks): S3600.2.1 (test tokenizer) + S3600.2.2 (test Mistral DGX) +
  S3601.2.1 (test Phi 4 DGX) + S3602.2.1 (test Gemma 3 DGX) + S3603.1.1 (test
  compute suite). Combine DGX tests into 1-2 agents. 2-3 teammates.

- Wave 4 (1 task): T3604.1 (all-model verification, DGX). 1 teammate.

### Dependency minimization checklist applied

a) All 4 diagnosis/fix tasks in Wave 1 are independent.
b) DGX tasks (T3601.1, T3602.1) can run in parallel via separate SSH sessions
   since they test different models and do not contend for GPU memory.
c) T3603.1 (Exp test fix) has zero dependencies and is purely local code.
d) Wave 1 saturates 4 teammates with independent work.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M800: Issues diagnosed | T3600.1, T3601.1, T3602.1 | All 3 regressions have root cause |
| M801: Fixes applied | T3600.2, T3601.2, T3602.2, T3603.1 | All fixes in code, tests pass |
| M802: All models verified | T3604.1 | 5/5 models produce usable output, Gemma 3 >= 230 tok/s |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R3600 | Mistral tokenizer fix breaks other SentencePiece models (Llama, Qwen) | Regression | Low | Test all SentencePiece models after fix. The U+2581 replacement is standard SentencePiece behavior. |
| R3601 | Phi 4 regression is architectural (attention pattern incompatible with ConstantOfShape fix) | Harder fix | Medium | Revert ConstantOfShape fix selectively for Phi 4 if needed. Check if Phi 4 uses ConstantOfShape differently. |
| R3602 | Gemma 3 regression is environmental (CUDA driver, Go version) not code | Cannot fix in code | Low | Bisect commits to isolate. If environmental, document as known limitation. |
| R3603 | Fixing Exp test reveals other precision issues across test suite | More work | Low | Survey float32 comparison patterns in tests. Use tolerance-based comparisons. |

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

### DGX Preflight (required before any DGX benchmark)

1. ssh ndungu@192.168.86.250
2. export PATH=/usr/local/cuda/bin:$PATH
3. cd ~/zerfoo && git fetch upstream <branch> && git checkout <branch>
4. cd internal/cuda/kernels && make clean && make shared CUDA_ARCH=sm_121
5. cd ~/zerfoo
6. export LD_LIBRARY_PATH=$(pwd)/internal/cuda/kernels:/usr/local/cuda/lib64
7. Verify: /usr/local/go/bin/go run ./cmd/bench_tps --help

---

## 8. Progress Log

### Change Summary -- 2026-03-15 (Phase 15: Remaining Issues)

Trimmed completed Phase 14 work into docs/design.md:
- GPU Cos/Sin/Expand/ScatterND kernels
- ConstantOfShape tensor fill fix
- Repetition penalty CLI
- Phase 14 verification results per model

Phase 15 created to address 4 remaining issues from Phase 14 verification:
- E3600: Mistral tokenizer SentencePiece space decoding
- E3601: Phi 4 output regression diagnosis
- E3602: Gemma 3 throughput regression diagnosis
- E3603: TestCPUEngine_Exp precision fix
- E3604: All-model final verification

No new ADRs needed.

---

## 9. Hand-off Notes

- **Current version:** v1.1.0.
- **Branch:** PRs #64 and #65 open for Phase 14 (feat/phase14-wave1, feat/phase14-wave2).
  These must be merged before Phase 15 work begins.
- **Model status after Phase 14 (DGX, feat/phase14-wave2 branch):**
  - Gemma 3 GGUF: 122.70 tok/s (REGRESSED from 232), poor output
  - Llama 3 ONNX: 12.90 tok/s, semi-coherent
  - Qwen 2.5 ONNX: 15.54 tok/s, FIXED (no more single-token repetition)
  - Mistral 7B ONNX: 3.65 tok/s, words coherent but no spaces (tokenizer)
  - Phi 4 ONNX: 4.53 tok/s, REGRESSED to "jjjjjjjj"
- **Key fixes applied in Phase 14:**
  - GPU Cos/Sin/Expand/ScatterND (eliminate D2H copies)
  - ConstantOfShape *zmf.Tensor fill value fix (constantofshape.go)
  - Repetition penalty (--repetition-penalty CLI flag)
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
  IMPORTANT: Use `upstream` HTTPS remote for fetch (origin SSH has host key issue).
  `export PATH=/usr/local/cuda/bin:$PATH` before `make shared`.
- **Key files for Phase 15:**
  - layers/tokenizers/ -- SentencePiece U+2581 decoding
  - layers/core/constantofshape.go -- may need Phi 4 specific handling
  - compute/cpu_engine_test.go:224 -- Exp precision test
  - graph/cuda_graph.go -- Gemma 3 graph capture performance
- **Models on DGX:** ~/models/gemma3-gguf/model.gguf, ~/models/llama3/,
  ~/models/qwen25/, ~/models/mistral/, ~/models/phi4/
- **Pre-commit hook:** Rejects multi-directory commits.
