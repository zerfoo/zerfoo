# Zerfoo Development Plan -- Runtime GPU Detection (ADR-025 Implementation)

## 1. Context

### Problem Statement

The megakernel cannot fire because of a build tag conflict: the megakernel
runner (internal/codegen/runner.go, //go:build !cuda) uses purego dlopen, while
the GPU engine (compute/gpu_engine.go, //go:build cuda) requires -tags cuda.
These are mutually exclusive -- no single build can include both. Additionally,
16 ops needed by Gemma 3 are missing from codegen.CheckSupport.

ADR-025 mandates runtime GPU detection via dlopen: one binary, no build tags,
graceful CPU fallback. T87.3 completed this for internal/cuda/ runtime. This
plan extends it to the remaining CUDA packages.

### Objectives

- O1: Remove //go:build cuda tags from all pure Go CUDA files (compute/,
  inference/, tensor/, internal/codegen/).
- O2: Replace CGo kernel files in internal/cuda/kernels/ with purego-only
  implementations (purego wrappers already exist).
- O3: Make GPU engine creation runtime-detected via cuda.Available().
- O4: Add the 16 missing op emitters to codegen.CheckSupport so the megakernel
  can fire on Gemma 3.
- O5: Verify megakernel fires on DGX Spark and produces correct output.

### Non-Goals

- Converting internal/cublas/ or internal/cudnn/ from CGo to purego (Phase 2).
- Converting ROCm or OpenCL backends to purego.
- Converting TensorRT or CUTLASS bindings to purego.
- Performance tuning (Track B -- after megakernel correctness is verified).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- internal/cublas/ and internal/cudnn/ remain behind //go:build cuda (CGo).
  The gpuapi layer must handle nil BLAS/DNN gracefully.
- The megakernel path does not use cuBLAS or cuDNN -- it runs everything via
  the kernel runner. So removing BLAS/DNN does not block the megakernel.
- Non-megakernel GPU inference (per-op dispatch) uses cuBLAS for MatMul. This
  path still requires -tags cuda until cublas is converted to purego (Phase 2).
- DGX Spark available at ssh ndungu@192.168.86.250.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| go build ./... | Passes without -tags cuda | go build ./... on macOS and Linux |
| Megakernel fires | "megakernel: compiled and loaded" log | bench_tps on DGX Spark |
| Correctness | 50 tokens match plan.Run() output | Token comparison test |
| Existing tests | All pass | go test ./... -timeout 120s |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D1 | Remove cuda tags from pure Go files | ADR-025: runtime detection |
| D2 | Delete CGo kernel files, purego-only | Eliminate CGo dependency for kernels |
| D3 | Runtime GPU engine creation | cuda.Available() guards |
| D4 | Optional BLAS/DNN in GPUEngine | Graceful nil handling when cublas absent |
| D5 | 16 missing op emitters | Unblock megakernel for Gemma 3 |
| D6 | Megakernel fires on DGX Spark | End-to-end verification |

### Out of Scope

- cuBLAS/cuDNN purego conversion (separate epic, Phase 2).
- ROCm, OpenCL, TensorRT, CUTLASS purego conversion.
- Performance tuning (Track B).

---

## 3. Checkable Work Breakdown

### E101: Remove Build Tags from internal/codegen/ -- COMPLETE

- [x] T101.1 Merge runner.go and runner_stub.go -- commit a64d831
- [x] S101.1.1 Test runner builds and skips without CUDA -- commit a64d831

### E102: Remove Build Tags from internal/cuda/kernels/

- [x] T102.1 Make purego kernel wrappers sole implementation -- commit d9375fb
  - Deleted 5 CGo files (elementwise, rmsnorm, transpose, gemm_q4, gather).
  - Removed !cuda from 6 purego files. flash_attention.go kept behind cuda&&cutlass.
- [x] S102.1.1 Kernel parity test passes without cuda tag -- commit d9375fb

### E103: Remove Build Tags from compute/ -- COMPLETE

- [x] T103.1 Remove cuda tag from gpu_engine.go -- commits eb7e77e, cd31b73
  - Added BLAS/DNN factory registration pattern in gpuapi/factory.go.
  - cuda_blas.go and cuda_dnn.go register via init() (stay behind cuda tag).
  - NewGPUEngine guarded with cuda.Available(). BLAS/DNN optional (nil-safe).
  - MatMul falls back to CPU when BLAS is nil.
- [x] T103.2 Remove cuda tag from gpu_kernels.go -- commit cd31b73
- [x] T103.3 Remove cuda tag from gpu_cudnn.go -- commit cd31b73
  - All DNN methods guarded with nil check, return descriptive error.
- [x] T103.4 Remove cuda tag from gpu_fused_rmsnorm.go -- commit cd31b73
- [x] S103.4.1 GPU engine tests use runtime skip -- commit 6b1af69
  - 5 test files updated: removed //go:build cuda, added cuda.Available() skip.

### E104: Remove Build Tags from inference/

- [ ] T104.1 Unify engine_cuda.go and engine_nocuda.go  Owner: TBD  Est: 45m
  - Merge into a single engine.go with runtime detection:
    if cuda.Available() -> create GPU engine, else -> CPU fallback.
  - Delete engine_cuda.go and engine_nocuda.go.
  - Keep engine_rocm.go and engine_opencl.go behind their build tags (out of
    scope).
  - Acceptance: go build ./inference/... passes without -tags cuda.
    Device "cuda" creates GPU engine when CUDA available, returns error when not.
  - Dependencies: T103.1 (GPUEngine compiles without cuda tag).

- [ ] T104.2 Update TensorRT files to remain behind cuda tag  Owner: TBD  Est: 15m
  - tensorrt_cache.go, tensorrt_convert.go, tensorrt_pipeline.go: keep
    //go:build cuda (TensorRT is CGo, out of scope for this epic).
  - Verify these do not break go build without -tags cuda.
  - Acceptance: go build ./inference/... passes.
  - Dependencies: T104.1.

- [ ] S104.2.1 Inference engine tests  Owner: TBD  Est: 15m
  - Verify createEngine("cuda") returns GPU engine when CUDA available, error
    when not. Verify createEngine("cpu") always works.
  - Acceptance: go test ./inference/... passes.
  - Dependencies: T104.2.

### E105: Remove Build Tags from tensor/

- [x] T105.1 Remove build tags from GPU storage files -- commit 44c68ba
  - Removed tags from gpu_storage.go, gpu_storage_default_cuda.go, transfer.go.
  - Added cuda.Available() runtime guards.
- [x] S105.1.1 Tensor GPU tests use runtime skip -- commit 44c68ba

### E106: Add Missing Op Emitters to codegen

S100.2.1 identified 16 ops rejected by CheckSupport:
AutoPositionIds, AutoZeroKVCache, Shape, Unsqueeze, Cast, Equal, Where,
ConstantOfShape, Expand, Range, Cos, Sin, Greater, Trilu, Max, ScatterND.

- [x] T106.1 Add RoPE op emitters (Cos, Sin, Range) -- commit 4bc6e9a
- [x] T106.2 Add attention masking emitters (Trilu, Where, Greater, Equal, ConstantOfShape, Expand) -- commit 4bc6e9a
- [x] T106.3 Add utility op emitters (Shape, Unsqueeze, Cast, Max, ScatterND) -- commit 51ea41d
- [x] T106.4 Add auto ops (AutoPositionIds, AutoZeroKVCache) -- commit 51ea41d
- [x] S106.4.1 Emitter unit tests -- commits 4bc6e9a, 51ea41d
- [x] T106.5 Run golangci-lint on internal/codegen/ -- no issues found
  - go vet, go build, go test all clean. No new lint warnings.

### E107: Full Build Verification

- [ ] T107.1 Verify go build ./... without -tags cuda  Owner: TBD  Est: 30m
  - Run go build ./... on macOS (no CUDA).
  - Fix any compilation errors from removed build tags.
  - Acceptance: go build ./... passes with zero errors (excluding pre-existing
    model/ package issues).
  - Dependencies: E101-E105.

- [ ] T107.2 Verify go build -tags cuda on DGX Spark  Owner: TBD  Est: 30m
  - SSH to DGX Spark, pull latest, run go build -tags cuda ./...
  - CGo files (cublas, cudnn, nccl) still compile with cuda tag.
  - Purego files also compile (no conflict).
  - Acceptance: go build -tags cuda ./... passes on DGX Spark.
  - Dependencies: T107.1.

- [ ] S107.2.1 Run full test suite  Owner: TBD  Est: 30m
  - go test ./... -race -timeout 120s on macOS (without cuda).
  - go test -tags cuda ./... -race -timeout 120s on DGX Spark.
  - Acceptance: All tests pass (GPU tests skip on macOS, run on DGX Spark).
  - Dependencies: T107.2.

### E108: Megakernel End-to-End Verification (replaces T100.3)

- [ ] T108.1 Run bench_tps on DGX Spark without -tags cuda  Owner: TBD  Est: 30m
  - Build without -tags cuda: go build ./cmd/bench_tps/
  - Run with -device cuda flag.
  - Verify "megakernel: compiled and loaded" log appears.
  - Record tok/s.
  - Acceptance: Megakernel fires. Performance recorded.
  - Dependencies: E106, E107.

- [ ] T108.2 Compare megakernel vs plan.Run() output  Owner: TBD  Est: 1h
  - Generate 50 tokens with megakernel enabled.
  - Generate 50 tokens with megakernel disabled.
  - Compare token-by-token.
  - Acceptance: Outputs match or differences documented with explanation.
  - Dependencies: T108.1.

- [ ] T108.3 Run golangci-lint on all modified packages  Owner: TBD  Est: 15m
  - Acceptance: No new lint warnings.
  - Dependencies: T108.2.

- [ ] S108.3.1 Update docs and checkpoint  Owner: TBD  Est: 15m
  - Update plan.md, docs/updates.md, .claude-checkpoint.md.
  - Acceptance: All results documented.
  - Dependencies: T108.3.

---

## 4. Parallel Work

| Track | Epics/Tasks | Notes |
|-------|-------------|-------|
| Track A: Build tag removal | E101, E102, E103, E104, E105 | Can run in parallel (different packages) |
| Track B: Op emitters | E106 | Independent of Track A |
| Track C: Verification | E107, E108 | Sync point: depends on Track A + B complete |

E101-E105 are independent (each touches a different package). E106 is independent.
All converge at E107 for full build verification, then E108 for DGX Spark test.

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M61: Build tag removal | E101-E105 | go build ./... passes without -tags cuda |
| M62: All emitters added | E106 | CheckSupport accepts all Gemma 3 ops |
| M63: Megakernel fires | E107, E108 | "megakernel: compiled and loaded" on DGX Spark |
| M64: Correctness verified | T108.2 | 50 tokens match plan.Run() output |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R104 | compute/ imports cublas/cudnn transitively | Build fails without cuda | High | Make BLAS/DNN optional via nil checks in GPUEngine |
| R105 | Removing kernel CGo breaks DGX Spark GPU tests | Test regression | Medium | Run full test suite on DGX Spark before and after |
| R106 | Some emitters generate incorrect CUDA code | Wrong megakernel output | Medium | Unit test each emitter, compare output on DGX Spark |
| R107 | purego dlopen path slower than CGo | Performance regression | Low | Benchmark before/after; CGo overhead was only ~0.5-1% |
| R92 | Register pressure: hidden_dim=2048 | Must tile, slower | High | Profile with nvcc --ptxas-options=-v |
| R95 | KV cache reads limit bandwidth | Cannot reach max | High | Focus on short contexts (<512) |
| R100 | Tracing captures wrong path | Wrong megakernel output | Medium | Only use for decode (seqLen=1) |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. go build ./... passes without -tags cuda.
3. go test for the modified package passes with -race.
4. Commit passes pre-commit hooks.
5. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run go vet before committing.

---

## 8. Progress Log

### Change Summary -- 2026-03-12 (Wave 4)

Wave 4: 2 agents, 2 tasks completed:
- E103 (T103.1-S103.4.1): Removed //go:build cuda from all compute/ files.
  Added BLAS/DNN factory pattern in gpuapi. BLAS and DNN now optional (nil-safe).
  Commits: eb7e77e, cd31b73, 6b1af69. Stub conflict fixed (fff635b).
- T106.5: golangci-lint on codegen -- no issues found.
Newly unblocked: E104 (inference/ tags), E107 (full build verification).

### Change Summary -- 2026-03-11 (Wave 3)

Wave 3: 5 agents, 7 tasks completed in parallel:
- T101.1+S101.1.1: codegen runner merged (a64d831)
- T102.1+S102.1.1: purego-only kernels, 5 CGo files deleted (d9375fb)
- T105.1+S105.1.1: tensor GPU tags removed (44c68ba)
- T106.1+T106.2: 9 RoPE+attention emitters (4bc6e9a)
- T106.3+T106.4+S106.4.1: 7 utility+auto emitters (51ea41d)
Newly unblocked: E103 (compute/ tags), E104 (inference/ tags), T106.5, E107.

### Change Summary -- 2026-03-11

New plan created for Option B: runtime GPU detection per ADR-025.
Scope: remove //go:build cuda from pure Go files, delete CGo kernel files
(purego is sole implementation), add 16 missing op emitters, verify
megakernel fires on DGX Spark.
Trimmed completed E1-E6, T87.3, S87.3.1, S88.2.1, S100.1.1, T100.2,
S100.2.1 to design.md/progress notes. Preserved pending Track A (S88.3.1,
T89.2, S89.3.1), Track B, and remaining Track C tasks in context.

Prior completed work (Waves 1-2):
- T87.3: CGo runtime replaced with purego dlopen (commit 8286656)
- S87.3.1: Runtime parity test (commit 8286656)
- S88.2.1: Elementwise kernel parity test (commit c50e4ab)
- S100.1.1: DGX Spark test -- GPU F32 12.84 tok/s (non-megakernel)
- T100.2: GPU KV cache wired into megakernel (commits 0b3ab3b, 6116588, 9ffa7bc)
- S100.2.1: Megakernel blocked -- 16 missing ops + cuda/!cuda tag conflict

---

## 9. Hand-off Notes

- **ADR-025:** docs/adr/025-purego-cuda-bindings.md -- the design decision
  driving this work.
- **PR workflow:** PRs go to zerfoo/zerfoo (upstream), not dndungu/zerfoo.
  Use `gh pr create --repo zerfoo/zerfoo --head dndungu:<branch>`.
- **DGX Spark:** ssh ndungu@192.168.86.250. Project at ~/zerfoo.
- **Pre-commit hook:** rejects multi-directory commits.
- **cublas/cudnn remain CGo:** internal/cublas/ and internal/cudnn/ keep their
  //go:build cuda tags. Purego conversion is a separate future epic.
  The GPUEngine must handle nil BLAS/DNN gracefully.
- **Non-megakernel GPU path:** Still requires -tags cuda for cuBLAS MatMul.
  The megakernel path works without -tags cuda.

## 10. Performance Baselines

| Config | tok/s | Source |
|--------|-------|--------|
| GPU F32 (non-megakernel) | 12.84 | S100.1.1 DGX Spark (2026-03-11) |
| GPU Q4 (non-megakernel) | 8.61 | S100.1.1 DGX Spark (2026-03-11) |
| CPU ARM64 (post Track D) | 8.15 median | Phase 34 Track D |
| Ollama GB10 | ~100 (est.) | Interpolated |

## 11. Pending Work (preserved from prior plan)

### Track A: Remaining purego Cleanup (after this plan)

- [ ] S88.3.1 Full kernel test suite  Owner: TBD  Est: 2h
  - Dependencies: S88.2.1 (done). Subsumed by E102 work.
- [ ] T89.2 Remove build tags from compute/ GPU files  Owner: TBD  Est: 2h
  - Subsumed by E103.
- [ ] S89.3.1 Cross-platform build verification  Owner: TBD  Est: 1h
  - Subsumed by E107.

### Track B: Megakernel Performance Tuning (after M64)

- [ ] T94.1 Profile megakernel with nsys  Owner: TBD  Est: 2h
- [ ] T94.2 Optimize memory access patterns  Owner: TBD  Est: 3h
- [ ] T94.3 Tune thread block and grid dimensions  Owner: TBD  Est: 2h
- [ ] T94.4 Run golangci-lint  Owner: TBD  Est: 15m
- [ ] T95.1 Profile GPU inference after all optimizations  Owner: TBD  Est: 2h
- [ ] S95.1.1 GPU profile report  Owner: TBD  Est: 30m
- [ ] T95.2 Compare all configurations  Owner: TBD  Est: 1.5h
- [ ] S95.2.1 Benchmark comparison report  Owner: TBD  Est: 30m
- [ ] T95.3 Verify output correctness across all paths  Owner: TBD  Est: 1h
- [ ] S95.3.1 Output correctness report  Owner: TBD  Est: 30m
- [ ] T95.4 Run golangci-lint  Owner: TBD  Est: 15m

### Key Milestones (pending)

| Milestone | Status |
|-----------|--------|
| M58: Megakernel fires | Replaced by M63 in this plan |
| M59: 50 tok/s GPU | PENDING (Track B after M64) |
| M60: 10 tok/s CPU ARM64 | PARTIAL (8.15 tok/s) |
