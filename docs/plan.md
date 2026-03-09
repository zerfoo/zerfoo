# Zerfoo Development Plan -- Documentation Cleanup and Code Commit

## 1. Context

### Problem Statement

The `feat/neon-softmax` branch has 31 commits ahead of origin with a mix of
uncommitted code and documentation. The docs/ directory has accumulated 12
markdown files totaling ~244KB, many of which are historical phase logs or
superseded proposals. The current plan.md (44KB) contains extensive detail on
completed work (Track 0, Track C waves C1-C4, Track D) that should be archived
to design.md. Binary build artifacts are untracked and polluting git status.

The branch needs to be cleaned up, committed, pushed, and merged.

### Objectives

- O1: Trim obsolete docs (suggestion.md, suggestion-plan.md, phase27.md,
  phase28.md) by removing them.
- O2: Consolidate docs/updates.md content into design.md, then remove it.
- O3: Archive docs/zerfoo-suggestions.md or integrate actionable items.
- O4: Commit all uncommitted code changes in small, single-directory commits.
- O5: Add binary artifacts to .gitignore.
- O6: Push branch and create PR to upstream (zerfoo/zerfoo).
- O7: Trim plan.md to contain only pending work.

### Non-Goals

- Refactoring any code (conv1d composition fix is separate work).
- DGX Spark integration tasks (T100.2, T100.3, T100.4).
- Performance tuning (Track B).

### Constraints and Assumptions

- Pre-commit hook rejects multi-directory commits.
- Must commit files from one directory per commit.
- Branch: feat/neon-softmax.
- PR target: zerfoo/zerfoo (upstream), head: dndungu:feat/neon-softmax.

### Success Metrics

| Metric | Target | How Measured |
|--------|--------|-------------|
| docs/ file count | <= 7 (from 12) | ls docs/*.md |
| docs/ total size | < 120KB (from 244KB) | du -sh docs/ |
| Untracked files | 0 (excluding build dirs) | git status |
| Uncommitted changes | 0 | git status |
| PR created | 1 | gh pr list |

---

## 2. Scope and Deliverables

### In Scope

| ID | Deliverable | Rationale |
|----|-------------|-----------|
| D1 | Remove 5 obsolete docs | Reduce noise, single source of truth |
| D2 | Trim plan.md | Remove completed epics, keep only pending work |
| D3 | Update design.md | Absorb stable knowledge from removed docs |
| D4 | Commit all code changes | Clean working tree |
| D5 | .gitignore update | Stop binary artifacts from cluttering status |
| D6 | PR to upstream | Get changes reviewed and merged |

### Out of Scope

- Code refactoring (conv1d composition, variable_selection GELU fix).
- New feature work.
- DGX Spark testing.

---

## 3. Checkable Work Breakdown

### E1: Trim Obsolete Documentation

- [ ] T1.1 Remove docs/suggestion.md  Owner: TBD  Est: 5m
  - Phase 26 proposal, fully superseded by design.md Phase 26 section.
  - Acceptance: File deleted. No broken references in remaining docs.
  - Dependencies: none.

- [ ] T1.2 Remove docs/suggestion-plan.md  Owner: TBD  Est: 5m
  - Phase 27-29 closure plan. All work documented in design.md.
  - Acceptance: File deleted. No broken references.
  - Dependencies: none.

- [ ] T1.3 Remove docs/phase27.md  Owner: TBD  Est: 5m
  - Historical execution log. Relevant facts already in design.md.
  - Acceptance: File deleted.
  - Dependencies: none.

- [ ] T1.4 Remove docs/phase28.md  Owner: TBD  Est: 5m
  - Historical execution log.
  - Acceptance: File deleted.
  - Dependencies: none.

- [ ] T1.5 Remove docs/updates.md  Owner: TBD  Est: 10m
  - Phase 34 execution log. Relevant Wave C1-C5 facts already in plan.md
    progress log and design.md section 15.16.
  - Acceptance: File deleted. Key facts preserved in design.md.
  - Dependencies: none.

- [ ] T1.6 Remove docs/zerfoo-suggestions.md  Owner: TBD  Est: 5m
  - Numerai training improvement proposals. training/optimizer/ema.go already
    implemented. Remaining proposals (SWA, SGDR) are future work.
  - Acceptance: File deleted.
  - Dependencies: none.

- [ ] T1.7 Commit docs removals  Owner: TBD  Est: 5m
  - Single commit: `docs: remove obsolete phase logs and proposals`
  - Acceptance: git status shows no deleted files. Commit passes pre-commit.
  - Dependencies: T1.1-T1.6.

### E2: Trim plan.md

- [ ] T2.1 Remove completed Track 0 (E96) from plan.md  Owner: TBD  Est: 15m
  - All 14 tasks complete. Summary already in design.md 15.16.
  - Remove the entire "Track 0: Remaining Composition Fixes" section.
  - Keep a one-line note in Context: "Track 0 complete (see design.md 15.16)."
  - Acceptance: No E96/T96 tasks in plan.md.
  - Dependencies: none.

- [ ] T2.2 Condense Track C completed work  Owner: TBD  Est: 20m
  - Replace detailed T97/T98/T99 task descriptions with a completion summary.
  - Keep only pending tasks: T100.2, T100.3, T100.4, S100.1.1, S100.2.1.
  - Keep Track B (E94, E95) as-is (all pending).
  - Track A (E87-E89): keep as-is (all pending).
  - Acceptance: plan.md < 15KB. All pending tasks preserved with acceptance
    criteria.
  - Dependencies: none.

- [ ] T2.3 Consolidate progress log  Owner: TBD  Est: 10m
  - Replace v10-v17 entries with single summary entry.
  - Add today's entry for this cleanup work.
  - Acceptance: Progress log has 2 entries: archive summary + today.
  - Dependencies: T2.1, T2.2.

- [ ] T2.4 Commit trimmed plan.md  Owner: TBD  Est: 5m
  - Commit: `docs: trim plan.md -- archive completed tracks to design.md`
  - Acceptance: Commit passes pre-commit.
  - Dependencies: T2.1-T2.3.

### E3: Update design.md

- [ ] T3.1 Stage and commit unstaged docs/design.md changes  Owner: TBD  Est: 5m
  - 143 lines of Phase 34 Track D (NEON SIMD) documentation already written.
  - Commit: `docs: add Phase 34 Track D NEON SIMD documentation to design.md`
  - Acceptance: docs/design.md committed. Content includes section 15.16.
  - Dependencies: none.

### E4: Commit Code Changes

All code changes are on feat/neon-softmax. Pre-commit hook requires
single-directory commits.

- [ ] T4.1 Commit staged normalization files  Owner: TBD  Est: 5m
  - Files: layers/normalization/rmsnorm.go,
    layers/normalization/simplified_layer_normalization.go
  - Already staged. These capture scales output from FusedRMSNormGPU.
  - Commit: `fix(normalization): cache GPU RMSNorm scales for backward pass`
  - Dependencies: none.

- [ ] T4.2 Commit ADRs 020-029  Owner: TBD  Est: 5m
  - 7 new ADR files in docs/adr/.
  - All complete and ready to commit.
  - Commit: `docs: add ADRs 020-029 (Q4 dot, worker pool, GPU-first, purego, megakernel, composition, tracing, NEON)`
  - Dependencies: none.

- [ ] T4.3 Commit new layer files  Owner: TBD  Est: 5m
  - Files: layers/core/conv1d.go, layers/core/conv1d_test.go,
    layers/core/temporal_conv_encoder.go,
    layers/core/temporal_conv_encoder_test.go,
    layers/core/variable_selection.go,
    layers/core/variable_selection_test.go
  - Commit: `feat(layers): add Conv1D, TemporalConvEncoder, VariableSelection layers`
  - Dependencies: none.

- [ ] T4.4 Commit training/optimizer/ema.go  Owner: TBD  Est: 5m
  - EMA optimizer wrapper. Clean implementation.
  - Commit: `feat(training): add EMA optimizer wrapper`
  - Dependencies: none.

- [ ] T4.5 Commit remaining compute/gpuapi changes  Owner: TBD  Est: 10m
  - Unstaged changes in compute/ and internal/gpuapi/ related to RMSNorm
    scales and kernel runner updates.
  - Review diffs, group by directory, commit separately.
  - Dependencies: T4.1.

### E5: Clean Up Build Artifacts

- [ ] T5.1 Add binary artifacts to .gitignore  Owner: TBD  Est: 5m
  - Add: softmax_arm64.o, bench-compare, bench_tps, coverage-gate,
    zerfoo-predict
  - Commit: `chore: add build artifacts to .gitignore`
  - Dependencies: none.

- [ ] T5.2 Remove softmax_arm64.o (0 bytes, empty)  Owner: TBD  Est: 2m
  - Delete the file.
  - Dependencies: T5.1.

### E6: Push and Create PR

- [ ] T6.1 Push feat/neon-softmax to origin  Owner: TBD  Est: 2m
  - `git push origin feat/neon-softmax`
  - Dependencies: E1-E5 all complete.

- [ ] T6.2 Create PR to upstream  Owner: TBD  Est: 5m
  - `gh pr create --repo zerfoo/zerfoo --head dndungu:feat/neon-softmax`
  - Title: "feat: tracing compiler, NEON SIMD, composition fixes, docs cleanup"
  - Body: Summary of all tracks (0, C, D) and docs cleanup.
  - Dependencies: T6.1.

---

## 4. Parallel Work

| Track | Tasks | Notes |
|-------|-------|-------|
| Track 1: Doc cleanup | T1.1-T1.7, T2.1-T2.4 | Can run in parallel with code commits |
| Track 2: Code commits | T4.1-T4.5 | Sequential (each commit must pass pre-commit) |
| Track 3: Artifacts | T5.1-T5.2 | Independent |
| Track 4: Design doc | T3.1 | Independent |

Sync point: All tracks converge at T6.1 (push).

---

## 5. Timeline and Milestones

| Milestone | Dependencies | Exit Criteria |
|-----------|-------------|---------------|
| M1: Docs trimmed | E1, E2, E3 | docs/ has <= 7 files, plan.md < 15KB |
| M2: Code committed | E4, E5 | git status clean (no modified/untracked) |
| M3: PR created | M1, M2 | PR URL returned, CI passes |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Pre-commit hook rejects multi-dir commit | Commit fails | Medium | Verify each commit touches only one directory |
| R2 | Unstaged gpuapi changes conflict with staged normalization | Merge conflict | Low | Commit normalization first, then gpuapi |
| R3 | Removing docs loses unpreserved knowledge | Information loss | Low | Review each doc before deleting; verify key facts in design.md |

---

## 7. Operating Procedure

### Definition of Done

A task is done when:
1. File changes match acceptance criteria.
2. Commit passes pre-commit hooks.
3. Single directory per commit.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Use Conventional Commits format.
- Run linters before committing.

---

## 8. Progress Log

### Change Summary -- 2026-03-09

Plan created for documentation cleanup and code commit work. Identified 5
obsolete docs for removal (suggestion.md, suggestion-plan.md, phase27.md,
phase28.md, updates.md, zerfoo-suggestions.md). Identified 4 groups of
uncommitted code changes. Binary artifacts need .gitignore entries.

---

## 9. Hand-off Notes

### For a New Contributor

- **This plan:** Covers only the cleanup and commit work for feat/neon-softmax.
- **Active development plan:** After this cleanup, the remaining Phase 34 work
  is in this same plan (pending tasks T100.2-T100.4, Track A, Track B).
- **PR workflow:** PRs go to zerfoo/zerfoo (upstream), not dndungu/zerfoo.
  Use `gh pr create --repo zerfoo/zerfoo --head dndungu:<branch>`.

### Pending Phase 34 Work (preserved from previous plan)

#### Track A: Remaining purego Cleanup

- [ ] T87.3 Replace runtime.go CGo functions with dlopen calls  Owner: TBD  Est: 3h
- [ ] S87.3.1 Runtime function parity test  Owner: TBD  Est: 1.5h
- [ ] S88.2.1 Elementwise kernel parity test  Owner: TBD  Est: 1.5h
- [ ] S88.3.1 Full kernel test suite  Owner: TBD  Est: 2h
- [ ] T89.2 Remove build tags from compute/ GPU files  Owner: TBD  Est: 2h  BLOCKED
- [ ] S89.3.1 Cross-platform build verification  Owner: TBD  Est: 1h

#### Track C: Tracing Compiler Integration (remaining)

Waves C1-C4 complete (E97, E98 code complete, E99 emitters complete, T100.1
wired). See design.md section 15.16 and docs/adr/028-tracing-compiler.md.

- [ ] S100.1.1 Integration test on DGX Spark  Owner: TBD  Est: 1.5h
  - Run bench_tps on DGX Spark. Verify "megakernel: compiled and loaded" log.
  - Dependencies: DGX Spark access.

- [ ] T100.2 Update tryCompileMegakernel for GPU KV cache  Owner: TBD  Est: 2h
  - Detect KVCache* ops in traced tape. Allocate GPUKVCache. Pass device
    pointers to runner Launch(). Pass seq_pos from Go KV cache.
  - Dependencies: T100.1, DGX Spark.

- [ ] S100.2.1 KV cache integration test  Owner: TBD  Est: 1.5h
  - Generate 50 tokens with megakernel. Compare with plan.Run() output.

- [ ] T100.3 End-to-end megakernel correctness test  Owner: TBD  Est: 2h
  - Load Gemma 3 2B Q4, generate 50 tokens, compare megakernel vs plan.Run().
  - Dependencies: T100.2.

- [ ] T100.4 Run golangci-lint on generate/, graph/  Owner: TBD  Est: 15m
  - Dependencies: T100.3.

#### Track B: Megakernel Performance Tuning

- [ ] T94.1 Profile megakernel with nsys  Owner: TBD  Est: 2h
  - Dependencies: E100 complete.
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

### Performance Baselines

| Config | tok/s | Source |
|--------|-------|--------|
| CPU ARM64 (post Track D) | 8.15 median | Phase 34 Track D |
| CPU ARM64 (pre Track D) | 6.86 | Phase 30 |
| GPU (cuda) | 10.32 peak / 7.78 median | Phase 33 |
| Ollama GB10 | ~100 (est.) | Interpolated |

### Key Milestones

| Milestone | Status |
|-----------|--------|
| M56: Tracing compiler works | DONE (CompileTraced produces primitive-op tape) |
| M57: GPU KV cache works | DONE (code complete, integration pending) |
| M58: Megakernel fires | PENDING (T100.2, T100.3 on DGX Spark) |
| M59: 50 tok/s GPU | PENDING (Track B after M58) |
| M60: 10 tok/s CPU ARM64 | PARTIAL (8.15 tok/s, needs GEMM cache tiling) |

### Risk Register (Phase 34)

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R92 | Register pressure: hidden_dim=2048 | Must tile, slower | High | Profile with nvcc --ptxas-options=-v |
| R95 | KV cache reads limit bandwidth | Cannot reach max | High | Focus on short contexts (<512) |
| R98 | 50 tok/s not achieved | Unknown bottleneck | Medium | If 30+, profile; if <30, fused kernels |
| R100 | Tracing captures wrong path | Wrong megakernel output | Medium | Only use for decode (seqLen=1) |
| R101 | Tensor identity via pointer fragile | Wrong slot wiring | Medium | Disable pooling during tracing |
| R102 | GPU KV cache memory budget | OOM for long contexts | Low | Default 512 tokens (~104MB) |
| R103 | EngineProxy dispatch overhead | Small regression | Low | ~1us total, negligible |
