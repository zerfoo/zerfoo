# Fix: RMSNorm Backward Pass Nil Pointer Dereference

## 1. Context

### Problem Statement

RMSNorm.Backward() in `layers/normalization/rmsnorm.go` uses `r.rms` and
`r.inputTensor` (cached forward-pass tensors) without nil checks. If Backward
is called before Forward, or if Forward fails partway through, the method panics
with a nil pointer dereference. This blocks downstream training workloads that
use RMSNorm (all modern transformer architectures: Llama, Mistral, Gemma, etc.).

The sibling class `SimplifiedLayerNormalization` already has the correct
defensive pattern at lines 152-154:

```go
if sln.invStdDev == nil || sln.normalizedInput == nil {
    return nil, errors.New("backward called before forward: missing cached tensors")
}
```

RMSNorm is missing this check. The `r.rms` field (line 23) is a nil pointer
from struct initialization. Forward sets it in three code paths (lines 131,
147, 178). Backward uses it in four places (lines 203, 240, 245, 250) with
no guard.

### Objectives

- O1: Prevent nil pointer panic in RMSNorm.Backward() when forward caches are missing.
- O2: Add regression test that fails without the fix.
- O3: Ensure all normalization layer backward passes follow the same defensive pattern.

### Non-Goals

- Refactoring RMSNorm forward pass logic.
- Changing the gradient computation math.
- Performance optimization of the backward pass.

### Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Nil safety | RMSNorm.Backward returns error, not panic | Unit test: call Backward without Forward |
| Regression test | Test fails without fix, passes with fix | `go test -run TestRMSNormBackwardNilCache` |
| Build clean | Zero errors, zero warnings | `go build ./...` and `go vet ./...` |
| Existing tests pass | No regressions | `go test ./layers/normalization/ -race` |

---

## 2. Scope and Deliverables

### In Scope

- Add nil check for `r.rms` and `r.inputTensor` at top of Backward method.
- Add regression test in `rmsnorm_test.go` or a new `rmsnorm_backward_test.go`.
- Run linter and existing tests.

### Out of Scope

- Changes to Forward pass.
- Changes to other normalization layers (they either have the check or lack Backward).
- Training infrastructure changes.

---

## 3. Checkable Work Breakdown

### E1: Fix RMSNorm Backward Nil Safety

- [ ] T1.1 Add nil guard at top of RMSNorm.Backward  Owner: TBD  Est: 15m
  - At `rmsnorm.go` line 199 (after input validation, before first use of `r.rms`),
    add:
    ```go
    if r.rms == nil || r.inputTensor == nil {
        return nil, fmt.Errorf("RMSNorm: backward called before forward: missing cached tensors")
    }
    ```
  - Acceptance: `go build ./layers/normalization/` passes.

- [ ] T1.2 Add regression test for Backward-before-Forward  Owner: TBD  Est: 30m
  - Deps: T1.1
  - Create table-driven test in `rmsnorm_test.go` (or `rmsnorm_backward_test.go`):
    1. Construct RMSNorm with NewRMSNorm (do NOT call Forward).
    2. Call Backward with a valid dOut tensor.
    3. Assert error is returned (not nil), error message contains "backward called before forward".
    4. Assert no panic.
  - Also test the normal path: Forward then Backward succeeds without error.
  - Use the standard `testing` package, table-driven style. No testify.
  - Acceptance: `go test -run TestRMSNormBackward -v ./layers/normalization/` passes.

- [ ] T1.3 Run `go build ./...` and `go vet ./...`  Owner: TBD  Est: 10m
  - Deps: T1.1
  - Acceptance: zero errors from `go build`, zero new warnings from `go vet`.

- [ ] T1.4 Run `go test ./layers/normalization/ -race -count=1`  Owner: TBD  Est: 10m
  - Deps: T1.1, T1.2
  - Acceptance: all tests pass, no race conditions, no regressions.

---

## 4. Parallel Work

This is a small fix. All tasks are sequential (T1.1 -> T1.2 -> T1.3/T1.4).

**Wave 1:** T1.1
**Wave 2:** T1.2, T1.3 (parallel -- test and build/vet are independent)
**Wave 3:** T1.4

---

## 5. Timeline and Milestones

| ID | Milestone | Exit Criteria | Deps |
|----|-----------|---------------|------|
| M1 | Fix applied | Nil guard added, builds clean | T1.1, T1.3 |
| M2 | Tests pass | Regression test + existing tests pass | T1.2, T1.4 |

---

## 6. Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|-----------|-----------|
| R1 | Backward test requires specific engine setup that is hard to construct in isolation | Low | Medium | Use CPUEngine with small tensors (e.g., [2, 3, 4] shape). SimplifiedLayerNormalization backward test is a reference. |

---

## 7. Operating Procedure

### Definition of Done

1. `go build ./...` passes in zerfoo repo.
2. `go test ./layers/normalization/ -race` passes with zero failures.
3. New test `TestRMSNormBackwardNilCache` exists and passes.
4. Commit uses Conventional Commits: `fix(normalization): prevent nil panic in RMSNorm.Backward`.

---

## 8. Progress Log

### 2026-03-17: Plan created

**Change summary:** Created plan to fix RMSNorm backward nil pointer dereference.
Bug confirmed by code inspection: `r.rms` used in 4 places without nil check.
SimplifiedLayerNormalization has the correct pattern to follow.

---

## 9. Hand-off Notes

- **Bug location:** `layers/normalization/rmsnorm.go`, Backward method (line 195+).
- **Nil fields:** `r.rms` (line 23) and `r.inputTensor` (line 22).
- **Reference pattern:** `layers/normalization/simplified_layer_normalization.go` lines 152-154.
- **Reference test:** `layers/normalization/simplified_layer_normalization_backward_test.go`.
- **Downstream impact:** Blocks any training workload using RMSNorm (all Llama/Mistral/Gemma architectures).
