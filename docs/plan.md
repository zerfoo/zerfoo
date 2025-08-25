# Project Plan: Zerfoo Gradient Strategy Architectural Refactoring

## 1. Context

### 1.1. Problem Statement

The current gradient strategy implementation in the Zerfoo framework is implicit and potentially confusing. The `DefaultBackpropStrategy` exhibits "magic" or surprising behavior, where its gradient computation method changes based on the type of node it is processing (e.g., a standard layer vs. a recurrent layer). This violates the principle of least surprise and makes the framework's behavior difficult to predict and debug. A cleaner, more explicit architecture is needed.

### 1.2. Objectives and Non-Goals

**Objectives:**
- Refactor the gradient computation architecture to be explicit, clear, and directly controlled by the selected `GradientStrategy`.
- Improve the long-term maintainability, clarity, and extensibility of the framework.
- Ensure the behavior of gradient computation is predictable and not dependent on the internal implementation of nodes.

**Non-Goals:**
- This refactoring will not implement new gradient computation features (e.g., full BPTT, Truncated BPTT). It will only establish the architectural foundation for them.
- The public-facing API of the root `zerfoo` package will not be changed.

### 1.3. Constraints and Assumptions

- All work must be implemented in Go, within the existing Zerfoo framework.
- The refactoring must not break existing, validated functionality.
- We assume that making the architecture more explicit will lead to fewer bugs and easier onboarding for new developers.

### 1.4. Success Metrics

- The new architecture using `BackwardMode` is successfully implemented.
- All existing unit and integration tests for the Zerfoo framework pass after the refactoring.
- The `audacity` project can be successfully trained using the new, explicit architecture.
- The code is demonstrably clearer, as measured by peer review.

## 2. Scope and Deliverables

### 2.1. In Scope

- Defining a `BackwardMode` enum in a central `types` package.
- Modifying the `graph.Node` interface to accept a `BackwardMode`.
- Updating the `graph.Graph` implementation to propagate the `BackwardMode`.
- Updating all existing layer implementations to conform to the new `Node` interface.
- Updating the `GradientStrategy` implementations to explicitly control the `BackwardMode`.

### 2.2. Out of Scope

- Implementation of new gradient strategies beyond `FullBackprop` and `OneStepApproximation`.
- Performance optimization of the gradient computation paths.

### 2.3. Deliverables

| ID   | Description                                      |
|------|--------------------------------------------------|
| D1   | Refactored core graph and node interfaces.       |
| D2   | All existing layers updated to the new interface.  |
| D3   | Updated Gradient Strategy implementations.       |
| D4   | A clean, self-contained project plan (this doc). |

## 3. Work Breakdown Structure

### E1: Define Core Types and Refactor Interfaces
- **T1.1: Define `BackwardMode` in `types` Package**
  - **S1.1.1:** Create a new directory and file: `types/backward_mode.go`.
  - **S1.1.2:** Define the `BackwardMode` enum in the `types` package with `FullBackprop` and `OneStepApproximation` values.
  - **S1.1.3:** Add comments explaining each mode.
- **T1.2: Update `graph.Node` Interface**
  - **Dependencies:** T1.1
  - **S1.2.1:** Modify the `Backward` method signature in `graph/node.go` to accept a `types.BackwardMode` parameter.
  - **S1.2.2:** Add `github.com/zerfoo/zerfoo/types` to the imports list in `graph/node.go`.
- **T1.3: Update `graph.Graph` Implementation**
  - **Dependencies:** T1.2
  - **S1.3.1:** Modify the `Backward` method signature in `graph/graph.go` to accept a `types.BackwardMode` parameter.
  - **S1.3.2:** Update the implementation to pass the `mode` down to each node's `Backward` call.
  - **S1.3.3:** Write unit tests to verify the mode is passed correctly.

### E2: Update Layer Implementations
- **T2.1: Update `layers/core`**
  - **Dependencies:** T1.2
  - **S2.1.1:** Update `add.go` `Backward` signature.
  - **S2.1.2:** Update `bias.go` `Backward` signature.
  - **S2.1.3:** Update `cast.go` `Backward` signature.
  - **S2.1.4:** Update `concat.go` `Backward` signature.
  - **S2.1.5:** Update `dense.go` `Backward` signature.
  - **S2.1.6:** Update `ffn.go` `Backward` signature.
  - **S2.1.7:** Update `film.go` `Backward` signature.
  - **S2.1.8:** Update `linear.go` `Backward` signature.
  - **S2.1.9:** Update other `core` layers (`matmul`, `mul`, `reshape`, etc.).
  - **S2.1.10:** Run `go test ./layers/core/...` to ensure compilation and basic test success.
- **T2.2: Update `layers/activations`**
  - **Dependencies:** T1.2
  - **S2.2.1:** Update all activation function layers (`relu`, `sigmoid`, `tanh`, etc.).
  - **S2.2.2:** Run `go test ./layers/activations/...`.
- **T2.3: Update `layers/attention`**
  - **Dependencies:** T1.2
  - **S2.3.1:** Update all attention layers.
  - **S2.3.2:** Run `go test ./layers/attention/...`.
- **T2.4: Update `layers/features`**
  - **Dependencies:** T1.2
  - **S2.4.1:** Update `spectral.go` `Backward` signature.
  - **S2.4.2:** Run `go test ./layers/features/...`.
- **T2.5: Update `layers/recurrent`**
  - **Dependencies:** T1.2
  - **S2.5.1:** Update `rnn.go` `Backward` signature.
  - **S2.5.2:** Implement logic within the RNN `Backward` method to switch behavior based on the `BackwardMode`.
  - **S2.5.3:** Add specific unit tests for each `BackwardMode` in `rnn_test.go`.

### E3: Update Training Strategies
- **T3.1: Update `DefaultBackpropStrategy`**
  - **Dependencies:** T1.3
  - **S3.1.1:** Modify `training/strategy_backprop.go` to call `g.Backward` with the `FullBackprop` mode.
  - **S3.1.2:** Add tests to verify the correct mode is used.
- **T3.2: Update `OneStepApproximationStrategy`**
  - **Dependencies:** T1.3
  - **S3.2.1:** Modify `training/strategy_one_step.go` to call `g.Backward` with the `OneStepApproximation` mode.
  - **S3.2.2:** Add tests to verify the correct mode is used.

### E4: Verification and Cleanup
- **T4.1: Full Project Testing**
  - **Dependencies:** E2, E3
  - **S4.1.1:** Execute the entire `zerfoo` test suite (`go test ./...`).
  - **S4.1.2:** Debug and fix any failing tests introduced by the refactoring.
- **T4.2: Update and Test `audacity`**
  - **Dependencies:** T4.1
  - **S4.2.1:** Update `audacity/cmd/train/main.go` to use one of the explicit strategies.
  - **S4.2.2:** Run the `audacity` training process to ensure it completes successfully.
- **T4.3: Code Formatting and Linting**
  - **Dependencies:** T4.2
  - **S4.3.1:** Run `go fmt ./...` on the `zerfoo` project.
  - **S4.3.2:** Run `golangci-lint run` on the `zerfoo` project and fix any issues.

## 4. Checkable Todo Board

### Not Started
- [ ] T4.2: Update and Test `audacity`  Owner: TBD  Est: 1h
- [ ] T4.3: Code Formatting and Linting  Owner: TBD  Est: 30m

### In Progress
- None

### Blocked
- None

### Done
- [x] T1.1: Define `BackwardMode` in `types` Package
- [x] T1.2: Update `graph.Node` Interface
- [x] T1.3: Update `graph.Graph` Implementation
- [x] T2.1: Update `layers/core` (complete and tested)
- [x] T2.2: Update `layers/activations` (complete and tested)
- [x] T2.3: Update `layers/attention` (complete and tested)
- [x] T2.4: Update `layers/features` (complete and tested)
- [x] T2.5: Update `layers/recurrent` (complete with BackwardMode switching logic)
- [x] T3.1: Update `DefaultBackpropStrategy` (complete and tested)
- [x] T3.2: Update `OneStepApproximationStrategy` (complete and tested)
- [x] T4.1: Full Project Testing (all packages passing)

## 5. Timeline and Milestones

| ID   | Task Description                      | Dependencies | Estimated End Date |
|------|---------------------------------------|--------------|--------------------|
| T1.1 | Define `BackwardMode` in `types` Package | -            | 2025 08 24         |
| T1.2 | Update `graph.Node` Interface         | T1.1         | 2025 08 24         |
| T1.3 | Update `graph.Graph` Implementation   | T1.2         | 2025 08 25         |
| M1   | **Milestone: Core Interfaces Refactored** | T1.3         | **2025 08 25**     |
| T2.x | Update All Layer Implementations      | T1.2         | 2025 08 26         |
| M2   | **Milestone: All Layers Updated**       | T2.x         | **2025 08 26**     |
| T3.x | Update Training Strategies            | T1.3         | 2025 08 27         |
| T4.1 | Full Project Testing                  | T2.x, T3.x   | 2025 08 27         |
| T4.2 | Update and Test `audacity`            | T4.1         | 2025 08 28         |
| T4.3 | Code Formatting and Linting           | T4.2         | 2025 08 28         |
| M3   | **Milestone: Refactoring Complete**     | T4.3         | **2025 08 28**     |

## 6. Operating Procedure

- **Definition of Done:** A task is done when its implementation is complete, it has been tested (unit and integration), the code has been formatted and linted, and it has been reviewed and merged.
- **QA Steps:** All code changes must be reviewed by at least one other team member. All tests must pass in the CI/CD pipeline.
- **Testing:** Always add tests when adding new implementation code. Aim for high test coverage.
- **Formatting:** Always run `go fmt ./...` and `golangci-lint run` after making code changes.
- **Commits:** Never commit files from different directories in the same commit. Make many small, logical commits with clear messages.

## 7. Progress Log

- **2025 08 24: Change Summary**
  - This plan was created to guide the architectural refactoring of Zerfoo's gradient computation strategy. The goal is to move from an implicit, "magic" system to an explicit one controlled by the `GradientStrategy` via a `BackwardMode` enum. This change was prompted by user feedback (you!) pointing out the lack of clarity in the original design. The plan breaks down the required changes to core interfaces (E1), layers (E2), strategies (E3), and verification (E4).

- **2025 08 24: Architectural Pivot**
  - After discussion, the initial plan to place `BackwardMode` in the `training` package was found to be flawed due to circular dependencies. The architecture was updated to introduce a new, dedicated `types` package at the project root. This package will house `BackwardMode` and other future shared types, ensuring a clean separation of concerns and a clear dependency graph (`training` -> `graph` -> `types`).

- **2025 08 24: Progress Update**
  - Completed the core interface refactoring (E1). The `types` package was created, `graph.Node` and `graph.Graph` were updated, and tests were added. Updated layer implementations (E2) for `layers/core`, `layers/activations` and `layers/attention` packages, but many tests are broken due to incorrect test signatures that need to be fixed.

- **2025 08 24: Major Completion**
  - ✅ **CORE BACKWARD METHOD REFACTORING COMPLETE**: All primary layer packages (`layers/activations`, `layers/embeddings`, `layers/features`, `layers/recurrent`) now have the correct `Backward(ctx, mode, grad, ...inputs)` signature and working tests.
  - ✅ **TRAINING STRATEGIES COMPLETE**: Both `DefaultBackpropStrategy` and `OneStepApproximationStrategy` properly use `types.FullBackprop` and `types.OneStepApproximation` respectively.
  - ✅ **RNN BACKWARD MODE LOGIC**: The `SimpleRNN` layer correctly implements different behaviors based on `BackwardMode` - truncating gradients for `OneStepApproximation` and passing them through for `FullBackprop`.
  - ✅ **BFLOAT16 INTEGRATION RESOLVED**: Unified `float16` package now provides both `Float16` and `BFloat16` types, resolving all undefined symbol errors and import issues.

- **2025 08 25: Float16 Package Consolidation**
  - ✅ **UNIFIED FLOAT16 PACKAGE**: Successfully consolidated separate `bfloat16` and `float16` packages into a single unified `float16` package.
  - ✅ **BFLOAT16 IMPLEMENTATION**: Added complete BFloat16 support with arithmetic, conversions, classifications, and cross-conversion with Float16.
  - ✅ **ZERFOO INTEGRATION**: Updated all zerfoo imports to use the unified package, resolving build failures in `model/tensor_decoder.go` and `tensor/tensor.go`.
  - ✅ **MISSING FUNCTIONS**: Implemented all missing functions in float16 package that tests were expecting: `FromFloat64WithMode`, `Parse`, `FromInt`, `ToFloat16`, `ToSlice16`, `shouldRound`, etc.
  - ✅ **TEST FIXES**: Fixed compilation errors and test signature mismatches, achieving successful package compilation.

- **2025 08 24: Test Suite Green**
  - ✅ Ran `go test ./...` with all packages passing after updating remaining tests (`layers/core/linear_extended_test.go`, `layers/core/spectral_fingerprint_test.go`) to the new Backward signature and imports.

## 8. Lessons Learned

### 8.1. Technical Insights

- **Package Consolidation Benefits**: Unifying the `bfloat16` and `float16` packages into a single package eliminated dependency management complexity and provided a consistent API. This approach reduced maintenance overhead and improved developer experience.

- **Interface Refactoring Strategy**: The systematic approach of updating interfaces first, then implementations, then tests proved effective. Starting with the core `graph.Node` interface and propagating changes outward ensured consistency across all layers.

- **BackwardMode Implementation**: The explicit `BackwardMode` parameter successfully eliminated the "magic" behavior in gradient computation. The `SimpleRNN` layer's implementation demonstrates how different gradient strategies can be clearly expressed through the mode parameter.

### 8.2. Project Management Insights

- **Test-Driven Debugging**: Compilation errors served as an excellent guide for missing implementations. Each undefined function error pointed directly to required work, making the refactoring process systematic and measurable.

- **Incremental Validation**: Testing each layer package individually (`./layers/activations/...`, `./layers/embeddings/...`, etc.) allowed for rapid iteration and immediate feedback on fixes.

- **Cross-Package Dependencies**: The original circular dependency issue between `training` and `graph` packages led to the creation of the `types` package. This architectural decision improved the overall dependency structure.

### 8.3. Code Quality Improvements

- **Explicit vs. Implicit Behavior**: The refactoring successfully eliminated implicit gradient computation behavior. The `BackwardMode` enum makes the intended behavior explicit and predictable.

- **Type Safety**: Adding the `types.BackwardMode` parameter to all `Backward` methods provides compile-time guarantees that gradient strategies are explicitly chosen.

- **API Consistency**: The unified float16 package provides consistent naming conventions and API patterns for both `Float16` and `BFloat16` types.

### 8.4. Future Recommendations

- **Testing Strategy**: Future layer implementations should include tests for both `FullBackprop` and `OneStepApproximation` modes to ensure correct behavior switching.

- **Documentation**: The explicit `BackwardMode` system should be documented with examples showing how different gradient strategies affect training behavior.

- **Package Architecture**: The `types` package pattern (shared types at project root) proved effective and should be considered for other shared concerns like error types or configuration structs.

## 9. Hand-off Notes

- **Context:** A new engineer should first read the `Context` section (1) of this plan to understand the problem and objectives. The core task is to refactor the framework to make gradient computation explicit.
- **Entry Point:** The Work Breakdown Structure (WBS) in section 3 provides a clear, hierarchical list of tasks. The `Checkable Todo Board` (section 4) shows the current status of each task.
- **Key Files:** The most important files for this refactoring are `types/backward_mode.go`, `graph/node.go`, `graph/graph.go`, and the `training/strategy_*.go` files.
- **Current Status:** The core backward method refactoring is complete and functional. The unified float16 package resolves all build issues. Remaining work focuses on `audacity` integration and final cleanup.

## 10. Appendix

- **References:**
  - Hierarchical Reasoning Model Paper: `docs/2506.21734v3.md`
