# Zerfoo Project Plan (Comprehensive)

This document provides a consolidated, comprehensive, and prioritized plan for the development of the Zerfoo project, based on all outstanding tasks in the codebase.

## 1. Understanding the Goal

The goal of this plan is to provide a single, actionable, and exhaustive roadmap for the Zerfoo project. It consolidates all TODOs from various files into a single document, prioritized by urgency, without losing the granularity of the original tasks.

## 2. Investigation & Analysis

The plan was formulated after a thorough review of the following files:
- `todos/architecture.md`
- `todos/inspect.md`
- `todos/lint.md`
- `todos/pending.md`
- `todos/performance.md`
- `todos/tokenizer.md`
- `todos.md`
- `review.md`
- `dirt.md`

The analysis confirmed that the major architectural refactoring to separate `zerfoo`, `zmf`, and `zonnx` is complete. The remaining tasks are focused on implementation, performance, and code quality.

## 3. Proposed Strategic Approach

The work is broken down into the following phases, in order of priority:

### Phase 1: Urgent - Blocking Issues & Critical Bugs

These tasks are critical for the stability, correctness, and core architecture of the framework.

-   **[ ] Architecture & Cleanup**
    -   **[ ] Remove CGo Dependency from `zonnx`**: As recommended in the architecture review, this is a critical cleanup to ensure the project is a pure Go solution.
        -   [ ] Remove the file `internal/onnx/frontend.go`.
        -   [ ] Remove the `github.com/yalue/onnxruntime_go` dependency from `zonnx/go.mod`.

-   **[ ] Core Engine Correctness**
    -   **[ ] Fix Softmax axis handling bug (`CPUEngine.Softmax`)**: A buggy core operator is a critical issue.
        -   [ ] Rewrite to compute max/sum along the specified axis, not globally.
        -   [ ] Use the numerically stable algorithm: `softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))`.
        -   [ ] Add comprehensive tests comparing against NumPy/SciPy.

-   **[ ] Training Functionality**
    -   **[ ] Implement Missing `Backward` Passes**: Without these, the framework cannot be used for training.
        -   [ ] `layers/normalization/simplified_layer_normalization.go`: Implement `SimplifiedLayerNormalization.Backward()`
        -   [ ] `layers/attention/attention_head.go`: Implement `AttentionHead.Backward()`
        -   [ ] `layers/core/concat.go`: Implement `Backward()` to split gradients.
        -   [ ] `layers/core/rotary_embedding.go`: Implement `Backward()` for RoPE.
        -   [ ] `layers/core/polynomial.go`: Improve `Backward()` to compute exact gradients.

### Phase 2: High - Core Functionality & Performance

These tasks are essential for making the framework usable and performant.

-   **[ ] Performance Optimizations**
    -   **[ ] Add Benchmarks First**: Before modifying any kernel, add focused benchmarks to capture a baseline.
    -   **[ ] Low-Level Engine Optimizations (`compute/cpu_engine.go`)**
        -   [ ] **`MatMul` BLAS Integration**: Replace the naive triple-loop `MatMul` with a BLAS-accelerated version (e.g., using `gonum/blas/gonum`).
        -   [ ] **Parallelize Element-wise Operations**: Parallelize loops in `binaryOp` and `UnaryOp` using goroutines.
        -   [ ] **Optimize `Transpose`**: Rewrite to avoid coordinate calculations in the loop.
        -   [ ] **Optimize `Sum` and `ReduceMean`**: Parallelize for large tensors.
        -   [ ] **Eliminate `BroadcastIndex` Hotspot**: Precompute stride mappings once per `binaryOp` call.
    -   **[ ] Tensor and Graph Optimizations**
        -   [ ] **Eliminate Redundant Tensor Methods**: Deprecate and remove arithmetic methods on `tensor.TensorNumeric`.
        -   [ ] **Optimize Tensor Views**: Avoid creating a new slice and copying data in `Data()`.
        -   [ ] **Cache Graph Parameters**: Collect all trainable parameters once and cache them.
    -   **[ ] Layer-Level Optimizations**
        -   [ ] **Optimize `Dense` Layer**: Remove reshaping logic in `Forward` and `Backward`.
        -   [ ] **Introduce Fused Operations**: Create fused operations like `MatMul + Add`.

-   **[ ] Core Functionality**
    -   **[ ] Implement `zonnx download` and Fix Tokenization**: This is crucial for the end-to-end usability of the Gemma 3 model.
        -   [ ] Create a new `download` command in `zonnx`.
        -   [ ] Use the new command to re-download the `gemma3` model.
        -   [ ] Integrate the `sugarme/tokenizer` into the `gemma3` example.
    -   **[ ] Implement Missing Core Layer Functionalities (`Forward` passes)**
        -   [ ] `layers/attention/group_query_attention.go`: Replace stub with real implementation.
        -   [ ] `layers/core/concat.go`: Implement actual concatenation.
        -   [ ] `layers/core/rotary_embedding.go`: Implement RoPE.

### Phase 3: Medium - Code Quality & Developer Experience

These tasks will improve the long-term maintainability of the codebase and the developer experience.

-   **[ ] Fix Linting Issues (`todos/lint.md`)**
    -   [ ] **High-priority fixes (revive: doc comments, stutter, unused params)**
        -   [ ] `layers/attention/attention_head.go`
        -   [ ] `layers/attention/global_attention.go`
        -   [ ] `layers/attention/local_attention.go`
        -   [ ] `layers/attention/scaled_dot_product_attention.go`
        -   [ ] `layers/normalization/simplified_layer_normalization.go`
        -   [ ] `layers/normalization/skip_simplified_layer_normalization.go`
        -   [ ] `layers/reducesum/reducesum.go`
        -   [ ] `layers/transformer/block.go`
        -   [ ] `layers/transpose/transpose.go`
        -   [ ] `model/builder.go`
        -   [ ] `tensor/tensor.go`
        -   [ ] `testing/testutils/test_helpers.go`
        -   [ ] `model/builder_graph_test.go`
    -   [ ] **Revive: empty-block in tests**
        -   [ ] `layers/embeddings/rotary_positional_embedding_test.go`
        -   [ ] `layers/embeddings/token_embedding_test.go`
    -   [ ] **Complexity and style (gocyclo, gocritic)**
    -   [ ] **Security (gosec)**

-   **[ ] Improve `zonnx inspect` Command (`todos/inspect.md`)**
    -   [ ] Add `--type` flag (`onnx` or `zmf`).
    -   [ ] Implement inspection with no `zerfoo` imports.
    -   [ ] Standardize JSON output.
    -   [ ] Add `--pretty` flag.

-   **[ ] Implement CI/CD and Automation (`todos/architecture.md`)**
    -   [ ] Add CI Architecture Guards.
    -   [ ] Add `make` targets for common tasks.
    -   [ ] Implement a pre-commit hook.

### Phase 4: Low - Documentation & Nice-to-haves

-   **[ ] Improve Documentation (`todos/architecture.md`)**
    -   [ ] Document architectural principles in READMEs.
    -   [ ] Update all documentation to reflect the new architecture.
    -   [ ] Create a comprehensive contributor's guide.

## 4. Verification Strategy

The success of this plan will be measured by the following:
-   All tasks in this plan are completed.
-   All tests pass, and code coverage remains high.
-   The `gemma3` example runs end-to-end and produces meaningful output.
-   The performance of the framework is significantly improved, as measured by the benchmarks.
-   The codebase is clean, well-documented, and easy to maintain.

## 5. Anticipated Challenges & Considerations

-   **Performance Tuning:** Performance optimization can be a time-consuming process. It will be important to focus on the most significant bottlenecks first.
-   **API Stability:** As the framework is still under development, some APIs may change. It will be important to communicate these changes clearly to users.
-   **Community Contributions:** As the project grows, it will be important to establish clear guidelines for community contributions.
