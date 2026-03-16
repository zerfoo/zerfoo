# Good First Issues

Issues suitable for new contributors across the Zerfoo ecosystem.
Each issue includes the repo, description, acceptance criteria, relevant code, and estimated effort.

## Beginner

These issues require minimal domain knowledge and are great for getting familiar with the codebase.

---

### 1. Fix `Exp10` returning a constant instead of computing 10^f

- **Repo:** float16
- **Labels:** good-first-issue, help-wanted, bug
- **Description:** `Exp10(f)` ignores its argument and always returns `FromFloat32(10)` (the value 10.0). It should compute `10^f` using `math.Pow(10, f)`, similar to how `Exp` and `Exp2` are implemented.
- **Acceptance criteria:**
  - `Exp10(FromFloat32(0))` returns 1.0
  - `Exp10(FromFloat32(1))` returns 10.0
  - `Exp10(FromFloat32(2))` returns 100.0 (clamped to Float16 max if needed)
  - Special cases (NaN, Inf, -Inf, zero) handled consistently with `Exp` and `Exp2`
  - Tests added for the above cases
- **Relevant code:** `math.go:125-128`
- **Estimated effort:** 30 minutes

---

### 2. Remove doc comment erroneously pasted into `Config.EnableFastMath` field

- **Repo:** float16
- **Labels:** good-first-issue, help-wanted, bug
- **Description:** The `Config` struct in `float16.go` has the entire package-level doc comment accidentally pasted after the `EnableFastMath` field on the same line (lines 67-105). This makes `EnableFastMath` appear to have a massive inline comment and causes GoDoc to render incorrectly. The duplicated comment should be removed, leaving only a short field comment.
- **Acceptance criteria:**
  - `EnableFastMath` has a clean one-line comment (e.g., `// EnableFastMath enables fast-math optimizations`)
  - The duplicated package doc block (lines 68-104) is removed
  - `go vet ./...` passes
  - GoDoc renders cleanly for the `Config` type
- **Relevant code:** `float16.go:62-105`
- **Estimated effort:** 15 minutes

---

### 3. Add `String()` method to `FloatClass` enum type

- **Repo:** float16
- **Labels:** good-first-issue, help-wanted, enhancement
- **Description:** The `FloatClass` type (used by `FpClassify` / `Class()`) is a bare `int` enum with no `String()` method. When printed in test output or debug logs, it shows as an opaque integer. Add a `String() string` method that returns human-readable names like `"PositiveZero"`, `"NegativeInfinity"`, `"QuietNaN"`, etc.
- **Acceptance criteria:**
  - `FloatClass.String()` returns a descriptive name for all 10 class values
  - Unknown values return `"Unknown(N)"` where N is the integer
  - Test verifying all class values round-trip through `String()`
- **Relevant code:** `types.go:170-184`
- **Estimated effort:** 30 minutes

---

### 4. Add missing doc comments to GGUF writer `AddMetadata*` methods

- **Repo:** zonnx
- **Labels:** good-first-issue, help-wanted, documentation
- **Description:** The `Writer` type in `pkg/gguf/writer.go` has several exported methods (`AddMetadataString`, `AddMetadataUint32`, `AddMetadataInt32`, `AddMetadataFloat32`, `AddMetadataBool`, `AddMetadataStringArray`) that lack GoDoc comments. Each should have a one-line doc comment describing what it does, consistent with Go conventions.
- **Acceptance criteria:**
  - All six `AddMetadata*` methods have GoDoc comments
  - Comments follow Go convention: `// MethodName does X.`
  - `golint` / `go vet` pass with no exported-without-comment warnings
- **Relevant code:** `pkg/gguf/writer.go:64-86`
- **Estimated effort:** 20 minutes

---

### 5. Add `String()` methods to `ConversionMode` and `ArithmeticMode` enums

- **Repo:** float8
- **Labels:** good-first-issue, help-wanted, enhancement
- **Description:** The `ConversionMode` and `ArithmeticMode` enum types in `types.go` lack `String()` methods. When these values appear in error messages or debug output (e.g., from `DebugInfo()`), they show as integers. Add `String() string` methods to both types.
- **Acceptance criteria:**
  - `ConversionMode.String()` returns `"Default"`, `"Strict"`, or `"Fast"`
  - `ArithmeticMode.String()` returns `"Auto"`, `"Algorithmic"`, or `"Lookup"`
  - Unknown values return `"Unknown(N)"`
  - Tests verifying all known values
- **Relevant code:** `types.go:52-73`
- **Estimated effort:** 30 minutes

---

### 6. Add table-driven tests for `BFloat16` comparison functions

- **Repo:** float16
- **Labels:** good-first-issue, help-wanted, testing
- **Description:** The `BFloat16Less`, `BFloat16Greater`, `BFloat16LessEqual`, and `BFloat16GreaterEqual` functions in `bfloat16.go` use direct float32 comparison via `ToFloat32()`, which means they return `true` when comparing NaN values (e.g., `BFloat16Less(NaN, 1.0)` would return `false` from float32 but this isn't explicitly tested). Add table-driven tests covering: normal values, zeros (+0/-0), infinities, NaN, subnormals, and mixed signs.
- **Acceptance criteria:**
  - Table-driven test for `BFloat16Less`, `BFloat16Greater`, `BFloat16LessEqual`, `BFloat16GreaterEqual`
  - Test cases include: normal positive/negative, +0 vs -0, +Inf vs -Inf, NaN comparisons, subnormal values
  - All tests pass
- **Relevant code:** `bfloat16.go:383-401`
- **Estimated effort:** 45 minutes

---

## Intermediate

These issues involve understanding the layer/op system or require changes across multiple files.

---

### 7. Fix `Mod(f, Inf)` returning NaN instead of `f`

- **Repo:** float16
- **Labels:** good-first-issue, help-wanted, bug
- **Description:** According to IEEE 754, `Mod(finite, +/-Inf) = finite` (the dividend is returned unchanged). Currently, lines 425-426 of `math.go` return `QuietNaN` when either operand is infinite. The check `f.IsInf(0) || divisor.IsInf(0)` should be split: `Mod(Inf, y)` is NaN, but `Mod(x, Inf)` where x is finite should return `x`.
- **Acceptance criteria:**
  - `Mod(FromFloat32(3.0), PositiveInfinity)` returns `FromFloat32(3.0)`
  - `Mod(PositiveInfinity, FromFloat32(3.0))` returns NaN
  - `Mod(PositiveInfinity, PositiveInfinity)` returns NaN
  - Tests added for these cases
- **Relevant code:** `math.go:414-433`
- **Estimated effort:** 30 minutes

---

### 8. Add NaN checks to `addAlgorithmic` and `subAlgorithmic` in float8

- **Repo:** float8
- **Labels:** good-first-issue, help-wanted, bug
- **Description:** The `mulAlgorithmic` and `divAlgorithmic` functions in `arithmetic.go` correctly check for NaN operands and return NaN. However, `addAlgorithmic` and `subAlgorithmic` skip NaN checks entirely, so `Add(NaN, x)` falls through to the float32 conversion path. While this happens to work (float32 NaN propagates), it is inconsistent and could mask issues if the conversion path changes. Add explicit NaN checks at the top of both functions for consistency and correctness.
- **Acceptance criteria:**
  - `Add(NaN, x)` explicitly returns NaN before any float32 conversion
  - `Sub(NaN, x)` explicitly returns NaN before any float32 conversion
  - `Add(x, NaN)` and `Sub(x, NaN)` likewise return NaN
  - Tests added verifying NaN propagation for add and sub
- **Relevant code:** `arithmetic.go:150-207`
- **Estimated effort:** 30 minutes

---

### 9. Add `SetNormalizer` public method to `BPETokenizer`

- **Repo:** ztoken
- **Labels:** good-first-issue, help-wanted, enhancement
- **Description:** The `BPETokenizer` has a `normalizer` field that is set internally during `LoadFromJSON`, but there is no public API to set a custom normalizer when constructing a tokenizer programmatically (e.g., via `NewBPETokenizer`). Add a `SetNormalizer(fn NormalizerFunc)` method, consistent with the existing `SetSentencePiece` and `SetSpecialTokenStrings` methods.
- **Acceptance criteria:**
  - `SetNormalizer(fn NormalizerFunc)` method added to `BPETokenizer`
  - Passing `nil` disables normalization
  - Test verifying that a custom normalizer (e.g., `strings.ToLower`) is applied during `Encode`
- **Relevant code:** `bpe.go:36-37`, `bpe.go:63-69`, `loader.go:87`
- **Estimated effort:** 30 minutes

---

### 10. Convert `downloadFile` to use `defer` for resource cleanup

- **Repo:** zonnx
- **Labels:** good-first-issue, help-wanted, code-quality
- **Description:** The `downloadFile` function in `pkg/downloader/downloader.go` manually closes `resp.Body` and the output file in multiple error branches (lines 80-118), which is repetitive and error-prone. Refactor to use `defer` for closing both resources, consistent with Go idioms and the pattern already used in `DownloadModel`.
- **Acceptance criteria:**
  - `resp.Body.Close()` is deferred immediately after the nil-error check
  - `out.Close()` is deferred immediately after file creation
  - Close errors are still checked (using named return values or a deferred closure)
  - All manual close calls in error branches are removed
  - Existing tests still pass
- **Relevant code:** `pkg/downloader/downloader.go:59-119`
- **Estimated effort:** 45 minutes

---

### 11. Add unit tests for `Div`, `Sqrt`, and `Neg` layers

- **Repo:** zerfoo
- **Labels:** good-first-issue, help-wanted, testing
- **Description:** The `layers/core/` package has `div.go`, `sqrt.go`, and `neg.go` which implement element-wise `Div`, `Sqrt`, and `Neg` graph nodes. None of them have corresponding `_test.go` files. Add table-driven unit tests for each, verifying forward pass behavior, input count validation, and the `OpType()` / `Attributes()` / `Parameters()` accessors.
- **Acceptance criteria:**
  - `div_test.go` — tests `Div.Forward` with valid inputs, division by zero, and wrong input count
  - `sqrt_test.go` — tests `Sqrt.Forward` with positive values, zero, and wrong input count
  - `neg_test.go` — tests `Neg.Forward` with positive, negative, and zero values
  - All three test files verify `OpType()` returns the correct string
  - Tests use the existing mock engine pattern from `layers/core/core_test.go`
- **Relevant code:** `layers/core/div.go`, `layers/core/sqrt.go`, `layers/core/neg.go`, `layers/core/core_test.go`
- **Estimated effort:** 1 hour

---

### 12. Add unit tests for `Softmax` activation layer

- **Repo:** zerfoo
- **Labels:** good-first-issue, help-wanted, testing
- **Description:** The `layers/activations/softmax.go` file has no corresponding test file. Add tests that verify the forward pass output sums to 1.0 (within tolerance), that the axis attribute is correctly returned, and that wrong input count returns an error.
- **Acceptance criteria:**
  - `softmax_test.go` created in `layers/activations/`
  - Test that `Forward` output sums to ~1.0 along the configured axis
  - Test that `OpType()` returns `"Softmax"` and `Attributes()` contains `"axis"`
  - Test that passing 0 or 2+ inputs to `Forward` returns an error
  - Test that `BuildSoftmax` correctly parses the `"axis"` attribute (int and int64)
- **Relevant code:** `layers/activations/softmax.go`, `layers/activations/comprehensive_test.go`
- **Estimated effort:** 45 minutes

---

### 13. Optimize `RecordRequest` to avoid per-token counter increment loop

- **Repo:** zerfoo
- **Labels:** good-first-issue, help-wanted, performance
- **Description:** In `serve/metrics.go`, `RecordRequest` increments `tokensTotal` in a `for range tokens` loop, calling `Inc()` once per token. For high-throughput requests generating hundreds of tokens, this is unnecessarily slow. Add an `Add(n int64)` method to the `CounterMetric` interface (and its `InMemoryCollector` implementation in ztensor) and use it here, or simply call `Inc()` once and multiply, depending on the collector API.
- **Acceptance criteria:**
  - `tokensTotal` is incremented by the token count in a single operation, not a loop
  - The metrics endpoint still reports the correct cumulative token count
  - Existing `serve/metrics_test.go` tests pass
- **Relevant code:** `serve/metrics.go:36-47`
- **Estimated effort:** 45 minutes

---

## Advanced

These issues require deeper understanding of the ML framework, GPU pipelines, or distributed systems.

---

### 14. Implement `Backward` pass for the `Gelu` activation's test coverage

- **Repo:** zerfoo
- **Labels:** help-wanted, testing
- **Description:** The `Gelu` activation in `layers/activations/gelu.go` has a full `Backward` implementation (lines 113-226), but no test file exists for it. Write tests that verify the backward pass gradient is numerically correct by comparing against finite-difference approximation. This ensures the analytical gradient formula matches the actual function behavior.
- **Acceptance criteria:**
  - `gelu_test.go` created in `layers/activations/`
  - Forward pass test: verify GELU(0) ~ 0, GELU(large positive) ~ input, GELU(large negative) ~ 0
  - Backward pass test: compare analytical gradient against `(GELU(x+eps) - GELU(x-eps)) / (2*eps)` for several x values
  - Gradient error < 1e-3 relative tolerance
- **Relevant code:** `layers/activations/gelu.go:47-111` (Forward), `layers/activations/gelu.go:113-226` (Backward)
- **Estimated effort:** 1.5 hours

---

### 15. Add JSON Schema `$ref` resolution to the grammar-constrained decoding converter

- **Repo:** zerfoo
- **Labels:** help-wanted, enhancement
- **Description:** The `generate/grammar/converter.go` rejects schemas containing `$ref` with an error. Many real-world JSON Schemas use `$ref` for reusable definitions (e.g., `"$ref": "#/$defs/Address"`). Add support for resolving local `$ref` references within the same schema document, enabling constrained decoding with more complex schemas.
- **Acceptance criteria:**
  - `Convert` resolves local `$ref` values (e.g., `#/$defs/Foo`) by looking up the referenced definition in the schema's `$defs` map
  - Circular references are detected and return an error
  - Test with a schema that uses `$ref` for a nested object type
  - Test that circular `$ref` returns a clear error
- **Relevant code:** `generate/grammar/converter.go:22-29`, `generate/grammar/schema.go`
- **Estimated effort:** 2 hours

---

### 16. Add a fine-tuning example application

- **Repo:** zerfoo
- **Labels:** help-wanted, documentation, enhancement
- **Description:** The `examples/` directory has examples for inference, chat, embedding, RAG, API server, and JSON output, but no example for fine-tuning/training. Add an `examples/fine-tune/` directory with a `main.go` that demonstrates loading a small model, running a few training steps with the `training.Trainer`, and saving the result. This would showcase the training API documented in the README.
- **Acceptance criteria:**
  - `examples/fine-tune/main.go` created with a working example
  - `examples/fine-tune/README.md` explains what the example does and how to run it
  - Example uses the `training.Trainer` and `training/optimizer` packages
  - Example compiles with `go build ./examples/fine-tune/`
- **Relevant code:** `training/trainer.go`, `training/interfaces.go`, `training/optimizer/optimizer.go`, `examples/inference/main.go` (reference pattern)
- **Estimated effort:** 2 hours

---

### 17. Implement `Backward` for `Div` and `Sqrt` core layers

- **Repo:** zerfoo
- **Labels:** help-wanted, enhancement
- **Description:** The `Div` and `Sqrt` nodes in `layers/core/` return `"backward not implemented"` errors. Implementing their backward passes would enable gradient computation through these operations during training. `Div` backward: `d/da(a/b) = 1/b`, `d/db(a/b) = -a/b^2`. `Sqrt` backward: `d/dx(sqrt(x)) = 1/(2*sqrt(x))`.
- **Acceptance criteria:**
  - `Div.Backward` returns gradients for both inputs using the quotient rule
  - `Sqrt.Backward` returns the gradient using `1/(2*sqrt(x))`
  - Tests verify gradients against finite-difference approximation (relative error < 1e-3)
  - Edge cases: division by zero gradient, sqrt of zero gradient
- **Relevant code:** `layers/core/div.go:36-38`, `layers/core/sqrt.go:36-38`
- **Estimated effort:** 2 hours
