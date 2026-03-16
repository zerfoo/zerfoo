# Good First Issues

Issues suitable for new contributors across the Zerfoo ecosystem.
Each issue includes the repo, description, acceptance criteria, relevant code, and estimated effort.

---

## 1. Fix `Exp10` returning a constant instead of computing 10^f

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

## 2. Remove doc comment erroneously pasted into `Config.EnableFastMath` field

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

## 3. Add `String()` method to `FloatClass` enum type

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

## 4. Add missing doc comments to GGUF writer `AddMetadata*` methods

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

## 5. Fix `Mod(f, Inf)` returning NaN instead of `f`

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

## 6. Add NaN checks to `addAlgorithmic` and `subAlgorithmic` in float8

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

## 7. Add `SetNormalizer` public method to `BPETokenizer`

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

## 8. Add `String()` methods to `ConversionMode` and `ArithmeticMode` enums

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

## 9. Convert `downloadFile` to use `defer` for resource cleanup

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

## 10. Add table-driven tests for `BFloat16` comparison functions

- **Repo:** float16
- **Labels:** good-first-issue, help-wanted, testing
- **Description:** The `BFloat16Less`, `BFloat16Greater`, `BFloat16LessEqual`, and `BFloat16GreaterEqual` functions in `bfloat16.go` use direct float32 comparison via `ToFloat32()`, which means they return `true` when comparing NaN values (e.g., `BFloat16Less(NaN, 1.0)` would return `false` from float32 but this isn't explicitly tested). Add table-driven tests covering: normal values, zeros (+0/-0), infinities, NaN, subnormals, and mixed signs.
- **Acceptance criteria:**
  - Table-driven test for `BFloat16Less`, `BFloat16Greater`, `BFloat16LessEqual`, `BFloat16GreaterEqual`
  - Test cases include: normal positive/negative, +0 vs -0, +Inf vs -Inf, NaN comparisons, subnormal values
  - All tests pass
- **Relevant code:** `bfloat16.go:383-401`
- **Estimated effort:** 45 minutes
