# pkg.go.dev Runnable Examples

## Context

Zerfoo has no `func Example` test functions. pkg.go.dev shows a "Run" button
for Example functions, letting users try the API in the Go Playground. The
Playground has: ~256MB RAM, ~10s CPU, no GPU, no network, no filesystem, Linux
amd64 only. Real model loading is impossible — we need stub-based examples.

### Problem
Users visiting pkg.go.dev/github.com/zerfoo/zerfoo cannot try any code.
There are zero Example functions in the codebase.

### Objectives
- Add 9 Example functions covering all major API features
- All examples runnable on pkg.go.dev (Go Playground)
- All examples pass `go test -run Example`
- Deterministic output (required for go test verification)

### Non-goals
- Actual model inference in examples (impossible on Playground)
- New third-party dependencies

### Success Metrics
- `go test -run Example ./...` passes
- All 9 examples appear on pkg.go.dev with working "Run" button

---

## Discovery Summary

### Current State
- Zero `func Example` functions in the repo
- `api_test.go` uses internal `generateFunc` field for testing
- `newTestModelWithEmbeddings()` helper exists in test file
- Grammar, tool call detection, and embeddings work without model files

### Approach: Exported `NewModel` Constructor
Add `NewModel(fn)` to the public API so examples can use `package zerfoo_test`
(external test package), enabling the pkg.go.dev "Run" button. Internal
package examples compile but the Run button fails for unexported symbols.

Decision rationale: docs/adr/070-pkg-godev-example-constructor.md

---

## Checkable Work Breakdown

### E1: Public Test Constructor

- [ ] T1.1 Add NewModel constructor to api.go  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: api.go (or zerfoo.go — wherever Model is defined)
  Add: `func NewModel(fn func(ctx context.Context, prompt string) (string, error)) *Model`
  Returns `&Model{generateFunc: fn}`. One function, 3 lines.
  Acceptance: `go build ./...` passes. Function appears in `go doc`.

- [ ] T1.2 Add NewModelWithEmbeddings constructor  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: api.go
  For embedding examples. Wraps the existing newTestModelWithEmbeddings pattern
  into an exported function that returns a Model with a fake tokenizer and
  embedding weights.
  Acceptance: `go build ./...` passes.

### E2: Core API Examples

- [ ] T2.1 ExampleModel_Chat  Owner: TBD  Est: 0.5h  verifies: [UC-002]
  File: example_test.go (package zerfoo_test)
  Construct Model with NewModel, call m.Chat, print result.
  // Output: Hello! I'm a language model.
  Acceptance: `go test -run ExampleModel_Chat` passes.

- [ ] T2.2 ExampleModel_Generate  Owner: TBD  Est: 0.5h  verifies: [UC-004]
  File: example_test.go
  Generate with WithMaxTokens, WithTemperature. Print result.
  Acceptance: deterministic output, test passes.

- [ ] T2.3 ExampleModel_ChatStream  Owner: TBD  Est: 0.5h  verifies: [UC-003]
  File: example_test.go
  Stream tokens, print each. Verify channel closes.
  Acceptance: deterministic output, test passes.

- [ ] T2.4 ExampleEmbedding_CosineSimilarity  Owner: TBD  Est: 0.5h  verifies: [UC-005]
  File: example_test.go
  Create two Embedding structs with known vectors.
  Print similarity with %.4f.
  Acceptance: exact numeric output matches.

- [ ] T2.5 ExampleModel_Embed  Owner: TBD  Est: 0.5h  verifies: [UC-005]
  File: example_test.go
  Use NewModelWithEmbeddings, call m.Embed([]string{"hello", "world"}).
  Print dimension and similarity.
  Acceptance: deterministic output, test passes.

### E3: Advanced API Examples

- [ ] T3.1 ExampleModel_Generate_toolCalling  Owner: TBD  Est: 0.5h  verifies: [UC-011]
  File: example_test.go
  NewModel returning JSON tool call text.
  Call with WithTools(weatherTool).
  Print ToolCalls[0].FunctionName.
  Acceptance: prints "get_weather".

- [ ] T3.2 ExampleModel_Generate_structuredOutput  Owner: TBD  Est: 0.5h  verifies: [UC-012]
  File: example_test.go
  NewModel returning valid JSON.
  Call with WithSchema(personSchema).
  Print result.Text.
  Acceptance: prints valid JSON.

### E4: Sub-Package Examples

- [ ] T4.1 ExampleDetectToolCall (serve package)  Owner: TBD  Est: 0.5h  verifies: [UC-011]
  File: serve/example_test.go (package serve_test)
  Call serve.DetectToolCall with JSON string and tools.
  Print result.
  Acceptance: test passes, appears on pkg.go.dev/serve.

- [ ] T4.2 ExampleGrammar (grammar package)  Owner: TBD  Est: 0.5h  verifies: [UC-012]
  File: generate/grammar/example_test.go (package grammar_test)
  Create JSONSchema, Convert to Grammar, Advance through valid JSON.
  Print IsComplete().
  Acceptance: prints "true".

### E5: Verification

- [ ] T5.1 Run all examples with go test  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Run: `go test -run Example -v ./...`
  All 9 examples must pass.
  Acceptance: zero failures.

- [ ] T5.2 Run go vet on example files  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Run: `go vet ./...`
  Acceptance: clean.

---

## Parallel Work

### Tracks
| Track | Tasks | Description |
|-------|-------|-------------|
| A: Constructors | T1.1, T1.2 | Public test model constructors |
| B: Core Examples | T2.1-T2.5 | Main API examples (deps on A) |
| C: Advanced Examples | T3.1-T3.2 | Tool calling + structured output (deps on A) |
| D: Sub-Package Examples | T4.1, T4.2 | serve + grammar examples (no deps on A) |
| E: Verification | T5.1, T5.2 | Final check (deps on B, C, D) |

### Waves

#### Wave 1: Constructors + Independent Examples (4 agents)
- [ ] T1.1 NewModel constructor
- [ ] T1.2 NewModelWithEmbeddings constructor
- [ ] T4.1 ExampleDetectToolCall (serve, no deps)
- [ ] T4.2 ExampleGrammar (grammar, no deps)

#### Wave 2: All API Examples (7 agents)
- [ ] T2.1 ExampleModel_Chat  Deps: T1.1
- [ ] T2.2 ExampleModel_Generate  Deps: T1.1
- [ ] T2.3 ExampleModel_ChatStream  Deps: T1.1
- [ ] T2.4 ExampleEmbedding_CosineSimilarity  (no deps)
- [ ] T2.5 ExampleModel_Embed  Deps: T1.2
- [ ] T3.1 ExampleModel_Generate_toolCalling  Deps: T1.1
- [ ] T3.2 ExampleModel_Generate_structuredOutput  Deps: T1.1

#### Wave 3: Verification (2 agents)
- [ ] T5.1 Run all examples  Deps: all above
- [ ] T5.2 Run go vet  Deps: all above

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | pkg.go.dev Run button may not work for some examples | Medium | Low | Use external test package + exported constructors |
| R2 | NewModel exposes test-only API surface | Low | Certain | Document clearly as test/demo utility |
| R3 | Example output may drift if API changes | Low | Medium | go test -run Example catches drift automatically |

---

## Operating Procedure

### Definition of Done
1. `go test -run Example -v ./...` passes
2. `go vet ./...` clean
3. `go build ./...` clean
4. All examples have `// Output:` comments
5. PR merged, CI green

---

## Progress Log

### 2026-03-28: Plan created
- Created plan for 9 pkg.go.dev Example functions
- Identified need for NewModel exported constructor (ADR 070)
- 3 waves, 13 tasks total
