# Zerfoo Issues Remediation Plan

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with a solid computational core (tensors, graph, compute engines,
automatic differentiation, multi-precision support). However, several high-level subsystems are broken
or incomplete. The ZMF model serialization uses string-based placeholder logic that is lossy and
incompatible with the binary decoder. The Transformer block cannot train because its backward pass is
a stub. The CLI predict and tokenize commands return hardcoded fake results. The data and features
packages are polluted with Numerai-specific structs and logic from a separate project. Multiple adapter
methods in the model package are stubs that return errors. HRM modules have low test coverage and are
not integrated as standard graph nodes.

These issues are documented in ISSUES.md (February 2026) and block Zerfoo from being usable as a
production framework.

### Objectives

- O1: Fix ZMF serialization so models round-trip correctly through export and import.
- O2: Implement the Transformer block backward pass so transformer-based models can train.
- O3: Replace hardcoded CLI outputs with real model loading and inference.
- O4: Remove all Numerai/audacity-specific code from the Zerfoo core packages.
- O5: Implement stub adapter methods in the model package.
- O6: Raise HRM module test coverage and standardize integration with the graph builder.

### Non-Goals

- GPU or BLAS compute engine implementation.
- New model architectures beyond what currently exists.
- Distributed training enhancements.
- Performance optimization of the tensor or compute packages.
- Reimplementing Numerai-specific functionality -- that belongs in the audacity/numerapi projects.

### Constraints and Assumptions

- Use Go standard library only. No third-party CLI frameworks (cobra, viper) or test frameworks (testify).
- Use the `flag` package for command-line tools.
- The pre-commit hook rejects commits spanning multiple directories. Each commit must touch files in one directory only.
- All changes must pass `gofmt` and `go vet` before commit.
- Existing binary-safe `model.EncodeTensor` and `model.DecodeTensor` functions in `model/tensor_encoder.go` and `model/tensor_decoder.go` are correct and should be reused.
- The `model.ModelRegistry` and `training.PluginRegistry` patterns are sound and should be leveraged, not replaced.

### Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| ZMF round-trip | Export then import a float32 model with zero data loss | Unit test comparing original and decoded tensor values |
| Transformer backward | Gradient flows through Block.Backward without error | Unit test with synthetic input and gradient check |
| CLI predict | Loads a ZMF model file and produces real predictions | Integration test with a small test model |
| CLI tokenize | Uses a real tokenizer and produces correct token IDs | Integration test with known input/output pairs |
| Data package clean | Zero references to NumeraiRow, era, stock in Zerfoo core | `grep -r "NumeraiRow\|\.Eras\|\.Stocks" data/ features/` returns nothing |
| Audacity references | Zero references to "audacity" in entire codebase | `grep -ri "audacity" .` returns nothing |
| Model adapters | All ModelLoader and ModelExporter methods functional | Unit tests for LoadFromPath, LoadFromReader, LoadFromBytes, ExportToWriter, ExportToBytes |
| HRM coverage | Test coverage above 60% for layers/hrm/ | `go test -cover ./layers/hrm/` |

---

## 2. Scope and Deliverables

### In Scope

- Fix ZMF exporter binary serialization (replace fmt.Sprintf with EncodeTensor).
- Implement TransformerBlock.Backward with gradient propagation through attention, FFN, and normalization.
- Connect CLI predict command to ModelRegistry for real model loading and inference.
- Connect CLI tokenize command to a real tokenizer implementation.
- Extract Numerai-specific code from data/ and features/ into the audacity/numerapi workspace projects.
- Replace Numerai-specific data structures with generic dataset interfaces.
- Implement all stub methods in model/adapters.go.
- Refactor HRM modules to integrate with graph.Builder as standard nodes.
- Remove all remaining "audacity" string references from source and docs.
- Add or update tests for every changed component.

### Out of Scope

- New data format support beyond what is needed to replace Numerai-specific code.
- GPU compute backend.
- New CLI commands beyond fixing predict and tokenize.
- Distributed training changes.
- Documentation website or tutorials.

### Deliverables

| ID | Description | Owner | Acceptance Criteria |
|----|-------------|-------|---------------------|
| D1 | Working ZMF export/import round-trip | TBD | Unit test: export float32 model, import, compare values bit-for-bit |
| D2 | TransformerBlock backward pass | TBD | Unit test: forward then backward with no error, gradients have correct shapes |
| D3 | Functional CLI predict and tokenize | TBD | Integration test: predict loads model, tokenize produces correct IDs |
| D4 | Generic data and features packages | TBD | No Numerai-specific types remain; generic CSV/Parquet loading works |
| D5 | Complete model adapter implementations | TBD | All 7 stub methods replaced with working code, tested |
| D6 | HRM graph integration and tests | TBD | HRM modules usable in graph.Builder; coverage above 60% |
| D7 | Codebase free of audacity references | TBD | grep returns zero matches |

---

## 3. Checkable Work Breakdown

### E1: Fix ZMF Serialization (Critical)

- [ ] T1.1 Replace serializeTensorData with EncodeTensor in zmf_exporter.go  Owner: TBD  Est: 1h
  - Dependencies: None
  - Acceptance: serializeTensorData function at model/zmf_exporter.go:233-247 replaced with call to EncodeTensor from model/tensor_encoder.go
  - Risk: EncodeTensor currently supports float32, float16, and int8 only. If other types are needed, EncodeTensor must be extended first.
  - [ ] S1.1.1 Read model/zmf_exporter.go and model/tensor_encoder.go to confirm type compatibility  Est: 15m
  - [ ] S1.1.2 Replace serializeTensorData body with EncodeTensor call, remove the old function  Est: 30m
  - [ ] S1.1.3 Run gofmt and go vet on model/  Est: 5m
- [ ] T1.2 Add round-trip unit test for ZMF export/import  Owner: TBD  Est: 1h
  - Dependencies: T1.1
  - Acceptance: Test creates a tensor, exports to ZMF bytes, decodes back, asserts values match exactly
  - [ ] S1.2.1 Write TestZMFExporter_RoundTrip in model/zmf_exporter_test.go  Est: 45m
  - [ ] S1.2.2 Run go test ./model/ and verify pass  Est: 15m

### E2: Implement Model Adapter Stubs (Critical)

- [ ] T2.1 Implement ZMFModelLoader.LoadFromPath  Owner: TBD  Est: 1.5h
  - Dependencies: T1.1 (needs working serialization)
  - Location: model/adapters.go:194-198
  - Acceptance: Opens a ZMF file, reads proto, decodes tensors, returns populated Model[T]
  - [ ] S2.1.1 Read model/adapters.go and model/zmf_loader.go to understand the loading pipeline  Est: 15m
  - [ ] S2.1.2 Implement LoadFromPath using existing ZMF proto parsing and DecodeTensor  Est: 45m
  - [ ] S2.1.3 Write TestZMFModelLoader_LoadFromPath in model/adapters_test.go  Est: 30m
  - [ ] S2.1.4 Run gofmt and go vet on model/  Est: 5m
- [ ] T2.2 Implement ZMFModelLoader.LoadFromReader  Owner: TBD  Est: 1h
  - Dependencies: T2.1
  - Location: model/adapters.go:200-202
  - Acceptance: Reads ZMF proto from io.Reader, decodes, returns Model[T]
  - [ ] S2.2.1 Implement LoadFromReader wrapping proto unmarshal from reader  Est: 30m
  - [ ] S2.2.2 Write TestZMFModelLoader_LoadFromReader  Est: 20m
  - [ ] S2.2.3 Run gofmt and go vet on model/  Est: 5m
- [ ] T2.3 Implement ZMFModelLoader.LoadFromBytes  Owner: TBD  Est: 45m
  - Dependencies: T2.2
  - Location: model/adapters.go:205-207
  - Acceptance: Wraps bytes in reader, delegates to LoadFromReader
  - [ ] S2.3.1 Implement LoadFromBytes using bytes.NewReader and LoadFromReader  Est: 15m
  - [ ] S2.3.2 Write TestZMFModelLoader_LoadFromBytes  Est: 20m
  - [ ] S2.3.3 Run gofmt and go vet on model/  Est: 5m
- [ ] T2.4 Implement ZMFModelExporter.ExportToWriter  Owner: TBD  Est: 1h
  - Dependencies: T1.1
  - Location: model/adapters.go:266-267
  - Acceptance: Serializes Model[T] to ZMF proto, writes to io.Writer
  - [ ] S2.4.1 Implement ExportToWriter using ZMFExporter and proto marshal  Est: 30m
  - [ ] S2.4.2 Write TestZMFModelExporter_ExportToWriter  Est: 20m
  - [ ] S2.4.3 Run gofmt and go vet on model/  Est: 5m
- [ ] T2.5 Implement ZMFModelExporter.ExportToBytes  Owner: TBD  Est: 45m
  - Dependencies: T2.4
  - Location: model/adapters.go:271-272
  - Acceptance: Wraps bytes.Buffer, delegates to ExportToWriter, returns bytes
  - [ ] S2.5.1 Implement ExportToBytes using bytes.Buffer and ExportToWriter  Est: 15m
  - [ ] S2.5.2 Write TestZMFModelExporter_ExportToBytes  Est: 20m
  - [ ] S2.5.3 Run gofmt and go vet on model/  Est: 5m
- [ ] T2.6 Implement StandardModelInstance.Backward  Owner: TBD  Est: 1.5h
  - Dependencies: E3 (needs TransformerBlock.Backward)
  - Location: model/adapters.go:67-71
  - Acceptance: Delegates backward to the underlying graph's backward pass
  - [ ] S2.6.1 Implement Backward using graph automatic differentiation  Est: 45m
  - [ ] S2.6.2 Write TestStandardModelInstance_Backward  Est: 30m
  - [ ] S2.6.3 Run gofmt and go vet on model/  Est: 5m

### E3: Implement Transformer Backward Pass (Critical)

- [ ] T3.1 Implement Block.Backward in layers/transformer/block.go  Owner: TBD  Est: 3h
  - Dependencies: None
  - Location: layers/transformer/block.go:134-136
  - Acceptance: Gradient propagates through normalization, attention, FFN, and residual connections in reverse order. Output gradient shapes match input shapes.
  - Risk: Must correctly handle residual connection gradient splitting and RMSNorm backward.
  - [ ] S3.1.1 Read layers/transformer/block.go Forward method to map the computation graph  Est: 30m
  - [ ] S3.1.2 Verify that sub-layers (attention, FFN, norm1, norm2, normPostAttention) each have Backward methods  Est: 15m
  - [ ] S3.1.3 Implement Block.Backward: reverse the forward pass order, propagate gradients through each sub-layer, split gradients at residual additions  Est: 1.5h
  - [ ] S3.1.4 Run gofmt and go vet on layers/transformer/  Est: 5m
- [ ] T3.2 Add backward pass unit tests  Owner: TBD  Est: 1.5h
  - Dependencies: T3.1
  - Acceptance: Test with synthetic input verifies: no error returned, output gradient shapes correct, gradients are non-zero
  - [ ] S3.2.1 Write TestTransformerBlock_Backward in layers/transformer/transformer_block_test.go  Est: 45m
  - [ ] S3.2.2 Write TestTransformerBlock_BackwardShapes verifying gradient dimensions  Est: 30m
  - [ ] S3.2.3 Run go test ./layers/transformer/ and verify pass  Est: 10m

### E4: Fix CLI Predict and Tokenize Commands (Critical)

- [ ] T4.1 Connect PredictCommand to ModelRegistry  Owner: TBD  Est: 2h
  - Dependencies: T2.1 (needs working model loading)
  - Location: cmd/cli/framework.go:308-368
  - Acceptance: PredictCommand.runPrediction loads a model via ModelRegistry, runs forward pass on input data, writes real predictions to output file
  - [ ] S4.1.1 Read cmd/cli/framework.go PredictCommand to understand current flag interface  Est: 15m
  - [ ] S4.1.2 Replace hardcoded numSamples/numFeatures/stats with model.Forward call  Est: 45m
  - [ ] S4.1.3 Replace hardcoded JSON/CSV output with actual prediction values  Est: 30m
  - [ ] S4.1.4 Run gofmt and go vet on cmd/cli/  Est: 5m
- [ ] T4.2 Connect TokenizeCommand to real tokenizer  Owner: TBD  Est: 1.5h
  - Dependencies: None
  - Location: cmd/cli/framework.go:425-431
  - Acceptance: TokenizeCommand uses a tokenizer loaded from vocabulary file, produces correct token IDs for known inputs
  - [ ] S4.2.1 Read existing tokenizer implementations in the codebase  Est: 15m
  - [ ] S4.2.2 Replace sequential ID assignment with actual tokenizer lookup  Est: 45m
  - [ ] S4.2.3 Run gofmt and go vet on cmd/cli/  Est: 5m
- [ ] T4.3 Add CLI integration tests  Owner: TBD  Est: 1.5h
  - Dependencies: T4.1, T4.2
  - Acceptance: Test predict with a small test model file, test tokenize with known vocabulary
  - [ ] S4.3.1 Create a small test model fixture in testdata/  Est: 30m
  - [ ] S4.3.2 Write TestPredictCommand_Integration in cmd/cli/framework_test.go  Est: 30m
  - [ ] S4.3.3 Write TestTokenizeCommand_Integration in cmd/cli/framework_test.go  Est: 20m
  - [ ] S4.3.4 Run go test ./cmd/cli/ and verify pass  Est: 10m

### E5: Remove Numerai/Audacity Pollution from Data and Features (Critical)

- [ ] T5.1 Define generic dataset interfaces in data/  Owner: TBD  Est: 1.5h
  - Dependencies: None
  - Acceptance: data/ exports a generic Dataset interface with methods for iterating rows of numeric features, no Numerai-specific types
  - [ ] S5.1.1 Read data/dataset.go to catalog all Numerai-specific types: NumeraiRow, StockData, EraData, LoadDatasetFromParquet  Est: 15m
  - [ ] S5.1.2 Define generic interfaces: DataSource, Row, FeatureSet  Est: 45m
  - [ ] S5.1.3 Implement a generic CSV data source as the default provider  Est: 30m
  - [ ] S5.1.4 Run gofmt and go vet on data/  Est: 5m
- [ ] T5.2 Remove NumeraiRow and era/stock structures from data/  Owner: TBD  Est: 1h
  - Dependencies: T5.1
  - Acceptance: NumeraiRow struct, StockData, EraData, and LoadDatasetFromParquet function removed from data/dataset.go. No compile errors.
  - [ ] S5.2.1 Delete NumeraiRow, StockData, EraData structs and LoadDatasetFromParquet from data/dataset.go  Est: 30m
  - [ ] S5.2.2 Update data/dataset_test.go to test generic interfaces only  Est: 20m
  - [ ] S5.2.3 Run gofmt and go vet on data/  Est: 5m
- [ ] T5.3 Generalize feature transformers in features/  Owner: TBD  Est: 2h
  - Dependencies: T5.1
  - Location: features/transformers.go
  - Acceptance: LaggedTransformer, RollingTransformer, FFTTransformer operate on generic DataSource interface, not on Eras/Stocks
  - Risk: If these transformers are inherently time-series specific, they may need to be moved to a separate time-series utility package rather than generalized.
  - [ ] S5.3.1 Read features/transformers.go to assess which parts are generalizable vs inherently domain-specific  Est: 15m
  - [ ] S5.3.2 Refactor transformers to accept generic row iterators instead of Dataset.Eras  Est: 1h
  - [ ] S5.3.3 Update features/transformers_test.go to use generic test data  Est: 30m
  - [ ] S5.3.4 Run gofmt and go vet on features/  Est: 5m
- [ ] T5.4 Remove all "audacity" string references from codebase  Owner: TBD  Est: 1h
  - Dependencies: T5.2, T5.3
  - Acceptance: `grep -ri "audacity" .` returns zero results (excluding .git/)
  - [ ] S5.4.1 Search entire codebase for "audacity" references  Est: 10m
  - [ ] S5.4.2 Remove or rewrite each reference  Est: 30m
  - [ ] S5.4.3 Run gofmt and go vet on all modified packages  Est: 10m
  - [ ] S5.4.4 Verify with final grep that zero references remain  Est: 5m
- [ ] T5.5 Add tests for generic data and features packages  Owner: TBD  Est: 1.5h
  - Dependencies: T5.2, T5.3
  - Acceptance: data/ and features/ packages each have test coverage above 70%
  - [ ] S5.5.1 Write TestGenericCSVDataSource loading a small CSV fixture  Est: 30m
  - [ ] S5.5.2 Write TestGenericTransformers with synthetic numeric data  Est: 45m
  - [ ] S5.5.3 Run go test ./data/ ./features/ -cover and verify coverage  Est: 10m

### E6: HRM Module Integration and Coverage (Medium Priority)

- [ ] T6.1 Assess HRM module graph.Node compatibility  Owner: TBD  Est: 1h
  - Dependencies: None
  - Location: layers/hrm/h_module.go, layers/hrm/l_module.go
  - Acceptance: Written assessment of what interface methods are missing for HModule and LModule to work as graph.Node types
  - [ ] S6.1.1 Read graph.Node interface definition  Est: 15m
  - [ ] S6.1.2 Read HModule and LModule to identify missing methods  Est: 15m
  - [ ] S6.1.3 Document the gap and implementation approach  Est: 30m
- [ ] T6.2 Implement graph.Node interface on HRM modules  Owner: TBD  Est: 2h
  - Dependencies: T6.1, T3.1 (HRM wraps transformer block, needs working backward)
  - Acceptance: HModule and LModule implement graph.Node and can be added to a graph.Builder
  - [ ] S6.2.1 Add missing interface methods to HModule  Est: 45m
  - [ ] S6.2.2 Add missing interface methods to LModule  Est: 45m
  - [ ] S6.2.3 Run gofmt and go vet on layers/hrm/  Est: 5m
- [ ] T6.3 Add comprehensive HRM tests  Owner: TBD  Est: 1.5h
  - Dependencies: T6.2
  - Acceptance: layers/hrm/ test coverage above 60%
  - [ ] S6.3.1 Write TestHModule_Forward, TestHModule_Backward in h_module_test.go  Est: 30m
  - [ ] S6.3.2 Write TestLModule_Forward, TestLModule_Backward in l_module_test.go  Est: 30m
  - [ ] S6.3.3 Write TestHRM_GraphBuilder_Integration testing HRM in a graph  Est: 20m
  - [ ] S6.3.4 Run go test ./layers/hrm/ -cover and verify 60%+ coverage  Est: 10m

### E7: Final Verification

- [ ] T7.1 Run full test suite  Owner: TBD  Est: 30m
  - Dependencies: E1, E2, E3, E4, E5, E6
  - Acceptance: `go test ./...` passes with zero failures
  - [ ] S7.1.1 Run go test ./... and capture output  Est: 20m
  - [ ] S7.1.2 Fix any test failures discovered  Est: varies
- [ ] T7.2 Run linters and formatters  Owner: TBD  Est: 30m
  - Dependencies: T7.1
  - Acceptance: `gofmt -l .` returns no files; `go vet ./...` returns no issues
  - [ ] S7.2.1 Run gofmt -l . and fix any formatting issues  Est: 10m
  - [ ] S7.2.2 Run go vet ./... and fix any issues  Est: 15m
- [ ] T7.3 Verify success metrics  Owner: TBD  Est: 30m
  - Dependencies: T7.2
  - Acceptance: All metrics from the Success Metrics table above are met
  - [ ] S7.3.1 Run ZMF round-trip test  Est: 5m
  - [ ] S7.3.2 Run transformer backward test  Est: 5m
  - [ ] S7.3.3 Run CLI integration tests  Est: 5m
  - [ ] S7.3.4 Run grep for NumeraiRow and audacity  Est: 5m
  - [ ] S7.3.5 Check HRM coverage report  Est: 5m

---

## 4. Timeline and Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|--------------|---------------|
| M1 | ZMF Serialization Fixed | E1 | Round-trip test passes: export then import with zero data loss |
| M2 | Model Adapters Complete | E2, E1 | All 7 stub methods implemented and tested |
| M3 | Transformer Training Enabled | E3 | Block.Backward propagates gradients, test passes |
| M4 | CLI Functional | E4, E2 | predict and tokenize produce real outputs from model files |
| M5 | Data Package Clean | E5 | Zero Numerai/audacity references, generic interfaces tested |
| M6 | HRM Integrated | E6, E3 | HRM modules work as graph nodes, 60%+ coverage |
| M7 | All Issues Resolved | E7 | Full test suite passes, all success metrics met |

### Recommended Sequence

1. E1 (ZMF serialization) -- no dependencies, unblocks E2
2. E3 (Transformer backward) -- no dependencies, unblocks E6
3. E2 (Model adapters) -- depends on E1, unblocks E4
4. E5 (Data cleanup) -- no dependencies, can run in parallel with E2/E3
5. E4 (CLI) -- depends on E2
6. E6 (HRM) -- depends on E3
7. E7 (Final verification) -- depends on all above

Parallelism: E1, E3, and E5 can all start simultaneously. E2 and E5 can proceed in parallel once E1 is done.

---

## 5. Operating Procedure

### Definition of Done

A task is done when:
1. The implementation is complete and compiles without errors.
2. Unit tests exist and pass for the new or changed code.
3. `gofmt` produces no changes on the affected files.
4. `go vet` reports no issues on the affected package.
5. The task's acceptance criteria are met.

### Review and QA Steps

1. Read the relevant source files before making changes.
2. Write or update tests before or alongside implementation.
3. Run `go test ./package/...` after each change.
4. Run `gofmt -w .` and `go vet ./...` before committing.
5. Verify acceptance criteria with a concrete test or check.

### Commit Discipline

- Never commit files from different directories in the same commit. The pre-commit hook will reject it.
- Make small, logical commits -- one task or subtask per commit.
- Never allow changes to pile up. Commit after each completed subtask.
- Write descriptive commit messages referencing the task ID (e.g., "fix(model): replace serializeTensorData with EncodeTensor [T1.1]").

---

## 6. Progress Log

- **2026 02 24:** Plan created to address all issues documented in ISSUES.md. Merged with existing audacity removal plan. Epics E1-E7 defined covering ZMF serialization (E1), model adapters (E2), transformer backward (E3), CLI (E4), data cleanup including audacity removal (E5), HRM integration (E6), and final verification (E7). Previous audacity-only tasks (old E1, E2) absorbed into E5 (T5.4) and E7. No implementation progress yet.

---

## 7. Hand-off Notes

### For a New Contributor

- **Start here:** Read ISSUES.md for the problem overview, then this plan for the solution.
- **Codebase entry points:**
  - `model/zmf_exporter.go` -- ZMF serialization (E1)
  - `model/tensor_encoder.go` and `model/tensor_decoder.go` -- Binary-safe encode/decode functions to reuse
  - `model/adapters.go` -- Stub methods to implement (E2)
  - `layers/transformer/block.go` -- Transformer backward pass (E3)
  - `cmd/cli/framework.go` -- CLI commands (E4)
  - `data/dataset.go` and `features/transformers.go` -- Numerai cleanup (E5)
  - `layers/hrm/` -- HRM modules (E6)
- **Key patterns:** The codebase uses Go generics with a `tensor.Numeric` type constraint. Model and training registries use factory functions. All interfaces are defined in `model/interfaces.go` and `training/interfaces.go`.
- **Testing:** Mock implementations exist in `model/interfaces_test.go` and `training/interfaces_test.go`. Reuse these patterns for new tests.
- **Build/test:** `go test ./...` from the repo root. `gofmt -w .` and `go vet ./...` for formatting and linting.
- **No credentials required.** All work is local.

---

## 8. Appendix

### References

- ISSUES.md -- Source issues report (February 2026)
- model/tensor_encoder.go -- EncodeTensor (supports float32, float16, int8 via binary.LittleEndian)
- model/tensor_decoder.go -- DecodeTensor (supports float32, float16, bfloat16, int8 with type conversion)
- model/model_registry.go -- ModelRegistry with factory pattern for providers, loaders, exporters, validators, optimizers
- training/registry.go -- PluginRegistry with factory pattern for workflows, data providers, model providers, sequence providers, metric computers, cross validators

### Archived Tasks (from Previous Plan)

The following tasks from the previous "Remove Audacity References" plan (2025 11 17) have been absorbed into this plan:

| Old ID | Old Label | New Location | Reason |
|--------|-----------|--------------|--------|
| T1.1 | Remove audacity refs from cmd/cli/framework.go | E5 T5.4 | Merged into comprehensive audacity removal |
| T1.2 | Remove audacity refs from cmd/cli/framework_test.go | E5 T5.4 | Merged into comprehensive audacity removal |
| T1.3 | Remove audacity refs from cmd/zerfoo-train/main.go | E5 T5.4 | cmd/zerfoo-train already deleted per recent commits |
| T1.4 | Remove audacity refs from integration/config_lock_integration_test.go | E5 T5.4 | File already deleted per recent commits |
| T1.5 | Remove audacity refs from training/interfaces_doc.go | E5 T5.4 | Merged into comprehensive audacity removal |
| T2.1-T2.3 | Verification tasks | E7 | Merged into final verification epic |
