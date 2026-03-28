# GPT-2 Architecture Support

## Context

Zerfoo needs GPT-2 support to load TinyStories-656K (~1MB GGUF), enabling
real inference in pkg.go.dev runnable examples. GPT-2 is the architecture
used by the smallest open-weight GGUF models available.

### Problem
The smallest GGUF models (TinyStories-656K, tiny-gpt2) use GPT-2 architecture
which Zerfoo does not support. Without it, pkg.go.dev examples must use stubs.

### Objectives
- Load and run GPT-2 GGUF models (TinyStories-656K, GPT-2 small)
- Reuse existing GQA + KV cache infrastructure (no duplicate attention code)
- Enable pkg.go.dev runnable examples with real inference

### Non-goals
- GPT-2 training support
- GPT-2 CUDA graph capture optimization
- Large GPT-2 model support (GPT-2 XL, etc.)

### GPT-2 vs Llama Differences

| Feature | Llama | GPT-2 |
|---------|-------|-------|
| Position encoding | RoPE (rotary) | Learned absolute embeddings |
| Normalization | RMSNorm | LayerNorm (with bias) |
| Activation | SiLU (SwiGLU gated) | GELU |
| FFN structure | 3 matrices (gate+up+down) | 2 matrices (up+down) |
| KV heads | GQA | MHA (kv_heads == heads) |
| Attention bias | No | Yes |
| LM head tying | Optional | Always tied to token_embd |

### Existing Components (no changes needed)
- GELU: layers/activations/gelu.go
- LayerNorm: layers/normalization/layer_normalization.go
- GGUF loader: handles all quant types, generic arch prefix parsing
- Embedding lookup: embeddingLookupNode in arch_llama.go

---

## Checkable Work Breakdown

### E1: Infrastructure Changes

- [x] T1.1 Make RoPE optional in GQA  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: layers/attention/grouped_query_attention.go
  Add nil checks around RoPE usage (SetPositionOffset, GetAngles).
  When RoPE is nil, Q and K pass through without rotation.
  Acceptance: GQA works with rope=nil. Existing tests still pass.

- [x] T1.2 Add position_embd.weight to GGUF tensor name mapping  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: model/gguf/arch.go
  Add "position_embd.weight" to globalTensorMap.
  Add output_norm.bias handling for architectures with LayerNorm bias.
  Acceptance: MapTensorName("gpt2", "position_embd.weight") returns mapped name.

- [x] T1.3 Add GPT-2 config parser  Owner: TBD  Est: 0.5h  verifies: [UC-001]
  File: inference/arch_config.go
  Parse GPT-2 config: n_embd -> HiddenSize, n_layer -> NumLayers,
  n_head -> NumQueryHeads=NumKVHeads, n_positions -> MaxPositionEmbeddings.
  IntermediateSize defaults to 4*HiddenSize.
  Register in DefaultArchConfigRegistry.
  Acceptance: parseGPT2Config returns correct metadata for TinyStories config.

### E2: GPT-2 Graph Builder

- [x] T2.1 Implement GPT-2 graph builder  Owner: TBD  Est: 3h  verifies: [UC-001]
  File: inference/arch_gpt2.go (new)
  Graph: TokenEmbed + PosEmbed -> [LayerNorm -> GQA(rope=nil) -> ResidualAdd
  -> LayerNorm -> Linear+GELU+Linear -> ResidualAdd] x N -> LayerNorm -> LMHead

  Key details:
  - gpt2EmbeddingNode: token lookup + position embedding addition with posOffset
  - Reuse GQA with rope=nil for causal self-attention + KV cache
  - 2-matrix FFN: Linear(hidden, ff_dim) -> GELU -> Linear(ff_dim, hidden)
  - LayerNorm with bias (not RMSNorm)
  - Pre-norm: normalize before attention/FFN, residual after
  - LM head tied to token_embd.weight if output.weight absent
  - Position offset tracking for autoregressive decode

  Acceptance: buildGPT2Graph returns valid graph. Forward produces [batch, seq, vocab].

- [x] T2.2 Register "gpt2" architecture  Owner: TBD  Est: 0.5h  verifies: [UC-001]
  File: inference/registry_init.go
  Add: RegisterArchitecture("gpt2", buildGPT2Graph)
  Acceptance: GetArchitecture("gpt2") returns non-nil builder.

### E3: Tests

- [x] T3.1 Unit tests for GPT-2 builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  File: inference/arch_gpt2_test.go (new)
  Tests: (1) Build graph with synthetic tensors (2 layers, 64 hidden, 2 heads).
  (2) Forward pass produces correct output shape. (3) Position offset works
  for autoregressive decode. (4) Tied embeddings work when output.weight absent.
  Acceptance: All tests pass with -race.

- [x] T3.2 Tensor name mapping tests  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: model/gguf/arch_test.go (add to existing)
  Test MapTensorName for "gpt2" with all GPT-2 tensor names.
  Acceptance: All mappings correct.

- [ ] T3.3 Integration test with TinyStories-656K  Owner: TBD  Est: 1h  verifies: [UC-001]
  File: tests/parity/gpt2_tinystories_test.go (new)
  Download TinyStories-656K-Q4_K_M GGUF, load via inference.LoadFile,
  generate 10 tokens. Verify: loads without error, output is English text.
  Gate behind env var (ZERFOO_TEST_MODELS=1) for CI.
  Acceptance: Model loads and generates coherent children's story text.

- [x] T3.4 Run go vet and linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Run: go build ./... && go vet ./... && go test ./inference/... -race
  Acceptance: Clean build, clean vet, all tests pass.

### E4: Documentation

- [x] T4.1 Update README with GPT-2 architecture  Owner: TBD  Est: 0.5h  delivers: [README update]
  Add GPT-2 row to architecture table. Update count to 25 (14 families).
  Acceptance: README reflects GPT-2 support.

---

## Parallel Work

### Tracks
| Track | Tasks | Description |
|-------|-------|-------------|
| A: GQA fix | T1.1 | Make RoPE optional |
| B: Tensor mapping | T1.2, T3.2 | GGUF name mapping |
| C: Config parser | T1.3 | GPT-2 metadata parsing |
| D: Graph builder | T2.1, T2.2 | Core implementation (deps A, B, C) |
| E: Tests | T3.1, T3.3, T3.4 | Verification (deps D) |
| F: Docs | T4.1 | README (deps D) |

### Waves

#### Wave 1: Foundation (3 agents)
- [x] T1.1 Make RoPE optional in GQA
- [x] T1.2 Add position_embd.weight to tensor mapping
- [x] T1.3 Add GPT-2 config parser

#### Wave 2: Core Implementation (2 agents)
- [x] T2.1 Implement GPT-2 graph builder  Deps: T1.1, T1.2, T1.3
- [x] T2.2 Register "gpt2" architecture  Deps: T2.1

#### Wave 3: Verification (4 agents)
- [x] T3.1 Unit tests for GPT-2 builder  Deps: T2.2
- [x] T3.2 Tensor name mapping tests  Deps: T1.2
- [ ] T3.3 Integration test with TinyStories  Deps: T2.2
- [x] T3.4 Run go vet and linters  Deps: T2.2
- [x] T4.1 Update README  Deps: T2.2

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | GQA tightly coupled to RoPE | High | Medium | Nil checks are minimal; fallback to custom attention node |
| R2 | Position offset mechanism differs from RoPE | Medium | Medium | Implement PositionOffsetter interface if generate package uses type assertions |
| R3 | TinyStories GGUF format differs from expected | Low | Low | Validate with hexdump of actual GGUF header before implementing |

---

## Operating Procedure

### Definition of Done
1. `go build ./...` passes
2. `go vet ./...` clean
3. `go test ./inference/... -race` passes
4. TinyStories-656K loads and generates text
5. PR merged, CI green

---

## Progress Log

### 2026-03-28: Plan created
- GPT-2 architecture support plan with 10 tasks across 3 waves
- Reuses GQA with rope=nil to avoid duplicate attention code
- Target model: TinyStories-656K (~1MB GGUF)
