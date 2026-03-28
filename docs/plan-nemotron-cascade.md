# Nemotron Cascade Architecture Support

## Context

NVIDIA's Nemotron Cascade family includes two distinct architectures:

1. **Nemotron-Cascade v1 (8B/14B)**: Post-trained Qwen3 models. Use `general.architecture = "qwen2"` in GGUF. Already supported by Zerfoo's existing Qwen builder -- no new work needed.

2. **Nemotron-Cascade-2 (30B-A3B)**: Hybrid Mamba-Transformer MoE architecture (`nemotron_h_moe`). 52 layers: 23 Mamba-2 (SSM) + 6 GQA attention + 23 MoE (128 routed + 2 shared experts, top-6). 30B total params, 3B active. This requires a new architecture builder.

The base **Nemotron-H** architecture (`nemotron_h`) is the dense variant (no MoE), used by Nemotron-H-8B and Nemotron-3-Super. Same hybrid Mamba+Attention pattern but with dense FFN instead of MoE.

### Objectives
- Load and run Nemotron-H and Nemotron-H-MoE GGUF models
- Reuse existing Mamba-2 (MIMOMambaBlock), GQA, and MoE infrastructure
- Support both dense (nemotron_h) and MoE (nemotron_h_moe) variants

### Non-goals
- Cascade routing (teacher/student model switching) -- this is an inference optimization, not an architecture requirement
- Training Nemotron models
- CUDA graph capture for hybrid layers (future optimization)

### Key References
- [Nemotron-H paper (arXiv:2504.03624)](https://arxiv.org/abs/2504.03624)
- [llama.cpp nemotron_h_moe issue #18064](https://github.com/ggml-org/llama.cpp/issues/18064)
- [bartowski GGUF quantizations](https://huggingface.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF)

### Existing Reusable Components
- Mamba-2 SSM: layers/ssm/mimo_ssm.go (MIMOMambaBlock)
- GQA attention: layers/attention/grouped_query_attention.go
- MoE: inference/arch_deepseek.go (stacked expert pattern)
- Hybrid dispatch: inference/arch_jamba.go (isAttentionLayer pattern)

---

## Checkable Work Breakdown

### E1: GGUF Metadata and Tensor Mapping

- [ ] T1.1 Add SSM config fields to ModelConfig  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: model/gguf/arch.go
  Add fields: SSMStateSize, SSMConvKernel, SSMNumHeads, ExpertSharedCount.
  Parse from nemotron_h_moe.ssm.* and nemotron_h.ssm.* GGUF keys.
  Acceptance: ExtractModelConfig populates SSM fields for nemotron_h_moe arch.

- [ ] T1.2 Add Nemotron-H tensor name mapping  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: model/gguf/arch.go
  Map blk.{i}.ssm_in, ssm_conv1d, ssm_dt, ssm_A, ssm_D, ssm_out,
  ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps to canonical names.
  Acceptance: MapTensorName("nemotron_h_moe", ...) returns correct mappings.

### E2: Architecture Graph Builder

- [ ] T2.1 Implement Nemotron-H dense graph builder  Owner: TBD  Est: 3h  verifies: [UC-001]
  File: inference/arch_nemotron_h.go (new)
  Layer dispatch: probe tensor names per layer to determine type
  (ssm_in -> Mamba, attn_q -> Attention, ffn_gate -> FFN).
  - Mamba layers: RMSNorm -> MIMOMambaBlock -> ResidualAdd
  - Attention layers: RMSNorm -> GQA -> ResidualAdd -> RMSNorm -> SwiGLU FFN -> ResidualAdd
  - Dense FFN layers: RMSNorm -> SwiGLU FFN -> ResidualAdd
  Postamble: Final RMSNorm -> LMHead.
  Follow arch_jamba.go pattern for hybrid dispatch.
  Acceptance: buildNemotronHGraph builds valid graph. Forward produces logits.

- [ ] T2.2 Implement Nemotron-H-MoE graph builder  Owner: TBD  Est: 2h  verifies: [UC-001]
  Deps: T2.1
  File: inference/arch_nemotron_h.go (extend)
  MoE layers: RMSNorm -> MoE(128 experts, top-6, 2 shared) -> ResidualAdd.
  Follow arch_deepseek.go stacked expert tensor pattern.
  Handle 2 shared experts (may need to extend MixtureOfExperts or concat).
  Acceptance: buildNemotronHMoEGraph builds valid graph.

- [ ] T2.3 Register nemotron_h and nemotron_h_moe  Owner: TBD  Est: 0.5h  verifies: [UC-001]
  File: inference/registry_init.go
  Add: RegisterArchitecture("nemotron_h", buildNemotronHGraph)
  Add: RegisterArchitecture("nemotron_h_moe", buildNemotronHMoEGraph)
  Acceptance: GetArchitecture returns non-nil for both.

### E3: Mamba-2 Verification

- [ ] T3.1 Verify MIMOMambaBlock compatibility with Nemotron-H  Owner: TBD  Est: 1h  verifies: [UC-001]
  File: layers/ssm/mimo_ssm.go
  Check: tensor shape expectations, in_proj/conv1d/dt/A/D/out_proj layout.
  GGUF maps backbone.layers.{bid}.mixer.* to blk.{bid}.ssm_*.
  If shapes differ from Jamba convention, add adapter or options.
  Acceptance: MIMOMambaBlock Forward produces correct shape with Nemotron-H tensors.

### E4: Tests

- [ ] T4.1 Unit tests for Nemotron-H builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  File: inference/arch_nemotron_h_test.go (new)
  Tests: (1) Build dense graph with synthetic tensors (4 layers: 2 Mamba, 1 Attn, 1 FFN).
  (2) Forward pass produces non-NaN logits. (3) Layer type detection from tensor names.
  Acceptance: All tests pass with -race.

- [ ] T4.2 Unit tests for Nemotron-H-MoE builder  Owner: TBD  Est: 1h  verifies: [UC-001]
  Deps: T4.1
  Tests: Build MoE graph with synthetic tensors (4 layers: 1 Mamba, 1 Attn, 2 MoE).
  Forward pass produces non-NaN logits. Router selects top-6 of 8 test experts.
  Acceptance: All tests pass with -race.

- [ ] T4.3 Tensor name mapping tests  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: model/gguf/arch_test.go (add to existing)
  Test MapTensorName for nemotron_h and nemotron_h_moe.
  Acceptance: All SSM, attention, MoE tensor names map correctly.

- [ ] T4.4 Run go vet and linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T2.3
  Run: go build ./... && go vet ./... && go test ./inference/... -race
  Acceptance: Clean.

### E5: Documentation

- [ ] T5.1 Update README with Nemotron-H architecture  Owner: TBD  Est: 0.5h  delivers: [README update]
  Add Nemotron-H and Nemotron-Cascade-2 rows to architecture table.
  Update count to 27 architectures (15 families).
  Note: Nemotron-Cascade v1 already works via Qwen2.
  Acceptance: README reflects Nemotron support.

---

## Parallel Work

### Tracks
| Track | Tasks | Description |
|-------|-------|-------------|
| A: GGUF | T1.1, T1.2, T4.3 | Metadata and tensor mapping |
| B: Mamba verify | T3.1 | SSM compatibility check |
| C: Builder | T2.1, T2.2, T2.3 | Graph builders (deps A, B) |
| D: Tests | T4.1, T4.2, T4.4 | Verification (deps C) |
| E: Docs | T5.1 | README (deps C) |

### Waves

#### Wave 1: Foundation (3 agents)
- [ ] T1.1 Add SSM config fields
- [ ] T1.2 Add Nemotron-H tensor name mapping
- [ ] T3.1 Verify MIMOMambaBlock compatibility

#### Wave 2: Implementation (3 agents)
- [ ] T2.1 Implement Nemotron-H dense builder  Deps: T1.1, T1.2, T3.1
- [ ] T2.2 Implement Nemotron-H-MoE builder  Deps: T2.1
- [ ] T2.3 Register architectures  Deps: T2.1

#### Wave 3: Verification (5 agents)
- [ ] T4.1 Unit tests dense  Deps: T2.3
- [ ] T4.2 Unit tests MoE  Deps: T2.3
- [ ] T4.3 Tensor name mapping tests  Deps: T1.2
- [ ] T4.4 Run go vet  Deps: T2.3
- [ ] T5.1 Update README  Deps: T2.3

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | MIMOMambaBlock weight layout differs from Nemotron-H | High | Medium | T3.1 verifies compatibility early; fallback to wrapper |
| R2 | 128-expert MoE memory pressure | Medium | Low | Stacked tensors already work for DeepSeek; profile later |
| R3 | 2 shared experts vs 1 in existing MoE | Medium | Medium | Concat into single wider FFN or extend SharedExpert to slice |
| R4 | llama.cpp nemotron_h_moe support not yet merged | High | Medium | Use bartowski GGUF quantizations which already exist |
| R5 | KV cache waste (only 12% of layers need it) | Low | Certain | Follow Jamba precedent; optimize later |

---

## Operating Procedure

### Definition of Done
1. `go build ./...` passes
2. `go vet ./...` clean
3. `go test ./inference/... -race` passes
4. Architecture registered and graph builds from synthetic tensors
5. PR merged, CI green

---

## Progress Log

### 2026-03-28: Clarification — Nemotron-Cascade-2 confirmed
- User confirmed the target is specifically Nemotron-Cascade-2 (30B-A3B).
- The plan already covers this correctly via nemotron_h_moe architecture.
- Nemotron-Cascade v1 (8B/14B) works via existing Qwen2 builder — no action needed.

### 2026-03-28: Plan created
- Nemotron Cascade architecture support plan: 12 tasks, 3 waves
- Two variants: nemotron_h (dense) and nemotron_h_moe (MoE)
- Reuses MIMOMambaBlock, GQA, and MoE infrastructure
- Note: Nemotron-Cascade v1 (8B/14B) already works via Qwen2 builder
