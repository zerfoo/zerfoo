# MiniMax M2 Architecture Support

## Context

MiniMax-M2 is an open-weight MoE transformer from MiniMax AI. The architecture
(`minimax_m2` / `minimax-m2` in GGUF) covers M2, M2.1, and M2.5 releases.
Note: MiniMax M2.7 is API-only with no open weights. This plan targets the
open-weight M2 family.

### Architecture Overview
- 62 layers, all identical: GQA attention + MoE FFN
- 256 routed experts, 8 active per token
- **Sigmoid gating** with routing bias (not softmax like DeepSeek)
- QK normalization (per-layer RMSNorm on Q and K)
- Partial RoPE: rotary_dim=64 of head_dim=128 (like Phi)
- 48 attention heads, 8 KV heads (GQA ratio 6:1)
- No shared experts, no lightning/linear attention
- MTP (Multi-Token Prediction) heads ignored at inference
- rope_theta = 5,000,000
- llama.cpp support merged (PR #16831, Oct 2025)

### Key Difference from Existing Builders
The only novel component is **sigmoid MoE gating with routing bias**. Everything
else (GQA, RoPE, RMSNorm, stacked expert tensors) exists in zerfoo.

---

## Checkable Work Breakdown

### E1: Sigmoid MoE Gate

- [x] T1.1 Add MoESigmoidGate to layers/core/moe.go  Owner: TBD  Est: 1.5h  verifies: [infrastructure]
  Add sigmoid gating option with optional bias tensor. Top-k selection and
  weight normalization remain unchanged. Accept WithSigmoidGating() and
  WithRoutingBias(bias) options.
  Acceptance: Sigmoid gate selects correct top-k. Unit test passes.

- [x] T1.2 Unit tests for sigmoid gate  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Tests: (1) Correct top-k with sigmoid. (2) Bias shifts selection.
  (3) Normalized weights sum to 1.0.

### E2: GGUF Metadata and Tensor Mapping

- [x] T2.1 Add MiniMax-M2 tensor name mapping  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Map: exp_probs_b (routing bias), ffn_gate_inp, ffn_gate_exps,
  ffn_up_exps, ffn_down_exps, attn_q_norm, attn_k_norm.
  Acceptance: MapTensorName("minimax-m2", ...) correct for all tensor names.

- [x] T2.2 Add ScoringFunc to ModelConfig  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Parse from {arch}.expert_gating_func GGUF key. Default: "softmax".
  Acceptance: ScoringFunc = "sigmoid" for minimax-m2.

### E3: Architecture Graph Builder

- [x] T3.1 Implement buildMiniMaxM2Graph  Owner: TBD  Est: 4h  verifies: [UC-001]
  Deps: T1.1, T2.1, T2.2
  File: inference/arch_minimax_m2.go (new)
  Graph: Embed -> [RMSNorm -> GQA(QKNorm, PartialRoPE) -> Add ->
  RMSNorm -> MoE(256, top-8, sigmoid+bias) -> Add] x 62 -> RMSNorm -> LMHead
  Follow arch_deepseek.go for MoE wiring. Use blk-style tensor names.
  Acceptance: Forward produces [batch, seq, vocab] logits.

- [x] T3.2 Register minimax-m2 architecture  Owner: TBD  Est: 0.25h  verifies: [UC-001]
  Deps: T3.1

### E4: Tests

- [x] T4.1 Unit tests for builder  Owner: TBD  Est: 2h  verifies: [UC-001]
  Deps: T3.2. Synthetic tensors (4 layers, 4 experts, top-2).

- [x] T4.2 Tensor name mapping tests  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T2.1

- [x] T4.3 Run go vet and linters  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T3.2

### E5: Documentation

- [x] T5.1 Update README  Owner: TBD  Est: 0.5h  delivers: [README update]

---

## Waves

#### Wave 1: Foundation (3 agents)
- [x] T1.1 Sigmoid MoE gate
- [x] T2.1 Tensor name mapping
- [x] T2.2 ScoringFunc config field

#### Wave 2: Implementation (2 agents)
- [x] T3.1 buildMiniMaxM2Graph  Deps: T1.1, T2.1, T2.2
- [x] T3.2 Register architecture  Deps: T3.1

#### Wave 3: Verification (5 agents)
- [x] T1.2 Sigmoid gate tests  Deps: T1.1
- [x] T4.1 Builder tests  Deps: T3.2
- [x] T4.2 Tensor mapping tests  Deps: T2.1
- [x] T4.3 go vet  Deps: T3.2
- [x] T5.1 README  Deps: T3.2

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | 256-expert memory at inference | Medium | Low | Stacked tensors work for DeepSeek |
| R2 | Sigmoid normalization differs from softmax | Medium | Medium | Validate against llama.cpp outputs |
| R3 | Partial RoPE (64/128) | Low | Low | Existing WithRotaryDimFraction handles this |

---

## Progress Log

### 2026-03-28: Plan created
- MiniMax M2 architecture plan: 11 tasks, 3 waves
- Key novel component: sigmoid MoE gating with routing bias
- Clarified: M2.7 is API-only; this covers open-weight M2/M2.1/M2.5
- Updated Nemotron Cascade plan: user confirmed Cascade-2 target
