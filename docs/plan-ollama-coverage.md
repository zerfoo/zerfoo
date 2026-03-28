# Ollama Model Coverage

## Context

Gap analysis of Ollama models updated in the last 6 months (as of 2026-03-28)
that Zerfoo cannot load. Cross-referenced against llama.cpp architecture
strings and Zerfoo's registered builders.

### Already Covered (28 architectures)
llama, llama4, gemma/gemma3/gemma3n, qwen2, mistral, mixtral, phi/phi3,
deepseek_v3/deepseek2, command-r, falcon, rwkv, mamba/mamba3, jamba, granite,
whisper, bert, gpt2, nemotron_h/nemotron_h_moe, minimax-m2, llava, qwen_vl

### Missing Architectures (9 new builders needed)

| Ollama Model | llama.cpp Arch | Type | Effort | Priority |
|-------------|---------------|------|--------|----------|
| glm-4.7/glm-5 | chatglm/glm4/glm4moe | Transformer + MoE | Medium | High (popular) |
| kimi-k2.5 | kimi-linear | Linear attention MoE | Medium | High (new) |
| lfm2 | lfm2/lfm2moe | Hybrid MoE | Medium | Medium |
| olmo-3/3.1 | olmo2 | Transformer (llama-like) | Low | Medium |
| starcoder2 | starcoder2 | Transformer (MQA) | Low | Low |
| internlm2 | internlm2 | Transformer (llama-like) | Low | Low |
| exaone3.5 | exaone/exaone4 | Transformer | Low | Low |
| dbrx | dbrx | MoE | Medium | Low |
| glm-ocr | deepseek2-ocr | Extend deepseek2 | Low | Low |

### Not Actionable
- minimax-m2.7: API-only, no open weights
- gemini-3-flash-preview: not in llama.cpp GGUF

---

## Checkable Work Breakdown

### E1: Low-Effort Llama-Like Architectures (olmo2, internlm2, exaone, starcoder2)

These architectures are structurally similar to Llama with minor differences.
Many can reuse buildTransformerGraph or buildLlamaGraph with config tweaks.

- [ ] T1.1 Add olmo2 architecture builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  File: inference/arch_olmo2.go (new)
  OLMo2 is nearly identical to Llama but uses:
  - No bias in attention (same as Llama)
  - Norm before and after attention (post-norm variant option)
  - Different tokenizer padding
  Check if buildLlamaGraph works directly with arch="olmo2" tensors.
  If yes, just register as alias. If no, create thin wrapper.
  Acceptance: olmo-3.1 GGUF loads and generates.

- [ ] T1.2 Add internlm2 architecture builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  File: inference/arch_internlm2.go (new)
  InternLM2 uses GQA with RoPE, SwiGLU FFN -- structurally identical to Llama.
  May need tensor name remapping only.
  Acceptance: internlm2 GGUF loads.

- [ ] T1.3 Add exaone architecture builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  File: inference/arch_exaone.go (new)
  EXAONE uses GQA with RoPE, SwiGLU -- Llama-like.
  Register both "exaone" and "exaone4".
  Acceptance: exaone3.5 GGUF loads.

- [ ] T1.4 Add starcoder2 architecture builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  File: inference/arch_starcoder2.go (new)
  StarCoder2 uses GQA (or MQA), RoPE, sliding window attention.
  Similar to Mistral. May reuse buildMistralGraph with config tweaks.
  Acceptance: starcoder2 GGUF loads.

- [ ] T1.5 Add dbrx architecture builder  Owner: TBD  Est: 2h  verifies: [UC-001]
  File: inference/arch_dbrx.go (new)
  DBRX is a fine-grained MoE (16 experts, top-4). Uses GQA + RoPE.
  Follow DeepSeek/Mixtral MoE pattern.
  Acceptance: dbrx GGUF loads.

- [ ] T1.6 Tests for llama-like architectures  Owner: TBD  Est: 2h  verifies: [UC-001]
  Deps: T1.1-T1.5
  Synthetic weight tests for each builder.

- [ ] T1.7 Register all E1 architectures  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: T1.1-T1.5
  File: inference/registry_init.go
  Register: olmo2, internlm2, exaone, exaone4, starcoder2, dbrx

### E2: GLM/ChatGLM Architecture

GLM (General Language Model) from Zhipu AI uses a unique architecture with:
- Multi-query attention (MQA) with RoPE
- SwiGLU FFN
- GLM-specific positional encoding for some variants
- GLM-5 uses MoE (glm4moe arch)
- GLM-OCR uses deepseek2-ocr architecture (separate)

- [ ] T2.1 Add GLM4 architecture builder  Owner: TBD  Est: 3h  verifies: [UC-001]
  File: inference/arch_glm.go (new)
  Build graph for chatglm/glm4 architecture.
  GQA with multi-query attention, RoPE, SwiGLU FFN.
  Register: "chatglm", "glm4"
  Acceptance: glm-4.7 GGUF loads and generates.

- [ ] T2.2 Add GLM4-MoE variant  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  Deps: T2.1
  Extend arch_glm.go for glm4moe architecture.
  GLM-5 is 744B total / 40B active MoE.
  Register: "glm4moe"
  Acceptance: glm-5 GGUF loads.

- [ ] T2.3 Add GLM-DSA (deepseek2-ocr) support  Owner: TBD  Est: 1h  verifies: [UC-001]
  Extend existing DeepSeek2 builder or create thin wrapper.
  Register: "deepseek2-ocr", "glm-dsa"
  Acceptance: glm-ocr GGUF loads.

- [ ] T2.4 GLM tensor name mapping  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  File: model/gguf/arch.go

- [ ] T2.5 Tests for GLM builders  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  Deps: T2.1-T2.3

### E3: Kimi (Moonshot) Architecture

Kimi K2/K2.5 uses a hybrid architecture with linear attention (kimi-linear).
This is a fundamentally new attention mechanism -- not softmax, not GQA.

- [ ] T3.1 Research Kimi linear attention mechanism  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Read llama.cpp implementation and Kimi papers.
  Document the attention computation (linear vs softmax).
  Determine if existing GQA can be adapted or if a new attention layer is needed.

- [ ] T3.2 Implement Kimi linear attention layer  Owner: TBD  Est: 3h  verifies: [UC-001]
  Deps: T3.1
  File: layers/attention/kimi_linear_attention.go (new)
  Linear attention: O = (Q * K^T) * V without softmax normalization.
  May use feature map phi(x) for efficient computation.
  Acceptance: Forward produces correct shape, matches llama.cpp reference.

- [ ] T3.3 Implement buildKimiGraph  Owner: TBD  Est: 3h  verifies: [UC-001]
  Deps: T3.2
  File: inference/arch_kimi.go (new)
  Kimi K2.5 is an MoE model with linear attention.
  Register: "kimi-linear"
  Acceptance: kimi-k2.5 GGUF loads.

- [ ] T3.4 Kimi tensor name mapping  Owner: TBD  Est: 0.5h  verifies: [infrastructure]

- [ ] T3.5 Tests for Kimi builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  Deps: T3.3

### E4: LFM2 (Liquid Foundation Model)

LFM2 from Liquid AI uses a hybrid architecture combining dense and sparse
(MoE) components. 24B total / 2B active per token.

- [ ] T4.1 Research LFM2 architecture  Owner: TBD  Est: 1h  verifies: [infrastructure]
  Read llama.cpp implementation. Determine layer types and dispatch pattern.
  LFM2 may use custom SSM or attention variants.

- [ ] T4.2 Implement buildLFM2Graph  Owner: TBD  Est: 3h  verifies: [UC-001]
  Deps: T4.1
  File: inference/arch_lfm2.go (new)
  Register: "lfm2", "lfm2moe"
  Acceptance: lfm2 GGUF loads.

- [ ] T4.3 Tests for LFM2 builder  Owner: TBD  Est: 1.5h  verifies: [UC-001]
  Deps: T4.2

### E5: Verification and Documentation

- [ ] T5.1 Run go vet and linters for all new architectures  Owner: TBD  Est: 0.5h  verifies: [infrastructure]
  Deps: all above

- [ ] T5.2 Update README with new architectures  Owner: TBD  Est: 0.5h  delivers: [README update]
  Update architecture count and model grid.

- [ ] T5.3 Update website model grid  Owner: TBD  Est: 0.5h  delivers: [website update]

---

## Waves

#### Wave 1: Llama-Like Architectures (5 agents)
- [ ] T1.1 olmo2
- [ ] T1.2 internlm2
- [ ] T1.3 exaone
- [ ] T1.4 starcoder2
- [ ] T1.5 dbrx

#### Wave 2: GLM + Research (5 agents)
- [ ] T2.1 GLM4 builder
- [ ] T2.4 GLM tensor mapping
- [ ] T3.1 Research Kimi linear attention
- [ ] T4.1 Research LFM2 architecture
- [ ] T1.6 Tests for Wave 1  Deps: T1.1-T1.5

#### Wave 3: Complex Architectures (5 agents)
- [ ] T2.2 GLM4-MoE  Deps: T2.1
- [ ] T2.3 GLM-DSA  Deps: T2.1
- [ ] T3.2 Kimi linear attention layer  Deps: T3.1
- [ ] T4.2 LFM2 builder  Deps: T4.1
- [ ] T1.7 Register all E1 architectures  Deps: T1.1-T1.5

#### Wave 4: Kimi Builder + Tests (5 agents)
- [ ] T3.3 buildKimiGraph  Deps: T3.2
- [ ] T3.4 Kimi tensor mapping
- [ ] T2.5 GLM tests  Deps: T2.1-T2.3
- [ ] T4.3 LFM2 tests  Deps: T4.2
- [ ] T3.5 Kimi tests  Deps: T3.3

#### Wave 5: Verification (3 agents)
- [ ] T5.1 go vet  Deps: all above
- [ ] T5.2 Update README
- [ ] T5.3 Update website

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Kimi linear attention requires fundamentally new layer | High | Medium | Research first (T3.1), may need new attention kernel |
| R2 | LFM2 architecture not well documented | Medium | Medium | Read llama.cpp source as reference |
| R3 | GLM positional encoding differs from standard RoPE | Medium | Low | GLM4+ uses standard RoPE; older ChatGLM had custom PE |
| R4 | Some "llama-like" architectures have subtle differences | Low | Medium | Test each with real GGUF before marking complete |

---

## Progress Log

### 2026-03-28: Plan created
- Ollama coverage gap analysis: 9 missing architectures
- 25 tasks across 5 waves
- Priority: GLM (popular in China), Kimi (new linear attention), then long tail
- ~60% of missing architectures are llama-like (low effort)
