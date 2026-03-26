# Granite Guardian Support Plan

## Context

IBM Granite Guardian is a family of safety and content moderation models built on
the Granite LLM backbone. Unlike regular LLMs that generate free-form text,
Guardian models are **classifiers**: they accept a structured prompt containing
content to evaluate and produce a binary safety verdict (Yes/No) with optional
confidence and reasoning traces.

### Model Variants

| Model | Params | Base | GGUF | Ollama | Key Feature |
|-------|--------|------|------|--------|-------------|
| Guardian 3.0 8B | 8B | Granite 3.0 8B Instruct | Yes (tensorblock, ibm-research) | `granite3-guardian:8b` | Original; Yes/No output + logprob confidence |
| Guardian 3.0 2B | 2B | Granite 3.0 2B Instruct | Yes | `granite3-guardian:2b` | Lightweight variant |
| Guardian 3.2 5B | 5B | Granite 3.1 8B (pruned 30%) | Yes (ibm-research) | `ibm/granite3.2-guardian` | Verbalized confidence (High/Low) |
| Guardian 3.2 3B-A800M | 3B (800M active) | Granite 3.2 MoE | Yes | -- | MoE sparse variant |
| Guardian 3.3 8B | 8B | Granite 3.3 8B | Yes (ibm-granite) | `ibm/granite3.3-guardian:8b` | Hybrid thinking mode with `<think>` traces |

All Guardian models use the `granite` architecture in GGUF metadata, which is a
standard transformer decoder that llama.cpp supports via its Granite handler.
The Granite architecture is close to Llama (RoPE, GQA, SiLU-gated FFN) with
minor differences in attention bias and embedding handling.

### Risk Categories

Guardian ships with pre-baked prompt templates for these risk types:

**Harm-related (input prompts and/or responses):**
- Harm (general-purpose, default)
- Social Bias
- Jailbreaking
- Violence
- Profanity
- Sexual Content
- Unethical Behavior
- Harm Engagement
- Evasiveness

**RAG hallucination (requires context document):**
- Context Relevance
- Groundedness
- Answer Relevance

**Agentic workflow:**
- Function Calling Hallucination

**Custom criteria:** users supply their own risk definition string.

### How Inference Works

1. A system prompt template is selected based on the risk category. The template
   embeds the content to evaluate and a risk definition.
2. The model generates a short response:
   - **3.0/3.2**: a single token `Yes` or `No`. Confidence derived from logprobs
     (3.0) or verbalized `<confidence>High</confidence>` tags (3.2).
   - **3.3**: optional `<think>...</think>` reasoning trace followed by
     `<score>yes</score>` or `<score>no</score>`.
3. The response is parsed into a structured verdict.

### Prompt Template (3.0/3.2 format)

```
You are a safety agent trying to help determine whether the user message is
harmful according to the safety risk definition below.

<start_of_turn>
User Message: {user_text}
<end_of_turn>

Our safety risk definition is defined below:

<start_of_risk_definition>
* {risk_definition}
<end_of_risk_definition>

Is the user message harmful based on the risk definition? Your answer must be
either 'Yes' or 'No'.
```

For RAG risks, a `context` role is used instead of `user`, and for
response-level checks, the `assistant` content is included.

### Existing Zerfoo Infrastructure

Zerfoo already has the pieces needed for Guardian support:

- **inference/arch_llama.go**: Granite's decoder architecture is Llama-compatible
  (RoPE, GQA, SiLU FFN). The existing Llama graph builder may work directly or
  need minor adjustments for Granite-specific attention bias.
- **serve/classify.go**: Classification endpoint (`POST /v1/classify`) with
  batch support, metrics, and the `Classifier` interface. Guardian can implement
  this interface or get a dedicated endpoint.
- **generate/**: Text generation pipeline with sampling, streaming, and KV cache.
  Guardian needs constrained generation (max 1-20 tokens).
- **model/gguf/**: GGUF parser already handles Granite GGUF files.

### Goal

Provide Go-native content moderation and safety guardrails via Granite Guardian.
Since Ollama already supports Guardian, the bar is feature parity plus a
purpose-built API and CLI experience for guardrails workflows (batch moderation,
multi-risk scanning, structured verdicts with confidence).

---

## Epics

| Epic | Description | Tasks |
|------|-------------|-------|
| GG-E1 | Granite architecture + model loading | 3 tasks |
| GG-E2 | Guardian inference pipeline (templates, parsing, verdict) | 4 tasks |
| GG-E3 | API, CLI, and integration | 3 tasks |
| GG-E4 | Testing and benchmarks | 3 tasks |

**Total: 13 tasks across 3 waves.**

---

## Wave 1: Foundation (3 agents, parallel)

### GG-E1: Granite Architecture and Model Loading

Granite Guardian GGUF files use the `granite` architecture identifier. Zerfoo's
registry does not yet have a Granite entry. The architecture is a transformer
decoder similar to Llama with potential differences in attention bias, embedding
tying, and RoPE parameters.

- [ ] GG-T1.1 Add Granite architecture builder
  Owner: ML Eng  Est: 8h
  Files: inference/arch_granite.go, inference/arch_granite_test.go
  Description: Implement a Granite-specific graph builder. Granite uses a Llama-like
  architecture: token embeddings, N transformer blocks (RMSNorm, GQA with RoPE,
  SiLU-gated FFN), final RMSNorm, LM head. Key differences from Llama to handle:
  1. Attention bias (some Granite variants include QKV bias like Qwen).
  2. Embedding multiplier (scaling factor applied after token embedding lookup).
  3. Logit softcapping (Granite 3.3 may use this like Gemma).
  4. RoPE theta value (check GGUF metadata for `rope_freq_base`).
  Start by attempting to load a Guardian 3.0 8B GGUF with the existing Llama
  builder; document which tensor names and config keys differ.
  Acceptance:
  - `buildGraniteGraph` loads Guardian 3.0 8B and 3.3 8B GGUF files.
  - Forward pass on a short prompt produces valid logits.
  - go test with a small test GGUF passes; go vet clean.

- [ ] GG-T1.2 Register Granite architecture and verify GGUF loading
  Owner: ML Eng  Est: 3h
  Files: inference/registry_init.go, inference/arch_granite_test.go
  Deps: GG-T1.1
  Description: Register `"granite"` in the architecture registry pointing to
  `buildGraniteGraph`. Verify that `inference.LoadModel` correctly dispatches
  to the Granite builder when loading a Guardian GGUF file. Test with both
  Guardian 3.0 8B (dense) and Guardian 3.2 3B-A800M (MoE, if the MoE variant
  uses a different architecture key, register that alias too).
  Acceptance:
  - `RegisterArchitecture("granite", buildGraniteGraph)` in registry_init.go.
  - `inference.LoadModel("granite-guardian-3.0-8b.gguf")` succeeds.
  - `inference.LoadModel("granite-guardian-3.3-8b.gguf")` succeeds.
  - go test passes.

- [ ] GG-T1.3 Guardian prompt template engine
  Owner: ML Eng  Est: 6h
  Files: inference/guardian/templates.go, inference/guardian/templates_test.go
  Description: Implement the Guardian prompt template system. Each risk category
  maps to a structured prompt template that wraps user content, assistant content,
  and/or RAG context into the Guardian evaluation format. Support:
  1. Pre-baked risk definitions for all 13 categories (harm, social_bias,
     jailbreaking, violence, profanity, sexual_content, unethical_behavior,
     harm_engagement, evasiveness, context_relevance, groundedness,
     answer_relevance, function_call_hallucination).
  2. Custom risk definitions supplied as a string.
  3. Template variants for 3.0/3.2 format (plain text) and 3.3 format (with
     `think=true/false` flag).
  4. Multiple input roles: user-only, user+assistant, context+assistant.
  Each template produces a tokenizer-ready chat message slice
  ([]ChatMessage with role and content fields).
  Acceptance:
  - RenderTemplate("harm", GuardianInput{User: "text"}) returns correct prompt.
  - RenderTemplate("groundedness", GuardianInput{Context: "ctx", Assistant: "resp"}) works.
  - RenderTemplate with custom criteria works.
  - All 13 pre-baked categories produce valid templates.
  - go test passes.

---

## Wave 2: Guardian Pipeline (3 agents, parallel)

### GG-E2: Guardian Inference Pipeline

This epic builds the end-to-end Guardian evaluation pipeline: template rendering,
constrained generation, and verdict parsing.

- [ ] GG-T2.1 Guardian verdict parser
  Owner: ML Eng  Est: 4h
  Files: inference/guardian/verdict.go, inference/guardian/verdict_test.go
  Description: Parse Guardian model output into a structured verdict. Handle all
  three output formats:
  1. **3.0 format**: Single token `Yes` or `No`. Extract logprob-based confidence
     as P(Yes) / (P(Yes) + P(No)) from the generation output scores.
  2. **3.2 format**: `Yes` or `No` followed by `<confidence>High</confidence>` or
     `<confidence>Low</confidence>`.
  3. **3.3 format**: Optional `<think>...</think>` reasoning trace followed by
     `<score>yes</score>` or `<score>no</score>`.
  Return a `Verdict` struct:
  ```go
  type Verdict struct {
      Unsafe     bool    // true if risk detected
      Risk       string  // risk category name
      Confidence float64 // 0.0-1.0 (from logprobs or verbalized)
      Reasoning  string  // thinking trace (3.3 only, empty otherwise)
  }
  ```
  Acceptance:
  - Parses "Yes" -> Unsafe=true, "No" -> Unsafe=false.
  - Parses confidence tags and thinking traces correctly.
  - Handles malformed output gracefully (returns Verdict with Unsafe=false, Confidence=0).
  - go test with all three format variants passes.

- [ ] GG-T2.2 Guardian evaluator (template + generate + parse)
  Owner: ML Eng  Est: 8h
  Files: inference/guardian/evaluator.go, inference/guardian/evaluator_test.go
  Deps: GG-T1.1, GG-T1.2, GG-T1.3, GG-T2.1
  Description: Implement `Evaluator`, the core Guardian type that orchestrates
  the full evaluation cycle:
  1. Accept a `GuardianRequest` (content to evaluate, risk categories to check).
  2. For each risk category, render the prompt template (GG-T1.3).
  3. Tokenize and run constrained generation through the Granite model. Use
     `generate.Generator` with `MaxTokens=20` and `Temperature=0` (greedy).
  4. Parse the output with the verdict parser (GG-T2.1).
  5. Return `[]Verdict` (one per risk category).
  Support evaluating a single text against multiple risk categories in one call
  (sequential generation with KV cache reuse where possible).
  Acceptance:
  - Evaluator loads a Guardian GGUF model and produces verdicts.
  - Multi-risk evaluation (e.g., harm + jailbreaking + profanity) returns one
    verdict per risk.
  - Custom criteria evaluation works end-to-end.
  - go test with mocked generation passes; integration test with real model on
    DGX Spark passes.

- [ ] GG-T2.3 Batch evaluation with concurrent processing
  Owner: ML Eng  Est: 6h
  Files: inference/guardian/batch.go, inference/guardian/batch_test.go
  Deps: GG-T2.2
  Description: Add batch evaluation support to the Guardian evaluator. Accept a
  slice of inputs and evaluate them concurrently (up to a configurable concurrency
  limit). For each input, run all requested risk categories. Aggregate results
  into a batch response:
  ```go
  type BatchResult struct {
      Results []InputResult // one per input
  }
  type InputResult struct {
      Index    int
      Verdicts []Verdict // one per risk category
      Flagged  bool      // true if any verdict is Unsafe
  }
  ```
  Use a worker pool pattern (bounded goroutines) since each evaluation requires
  a forward pass. On GPU, serialize forward passes but parallelize template
  rendering and result parsing.
  Acceptance:
  - Batch of 10 inputs evaluated correctly.
  - Concurrency limit is respected (no more than N simultaneous forward passes).
  - Results are returned in input order.
  - go test -race passes.

- [ ] GG-T2.4 Multi-risk scanning with aggregate verdict
  Owner: ML Eng  Est: 4h
  Files: inference/guardian/scan.go, inference/guardian/scan_test.go
  Deps: GG-T2.2
  Description: Implement a convenience `Scan` function that evaluates content
  against all harm-related risk categories (or a configurable subset) and returns
  an aggregate safety verdict. This is the primary entry point for guardrails
  use cases:
  ```go
  func (e *Evaluator) Scan(ctx context.Context, input GuardianInput) (*ScanResult, error)

  type ScanResult struct {
      Safe       bool      // true if no risks detected
      Verdicts   []Verdict // all individual verdicts
      TopRisk    *Verdict  // highest-confidence unsafe verdict, nil if safe
      LatencyMs  float64
  }
  ```
  Default risk set: harm, social_bias, jailbreaking, violence, profanity,
  sexual_content, unethical_behavior. Users can override via options.
  Acceptance:
  - Scan("I want to harm someone") returns Safe=false with violence/harm flagged.
  - Scan("What is the weather?") returns Safe=true.
  - TopRisk points to the highest-confidence flagged category.
  - Custom risk subset works.
  - go test passes.

---

## Wave 3: Integration and Testing (3 agents, parallel)

### GG-E3: API, CLI, and Integration

- [ ] GG-T3.1 REST API endpoint for content moderation
  Owner: ML Eng  Est: 6h
  Files: serve/guardian.go, serve/guardian_test.go
  Deps: GG-T2.3, GG-T2.4
  Description: Add a `POST /v1/guard` endpoint to the serve package. Request and
  response types:
  ```go
  // POST /v1/guard
  type GuardRequest struct {
      Model  string        `json:"model"`
      Input  []GuardInput  `json:"input"`
      Risks  []string      `json:"risks,omitempty"`  // default: all harm categories
      Think  bool          `json:"think,omitempty"`   // enable reasoning (3.3 only)
  }
  type GuardInput struct {
      User      string `json:"user,omitempty"`
      Assistant string `json:"assistant,omitempty"`
      Context   string `json:"context,omitempty"`
  }
  type GuardResponse struct {
      Results []GuardResult `json:"results"`
      Model   string        `json:"model"`
      Usage   UsageInfo     `json:"usage"`
  }
  type GuardResult struct {
      Index    int             `json:"index"`
      Flagged  bool            `json:"flagged"`
      Verdicts []VerdictResult `json:"verdicts"`
  }
  type VerdictResult struct {
      Risk       string  `json:"risk"`
      Flagged    bool    `json:"flagged"`
      Confidence float64 `json:"confidence"`
      Reasoning  string  `json:"reasoning,omitempty"`
  }
  ```
  Wire into the existing `serve.Server` router. Add `GuardianMetrics` (request
  count, latency histogram, flagged count). Implement the `WithGuardian` server
  option following the `WithClassifier` pattern.
  Acceptance:
  - `POST /v1/guard` with single input returns correct verdict JSON.
  - Batch of 5 inputs returns 5 results.
  - Metrics are recorded.
  - Invalid requests return 400 with descriptive error.
  - go test with httptest passes.

- [ ] GG-T3.2 CLI `zerfoo guard` command
  Owner: ML Eng  Est: 4h
  Files: cmd/zerfoo/guard.go
  Deps: GG-T2.4
  Description: Add a `guard` subcommand to the zerfoo CLI:
  ```
  zerfoo guard --model ibm-granite/granite-guardian-3.3-8b \
               --input "text to evaluate" \
               --risks harm,jailbreaking,profanity \
               --think
  ```
  Support:
  - `--input` flag or stdin (pipe content to evaluate).
  - `--file` flag to read input from a file.
  - `--risks` comma-separated list (default: all harm categories).
  - `--think` enable reasoning traces (3.3 models).
  - `--json` output as JSON (default: human-readable table).
  - `--batch` read JSONL from stdin for batch evaluation.
  Auto-download model from HuggingFace if not cached.
  Acceptance:
  - `zerfoo guard --input "hello" --model granite-guardian-3.3-8b` prints verdict.
  - `echo "harmful text" | zerfoo guard --model granite-guardian-3.3-8b` works.
  - `--json` produces valid JSON output.
  - `--batch` processes JSONL input correctly.
  - `--help` is clear and complete.
  - go vet clean.

- [ ] GG-T3.3 Guardrails middleware for chat completions
  Owner: ML Eng  Est: 6h
  Files: serve/guardian_middleware.go, serve/guardian_middleware_test.go
  Deps: GG-T2.4
  Description: Implement an optional guardrails middleware that wraps the
  `/v1/chat/completions` endpoint. When enabled, it evaluates the user prompt
  before generation and (optionally) the assistant response after generation.
  If content is flagged, return a 400 error with the verdict details instead
  of the completion. Configuration:
  ```go
  type GuardianMiddlewareConfig struct {
      Model         string   // Guardian model path
      Risks         []string // risk categories to check
      CheckInput    bool     // scan user prompts (default: true)
      CheckOutput   bool     // scan assistant responses (default: false)
      BlockOnFlag   bool     // return error if flagged (default: true)
  }
  ```
  This enables using Guardian as an inline safety layer for any LLM served by
  Zerfoo, not just as a standalone moderation endpoint.
  Acceptance:
  - Chat request with harmful prompt is blocked with 400 + verdict JSON.
  - Chat request with safe prompt passes through normally.
  - Output checking scans assistant response before returning to client.
  - Middleware can be disabled at runtime via config.
  - go test with httptest passes.

### GG-E4: Testing and Benchmarks

- [ ] GG-T4.1 Parity tests against Ollama granite3-guardian
  Owner: ML Eng  Est: 8h
  Files: tests/parity/guardian_test.go, tests/parity/testdata/guardian/
  Deps: GG-T2.2
  Description: Parity tests that verify Zerfoo's Guardian verdicts match Ollama's
  output. For each risk category:
  1. Run Ollama `granite3-guardian:8b` on a set of test inputs, save verdicts
     as golden files.
  2. Run Zerfoo on same inputs with same model, compare verdicts.
  Test cases should cover:
  - 5 clearly harmful inputs (expected: Yes for relevant categories).
  - 5 clearly safe inputs (expected: No for all categories).
  - 3 RAG hallucination cases (groundedness, context relevance).
  - 2 edge cases (ambiguous content).
  Golden files checked into tests/parity/testdata/guardian/.
  Acceptance:
  - Verdict (Yes/No) matches Ollama on all 15 test cases.
  - Confidence values within 0.05 of Ollama reference.
  - Tests run on DGX Spark with `go test -run TestParity -count=1`.

- [ ] GG-T4.2 Latency benchmarks
  Owner: Kernel Eng  Est: 6h
  Files: tests/benchmark/guardian_bench_test.go
  Deps: GG-T2.3
  Description: Benchmark Guardian evaluation latency:
  - **Single evaluation**: Time to evaluate one input against one risk category.
  - **Multi-risk scan**: Time to evaluate one input against all 7 harm categories.
  - **Batch throughput**: Evaluations per second for batch sizes 1, 8, 32.
  Compare against Ollama `granite3-guardian:8b` on same hardware (DGX Spark).
  Target: match or beat Ollama latency (Guardian prompts are short, so generation
  overhead should be minimal and CUDA graph capture should help).
  Profile the hot path:
  - Template rendering time vs. generation time vs. parsing time.
  - KV cache reuse across multi-risk evaluations on same input.
  Acceptance:
  - Benchmark results for all three scenarios.
  - Comparison table vs. Ollama included in results.
  - Single evaluation latency < 100ms on GPU (Guardian generates 1-20 tokens).

- [ ] GG-T4.3 Safety benchmark accuracy evaluation
  Owner: ML Eng  Est: 6h
  Files: tests/benchmark/guardian_accuracy_test.go
  Deps: GG-T2.2
  Description: Evaluate Zerfoo's Guardian implementation on standard safety
  benchmarks to verify the model performs as reported by IBM:
  - **HarmBench** subset: F1 score on harm detection.
  - **ToxiGen** subset: social bias detection accuracy.
  - **XSTest** subset: false positive rate on safe content.
  Use the same evaluation methodology as the Guardian model card (binary
  classification metrics: precision, recall, F1, balanced accuracy).
  Acceptance:
  - F1 scores within 2% of IBM's reported numbers.
  - False positive rate on XSTest safe subset < 5%.
  - Results logged with per-category breakdown.

---

## Wave Schedule

| Wave | Tasks | Agents | Deps | Est Duration |
|------|-------|--------|------|-------------|
| W1 | GG-T1.1, GG-T1.2, GG-T1.3 | 3 | None | 1 day |
| W2 | GG-T2.1, GG-T2.2, GG-T2.3, GG-T2.4 | 3 | W1 | 2 days |
| W3 | GG-T3.1, GG-T3.2, GG-T3.3, GG-T4.1, GG-T4.2, GG-T4.3 | 3 | W2 | 2 days |

**Total estimated: 5 working days with 3 parallel agents.**

Within each wave, tasks are assigned to agents for parallel execution.
In Wave 2, GG-T2.1 can run independently; GG-T2.2 depends on W1 + GG-T2.1;
GG-T2.3 and GG-T2.4 both depend on GG-T2.2.
In Wave 3, all E3 and E4 tasks depend only on W2 and run fully in parallel.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Granite GGUF uses different tensor names than Llama | High | Medium | Inspect actual GGUF files; compare tensor names with llama.cpp granite handler |
| Granite architecture has undocumented differences from Llama | Medium | High | Compare llama.cpp `build_granite` function; test forward pass layer-by-layer |
| Guardian 3.3 thinking mode produces variable-length output | Low | Low | Set generous `MaxTokens=200` for thinking mode; parse `<score>` tag from end |
| Logprob-based confidence extraction differs from HuggingFace | Medium | Medium | Compare softmax(logprob) values against Python reference output |
| MoE variant (3.2 3B-A800M) needs separate architecture handling | Medium | Medium | Check if llama.cpp treats it as `granite` or `granite_moe`; add alias if needed |
| Guardian prompt template format changes between versions | Low | Low | Template engine supports version-specific rendering; test each version |

---

## References

- [Granite Guardian 3.0 8B Model Card](https://huggingface.co/ibm-granite/granite-guardian-3.0-8b)
- [Granite Guardian 3.2 5B Model Card](https://huggingface.co/ibm-granite/granite-guardian-3.2-5b)
- [Granite Guardian 3.3 8B Model Card](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b)
- [Granite Guardian 3.3 8B GGUF](https://huggingface.co/ibm-granite/granite-guardian-3.3-8b-GGUF)
- [Granite Guardian GitHub](https://github.com/ibm-granite/granite-guardian)
- [Granite Guardian Ollama Cookbook](https://github.com/ibm-granite/granite-guardian/blob/main/cookbooks/granite-guardian-3.0/detailed_guide_ollama.ipynb)
- [Ollama granite3-guardian](https://ollama.com/library/granite3-guardian)
- [Ollama granite3.3-guardian](https://ollama.com/ibm/granite3.3-guardian)
- [IBM Granite Guardian Docs](https://www.ibm.com/granite/docs/models/guardian)
- [Granite Guardian Paper (arXiv:2412.07724)](https://arxiv.org/abs/2412.07724)
- [IBM GGUF Conversion Repo](https://github.com/ibm-granite/gguf)
