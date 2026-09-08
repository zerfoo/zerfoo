# zerfoo

Pure Go ML framework -- inference, training, and serving. Embed any GGUF model in your Go application with `go build ./...`.

[![CI](https://github.com/zerfoo/zerfoo/actions/workflows/ci.yml/badge.svg)](https://github.com/zerfoo/zerfoo/actions/workflows/ci.yml)
[![Go 1.26+](https://img.shields.io/badge/Go-1.26+-00ADD8.svg)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/zerfoo.svg)](https://pkg.go.dev/github.com/zerfoo/zerfoo)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**235 tok/s** on Gemma 3 1B Q4_K_M -- 1.25x [Ollama](https://ollama.com/) 0.17.7 on an NVIDIA DGX Spark GB10, at parity by 3B. Zero CGo in the default and release builds. 38 model builders covering 45 GGUF architecture strings. **Run models larger than RAM** via memory-mapped I/O: a 229B MiniMax-M2 in 128.8 GB of weights loads in 6.3s on a 128 GB machine. QuaRot weight fusion, quantized and tiered KV cache, Multi-LoRA serving, BitNet ternary inference, CUDA graph capture. Tabular ML and time-series forecasting built in.

### Benchmarks

Decode throughput against [Ollama](https://ollama.com/) on NVIDIA DGX Spark GB10 (Grace Blackwell, sm_121, 128 GB LPDDR5x unified memory).

| Model | Size | Quant | Zerfoo (tok/s) | Ollama (tok/s) | Ratio |
|-------|------|-------|----------------|----------------|-------|
| Gemma 3 1B | 1B | Q4_K_M | **235** | 188 | **1.25x** |
| DeepSeek-R1-Distill 1.5B | 1.5B | Q4_K_M | **186** | 167 | **1.11x** |
| Llama 3.2 3B | 3B | Q4_K_M | 92 | 93 | 0.99x |
| Mistral 7B | 7B | Q5_K_M | 44 | 44 | 1.00x |

**Method:** DGX Spark GB10, 2026-03-27. 128 generated tokens from a fixed 8-word
prompt, greedy sampling (temperature 0), 3-run median, batch size 1, fp32 compute
and KV. Compared against Ollama 0.17.7. Raw results:
[`results/benchmark-2026-03-27.json`](results/benchmark-2026-03-27.json).

Zerfoo is meaningfully faster at 1B-1.5B and at parity by 3B -- the advantage
shrinks as the model grows, which is what a bandwidth-bound decode with a
well-optimized small-model path looks like. We do not claim a win at 3B or above.

<details>
<summary>Full methodology, asymmetries, and reproduction</summary>

- **Hardware**: NVIDIA DGX Spark GB10 (Grace Blackwell, sm_121, 128 GB LPDDR5x unified memory)
- **Prompt**: "Explain the theory of relativity in simple terms."
- **Tokens**: 128 generated tokens per run
- **Sampling**: greedy (temperature = 0), batch size 1
- **Runs**: 3 per model per runtime, median reported
- **Date**: 2026-03-27
- **Ollama version**: 0.17.7 (recorded in the sibling `results/benchmark-2026-03-25.json`)
- **Harness**: [`scripts/bench-compare-ollama.sh`](scripts/bench-compare-ollama.sh) driving [`cmd/bench_tps`](cmd/bench_tps/main.go)
- **Acceleration**: CUDA graph capture (184/185 instructions, 99.5%) with flash attention decode, fused AddRMSNorm, fused SwiGLU, fused QKNormRoPE, merged QKV, merged gate+up. mmap is auto-disabled on CUDA devices, so the GPU numbers and the over-RAM mmap path below are never exercised together.

Known asymmetries, stated because they affect how the ratio should be read:

- **The two runtimes report different metrics.** Zerfoo's timer spans prefill plus decode; Ollama's `eval rate` is decode-only. For a short prompt and 128 generated tokens the gap is small, and it runs *against* Zerfoo.
- **Quantization labels describe the file, not the arithmetic.** The GGUF loader re-quantizes Q4_K to Q4_0 at load ([`model/gguf/loader.go:265`](model/gguf/loader.go)), so Zerfoo runs Q4_0 math on a Q4_K_M file while Ollama runs true Q4_K_M. No perplexity comparison has been run, so this is **not quality-normalized**.
- **Ollama model tags are unpinned**, so the server-side quantization is whatever the registry serves.
- **Single operator, single machine, single configuration.** One prompt, one token count, greedy, batch 1, short context. Nothing here has been independently reproduced, and nothing here says anything about batched or concurrent serving.
- **GB10-specific.** Grace Blackwell unified memory. Results will not transfer to a discrete-VRAM datacenter GPU, and the ranking may invert.
- **Ollama is a convenience baseline**, not the performance ceiling. No comparison against llama.cpp, vLLM, TensorRT-LLM, or MLX exists.

Reproduction instructions, the CPU-only microbenchmarks that need no GPU, and
the full evidence table are in [`docs/benchmarks.md`](docs/benchmarks.md).

</details>

<details>
<summary>Where our kernels lost</summary>

A fused Q4_K GEMV kernel exists ([`internal/cuda/kernels/gemv_q4k.cu`](internal/cuda/kernels/gemv_q4k.cu),
with an `sm_121` variant) and it is **not on the default inference path** -- because
it was slower than the path it was written to replace. Measured on GB10 at 256
tokens: native Q4_K plus the `sm_121` kernel reached 174.44 tok/s against 245.15
tok/s for the Q4_0 re-quantization path (`docs/devlog.md`, 2026-03-18). Q4_0's
18-byte blocks cache better than Q4_K's 144-byte super-blocks, and that beats the
arithmetic saved by fusing. The loader therefore re-quantizes, and the Q4_K kernel
has no call site in `inference/`, `layers/`, or `model/`.

Also unresolved: flash decode is not yet benchmarked, and the Gemma 4 E2B edge
path produces degenerate decode output, so its throughput figures are a
regression floor rather than a performance claim. See
[`docs/benchmarks.md`](docs/benchmarks.md).

</details>

### Memory-Mapped Model Loading

Zerfoo memory-maps GGUF files by default — no flags, no configuration. The entire file (or all shards of a split GGUF) is mapped rather than read into the heap. Tensor data stays on disk and is paged into RAM on demand by the OS. Split GGUF files (multiple shards) are detected and mapped automatically from any shard path.

**Single recorded run on DGX Spark, 2026-03-29 (128 GB RAM, CPU-only):**

| Model | Params | Quant | File Size | Shards | Load time | Throughput | Ollama |
|-------|--------|-------|-----------|--------|-----------|-----------|--------|
| MiniMax-M2 | 229B (MoE) | Q4_K_M | 128.8 GB | 3 | **6.3s** | 0.06 tok/s | fails to load (HTTP 500) |

This is a **capability result, not a throughput result.** A 128.8 GB model
decoded on a 128 GB machine at all is the claim; 0.06 tok/s is NVMe-bound and
unusable for interactive work. The run generated 4 tokens, checked for
non-degeneracy rather than quality, on CPU — there is no GPU path for over-RAM
inference. Recorded once, in `docs/devlog.md` (2026-03-29).

```go
// 128.8 GB model across 3 shards on a 128 GB machine.
// 809 tensors mapped. No heap allocation for weights.
m, _ := zerfoo.Load("./MiniMax-M2-Q4_K_M-00001-of-00003.gguf")
defer m.Close()
result, _ := m.Generate(ctx, "The meaning of life is")
// → "a priori is something"
```

Startup maps all shards and parses tensor metadata — no weight data is read until inference. The OS pages 128.8 GB of Q4_K_M quantized weights from NVMe as each matrix multiply streams through its superblocks. Ollama returns a 500 error on the same model on the same hardware.

## Advanced Inference Features

### EAGLE Speculative Decoding — implemented, not yet wired to the Model API

Self-speculative decoding using a lightweight prediction head — no draft model
needed. Based on [EAGLE-3](https://arxiv.org/abs/2503.01840). The head, the
draft loop, the weight loader, and head training are implemented and unit-tested
(`generate/eagle_speculative.go`, `layers/core/eagle_head.go`, `inference/eagle.go`).

**Status, stated plainly:** the generator has no caller outside its tests, and
`generate.WithEAGLE` records a head path that no generation path currently reads.
There is no measured speedup for EAGLE in this repository. Treat it as
implemented infrastructure, not a shipping feature — do not plan around it yet.

Head training works today and is reachable from the CLI:

```bash
zerfoo eagle-train --model model.gguf --corpus data.txt --output eagle-head.gguf --epochs 5
```

Training currently uses synthetic feature pairs; capturing real graph-level
intermediates is not yet implemented.

### QuaRot Weight Fusion

Hadamard rotation fused into weights at load time for uniform 4-bit quantization. Based on [QuaRot](https://arxiv.org/abs/2404.00456). Wired end to end — `inference.WithQuaRot` is applied before graph build — and covered by 13 tests.

```bash
zerfoo run --quarot model.gguf "Hello world"
```

No accuracy or throughput number has been measured for QuaRot in this repository,
so none is claimed. Known limitation: the hidden dimension must be a power of two;
non-power-of-two tensors are skipped with a warning.

### Quantized KV Cache

Store the KV cache quantized instead of fp32. Q4 packs two 4-bit values per byte
with a per-block scale (~7.5x smaller than fp32); Q3 is ~6.4x. Configured on the
generator:

```go
import "github.com/zerfoo/zerfoo/generate"

gen := generate.NewGenerator(/* ... */,
    generate.WithGeneratorKVDtype("q4"),  // "fp32" (default), "fp16", "q4", "q3"
)
```

These are memory ratios, not throughput results; no decode-quality comparison
across KV dtypes has been run.

### Tiered KV Cache

Automatically spill KV cache across three storage tiers as sequences grow — no OOM, no manual tuning:

- **Hot**: uncompressed tensors in GPU/CPU memory (recent tokens)
- **Warm**: compressed in CPU memory via block quantization
- **Cold**: serialized to disk as binary files (oldest tokens)

Layers are promoted and demoted automatically based on access frequency. Async prefetch moves cold layers back to hot before the decoder needs them.

```go
import "github.com/zerfoo/zerfoo/generate"

gen := generate.NewGenerator(/* ... */,
    generate.WithTieredKV(generate.TieredKVStoreConfig{
        ChunkSize:        64,  // warm-tier compression chunk size
        DemoteThreshold:  2,   // demote layers accessed < 2 times
        PromoteThreshold: 8,   // promote layers accessed > 8 times
        // ColdDir: "/var/cache/kv" // optional: persist cold tier across calls
    }),
)
```

Configured on `generate.Generator`, not on the high-level `zerfoo.Model` API.
Useful for long-context inference where the KV cache exceeds GPU memory.

### TransMLA — MHA-to-MLA Conversion

Convert any MHA/GQA model to Multi-Head Latent Attention via truncated SVD, shrinking the KV cache by the chosen rank. Based on [TransMLA](https://arxiv.org/abs/2502.07864). Use `transmla-validate` to compare perplexity against the original before and after conversion — no fixed reduction ratio or quality delta is claimed here.

```bash
zerfoo transmla --rank 512 --input model.gguf --output model-mla.gguf
```

### Multi-LoRA Serving

Serve multiple LoRA adapters from a single base model. Per-request adapter selection via the OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "gemma3-1b:my-lora", "messages": [{"role": "user", "content": "Hello"}]}'
```

### BitNet Ternary Inference

Native support for ternary weight models ({-1, 0, 1}) where matrix multiplication becomes integer addition/subtraction. Based on [BitNet b1.58](https://arxiv.org/abs/2402.17764).

### Native Sparse Attention (NSA)

Hardware-aligned three-path sparse attention: coarse compression, fine-grained selection, and sliding window. Fused CUDA kernel. Based on [NSA](https://arxiv.org/abs/2502.11089).

### Hybrid CPU/GPU MoE

Place shared MoE experts on GPU, offload routed experts to CPU with SIMD kernels, and predictively prefetch the experts a router is likely to select. Based on [KTransformers](https://arxiv.org/abs/2501.14018). Prefetch hit rate is exercised by unit tests over synthetic routing patterns (`inference/moe_prefetch_test.go`); no hit rate from a real workload has been recorded, so none is quoted.

### Audio Transcription

Transcribe WAV audio to text using Whisper or Voxtral speech-to-text models. Audio is chunked into 30-second segments, mel spectrograms are extracted, and each chunk is decoded:

```go
import "github.com/zerfoo/zerfoo/inference"

wavData, _ := os.ReadFile("speech.wav")

m, _ := inference.Load("openai/whisper-large-v3")
defer m.Close()

text, err := m.Transcribe(context.Background(), wavData)
fmt.Println(text)
```

Supports 16 kHz mono WAV input. Whisper uses 80 mel bins; Voxtral uses 128. Long
audio is automatically chunked into 30-second segments and concatenated.
`Transcribe` lives on `inference.Model`, not on the high-level `zerfoo.Model`.
For a causal decoder this path is an approximation — the two-phase graph
(encode once, decode autoregressively) is not yet wired.

## Quick Start

```go
m, _ := zerfoo.Load("google/gemma-3-4b")  // downloads from HuggingFace
defer m.Close()
response, _ := m.Chat("Explain Go interfaces in one sentence.")
fmt.Println(response)
```

## Installation

```bash
go get github.com/zerfoo/zerfoo
```

### Zero CGo

`go build ./...` needs no C toolchain. All three release builds pin
`CGO_ENABLED=0` (`.goreleaser.yml`), and CUDA is reached at runtime through a
hand-rolled `dlopen`/`dlsym` layer rather than CGo (`internal/cuda/purego.go`),
so a CGo-free binary can still drive a GPU.

**The qualification:** this holds for the default and release builds. The
optional GPU kernel build (`-tags cuda`), along with `cutlass`, `tensorrt`, and
`opencl`, *is* a CGo path — 27 files contain `import "C"`, every one behind a
build tag that is off by default. If you build with `-tags cuda`, you need a C
toolchain and you are not running a zero-CGo binary.

## HuggingFace Download

`Load` accepts HuggingFace model IDs. Models are downloaded and cached automatically:

```go
// Download by repo ID (defaults to Q4_K_M quantization)
m, err := zerfoo.Load("google/gemma-3-4b")

// Specify a quantization variant
m, err := zerfoo.Load("google/gemma-3-4b/Q8_0")

// Or load a local GGUF file
m, err := zerfoo.Load("./models/gemma-3-1b.gguf")
```

## Streaming

Stream tokens as they are generated via a channel:

```go
m, _ := zerfoo.Load("google/gemma-3-4b")
defer m.Close()

ch, err := m.ChatStream(context.Background(), "Tell me a joke.")
if err != nil {
    log.Fatal(err)
}
for tok := range ch {
    if !tok.Done {
        fmt.Print(tok.Text)
    }
}
fmt.Println()
```

## Embeddings

Extract L2-normalized embeddings and compute similarity:

```go
m, _ := zerfoo.Load("google/gemma-3-4b")
defer m.Close()

embeddings, _ := m.Embed([]string{
    "Go is a statically typed language.",
    "Rust has a borrow checker.",
})
score := embeddings[0].CosineSimilarity(embeddings[1])
fmt.Printf("similarity: %.4f\n", score)
```

## Structured Output

Constrain model output to valid JSON matching a schema:

```go
import "github.com/zerfoo/zerfoo/generate/grammar"

m, _ := zerfoo.Load("google/gemma-3-4b")
defer m.Close()

schema := grammar.JSONSchema{
    Type: "object",
    Properties: map[string]*grammar.JSONSchema{
        "name": {Type: "string"},
        "age":  {Type: "number"},
    },
    Required: []string{"name", "age"},
}

result, _ := m.Generate(context.Background(),
    "Generate a person named Alice who is 30.",
    zerfoo.WithSchema(schema),
)
fmt.Println(result.Text) // {"name": "Alice", "age": 30}
```

## Tool Calling

Detect tool/function calls in model output (OpenAI-compatible):

```go
import "github.com/zerfoo/zerfoo/serve"

m, _ := zerfoo.Load("google/gemma-3-4b")
defer m.Close()

tools := []serve.Tool{{
    Type: "function",
    Function: serve.ToolFunction{
        Name:        "get_weather",
        Description: "Get the current weather for a city",
        Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
    },
}}

result, _ := m.Generate(context.Background(),
    "What is the weather in Paris?",
    zerfoo.WithTools(tools...),
)

for _, tc := range result.ToolCalls {
    fmt.Printf("call %s(%s)\n", tc.FunctionName, tc.Arguments)
}
```

## Supported Models

### LLM Inference (38 model builders, 45 architecture strings)

The registry maps 45 GGUF architecture strings onto 38 distinct builder
functions (aliases such as `gemma`/`gemma3` and `phi`/`phi3` share a builder).
**Registration is not verification.** A builder means the architecture is
recognized and graphed, not that output has been checked against a reference.
[`docs/verified-models.md`](docs/verified-models.md) is the authority on which
models are actually verified, and it lists far fewer than 38.

| Architecture | Format | Special Features |
|-------------|--------|-----------------|
| Gemma 3 | GGUF Q4_K | Production. CUDA graph capture, 235 tok/s (1B) |
| Gemma 3n | GGUF | Mobile-optimized variant |
| Llama 3 | GGUF | RoPE theta=500K |
| Llama 4 | GGUF | Latest generation |
| Mistral | GGUF | Sliding window attention, 44 tok/s (7B Q5_K_M) |
| Mixtral | GGUF | Mixture of Experts |
| Qwen 2 | GGUF | Attention bias, RoPE theta=1M |
| Qwen 3 (dense) | GGUF | QK RMSNorm, no attention bias, explicit head dim |
| Phi 3/4 | GGUF | Partial rotary factor, Q2_K/Q3_K support |
| DeepSeek V3 | GGUF | MLA + MoE (batched) |
| Command R | GGUF | Cohere architecture |
| Falcon | GGUF | Multi-query attention |
| RWKV | GGUF | Linear attention |
| GPT-2 | GGUF | TinyStories, learned position embeddings |
| Nemotron-H | GGUF | Hybrid Mamba-2 + Attention (NVIDIA) |
| Nemotron-Cascade-2 | GGUF | Hybrid Mamba-2 + Attention + MoE (30B-A3B) |
| MiniMax M2 | GGUF | Sigmoid MoE (256 experts), QK norm |
| OLMo 2 | GGUF | AI2 open language model |
| InternLM 2 | GGUF | Shanghai AI Lab |
| EXAONE | GGUF | LG AI Research |
| StarCoder 2 | GGUF | Code generation, sliding window |
| DBRX | GGUF | Fine-grained MoE (16 experts, top-4) |
| GLM-4 / ChatGLM | GGUF | Zhipu AI, dense + MoE variants |
| Kimi K2 | GGUF | Linear attention MoE (Moonshot AI) |
| LFM2 | GGUF | Liquid Foundation Model, hybrid MoE |
| Mamba / Mamba 3 | GGUF | State space models (MIMO SSM) |
| Jamba | GGUF | Hybrid Mamba-Transformer |
| Whisper | GGUF | Audio transcription |
| Voxtral | GGUF | Mistral speech-to-text (encoder-projector-decoder) |
| LLaVA | GGUF | Vision-language |
| Qwen-VL | GGUF | Vision-language |

New architectures are auto-detected from GGUF metadata.

### Tabular ML

| Architecture | Package | Use Case |
|-------------|---------|----------|
| MLP / Ensemble | `tabular` | Baseline tabular prediction |
| FTTransformer | `tabular` | Attention-based tabular |
| TabNet | `tabular` | Attentive feature selection |
| SAINT | `tabular` | Self-attention + inter-sample |
| TabResNet | `tabular` | Residual tabular networks |

### Time-Series Forecasting

| Architecture | Package | Use Case |
|-------------|---------|----------|
| TFT | `timeseries` | Temporal Fusion Transformer |
| N-BEATS | `timeseries` | Basis expansion forecasting |
| PatchTST | `timeseries` | Patch-based transformer |

### IBM Granite Time Series

| Architecture | Format | Use Case |
|-------------|--------|----------|
| Granite TTM | GGUF | Zero-shot/few-shot time series forecasting |
| Granite FlowState | GGUF | Continuous forecasting, timescale-invariant |
| Granite TSPulse | GGUF | Anomaly detection, classification, imputation |

Granite Time Series models are converted from HuggingFace using `granite2gguf`
(part of `zonnx`). Supported tasks: forecasting, anomaly detection,
classification, imputation, and embedding extraction.

## Training

Train tabular and time-series models with built-in AdamW, learning rate schedulers, and early stopping:

```go
import "github.com/zerfoo/zerfoo/tabular"

model := tabular.NewEnsemble[float32](engine, tabular.EnsembleConfig{
    InputDim:  10,
    OutputDim: 1,
    Models:    3,
})
trainer := tabular.NewTrainer(model, engine, tabular.TrainerConfig{
    LR:     0.001,
    Epochs: 50,
})
trainer.Fit(ctx, trainX, trainY)
predictions, _ := model.Predict(ctx, testX)
```

### Fused SDPA graph node

`layers/attention.FusedSDPA[T]` wraps the existing `ScaledDotProductAttention`
as a `graph.Node[T]` so callers can compose fused scale + softmax + matmul
attention inside autograd graphs without re-implementing the math:

```go
import (
    "github.com/zerfoo/zerfoo/layers/attention"
)

// Causal (decoder) SDPA, head_dim=64.
sdpa := attention.NewFusedSDPA[float32](engine, 64)

// Bidirectional (encoder) SDPA with explicit Q/KV head counts.
enc := attention.NewFusedSDPA[float32](engine, 64,
    attention.WithFusedSDPABidirectional[float32](),
    attention.WithFusedSDPAHeadCounts[float32](8, 8),
)

// Forward accepts (Q, K, V) or (Q, K, V, mask); Backward returns gradients
// for [Q, K, V] (and a nil slot for mask when one was supplied).
out, err := sdpa.Forward(ctx, q, k, v)
```

The node is numerically equivalent to the unfused
`Q @ Kᵀ → scale → softmax → dropout → V` chain (fp32 ≤ 1e-5 fwd / 1e-5 bwd,
fp64 ≤ 1e-12 fwd / 1e-10 bwd; see `layers/attention/fused_sdpa_node_test.go`).

## CLI

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
```

| Command             | Description                                                             |
|---------------------|-------------------------------------------------------------------------|
| `predict`           | Perform model inference on data using configurable model and data providers |
| `tokenize`          | Tokenize text using the Zerfoo tokenizer                                |
| `worker`            | Start a distributed training worker                                     |
| `pull`              | Download and cache a model                                              |
| `list`              | List cached models                                                      |
| `rm`                | Remove a cached model                                                   |
| `run`               | Run interactive chat with a model                                       |
| `serve`             | Start an OpenAI-compatible inference server                             |
| `version`           | Print the Zerfoo version                                                |
| `automl`            | Run automated hyperparameter optimization                               |
| `train`             | Train a model locally or distributed across multiple GPUs               |
| `guard`             | Evaluate content safety using Granite Guardian                          |
| `sentiment`         | Run sentiment classification on text                                    |
| `finetune-sentiment`| Fine-tune a sentiment classification model                              |
| `transmla`          | Convert MHA GGUF weights to multi-head latent attention (MLA) via truncated SVD |
| `eagle-train`       | Train an EAGLE speculative decoding head                                |
| `transcribe`        | Transcribe audio to text using a speech-to-text model                   |
| `transmla-validate` | Compare perplexity between original and TransMLA-converted models       |

## Examples

See the [`examples/`](examples/) directory for runnable programs:

- **[chat](examples/chat/)** -- interactive chatbot CLI
- **[rag](examples/rag/)** -- retrieval-augmented generation with embeddings
- **[json-output](examples/json-output/)** -- grammar-guided structured JSON output
- **[embedding](examples/embedding/)** -- embed inference in an HTTP server
- **[api-server](examples/api-server/)** -- standalone API server
- **[inference](examples/inference/)** -- basic text generation
- **[streaming](examples/streaming/)** -- token streaming
- **[fine-tuning](examples/fine-tuning/)** -- LoRA fine-tuning
- **[automl](examples/automl/)** -- automated model selection
- **[timeseries](examples/timeseries/)** -- time-series forecasting
- **[distributed-training](examples/distributed-training/)** -- multi-node training
- **[agentic-tool-use](examples/agentic-tool-use/)** -- function calling agent
- **[audio-transcription](examples/audio-transcription/)** -- Whisper transcription

## Documentation

Full documentation at **[zerfoo.feza.ai/docs/](https://zerfoo.feza.ai/docs/)**

- **[Getting Started](https://zerfoo.feza.ai/docs/getting-started/installation/)** -- install, pull a model, run inference
- **[Tutorials](https://zerfoo.feza.ai/docs/tutorials/)** -- step-by-step guides
- **[API Reference](https://zerfoo.feza.ai/docs/api/)** -- generate, inference, serve APIs
- **[Cookbooks](https://zerfoo.feza.ai/docs/cookbooks/)** -- 12 runnable code recipes
- **[Architecture](https://zerfoo.feza.ai/docs/architecture/)** -- GPU setup, architecture overview
- **[Benchmarks](docs/benchmarks.md)** -- every throughput figure, its evidence, its methodology, and what is explicitly not claimed
- **[Verified models](docs/verified-models.md)** -- which architectures are actually verified, and against what
- **[Blog](https://zerfoo.feza.ai/docs/blog/)** -- development updates and deep dives
- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- how to contribute

## License

Apache 2.0
