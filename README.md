# zerfoo

Pure Go ML framework -- inference, training, and serving. Embed any GGUF model in your Go application with `go build ./...`.

[![CI](https://github.com/zerfoo/zerfoo/actions/workflows/ci.yml/badge.svg)](https://github.com/zerfoo/zerfoo/actions/workflows/ci.yml)
[![Go 1.26+](https://img.shields.io/badge/Go-1.26+-00ADD8.svg)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/zerfoo.svg)](https://pkg.go.dev/github.com/zerfoo/zerfoo)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**235 tok/s** on Gemma 3 1B Q4_K_M -- 25% faster than Ollama. Zero CGo. 20 model architectures. EAGLE speculative decoding, QuaRot quantization, Multi-LoRA serving, BitNet ternary inference. CUDA graph capture. Tabular ML and time-series forecasting built in.

### Benchmarks

Decode throughput comparison against [Ollama](https://ollama.com/) on NVIDIA DGX Spark GB10 (Grace Blackwell, sm_121, 128 GB LPDDR5x).

| Model | Size | Quant | Zerfoo (tok/s) | Ollama (tok/s) | Ratio |
|-------|------|-------|----------------|----------------|-------|
| Gemma 3 1B | 1B | Q4_K_M | **235** | 188 | **1.25x** |
| DeepSeek R1 1.5B | 1.5B | Q4_K_M | **186** | 167 | **1.11x** |
| Llama 3.2 3B | 3B | Q4_K_M | 92 | 93 | 0.99x |
| Mistral 7B | 7B | Q5_K_M | 44 | 44 | 1.00x |

25% faster on small models, parity at 7B. All models produce coherent, verified output.

<details>
<summary>Methodology</summary>

- **Hardware**: NVIDIA DGX Spark GB10 (Grace Blackwell, sm_121, 128 GB LPDDR5x unified memory)
- **Prompt**: "Explain the theory of relativity in simple terms."
- **Tokens**: 128 decode tokens per run
- **Sampling**: greedy (temperature = 0)
- **Runs**: 3-run median
- **Date**: 2026-03-27
- **Ollama version**: 0.17.7
- **Notes**: All results verified for coherent output. Zerfoo uses CUDA graph capture with flash attention decode. GQA repeat fix applied (ztensor v0.6.3, zerfoo v1.25.5).

Raw results: [`results/benchmark-2026-03-27.json`](results/benchmark-2026-03-27.json)

</details>

## Advanced Inference Features

### EAGLE Speculative Decoding

Self-speculative decoding using a lightweight prediction head — no draft model needed. Based on [EAGLE-3](https://arxiv.org/abs/2503.01840).

```go
m, _ := zerfoo.Load("google/gemma-3-1b")
defer m.Close()
result, _ := m.Generate(ctx, "Explain quantum computing.",
    zerfoo.WithEAGLE("eagle-head.gguf"),
)
```

### QuaRot Weight Fusion

Hadamard rotation fused into weights at load time for uniform 4-bit quantization. Based on [QuaRot](https://arxiv.org/abs/2404.00456).

```bash
zerfoo run --quarot model.gguf "Hello world"
```

### Quantized KV Cache

Reduce KV cache memory by 6-7x with Q4 or Q3 quantization:

```go
result, _ := m.Generate(ctx, prompt,
    zerfoo.WithKVDtype("q4"),  // 7.5x memory reduction
)
```

### TransMLA — MHA-to-MLA Conversion

Convert any MHA/GQA model to Multi-Head Latent Attention via SVD decomposition. Reduces KV cache by 80%+. Based on [TransMLA](https://arxiv.org/abs/2502.07864).

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

Place shared MoE experts on GPU, offload routed experts to CPU with SIMD kernels. Predictive prefetching achieves 98% hit rate. Based on [KTransformers](https://arxiv.org/abs/2501.14018).

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

### LLM Inference (20 architectures)

| Architecture | Format | Special Features |
|-------------|--------|-----------------|
| Gemma 3 | GGUF Q4_K | Production. CUDA graph capture, 235 tok/s |
| Gemma 3n | GGUF | Mobile-optimized variant |
| Llama 3 | GGUF | RoPE theta=500K |
| Llama 4 | GGUF | Latest generation |
| Mistral | GGUF | Sliding window attention, 44 tok/s (7B Q4_K_M) |
| Mixtral | GGUF | Mixture of Experts |
| Qwen 2 | GGUF | Attention bias, RoPE theta=1M |
| Phi 3/4 | GGUF | Partial rotary factor, Q2_K/Q3_K support |
| DeepSeek V3 | GGUF | MLA + MoE (batched) |
| Command R | GGUF | Cohere architecture |
| Falcon | GGUF | Multi-query attention |
| RWKV | GGUF | Linear attention |
| Mamba / Mamba 3 | GGUF | State space models (MIMO SSM) |
| Jamba | GGUF | Hybrid Mamba-Transformer |
| Whisper | GGUF | Audio transcription |
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

## CLI

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest

zerfoo pull gemma-3-1b-q4              # download a model
zerfoo run gemma-3-1b-q4 "Hello"       # generate text
zerfoo run --quarot model.gguf "Hello" # QuaRot weight fusion
zerfoo serve gemma-3-1b-q4             # OpenAI-compatible API server
zerfoo transmla --input m.gguf --output m-mla.gguf  # MHA→MLA conversion
zerfoo train -backend tabular ...      # train a tabular model
zerfoo list                             # list cached models
```

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
- **[Benchmarks](https://zerfoo.feza.ai/docs/reference/benchmarks/)** -- throughput numbers
- **[Blog](https://zerfoo.feza.ai/docs/blog/)** -- development updates and deep dives
- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- how to contribute

## License

Apache 2.0
