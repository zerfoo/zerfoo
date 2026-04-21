# zerfoo

Pure Go ML framework -- inference, training, and serving. Embed any GGUF model in your Go application with `go build ./...`.

[![CI](https://github.com/zerfoo/zerfoo/actions/workflows/ci.yml/badge.svg)](https://github.com/zerfoo/zerfoo/actions/workflows/ci.yml)
[![Go 1.26+](https://img.shields.io/badge/Go-1.26+-00ADD8.svg)](https://go.dev/)
[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/zerfoo.svg)](https://pkg.go.dev/github.com/zerfoo/zerfoo)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**241 tok/s** on Gemma 3 1B Q4_K_M -- up to 28% faster than Ollama. Faster on all 4 benchmarked models. Zero CGo. 41 model architectures (25 families). **Run models larger than RAM** via memory-mapped I/O (229B MiniMax-M2 on 128 GB). EAGLE speculative decoding with built-in head training, QuaRot quantization, Q4_K fused GEMV (14x faster), Multi-LoRA serving, BitNet ternary inference. CUDA graph capture, Apple Metal kernels. Time-series training 4.6x faster with CUDA graphs. Tabular ML and time-series forecasting built in.

### Benchmarks

Decode throughput comparison against [Ollama](https://ollama.com/) on NVIDIA DGX Spark GB10 (Grace Blackwell, sm_121, 128 GB LPDDR5x).

| Model | Size | Quant | Zerfoo (tok/s) | Ollama (tok/s) | Ratio |
|-------|------|-------|----------------|----------------|-------|
| Gemma 3 1B | 1B | Q4_K_M | **241** | 188 | **1.28x** |
| DeepSeek R1 1.5B | 1.5B | Q4_K_M | **190** | 174 | **1.09x** |
| Llama 3.2 3B | 3B | Q4_K_M | **95** | 93 | **1.02x** |
| Mistral 7B | 7B | Q5_K_M | **46** | 45 | **1.02x** |

Faster than Ollama on all models. Up to 28% faster on small models, 2% faster at 7B.

<details>
<summary>Methodology</summary>

- **Hardware**: NVIDIA DGX Spark GB10 (Grace Blackwell, sm_121, 128 GB LPDDR5x unified memory)
- **Prompt**: "Explain the theory of relativity in simple terms."
- **Tokens**: 128 decode tokens per run
- **Sampling**: greedy (temperature = 0)
- **Runs**: 3-run median
- **Date**: 2026-03-31
- **Ollama version**: 0.17.7
- **Zerfoo version**: v1.38.4+ (ztensor v1.1.2+, 7 GPU regression fixes)
- **Notes**: Zerfoo uses CUDA graph capture (184/185 instructions, 99.5%) with flash attention decode. Fused kernels: softmax+V multiply, repeat-interleave for GQA, fused AddRMSNorm, fused SwiGLU, fused QKNormRoPE, merged QKV, merged gate+up. Auto-disable mmap on CUDA for ARM64 compatibility. Q4_K/Q5_K/Q6_K/Q5_0 weights re-quantized to Q4_0 for fast vectorized GEMV.

</details>

### Memory-Mapped Model Loading

Zerfoo memory-maps GGUF files by default — no flags, no configuration. The entire file (or all shards of a split GGUF) is mmap'd via `syscall.Mmap`. Tensor data stays on disk and is paged into RAM on demand by the OS. Split GGUF files (multiple shards) are detected and mapped automatically from any shard path.

**Results on DGX Spark (128 GB RAM, CPU-only):**

| Model | Params | Quant | File Size | Shards | Load time | Generates tokens | Ollama |
|-------|--------|-------|-----------|--------|-----------|-----------------|--------|
| MiniMax-M2 | 229B (MoE) | Q4_K_M | 128.8 GB | 3 | **6.3s** | ✅ yes | ❌ fails to load |

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

### EAGLE Speculative Decoding

Self-speculative decoding using a lightweight prediction head — no draft model needed. Based on [EAGLE-3](https://arxiv.org/abs/2503.01840).

```go
m, _ := zerfoo.Load("google/gemma-3-1b")
defer m.Close()
result, _ := m.Generate(ctx, "Explain quantum computing.",
    zerfoo.WithEAGLE("eagle-head.gguf"),
)
```

Train your own EAGLE head:

```bash
zerfoo eagle-train --model model.gguf --corpus data.txt --output eagle-head.gguf --epochs 5
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

### Tiered KV Cache

Automatically spill KV cache across three storage tiers as sequences grow — no OOM, no manual tuning:

- **Hot**: uncompressed tensors in GPU/CPU memory (recent tokens)
- **Warm**: compressed in CPU memory via block quantization
- **Cold**: serialized to disk as binary files (oldest tokens)

Layers are promoted and demoted automatically based on access frequency. Async prefetch moves cold layers back to hot before the decoder needs them.

```go
result, _ := m.Generate(ctx, prompt,
    zerfoo.WithTieredKV(generate.TieredKVStoreConfig{
        ChunkSize:        64,  // warm-tier compression chunk size
        DemoteThreshold:  2,   // demote layers accessed < 2 times
        PromoteThreshold: 8,   // promote layers accessed > 8 times
        // ColdDir: "/var/cache/kv" // optional: persist cold tier across calls
    }),
)
```

Enable it on the Model API via `zerfoo.WithTieredKV` (wraps `generate.WithTieredKV`). Useful for long-context inference where the KV cache exceeds GPU memory.

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

### Audio Transcription

Transcribe WAV audio to text using Whisper or Voxtral speech-to-text models. Audio is chunked into 30-second segments, mel spectrograms are extracted, and each chunk is decoded in parallel:

```go
wavData, _ := os.ReadFile("speech.wav")

m, _ := zerfoo.Load("openai/whisper-large-v3")
defer m.Close()

text, err := m.Transcribe(context.Background(), wavData)
fmt.Println(text)
```

Supports 16 kHz mono WAV input. Whisper uses 80 mel bins; Voxtral uses 128. Long audio is automatically chunked into 30-second segments and concatenated.

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

### LLM Inference (41 architectures, 25 model families)

| Architecture | Format | Special Features |
|-------------|--------|-----------------|
| Gemma 3 | GGUF Q4_K | Production. CUDA graph capture, 241 tok/s |
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
- **[Benchmarks](https://zerfoo.feza.ai/docs/reference/benchmarks/)** -- throughput numbers
- **[Blog](https://zerfoo.feza.ai/docs/blog/)** -- development updates and deep dives
- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- how to contribute

## License

Apache 2.0
