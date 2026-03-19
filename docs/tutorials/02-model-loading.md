# Part 2: Model Loading and Architecture Support

This tutorial covers the GGUF model format, the architectures Zerfoo supports, how to load models programmatically with various options, and what quantization levels mean for memory and quality.

## The GGUF Format

GGUF (GPT-Generated Unified Format) is a single-file model format designed for efficient inference. A GGUF file contains everything needed to run a model:

- **Metadata**: architecture name, vocabulary size, hidden dimensions, RoPE parameters, chat template, and more.
- **Tokenizer**: the full BPE vocabulary and merge rules embedded in the file's metadata section.
- **Tensors**: all model weights, stored in their quantized or full-precision representation with shape information.

Zerfoo uses GGUF as its sole model format. When you call `inference.LoadFile`, the framework parses the GGUF header, extracts the tokenizer, reads the architecture metadata, and builds a typed computation graph -- all without any external config files.

```go
model, err := inference.LoadFile("path/to/model.gguf")
```

GGUF files are mmap-friendly. On Unix platforms, you can enable memory-mapped loading to avoid copying weights into the Go heap:

```go
model, err := inference.LoadFile("model.gguf",
	inference.WithMmap(true),
)
```

## Supported Architectures

Zerfoo includes architecture-specific graph builders for each model family. The architecture is detected automatically from GGUF metadata -- you do not need to specify it.

| Architecture | Key Features | Example Models |
|-------------|-------------|----------------|
| Llama 3 | RoPE theta=500K, GQA | Llama 3.2 1B/3B, Llama 3.1 8B/70B |
| Llama 4 | Extended Llama architecture | Llama 4 Scout |
| Gemma 3 | Tied embeddings, embedding scaling, QK norms, logit softcap | Gemma 3 1B/4B/12B/27B |
| Gemma 3n | Gemma 3 nano variant | Gemma 3n |
| Mistral | Sliding window attention | Mistral 7B v0.3 |
| Mixtral | Mixture of experts (MoE) with sliding window | Mixtral 8x7B |
| Qwen 2 | Attention bias, RoPE theta=1M | Qwen 2.5 7B/14B/72B |
| Phi 3/4 | Partial rotary factor | Phi-3 Mini, Phi-4 |
| DeepSeek V3 | Multi-head Latent Attention (MLA), batched MoE | DeepSeek V3 |
| Falcon | Multi-query attention | Falcon 7B/40B |
| Command-R | Retrieval-augmented generation architecture | Command-R |
| Jamba | Hybrid Mamba-Transformer architecture | Jamba |
| Mamba/Mamba3 | State-space model (SSM), no attention | Mamba |
| LLaVA | Vision-language multimodal | LLaVA |

Each architecture has a dedicated builder in the `inference/` package (e.g., `arch_llama.go`, `arch_gemma.go`, `arch_deepseek.go`). The builder reads architecture-specific metadata fields and constructs the computation graph with the correct layer structure, attention mechanism, and normalization.

## Loading Models Programmatically

The `inference.LoadFile` function accepts functional options that control device placement, precision, and sequence length.

### Device Selection

```go
// CPU inference (default).
model, err := inference.LoadFile("model.gguf")

// CUDA GPU inference.
model, err := inference.LoadFile("model.gguf",
	inference.WithDevice("cuda"),
)
```

### Compute Precision

```go
// FP16 compute -- activations are converted F32->FP16 before GPU kernels.
model, err := inference.LoadFile("model.gguf",
	inference.WithDevice("cuda"),
	inference.WithDType("fp16"),
)

// FP8 quantization -- weights are quantized to FP8 E4M3 at load time.
model, err := inference.LoadFile("model.gguf",
	inference.WithDevice("cuda"),
	inference.WithDType("fp8"),
)
```

### Sequence Length

Override the model's default maximum context length:

```go
model, err := inference.LoadFile("model.gguf",
	inference.WithMaxSeqLen(4096),
)
```

### TensorRT Backend

For maximum throughput on NVIDIA GPUs, enable the TensorRT backend:

```go
model, err := inference.LoadFile("model.gguf",
	inference.WithDevice("cuda"),
	inference.WithBackend("tensorrt"),
	inference.WithPrecision("fp16"),
)
```

### Model Aliases

Zerfoo maintains a table of short aliases for popular HuggingFace repositories. You can resolve an alias to its full repo ID or register your own:

```go
// Resolves "gemma-3-1b-q4" -> "google/gemma-3-1b-it-qat-q4_0-gguf"
repoID := inference.ResolveAlias("gemma-3-1b-q4")

// Register a custom alias.
inference.RegisterAlias("my-model", "myorg/my-model-GGUF")
```

## Understanding Quantization

Quantization reduces model weights from 16- or 32-bit floats to lower-precision integers, trading a small amount of quality for significant memory savings and faster inference.

Common GGUF quantization types:

| Type | Bits/Weight | Memory (7B model) | Quality | Use Case |
|------|------------|-------------------|---------|----------|
| F16 | 16 | ~14 GB | Baseline | Full quality, GPU with ample VRAM |
| Q8_0 | 8 | ~7 GB | Near-lossless | Best quality-to-size ratio |
| Q4_K_M | ~4.5 | ~4 GB | Good | Recommended default for most users |
| Q4_0 | 4 | ~3.5 GB | Acceptable | Minimum viable quality |

The quantization type is baked into the GGUF file at conversion time. Zerfoo reads the quantization metadata from each tensor and applies the correct dequantization during inference. You do not need to specify the quantization type at load time.

For a 1B parameter model like Gemma 3 1B with Q4_K_M quantization, expect roughly 800 MB of memory usage -- small enough to run on a laptop CPU.

## Inspecting Model Metadata

After loading a model, you can access its metadata:

```go
model, err := inference.LoadFile("model.gguf")
if err != nil {
	log.Fatal(err)
}
defer model.Close()

info := model.Info()
fmt.Printf("Architecture: %s\n", info.Architecture)
fmt.Printf("Parameters: %d\n", info.Parameters)
```

## What is Next

- [Part 3: Text Generation Deep Dive](03-text-generation.md) -- sampling strategies, streaming, and performance tuning.
- [Part 4: Running the OpenAI-Compatible API Server](04-api-server.md) -- serve models over HTTP.
- [Part 5: Tabular and Time-Series ML](05-tabular-timeseries.md) -- structured data and forecasting.
