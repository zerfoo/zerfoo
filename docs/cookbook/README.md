# Zerfoo Cookbook

Practical, copy-paste recipes for common tasks with the Zerfoo ML framework.

Each recipe is self-contained and can be adapted to your use case. Recipes use
either the high-level `zerfoo` package (one-line API) or the lower-level
`inference`, `generate`, `serve`, and `training` packages for full control.

## Recipes

| # | Recipe | API Level | Description |
|---|--------|-----------|-------------|
| 1 | [Basic Text Generation](01-basic-text-generation/) | High | Load a model and generate text with a single function call |
| 2 | [Streaming Chat](02-streaming-chat/) | High | Stream tokens to the terminal as they are generated |
| 3 | [Embedding and Similarity](03-embedding-similarity/) | High | Compute embeddings and rank documents by cosine similarity |
| 4 | [OpenAI-Compatible Server](04-openai-server/) | Low | Serve a model behind an OpenAI-compatible HTTP API |
| 5 | [Custom Sampling](05-custom-sampling/) | Low | Fine-tune temperature, top-K, top-P, and repetition penalty |
| 6 | [Structured JSON Output](06-structured-json-output/) | Both | Constrain model output to a JSON schema using grammar-guided decoding |
| 7 | [Fine-Tuning with LoRA](07-lora-fine-tuning/) | Low | Inject LoRA adapters and train on custom data |
| 8 | [Batch Inference](08-batch-inference/) | High | Run inference over many prompts concurrently |
| 9 | [Speculative Decoding](09-speculative-decoding/) | Low | Use a draft model to accelerate generation |
| 10 | [Tool / Function Calling](10-tool-calling/) | High | Let the model invoke tools and feed results back |
| 11 | [Retrieval-Augmented Generation](11-rag/) | High | Embed, retrieve, and generate with context |
| 12 | [Vision / Multimodal](12-vision-multimodal/) | Low | Analyze images with a vision-capable model |

## Prerequisites

- Go 1.25+
- A GGUF model file (e.g. Gemma 3 1B, Llama 3 8B)
- For GPU recipes: CUDA 12+ or ROCm 6+

## Running a Recipe

Each recipe includes a complete `main.go`. To run one:

```bash
# From the zerfoo repo root:
go run ./docs/cookbook/01-basic-text-generation/main.go --model path/to/model.gguf
```

Or build first:

```bash
go build -o recipe ./docs/cookbook/01-basic-text-generation/
./recipe --model path/to/model.gguf
```
