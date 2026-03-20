# Text Embedding Generation

Demonstrates extracting text embedding vectors from a GGUF model using the `inference` package and computing semantic similarity between texts.

## Prerequisites

- Go 1.25+
- A GGUF model file (any supported architecture: Llama, Gemma, Mistral, Qwen, Phi)

### Downloading a test model

```bash
pip install huggingface-hub

huggingface-cli download google/gemma-3-1b-it-qat-q4_0-gguf \
  --local-dir ./models
```

## Build

```bash
go build -o text-embedding ./examples/text-embedding/
```

## Run

```bash
./text-embedding ./models/gemma-3-1b-it-qat-q4_0.gguf
```

With GPU acceleration:

```bash
./text-embedding -device cuda ./models/gemma-3-1b-it-qat-q4_0.gguf
```

## Expected output

```
=== Text Embedding Example ===

Loading model: ./models/gemma-3-1b-it-qat-q4_0.gguf
Device: cpu

Model: gemma (18 layers, hidden=1536, vocab=262144)

--- Generating Embeddings ---
Text 0: "The cat sat on the mat." -> 1536-dim embedding (first 5: [...])
Text 1: "A kitten rested on the rug." -> 1536-dim embedding (first 5: [...])
Text 2: "Stock prices rose sharply today." -> 1536-dim embedding (first 5: [...])
Text 3: "The financial markets surged in early trading." -> 1536-dim embedding (first 5: [...])

--- Cosine Similarity Matrix ---
                           Text 0  Text 1  Text 2  Text 3
  Text 0 (The cat sat on th...)  1.0000  0.8xxx  0.3xxx  0.2xxx
  Text 1 (A kitten rested o...)  0.8xxx  1.0000  0.3xxx  0.2xxx
  Text 2 (Stock prices rose...)  0.3xxx  0.3xxx  1.0000  0.8xxx
  Text 3 (The financial mar...)  0.2xxx  0.2xxx  0.8xxx  1.0000

--- Interpretation ---
Texts 0-1 (cats) should have high similarity.
Texts 2-3 (finance) should have high similarity.
Cross-topic pairs should have lower similarity.

=== Done ===
```

## How it works

1. **Load** -- `inference.LoadFile` loads the GGUF model file into memory (with optional mmap)
2. **Embed** -- `model.Embed(text)` tokenizes the input, looks up token embeddings from the model's embedding table, mean-pools across tokens, and L2-normalizes the result
3. **Compare** -- cosine similarity between L2-normalized vectors is just their dot product

The embedding dimension matches the model's hidden size (e.g., 1536 for Gemma 3 1B).

## Key APIs

| Function/Type | Package | Purpose |
|---------------|---------|---------|
| `inference.LoadFile` | `inference/` | Load a GGUF model with device/dtype options |
| `model.Embed(text)` | `inference/` | Generate L2-normalized embedding vector |
| `model.EmbeddingWeights()` | `inference/` | Access raw embedding table and hidden size |
| `model.Config()` | `inference/` | Model metadata (architecture, layers, vocab) |

## Use cases

- **Semantic search**: embed queries and documents, find nearest neighbors by cosine similarity
- **Clustering**: group similar texts by embedding distance
- **RAG**: embed chunks of a knowledge base for retrieval-augmented generation
- **Deduplication**: detect near-duplicate texts by high cosine similarity
