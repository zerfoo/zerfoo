# RAG (Retrieval-Augmented Generation) Example

Demonstrates the retrieval-augmented generation pattern using Zerfoo. A hardcoded 5-document corpus is embedded, a query is matched against the corpus via cosine similarity, and the top-3 most relevant documents are passed as context to the model for generation.

## What is RAG?

Retrieval-Augmented Generation grounds a language model's responses in specific documents rather than relying solely on its training data. The pattern has three steps:

1. **Embed** — Convert documents and the query into vector embeddings using the model.
2. **Retrieve** — Find the most relevant documents by comparing embedding similarity.
3. **Generate** — Pass the retrieved documents as context in the prompt, so the model answers based on the provided facts.

This approach reduces hallucination and lets you inject domain-specific knowledge without fine-tuning.

## Prerequisites

- Go 1.25+
- A GGUF model file (e.g., Gemma 3 1B or Llama 3.2 1B)

### Downloading a test model

```bash
pip install huggingface-hub

huggingface-cli download google/gemma-3-1b-it-qat-q4_0-gguf \
  --local-dir ./models
```

## Build

```bash
go build -o rag ./examples/rag/
```

## Run

```bash
./rag --model ./models/gemma-3-1b-it-qat-q4_0.gguf
```

With a custom query:

```bash
./rag --model ./models/gemma-3-1b-it-qat-q4_0.gguf \
  --query "What algorithm does Go's garbage collector use?"
```

## Example output

```
Top-3 documents (by similarity):
  1. [0.9234] Go's garbage collector uses a concurrent tri-color mark-and-sweep algorithm.
  2. [0.7812] Go uses goroutines for concurrency, which are lightweight threads managed by the Go runtime.
  3. [0.7456] Go was created at Google in 2009 by Robert Griesemer, Rob Pike, and Ken Thompson.

Go's garbage collector uses a concurrent tri-color mark-and-sweep algorithm...
```

## How it works

1. The model is loaded via `zerfoo.Load()`, which accepts a local GGUF path or a HuggingFace model ID.
2. `model.Embed()` computes vector embeddings for each corpus document and the query.
3. `embedding.CosineSimilarity()` ranks corpus documents by relevance to the query.
4. The top-3 documents are injected into a prompt as context, and `model.Chat()` generates an answer grounded in those facts.

In a production system, the corpus would be stored in a vector database and the embedding step would happen at index time rather than query time.
