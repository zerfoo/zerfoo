# Embedding Search

Semantic search over a document corpus using model embeddings and cosine similarity.

This example embeds a set of documents and a user query, then ranks documents by similarity to find the most relevant matches. This is the retrieval component of a RAG (retrieval-augmented generation) pipeline.

## How it works

1. Loads a GGUF model using the `zerfoo.Load` one-line API
2. Embeds all corpus documents with `model.Embed`
3. Embeds the user query
4. Ranks documents by cosine similarity using `Embedding.CosineSimilarity`
5. Prints the top-N most relevant results

## Usage

```bash
go build -o embedding-search ./examples/embedding-search/
./embedding-search --model path/to/model.gguf
./embedding-search --model path/to/model.gguf --query "memory management" --top 5
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to GGUF model file or HuggingFace model ID |
| `--query` | "How does Go handle memory?" | Search query |
| `--top` | 3 | Number of results to display |
