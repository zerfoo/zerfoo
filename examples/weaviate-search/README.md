# Semantic Search with Zerfoo Embeddings

This example demonstrates using Zerfoo's embedding API to perform
cosine-similarity semantic search over a small document corpus. The
`integrations/weaviate` adapter wraps the `/v1/embeddings` endpoint and
produces `[]float32` vectors suitable for insertion into Weaviate or any
other vector database.

## Prerequisites

- A compiled `zerfoo` binary (or `go run ./cmd/zerfoo`)
- An embedding model in GGUF format (e.g. nomic-embed-text)

## Setup

1. Start the Zerfoo server with an embedding model:

```bash
zerfoo serve --model path/to/nomic-embed-text.gguf --port 8080
```

2. Run the search example:

```bash
go run ./examples/weaviate-search/ \
  --server http://localhost:8080 \
  --model nomic-embed-text \
  --query "How do I run ML inference in Go?" \
  --top-k 3
```

## How It Works

The adapter (`integrations/weaviate.Adapter`) sends text to Zerfoo's
`/v1/embeddings` endpoint and returns `[]float32` vectors. This example
embeds a hardcoded corpus of 8 documents, then embeds the query and ranks
documents by cosine similarity.

```go
emb := weaviate.NewAdapter("http://localhost:8080", "nomic-embed-text")

// Embed documents
docVecs, err := emb.EmbedDocuments(ctx, documents)

// Embed a query
queryVec, err := emb.EmbedQuery(ctx, "How do I run ML inference in Go?")
```

In production, you would store `docVecs` in Weaviate using its Go client
and use `queryVec` for near-vector searches.

## Using with Weaviate

To insert vectors into a live Weaviate instance:

```go
import "github.com/weaviate/weaviate-go-client/v4/weaviate"

client, _ := weaviate.NewClient(weaviate.Config{Host: "localhost:8080", Scheme: "http"})

// Use the adapter to get vectors, then batch-import into Weaviate
vecs, _ := emb.EmbedDocuments(ctx, documents)
for i, doc := range documents {
    client.Data().Creator().
        WithClassName("Document").
        WithProperties(map[string]any{"text": doc}).
        WithVector(vecs[i]).
        Do(ctx)
}
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://localhost:8080` | Zerfoo server URL |
| `--model` | `nomic-embed-text` | Embedding model name |
| `--query` | `How do I run ML inference in Go?` | Search query |
| `--top-k` | `3` | Number of results to return |
