# Embedded retrieval

`retrieval` indexes arbitrary text records in process. It has no dependency on
an agent runtime, network service, catalog, or vector database.

```go
docs := []retrieval.Document{
    {ID: "record-1", Text: "Parse and validate JSON", Body: "Full record content"},
    {ID: "record-2", Text: "Format a spreadsheet", Body: "Other content"},
}
index, err := retrieval.NewIndex(ctx, docs, retrieval.Options{})
if err != nil { return err }
matches, err := index.Search(ctx, "validate a JSON document", 5)
if err != nil { return err }
if len(matches) > 0 {
    full, ok := index.Get(matches[0].Document.ID)
    _ = full
    _ = ok
}
```

The default is local BM25. Supply an `Embedder` to add vector retrieval and
reciprocal-rank fusion. Its separate `EmbedDocuments` and `EmbedQuery` methods
allow role-specific formatting; document calls are batched (64 by default).
`MinDenseSimilarity` defaults to zero and excludes candidates with cosine
similarity at or below the threshold. Supply a `Reranker` to reorder the shortlist. Rerankers
receive full candidate records. Search results omit `Body`; call `Get` after
selection. The index is immutable; build
a new one when records change. The exact dense scan is intended as a baseline
for catalogs of roughly tens of thousands of records, pending measured latency.

`Evaluate` accepts generic query/relevant-ID labels and reports retrieval
metrics at a specified cutoff. It does not choose dataset splits or establish
model quality by itself. Every relevant ID must exist in the index. The current `zerfoo.Model.Embed` is a non-contextual
embedding-table average, so it should not be used as evidence of neural
retriever quality. See [ADR 099](../docs/adr/099-embedded-capability-retrieval.md)
and the [work plan](../docs/plan-embedded-retrieval.md).
