# ADR 099: Generic embedded retrieval in Zerfoo

Status: Accepted for the retrieval interface; model training remains planned.

Date: 2026-09-22

## Context

Go applications need local retrieval over arbitrary text records. Agent skills
and MCP/OpenAPI operations are initial consumers, but the Zerfoo API must make
no assumptions about their formats, authorization rules, or execution. An
agent can separately use one small discovery skill to call search and fetch.
Zatiti is a possible consumer for operation discovery and will validate
bindings, authorization, and schemas after retrieval.

The current public `Model.Embed` implementation averages token embedding-table
rows. It does not run a contextual encoder and cannot be presented as a trained
dense retriever or a ColBERT encoder. `RoutingContrastive` trains attention-head
routing diversity, not query-document relevance.

## Decision

Introduce a generic `retrieval` Go package with immutable documents, local
BM25 search, optional application-supplied dense embeddings, reciprocal-rank
fusion, optional shortlist reranking, and lookup by stable ID. Search returns
short text; fetch returns the body. The package does not execute retrieved
content. Dataset adapters and agent integrations live outside this core API.
At 10,000–50,000 documents an exact vector scan is the
initial dense baseline; add an approximate index only after measured latency
requires one.
The exact scan is a CPU ranking utility over supplied vectors, outside the
trainable tensor graph; its scalar dot products are an explicit exception to
the engine-only rule for model arithmetic. Benchmark it before relying on it
at catalog scale.

The encoder interface must remain separate from `Model.Embed` until Zerfoo has
contextual hidden-state inference with the pooling and query/document prompts
required by a reference retrieval model. Do not silently substitute embedding
table averages. Generic training primitives should learn from labeled
query-record pairs. A one-skill retriever will be a separately trained,
versioned open-weights checkpoint with its own data and model card; its
specialization does not enter the core package. Go applications can load that
checkpoint locally.

## Verification contract

1. Compare BM25, a reference pretrained encoder, hybrid fusion, and reranking
   on SkillRet v1.1, using its disjoint skill split. Report Recall@5/20,
   NDCG@10, completeness for multi-skill requests, no-skill false-positive
   rate on a separate hand-labeled set, p50/p95 latency, and index memory.
2. Train a Zerfoo bi-encoder with a query-document contrastive objective and
   hard negatives, holding out skill sources and API families. Compare its
   embeddings and ranking to a pinned reference implementation on the same
   records. A green loss curve alone is insufficient.
3. Only then evaluate late interaction and a learned reranker against the
   hybrid baseline. Require measured gain large enough to justify memory and
   latency cost. The agent integration must test task completion, not ranking
   metrics alone.
4. Before releasing open weights, publish the exact training data provenance,
   licenses, held-out evaluation, model format, inference example, and known
   failure cases. Keep release approval separate from local training.

## Consequences

The embedded search API is usable immediately with BM25 or a supplied encoder.
The generic in-batch contrastive loss is available for model training graphs.
Native contextual encoder inference, end-to-end retrieval training, model export/load,
catalog ingestion, and production qualification remain explicit work in the
[retrieval plan](../plan-embedded-retrieval.md). This ADR does not claim those
pieces are implemented.
