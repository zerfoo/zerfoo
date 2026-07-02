# ADR 047: Disaggregated Prefill/Decode Serving

## Status
Accepted

## Date
2026-03-17

## Context
In standard serving, prefill (processing the input prompt) and decode (generating
output tokens one at a time) share GPU resources. Prefill is compute-bound and
latency-sensitive (time-to-first-token); decode is memory-bandwidth-bound and
throughput-sensitive. Mixing them causes prefill to steal GPU cycles from decode,
inflating TTFT, and decode to inflate prefill queue depth.

SGLang demonstrated 3.8x prefill / 4.8x decode throughput improvement on GB200
NVL72 using disaggregated serving with RDMA-based KV transfer. This architecture
is now table-stakes for high-throughput serving in 2026.

## Decision
Implement disaggregated serving in serve/disaggregated/ with two worker roles:

Prefill Worker (serve/disaggregated/prefill_worker.go):
- Receives prompt text via gRPC (PreFillRequest)
- Runs full prefill forward pass, produces KV cache for all prompt tokens
- Transfers KV cache blocks to decode worker via gRPC stream (KVBlockStream)
- KV blocks are serialized as raw FP16 bytes; no compression initially
- One or more prefill workers per serving cluster

Decode Worker (serve/disaggregated/decode_worker.go):
- Receives KV block stream from prefill worker
- Reconstructs KV cache in local block manager
- Runs autoregressive decode until EOS or max_tokens
- Streams tokens back to API gateway via SSE

API Gateway (serve/gateway.go):
- Routes requests to least-loaded prefill worker
- Multiplexes decode stream back to client
- Health-checks both worker pools with exponential backoff

KV transfer uses existing gRPC infrastructure. RDMA (UCX/NCCL) is deferred to
Year 2 when multi-node deployment is required.

## Consequences
Positive:
- Prefill and decode can be scaled independently (more prefill GPUs for chatty
  workloads, more decode GPUs for long-generation workloads)
- Eliminates prefill-decode interference; reduces P99 TTFT by 40-60%
- Enables heterogeneous hardware (fast Blackwell for prefill, cheaper Hopper for decode)

Negative:
- KV transfer over gRPC adds 10-50ms latency per request (vs zero in collocated mode)
- Stateful decode workers are harder to scale horizontally; sticky routing needed
- Increased operational complexity: two worker pools to deploy, monitor, and scale
- Initial implementation targets single-machine (shared memory); gRPC adds overhead
  only in multi-machine deployment
