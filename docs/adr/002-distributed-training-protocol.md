# ADR-002: Distributed Training Protocol

**Status:** Accepted
**Phase:** 5
**Date:** 2026-03-01

## Context

The distributed package had auto-generated protobuf stubs, a coordinator server,
InternalStrategy[T] interface, AllReduceStrategy[T], NetworkManager, and
ServerManager. Missing: concrete DistributedServiceServer, GrpcStrategy[T]
connecting strategy to transport, WorkerNode lifecycle management, and
multi-worker integration tests.

## Decision

### AllReduce Protocol (Star Topology)

Root (rank 0) runs the server collecting gradients from all peers. Each non-root
worker opens a bidi stream to root, sends gradients as AllReduceRequest messages
(one per named tensor), then waits for AllReduceResponse with averaged results.
Root accumulates peer gradients plus its own, computes element-wise average
(sum / world_size), and streams results back.

A `reduceSession` struct coordinates across concurrent AllReduce stream handlers:
collects tensors by name from each peer, waits via sync barrier, computes
reduction, distributes result.

Ring all-reduce optimization was explicitly deferred (correctness first).

### Barrier Protocol

Counter-based. Each worker calls Barrier RPC on root. Root counts arrivals and
blocks via sync.Cond until all workers arrive. Epoch numbers prevent stale
barrier responses.

### Broadcast Protocol

Root stores broadcast tensor in thread-safe map keyed by name. Non-root workers
call Broadcast RPC on root. If tensor not yet available, handler waits with
context deadline.

### Tensor Serialization

pb.Tensor uses repeated float (float32 only). GrpcStrategy[T] converts
tensor.TensorNumeric[T] to/from pb.Tensor. For T=float64, values narrow to
float32 for transport (acceptable for gradient averaging).

### Worker Lifecycle

WorkerNode struct encapsulates: GrpcStrategy, coordinator connection, health
check registration, shutdown.Closer implementation. Start(ctx, cfg) initializes
strategy, registers with coordinator, starts gRPC server, connects to peers.
Close(ctx) triggers orderly shutdown.

CLI `worker` subcommand: --coordinator-address, --worker-address, --worker-id
flags. Signal handling via cli.SignalContext.

## Consequences

- Star topology is simple and correct but O(N) at root. Ring optimization
  deferred to future phase.
- All distributed operations tested end-to-end over real gRPC (bufconn).
- TLS integration tested with self-signed certificates.
- Worker lifecycle integrated with health checks and shutdown coordinator.
- distributed/ package at 96% coverage.

### Key Files

- `distributed/worker_service.go` -- DistributedServiceServer implementation
- `distributed/grpc_strategy.go` -- GrpcStrategy[T] (InternalStrategy over gRPC)
- `distributed/worker_node.go` -- WorkerNode lifecycle
- `distributed/integration_test.go` -- Multi-worker tests (bufconn)
- `cmd/cli/worker.go` -- Worker CLI subcommand
