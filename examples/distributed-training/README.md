# Distributed Training with FSDP

Demonstrates Fully Sharded Data Parallelism (FSDP) and gradient accumulation using the `distributed` and `distributed/fsdp` packages.

FSDP shards model parameters across GPUs so each rank holds only `1/worldSize` of each parameter. Before forward, `AllGather` reconstructs full parameters; after backward, `ReduceScatter` aggregates gradient shards.

## What this example shows

1. **Model creation** -- a toy model implementing `training.Model[float32]`
2. **WorkerNode configuration** -- how to configure a distributed worker
3. **FSDP sharding** -- wrapping a model with `fsdp.NewShardedModule`
4. **Gradient accumulation** -- using `fsdp.NewGradAccum` to accumulate gradients across micro-steps before synchronization

## Build

```bash
go build -o distributed-training ./examples/distributed-training/
```

## Run

```bash
./distributed-training
```

## Expected output

```
=== Distributed Training with FSDP Example ===
World size: 2, Gradient accumulation steps: 4

Created toy model with 2 parameters
  linear.weight: shape [64 16] (1024 elements)
  linear.bias: shape [16] (16 elements)

WorkerNode config: worker=localhost:50051 coordinator=localhost:50050 world_size=2

--- FSDP Sharding (simulated, rank 0) ---
In production, each rank creates a ShardedModule with a live NCCLComm:
  sharded := fsdp.NewShardedModule(model, rank, worldSize, comm)
  accum   := fsdp.NewGradAccum(sharded, microSteps)

--- Training Loop (8 micro-steps) ---
  Step 1: accumulated gradients, ready=false
  Step 2: accumulated gradients, ready=false
  Step 3: accumulated gradients, ready=false
  Step 4: accumulated gradients, ready=true
  -> Synced! Averaged gradients for 2 parameters
     ...
  Step 5: accumulated gradients, ready=false
  Step 6: accumulated gradients, ready=false
  Step 7: accumulated gradients, ready=false
  Step 8: accumulated gradients, ready=true
  -> Synced! Averaged gradients for 2 parameters
     ...

=== Done ===
```

## Production usage

In a real multi-GPU setup:

1. Each rank runs in a separate process
2. A coordinator process manages worker registration
3. Use `distributed.NewWorkerNode(cfg)` and call `Start(ctx)` to join the cluster
4. Pass a live `*distributed.NCCLComm` to `fsdp.NewShardedModule`
5. Call `sharded.Forward(ctx, inputs...)` instead of `model.Forward` -- FSDP handles AllGather/ReduceScatter transparently

## Key APIs

| Type | Package | Purpose |
|------|---------|---------|
| `distributed.WorkerNode` | `distributed/` | Manages gRPC server, peer connections, lifecycle |
| `fsdp.ShardedModule[T]` | `distributed/fsdp/` | Shards parameters across ranks, handles AllGather/ReduceScatter |
| `fsdp.GradAccum[T]` | `distributed/fsdp/` | Accumulates gradients across micro-steps |
| `training.Model[T]` | `training/` | Interface: Forward, Backward, Parameters |
