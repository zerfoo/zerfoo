# ADR 050: Distributed Training FSDP-Equivalent

## Status
Accepted

## Date
2026-03-17

## Context
Wolf will eventually require fine-tuning of 70B+ models on the DGX cluster.
Single-GPU training cannot fit these models even with QLoRA at extreme scale.
PyTorch FSDP (Fully Sharded Data Parallel) shards model parameters, optimizer
states, and gradients across devices, loading/releasing shards during forward
and backward passes. ZeRO-3 (DeepSpeed) follows the same pattern.

Zerfoo's distributed package already has gRPC infrastructure and NCCL bindings
for gradient exchange. FSDP-equivalent sharding is the natural extension.

## Decision
Implement FSDP-equivalent in distributed/fsdp/ using NCCL as the collective
communication backend:

ShardedModule (distributed/fsdp/sharded_module.go):
- Wraps a model's parameter tensors; each device holds 1/N of each parameter
- Before forward: AllGather to reconstruct full parameter on each device
- After forward: free gathered parameter, retain only local shard
- Before backward: AllGather again for gradient computation
- After backward: ReduceScatter gradient contributions; free gathered

GradientAccumulation (distributed/fsdp/grad_accum.go):
- Accumulates gradients locally for M steps before synchronization
- Reduces communication frequency by M; critical for slow networks

OptimizerShard (distributed/fsdp/optimizer_shard.go):
- Each device runs optimizer step on its local parameter shard only
- AdamW moment tensors stored per-shard; 1/N memory for optimizer states

CheckpointRestart (distributed/fsdp/checkpoint.go):
- Saves full model by AllGather + rank-0 write
- Restart re-shards from full checkpoint
- Flash Checkpoint: in-memory sharded checkpoint for fast recovery within cluster

Fault Tolerance:
- Worker health monitored via gRPC health check protocol
- On worker failure: surviving workers reload last checkpoint; failed worker restarts
  and rejoins via dynamic membership in distributed/coordinator.go
- Recovery time target: under 60 seconds for 8-GPU cluster

NCCL is loaded via purego (no CGo) using existing distributed/nccl.go bindings.

## Consequences
Positive:
- Enables training of models larger than single-GPU VRAM (70B on 8x H100)
- Optimizer state sharding reduces peak memory by 8x on 8-GPU cluster
- Go-native implementation; no Python dependency for distributed training

Negative:
- AllGather/ReduceScatter communication overhead is significant on PCIe (vs NVLink)
  -- DGX Spark has NVLink, mitigating this
- Implementation complexity is high; correctness bugs in gradient accumulation are
  subtle and may not manifest until large-scale runs
- Dynamic membership (fault tolerance) adds significant complexity; deferred to
  Year 2 Q3 after core FSDP is validated
