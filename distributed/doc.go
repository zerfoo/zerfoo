// Package distributed provides multi-node distributed training for the Zerfoo
// ML framework. It implements gradient synchronization, barrier coordination,
// and tensor broadcasting across a cluster of worker nodes. (Stability: beta)
//
// # Architecture
//
// The package is built around the [InternalStrategy] interface, which defines
// the collective operations required for distributed training: all-reduce,
// barrier, and broadcast. Two concrete implementations are provided:
//
//   - [GrpcStrategy] uses gRPC for CPU-based gradient exchange over the
//     network. Workers register with a coordinator, discover peers, and
//     perform a star-topology all-reduce through bidirectional streaming RPCs.
//
//   - [NcclStrategy] (build tag: cuda) uses NVIDIA NCCL for GPU-native
//     collective operations. Gradient tensors remain on-device throughout
//     the all-reduce with no CPU round-trip, delivering significantly higher
//     throughput on multi-GPU nodes.
//
// [AllReduceStrategy] composes two InternalStrategy instances into a
// hierarchical scheme: a local strategy handles intra-node communication
// (typically NCCL) while a cross-node strategy handles inter-node
// communication (typically gRPC). Node leaders participate in both layers.
//
// # Coordinator and Worker Lifecycle
//
// A coordinator process (defined in the distributed/pb protobuf service)
// manages worker registration, heartbeats, and peer discovery. Each worker
// follows this lifecycle:
//
//  1. Create a [WorkerNode] or [GrpcStrategy] with the desired configuration.
//  2. Call Init (or Start for WorkerNode), which registers with the
//     coordinator, starts a local gRPC server, and connects to all peers.
//  3. Use AllReduceGradients, Barrier, and BroadcastTensor during training.
//  4. Call Shutdown (or Close) for orderly teardown: unregister from the
//     coordinator, close peer connections, and stop the local gRPC server.
//
// [WorkerNode] wraps [GrpcStrategy] with mutex-guarded lifecycle management,
// health check integration, and compatibility with shutdown.Coordinator.
//
// # gRPC Protocol
//
// The protobuf service (distributed/pb) defines three RPCs on the worker
// service:
//
//   - AllReduce: bidirectional streaming. Each non-root worker sends its
//     gradient tensors to the root (rank 0), which collects all submissions,
//     computes the element-wise average, and streams the result back.
//
//   - Barrier: unary RPC. Each worker calls Barrier on the root, which
//     blocks all callers until every rank has arrived, then releases them.
//
//   - Broadcast: unary RPC. The root sets a tensor via SetBroadcastTensor,
//     and non-root workers retrieve it by calling Broadcast on the root.
//
// A separate coordinator service handles RegisterWorker, UnregisterWorker,
// and Heartbeat RPCs for cluster membership.
//
// # NCCL Gradient Exchange
//
// When built with the cuda build tag, [NcclStrategy] provides GPU-native
// collectives via NCCL. It groups multiple tensor reductions into a single
// NCCL launch (ncclGroupStart/ncclGroupEnd) for efficiency, synchronizes
// on a dedicated CUDA stream, and implements barriers as zero-byte
// all-reduce operations. Use [NcclStrategy.InitWithUID] for single-process
// multi-GPU setups where the coordinator can distribute the NCCL UniqueID
// directly.
//
// # TLS
//
// [TLSConfig] provides optional TLS and mutual TLS (mTLS) for all gRPC
// connections, including coordinator registration and peer-to-peer traffic.
// When TLSConfig is nil, plaintext connections are used.
//
// # Metrics
//
// All strategies and the worker service emit Prometheus-compatible metrics
// (counters and histograms) through the ztensor metrics.Collector interface.
// Use SetCollector to wire in a concrete collector.
// Stability: beta
package distributed
