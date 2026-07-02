// Package fsdp implements Fully Sharded Data Parallelism (FSDP) for distributed
// training. It shards model parameters across ranks so each rank holds only
// 1/worldSize of each parameter, reducing per-GPU memory proportionally.
// Before forward, AllGather reconstructs full parameters; after backward,
// ReduceScatter aggregates gradient shards.
package fsdp

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// ShardedModule wraps a model and shards its parameters across worldSize devices.
// Before forward: AllGather reconstructs full parameter tensors.
// After backward: ReduceScatter aggregates gradient shards.
type ShardedModule[T tensor.Numeric] struct {
	module    training.Model[T]
	rank      int
	worldSize int
	comm      *distributed.NCCLComm

	// shards holds the local shard of each parameter (1/worldSize of full data).
	shards map[string][]T
	// originalSizes holds the original full size of each parameter.
	originalSizes map[string]int
}

// NewShardedModule creates a ShardedModule that shards all model parameters
// across worldSize ranks. Each rank retains only its 1/worldSize slice.
func NewShardedModule[T tensor.Numeric](module training.Model[T], rank, worldSize int, comm *distributed.NCCLComm) *ShardedModule[T] {
	s := &ShardedModule[T]{
		module:        module,
		rank:          rank,
		worldSize:     worldSize,
		comm:          comm,
		shards:        make(map[string][]T),
		originalSizes: make(map[string]int),
	}
	s.shardParameters()
	return s
}

// shardParameters splits each parameter's data into worldSize equal shards
// and retains only the shard for this rank.
func (s *ShardedModule[T]) shardParameters() {
	params := s.module.Parameters()
	for _, p := range params {
		data := p.Value.Data()
		fullSize := len(data)
		shardSize := fullSize / s.worldSize

		// Store the local shard.
		start := s.rank * shardSize
		end := start + shardSize
		shard := make([]T, shardSize)
		copy(shard, data[start:end])

		s.shards[p.Name] = shard
		s.originalSizes[p.Name] = fullSize

		// Replace parameter data with just the shard to free memory.
		p.Value.SetData(shard)
		p.Value.SetShape([]int{shardSize})
	}
}

// Forward performs AllGather on all parameters, runs the model forward pass,
// then restores sharded state.
func (s *ShardedModule[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := s.gatherParameters(); err != nil {
		return nil, fmt.Errorf("fsdp: allgather failed: %w", err)
	}

	output, err := s.module.Forward(ctx, inputs...)

	// Restore sharded state after forward to free full-size memory.
	s.restoreShards()

	return output, err
}

// Backward performs AllGather, runs the model backward pass, then
// ReduceScatters gradients so each rank holds its gradient shard.
func (s *ShardedModule[T]) Backward(ctx context.Context, grad *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if err := s.gatherParameters(); err != nil {
		return nil, fmt.Errorf("fsdp: allgather failed: %w", err)
	}

	grads, err := s.module.Backward(ctx, grad, inputs...)
	if err != nil {
		s.restoreShards()
		return nil, err
	}

	if err := s.scatterGradients(); err != nil {
		s.restoreShards()
		return nil, fmt.Errorf("fsdp: reduce-scatter failed: %w", err)
	}

	s.restoreShards()
	return grads, nil
}

// Parameters returns the underlying model's parameters (in sharded state).
func (s *ShardedModule[T]) Parameters() []*graph.Parameter[T] {
	return s.module.Parameters()
}

// ShardMemoryBytes returns the memory used by sharded parameters (per rank).
func (s *ShardedModule[T]) ShardMemoryBytes() int64 {
	var total int64
	for _, shard := range s.shards {
		total += int64(len(shard)) * int64(unsafe.Sizeof(shard[0]))
	}
	return total
}

// ReplicatedMemoryBytes returns the memory that would be used by fully
// replicated (unsharded) parameters.
func (s *ShardedModule[T]) ReplicatedMemoryBytes() int64 {
	var total int64
	for _, shard := range s.shards {
		var zero T
		total += int64(len(shard)) * int64(s.worldSize) * int64(unsafe.Sizeof(zero))
	}
	return total
}

// gatherParameters reconstructs full parameter tensors via AllGather.
// If NCCL is not available, it simulates by tiling the local shard.
func (s *ShardedModule[T]) gatherParameters() error {
	params := s.module.Parameters()
	for _, p := range params {
		shard := s.shards[p.Name]
		fullSize := s.originalSizes[p.Name]
		fullData := make([]T, fullSize)

		if s.comm != nil {
			// Use NCCL AllGather: each rank contributes its shard.
			err := distributed.NCCLAllGather(
				s.comm,
				unsafe.Pointer(&shard[0]),
				unsafe.Pointer(&fullData[0]),
				len(shard),
				0, // default CUDA stream
			)
			if err != nil {
				return err
			}
		} else {
			// In-process simulation: tile the shard for testing without NCCL.
			for i := 0; i < s.worldSize; i++ {
				copy(fullData[i*len(shard):], shard)
			}
		}

		p.Value.SetData(fullData)
		p.Value.SetShape([]int{fullSize})
	}
	return nil
}

// restoreShards replaces full parameter data with local shards.
func (s *ShardedModule[T]) restoreShards() {
	params := s.module.Parameters()
	for _, p := range params {
		shard := s.shards[p.Name]
		p.Value.SetData(shard)
		p.Value.SetShape([]int{len(shard)})
	}
}

// scatterGradients performs ReduceScatter on gradients so each rank holds
// its reduced gradient shard.
func (s *ShardedModule[T]) scatterGradients() error {
	params := s.module.Parameters()
	for _, p := range params {
		if p.Gradient == nil {
			continue
		}
		gradData := p.Gradient.Data()
		shardSize := len(gradData) / s.worldSize
		recvBuf := make([]T, shardSize)

		if s.comm != nil {
			err := distributed.NCCLReduceScatter(
				s.comm,
				unsafe.Pointer(&gradData[0]),
				unsafe.Pointer(&recvBuf[0]),
				shardSize,
				0,
			)
			if err != nil {
				return err
			}
		} else {
			// Simulation: take the shard for this rank (no actual reduction).
			start := s.rank * shardSize
			copy(recvBuf, gradData[start:start+shardSize])
		}

		p.Gradient.SetData(recvBuf)
		p.Gradient.SetShape([]int{shardSize})
	}
	return nil
}
