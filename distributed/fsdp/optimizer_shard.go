// Package fsdp — see sharded_module.go for package doc.

package fsdp

import (
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

// ShardedAdamW implements AdamW where each rank only maintains optimizer state
// for its own parameter shard (ZeRO Stage 2). Moment tensors are sized 1/worldSize
// of the full parameter, so total optimizer memory scales as O(params/N).
type ShardedAdamW[T tensor.Numeric] struct {
	rank      int
	worldSize int
	lr        float32
	beta1     float32
	beta2     float32
	eps       float32
	wd        float32
	step      int

	// moment buffers sized to local shard only
	m1 map[string][]float32
	m2 map[string][]float32
}

// NewShardedAdamW creates a sharded AdamW optimizer for the given rank.
// Each rank maintains moment buffers only for its local parameter shard.
func NewShardedAdamW[T tensor.Numeric](rank, worldSize int, lr, beta1, beta2, eps, wd float32) *ShardedAdamW[T] {
	return &ShardedAdamW[T]{
		rank:      rank,
		worldSize: worldSize,
		lr:        lr,
		beta1:     beta1,
		beta2:     beta2,
		eps:       eps,
		wd:        wd,
		m1:        make(map[string][]float32),
		m2:        make(map[string][]float32),
	}
}

// Step performs one AdamW update on the local parameter shards.
// shardGrads maps parameter names to gradient slices (already reduced/scattered
// to this rank's shard). Returns the updated parameter shard delta values.
func (o *ShardedAdamW[T]) Step(shardGrads map[string][]T) map[string][]T {
	o.step++

	bc1 := 1.0 - math.Pow(float64(o.beta1), float64(o.step))
	bc2 := 1.0 - math.Pow(float64(o.beta2), float64(o.step))
	alpha := float64(o.lr) * math.Sqrt(bc2) / bc1

	b1 := float64(o.beta1)
	b2 := float64(o.beta2)

	updated := make(map[string][]T, len(shardGrads))

	for name, grad := range shardGrads {
		n := len(grad)

		m1, ok := o.m1[name]
		if !ok {
			m1 = make([]float32, n)
			o.m1[name] = m1
			o.m2[name] = make([]float32, n)
		}
		m2 := o.m2[name]

		params := make([]T, n)

		for i := range n {
			g := float64(grad[i])

			m1[i] = float32(b1*float64(m1[i]) + (1-b1)*g)
			m2[i] = float32(b2*float64(m2[i]) + (1-b2)*g*g)

			update := alpha * float64(m1[i]) / (math.Sqrt(float64(m2[i])) + float64(o.eps))
			params[i] = T(-update)
		}

		updated[name] = params
	}

	return updated
}

// StepOnParams performs one AdamW update directly on parameter shard slices.
// This is the primary entry point for FSDP training: it updates shardParams
// in-place using the corresponding shardGrads.
func (o *ShardedAdamW[T]) StepOnParams(shardParams, shardGrads map[string][]T) {
	o.step++

	bc1 := 1.0 - math.Pow(float64(o.beta1), float64(o.step))
	bc2 := 1.0 - math.Pow(float64(o.beta2), float64(o.step))
	alpha := float64(o.lr) * math.Sqrt(bc2) / bc1

	b1 := float64(o.beta1)
	b2 := float64(o.beta2)

	for name, grad := range shardGrads {
		params := shardParams[name]
		n := len(grad)

		m1, ok := o.m1[name]
		if !ok {
			m1 = make([]float32, n)
			o.m1[name] = m1
			o.m2[name] = make([]float32, n)
		}
		m2 := o.m2[name]

		for i := range n {
			g := float64(grad[i])

			m1[i] = float32(b1*float64(m1[i]) + (1-b1)*g)
			m2[i] = float32(b2*float64(m2[i]) + (1-b2)*g*g)

			update := alpha * float64(m1[i]) / (math.Sqrt(float64(m2[i])) + float64(o.eps))
			decay := float64(o.lr) * float64(o.wd) * float64(params[i])
			params[i] = T(float64(params[i]) - update - decay)
		}
	}
}

// MemoryBytes returns the total memory used by moment buffers across all
// tracked parameter shards.
func (o *ShardedAdamW[T]) MemoryBytes() int64 {
	var total int64
	var zero float32
	elemSize := int64(unsafe.Sizeof(zero))
	for name, m1 := range o.m1 {
		total += int64(len(m1)) * elemSize
		total += int64(len(o.m2[name])) * elemSize
	}
	return total
}
