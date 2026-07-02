package fsdp

import "github.com/zerfoo/ztensor/tensor"

// GradAccum accumulates gradients across M micro-steps before triggering
// a synchronization (e.g., AllReduce or ReduceScatter for FSDP).
// This allows effective batch sizes larger than what fits in GPU memory.
type GradAccum[T tensor.Numeric] struct {
	module  *ShardedModule[T]
	steps   int            // micro-steps per sync
	current int            // current micro-step count
	accum   map[string][]T // accumulated gradients per param
}

// NewGradAccum creates a GradAccum that accumulates gradients for stepsPerSync
// micro-steps before triggering synchronization through the ShardedModule.
func NewGradAccum[T tensor.Numeric](module *ShardedModule[T], stepsPerSync int) *GradAccum[T] {
	if stepsPerSync < 1 {
		stepsPerSync = 1
	}
	return &GradAccum[T]{
		module:  module,
		steps:   stepsPerSync,
		current: 0,
		accum:   make(map[string][]T),
	}
}

// Accumulate adds a set of gradients (keyed by parameter name) to the
// accumulator. Returns true when the accumulation window is full and
// Sync should be called.
func (g *GradAccum[T]) Accumulate(grads map[string][]T) bool {
	for name, grad := range grads {
		existing, ok := g.accum[name]
		if !ok {
			buf := make([]T, len(grad))
			copy(buf, grad)
			g.accum[name] = buf
			continue
		}
		for i, v := range grad {
			existing[i] += v
		}
	}
	g.current++
	return g.current >= g.steps
}

// Sync returns the averaged accumulated gradients (sum / steps) and resets
// the accumulator. This should be called after Accumulate returns true.
func (g *GradAccum[T]) Sync() map[string][]T {
	result := make(map[string][]T, len(g.accum))
	divisor := T(g.steps)
	for name, grad := range g.accum {
		averaged := make([]T, len(grad))
		for i, v := range grad {
			averaged[i] = v / divisor
		}
		result[name] = averaged
	}
	g.Reset()
	return result
}

// Reset clears the accumulator and resets the step counter.
func (g *GradAccum[T]) Reset() {
	g.current = 0
	g.accum = make(map[string][]T)
}
