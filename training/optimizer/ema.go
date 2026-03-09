// Package optimizer provides various optimization algorithms for neural networks.
package optimizer

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// EMA wraps an Optimizer with Exponential Moving Average weight averaging.
// After each inner optimizer step, it updates shadow weights:
//
//	shadow = decay * shadow + (1-decay) * param.Value
//
// Call SwapShadow before validation to use averaged weights,
// then SwapBack to restore training weights.
type EMA[T tensor.Numeric] struct {
	inner  Optimizer[T]
	engine compute.Engine[T]
	decay  T
	shadow map[*graph.Parameter[T]]*tensor.TensorNumeric[T]
}

// NewEMA creates a new EMA wrapper around the given optimizer.
func NewEMA[T tensor.Numeric](inner Optimizer[T], engine compute.Engine[T], decay T) *EMA[T] {
	return &EMA[T]{
		inner:  inner,
		engine: engine,
		decay:  decay,
		shadow: make(map[*graph.Parameter[T]]*tensor.TensorNumeric[T]),
	}
}

// Step runs the inner optimizer step and then updates shadow weights.
func (e *EMA[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	// Run inner optimizer step first
	if err := e.inner.Step(ctx, params); err != nil {
		return err
	}

	ops := e.engine.Ops()
	one := ops.FromFloat64(1.0)
	oneMinusDecay := ops.Sub(one, e.decay)

	for _, param := range params {
		if param.Value == nil {
			continue
		}

		shadow, ok := e.shadow[param]
		if !ok {
			// Initialize shadow as copy of current param value
			shadowTensor, err := tensor.New[T](param.Value.Shape(), nil)
			if err != nil {
				return err
			}
			if err := e.engine.Copy(ctx, shadowTensor, param.Value); err != nil {
				return err
			}
			e.shadow[param] = shadowTensor
			continue
		}

		// shadow = decay * shadow + (1-decay) * param.Value
		decayed, err := e.engine.MulScalar(ctx, shadow, e.decay, nil)
		if err != nil {
			return err
		}
		current, err := e.engine.MulScalar(ctx, param.Value, oneMinusDecay, nil)
		if err != nil {
			return err
		}
		_, err = e.engine.Add(ctx, decayed, current, shadow)
		if err != nil {
			return err
		}
	}
	return nil
}

// SwapShadow swaps param.Value with shadow weights for validation/checkpointing.
func (e *EMA[T]) SwapShadow(ctx context.Context, params []*graph.Parameter[T]) error {
	for _, param := range params {
		shadow, ok := e.shadow[param]
		if !ok || param.Value == nil {
			continue
		}
		// temp = copy of param.Value
		tmp, err := tensor.New[T](param.Value.Shape(), nil)
		if err != nil {
			return err
		}
		if err := e.engine.Copy(ctx, tmp, param.Value); err != nil {
			return err
		}
		// param.Value = shadow
		if err := e.engine.Copy(ctx, param.Value, shadow); err != nil {
			return err
		}
		// shadow = temp (old param.Value)
		if err := e.engine.Copy(ctx, shadow, tmp); err != nil {
			return err
		}
	}
	return nil
}

// SwapBack is a semantic alias for SwapShadow — the swap operation is symmetric.
func (e *EMA[T]) SwapBack(ctx context.Context, params []*graph.Parameter[T]) error {
	return e.SwapShadow(ctx, params)
}

// Statically assert that the type implements the Optimizer interface.
var _ Optimizer[float32] = (*EMA[float32])(nil)
