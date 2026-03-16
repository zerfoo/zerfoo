package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// fusedNormAddNode fuses RMSNorm + Add into a single GPU kernel launch.
// It takes two inputs (data to normalize, residual to add), computes:
//   output = rmsnorm(input, weight, eps) + residual
//
// This replaces the postFfnNorm (RMSNorm) + residualAdd (Add) pair,
// saving one kernel launch per transformer layer.
type fusedNormAddNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	weight *tensor.TensorNumeric[T]
	eps    float32
}

func (n *fusedNormAddNode[T]) OpType() string { return "FusedNormAdd" }

func (n *fusedNormAddNode[T]) Attributes() map[string]any {
	return map[string]any{"eps": n.eps}
}

func (n *fusedNormAddNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("FusedNormAdd: expected 2 inputs (data, residual), got %d", len(inputs))
	}
	data := inputs[0]     // e.g. ffnOut
	residual := inputs[1] // e.g. stored residual from fusedAddRMSNormNode

	// Try fused GPU path.
	realEngine := compute.Engine[T](n.engine)
	if proxy, ok := n.engine.(*compute.EngineProxy[T]); ok {
		realEngine = proxy.Real()
	}

	if provider, ok := realEngine.(compute.FusedNormAddProvider[T]); ok {
		out, err := provider.GPUFusedNormAdd(data, n.weight, residual, n.eps)
		if err == nil {
			return out, nil
		}
		// Fall through to unfused path on error.
	}

	// Fallback: separate RMSNorm + Add.
	if f32Data, ok := any(data).(*tensor.TensorNumeric[float32]); ok {
		if f32Weight, ok2 := any(n.weight).(*tensor.TensorNumeric[float32]); ok2 {
			normed, _, err := compute.FusedRMSNorm(f32Data, f32Weight, n.eps)
			if err != nil {
				return nil, err
			}
			f32Residual := any(residual).(*tensor.TensorNumeric[float32])
			result, err := n.engine.Add(ctx, any(normed).(*tensor.TensorNumeric[T]), any(f32Residual).(*tensor.TensorNumeric[T]))
			if err != nil {
				return nil, err
			}
			return result, nil
		}
	}
	return nil, fmt.Errorf("FusedNormAdd: no GPU provider and CPU fallback failed")
}

func (n *fusedNormAddNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("FusedNormAdd: backward not implemented")
}

func (n *fusedNormAddNode[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{{Name: "weight", Value: n.weight}}
}

func (n *fusedNormAddNode[T]) OutputShape() []int { return nil }
