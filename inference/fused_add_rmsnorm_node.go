package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// fusedAddRMSNormNode fuses Add + RMSNorm into a single GPU kernel launch.
// It takes two inputs (addend and residual), computes residual = addend + residual
// in-place, then applies RMSNorm. The updated residual is stored internally
// so that the subsequent residualAddNode can retrieve it without recomputing.
//
// This saves one kernel launch per fusion point (2 per transformer layer).
type fusedAddRMSNormNode[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	weight  *tensor.TensorNumeric[T]
	eps     float32
	residual *tensor.TensorNumeric[T] // stored after Forward
}

func (n *fusedAddRMSNormNode[T]) OpType() string { return "FusedAddRMSNorm" }

func (n *fusedAddRMSNormNode[T]) Attributes() map[string]any {
	return map[string]any{"eps": n.eps}
}

func (n *fusedAddRMSNormNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("FusedAddRMSNorm: expected 2 inputs, got %d", len(inputs))
	}
	addend := inputs[0]  // e.g. attnOut or ffnOut
	residual := inputs[1] // e.g. hidden or residual1

	if provider, ok := n.engine.(compute.FusedAddRMSNormProvider[T]); ok {
		normed, residualOut, _, err := provider.GPUFusedAddRMSNorm(addend, residual, n.weight, n.eps)
		if err == nil {
			n.residual = residualOut
			return normed, nil
		}
		// Fall through to unfused path on error.
	}

	// Unfused fallback: separate Add + GPU RMSNorm.
	sum, err := n.engine.Add(ctx, addend, residual)
	if err != nil {
		return nil, err
	}
	n.residual = sum

	// Use GPU-resident FusedRMSNormGPU to avoid D2H copies that break CUDA
	// graph capture. The old path used CPU compute.FusedRMSNorm which called
	// .Data() on GPU tensors, triggering cudaMemcpy D2H.
	type gpuRMSNormer interface {
		FusedRMSNormGPU(input, weight *tensor.TensorNumeric[float32], epsilon float32) (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], error)
	}
	if gpuNorm, ok := any(n.engine).(gpuRMSNormer); ok {
		if f32Sum, ok2 := any(sum).(*tensor.TensorNumeric[float32]); ok2 {
			if f32Weight, ok3 := any(n.weight).(*tensor.TensorNumeric[float32]); ok3 {
				normed, _, err := gpuNorm.FusedRMSNormGPU(f32Sum, f32Weight, n.eps)
				if err == nil {
					if result, ok4 := any(normed).(*tensor.TensorNumeric[T]); ok4 {
						return result, nil
					}
				}
				// Fall through to CPU path on GPU RMSNorm error.
			}
		}
	}

	// Last resort: CPU FusedRMSNorm. WARNING: this calls .Data() on GPU
	// tensors, which triggers D2H cudaMemcpy and breaks CUDA graph capture.
	// This path should only be reached on CPU-only engines.
	if f32Sum, ok := any(sum).(*tensor.TensorNumeric[float32]); ok {
		if f32Weight, ok2 := any(n.weight).(*tensor.TensorNumeric[float32]); ok2 {
			normed, _, err := compute.FusedRMSNorm(f32Sum, f32Weight, n.eps)
			if err != nil {
				return nil, err
			}
			if result, ok3 := any(normed).(*tensor.TensorNumeric[T]); ok3 {
				return result, nil
			}
		}
	}
	return nil, fmt.Errorf("FusedAddRMSNorm: all paths failed")
}

func (n *fusedAddRMSNormNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("FusedAddRMSNorm: backward not implemented")
}

func (n *fusedAddRMSNormNode[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{{Name: "weight", Value: n.weight}}
}

func (n *fusedAddRMSNormNode[T]) OutputShape() []int { return nil }

// Residual returns the stored residual (addend + original residual) from the
// most recent Forward call. Used by residualAddNode.
func (n *fusedAddRMSNormNode[T]) Residual() *tensor.TensorNumeric[T] {
	return n.residual
}

// residualAddNode computes Add(input, stored_residual) where the residual
// comes from a preceding fusedAddRMSNormNode. This avoids recomputing the
// residual sum.
type residualAddNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	source *fusedAddRMSNormNode[T]
}

func (n *residualAddNode[T]) OpType() string                 { return "ResidualAdd" }
func (n *residualAddNode[T]) Attributes() map[string]any     { return nil }
func (n *residualAddNode[T]) Parameters() []*graph.Parameter[T] { return nil }
func (n *residualAddNode[T]) OutputShape() []int             { return nil }

func (n *residualAddNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("ResidualAdd: expected at least 1 input, got %d", len(inputs))
	}
	// inputs[0] = ffnOut (or attnOut in the cross-layer case)
	// The residual comes from the fused node, not from graph inputs.
	res := n.source.Residual()
	if res == nil {
		return nil, fmt.Errorf("ResidualAdd: fused node has no stored residual")
	}
	return n.engine.Add(ctx, inputs[0], res)
}

func (n *residualAddNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("ResidualAdd: backward not implemented")
}

// residualRefNode retrieves the stored residual from a fusedAddRMSNormNode
// without adding anything. Used by fusedNormAddNode to access the residual
// as a graph input.
type residualRefNode[T tensor.Numeric] struct {
	source *fusedAddRMSNormNode[T]
}

func (n *residualRefNode[T]) OpType() string                 { return "ResidualRef" }
func (n *residualRefNode[T]) Attributes() map[string]any     { return nil }
func (n *residualRefNode[T]) Parameters() []*graph.Parameter[T] { return nil }
func (n *residualRefNode[T]) OutputShape() []int             { return nil }

func (n *residualRefNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	res := n.source.Residual()
	if res == nil {
		return nil, fmt.Errorf("ResidualRef: fused node has no stored residual")
	}
	return res, nil
}

func (n *residualRefNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("ResidualRef: backward not implemented")
}
