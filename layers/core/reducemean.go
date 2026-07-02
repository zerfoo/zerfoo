package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ReduceMean reduces a tensor by computing the mean along specified axes.
type ReduceMean[T tensor.Numeric] struct {
	engine   compute.Engine[T]
	axes     []int
	keepDims bool
}

func (r *ReduceMean[T]) OpType() string                  { return "ReduceMean" }
func (r *ReduceMean[T]) OutputShape() []int               { return nil }
func (r *ReduceMean[T]) Parameters() []*graph.Parameter[T] { return nil }

func (r *ReduceMean[T]) Attributes() map[string]any {
	return map[string]any{"axes": r.axes, "keepdims": r.keepDims}
}

func (r *ReduceMean[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("ReduceMean requires at least 1 input, got %d", len(inputs))
	}

	result := inputs[0]

	// ONNX ReduceMean opset 18+: axes come from second input tensor.
	axes := r.axes
	if len(axes) == 0 && len(inputs) >= 2 {
		axesData := inputs[1].Data()
		axes = make([]int, len(axesData))
		for i, v := range axesData {
			axes[i] = int(v)
		}
	}

	// Reduce along each axis (from highest to lowest to preserve indices).
	sorted := make([]int, len(axes))
	copy(sorted, axes)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] > sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	for _, axis := range sorted {
		// Normalize negative axes.
		ndim := len(result.Shape())
		if axis < 0 {
			axis += ndim
		}
		var err error
		result, err = r.engine.ReduceMean(ctx, result, axis, r.keepDims)
		if err != nil {
			return nil, fmt.Errorf("ReduceMean axis %d: %w", axis, err)
		}
	}

	return result, nil
}

func (r *ReduceMean[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("ReduceMean backward not implemented")
}

// BuildReduceMean constructs a ReduceMean node from attributes.
func BuildReduceMean[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], attrs map[string]any,
) (graph.Node[T], error) {
	var axes []int
	if a, ok := attrs["axes"]; ok {
		switch v := a.(type) {
		case []any:
			axes = make([]int, len(v))
			for i, val := range v {
				axes[i] = int(val.(int64))
			}
		case []int64:
			axes = make([]int, len(v))
			for i, val := range v {
				axes[i] = int(val)
			}
		}
	}

	keepDims := true
	if kd, ok := attrs["keepdims"]; ok {
		switch v := kd.(type) {
		case int64:
			keepDims = v != 0
		case bool:
			keepDims = v
		}
	}

	return &ReduceMean[T]{engine: engine, axes: axes, keepDims: keepDims}, nil
}

var _ graph.Node[float32] = (*ReduceMean[float32])(nil)
