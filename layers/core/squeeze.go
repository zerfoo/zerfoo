package core

import (
	"context"
	"fmt"
	"sort"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Squeeze removes dimensions of size 1 at the specified axes.
// If no axes are specified, all size-1 dimensions are removed.
type Squeeze[T tensor.Numeric] struct {
	engine compute.Engine[T]
	axes   []int // empty means squeeze all size-1 dims
}

func (s *Squeeze[T]) OpType() string                  { return "Squeeze" }
func (s *Squeeze[T]) Attributes() map[string]any       { return map[string]any{"axes": s.axes} }
func (s *Squeeze[T]) OutputShape() []int               { return nil }
func (s *Squeeze[T]) Parameters() []*graph.Parameter[T] { return nil }

func (s *Squeeze[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 || len(inputs) > 2 {
		return nil, fmt.Errorf("Squeeze requires 1 or 2 inputs, got %d", len(inputs))
	}
	inShape := inputs[0].Shape()

	axes := s.axes
	// ONNX opset 13+: axes come as second input tensor.
	if len(inputs) == 2 {
		axData := inputs[1].Data()
		axes = make([]int, len(axData))
		for i, v := range axData {
			axes[i] = int(v)
		}
	}

	// Normalize negative axes.
	rank := len(inShape)
	squeezeSet := make(map[int]bool, len(axes))
	for _, a := range axes {
		if a < 0 {
			a += rank
		}
		if a < 0 || a >= rank {
			return nil, fmt.Errorf("Squeeze: axis out of range for rank %d", rank)
		}
		if inShape[a] != 1 {
			return nil, fmt.Errorf("Squeeze: dim %d has size %d, not 1", a, inShape[a])
		}
		squeezeSet[a] = true
	}

	var newShape []int
	for i, d := range inShape {
		if len(axes) == 0 {
			if d != 1 {
				newShape = append(newShape, d)
			}
		} else {
			if !squeezeSet[i] {
				newShape = append(newShape, d)
			}
		}
	}
	if len(newShape) == 0 {
		// Squeezing all dimensions produces a scalar (0D tensor).
		// Use tensor.New directly since Reshape may reject empty shapes.
		return tensor.New[T](nil, inputs[0].Data()[:1])
	}

	return s.engine.Reshape(ctx, inputs[0], newShape)
}

func (s *Squeeze[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Squeeze backward not implemented")
}

// BuildSqueeze constructs a Squeeze node from attributes.
func BuildSqueeze[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], attrs map[string]any,
) (graph.Node[T], error) {
	var axes []int
	if v, ok := attrs["axes"]; ok {
		switch a := v.(type) {
		case []int64:
			axes = make([]int, len(a))
			for i, x := range a {
				axes[i] = int(x)
			}
		case []any:
			axes = make([]int, len(a))
			for i, x := range a {
				axes[i] = int(x.(int64))
			}
		default:
			return nil, fmt.Errorf("Squeeze: unsupported axes type %T", v)
		}
		sort.Ints(axes)
	}
	return &Squeeze[T]{engine: engine, axes: axes}, nil
}

var _ graph.Node[float32] = (*Squeeze[float32])(nil)
