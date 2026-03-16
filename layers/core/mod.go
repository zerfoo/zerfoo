package core

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Mod computes element-wise remainder (fmod). Supports scalar broadcasting.
// ONNX Mod op with fmod=1 (default for floats).
type Mod[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (m *Mod[T]) OpType() string                  { return "Mod" }
func (m *Mod[T]) Attributes() map[string]any       { return nil }
func (m *Mod[T]) OutputShape() []int               { return nil }
func (m *Mod[T]) Parameters() []*graph.Parameter[T] { return nil }

func (m *Mod[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Mod requires 2 inputs, got %d", len(inputs))
	}
	a, b := inputs[0].Data(), inputs[1].Data()

	if len(b) == 1 {
		out := make([]T, len(a))
		bv := float64(b[0])
		for i := range a {
			out[i] = T(math.Mod(float64(a[i]), bv))
		}
		return tensor.New(inputs[0].Shape(), out)
	}
	if len(a) == 1 {
		out := make([]T, len(b))
		av := float64(a[0])
		for i := range b {
			out[i] = T(math.Mod(av, float64(b[i])))
		}
		return tensor.New(inputs[1].Shape(), out)
	}

	if len(a) != len(b) {
		return nil, fmt.Errorf("Mod: input sizes differ (%d vs %d)", len(a), len(b))
	}
	out := make([]T, len(a))
	for i := range a {
		out[i] = T(math.Mod(float64(a[i]), float64(b[i])))
	}
	return tensor.New(inputs[0].Shape(), out)
}

func (m *Mod[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Mod backward not implemented")
}

// BuildMod constructs a Mod node from attributes.
func BuildMod[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Mod[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Mod[float32])(nil)
