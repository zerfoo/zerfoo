package core

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// tensorDType mirrors zmf.Tensor_DataType enum values.
type tensorDType int32

const (
	tensorDTypeFloat32 tensorDType = 3
	tensorDTypeFloat64 tensorDType = 4
	tensorDTypeInt64   tensorDType = 8
)

// tensorValue holds a serialized tensor value used in ConstantOfShape attributes.
type tensorValue struct {
	Dtype tensorDType
	Shape []int64
	Data  []byte
}

// ConstantOfShape creates a tensor of a given shape filled with a constant value.
type ConstantOfShape[T tensor.Numeric] struct {
	engine compute.Engine[T]
	value  T
}

func (c *ConstantOfShape[T]) OpType() string { return "ConstantOfShape" }
func (c *ConstantOfShape[T]) Attributes() map[string]any {
	return map[string]any{"value": c.value}
}
func (c *ConstantOfShape[T]) OutputShape() []int               { return nil }
func (c *ConstantOfShape[T]) Parameters() []*graph.Parameter[T] { return nil }

func (c *ConstantOfShape[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("ConstantOfShape requires 1 input (shape), got %d", len(inputs))
	}

	shapeData := inputs[0].Data()
	shape := make([]int, len(shapeData))
	size := 1
	for i, v := range shapeData {
		shape[i] = int(v)
		size *= shape[i]
	}

	out := make([]T, size)
	for i := range out {
		out[i] = c.value
	}

	return tensor.New(shape, out)
}

func (c *ConstantOfShape[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("ConstantOfShape backward not implemented")
}

// BuildConstantOfShape constructs a ConstantOfShape node from attributes.
func BuildConstantOfShape[T tensor.Numeric](
	engine compute.Engine[T], ops numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], attrs map[string]any,
) (graph.Node[T], error) {
	var value T
	if v, ok := attrs["value"]; ok {
		switch val := v.(type) {
		case float64:
			value = ops.FromFloat64(val)
		case float32:
			value = ops.FromFloat32(val)
		case int64:
			value = ops.FromFloat64(float64(val))
		case *tensorValue:
			if val != nil && len(val.Data) > 0 {
				switch val.Dtype {
				case tensorDTypeFloat32:
					bits := binary.LittleEndian.Uint32(val.Data[:4])
					value = ops.FromFloat32(math.Float32frombits(bits))
				case tensorDTypeFloat64:
					bits := binary.LittleEndian.Uint64(val.Data[:8])
					value = ops.FromFloat64(math.Float64frombits(bits))
				case tensorDTypeInt64:
					n := int64(binary.LittleEndian.Uint64(val.Data[:8]))
					value = ops.FromFloat64(float64(n))
				}
			}
		}
	}
	return &ConstantOfShape[T]{engine: engine, value: value}, nil
}

var _ graph.Node[float32] = (*ConstantOfShape[float32])(nil)
