package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Pad applies constant-value padding to a tensor.
// pads has shape [2*ndim]: [begin_0, begin_1, ..., end_0, end_1, ...].
type Pad[T tensor.Numeric] struct {
	engine        compute.Engine[T]
	pads          []int64
	constantValue T
	outputShape   []int
}

// NewPad creates a new Pad layer.
func NewPad[T tensor.Numeric](engine compute.Engine[T], pads []int64, constantValue T) *Pad[T] {
	return &Pad[T]{engine: engine, pads: pads, constantValue: constantValue}
}

// Forward pads the input tensor with the configured constant value.
func (p *Pad[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Pad expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	ndim := len(shape)

	if len(p.pads) != 2*ndim {
		return nil, fmt.Errorf("Pad: pads length %d does not match 2*ndim=%d", len(p.pads), 2*ndim)
	}

	// Compute output shape.
	outShape := make([]int, ndim)
	for d := range ndim {
		outShape[d] = shape[d] + int(p.pads[d]) + int(p.pads[ndim+d])
	}

	outSize := 1
	for _, d := range outShape {
		outSize *= d
	}

	// Allocate output filled with the constant value.
	outData := make([]T, outSize)
	for i := range outData {
		outData[i] = p.constantValue
	}

	// Copy input data into the padded positions using flat indexing.
	inData := input.Data()
	inSize := len(inData)
	inStrides := make([]int, ndim)
	outStrides := make([]int, ndim)
	stride := 1
	for d := ndim - 1; d >= 0; d-- {
		inStrides[d] = stride
		stride *= shape[d]
	}
	stride = 1
	for d := ndim - 1; d >= 0; d-- {
		outStrides[d] = stride
		stride *= outShape[d]
	}

	for flatIn := range inSize {
		// Convert flat input index to multi-dimensional indices.
		// idx[d] = (flatIn / inStrides[d]) % shape[d]
		idx := make([]int, ndim)
		for d := range ndim {
			idx[d] = (flatIn / inStrides[d]) % shape[d]
		}

		// Compute output flat index with padding offset.
		flatOut := 0
		for d := range ndim {
			flatOut += (idx[d] + int(p.pads[d])) * outStrides[d]
		}
		outData[flatOut] = inData[flatIn]
	}

	out, err := tensor.New[T](outShape, outData)
	if err != nil {
		return nil, fmt.Errorf("Pad.Forward: %w", err)
	}
	p.outputShape = outShape
	return out, nil
}

// Backward returns nil (not required for inference).
func (p *Pad[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "Pad".
func (p *Pad[T]) OpType() string { return "Pad" }

// Attributes returns the pad configuration.
func (p *Pad[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"pads": p.pads}
}

// OutputShape returns the output shape from the last forward call.
func (p *Pad[T]) OutputShape() []int { return p.outputShape }

// Parameters returns nil (no trainable parameters).
func (p *Pad[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildPad constructs a Pad layer from ZMF attributes.
// Supported attribute keys: "pads" ([]int64), "constant_value" (float32/float64).
func BuildPad[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	pads := extractInt64Slice(attributes, "pads")

	var constantValue T
	if v, ok := attributes["constant_value"]; ok {
		switch cv := v.(type) {
		case float64:
			constantValue = ops.FromFloat64(cv)
		case float32:
			constantValue = ops.FromFloat64(float64(cv))
		}
	}

	return NewPad(engine, pads, constantValue), nil
}

// Statically assert that Pad implements graph.Node.
var _ graph.Node[float32] = (*Pad[float32])(nil)
