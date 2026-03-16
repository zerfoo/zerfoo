// Package gather provides the Gather layer for the Zerfoo ML framework.
package gather

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Gather is a layer that gathers slices from a tensor.
type Gather[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	outputShape []int
	weights     *tensor.TensorNumeric[T]  // Optional embedded weights (data)
	indices     *tensor.TensorNumeric[int] // Optional embedded indices
}

// New creates a new Gather layer.
func New[T tensor.Numeric](engine compute.Engine[T]) *Gather[T] {
	return &Gather[T]{
		engine: engine,
	}
}

// NewWithWeights creates a new Gather layer with embedded weights.
func NewWithWeights[T tensor.Numeric](engine compute.Engine[T], weights *tensor.TensorNumeric[T]) *Gather[T] {
	return &Gather[T]{
		engine:  engine,
		weights: weights,
	}
}

// NewWithIndices creates a new Gather layer with embedded constant indices.
// At forward time, input[0] is the data tensor; indices come from the layer.
func NewWithIndices[T tensor.Numeric](engine compute.Engine[T], indices *tensor.TensorNumeric[int]) *Gather[T] {
	return &Gather[T]{
		engine:  engine,
		indices: indices,
	}
}

// OutputShape returns the output shape of the Gather layer.
func (g *Gather[T]) OutputShape() []int {
	return g.outputShape
}

// Parameters returns no trainable parameters for the Gather layer.
func (g *Gather[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// HasEmbeddedWeights returns true if this Gather layer has embedded weights.
func (g *Gather[T]) HasEmbeddedWeights() bool {
	return g.weights != nil
}

// EmbeddedFrozen returns the embedded frozen tensors (weights) that should
// be registered as frozen slots during compilation. Returns nil if no weights
// are embedded. Implements graph.EmbeddedFrozenProvider.
func (g *Gather[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	if g.weights == nil {
		return nil
	}
	return []*tensor.TensorNumeric[T]{g.weights}
}

// Forward computes the gather operation.
func (g *Gather[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	var (
		params  *tensor.TensorNumeric[T]
		indices *tensor.TensorNumeric[int]
	)

	switch {
	case g.indices != nil:
		// Embedded indices mode: single data input, indices from layer.
		if len(inputs) != 1 {
			return nil, fmt.Errorf("Gather layer with embedded indices expects 1 input (data), got %d", len(inputs))
		}
		params = inputs[0]
		indices = g.indices
	case g.weights != nil:
		if len(inputs) != 1 {
			return nil, fmt.Errorf("Gather layer with embedded weights expects 1 input (indices), got %d", len(inputs))
		}

		params = g.weights

		// Create int tensor with same shape as input
		inputTensor := inputs[0]

		intTensor, err := tensor.New[int](inputTensor.Shape(), nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create int tensor: %w", err)
		}

		// Convert input values to int (assuming they represent valid indices)
		// For now, we'll access the underlying data directly since we need to convert types
		// This is a simplified approach - in a production system, we'd want better type handling
		inputData := inputTensor.Data()

		intData := make([]int, len(inputData))
		for i, val := range inputData {
			// Convert to int (assuming T represents valid indices)
			intData[i] = int(float64(val))
		}

		intTensor.SetData(intData)

		// Ensure indices tensor is 2D as required by the engine
		if len(intTensor.Shape()) == 1 {
			// Reshape 1D tensor to 2D: [N] -> [1, N]
			newShape := []int{1, intTensor.Shape()[0]}

			reshapedTensor, err := tensor.New[int](newShape, intData)
			if err != nil {
				return nil, fmt.Errorf("failed to reshape indices tensor: %w", err)
			}

			indices = reshapedTensor
		} else {
			indices = intTensor
		}
	default:
		// General ONNX Gather: data and indices as inputs, axis=0.
		if len(inputs) < 2 {
			return nil, fmt.Errorf("Gather layer expects 2 inputs (data, indices), got %d", len(inputs))
		}

		params = inputs[0]

		// Convert float32 indices to int (ONNX Gather indices are int64 but
		// zerfoo uses float32 throughout).
		idxTensor := inputs[1]
		idxData := idxTensor.Data()
		intData := make([]int, len(idxData))
		for i, v := range idxData {
			intData[i] = int(float64(v))
		}

		var err error
		indices, err = tensor.New[int](idxTensor.Shape(), intData)
		if err != nil {
			return nil, fmt.Errorf("failed to create int indices tensor: %w", err)
		}
	}

	idxShape := indices.Shape()
	paramShape := params.Shape()

	// Handle scalar/1D index gathering from 1D data (e.g. gather one element
	// from a Shape output).
	if len(idxShape) == 0 || (len(idxShape) == 1 && idxShape[0] == 1) {
		// Scalar gather: return a single element from params axis 0.
		idx := indices.Data()[0]
		if idx < 0 {
			idx += paramShape[0]
		}
		subShape := paramShape[1:]
		stride := 1
		for _, d := range subShape {
			stride *= d
		}
		start := idx * stride
		end := start + stride
		if end > len(params.Data()) {
			end = len(params.Data())
		}
		if start >= len(params.Data()) {
			start = len(params.Data()) - stride
		}
		result := params.Data()[start:end]
		g.outputShape = subShape
		return tensor.New(subShape, result)
	}

	// Embedding-style gather: 2D indices, 2D+ params.
	if len(idxShape) < 2 {
		// Reshape 1D indices to [1, N]
		newShape := []int{1, idxShape[0]}
		var err error
		indices, err = tensor.New[int](newShape, indices.Data())
		if err != nil {
			return nil, fmt.Errorf("failed to reshape indices: %w", err)
		}
		idxShape = newShape
	}

	batchSize := idxShape[0]
	numIndices := idxShape[1]
	embeddingDim := paramShape[len(paramShape)-1]
	outputShape := []int{batchSize, numIndices, embeddingDim}
	g.outputShape = outputShape

	// Clamp indices to valid range [0, vocab) to handle cases where
	// position IDs exceed dynamically-sliced tables (e.g. RoPE cos/sin).
	vocab := paramShape[0]
	idxData := indices.Data()
	clamped := false
	for i, v := range idxData {
		if v < 0 {
			idxData[i] = 0
			clamped = true
		} else if v >= vocab {
			idxData[i] = vocab - 1
			clamped = true
		}
	}
	if clamped {
		indices, _ = tensor.New[int](idxShape, idxData)
	}

	output, err := tensor.New[T](outputShape, nil)
	if err != nil {
		return nil, err
	}

	return output, g.engine.Gather(ctx, params, indices, output)
}

// Backward computes the gradients for the Gather layer.
func (g *Gather[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The Gather layer has no trainable parameters, so the gradient is passed
	// through to the params tensor.
	params := inputs[0]

	indices, ok := any(inputs[1]).(*tensor.TensorNumeric[int])
	if !ok {
		return nil, fmt.Errorf("Gather layer expects indices to be of type *tensor.TensorNumeric[int], got %T", inputs[1])
	}

	dParams, err := tensor.New[T](params.Shape(), nil)
	if err != nil {
		return nil, err
	}

	if err := g.engine.ScatterAdd(ctx, dParams, indices, outputGradient); err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dParams, nil}, nil
}

// OpType returns the operation type of the Gather layer.
func (g *Gather[T]) OpType() string {
	return "Gather"
}

// Attributes returns nil for the Gather layer.
func (g *Gather[T]) Attributes() map[string]interface{} {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Gather[float32])(nil)

// Statically assert that the type implements the graph.EmbeddedFrozenProvider interface.
var _ graph.EmbeddedFrozenProvider[float32] = (*Gather[float32])(nil)
