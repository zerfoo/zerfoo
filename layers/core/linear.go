package core

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Linear is a linear layer.
type Linear[T tensor.Numeric] struct {
	name         string
	engine       compute.Engine[T]
	ops          numeric.Arithmetic[T]
	weights      *graph.Parameter[T]
	inputFeatures int
	outputFeatures int
}

func randomData[T tensor.Numeric](size int) []T {
	data := make([]T, size)
	for i := range data {
		data[i] = T(rand.Float32()) //nolint:gosec
	}
	return data
}

// NewLinear creates a new Linear layer.
func NewLinear[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputFeatures, outputFeatures int,
) (*Linear[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if inputFeatures <= 0 || outputFeatures <= 0 {
		return nil, fmt.Errorf("input and output features must be positive")
	}
	weightsTensor, err := tensor.New[T](
		[]int{inputFeatures, outputFeatures},
		randomData[T](inputFeatures*outputFeatures),
	)
	if err != nil {
		return nil, err
	}
	weights, err := graph.NewParameter[T](
		name+"_weights",
		weightsTensor,
		tensor.New[T],
	)
	if err != nil {
		return nil, err
	}

	return &Linear[T]{
		name:         name,
		engine:       engine,
		ops:          ops,
		weights:      weights,
		inputFeatures: inputFeatures,
		outputFeatures: outputFeatures,
	}, nil
}

// OpType returns the operation type of the layer.
func (l *Linear[T]) OpType() string {
	return "Linear"
}

// Attributes returns the attributes of the layer.
func (l *Linear[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_features":  l.inputFeatures,
		"output_features": l.outputFeatures,
	}
}

// OutputShape returns the output shape of the layer.
func (l *Linear[T]) OutputShape() []int {
	return []int{-1, l.outputFeatures} // -1 for batch dimension
}

// Forward computes the forward pass of the layer.
func (l *Linear[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Linear layer requires exactly one input, got %d", len(inputs))
	}

	return l.engine.MatMul(ctx, inputs[0], l.weights.Value)
}

// Backward computes the gradients.
func (l *Linear[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Linear layer requires exactly one input, got %d", len(inputs))
	}

	input := inputs[0]

	// Gradient with respect to weights
	inputShape := input.Shape()
	gradShape := outputGradient.Shape()
	
	// Reshape input and gradient to 2D for matrix operations
	batchSize := 1
	for i := 0; i < len(inputShape)-1; i++ {
		batchSize *= inputShape[i]
	}
	
	inputReshaped, err := l.engine.Reshape(ctx, input, []int{batchSize, inputShape[len(inputShape)-1]})
	if err != nil {
		return nil, err
	}
	
	gradReshaped, err := l.engine.Reshape(ctx, outputGradient, []int{batchSize, gradShape[len(gradShape)-1]})
	if err != nil {
		return nil, err
	}
	
	transposedInput, err := l.engine.Transpose(ctx, inputReshaped, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dw, err := l.engine.MatMul(ctx, transposedInput, gradReshaped)
	if err != nil {
		return nil, err
	}
	l.weights.Gradient, err = l.engine.Add(ctx, l.weights.Gradient, dw)
	if err != nil {
		return nil, err
	}

	// Gradient with respect to input
	transposedWeights, err := l.engine.Transpose(ctx, l.weights.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dx, err := l.engine.MatMul(ctx, gradReshaped, transposedWeights)
	if err != nil {
		return nil, err
	}
	
	// Reshape back to original input shape
	dxReshaped, err := l.engine.Reshape(ctx, dx, inputShape)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dxReshaped}, nil
}

// Parameters returns the parameters of the layer.
func (l *Linear[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{l.weights}
}

func init() {
	model.RegisterLayer("Linear", func(engine compute.Engine[float32], ops numeric.Arithmetic[float32], name string, params map[string]*graph.Parameter[float32], attributes map[string]interface{}) (graph.Node[float32], error) {
		inputFeatures, ok := attributes["input_features"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'input_features' for Linear")
		}
		outputFeatures, ok := attributes["output_features"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'output_features' for Linear")
		}
		return NewLinear[float32](name, engine, ops, inputFeatures, outputFeatures)
	})
}

// SetName sets the name of the Linear layer.
func (l *Linear[T]) SetName(name string) {
	l.name = name
	l.weights.Name = name + "_weights"
}

// Name returns the name of the Linear layer.
func (l *Linear[T]) Name() string {
	return l.name
}

// NewLinearFromParam creates a Linear layer from an existing parameter.
// This is used for constructing layers from pre-existing parameters during model loading.
func NewLinearFromParam[T tensor.Numeric](engine compute.Engine[T], param *graph.Parameter[T]) *Linear[T] {
	shape := param.Value.Shape()
	return &Linear[T]{
		name:           param.Name,
		engine:         engine,
		weights:        param,
		inputFeatures:  shape[0],
		outputFeatures: shape[1],
	}
}
