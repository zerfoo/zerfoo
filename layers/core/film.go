package core

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// FiLM (Feature-wise Linear Modulation) layer.
// It takes a feature tensor and a context tensor as input.
// It generates scale and bias vectors from the context tensor and applies them to the feature tensor.
// Output = (feature * scale) + bias
type FiLM[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	scaleGen    *Linear[T]
	biasGen     *Linear[T]
	lastFeature *tensor.TensorNumeric[T]
	lastScale   *tensor.TensorNumeric[T]
	lastBias    *tensor.TensorNumeric[T]
	outputShape []int
}

// NewFiLM creates a new FiLM layer.
func NewFiLM[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	contextDim, featureDim int,
) (*FiLM[T], error) {
	if name == "" {
		return nil, errors.New("layer name cannot be empty")
	}

	scaleGen, err := NewLinear[T](name+"_scale_generator", engine, ops, contextDim, featureDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create scale generator: %w", err)
	}

	biasGen, err := NewLinear[T](name+"_bias_generator", engine, ops, contextDim, featureDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create bias generator: %w", err)
	}

	return &FiLM[T]{
		engine:      engine,
		scaleGen:    scaleGen,
		biasGen:     biasGen,
		outputShape: []int{1, featureDim}, // Assuming batch size of 1 for now
	}, nil
}

// OutputShape returns the output shape of the FiLM layer.
func (f *FiLM[T]) OutputShape() []int {
	return f.outputShape
}

// Forward performs the forward pass for the FiLM layer.
// It expects two inputs: inputs[0] is the feature tensor, inputs[1] is the context tensor.
func (f *FiLM[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("FiLM: %w, expected %d, got %d", graph.ErrInvalidInputCount, 2, len(inputs))
	}

	feature, ctxInput := inputs[0], inputs[1]
	f.lastFeature = feature

	// Generate scale and bias from the context
	scale, err := f.scaleGen.Forward(ctx, ctxInput)
	if err != nil {
		return nil, fmt.Errorf("FiLM scale generation failed: %w", err)
	}
	f.lastScale = scale

	bias, err := f.biasGen.Forward(ctx, ctxInput)
	if err != nil {
		return nil, fmt.Errorf("FiLM bias generation failed: %w", err)
	}
	f.lastBias = bias

	// Apply FiLM: (feature * scale) + bias
	scaledFeature, err := f.engine.Mul(ctx, feature, scale)
	if err != nil {
		return nil, fmt.Errorf("FiLM scaling failed: %w", err)
	}

	output, err := f.engine.Add(ctx, scaledFeature, bias)
	if err != nil {
		return nil, fmt.Errorf("FiLM bias addition failed: %w", err)
	}

	f.outputShape = output.Shape()
	return output, nil
}

// Backward computes the gradients for the FiLM layer.
func (f *FiLM[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("FiLM: %w, expected %d, got %d", graph.ErrInvalidInputCount, 2, len(inputs))
	}

	ctxInput := inputs[1]

	// Gradient of the bias is just the output gradient
	dBias, err := f.biasGen.Backward(ctx, mode, outputGradient, ctxInput)
	if err != nil {
		return nil, fmt.Errorf("FiLM bias backward pass failed: %w", err)
	}

	// Gradient for the scaled feature is also the output gradient
	dScaledFeature := outputGradient

	// Gradient for the scale is dScaledFeature * feature
	dScale, err := f.engine.Mul(ctx, dScaledFeature, f.lastFeature)
	if err != nil {
		return nil, fmt.Errorf("FiLM dScale calculation failed: %w", err)
	}

	dScaleContext, err := f.scaleGen.Backward(ctx, mode, dScale, ctxInput)
	if err != nil {
		return nil, fmt.Errorf("FiLM scale backward pass failed: %w", err)
	}

	// Gradient for the feature is dScaledFeature * scale
	dFeature, err := f.engine.Mul(ctx, dScaledFeature, f.lastScale)
	if err != nil {
		return nil, fmt.Errorf("FiLM dFeature calculation failed: %w", err)
	}

	// The context gradient is the sum of the gradients from the scale and bias generators.
	dContext, err := f.engine.Add(ctx, dScaleContext[0], dBias[0])
	if err != nil {
		return nil, fmt.Errorf("FiLM dContext calculation failed: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dFeature, dContext}, nil
}

// Parameters returns the parameters of the FiLM layer.
func (f *FiLM[T]) Parameters() []*graph.Parameter[T] {
	params := f.scaleGen.Parameters()
	params = append(params, f.biasGen.Parameters()...)
	return params
}

// OpType returns the operation type of the FiLM layer.
func (f *FiLM[T]) OpType() string {
	return "FiLM"
}

// Attributes returns nil for the FiLM layer.
func (f *FiLM[T]) Attributes() map[string]interface{} {
	return nil
}

// Ensure FiLM implements the graph.Node interface.
var _ graph.Node[float32] = (*FiLM[float32])(nil)
