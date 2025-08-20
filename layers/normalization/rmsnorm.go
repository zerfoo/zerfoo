package normalization

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// RMSNorm is a struct that implements the graph.Node interface for RMSNorm.
type RMSNorm[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	epsilon T
	gain    *graph.Parameter[T] // Learnable gain parameter

	// Cache for backward pass
	inputTensor *tensor.Tensor[T]
	rms         *tensor.Tensor[T]
	outputShape []int
}

// NewRMSNorm creates a new RMSNorm layer.
func NewRMSNorm[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim int, epsilon T) (*RMSNorm[T], error) {
	// Initialize gain as a learnable parameter with value 1.0
	gainData := make([]T, modelDim)
	for i := range gainData {
		gainData[i] = ops.One()
	}
	gainTensor, err := tensor.New[T]([]int{modelDim}, gainData)
	if err != nil {
		return nil, fmt.Errorf("failed to create gain tensor: %w", err)
	}

	gainParam, err := graph.NewParameter[T](name+"_gain", gainTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("failed to create gain parameter: %w", err)
	}

	return &RMSNorm[T]{
		engine:  engine,
		ops:     ops,
		epsilon: epsilon,
		gain:    gainParam,
	}, nil
}

// OutputShape returns the output shape of the RMSNorm layer.
func (r *RMSNorm[T]) OutputShape() []int {
	return r.outputShape
}

// Parameters returns the learnable parameters of the RMSNorm layer.
func (r *RMSNorm[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{r.gain}
}

// Forward computes the forward pass of the RMSNorm layer.
func (r *RMSNorm[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RMSNorm: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	input := inputs[0]
	r.inputTensor = input // Cache for backward pass
	r.outputShape = input.Shape()

	// Calculate sum of squares along the last dimension
	squared, err := r.engine.Mul(ctx, input, input, nil)
	if err != nil {
		return nil, err
	}
	lastDim := len(input.Shape()) - 1
	meanSq, err := r.engine.ReduceMean(ctx, squared, lastDim, true)
	if err != nil {
		return nil, err
	}

	// Add epsilon and compute reciprocal square root (rsqrt)
	meanSqPlusEpsilon, err := r.engine.AddScalar(ctx, meanSq, r.epsilon, nil)
	if err != nil {
		return nil, err
	}
	rsqrt, err := r.engine.Rsqrt(ctx, meanSqPlusEpsilon, nil)
	if err != nil {
		return nil, err
	}
	r.rms = rsqrt // Cache for backward pass

	// Normalize input and scale by gain
	normalized, err := r.engine.Mul(ctx, input, rsqrt, nil)
	if err != nil {
		return nil, err
	}
	output, err := r.engine.Mul(ctx, normalized, r.gain.Value, nil)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the backward pass of the RMSNorm layer.
func (r *RMSNorm[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RMSNorm: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	input := r.inputTensor

	// Gradient of the gain parameter
	normalized, err := r.engine.Mul(ctx, input, r.rms, nil)
	if err != nil {
		return nil, err
	}
	dGain, err := r.engine.Mul(ctx, dOut, normalized, nil)
	if err != nil {
		return nil, err
	}
	// Sum gradients over batch and sequence dimensions
	dGainSum, err := r.engine.ReduceSum(ctx, dGain, 0, true)
	if err != nil {
		return nil, err
	}
	dGainSum, err = r.engine.ReduceSum(ctx, dGainSum, 1, true)
	if err != nil {
		return nil, err
	}
	r.gain.Gradient, err = r.engine.Add(ctx, r.gain.Gradient, dGainSum, r.gain.Gradient)
	if err != nil {
		return nil, err
	}

	// Gradient of the input
	dNormalized, err := r.engine.Mul(ctx, dOut, r.gain.Value, nil)
	if err != nil {
		return nil, err
	}

	term1, err := r.engine.Mul(ctx, dNormalized, r.rms, nil)
	if err != nil {
		return nil, err
	}

	rmsCubed, err := r.engine.Mul(ctx, r.rms, r.rms, nil)
	if err != nil {
		return nil, err
	}
	rmsCubed, err = r.engine.Mul(ctx, rmsCubed, r.rms, nil)
	if err != nil {
		return nil, err
	}

	invN, err := tensor.New[T]([]int{1}, []T{r.ops.FromFloat64(1.0 / float64(input.Shape()[len(input.Shape())-1]))})
	if err != nil {
		return nil, err
	}

	sumDNormX, err := r.engine.Mul(ctx, dNormalized, input, nil)
	if err != nil {
		return nil, err
	}
	sumDNormX, err = r.engine.ReduceSum(ctx, sumDNormX, -1, true)
	if err != nil {
		return nil, err
	}

	term2, err := r.engine.Mul(ctx, input, sumDNormX, nil)
	if err != nil {
		return nil, err
	}
	term2, err = r.engine.Mul(ctx, term2, rmsCubed, nil)
	if err != nil {
		return nil, err
	}
	term2, err = r.engine.Mul(ctx, term2, invN, nil)
	if err != nil {
		return nil, err
	}

	dInput, err := r.engine.Sub(ctx, term1, term2, nil)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor[T]{dInput}, nil
}