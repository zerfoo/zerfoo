package normalization

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// RMSNorm is a struct that implements the graph.Node interface for RMSNorm.
type RMSNorm[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	epsilon T
	gain    *graph.Parameter[T] // Learnable gain parameter

	outputShape []int
}

// OpType returns the operation type.
func (r *RMSNorm[T]) OpType() string {
	return "RMSNorm"
}

// Attributes returns the attributes.
func (r *RMSNorm[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"epsilon": r.epsilon,
	}
}

// RMSNormOptions holds configuration options for RMSNorm layers.
type RMSNormOptions[T tensor.Numeric] struct {
	Epsilon T // Small constant to avoid division by zero
}

// RMSNormOption is a functional option for configuring RMSNorm layers.
type RMSNormOption[T tensor.Numeric] func(*RMSNormOptions[T])

// WithRMSNormEpsilon sets the epsilon parameter for RMSNorm.
func WithRMSNormEpsilon[T tensor.Numeric](epsilon T) RMSNormOption[T] {
	return func(opts *RMSNormOptions[T]) {
		opts.Epsilon = epsilon
	}
}

// NewRMSNorm creates a new RMSNorm layer.
func NewRMSNorm[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim int, options ...RMSNormOption[T]) (*RMSNorm[T], error) {
	// Apply functional options
	opts := &RMSNormOptions[T]{
		Epsilon: ops.FromFloat64(1e-6), // Default epsilon value
	}
	for _, option := range options {
		option(opts)
	}

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
		epsilon: opts.Epsilon,
		gain:    gainParam,
	}, nil
}

// NewRMSNormFromParam creates a new RMSNorm layer from an existing gain parameter.
func NewRMSNormFromParam[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], epsilon T, gain *graph.Parameter[T]) (*RMSNorm[T], error) {
	return &RMSNorm[T]{
		engine:  engine,
		ops:     ops,
		epsilon: epsilon,
		gain:    gain,
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
func (r *RMSNorm[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RMSNorm: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}

	input := inputs[0]
	r.outputShape = input.Shape()

	res, err := rmsNormalize(ctx, r.engine, input, r.gain.Value, r.epsilon)
	if err != nil {
		return nil, err
	}

	return res.output, nil
}

// Backward computes the backward pass of the RMSNorm layer.
func (r *RMSNorm[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RMSNorm: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}

	// Recompute the RMS statistics from the live input instead of reading
	// tensors cached during Forward (ztensor ADR 006; zerfoo#842 bug class).
	input := inputs[0]

	rms, normalized, err := rmsRecomputeStats(ctx, r.engine, input, r.epsilon)
	if err != nil {
		return nil, err
	}

	dGain, err := r.engine.Mul(ctx, dOut, normalized, nil)
	if err != nil {
		return nil, err
	}
	// Sum gradients over all dimensions except the last (feature) dimension.
	dGainSum := dGain
	ndim := len(dGain.Shape())
	for dim := range ndim - 1 {
		dGainSum, err = r.engine.ReduceSum(ctx, dGainSum, dim, true)
		if err != nil {
			return nil, err
		}
	}

	// Reshape reduced gradient to [dim] to match gain shape.
	dGainSum, err = dGainSum.Reshape(r.gain.Value.Shape())
	if err != nil {
		return nil, fmt.Errorf("RMSNorm: failed to reshape gain gradient: %w", err)
	}

	if r.gain.Gradient == nil {
		r.gain.Gradient = dGainSum
	} else {
		r.gain.Gradient, err = r.engine.Add(ctx, r.gain.Gradient, dGainSum, r.gain.Gradient)
		if err != nil {
			return nil, err
		}
	}

	// Gradient of the input
	dNormalized, err := r.engine.Mul(ctx, dOut, r.gain.Value, nil)
	if err != nil {
		return nil, err
	}

	term1, err := r.engine.Mul(ctx, dNormalized, rms, nil)
	if err != nil {
		return nil, err
	}

	rmsCubed, err := r.engine.Mul(ctx, rms, rms, nil)
	if err != nil {
		return nil, err
	}

	rmsCubed, err = r.engine.Mul(ctx, rmsCubed, rms, nil)
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

	lastDim := len(input.Shape()) - 1
	sumDNormX, err = r.engine.ReduceSum(ctx, sumDNormX, lastDim, true)
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

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// SetName sets the name of the RMSNorm layer.
func (r *RMSNorm[T]) SetName(name string) {
	r.gain.Name = name + "_gain"
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*RMSNorm[float32])(nil)
