// Package loss provides various loss functions for neural networks.
package loss

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// CrossEntropyLoss computes the cross-entropy loss.
type CrossEntropyLoss[T tensor.Numeric] struct {
	engine compute.Engine[T]

	// Cached tensors for backward pass
	predictions   *tensor.TensorNumeric[T]   // Model's output (logits)
	targets       *tensor.TensorNumeric[int] // True labels (int type)
	softmaxOutput *tensor.TensorNumeric[T]   // Softmax of predictions
	outputShape   []int
}

// NewCrossEntropyLoss creates a new CrossEntropyLoss layer.
func NewCrossEntropyLoss[T tensor.Numeric](engine compute.Engine[T]) *CrossEntropyLoss[T] {
	return &CrossEntropyLoss[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the loss (a scalar).
func (cel *CrossEntropyLoss[T]) OutputShape() []int {
	return cel.outputShape
}

// Parameters returns an empty slice as CrossEntropyLoss has no trainable parameters.
func (cel *CrossEntropyLoss[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward computes the cross-entropy loss.
// Inputs: predictions (logits), targets (int labels).
func (cel *CrossEntropyLoss[T]) Forward(ctx context.Context, predictions *tensor.TensorNumeric[T], targets *tensor.TensorNumeric[int]) (*tensor.TensorNumeric[T], error) {
	cel.outputShape = []int{1} // Loss is a scalar

	cel.predictions = predictions // Cache for backward
	cel.targets = targets         // Cache for backward

	// Apply softmax to predictions
	// Assuming a Softmax function is available in compute.Engine or as a helper.
	// For numerical stability, it's often combined with log.
	// Here, we'll do softmax then log.
	softmaxOutput, err := cel.engine.Softmax(ctx, predictions, len(predictions.Shape())-1, nil) // Assuming Softmax is available
	if err != nil {
		return nil, err
	}

	cel.softmaxOutput = softmaxOutput // Cache for backward

	// Take log of softmax output
	logSoftmaxOutput, err := cel.engine.Log(ctx, softmaxOutput, nil) // Assuming Log is available
	if err != nil {
		return nil, err
	}

	// Gather negative log-probabilities for target classes
	// -log(softmax(predictions)[target_index])
	// This requires a gather operation on logSoftmaxOutput using targets as indices.
	// The result will be (batch_size, seq_len) if predictions are (batch_size, seq_len, vocab_size)
	gatheredLossShape := targets.Shape()

	gatheredLoss, err := tensor.New[T](gatheredLossShape, nil) // Create a new tensor for gatheredLoss
	if err != nil {
		return nil, err
	}

	err = cel.engine.Gather(ctx, logSoftmaxOutput, targets, gatheredLoss)
	if err != nil {
		return nil, err
	}

	// Sum over all elements and negate for total loss
	// Average over batch and sequence length
	totalLoss, err := cel.engine.ReduceSum(ctx, gatheredLoss, -1, false, nil) // Sum all elements
	if err != nil {
		return nil, err
	}

	// Negate the sum
	negatedLoss, err := cel.engine.MulScalar(ctx, totalLoss, cel.engine.Ops().FromFloat64(-1.0), nil)
	if err != nil {
		return nil, err
	}

	// Average loss over batch size and sequence length
	// Compute denominator as float64 and convert once.
	denomF64 := 1.0
	for _, dim := range predictions.Shape() {
		denomF64 *= float64(dim)
	}
	// Divide by vocab size to get average per token
	denomF64 /= float64(predictions.Shape()[len(predictions.Shape())-1])
	denom := cel.engine.Ops().FromFloat64(denomF64)

	averageLoss, err := cel.engine.DivScalar(ctx, negatedLoss, denom, nil)
	if err != nil {
		return nil, err
	}

	return averageLoss, nil
}

// Backward computes the gradients for CrossEntropyLoss.
func (cel *CrossEntropyLoss[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The gradient of Cross-Entropy Loss with respect to logits (predictions)
	// is (softmax(predictions) - one_hot(targets))
	// dOut is the gradient from the subsequent layer (usually 1.0 for loss)

	// Create one-hot encoded targets
	// targets: (batch_size, seq_len)
	// oneHotTargets: (batch_size, seq_len, vocab_size)
	vocabSize := cel.predictions.Shape()[len(cel.predictions.Shape())-1]

	oneHotTargets, err := cel.engine.OneHot(ctx, cel.targets, vocabSize, nil)
	if err != nil {
		return nil, err
	}

	// (softmax(predictions) - one_hot(targets))
	gradPredictions, err := cel.engine.Sub(ctx, cel.softmaxOutput, oneHotTargets, nil)
	if err != nil {
		return nil, err
	}

	// Multiply by dOut (which is usually 1.0 for loss)
	// If dOut is a scalar, it will broadcast.
	finalGradPredictions, err := cel.engine.Mul(ctx, gradPredictions, dOut, nil)
	if err != nil {
		return nil, err
	}

	// Loss function does not pass gradients back to targets (they are ground truth).
	return []*tensor.TensorNumeric[T]{finalGradPredictions, nil}, nil
}
