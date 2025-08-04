// Package loss provides various loss functions for neural networks.
package loss

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// CrossEntropyLoss computes the cross-entropy loss.
type CrossEntropyLoss[T tensor.Numeric] struct {
	engine compute.Engine[T]

	// Cached tensors for backward pass
	predictions   *tensor.Tensor[T]   // Model's output (logits)
	targets       *tensor.Tensor[int] // True labels (int type)
	softmaxOutput *tensor.Tensor[T]   // Softmax of predictions
}

// NewCrossEntropyLoss creates a new CrossEntropyLoss layer.
func NewCrossEntropyLoss[T tensor.Numeric](engine compute.Engine[T]) *CrossEntropyLoss[T] {
	return &CrossEntropyLoss[T]{
		engine: engine,
	}
}

// OutputShape returns the output shape of the loss (a scalar).
func (cel *CrossEntropyLoss[T]) OutputShape(inputShapes ...[]int) ([]int, error) {
	if len(inputShapes) != 2 { // predictions, targets
		return nil, fmt.Errorf("CrossEntropyLoss: %w, expected %d, got %d", graph.ErrInvalidInputCount, 2, len(inputShapes))
	}
	// Loss is a scalar
	return []int{1}, nil
}

// Parameters returns an empty slice as CrossEntropyLoss has no trainable parameters.
func (cel *CrossEntropyLoss[T]) Parameters() []graph.Parameter[T] {
	return nil
}

// Forward computes the cross-entropy loss.
// Inputs: predictions (logits), targets (int labels).
func (cel *CrossEntropyLoss[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("CrossEntropyLoss: %w, expected %d, got %d", graph.ErrInvalidInputCount, 2, len(inputs))
	}
	predictions := inputs[0]   // Logits
	targetsTensor := inputs[1] // Targets are passed as *tensor.Tensor[T]

	// Convert targetsTensor to *tensor.Tensor[int] for Gather and OneHot
	targetsData := make([]int, targetsTensor.Size())
	for i, v := range targetsTensor.Data() {
		targetsData[i] = int(v) // Assuming T can be safely cast to int for labels
	}
	targets, err := tensor.New[int](targetsTensor.Shape(), targetsData)
	if err != nil {
		return nil, err
	}

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
	numElements := T(1.0)
	for _, dim := range predictions.Shape() {
		numElements *= T(dim)
	}
	// Divide by vocab size to get average per token
	numElements /= T(predictions.Shape()[len(predictions.Shape())-1])

	averageLoss, err := cel.engine.DivScalar(ctx, negatedLoss, numElements, nil)
	if err != nil {
		return nil, err
	}

	return averageLoss, nil
}

// Backward computes the gradients for CrossEntropyLoss.
// dOut is typically a scalar (1.0) for loss functions.
func (cel *CrossEntropyLoss[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("CrossEntropyLoss: %w, expected %d, got %d", graph.ErrInvalidInputCount, 2, len(inputs))
	}
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
	return []*tensor.Tensor[T]{finalGradPredictions, nil}, nil
}
