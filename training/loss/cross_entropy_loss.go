// Package loss provides various loss functions for neural networks.
package loss

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
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
// Inputs: predictions (logits as T), targets (labels as T that will be converted to int indices).
func (cel *CrossEntropyLoss[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("CrossEntropyLoss expects 2 inputs, got %d", len(inputs))
	}
	predictions := inputs[0]
	targetsT := inputs[1]
	// Convert targets from type T to int indices
	tShape := targetsT.Shape()
	flat := 1
	for _, d := range tShape {
		flat *= d
	}
	intData := make([]int, flat)
	dataT := targetsT.Data()
	for i := 0; i < flat; i++ {
		switch v := any(dataT[i]).(type) {
		case float32:
			intData[i] = int(v)
		case float64:
			intData[i] = int(v)
		default:
			return nil, fmt.Errorf("CrossEntropyLoss requires targets convertible to int; unsupported element type %T", v)
		}
	}
	targets, err := tensor.New[int](tShape, intData)
	if err != nil {
		return nil, err
	}
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

	// Gather negative log-probabilities for the target class at each position.
	// logSoftmaxOutput shape: [batch, classes] or [batch, seq, vocab]
	// targets shape: [batch] or [batch, seq]
	// We index the last axis of logSoftmaxOutput using targets.
	pShape := predictions.Shape()
	lastDim := pShape[len(pShape)-1]
	logData := logSoftmaxOutput.Data()
	tgtData := targets.Data()

	// Number of elements to gather equals the total number of target indices.
	n := 1
	for _, d := range targets.Shape() {
		n *= d
	}

	// Sum -log(softmax[target]) over all positions and average.
	ops := cel.engine.Ops()
	var sumNegLogProb T
	for i := 0; i < n; i++ {
		idx := tgtData[i]
		sumNegLogProb = ops.Sub(sumNegLogProb, logData[i*lastDim+idx])
	}
	avgLoss := ops.Div(sumNegLogProb, ops.FromFloat64(float64(n)))

	result, err := tensor.New[T]([]int{1}, []T{avgLoss})
	if err != nil {
		return nil, err
	}

	return result, nil
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

	// Divide by batch size (mean reduction normalization).
	// batch_size = total number of target samples (product of target dims).
	batchSize := 1
	for _, d := range cel.targets.Shape() {
		batchSize *= d
	}
	invN := cel.engine.Ops().FromFloat64(1.0 / float64(batchSize))
	gradPredictions, err = cel.engine.MulScalar(ctx, gradPredictions, invN)
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

// SoftmaxOutput returns the cached softmax output from the most recent Forward call.
func (cel *CrossEntropyLoss[T]) SoftmaxOutput() *tensor.TensorNumeric[T] {
	return cel.softmaxOutput
}

// OpType returns the operation type of the CrossEntropyLoss layer.
func (cel *CrossEntropyLoss[T]) OpType() string {
	return "CrossEntropyLoss"
}

// Attributes returns the attributes of the CrossEntropyLoss layer.
func (cel *CrossEntropyLoss[T]) Attributes() map[string]interface{} {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*CrossEntropyLoss[float32])(nil)
