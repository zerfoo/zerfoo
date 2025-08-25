// Package training provides tools for training neural networks.
package training

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// Trainer is an interface for model-specific training orchestration.
type Trainer[T tensor.Numeric] interface {
	// TrainStep performs a single training step for a model.
	// It takes the model's parameters, the optimizer, and the input/target data.
	// It is responsible for computing the loss, gradients, and updating the parameters.
	TrainStep(
		ctx context.Context,
		modelGraph *graph.Graph[T],
		optimizer optimizer.Optimizer[T],
		inputs map[graph.Node[T]]*tensor.TensorNumeric[T],
		targets *tensor.TensorNumeric[T],
	) (loss T, err error)
}

// ... (rest of the file)
