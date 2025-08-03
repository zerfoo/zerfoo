package training

import (
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// Trainer encapsulates the training logic for a model.
type Trainer[T tensor.Numeric] struct {
	model     Model[T]
	optimizer optimizer.Optimizer[T]
	lossFn    loss.Loss[T]
}

// NewTrainer creates a new trainer.
func NewTrainer[T tensor.Numeric](model Model[T], optimizer optimizer.Optimizer[T], lossFn loss.Loss[T]) *Trainer[T] {
	return &Trainer[T]{
		model:     model,
		optimizer: optimizer,
		lossFn:    lossFn,
	}
}

// Train performs a single training step.
func (t *Trainer[T]) Train(inputs, targets *tensor.Tensor[T]) (T, error) {
	// Forward pass
	predictions := t.model.Forward(inputs)

	// Calculate loss
	lossValue, lossGrad := t.lossFn.Forward(predictions, targets)

	// Backward pass
	t.model.Backward(lossGrad)

	// Update parameters
	t.optimizer.Step(t.model.Parameters())

	return lossValue, nil
}
