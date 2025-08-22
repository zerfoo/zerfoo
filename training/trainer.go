package training

import (
	"context"

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
func (t *Trainer[T]) Train(ctx context.Context, inputs, targets *tensor.TensorNumeric[T]) (T, error) {
	// Forward pass
	predictions, err := t.model.Forward(ctx, inputs)
	if err != nil {
		return *new(T), err
	}

	// Calculate loss
	lossValue, lossGrad, err := t.lossFn.Forward(ctx, predictions, targets)
	if err != nil {
		return *new(T), err
	}

	// Backward pass
	_, err = t.model.Backward(ctx, lossGrad, inputs)
	if err != nil {
		return *new(T), err
	}

	// Update parameters
	err = t.optimizer.Step(ctx, t.model.Parameters())
	if err != nil {
		return *new(T), err
	}

	return lossValue, nil
}
