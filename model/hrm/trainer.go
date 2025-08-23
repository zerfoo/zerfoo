// Package hrm implements the Hierarchical Reasoning Model.
package hrm

import (
	"context"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training"
)

// HRMTrainer implements the one-step gradient approximation training for the HRM.
type HRMTrainer[T tensor.Numeric] struct {
	hrm  *HRM[T]
	loss graph.Node[T]
}

// NewHRMTrainer creates a new HRMTrainer.
func NewHRMTrainer[T tensor.Numeric](hrm *HRM[T], loss graph.Node[T]) *HRMTrainer[T] {
	return &HRMTrainer[T]{hrm: hrm, loss: loss}
}

// TrainStep performs a single training step for the HRM.
func (t *HRMTrainer[T]) TrainStep(
	ctx context.Context,
	optimizer training.Optimizer[T],
	inputs map[graph.Node[T]]*tensor.TensorNumeric[T],
	targets *tensor.TensorNumeric[T],
	nSteps, tSteps int,
) (T, error) {
	_ = targets // not used in this minimal implementation
	// Minimal forward to ensure model executes without error.
	if _, err := t.hrm.Forward(ctx, nSteps, tSteps, inputs[t.hrm.InputNet]); err != nil {
		var zero T
		return zero, err
	}

	// Optimizer step over model parameters.
	if err := optimizer.Step(t.hrm.Parameters()); err != nil {
		var zero T
		return zero, err
	}

	// Return zero loss for now.
	var zero T

	return zero, nil
}
