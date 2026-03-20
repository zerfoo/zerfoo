// Package scheduler provides learning rate scheduling strategies for optimizers.
package scheduler

import "github.com/zerfoo/ztensor/tensor"

// Scheduler adjusts the learning rate over the course of training.
type Scheduler[T tensor.Numeric] interface {
	// Step advances the scheduler by one epoch. The metric value is used
	// by metric-aware schedulers such as ReduceOnPlateau; schedulers that
	// do not use a metric may ignore it.
	Step(epoch int, metric float64)

	// GetLR returns the current learning rate.
	GetLR() T
}
