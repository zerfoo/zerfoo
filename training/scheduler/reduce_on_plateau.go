package scheduler

import (
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// ReduceOnPlateauConfig holds configuration for the ReduceOnPlateau scheduler.
type ReduceOnPlateauConfig[T tensor.Numeric] struct {
	// InitialLR is the starting learning rate.
	InitialLR T

	// Factor is the multiplier applied to the LR when reducing (e.g. 0.1).
	Factor float64

	// Patience is the number of epochs with no improvement before reducing.
	Patience int

	// MinLR is the minimum learning rate. LR will not be reduced below this value.
	MinLR float64

	// Mode is "min" (lower metric is better) or "max" (higher metric is better).
	Mode string
}

// ReduceOnPlateau reduces the learning rate when a metric has stopped improving.
type ReduceOnPlateau[T tensor.Numeric] struct {
	lr         float64
	factor     float64
	patience   int
	minLR      float64
	modeMin    bool
	best       float64
	waitEpochs int
	toT        func(float64) T
}

// NewReduceOnPlateau creates a new ReduceOnPlateau scheduler.
func NewReduceOnPlateau[T tensor.Numeric](cfg ReduceOnPlateauConfig[T]) *ReduceOnPlateau[T] {
	modeMin := cfg.Mode != "max"

	var best float64
	if modeMin {
		best = math.Inf(1)
	} else {
		best = math.Inf(-1)
	}

	return &ReduceOnPlateau[T]{
		lr:       float64FromNumeric(cfg.InitialLR),
		factor:   cfg.Factor,
		patience: cfg.Patience,
		minLR:    cfg.MinLR,
		modeMin:  modeMin,
		best:     best,
		toT:      converterFor[T](),
	}
}

// Step checks the metric and reduces LR if no improvement for patience epochs.
func (r *ReduceOnPlateau[T]) Step(_ int, metric float64) {
	improved := false
	if r.modeMin {
		improved = metric < r.best
	} else {
		improved = metric > r.best
	}

	if improved {
		r.best = metric
		r.waitEpochs = 0
		return
	}

	r.waitEpochs++
	if r.waitEpochs >= r.patience {
		r.lr *= r.factor
		if r.lr < r.minLR {
			r.lr = r.minLR
		}
		r.waitEpochs = 0
	}
}

// GetLR returns the current learning rate.
func (r *ReduceOnPlateau[T]) GetLR() T {
	return r.toT(r.lr)
}

// Compile-time interface check.
var _ Scheduler[float32] = (*ReduceOnPlateau[float32])(nil)
