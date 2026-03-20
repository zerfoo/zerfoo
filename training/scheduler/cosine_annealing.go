package scheduler

import (
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// CosineAnnealingConfig holds configuration for the CosineAnnealing scheduler.
type CosineAnnealingConfig[T tensor.Numeric] struct {
	// EtaMax is the initial (maximum) learning rate.
	EtaMax T

	// EtaMin is the minimum learning rate.
	EtaMin float64

	// TMax is the number of epochs for a full cosine cycle.
	TMax int

	// WarmRestarts enables cosine annealing with warm restarts. When true,
	// the epoch counter resets every TMax epochs.
	WarmRestarts bool
}

// CosineAnnealing implements cosine annealing learning rate scheduling.
type CosineAnnealing[T tensor.Numeric] struct {
	etaMax       float64
	etaMin       float64
	tMax         int
	warmRestarts bool
	lr           float64
	toT          func(float64) T
}

// NewCosineAnnealing creates a new CosineAnnealing scheduler.
func NewCosineAnnealing[T tensor.Numeric](cfg CosineAnnealingConfig[T]) *CosineAnnealing[T] {
	etaMax := float64FromNumeric(cfg.EtaMax)
	return &CosineAnnealing[T]{
		etaMax:       etaMax,
		etaMin:       cfg.EtaMin,
		tMax:         cfg.TMax,
		warmRestarts: cfg.WarmRestarts,
		lr:           etaMax,
		toT:          converterFor[T](),
	}
}

// Step computes the learning rate for the given epoch.
func (c *CosineAnnealing[T]) Step(epoch int, _ float64) {
	t := epoch
	if c.warmRestarts && c.tMax > 0 {
		t = epoch % c.tMax
	}

	if c.tMax > 0 {
		c.lr = c.etaMin + 0.5*(c.etaMax-c.etaMin)*(1+math.Cos(math.Pi*float64(t)/float64(c.tMax)))
	}
}

// GetLR returns the current learning rate.
func (c *CosineAnnealing[T]) GetLR() T {
	return c.toT(c.lr)
}

// Compile-time interface check.
var _ Scheduler[float32] = (*CosineAnnealing[float32])(nil)
