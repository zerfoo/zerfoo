package optimizer

import (
	"context"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// SWA wraps an Optimizer with Stochastic Weight Averaging.
// Unlike EMA which averages every step, SWA averages at epoch boundaries.
// Call UpdateAverage at the end of each epoch (after startEpoch).
// Call SwapWeights before validation to use averaged weights.
type SWA[T tensor.Numeric] struct {
	inner      Optimizer[T]
	engine     compute.Engine[T]
	avgParams  map[*graph.Parameter[T]]*tensor.TensorNumeric[T]
	nAveraged  int
	startEpoch int
}

// NewSWA creates a new SWA wrapper around the given optimizer.
func NewSWA[T tensor.Numeric](inner Optimizer[T], engine compute.Engine[T], startEpoch int) *SWA[T] {
	return &SWA[T]{
		inner:      inner,
		engine:     engine,
		avgParams:  make(map[*graph.Parameter[T]]*tensor.TensorNumeric[T]),
		startEpoch: startEpoch,
	}
}

// Step delegates to the inner optimizer.
func (s *SWA[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	return s.inner.Step(ctx, params)
}

// UpdateAverage updates the running average of parameters.
// Should be called at the end of each epoch. Only averages when epoch >= startEpoch.
// Formula: avg = avg + (param - avg) / (n + 1)
func (s *SWA[T]) UpdateAverage(ctx context.Context, params []*graph.Parameter[T], epoch int) error {
	if epoch < s.startEpoch {
		return nil
	}

	ops := s.engine.Ops()
	n1 := ops.FromFloat64(float64(s.nAveraged + 1))

	for _, param := range params {
		if param.Value == nil {
			continue
		}

		avg, ok := s.avgParams[param]
		if !ok {
			avgTensor, err := tensor.New[T](param.Value.Shape(), nil)
			if err != nil {
				return err
			}
			if err := s.engine.Copy(ctx, avgTensor, param.Value); err != nil {
				return err
			}
			s.avgParams[param] = avgTensor
			continue
		}

		// diff = param.Value - avg
		diff, err := s.engine.Sub(ctx, param.Value, avg, nil)
		if err != nil {
			return err
		}
		// delta = diff / (n+1)
		delta, err := s.engine.DivScalar(ctx, diff, n1, nil)
		if err != nil {
			return err
		}
		// avg = avg + delta
		if _, err := s.engine.Add(ctx, avg, delta, avg); err != nil {
			return err
		}
	}

	s.nAveraged++
	return nil
}

// SwapWeights swaps live params with averaged params.
func (s *SWA[T]) SwapWeights(ctx context.Context, params []*graph.Parameter[T]) error {
	for _, param := range params {
		avg, ok := s.avgParams[param]
		if !ok || param.Value == nil {
			continue
		}
		tmp, err := tensor.New[T](param.Value.Shape(), nil)
		if err != nil {
			return err
		}
		if err := s.engine.Copy(ctx, tmp, param.Value); err != nil {
			return err
		}
		if err := s.engine.Copy(ctx, param.Value, avg); err != nil {
			return err
		}
		if err := s.engine.Copy(ctx, avg, tmp); err != nil {
			return err
		}
	}
	return nil
}

// NAveraged returns the number of checkpoints averaged so far.
func (s *SWA[T]) NAveraged() int {
	return s.nAveraged
}

var _ Optimizer[float32] = (*SWA[float32])(nil)
