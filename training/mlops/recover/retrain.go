package recover

import (
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/training/mlops/monitor"
)

// Phase represents a stage in the auto-retrain pipeline.
type Phase int

const (
	// PhaseDetect is the drift detection phase.
	PhaseDetect Phase = iota
	// PhaseRollback rolls back to the last known good model.
	PhaseRollback
	// PhaseRetrain retrains the model.
	PhaseRetrain
	// PhaseValidate validates the retrained model.
	PhaseValidate
	// PhaseRedeploy redeploys the validated model.
	PhaseRedeploy
)

// String returns the phase name.
func (p Phase) String() string {
	switch p {
	case PhaseDetect:
		return "detect"
	case PhaseRollback:
		return "rollback"
	case PhaseRetrain:
		return "retrain"
	case PhaseValidate:
		return "validate"
	case PhaseRedeploy:
		return "redeploy"
	default:
		return "unknown"
	}
}

// PipelineError wraps an error with the phase in which it occurred.
type PipelineError struct {
	Phase Phase
	Err   error
}

func (e *PipelineError) Error() string {
	return fmt.Sprintf("recover: %s phase failed: %v", e.Phase, e.Err)
}

func (e *PipelineError) Unwrap() error {
	return e.Err
}

// AutoRetrainConfig holds callbacks for each stage of the recovery pipeline.
type AutoRetrainConfig struct {
	// RollbackFn rolls back to the last known good model state.
	// Called when drift is detected. May be nil to skip rollback.
	RollbackFn func() error

	// RetrainFn retrains the model. Required.
	RetrainFn func() error

	// ValidateFn validates the retrained model.
	// Should return an error if validation fails. May be nil to skip.
	ValidateFn func() error

	// RedeployFn redeploys the validated model.
	// May be nil to skip.
	RedeployFn func() error
}

// AutoRetrain orchestrates an automated recovery pipeline:
// detect drift -> rollback -> retrain -> validate -> redeploy.
type AutoRetrain struct {
	cfg AutoRetrainConfig
}

// NewAutoRetrain creates an AutoRetrain with the given config.
// RetrainFn is required and must not be nil.
func NewAutoRetrain(cfg AutoRetrainConfig) (*AutoRetrain, error) {
	if cfg.RetrainFn == nil {
		return nil, errors.New("recover: RetrainFn is required")
	}
	return &AutoRetrain{cfg: cfg}, nil
}

// Run feeds values from the stream to the detector. When drift is detected,
// it executes the full recovery pipeline: rollback -> retrain -> validate -> redeploy.
// The stream function is called repeatedly to get the next observation value.
// It should return the value and true to continue, or 0 and false to stop.
func (ar *AutoRetrain) Run(detector monitor.DriftDetector, stream func() (float64, bool)) error {
	for {
		value, ok := stream()
		if !ok {
			return nil
		}
		if detector.Observe(value) {
			if err := ar.execute(); err != nil {
				return err
			}
		}
	}
}

// RunOnDrift executes the recovery pipeline immediately, without
// waiting for drift detection. Useful when drift has already been
// detected externally.
func (ar *AutoRetrain) RunOnDrift() error {
	return ar.execute()
}

// execute runs the recovery pipeline phases in order.
func (ar *AutoRetrain) execute() error {
	if ar.cfg.RollbackFn != nil {
		if err := ar.cfg.RollbackFn(); err != nil {
			return &PipelineError{Phase: PhaseRollback, Err: err}
		}
	}

	if err := ar.cfg.RetrainFn(); err != nil {
		return &PipelineError{Phase: PhaseRetrain, Err: err}
	}

	if ar.cfg.ValidateFn != nil {
		if err := ar.cfg.ValidateFn(); err != nil {
			return &PipelineError{Phase: PhaseValidate, Err: err}
		}
	}

	if ar.cfg.RedeployFn != nil {
		if err := ar.cfg.RedeployFn(); err != nil {
			return &PipelineError{Phase: PhaseRedeploy, Err: err}
		}
	}

	return nil
}
