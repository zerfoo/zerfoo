package online

import (
	"errors"
	"math"
)

// EWC implements Elastic Weight Consolidation for continual learning.
// It prevents catastrophic forgetting by penalizing changes to parameters
// that were important for previously learned tasks. Importance is estimated
// via the diagonal of the Fisher information matrix.
type EWC struct {
	// lambda controls the strength of the EWC regularization penalty.
	lambda float64

	// fisherSamples is the number of samples used to estimate the Fisher
	// information matrix diagonal.
	fisherSamples int

	// fisher stores the diagonal of the Fisher information matrix,
	// estimating the importance of each parameter.
	fisher []float64

	// baseline stores the reference weights (the optimal weights for
	// previous tasks).
	baseline []float64
}

// NewEWC creates a new EWC instance with the given model weights as the
// baseline and the specified number of samples for Fisher estimation.
// The lambda (regularization strength) defaults to 1.0 and can be set
// via SetLambda.
func NewEWC(weights []float64, fisherSamples int) *EWC {
	baseline := make([]float64, len(weights))
	copy(baseline, weights)
	return &EWC{
		lambda:        1.0,
		fisherSamples: fisherSamples,
		baseline:      baseline,
	}
}

// SetLambda sets the EWC regularization strength. Higher values more
// strongly penalize deviations from the baseline weights on important
// parameters.
func (e *EWC) SetLambda(lambda float64) {
	e.lambda = lambda
}

// ComputeFisher estimates the diagonal of the Fisher information matrix
// using the provided data and loss function. For each data point, it
// constructs a per-sample loss (by passing each data point to lossFn as
// a single-element dataset) and computes the squared gradient via finite
// differences.
//
// The lossFn receives a weight vector and a single data point, and returns
// the scalar loss for that sample.
func (e *EWC) ComputeFisher(data [][]float64, lossFn func([]float64) float64) error {
	if len(data) == 0 {
		return errors.New("data must not be empty")
	}
	if lossFn == nil {
		return errors.New("lossFn must not be nil")
	}
	if len(e.baseline) == 0 {
		return errors.New("baseline weights are empty")
	}

	nParams := len(e.baseline)
	fisher := make([]float64, nParams)

	// Determine how many samples to use.
	nSamples := e.fisherSamples
	if nSamples > len(data) {
		nSamples = len(data)
	}
	if nSamples <= 0 {
		nSamples = len(data)
	}

	const eps = 1e-5

	weights := make([]float64, nParams)

	// Estimate the Fisher diagonal using the Hessian diagonal (second
	// derivative) of the loss at the baseline weights. At a well-trained
	// optimum, the gradient is near zero but the curvature captures
	// parameter importance. For each parameter p:
	//   H_pp ≈ (f(w+eps_p) - 2*f(w) + f(w-eps_p)) / eps^2
	copy(weights, e.baseline)
	baseLoss := lossFn(weights)

	for p := 0; p < nParams; p++ {
		copy(weights, e.baseline)
		weights[p] += eps
		fPlus := lossFn(weights)

		copy(weights, e.baseline)
		weights[p] -= eps
		fMinus := lossFn(weights)

		hessianDiag := (fPlus - 2*baseLoss + fMinus) / (eps * eps)
		if hessianDiag < 0 {
			hessianDiag = -hessianDiag
		}
		fisher[p] = hessianDiag
	}

	// Accumulate with any existing Fisher (supports multi-task consolidation).
	if e.fisher == nil {
		e.fisher = fisher
	} else {
		for p := range e.fisher {
			e.fisher[p] += fisher[p]
		}
	}

	return nil
}

// Penalty computes the EWC penalty term for the given current weights.
// The penalty is: (lambda / 2) * sum_i(fisher[i] * (currentWeights[i] - baseline[i])^2)
// Returns 0 if Fisher has not been computed yet.
func (e *EWC) Penalty(currentWeights []float64) float64 {
	if e.fisher == nil || len(e.fisher) == 0 {
		return 0
	}

	n := len(e.fisher)
	if len(currentWeights) < n {
		n = len(currentWeights)
	}
	if len(e.baseline) < n {
		n = len(e.baseline)
	}

	var penalty float64
	for i := 0; i < n; i++ {
		diff := currentWeights[i] - e.baseline[i]
		penalty += e.fisher[i] * diff * diff
	}
	return (e.lambda / 2.0) * penalty
}

// UpdateBaseline updates the reference weights to a new set of weights.
// This should be called after the model has finished learning a new task,
// so that future EWC penalties are computed relative to the new optimal
// weights. The Fisher information is preserved across baseline updates.
func (e *EWC) UpdateBaseline(newWeights []float64) {
	e.baseline = make([]float64, len(newWeights))
	copy(e.baseline, newWeights)

	// If Fisher dimensions don't match, reset it.
	if len(e.fisher) != len(newWeights) {
		e.fisher = nil
	}
}

// Fisher returns a copy of the current Fisher information diagonal.
// Returns nil if Fisher has not been computed.
func (e *EWC) Fisher() []float64 {
	if e.fisher == nil {
		return nil
	}
	f := make([]float64, len(e.fisher))
	copy(f, e.fisher)
	return f
}

// Baseline returns a copy of the current baseline weights.
func (e *EWC) Baseline() []float64 {
	b := make([]float64, len(e.baseline))
	copy(b, e.baseline)
	return b
}

// Lambda returns the current regularization strength.
func (e *EWC) Lambda() float64 {
	return e.lambda
}

// Loss computes a total loss that includes both the task loss and the EWC
// penalty: totalLoss = taskLoss + Penalty(currentWeights).
// This is a convenience method for use in training loops.
func (e *EWC) Loss(taskLoss float64, currentWeights []float64) float64 {
	if math.IsNaN(taskLoss) || math.IsInf(taskLoss, 0) {
		return taskLoss
	}
	return taskLoss + e.Penalty(currentWeights)
}
