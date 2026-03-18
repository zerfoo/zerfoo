package online

import (
	"errors"
	"math"
	"math/rand/v2"
)

// Sample represents a single training example with input features and target labels.
type Sample struct {
	Input []float32
	Label []float32
}

// LoRAUpdateConfig holds parameters for an incremental LoRA update pass.
type LoRAUpdateConfig struct {
	// Rank is the low-rank dimension for LoRA adapter matrices.
	Rank int

	// Alpha is the scaling factor; the LoRA output is multiplied by Alpha/Rank.
	Alpha int

	// LR is the learning rate for SGD updates.
	LR float64

	// MaxSteps is the maximum number of gradient descent steps per Update call.
	MaxSteps int

	// TargetModules lists the module names that LoRA adapters are applied to.
	TargetModules []string
}

// loraAdapter holds a pair of low-rank matrices A (rank x dIn) and B (dOut x rank).
type loraAdapter struct {
	A    []float32 // rank x dIn
	B    []float32 // dOut x rank
	rank int
	dIn  int
	dOut int
}

// newLoraAdapter creates an adapter with Kaiming-initialized A and
// zero-initialized B, following standard LoRA convention.
func newLoraAdapter(rank, dIn, dOut int) *loraAdapter {
	a := make([]float32, rank*dIn)
	for i := range a {
		a[i] = float32(rand.NormFloat64()) * float32(math.Sqrt(2.0/float64(dIn)))
	}
	return &loraAdapter{
		A:    a,
		B:    make([]float32, dOut*rank),
		rank: rank,
		dIn:  dIn,
		dOut: dOut,
	}
}

// clone returns a deep copy of the adapter.
func (a *loraAdapter) clone() *loraAdapter {
	c := &loraAdapter{
		A:    make([]float32, len(a.A)),
		B:    make([]float32, len(a.B)),
		rank: a.rank,
		dIn:  a.dIn,
		dOut: a.dOut,
	}
	copy(c.A, a.A)
	copy(c.B, a.B)
	return c
}

// forward computes output = input + scale * B @ A @ input for a single sample.
// input is (dIn,), output is (dOut,).
func (a *loraAdapter) forward(input []float32, scale float32) []float32 {
	// A @ input => (rank,)
	ax := make([]float32, a.rank)
	for r := 0; r < a.rank; r++ {
		var sum float32
		for j := 0; j < a.dIn; j++ {
			sum += a.A[r*a.dIn+j] * input[j]
		}
		ax[r] = sum
	}

	// B @ ax => (dOut,)
	out := make([]float32, a.dOut)
	for i := 0; i < a.dOut; i++ {
		var sum float32
		for r := 0; r < a.rank; r++ {
			sum += a.B[i*a.rank+r] * ax[r]
		}
		out[i] = sum * scale
	}
	return out
}

// IncrementalUpdater applies incremental LoRA updates using SGD.
// It maintains adapter weights and supports rollback to the pre-update state.
type IncrementalUpdater struct {
	cfg      LoRAUpdateConfig
	adapter  *loraAdapter
	snapshot *loraAdapter // pre-update snapshot for rollback; nil after CommitUpdate
	scale    float32
}

// NewIncrementalUpdater creates a new updater with the given configuration.
// The adapter dimensions are inferred from the first Update call's sample sizes.
func NewIncrementalUpdater(cfg LoRAUpdateConfig) *IncrementalUpdater {
	return &IncrementalUpdater{
		cfg:   cfg,
		scale: float32(cfg.Alpha) / float32(cfg.Rank),
	}
}

// ensureAdapter lazily initializes the adapter from sample dimensions.
func (u *IncrementalUpdater) ensureAdapter(dIn, dOut int) {
	if u.adapter == nil {
		u.adapter = newLoraAdapter(u.cfg.Rank, dIn, dOut)
	}
}

// Update applies MaxSteps of SGD on the LoRA adapter parameters using the
// provided samples. Each step iterates over all samples, computing MSE loss
// gradients and updating A and B matrices in-place.
func (u *IncrementalUpdater) Update(samples []Sample) error {
	if len(samples) == 0 {
		return errors.New("no samples provided")
	}
	dIn := len(samples[0].Input)
	dOut := len(samples[0].Label)
	if dIn == 0 || dOut == 0 {
		return errors.New("sample input and label must be non-empty")
	}
	for i, s := range samples {
		if len(s.Input) != dIn || len(s.Label) != dOut {
			return errors.New("all samples must have consistent input and label dimensions")
		}
		_ = i
	}

	u.ensureAdapter(dIn, dOut)

	// Snapshot current weights for rollback.
	u.snapshot = u.adapter.clone()

	lr := float32(u.cfg.LR)
	rank := u.adapter.rank
	n := float32(len(samples))

	for step := 0; step < u.cfg.MaxSteps; step++ {
		// Accumulate gradients over all samples.
		dA := make([]float32, len(u.adapter.A))
		dB := make([]float32, len(u.adapter.B))

		for _, s := range samples {
			// Forward: ax = A @ input
			ax := make([]float32, rank)
			for r := 0; r < rank; r++ {
				var sum float32
				for j := 0; j < dIn; j++ {
					sum += u.adapter.A[r*dIn+j] * s.Input[j]
				}
				ax[r] = sum
			}

			// Forward: pred = scale * B @ ax
			pred := make([]float32, dOut)
			for i := 0; i < dOut; i++ {
				var sum float32
				for r := 0; r < rank; r++ {
					sum += u.adapter.B[i*rank+r] * ax[r]
				}
				pred[i] = sum * u.scale
			}

			// Error: e = pred - label (MSE gradient factor: 2/n * e)
			e := make([]float32, dOut)
			for i := 0; i < dOut; i++ {
				e[i] = pred[i] - s.Label[i]
			}

			// dL/dB: for MSE loss L = (1/n) * sum(||pred - label||^2)
			// dL/dB[i][r] = (2/n) * scale * e[i] * ax[r]
			for i := 0; i < dOut; i++ {
				for r := 0; r < rank; r++ {
					dB[i*rank+r] += (2.0 / n) * u.scale * e[i] * ax[r]
				}
			}

			// dL/dA: dL/dA[r][j] = (2/n) * scale * sum_i(e[i] * B[i][r]) * input[j]
			// First compute: gradR[r] = sum_i(e[i] * B[i][r])
			gradR := make([]float32, rank)
			for r := 0; r < rank; r++ {
				var sum float32
				for i := 0; i < dOut; i++ {
					sum += e[i] * u.adapter.B[i*rank+r]
				}
				gradR[r] = sum
			}
			for r := 0; r < rank; r++ {
				for j := 0; j < dIn; j++ {
					dA[r*dIn+j] += (2.0 / n) * u.scale * gradR[r] * s.Input[j]
				}
			}
		}

		// SGD update.
		for i := range u.adapter.A {
			u.adapter.A[i] -= lr * dA[i]
		}
		for i := range u.adapter.B {
			u.adapter.B[i] -= lr * dB[i]
		}
	}
	return nil
}

// Rollback restores the adapter weights to the state before the last Update call.
// Returns an error if no snapshot is available (e.g. after CommitUpdate or before
// any Update).
func (u *IncrementalUpdater) Rollback() error {
	if u.snapshot == nil {
		return errors.New("no snapshot available for rollback")
	}
	u.adapter = u.snapshot
	u.snapshot = nil
	return nil
}

// CommitUpdate finalizes the current update by discarding the rollback snapshot.
func (u *IncrementalUpdater) CommitUpdate() {
	u.snapshot = nil
}

// CurrentLoss computes the MSE loss over the given samples using the current
// adapter weights. Returns +Inf if there are no samples or the adapter is not
// initialized.
func (u *IncrementalUpdater) CurrentLoss(samples []Sample) float64 {
	if len(samples) == 0 || u.adapter == nil {
		return math.Inf(1)
	}

	var totalLoss float64
	for _, s := range samples {
		pred := u.adapter.forward(s.Input, u.scale)
		for i := 0; i < len(s.Label) && i < len(pred); i++ {
			diff := float64(pred[i] - s.Label[i])
			totalLoss += diff * diff
		}
	}
	return totalLoss / float64(len(samples))
}
