package federated

import (
	"errors"
	"math"
	"math/rand"
	"sync"
)

// DPConfig configures differential privacy noise injection.
type DPConfig struct {
	// Epsilon is the privacy budget parameter (must be > 0).
	Epsilon float64
	// Delta is the failure probability (must be in (0, 1)).
	Delta float64
	// ClipNorm is the L2 norm bound for gradient clipping.
	ClipNorm float64
	// Mechanism selects the noise distribution: "gaussian" or "laplacian".
	Mechanism string
}

// DPStrategy wraps any Strategy and adds differential privacy noise to
// aggregated weights. It clips each client update to ClipNorm before
// delegating to the inner strategy, then adds calibrated noise.
type DPStrategy struct {
	inner      Strategy
	config     DPConfig
	rng        *rand.Rand
	accountant *PrivacyAccountant
}

// NewDPStrategy creates a DPStrategy that wraps inner with the given DP config.
func NewDPStrategy(inner Strategy, config DPConfig) *DPStrategy {
	return &DPStrategy{
		inner:      inner,
		config:     config,
		rng:        rand.New(rand.NewSource(42)),
		accountant: &PrivacyAccountant{},
	}
}

// Accountant returns the privacy accountant tracking cumulative budget.
func (d *DPStrategy) Accountant() *PrivacyAccountant {
	return d.accountant
}

// Aggregate clips each client update, delegates to the inner strategy,
// then adds calibrated DP noise to the aggregated weights.
func (d *DPStrategy) Aggregate(updates []ModelUpdate) (*AggregatedModel, error) {
	if len(updates) == 0 {
		return nil, errors.New("dp: no updates to aggregate")
	}

	// Clip each client update to ClipNorm.
	clipped := make([]ModelUpdate, len(updates))
	for i, u := range updates {
		clipped[i] = ModelUpdate{
			ClientID: u.ClientID,
			Weights:  clipL2(u.Weights, d.config.ClipNorm),
			NSamples: u.NSamples,
			Metrics:  u.Metrics,
		}
	}

	// Delegate aggregation to inner strategy.
	agg, err := d.inner.Aggregate(clipped)
	if err != nil {
		return nil, err
	}

	// Add DP noise to aggregated weights.
	switch d.config.Mechanism {
	case "gaussian":
		sigma := d.gaussianSigma()
		for i := range agg.Weights {
			agg.Weights[i] += d.rng.NormFloat64() * sigma
		}
	case "laplacian":
		scale := d.config.ClipNorm / d.config.Epsilon
		for i := range agg.Weights {
			agg.Weights[i] += laplaceSample(d.rng, scale)
		}
	default:
		return nil, errors.New("dp: unsupported mechanism: " + d.config.Mechanism)
	}

	// Account for privacy spent this round.
	d.accountant.addRound(d.config.Epsilon, d.config.Delta)

	return agg, nil
}

// SelectClients delegates to the inner strategy.
func (d *DPStrategy) SelectClients(round int, available []ClientID) []ClientID {
	return d.inner.SelectClients(round, available)
}

// gaussianSigma computes σ = ClipNorm * sqrt(2 * ln(1.25/δ)) / ε.
func (d *DPStrategy) gaussianSigma() float64 {
	return d.config.ClipNorm * math.Sqrt(2*math.Log(1.25/d.config.Delta)) / d.config.Epsilon
}

// clipL2 clips a weight vector to the given L2 norm bound.
func clipL2(weights []float64, maxNorm float64) []float64 {
	norm := 0.0
	for _, w := range weights {
		norm += w * w
	}
	norm = math.Sqrt(norm)

	out := make([]float64, len(weights))
	if norm <= maxNorm {
		copy(out, weights)
		return out
	}
	scale := maxNorm / norm
	for i, w := range weights {
		out[i] = w * scale
	}
	return out
}

// laplaceSample draws from a Laplace(0, scale) distribution using inverse CDF.
func laplaceSample(rng *rand.Rand, scale float64) float64 {
	u := rng.Float64() - 0.5
	if u < 0 {
		return scale * math.Log(1+2*u)
	}
	return -scale * math.Log(1-2*u)
}

// PrivacyAccountant tracks the cumulative differential privacy budget spent
// across federated rounds using basic composition.
type PrivacyAccountant struct {
	mu      sync.Mutex
	epsilon float64
	delta   float64
}

// Spent returns the cumulative (epsilon, delta) privacy budget consumed.
func (p *PrivacyAccountant) Spent() (epsilon, delta float64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.epsilon, p.delta
}

// CanContinue returns true if the cumulative epsilon is below maxEpsilon.
func (p *PrivacyAccountant) CanContinue(maxEpsilon float64) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.epsilon < maxEpsilon
}

func (p *PrivacyAccountant) addRound(epsilon, delta float64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.epsilon += epsilon
	p.delta += delta
}
