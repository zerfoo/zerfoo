package federated

import "errors"

// FedProx implements the Federated Proximal strategy. It extends FedAvg by
// adding a proximal term (mu * ||w - w_global||^2) to the local training
// objective, which limits client model divergence from the global model.
// The aggregation step is identical to FedAvg (sample-weighted average),
// but clients are expected to incorporate the proximal penalty during training.
type FedProx struct {
	mu float64
}

// NewFedProx returns a new FedProx strategy. The mu parameter controls the
// strength of the proximal term: higher values penalize divergence from the
// global model more strongly.
func NewFedProx(mu float64) *FedProx {
	return &FedProx{mu: mu}
}

// Mu returns the proximal term strength.
func (f *FedProx) Mu() float64 {
	return f.mu
}

// Aggregate computes a sample-weighted average of client updates, identical
// to FedAvg. The proximal penalty is applied during client-side training
// (reflected in the weights clients send back), not during aggregation.
func (f *FedProx) Aggregate(updates []ModelUpdate) (*AggregatedModel, error) {
	if len(updates) == 0 {
		return nil, errors.New("fedprox: no updates to aggregate")
	}

	dim := len(updates[0].Weights)
	for _, u := range updates[1:] {
		if len(u.Weights) != dim {
			return nil, errors.New("fedprox: weight dimension mismatch")
		}
	}

	totalSamples := 0
	for _, u := range updates {
		if u.NSamples <= 0 {
			return nil, errors.New("fedprox: client reported non-positive sample count")
		}
		totalSamples += u.NSamples
	}

	aggregated := make([]float64, dim)
	for _, u := range updates {
		scale := float64(u.NSamples) / float64(totalSamples)
		for j, w := range u.Weights {
			aggregated[j] += w * scale
		}
	}

	return &AggregatedModel{
		Weights:       aggregated,
		NParticipants: len(updates),
	}, nil
}

// SelectClients returns all available clients for every round.
func (f *FedProx) SelectClients(_ int, available []ClientID) []ClientID {
	result := make([]ClientID, len(available))
	copy(result, available)
	return result
}

// ProximalLoss computes the proximal penalty term: (mu / 2) * ||localWeights - globalWeights||^2.
// This value is added to the local training loss to penalize client model divergence.
func ProximalLoss(localWeights, globalWeights []float64, mu float64) float64 {
	var sum float64
	for i := range localWeights {
		diff := localWeights[i] - globalWeights[i]
		sum += diff * diff
	}
	return (mu / 2.0) * sum
}
