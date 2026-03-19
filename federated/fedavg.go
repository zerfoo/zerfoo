package federated

import "errors"

// FedAvg implements the Federated Averaging strategy. It computes a weighted
// average of client model updates, where each client's contribution is
// proportional to its dataset size (NSamples).
type FedAvg struct{}

// NewFedAvg returns a new FedAvg strategy.
func NewFedAvg() *FedAvg {
	return &FedAvg{}
}

// Aggregate computes the weighted average of model updates. Each update's
// weights are scaled by the proportion of samples that client contributed.
func (f *FedAvg) Aggregate(updates []ModelUpdate) (*AggregatedModel, error) {
	if len(updates) == 0 {
		return nil, errors.New("fedavg: no updates to aggregate")
	}

	// All updates must have the same weight dimension.
	dim := len(updates[0].Weights)
	for _, u := range updates[1:] {
		if len(u.Weights) != dim {
			return nil, errors.New("fedavg: weight dimension mismatch")
		}
	}

	// Compute total samples across all participants.
	totalSamples := 0
	for _, u := range updates {
		if u.NSamples <= 0 {
			return nil, errors.New("fedavg: client reported non-positive sample count")
		}
		totalSamples += u.NSamples
	}

	// Weighted average.
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
func (f *FedAvg) SelectClients(_ int, available []ClientID) []ClientID {
	result := make([]ClientID, len(available))
	copy(result, available)
	return result
}
