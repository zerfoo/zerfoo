package data

import (
	"math"
)

// Sample represents a single data point with an identifier, feature vector, and target value.
type Sample struct {
	ID       string
	Features []float64
	Target   float64
}

// Batch represents a group of samples collected at the same time index.
type Batch struct {
	Index   int
	Samples []Sample
	Stats   []float64 // Precomputed aggregate stats for this batch (mean, var, etc.)
}

// Dataset represents a collection of time-ordered batches of samples.
type Dataset struct {
	Batches []Batch
}

// NormalizeFeatures applies z-score normalization to the features of the dataset.
func (d *Dataset) NormalizeFeatures() {
	if len(d.Batches) == 0 || len(d.Batches[0].Samples) == 0 {
		return
	}

	numFeatures := len(d.Batches[0].Samples[0].Features)
	means := make([]float64, numFeatures)
	stdDevs := make([]float64, numFeatures)
	counts := make([]float64, numFeatures)

	// Calculate mean
	for _, batch := range d.Batches {
		for _, sample := range batch.Samples {
			for i, feature := range sample.Features {
				if i < numFeatures {
					means[i] += feature
					counts[i]++
				}
			}
		}
	}

	for i := range means {
		if counts[i] > 0 {
			means[i] /= counts[i]
		}
	}

	// Calculate standard deviation
	for _, batch := range d.Batches {
		for _, sample := range batch.Samples {
			for i, feature := range sample.Features {
				if i < numFeatures {
					stdDevs[i] += (feature - means[i]) * (feature - means[i])
				}
			}
		}
	}

	for i := range stdDevs {
		if counts[i] > 1 {
			stdDevs[i] = math.Sqrt(stdDevs[i] / (counts[i] - 1))
		} else {
			stdDevs[i] = 0
		}
	}

	// Apply z-score normalization
	for i := range d.Batches {
		for j := range d.Batches[i].Samples {
			for k, feature := range d.Batches[i].Samples[j].Features {
				if k < numFeatures {
					if stdDevs[k] > 0 {
						d.Batches[i].Samples[j].Features[k] = (feature - means[k]) / stdDevs[k]
					} else {
						d.Batches[i].Samples[j].Features[k] = 0
					}
				}
			}
		}
	}
}
