package data

import (
	"math"
)

// StockData represents a single stock's data for a given era.
type StockData struct {
	ID       string
	Features []float64
	Target   float64 // Target value (if known; training data has it, tournament data will not)
}

// EraData represents all the data for a single era.
type EraData struct {
	Era      int
	Stocks   []StockData
	EraStats []float64 // e.g. precomputed aggregate stats for the era (mean, var, etc.)
}

// Dataset represents the entire dataset, composed of multiple eras.
type Dataset struct {
	Eras []EraData
}

// NormalizeFeatures applies z-score normalization to the features of the dataset.
func (d *Dataset) NormalizeFeatures() {
	if len(d.Eras) == 0 || len(d.Eras[0].Stocks) == 0 {
		return
	}

	numFeatures := len(d.Eras[0].Stocks[0].Features)
	means := make([]float64, numFeatures)
	stdDevs := make([]float64, numFeatures)
	counts := make([]float64, numFeatures)

	// Calculate mean
	for _, era := range d.Eras {
		for _, stock := range era.Stocks {
			for i, feature := range stock.Features {
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
	for _, era := range d.Eras {
		for _, stock := range era.Stocks {
			for i, feature := range stock.Features {
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
	for i := range d.Eras {
		for j := range d.Eras[i].Stocks {
			for k, feature := range d.Eras[i].Stocks[j].Features {
				if k < numFeatures {
					if stdDevs[k] > 0 {
						d.Eras[i].Stocks[j].Features[k] = (feature - means[k]) / stdDevs[k]
					} else {
						d.Eras[i].Stocks[j].Features[k] = 0
					}
				}
			}
		}
	}
}