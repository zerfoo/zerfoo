package features

import (
	"math"
	"math/cmplx"
	"sort"

	"github.com/zerfoo/zerfoo/data"
	"gonum.org/v1/gonum/dsp/fourier"
)

// Transformer is the interface for feature transformers.
type Transformer interface {
	Transform(dataset *data.Dataset) error
}

// LaggedTransformer adds lagged features to the dataset.
type LaggedTransformer struct {
	lags []int
}

// NewLaggedTransformer creates a new lagged features transformer.
func NewLaggedTransformer(lags []int) *LaggedTransformer {
	return &LaggedTransformer{
		lags: lags,
	}
}

// Transform adds lagged features to the dataset.
// It assumes the Eras in the dataset are sorted chronologically.
func (t *LaggedTransformer) Transform(dataset *data.Dataset) error {
	if len(dataset.Eras) == 0 || len(dataset.Eras[0].Stocks) == 0 {
		return nil
	}

	// Create a map for quick lookup of stock data by era and stock ID
	stockHistory := make(map[int]map[string][]float64)
	for _, era := range dataset.Eras {
		stockHistory[era.Era] = make(map[string][]float64)
		for _, stock := range era.Stocks {
			// Store a copy of the original features
			originalFeatures := make([]float64, len(stock.Features))
			copy(originalFeatures, stock.Features)
			stockHistory[era.Era][stock.ID] = originalFeatures
		}
	}

	// Sort eras to ensure chronological processing
	sort.Slice(dataset.Eras, func(i, j int) bool {
		return dataset.Eras[i].Era < dataset.Eras[j].Era
	})

	for eraIdx := range dataset.Eras {
		era := &dataset.Eras[eraIdx]
		numOriginalFeatures := 0
		if len(era.Stocks) > 0 {
			numOriginalFeatures = len(stockHistory[era.Era][era.Stocks[0].ID])
		}

		for stockIdx := range era.Stocks {
			stock := &era.Stocks[stockIdx]

			newFeatures := make([]float64, numOriginalFeatures)
			copy(newFeatures, stock.Features)

			for _, lag := range t.lags {
				prevEra := era.Era - lag
				if prevFeatures, ok := stockHistory[prevEra][stock.ID]; ok {
					newFeatures = append(newFeatures, prevFeatures...)
				} else {
					// If past data is missing, append zeros.
					newFeatures = append(newFeatures, make([]float64, numOriginalFeatures)...)
				}
			}
			stock.Features = newFeatures
		}
	}

	return nil
}

// RollingTransformer adds rolling statistics to the dataset.
type RollingTransformer struct {
	window int
}

// NewRollingTransformer creates a new rolling statistics transformer.
func NewRollingTransformer(window int) *RollingTransformer {
	return &RollingTransformer{
		window: window,
	}
}

// Transform adds rolling statistics to the dataset.
// It calculates the moving average and standard deviation over a given window.
func (t *RollingTransformer) Transform(dataset *data.Dataset) error {
	if len(dataset.Eras) == 0 || len(dataset.Eras[0].Stocks) == 0 {
		return nil
	}

	// Create a map for quick lookup of stock data by era and stock ID
	stockHistory := make(map[int]map[string][]float64)
	for _, era := range dataset.Eras {
		stockHistory[era.Era] = make(map[string][]float64)
		for _, stock := range era.Stocks {
			stockHistory[era.Era][stock.ID] = stock.Features
		}
	}

	// Sort eras to ensure chronological processing
	sort.Slice(dataset.Eras, func(i, j int) bool {
		return dataset.Eras[i].Era < dataset.Eras[j].Era
	})

	for eraIdx := range dataset.Eras {
		era := &dataset.Eras[eraIdx]
		numOriginalFeatures := 0
		if len(era.Stocks) > 0 {
			numOriginalFeatures = len(stockHistory[era.Era][era.Stocks[0].ID])
		}

		for stockIdx := range era.Stocks {
			stock := &era.Stocks[stockIdx]

			newFeatures := make([]float64, len(stock.Features))
			copy(newFeatures, stock.Features)

			for featureIdx := 0; featureIdx < numOriginalFeatures; featureIdx++ {
				values := []float64{}
				for i := 0; i < t.window; i++ {
					prevEra := era.Era - i
					if prevFeatures, ok := stockHistory[prevEra][stock.ID]; ok && len(prevFeatures) > featureIdx {
						values = append(values, prevFeatures[featureIdx])
					}
				}

				if len(values) > 0 {
					// Calculate moving average
					sum := 0.0
					for _, v := range values {
						sum += v
					}
					mean := sum / float64(len(values))
					newFeatures = append(newFeatures, mean)

					// Calculate standard deviation
					sd := 0.0
					if len(values) > 1 {
						for _, v := range values {
							sd += (v - mean) * (v - mean)
						}
						sd = math.Sqrt(sd / float64(len(values)-1))
					}
					newFeatures = append(newFeatures, sd)
				} else {
					// Pad with zeros if no historical data is available
					newFeatures = append(newFeatures, 0.0, 0.0)
				}
			}
			stock.Features = newFeatures
		}
	}

	return nil
}

// FFTTransformer adds FFT-based features to the dataset.
type FFTTransformer struct {
	window int
	k      int // Number of top frequencies to extract
}

// NewFFTTransformer creates a new FFT features transformer.
func NewFFTTransformer(window, k int) *FFTTransformer {
	return &FFTTransformer{
		window: window,
		k:      k,
	}
}

// Transform adds FFT-based features to the dataset.
func (t *FFTTransformer) Transform(dataset *data.Dataset) error {
	if len(dataset.Eras) == 0 || len(dataset.Eras[0].Stocks) == 0 {
		return nil
	}

	// Create a map for quick lookup of stock data by era and stock ID
	stockHistory := make(map[int]map[string][]float64)
	for _, era := range dataset.Eras {
		stockHistory[era.Era] = make(map[string][]float64)
		for _, stock := range era.Stocks {
			stockHistory[era.Era][stock.ID] = stock.Features
		}
	}

	// Sort eras to ensure chronological processing
	sort.Slice(dataset.Eras, func(i, j int) bool {
		return dataset.Eras[i].Era < dataset.Eras[j].Era
	})

	for eraIdx := range dataset.Eras {
		era := &dataset.Eras[eraIdx]
		numOriginalFeatures := 0
		if len(era.Stocks) > 0 {
			numOriginalFeatures = len(stockHistory[era.Era][era.Stocks[0].ID])
		}

		for stockIdx := range era.Stocks {
			stock := &era.Stocks[stockIdx]

			newFeatures := make([]float64, len(stock.Features))
			copy(newFeatures, stock.Features)

			for featureIdx := 0; featureIdx < numOriginalFeatures; featureIdx++ {
				series := make([]float64, 0, t.window)
				for i := 0; i < t.window; i++ {
					prevEra := era.Era - i
					if prevFeatures, ok := stockHistory[prevEra][stock.ID]; ok && len(prevFeatures) > featureIdx {
						series = append(series, prevFeatures[featureIdx])
					}
				}

				if len(series) > 1 {
					fft := fourier.NewFFT(len(series))
					coeffs := fft.Coefficients(nil, series)

					// Get top k frequencies
					for i := 1; i <= t.k && i < len(coeffs); i++ {
						newFeatures = append(newFeatures, cmplx.Abs(coeffs[i]))
					}
					// Pad with zeros if not enough frequencies
					for i := len(coeffs); i <= t.k; i++ {
						newFeatures = append(newFeatures, 0.0)
					}
				} else {
					// Pad with zeros if not enough data for FFT
					for i := 0; i < t.k; i++ {
						newFeatures = append(newFeatures, 0.0)
					}
				}
			}
			stock.Features = newFeatures
		}
	}

	return nil
}
