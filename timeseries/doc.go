// Package timeseries provides time-series forecasting models built on ztensor.
//
// Models in this package accept static covariates and temporal features, producing
// multi-horizon forecasts with optional quantile estimates.
package timeseries

import (
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/tensor"
)

// Forecast represents a multi-horizon forecast with optional quantile estimates.
type Forecast struct {
	Horizons  []float64             // point forecasts per horizon
	Quantiles map[float64][]float64 // quantile -> values per horizon (nil if no quantiles)
}

// linearLayer holds weights and biases for a single linear transformation.
type linearLayer struct {
	weights *tensor.TensorNumeric[float32]
	biases  *tensor.TensorNumeric[float32]
}

// newLinearLayer creates a linear layer with He (Kaiming) initialization.
func newLinearLayer(in, out int) (linearLayer, error) {
	scale := float32(math.Sqrt(2.0 / float64(in)))
	wData := make([]float32, in*out)
	for i := range wData {
		wData[i] = float32(rand.NormFloat64()) * scale
	}
	w, err := tensor.New[float32]([]int{in, out}, wData)
	if err != nil {
		return linearLayer{}, err
	}

	bData := make([]float32, out)
	b, err := tensor.New[float32]([]int{1, out}, bData)
	if err != nil {
		return linearLayer{}, err
	}

	return linearLayer{weights: w, biases: b}, nil
}

// newLinearXavier creates a linear layer with Xavier initialization.
func newLinearXavier(in, out int) (linearLayer, error) {
	scale := float32(math.Sqrt(2.0 / float64(in+out)))
	wData := make([]float32, in*out)
	for i := range wData {
		wData[i] = float32(rand.NormFloat64()) * scale
	}
	w, err := tensor.New[float32]([]int{in, out}, wData)
	if err != nil {
		return linearLayer{}, err
	}

	bData := make([]float32, out)
	b, err := tensor.New[float32]([]int{1, out}, bData)
	if err != nil {
		return linearLayer{}, err
	}

	return linearLayer{weights: w, biases: b}, nil
}
