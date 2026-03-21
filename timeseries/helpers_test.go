package timeseries

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func newTestEngine() (compute.Engine[float32], numeric.Arithmetic[float32]) {
	ops := numeric.Float32Ops{}
	return compute.NewCPUEngine[float32](ops), ops
}

// makeMultiScaleWindows creates synthetic training data with features spanning
// 10 orders of magnitude, simulating the conditions in issue #121.
// Returns windows [nSamples][nChannels][inputLen] and labels [nSamples * outputDim].
func makeMultiScaleWindows(nSamples, nChannels, inputLen, outputDim int) ([][][]float64, []float64) {
	// Scales from 1e-4 to 1e6 (10 orders of magnitude).
	scales := make([]float64, nChannels)
	for c := 0; c < nChannels; c++ {
		scales[c] = math.Pow(10, -4+10*float64(c)/float64(nChannels-1))
	}

	rng := rand.New(rand.NewPCG(42, 0))
	windows := make([][][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, nChannels)
		for c := 0; c < nChannels; c++ {
			windows[i][c] = make([]float64, inputLen)
			for t := 0; t < inputLen; t++ {
				windows[i][c][t] = scales[c] * (1.0 + 0.1*rng.Float64())
			}
		}
	}

	labels := make([]float64, nSamples*outputDim)
	for i := range labels {
		labels[i] = 0.01 * rng.Float64()
	}
	return windows, labels
}

// assertFiniteWeights checks that all model weights in params are finite.
func assertFiniteWeights(t *testing.T, params []*float64) {
	t.Helper()
	for i, p := range params {
		if !isFinite(*p) {
			t.Errorf("param[%d] = %v, want finite", i, *p)
		}
	}
}
