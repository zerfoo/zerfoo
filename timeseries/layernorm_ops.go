package timeseries

import "math"

// layerNormF64 applies layer normalization in float64.
// x: [seq][d], scale/bias: [d]. Returns normalized output [seq][d].
func layerNormF64(x [][]float64, scale, bias []float64, d int) [][]float64 {
	seq := len(x)
	out := make([][]float64, seq)
	for s := 0; s < seq; s++ {
		mean := 0.0
		for j := 0; j < d; j++ {
			mean += x[s][j]
		}
		mean /= float64(d)

		variance := 0.0
		for j := 0; j < d; j++ {
			diff := x[s][j] - mean
			variance += diff * diff
		}
		variance /= float64(d)

		invStd := 1.0 / math.Sqrt(variance+1e-5)
		out[s] = make([]float64, d)
		for j := 0; j < d; j++ {
			out[s][j] = (x[s][j]-mean)*invStd*scale[j] + bias[j]
		}
	}
	return out
}

// layerNormF64WithCache applies layer normalization and returns cached
// intermediates for backward: means, invStds, and centered values.
func layerNormF64WithCache(x [][]float64, scale, bias []float64, d int) (normed [][]float64, means []float64, invStds []float64, centered [][]float64) {
	seq := len(x)
	normed = make([][]float64, seq)
	means = make([]float64, seq)
	invStds = make([]float64, seq)
	centered = make([][]float64, seq)

	for s := 0; s < seq; s++ {
		mean := 0.0
		for j := 0; j < d; j++ {
			mean += x[s][j]
		}
		mean /= float64(d)
		means[s] = mean

		variance := 0.0
		centered[s] = make([]float64, d)
		for j := 0; j < d; j++ {
			centered[s][j] = x[s][j] - mean
			variance += centered[s][j] * centered[s][j]
		}
		variance /= float64(d)

		invStd := 1.0 / math.Sqrt(variance+1e-5)
		invStds[s] = invStd

		normed[s] = make([]float64, d)
		for j := 0; j < d; j++ {
			normed[s][j] = centered[s][j]*invStd*scale[j] + bias[j]
		}
	}
	return
}

// layerNormBackwardF64 computes the backward pass through 2D layer normalization.
// dOut: [seq][d], centered: [seq][d], invStd: [seq], scale: [d].
// Accumulates into dScale and dBias. Returns dInput: [seq][d].
func layerNormBackwardF64(dOut [][]float64, centered [][]float64, invStd []float64, scale []float64, dScale, dBias []float64, d int) [][]float64 {
	seq := len(dOut)
	dInput := make([][]float64, seq)
	df := float64(d)

	for s := 0; s < seq; s++ {
		dNormed := make([]float64, d)
		for j := 0; j < d; j++ {
			dNormed[j] = dOut[s][j] * scale[j]
			dScale[j] += dOut[s][j] * centered[s][j] * invStd[s]
			dBias[j] += dOut[s][j]
		}

		meanDN := 0.0
		meanDNxhat := 0.0
		for j := 0; j < d; j++ {
			xhat := centered[s][j] * invStd[s]
			meanDN += dNormed[j]
			meanDNxhat += dNormed[j] * xhat
		}
		meanDN /= df
		meanDNxhat /= df

		dInput[s] = make([]float64, d)
		for j := 0; j < d; j++ {
			xhat := centered[s][j] * invStd[s]
			dInput[s][j] = invStd[s] * (dNormed[j] - meanDN - xhat*meanDNxhat)
		}
	}
	return dInput
}

// layerNorm1D applies layer normalization to a single 1D float64 vector.
// x, scale, bias: [n]. Returns normalized output [n].
func layerNorm1D(x, scale, bias []float64) []float64 {
	n := len(x)
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(n)

	variance := 0.0
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float64(n)
	std := math.Sqrt(variance + 1e-5)

	out := make([]float64, n)
	for i := range x {
		out[i] = scale[i]*(x[i]-mean)/std + bias[i]
	}
	return out
}

// layerNorm1DCached computes 1D layer norm and returns output, mean, and std.
func layerNorm1DCached(x, scale, bias []float64) ([]float64, float64, float64) {
	n := len(x)
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(n)

	variance := 0.0
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float64(n)
	std := math.Sqrt(variance + 1e-5)

	out := make([]float64, n)
	for i := range x {
		out[i] = scale[i]*(x[i]-mean)/std + bias[i]
	}
	return out, mean, std
}
