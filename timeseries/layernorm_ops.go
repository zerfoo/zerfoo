package timeseries

import (
	"context"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/functional"
)

// cpuEngine64 is a package-level CPU engine for float64 layer norm operations.
var cpuEngine64 = compute.NewCPUEngine[float64](numeric.Float64Ops{})

// layerNormF64 applies layer normalization in float64 via functional.LayerNorm.
// x: [seq][d], scale/bias: [d]. Returns normalized output [seq][d].
func layerNormF64(x [][]float64, scale, bias []float64, d int) [][]float64 {
	seq := len(x)
	flat := make([]float64, seq*d)
	for s := 0; s < seq; s++ {
		copy(flat[s*d:], x[s])
	}

	xT, _ := tensor.New[float64]([]int{seq, d}, flat)
	sT, _ := tensor.New[float64]([]int{1, d}, scale)
	bT, _ := tensor.New[float64]([]int{1, d}, bias)

	ctx := context.Background()
	out, err := functional.LayerNorm(ctx, cpuEngine64, xT, sT, bT, 1e-5)
	if err != nil {
		panic("layerNormF64: " + err.Error())
	}

	result := make([][]float64, seq)
	data := out.Data()
	for s := 0; s < seq; s++ {
		result[s] = make([]float64, d)
		copy(result[s], data[s*d:(s+1)*d])
	}
	return result
}

// layerNormF64WithCache applies layer normalization via functional.LayerNorm
// and returns cached intermediates for backward: means, invStds, and centered values.
func layerNormF64WithCache(x [][]float64, scale, bias []float64, d int) (normed [][]float64, means []float64, invStds []float64, centered [][]float64) {
	seq := len(x)
	ctx := context.Background()

	flat := make([]float64, seq*d)
	for s := 0; s < seq; s++ {
		copy(flat[s*d:], x[s])
	}
	xT, _ := tensor.New[float64]([]int{seq, d}, flat)
	sT, _ := tensor.New[float64]([]int{1, d}, scale)
	bT, _ := tensor.New[float64]([]int{1, d}, bias)

	out, err := functional.LayerNorm(ctx, cpuEngine64, xT, sT, bT, 1e-5)
	if err != nil {
		panic("layerNormF64WithCache: " + err.Error())
	}

	normed = make([][]float64, seq)
	data := out.Data()
	for s := 0; s < seq; s++ {
		normed[s] = make([]float64, d)
		copy(normed[s], data[s*d:(s+1)*d])
	}

	// Compute and return cached intermediates for backward.
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
		invStds[s] = 1.0 / math.Sqrt(variance+1e-5)
	}
	return
}

// layerNormBackwardF64 computes the backward pass through 2D layer normalization
// using engine operations.
// dOut: [seq][d], centered: [seq][d], invStd: [seq], scale: [d].
// Accumulates into dScale and dBias. Returns dInput: [seq][d].
func layerNormBackwardF64(dOut [][]float64, centered [][]float64, invStd []float64, scale []float64, dScale, dBias []float64, d int) [][]float64 {
	seq := len(dOut)
	ctx := context.Background()

	// Flatten inputs into tensors.
	dOutFlat := make([]float64, seq*d)
	centFlat := make([]float64, seq*d)
	for s := 0; s < seq; s++ {
		copy(dOutFlat[s*d:], dOut[s])
		copy(centFlat[s*d:], centered[s])
	}
	dOutT, _ := tensor.New[float64]([]int{seq, d}, dOutFlat)
	centT, _ := tensor.New[float64]([]int{seq, d}, centFlat)
	scaleT, _ := tensor.New[float64]([]int{1, d}, scale)

	// Build invStd as [seq, 1] for broadcasting.
	invStdT, _ := tensor.New[float64]([]int{seq, 1}, invStd)

	// xhat = centered * invStd  (broadcast [seq,1] over [seq,d])
	xhat, _ := cpuEngine64.Mul(ctx, centT, invStdT, nil)

	// dNormed = dOut * scale
	dNormed, _ := cpuEngine64.Mul(ctx, dOutT, scaleT, nil)

	// Accumulate dScale and dBias.
	// dScale[j] += sum_s(dOut[s][j] * xhat[s][j])
	// dBias[j] += sum_s(dOut[s][j])
	dOutData := dOutT.Data()
	xhatData := xhat.Data()
	for s := 0; s < seq; s++ {
		for j := 0; j < d; j++ {
			idx := s*d + j
			dScale[j] += dOutData[idx] * xhatData[idx]
			dBias[j] += dOutData[idx]
		}
	}

	// meanDN = ReduceSum(dNormed, axis=1, keepDims=true) / d
	sumDN, _ := cpuEngine64.ReduceSum(ctx, dNormed, 1, true)
	invD := 1.0 / float64(d)
	meanDN, _ := cpuEngine64.MulScalar(ctx, sumDN, invD)

	// meanDNxhat = ReduceSum(dNormed * xhat, axis=1, keepDims=true) / d
	dnXhat, _ := cpuEngine64.Mul(ctx, dNormed, xhat, nil)
	sumDNxhat, _ := cpuEngine64.ReduceSum(ctx, dnXhat, 1, true)
	meanDNxhat, _ := cpuEngine64.MulScalar(ctx, sumDNxhat, invD)

	// dInput = invStd * (dNormed - meanDN - xhat * meanDNxhat)
	sub1, _ := cpuEngine64.Sub(ctx, dNormed, meanDN, nil)
	xhatMean, _ := cpuEngine64.Mul(ctx, xhat, meanDNxhat, nil)
	sub2, _ := cpuEngine64.Sub(ctx, sub1, xhatMean, nil)
	dInputT, _ := cpuEngine64.Mul(ctx, invStdT, sub2, nil)

	// Extract result.
	dInput := make([][]float64, seq)
	dData := dInputT.Data()
	for s := 0; s < seq; s++ {
		dInput[s] = make([]float64, d)
		copy(dInput[s], dData[s*d:(s+1)*d])
	}
	return dInput
}

// layerNorm1D applies layer normalization to a single 1D float64 vector
// via functional.LayerNorm.
// x, scale, bias: [n]. Returns normalized output [n].
func layerNorm1D(x, scale, bias []float64) []float64 {
	n := len(x)
	xT, _ := tensor.New[float64]([]int{1, n}, x)
	sT, _ := tensor.New[float64]([]int{1, n}, scale)
	bT, _ := tensor.New[float64]([]int{1, n}, bias)

	ctx := context.Background()
	out, err := functional.LayerNorm(ctx, cpuEngine64, xT, sT, bT, 1e-5)
	if err != nil {
		panic("layerNorm1D: " + err.Error())
	}

	result := make([]float64, n)
	copy(result, out.Data())
	return result
}

// layerNorm1DCached computes 1D layer norm via functional.LayerNorm and
// returns output, mean, and std.
func layerNorm1DCached(x, scale, bias []float64) ([]float64, float64, float64) {
	n := len(x)

	// Compute mean and std for backward-pass callers.
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

	// Use functional.LayerNorm for the actual normalization.
	xT, _ := tensor.New[float64]([]int{1, n}, x)
	sT, _ := tensor.New[float64]([]int{1, n}, scale)
	bT, _ := tensor.New[float64]([]int{1, n}, bias)

	ctx := context.Background()
	out, err := functional.LayerNorm(ctx, cpuEngine64, xT, sT, bT, 1e-5)
	if err != nil {
		panic("layerNorm1DCached: " + err.Error())
	}

	result := make([]float64, n)
	copy(result, out.Data())
	return result, mean, std
}
