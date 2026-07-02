package timeseries

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TimeMixerOption configures a TimeMixer model.
type TimeMixerOption func(*TimeMixer)

// WithTimeMixerEngine sets the compute engine for GPU-accelerated forward pass.
// When nil (the default), TimeMixer uses the pure-Go CPU path.
func WithTimeMixerEngine(engine compute.Engine[float32], ops numeric.Arithmetic[float32]) TimeMixerOption {
	return func(m *TimeMixer) {
		m.engine = engine
		m.ops = ops
	}
}

// WithTimeMixerRNG sets a deterministic RNG used for weight initialization
// in NewTimeMixer. Without this option NewTimeMixer draws weights from the
// global math/rand/v2 generator, which is runtime-seeded and therefore
// non-deterministic across test runs. Pass a seeded *rand.Rand to make
// model construction reproducible (e.g. for gradient checks).
func WithTimeMixerRNG(r *rand.Rand) TimeMixerOption {
	return func(m *TimeMixer) {
		m.initRNG = r
	}
}

// ForwardEngine runs the engine-accelerated forward pass, producing the same
// MultiScaleOutput as Forward. The weighted moving average at each scale is
// computed via engine.MatMul by constructing a causal convolution (Toeplitz)
// matrix from the learnable kernel weights. Falls back to the CPU path on
// any engine error.
func (m *TimeMixer) ForwardEngine(ctx context.Context, input [][]float64) (*TimeMixerOutput, error) {
	if m.engine == nil {
		fwdOut, fwdErr := m.Forward(input)
		return fwdOut, fwdErr
	}

	if len(input) == 0 {
		return nil, fmt.Errorf("timemixer: empty input")
	}
	if len(input) != m.config.NumFeatures {
		return nil, fmt.Errorf("timemixer: expected %d features, got %d", m.config.NumFeatures, len(input))
	}
	for f, ch := range input {
		if len(ch) != m.config.InputLen {
			return nil, fmt.Errorf("timemixer: feature %d has length %d, expected %d", f, len(ch), m.config.InputLen)
		}
	}

	nf := m.config.NumFeatures
	inputLen := m.config.InputLen

	// Pack all features into a single [numFeatures x inputLen] float32 tensor.
	xFlat := make([]float32, nf*inputLen)
	for f := 0; f < nf; f++ {
		for i := 0; i < inputLen; i++ {
			xFlat[f*inputLen+i] = float32(input[f][i])
		}
	}
	xT, err := tensor.New[float32]([]int{nf, inputLen}, xFlat)
	if err != nil {
		fwdOut, fwdErr := m.Forward(input)
		return fwdOut, fwdErr
	}

	scales := make([]scaleDecomposition, m.config.NumScales)

	for s := 0; s < m.config.NumScales; s++ {
		kernel := m.maWeights[s]

		// Build causal convolution (Toeplitz) matrix [inputLen x inputLen].
		// Row i, col j: kernel[i-j] if 0 <= i-j < kernelSize, else edge-padded.
		// Edge padding: for j where i-j < 0, use kernel[-(i-j)] contribution
		// via the idx=0 clamping in weightedMovingAverage.
		//
		// The CPU path computes: out[i] = sum_{j=0}^{k-1} kernel[j] * x[max(0, i-j)]
		// This is equivalent to: out = toeplitz @ x^T  where toeplitz[i][col] is the
		// sum of kernel weights that map to x[col] for output position i.
		toepFlat := make([]float32, inputLen*inputLen)
		k := len(kernel)
		for i := 0; i < inputLen; i++ {
			for j := 0; j < k; j++ {
				col := i - j
				if col < 0 {
					col = 0 // edge padding
				}
				toepFlat[i*inputLen+col] += float32(kernel[j])
			}
		}

		toepT, err := tensor.New[float32]([]int{inputLen, inputLen}, toepFlat)
		if err != nil {
			fwdOut, fwdErr := m.Forward(input)
			return fwdOut, fwdErr
		}

		// Transpose x to [inputLen x numFeatures] so we can do toeplitz @ xT
		// yielding [inputLen x numFeatures], then transpose back.
		// Alternatively, compute x @ toeplitz^T = [nf x inputLen] @ [inputLen x inputLen].
		// Since toeplitz operates on columns, we need trend = (toeplitz @ x^T)^T = x @ toeplitz^T.
		// But toeplitz is not symmetric, so we need toeplitz^T.
		// Actually: trend[f][i] = sum_col toeplitz[i][col] * x[f][col]
		//   = row i of (toeplitz @ x[f]^T)
		// For all features at once: trend = x @ toeplitz^T where trend is [nf x inputLen].
		//
		// Transpose toeplitz: toepT_trans[col][i] = toepT[i][col]
		toepTransFlat := make([]float32, inputLen*inputLen)
		for i := 0; i < inputLen; i++ {
			for j := 0; j < inputLen; j++ {
				toepTransFlat[j*inputLen+i] = toepFlat[i*inputLen+j]
			}
		}
		toepTransT, err := tensor.New[float32]([]int{inputLen, inputLen}, toepTransFlat)
		if err != nil {
			fwdOut, fwdErr := m.Forward(input)
			return fwdOut, fwdErr
		}
		_ = toepT

		// trend = x @ toeplitz^T : [nf x inputLen] @ [inputLen x inputLen] = [nf x inputLen]
		trendT, err := m.engine.MatMul(ctx, xT, toepTransT)
		if err != nil {
			fwdOut, fwdErr := m.Forward(input)
			return fwdOut, fwdErr
		}

		// seasonal = x - trend (via engine.Sub)
		seasonalT, err := m.engine.Sub(ctx, xT, trendT)
		if err != nil {
			fwdOut, fwdErr := m.Forward(input)
			return fwdOut, fwdErr
		}

		// Unpack trend and seasonal back to [][]float64.
		trendData := trendT.Data()
		seasonalData := seasonalT.Data()

		scales[s] = scaleDecomposition{
			trend:    make([][]float64, nf),
			seasonal: make([][]float64, nf),
		}
		for f := 0; f < nf; f++ {
			scales[s].trend[f] = make([]float64, inputLen)
			scales[s].seasonal[f] = make([]float64, inputLen)
			off := f * inputLen
			for i := 0; i < inputLen; i++ {
				scales[s].trend[f][i] = float64(trendData[off+i])
				scales[s].seasonal[f][i] = float64(seasonalData[off+i])
			}
		}
	}

	mixed := m.pastDecomposableMixing(scales)

	// Compute softmax mixing weights.
	smWeights := make([]float64, len(m.mixWeights))
	copy(smWeights, m.mixWeights)
	normalizeWeights(smWeights)

	// Project each scale and combine forecasts.
	outLen := m.config.OutputLen
	forecast := make([][]float64, nf)
	for f := 0; f < nf; f++ {
		forecast[f] = make([]float64, outLen)
		for s := 0; s < len(mixed); s++ {
			trendProj := linearProject(mixed[s].trend[f], m.trendHeads[s], outLen)
			seasonProj := linearProject(mixed[s].seasonal[f], m.seasonalHeads[s], outLen)
			for j := 0; j < outLen; j++ {
				forecast[f][j] += smWeights[s] * (trendProj[j] + seasonProj[j])
			}
		}
	}

	return &TimeMixerOutput{
		Forecast:         forecast,
		MultiScaleOutput: MultiScaleOutput{Scales: mixed},
	}, nil
}
