package timeseries

import (
	"encoding/json"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"sort"
)

// FreTSConfig holds the configuration for a FreTS model.
type FreTSConfig struct {
	Channels   int // number of input channels/features
	InputLen   int // input sequence length (lookback window)
	OutputLen  int // forecast horizon
	TopK       int // number of frequency components to keep
	HiddenSize int // hidden size for channel/temporal mixing MLPs
}

// FreTS implements the Frequency-enhanced Time Series forecasting model (ICML 2023).
// FreTS uses discrete Fourier transform for channel and temporal mixing:
//  1. Real FFT on input -> select top-K frequency components
//  2. Channel mixing MLP (mix across channels in frequency domain)
//  3. Temporal mixing MLP (mix across time in frequency domain)
//  4. Inverse FFT -> linear projection to outputLen
type FreTS struct {
	config FreTSConfig

	// Channel mixing MLP weights (operates on real and imaginary parts separately).
	// Maps channels -> hiddenSize -> channels in frequency domain.
	chanW1 []float64 // [channels * hiddenSize]
	chanB1 []float64 // [hiddenSize]
	chanW2 []float64 // [hiddenSize * channels]
	chanB2 []float64 // [channels]

	// Temporal mixing MLP weights.
	// Maps topK -> hiddenSize -> topK in frequency domain.
	tempW1 []float64 // [topK * hiddenSize]
	tempB1 []float64 // [hiddenSize]
	tempW2 []float64 // [hiddenSize * topK]
	tempB2 []float64 // [topK]

	// Output projection: inputLen -> outputLen per channel.
	outW []float64 // [channels * outputLen * inputLen]
	outB []float64 // [channels * outputLen]

	normMeans [][]float64 // per-channel normalization means from training
	normStds  [][]float64 // per-channel normalization stds from training
}

// NewFreTS creates a new FreTS model with the given configuration.
func NewFreTS(config FreTSConfig) (*FreTS, error) {
	if config.Channels <= 0 {
		return nil, fmt.Errorf("frets: Channels must be positive, got %d", config.Channels)
	}
	if config.InputLen <= 0 {
		return nil, fmt.Errorf("frets: InputLen must be positive, got %d", config.InputLen)
	}
	if config.OutputLen <= 0 {
		return nil, fmt.Errorf("frets: OutputLen must be positive, got %d", config.OutputLen)
	}
	if config.TopK <= 0 {
		return nil, fmt.Errorf("frets: TopK must be positive, got %d", config.TopK)
	}
	if config.HiddenSize <= 0 {
		return nil, fmt.Errorf("frets: HiddenSize must be positive, got %d", config.HiddenSize)
	}
	maxFreqs := config.InputLen/2 + 1
	if config.TopK > maxFreqs {
		return nil, fmt.Errorf("frets: TopK=%d exceeds max frequency bins=%d for InputLen=%d", config.TopK, maxFreqs, config.InputLen)
	}

	f := &FreTS{config: config}

	// Xavier initialization.
	chanScale1 := math.Sqrt(2.0 / float64(config.Channels+config.HiddenSize))
	f.chanW1 = randNormSlice(config.Channels*config.HiddenSize, chanScale1)
	f.chanB1 = make([]float64, config.HiddenSize)
	chanScale2 := math.Sqrt(2.0 / float64(config.HiddenSize+config.Channels))
	f.chanW2 = randNormSlice(config.HiddenSize*config.Channels, chanScale2)
	f.chanB2 = make([]float64, config.Channels)

	tempScale1 := math.Sqrt(2.0 / float64(config.TopK+config.HiddenSize))
	f.tempW1 = randNormSlice(config.TopK*config.HiddenSize, tempScale1)
	f.tempB1 = make([]float64, config.HiddenSize)
	tempScale2 := math.Sqrt(2.0 / float64(config.HiddenSize+config.TopK))
	f.tempW2 = randNormSlice(config.HiddenSize*config.TopK, tempScale2)
	f.tempB2 = make([]float64, config.TopK)

	outScale := math.Sqrt(2.0 / float64(config.InputLen+config.OutputLen))
	f.outW = randNormSlice(config.Channels*config.OutputLen*config.InputLen, outScale)
	f.outB = make([]float64, config.Channels*config.OutputLen)

	return f, nil
}

// randNormSlice returns a slice of n values drawn from N(0, scale).
func randNormSlice(n int, scale float64) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = rand.NormFloat64() * scale
	}
	return s
}

// dft computes the discrete Fourier transform of a real signal.
// Returns complex coefficients of length n/2+1 (positive frequencies only).
func dft(x []float64) []complex128 {
	n := len(x)
	nFreqs := n/2 + 1
	out := make([]complex128, nFreqs)
	for k := 0; k < nFreqs; k++ {
		var sum complex128
		for t := 0; t < n; t++ {
			angle := -2 * math.Pi * float64(k) * float64(t) / float64(n)
			sum += complex(x[t]*math.Cos(angle), x[t]*math.Sin(angle))
		}
		out[k] = sum
	}
	return out
}

// idft computes the inverse DFT from positive-frequency coefficients back to a real signal of length n.
func idft(coeffs []complex128, n int) []float64 {
	out := make([]float64, n)
	nFreqs := len(coeffs)
	for t := 0; t < n; t++ {
		var sum complex128
		for k := 0; k < nFreqs; k++ {
			angle := 2 * math.Pi * float64(k) * float64(t) / float64(n)
			c := complex(math.Cos(angle), math.Sin(angle))
			if k == 0 || (n%2 == 0 && k == n/2) {
				sum += coeffs[k] * c
			} else {
				sum += coeffs[k]*c + cmplx.Conj(coeffs[k])*cmplx.Conj(c)
			}
		}
		out[t] = real(sum) / float64(n)
	}
	return out
}

// topKIndices returns the indices of the top-K frequency components by magnitude.
func topKIndices(coeffs []complex128, k int) []int {
	type freqMag struct {
		idx int
		mag float64
	}
	mags := make([]freqMag, len(coeffs))
	for i, c := range coeffs {
		mags[i] = freqMag{i, cmplx.Abs(c)}
	}
	sort.Slice(mags, func(a, b int) bool {
		return mags[a].mag > mags[b].mag
	})
	if k > len(mags) {
		k = len(mags)
	}
	indices := make([]int, k)
	for i := 0; i < k; i++ {
		indices[i] = mags[i].idx
	}
	sort.Ints(indices)
	return indices
}

// fretsCache stores intermediate values from the forward pass for backpropagation.
type fretsCache struct {
	// Per-channel DFT results and top-K selection.
	allCoeffs  [][]complex128 // [channels][nFreqs]
	topIndices [][]int        // [channels][topK]

	// Frequency domain values before and after mixing.
	freqRealPreChan [][]float64 // [channels][topK] - from DFT, before channel mixing
	freqImagPreChan [][]float64

	// Channel MLP caches per freq bin: chanHiddenReal[k][hidden], chanPreActReal[k][hidden]
	chanHiddenReal [][]float64 // post-ReLU hidden activations
	chanHiddenImag [][]float64
	chanPreActReal [][]float64 // pre-ReLU values (for ReLU grad)
	chanPreActImag [][]float64
	chanInputReal  [][]float64 // [topK][channels] input to channel MLP
	chanInputImag  [][]float64

	// Freq values after channel mixing, before temporal mixing.
	freqRealPreTemp [][]float64 // [channels][topK]
	freqImagPreTemp [][]float64

	// Temporal MLP caches per channel.
	tempHiddenReal [][]float64 // [channels][hidden]
	tempHiddenImag [][]float64
	tempPreActReal [][]float64
	tempPreActImag [][]float64
	tempInputReal  [][]float64 // [channels][topK]
	tempInputImag  [][]float64

	// Reconstructed time-domain signal per channel.
	reconstructed [][]float64 // [channels][inputLen]
}

// forwardWithCache runs the FreTS forward pass, caching intermediates for backprop.
func (f *FreTS) forwardWithCache(input [][]float64) ([][]float64, *fretsCache) {
	channels := f.config.Channels
	inputLen := f.config.InputLen
	topK := f.config.TopK
	hidden := f.config.HiddenSize
	cache := &fretsCache{}

	// Step 1: DFT per channel, select top-K frequencies.
	cache.allCoeffs = make([][]complex128, channels)
	cache.topIndices = make([][]int, channels)
	cache.freqRealPreChan = make([][]float64, channels)
	cache.freqImagPreChan = make([][]float64, channels)

	freqReal := make([][]float64, channels)
	freqImag := make([][]float64, channels)

	for c := 0; c < channels; c++ {
		cache.allCoeffs[c] = dft(input[c])
		cache.topIndices[c] = topKIndices(cache.allCoeffs[c], topK)
		freqReal[c] = make([]float64, topK)
		freqImag[c] = make([]float64, topK)
		cache.freqRealPreChan[c] = make([]float64, topK)
		cache.freqImagPreChan[c] = make([]float64, topK)
		for i, idx := range cache.topIndices[c] {
			freqReal[c][i] = real(cache.allCoeffs[c][idx])
			freqImag[c][i] = imag(cache.allCoeffs[c][idx])
			cache.freqRealPreChan[c][i] = freqReal[c][i]
			cache.freqImagPreChan[c][i] = freqImag[c][i]
		}
	}

	// Step 2: Channel mixing.
	cache.chanHiddenReal = make([][]float64, topK)
	cache.chanHiddenImag = make([][]float64, topK)
	cache.chanPreActReal = make([][]float64, topK)
	cache.chanPreActImag = make([][]float64, topK)
	cache.chanInputReal = make([][]float64, topK)
	cache.chanInputImag = make([][]float64, topK)

	for k := 0; k < topK; k++ {
		realIn := make([]float64, channels)
		imagIn := make([]float64, channels)
		for c := 0; c < channels; c++ {
			realIn[c] = freqReal[c][k]
			imagIn[c] = freqImag[c][k]
		}
		cache.chanInputReal[k] = make([]float64, channels)
		cache.chanInputImag[k] = make([]float64, channels)
		copy(cache.chanInputReal[k], realIn)
		copy(cache.chanInputImag[k], imagIn)

		// Channel MLP forward with cache.
		hReal := make([]float64, hidden)
		hImag := make([]float64, hidden)
		preActReal := make([]float64, hidden)
		preActImag := make([]float64, hidden)
		for j := 0; j < hidden; j++ {
			val := f.chanB1[j]
			for i := 0; i < channels; i++ {
				val += f.chanW1[i*hidden+j] * realIn[i]
			}
			preActReal[j] = val
			if val > 0 {
				hReal[j] = val
			}
			val = f.chanB1[j]
			for i := 0; i < channels; i++ {
				val += f.chanW1[i*hidden+j] * imagIn[i]
			}
			preActImag[j] = val
			if val > 0 {
				hImag[j] = val
			}
		}
		cache.chanHiddenReal[k] = hReal
		cache.chanHiddenImag[k] = hImag
		cache.chanPreActReal[k] = preActReal
		cache.chanPreActImag[k] = preActImag

		realOut := make([]float64, channels)
		imagOut := make([]float64, channels)
		for j := 0; j < channels; j++ {
			val := f.chanB2[j]
			for i := 0; i < hidden; i++ {
				val += f.chanW2[i*channels+j] * hReal[i]
			}
			realOut[j] = val
			val = f.chanB2[j]
			for i := 0; i < hidden; i++ {
				val += f.chanW2[i*channels+j] * hImag[i]
			}
			imagOut[j] = val
		}

		// Residual connection.
		for c := 0; c < channels; c++ {
			freqReal[c][k] = realIn[c] + realOut[c]
			freqImag[c][k] = imagIn[c] + imagOut[c]
		}
	}

	// Save pre-temporal values.
	cache.freqRealPreTemp = make([][]float64, channels)
	cache.freqImagPreTemp = make([][]float64, channels)
	for c := 0; c < channels; c++ {
		cache.freqRealPreTemp[c] = make([]float64, topK)
		cache.freqImagPreTemp[c] = make([]float64, topK)
		copy(cache.freqRealPreTemp[c], freqReal[c])
		copy(cache.freqImagPreTemp[c], freqImag[c])
	}

	// Step 3: Temporal mixing.
	cache.tempHiddenReal = make([][]float64, channels)
	cache.tempHiddenImag = make([][]float64, channels)
	cache.tempPreActReal = make([][]float64, channels)
	cache.tempPreActImag = make([][]float64, channels)
	cache.tempInputReal = make([][]float64, channels)
	cache.tempInputImag = make([][]float64, channels)

	for c := 0; c < channels; c++ {
		cache.tempInputReal[c] = make([]float64, topK)
		cache.tempInputImag[c] = make([]float64, topK)
		copy(cache.tempInputReal[c], freqReal[c])
		copy(cache.tempInputImag[c], freqImag[c])

		// Temporal MLP forward with cache.
		hReal := make([]float64, hidden)
		hImag := make([]float64, hidden)
		preActReal := make([]float64, hidden)
		preActImag := make([]float64, hidden)
		for j := 0; j < hidden; j++ {
			val := f.tempB1[j]
			for i := 0; i < topK; i++ {
				val += f.tempW1[i*hidden+j] * freqReal[c][i]
			}
			preActReal[j] = val
			if val > 0 {
				hReal[j] = val
			}
			val = f.tempB1[j]
			for i := 0; i < topK; i++ {
				val += f.tempW1[i*hidden+j] * freqImag[c][i]
			}
			preActImag[j] = val
			if val > 0 {
				hImag[j] = val
			}
		}
		cache.tempHiddenReal[c] = hReal
		cache.tempHiddenImag[c] = hImag
		cache.tempPreActReal[c] = preActReal
		cache.tempPreActImag[c] = preActImag

		realOut := make([]float64, topK)
		imagOut := make([]float64, topK)
		for j := 0; j < topK; j++ {
			val := f.tempB2[j]
			for i := 0; i < hidden; i++ {
				val += f.tempW2[i*topK+j] * hReal[i]
			}
			realOut[j] = val
			val = f.tempB2[j]
			for i := 0; i < hidden; i++ {
				val += f.tempW2[i*topK+j] * hImag[i]
			}
			imagOut[j] = val
		}

		// Residual connection.
		for k := 0; k < topK; k++ {
			freqReal[c][k] += realOut[k]
			freqImag[c][k] += imagOut[k]
		}
	}

	// Step 4: Reconstruct time domain signal via inverse DFT.
	cache.reconstructed = make([][]float64, channels)
	for c := 0; c < channels; c++ {
		mixed := make([]complex128, len(cache.allCoeffs[c]))
		for i, idx := range cache.topIndices[c] {
			mixed[idx] = complex(freqReal[c][i], freqImag[c][i])
		}
		cache.reconstructed[c] = idft(mixed, inputLen)
	}

	// Step 5: Linear projection to output length.
	output := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		output[c] = make([]float64, f.config.OutputLen)
		wOff := c * f.config.OutputLen * inputLen
		bOff := c * f.config.OutputLen
		for o := 0; o < f.config.OutputLen; o++ {
			val := f.outB[bOff+o]
			for i := 0; i < inputLen; i++ {
				val += f.outW[wOff+o*inputLen+i] * cache.reconstructed[c][i]
			}
			output[c][o] = val
		}
	}

	return output, cache
}

// forward runs the FreTS forward pass on a single sample.
// Input: [channels][inputLen], returns: [channels][outputLen].
func (f *FreTS) forward(input [][]float64) [][]float64 {
	out, _ := f.forwardWithCache(input)
	return out
}

// backward computes gradients for a single sample given output error dOut[channels][outputLen].
// Returns gradients in the same order as flatParams.
func (f *FreTS) backward(dOut [][]float64, cache *fretsCache) []float64 {
	channels := f.config.Channels
	inputLen := f.config.InputLen
	topK := f.config.TopK
	hidden := f.config.HiddenSize

	nParams := f.paramCount()
	grads := make([]float64, nParams)

	// Param layout offsets.
	chanW1Off := 0
	chanB1Off := chanW1Off + channels*hidden
	chanW2Off := chanB1Off + hidden
	chanB2Off := chanW2Off + hidden*channels
	tempW1Off := chanB2Off + channels
	tempB1Off := tempW1Off + topK*hidden
	tempW2Off := tempB1Off + hidden
	tempB2Off := tempW2Off + hidden*topK
	outWOff := tempB2Off + topK
	outBOff := outWOff + channels*f.config.OutputLen*inputLen

	// Step 5 backward: output projection.
	// output[c][o] = outB[c*outputLen+o] + sum_i(outW[c*outputLen*inputLen + o*inputLen + i] * reconstructed[c][i])
	dReconstructed := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		dReconstructed[c] = make([]float64, inputLen)
		wOff := c * f.config.OutputLen * inputLen
		bOff := c * f.config.OutputLen
		for o := 0; o < f.config.OutputLen; o++ {
			d := dOut[c][o]
			grads[outBOff+bOff+o] += d
			for i := 0; i < inputLen; i++ {
				grads[outWOff+wOff+o*inputLen+i] += d * cache.reconstructed[c][i]
				dReconstructed[c][i] += d * f.outW[wOff+o*inputLen+i]
			}
		}
	}

	// Step 4 backward: IDFT is linear, so we can compute dFreqReal/dFreqImag from dReconstructed.
	// reconstructed[c][t] = sum_k (real_part_of(mixed[idx] * exp(j*2pi*idx*t/n)) ...) / n
	// For the selected frequency bins, we need the Jacobian of IDFT.
	dFreqReal := make([][]float64, channels)
	dFreqImag := make([][]float64, channels)
	n := float64(inputLen)
	for c := 0; c < channels; c++ {
		dFreqReal[c] = make([]float64, topK)
		dFreqImag[c] = make([]float64, topK)
		for i, idx := range cache.topIndices[c] {
			var dR, dI float64
			for t := 0; t < inputLen; t++ {
				angle := 2 * math.Pi * float64(idx) * float64(t) / n
				cosA := math.Cos(angle)
				sinA := math.Sin(angle)
				// d(reconstructed[t])/d(real(coeff[idx])):
				// For k=0 or k=n/2: just cosA/n
				// For other k: 2*cosA/n (conjugate symmetry doubles the contribution)
				// d(reconstructed[t])/d(imag(coeff[idx])):
				// For k=0 or k=n/2: -sinA/n
				// For other k: -2*sinA/n
				mult := 1.0
				if idx != 0 && !(inputLen%2 == 0 && idx == inputLen/2) {
					mult = 2.0
				}
				dR += dReconstructed[c][t] * mult * cosA / n
				dI += dReconstructed[c][t] * mult * (-sinA) / n
			}
			dFreqReal[c][i] = dR
			dFreqImag[c][i] = dI
		}
	}

	// Step 3 backward: Temporal mixing.
	// freqReal[c][k] (after) = freqReal[c][k] (before) + tempMLPOutput_real[c][k]
	// So dTempOut_real[c][k] = dFreqReal[c][k], and dFreqRealPreTemp[c][k] += dFreqReal[c][k]
	for c := 0; c < channels; c++ {
		// Temporal MLP backward for real part.
		dTempOutReal := make([]float64, topK)
		dTempOutImag := make([]float64, topK)
		copy(dTempOutReal, dFreqReal[c])
		copy(dTempOutImag, dFreqImag[c])

		// Layer 2 backward: out[j] = tempB2[j] + sum_i(tempW2[i*topK+j] * h[i])
		dHReal := make([]float64, hidden)
		dHImag := make([]float64, hidden)
		for j := 0; j < topK; j++ {
			grads[tempB2Off+j] += dTempOutReal[j] + dTempOutImag[j]
			for i := 0; i < hidden; i++ {
				grads[tempW2Off+i*topK+j] += dTempOutReal[j]*cache.tempHiddenReal[c][i] + dTempOutImag[j]*cache.tempHiddenImag[c][i]
				dHReal[i] += dTempOutReal[j] * f.tempW2[i*topK+j]
				dHImag[i] += dTempOutImag[j] * f.tempW2[i*topK+j]
			}
		}

		// ReLU backward.
		dPreActReal := make([]float64, hidden)
		dPreActImag := make([]float64, hidden)
		for j := 0; j < hidden; j++ {
			if cache.tempPreActReal[c][j] > 0 {
				dPreActReal[j] = dHReal[j]
			}
			if cache.tempPreActImag[c][j] > 0 {
				dPreActImag[j] = dHImag[j]
			}
		}

		// Layer 1 backward: preact[j] = tempB1[j] + sum_i(tempW1[i*hidden+j] * input[i])
		for j := 0; j < hidden; j++ {
			grads[tempB1Off+j] += dPreActReal[j] + dPreActImag[j]
			for i := 0; i < topK; i++ {
				grads[tempW1Off+i*hidden+j] += dPreActReal[j]*cache.tempInputReal[c][i] + dPreActImag[j]*cache.tempInputImag[c][i]
			}
		}

		// Gradient flows through residual to pre-temp values (which are post-channel values).
		// dFreqReal/Imag already contains the residual gradient.
	}

	// Step 2 backward: Channel mixing.
	// freqReal[c][k] (post-chan) = freqReal[c][k] (pre-chan) + chanMLPOutput_real[c][k]
	// The gradient from temporal mixing flows to post-channel values.
	// dFreqReal[c][k] is the gradient on the post-temporal value.
	// Through the temporal residual, it flows to both the temporal MLP input and the pre-temporal value.
	// The pre-temporal value IS the post-channel value.
	// Through the channel residual, it flows to both the channel MLP output and the pre-channel value.

	for k := 0; k < topK; k++ {
		// Gather dOut for channel MLP at this freq bin.
		dChanOutReal := make([]float64, channels)
		dChanOutImag := make([]float64, channels)
		for c := 0; c < channels; c++ {
			dChanOutReal[c] = dFreqReal[c][k]
			dChanOutImag[c] = dFreqImag[c][k]
		}

		// Layer 2 backward: out[j] = chanB2[j] + sum_i(chanW2[i*channels+j] * h[i])
		dHReal := make([]float64, hidden)
		dHImag := make([]float64, hidden)
		for j := 0; j < channels; j++ {
			grads[chanB2Off+j] += dChanOutReal[j] + dChanOutImag[j]
			for i := 0; i < hidden; i++ {
				grads[chanW2Off+i*channels+j] += dChanOutReal[j]*cache.chanHiddenReal[k][i] + dChanOutImag[j]*cache.chanHiddenImag[k][i]
				dHReal[i] += dChanOutReal[j] * f.chanW2[i*channels+j]
				dHImag[i] += dChanOutImag[j] * f.chanW2[i*channels+j]
			}
		}

		// ReLU backward.
		dPreActReal := make([]float64, hidden)
		dPreActImag := make([]float64, hidden)
		for j := 0; j < hidden; j++ {
			if cache.chanPreActReal[k][j] > 0 {
				dPreActReal[j] = dHReal[j]
			}
			if cache.chanPreActImag[k][j] > 0 {
				dPreActImag[j] = dHImag[j]
			}
		}

		// Layer 1 backward: preact[j] = chanB1[j] + sum_i(chanW1[i*hidden+j] * input[i])
		for j := 0; j < hidden; j++ {
			grads[chanB1Off+j] += dPreActReal[j] + dPreActImag[j]
			for i := 0; i < channels; i++ {
				grads[chanW1Off+i*hidden+j] += dPreActReal[j]*cache.chanInputReal[k][i] + dPreActImag[j]*cache.chanInputImag[k][i]
			}
		}
	}

	// Clamp NaN/Inf gradients.
	for i := range grads {
		if !isFinite(grads[i]) {
			grads[i] = 0
		}
	}

	return grads
}

// channelMLP applies the channel mixing MLP: [channels] -> [hiddenSize] -> [channels].
func (f *FreTS) channelMLP(input []float64) []float64 {
	channels := f.config.Channels
	hidden := f.config.HiddenSize

	h := make([]float64, hidden)
	for j := 0; j < hidden; j++ {
		val := f.chanB1[j]
		for i := 0; i < channels; i++ {
			val += f.chanW1[i*hidden+j] * input[i]
		}
		if val > 0 {
			h[j] = val
		}
	}

	out := make([]float64, channels)
	for j := 0; j < channels; j++ {
		val := f.chanB2[j]
		for i := 0; i < hidden; i++ {
			val += f.chanW2[i*channels+j] * h[i]
		}
		out[j] = val
	}
	return out
}

// temporalMLP applies the temporal mixing MLP: [topK] -> [hiddenSize] -> [topK].
func (f *FreTS) temporalMLP(input []float64) []float64 {
	topK := f.config.TopK
	hidden := f.config.HiddenSize

	h := make([]float64, hidden)
	for j := 0; j < hidden; j++ {
		val := f.tempB1[j]
		for i := 0; i < topK; i++ {
			val += f.tempW1[i*hidden+j] * input[i]
		}
		if val > 0 {
			h[j] = val
		}
	}

	out := make([]float64, topK)
	for j := 0; j < topK; j++ {
		val := f.tempB2[j]
		for i := 0; i < hidden; i++ {
			val += f.tempW2[i*topK+j] * h[i]
		}
		out[j] = val
	}
	return out
}

// TrainWindowed trains the FreTS model on windowed data using AdamW.
// windows: [nSamples][channels][inputLen] input windows.
// labels: flat slice of length nSamples * channels * outputLen (row-major: sample, channel, time).
func (f *FreTS) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("frets: empty training set")
	}

	expectedLabels := nSamples * f.config.Channels * f.config.OutputLen
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("frets: expected %d labels, got %d", expectedLabels, len(labels))
	}

	for i, w := range windows {
		if len(w) != f.config.Channels {
			return nil, fmt.Errorf("frets: window %d has %d channels, expected %d", i, len(w), f.config.Channels)
		}
		for c, ch := range w {
			if len(ch) != f.config.InputLen {
				return nil, fmt.Errorf("frets: window %d channel %d has length %d, expected %d", i, c, len(ch), f.config.InputLen)
			}
		}
	}

	if config.Epochs <= 0 {
		config.Epochs = 100
	}
	if config.LR <= 0 {
		config.LR = 1e-3
	}
	if config.Beta1 <= 0 {
		config.Beta1 = 0.9
	}
	if config.Beta2 <= 0 {
		config.Beta2 = 0.999
	}
	if config.Epsilon <= 0 {
		config.Epsilon = 1e-8
	}

	// Z-score normalize inputs.
	windows, f.normMeans, f.normStds = normalizeWindows(windows)

	nParams := f.paramCount()
	m := make([]float64, nParams)
	v := make([]float64, nParams)

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			batchWindows := windows[start:end]
			batchLabels := labels[start*f.config.Channels*f.config.OutputLen : end*f.config.Channels*f.config.OutputLen]

			grads := make([]float64, nParams)
			batchLoss := 0.0
			bs := end - start

			for s := 0; s < bs; s++ {
				pred, cache := f.forwardWithCache(batchWindows[s])

				// Compute dOut = 2 * (pred - label) / total_elements.
				dOut := make([][]float64, f.config.Channels)
				for c := 0; c < f.config.Channels; c++ {
					dOut[c] = make([]float64, f.config.OutputLen)
					for o := 0; o < f.config.OutputLen; o++ {
						labelIdx := s*f.config.Channels*f.config.OutputLen + c*f.config.OutputLen + o
						diff := pred[c][o] - batchLabels[labelIdx]
						if !isFinite(diff) {
							diff = 0
						}
						batchLoss += diff * diff
						dOut[c][o] = 2.0 * diff / float64(bs*f.config.Channels*f.config.OutputLen)
					}
				}

				sampleGrads := f.backward(dOut, cache)
				for i := range grads {
					grads[i] += sampleGrads[i]
				}
			}

			batchLoss /= float64(bs * f.config.Channels * f.config.OutputLen)
			epochLoss += batchLoss
			nBatches++

			// Gradient clipping.
			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := config.GradClip / norm
					for i := range grads {
						grads[i] *= scale
					}
				}
			}

			// AdamW update with LR warmup.
			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			params := f.flatParams()
			for i := range params {
				m[i] = config.Beta1*m[i] + (1-config.Beta1)*grads[i]
				v[i] = config.Beta2*v[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := m[i] / (1 - math.Pow(config.Beta1, t))
				vHat := v[i] / (1 - math.Pow(config.Beta2, t))
				*params[i] = *params[i] - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*params[i]))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("frets: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// PredictWindowed runs inference on windowed data.
// windows: [nSamples][channels][inputLen].
// Returns flat predictions of length nSamples * channels * outputLen.
func (f *FreTS) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	if modelPath != "" {
		if err := f.loadWeights(modelPath); err != nil {
			return nil, fmt.Errorf("frets: load weights: %w", err)
		}
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("frets: empty input")
	}

	if f.normMeans != nil {
		windows = applyNormalization(windows, f.normMeans, f.normStds)
	}

	out := make([]float64, 0, nSamples*f.config.Channels*f.config.OutputLen)
	for _, w := range windows {
		if len(w) != f.config.Channels {
			return nil, fmt.Errorf("frets: expected %d channels, got %d", f.config.Channels, len(w))
		}
		pred := f.forward(w)
		for c := 0; c < f.config.Channels; c++ {
			out = append(out, pred[c]...)
		}
	}
	return out, nil
}

// paramCount returns the total number of trainable parameters.
func (f *FreTS) paramCount() int {
	channels := f.config.Channels
	hidden := f.config.HiddenSize
	topK := f.config.TopK
	inputLen := f.config.InputLen
	outputLen := f.config.OutputLen

	chanParams := channels*hidden + hidden + hidden*channels + channels
	tempParams := topK*hidden + hidden + hidden*topK + topK
	outParams := channels*outputLen*inputLen + channels*outputLen

	return chanParams + tempParams + outParams
}

// flatParams returns pointers to all trainable parameters in a flat slice.
func (f *FreTS) flatParams() []*float64 {
	n := f.paramCount()
	params := make([]*float64, 0, n)

	for i := range f.chanW1 {
		params = append(params, &f.chanW1[i])
	}
	for i := range f.chanB1 {
		params = append(params, &f.chanB1[i])
	}
	for i := range f.chanW2 {
		params = append(params, &f.chanW2[i])
	}
	for i := range f.chanB2 {
		params = append(params, &f.chanB2[i])
	}
	for i := range f.tempW1 {
		params = append(params, &f.tempW1[i])
	}
	for i := range f.tempB1 {
		params = append(params, &f.tempB1[i])
	}
	for i := range f.tempW2 {
		params = append(params, &f.tempW2[i])
	}
	for i := range f.tempB2 {
		params = append(params, &f.tempB2[i])
	}
	for i := range f.outW {
		params = append(params, &f.outW[i])
	}
	for i := range f.outB {
		params = append(params, &f.outB[i])
	}

	return params
}

// fretsWeights is the JSON-serializable form of FreTS parameters.
type fretsWeights struct {
	Config    FreTSConfig `json:"config"`
	ChanW1   []float64   `json:"chan_w1"`
	ChanB1   []float64   `json:"chan_b1"`
	ChanW2   []float64   `json:"chan_w2"`
	ChanB2   []float64   `json:"chan_b2"`
	TempW1   []float64   `json:"temp_w1"`
	TempB1   []float64   `json:"temp_b1"`
	TempW2   []float64   `json:"temp_w2"`
	TempB2   []float64   `json:"temp_b2"`
	OutW     []float64   `json:"out_w"`
	OutB     []float64   `json:"out_b"`
	NormMeans [][]float64 `json:"norm_means,omitempty"`
	NormStds  [][]float64 `json:"norm_stds,omitempty"`
}

// SaveWeights writes the model weights to a JSON file.
func (f *FreTS) SaveWeights(path string) error {
	w := fretsWeights{
		Config:    f.config,
		ChanW1:   f.chanW1,
		ChanB1:   f.chanB1,
		ChanW2:   f.chanW2,
		ChanB2:   f.chanB2,
		TempW1:   f.tempW1,
		TempB1:   f.tempB1,
		TempW2:   f.tempW2,
		TempB2:   f.tempB2,
		OutW:     f.outW,
		OutB:     f.outB,
		NormMeans: f.normMeans,
		NormStds:  f.normStds,
	}
	data, err := json.Marshal(w)
	if err != nil {
		return fmt.Errorf("frets: marshal weights: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// loadWeights reads model weights from a JSON file.
func (f *FreTS) loadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var w fretsWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Config != f.config {
		return fmt.Errorf("frets: config mismatch: file has %+v, model has %+v", w.Config, f.config)
	}
	f.chanW1 = w.ChanW1
	f.chanB1 = w.ChanB1
	f.chanW2 = w.ChanW2
	f.chanB2 = w.ChanB2
	f.tempW1 = w.TempW1
	f.tempB1 = w.TempB1
	f.tempW2 = w.TempW2
	f.tempB2 = w.TempB2
	f.outW = w.OutW
	f.outB = w.OutB
	f.normMeans = w.NormMeans
	f.normStds = w.NormStds
	return nil
}
