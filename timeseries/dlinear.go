package timeseries

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

// TrainConfig holds training hyperparameters for windowed time-series backends.
type TrainConfig struct {
	Epochs       int     // number of training epochs
	LR           float64 // learning rate
	WeightDecay  float64 // AdamW weight decay
	GradClip     float64 // max gradient norm (0 = no clipping)
	BatchSize    int     // mini-batch size (0 = full batch)
	Beta1        float64 // AdamW beta1
	Beta2        float64 // AdamW beta2
	Epsilon      float64 // AdamW epsilon
	WarmupEpochs int     // linear LR warmup over this many epochs (0 = no warmup)
}

// DefaultTrainConfig returns sensible defaults for training.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Epochs:       100,
		LR:           1e-3,
		WeightDecay:  1e-4,
		GradClip:     1.0,
		BatchSize:    0,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WarmupEpochs: 5,
	}
}

// warmupLR returns the effective learning rate for the given epoch,
// applying linear warmup over the first warmupEpochs epochs.
func warmupLR(baseLR float64, epoch, warmupEpochs int) float64 {
	if warmupEpochs <= 0 {
		return baseLR
	}
	scale := float64(epoch+1) / float64(warmupEpochs)
	if scale > 1.0 {
		scale = 1.0
	}
	return baseLR * scale
}

// isFinite returns true if v is neither NaN nor Inf.
func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// normalizeWindows applies z-score normalization per channel across all samples.
// It returns the normalized windows, per-channel means, and per-channel standard
// deviations. Each channel is normalized independently: x' = (x - mean) / (std + 1e-8).
func normalizeWindows(windows [][][]float64) ([][][]float64, [][]float64, [][]float64) {
	if len(windows) == 0 {
		return windows, nil, nil
	}
	nChannels := len(windows[0])
	inputLen := 0
	if nChannels > 0 {
		inputLen = len(windows[0][0])
	}
	nSamples := len(windows)

	means := make([][]float64, nChannels)
	stds := make([][]float64, nChannels)

	for c := 0; c < nChannels; c++ {
		means[c] = make([]float64, inputLen)
		stds[c] = make([]float64, inputLen)

		// Compute mean per timestep.
		for i := 0; i < nSamples; i++ {
			for t := 0; t < inputLen; t++ {
				means[c][t] += windows[i][c][t]
			}
		}
		for t := 0; t < inputLen; t++ {
			means[c][t] /= float64(nSamples)
		}

		// Compute std per timestep.
		for i := 0; i < nSamples; i++ {
			for t := 0; t < inputLen; t++ {
				d := windows[i][c][t] - means[c][t]
				stds[c][t] += d * d
			}
		}
		for t := 0; t < inputLen; t++ {
			stds[c][t] = math.Sqrt(stds[c][t] / float64(nSamples))
		}
	}

	// Normalize in-place copy.
	out := make([][][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		out[i] = make([][]float64, nChannels)
		for c := 0; c < nChannels; c++ {
			out[i][c] = make([]float64, inputLen)
			for t := 0; t < inputLen; t++ {
				out[i][c][t] = (windows[i][c][t] - means[c][t]) / (stds[c][t] + 1e-8)
			}
		}
	}
	return out, means, stds
}

// TrainResult holds training metrics.
type TrainResult struct {
	FinalLoss   float64            // loss at last epoch
	LossHistory []float64          // loss per epoch
	ModelPath   string             // path where model weights were saved
	Metrics     map[string]float64 // e.g., "mse", "correlation", "directional_accuracy"
}

// DLinearConfig holds the configuration for a DLinear model.
type DLinearConfig struct {
	InputLen   int // input sequence length (lookback window)
	OutputLen  int // forecast horizon
	Channels   int // number of channels/features
	KernelSize int // moving average kernel size (must be odd)
}

// DLinear implements the DLinear time-series forecasting model (AAAI 2023).
// It decomposes input into trend and seasonal components using moving average,
// then applies separate linear projections for each component per channel.
type DLinear struct {
	config     DLinearConfig
	trendW     [][]float64 // [channels][outputLen * inputLen]
	trendB     [][]float64 // [channels][outputLen]
	seasonalW  [][]float64 // [channels][outputLen * inputLen]
	seasonalB  [][]float64 // [channels][outputLen]
}

// NewDLinear creates a new DLinear model with the given configuration.
func NewDLinear(inputLen, outputLen, channels, kernelSize int) (*DLinear, error) {
	if inputLen <= 0 {
		return nil, fmt.Errorf("dlinear: inputLen must be positive, got %d", inputLen)
	}
	if outputLen <= 0 {
		return nil, fmt.Errorf("dlinear: outputLen must be positive, got %d", outputLen)
	}
	if channels <= 0 {
		return nil, fmt.Errorf("dlinear: channels must be positive, got %d", channels)
	}
	if kernelSize <= 0 || kernelSize%2 == 0 {
		return nil, fmt.Errorf("dlinear: kernelSize must be positive and odd, got %d", kernelSize)
	}

	d := &DLinear{
		config: DLinearConfig{
			InputLen:   inputLen,
			OutputLen:  outputLen,
			Channels:   channels,
			KernelSize: kernelSize,
		},
		trendW:    make([][]float64, channels),
		trendB:    make([][]float64, channels),
		seasonalW: make([][]float64, channels),
		seasonalB: make([][]float64, channels),
	}

	// Xavier initialization for linear projections.
	scale := math.Sqrt(2.0 / float64(inputLen+outputLen))
	for c := 0; c < channels; c++ {
		d.trendW[c] = make([]float64, outputLen*inputLen)
		d.trendB[c] = make([]float64, outputLen)
		d.seasonalW[c] = make([]float64, outputLen*inputLen)
		d.seasonalB[c] = make([]float64, outputLen)
		for i := range d.trendW[c] {
			d.trendW[c][i] = rand.NormFloat64() * scale
		}
		for i := range d.seasonalW[c] {
			d.seasonalW[c][i] = rand.NormFloat64() * scale
		}
	}

	return d, nil
}

// movingAverage computes a centered moving average with edge padding.
// Input: [length], output: [length].
func movingAverage(x []float64, kernelSize int) []float64 {
	n := len(x)
	out := make([]float64, n)
	half := kernelSize / 2

	for i := 0; i < n; i++ {
		sum := 0.0
		count := 0
		for j := i - half; j <= i+half; j++ {
			idx := j
			if idx < 0 {
				idx = 0
			} else if idx >= n {
				idx = n - 1
			}
			sum += x[idx]
			count++
		}
		out[i] = sum / float64(count)
	}
	return out
}

// decompose splits input into trend and seasonal components.
// Input: [channels][inputLen], returns (trend, seasonal) each [channels][inputLen].
func (d *DLinear) decompose(input [][]float64) ([][]float64, [][]float64) {
	channels := len(input)
	trend := make([][]float64, channels)
	seasonal := make([][]float64, channels)

	for c := 0; c < channels; c++ {
		trend[c] = movingAverage(input[c], d.config.KernelSize)
		seasonal[c] = make([]float64, len(input[c]))
		for i := range input[c] {
			seasonal[c][i] = input[c][i] - trend[c][i]
		}
	}
	return trend, seasonal
}

// forward runs the DLinear forward pass on a single sample.
// Input: [channels][inputLen], returns: [channels][outputLen].
func (d *DLinear) forward(input [][]float64) [][]float64 {
	trend, seasonal := d.decompose(input)
	output := make([][]float64, d.config.Channels)

	for c := 0; c < d.config.Channels; c++ {
		output[c] = make([]float64, d.config.OutputLen)
		// trendOut = trendW @ trend + trendB
		// seasonalOut = seasonalW @ seasonal + seasonalB
		for o := 0; o < d.config.OutputLen; o++ {
			trendVal := d.trendB[c][o]
			seasonVal := d.seasonalB[c][o]
			for i := 0; i < d.config.InputLen; i++ {
				trendVal += d.trendW[c][o*d.config.InputLen+i] * trend[c][i]
				seasonVal += d.seasonalW[c][o*d.config.InputLen+i] * seasonal[c][i]
			}
			output[c][o] = trendVal + seasonVal
		}
	}
	return output
}

// TrainWindowed trains the DLinear model on windowed data using AdamW.
// windows: [nSamples][channels][inputLen] input windows.
// labels: flat slice of length nSamples * channels * outputLen (row-major: sample, channel, time).
func (d *DLinear) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("dlinear: empty training set")
	}

	expectedLabels := nSamples * d.config.Channels * d.config.OutputLen
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("dlinear: expected %d labels, got %d", expectedLabels, len(labels))
	}

	// Validate window shapes.
	for i, w := range windows {
		if len(w) != d.config.Channels {
			return nil, fmt.Errorf("dlinear: window %d has %d channels, expected %d", i, len(w), d.config.Channels)
		}
		for c, ch := range w {
			if len(ch) != d.config.InputLen {
				return nil, fmt.Errorf("dlinear: window %d channel %d has length %d, expected %d", i, c, len(ch), d.config.InputLen)
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

	// Z-score normalize inputs to prevent gradient explosion on multi-scale data.
	windows, _, _ = normalizeWindows(windows)

	// Flatten all parameters and their gradients for AdamW.
	nParams := d.paramCount()
	m := make([]float64, nParams) // first moment
	v := make([]float64, nParams) // second moment

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
			batchLabels := labels[start*d.config.Channels*d.config.OutputLen : end*d.config.Channels*d.config.OutputLen]

			grads := make([]float64, nParams)
			batchLoss := 0.0
			bs := end - start

			for s := 0; s < bs; s++ {
				pred := d.forward(batchWindows[s])
				trend, seasonal := d.decompose(batchWindows[s])

				for c := 0; c < d.config.Channels; c++ {
					for o := 0; o < d.config.OutputLen; o++ {
						labelIdx := s*d.config.Channels*d.config.OutputLen + c*d.config.OutputLen + o
						diff := pred[c][o] - batchLabels[labelIdx]
						batchLoss += diff * diff

						// Gradient of MSE: 2*diff / total_elements
						dOut := 2.0 * diff / float64(bs*d.config.Channels*d.config.OutputLen)

						// Gradients for trend linear.
						tOff := d.trendParamOffset(c)
						for i := 0; i < d.config.InputLen; i++ {
							grads[tOff+o*d.config.InputLen+i] += dOut * trend[c][i]
						}
						grads[tOff+d.config.OutputLen*d.config.InputLen+o] += dOut

						// Gradients for seasonal linear.
						sOff := d.seasonalParamOffset(c)
						for i := 0; i < d.config.InputLen; i++ {
							grads[sOff+o*d.config.InputLen+i] += dOut * seasonal[c][i]
						}
						grads[sOff+d.config.OutputLen*d.config.InputLen+o] += dOut
					}
				}
			}

			batchLoss /= float64(bs * d.config.Channels * d.config.OutputLen)
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
			params := d.flatParams()
			for i := range params {
				m[i] = config.Beta1*m[i] + (1-config.Beta1)*grads[i]
				v[i] = config.Beta2*v[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := m[i] / (1 - math.Pow(config.Beta1, t))
				vHat := v[i] / (1 - math.Pow(config.Beta2, t))
				// AdamW: weight decay applied to param directly, not through gradient.
				*params[i] = *params[i] - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*params[i]))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		// Early halt on NaN/Inf loss.
		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("dlinear: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// PredictWindowed runs inference on windowed data.
// windows: [nSamples][channels][inputLen].
// Returns flat predictions of length nSamples * channels * outputLen.
func (d *DLinear) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	// If modelPath is non-empty, load weights from file.
	if modelPath != "" {
		if err := d.loadWeights(modelPath); err != nil {
			return nil, fmt.Errorf("dlinear: load weights: %w", err)
		}
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("dlinear: empty input")
	}

	out := make([]float64, 0, nSamples*d.config.Channels*d.config.OutputLen)
	for _, w := range windows {
		if len(w) != d.config.Channels {
			return nil, fmt.Errorf("dlinear: expected %d channels, got %d", d.config.Channels, len(w))
		}
		pred := d.forward(w)
		for c := 0; c < d.config.Channels; c++ {
			out = append(out, pred[c]...)
		}
	}
	return out, nil
}

// paramCount returns the total number of trainable parameters.
func (d *DLinear) paramCount() int {
	// Per channel: outputLen*inputLen weights + outputLen biases, for both trend and seasonal.
	perChannel := 2 * (d.config.OutputLen*d.config.InputLen + d.config.OutputLen)
	return d.config.Channels * perChannel
}

// trendParamOffset returns the flat parameter offset for trend weights+bias of channel c.
func (d *DLinear) trendParamOffset(c int) int {
	perChannel := 2 * (d.config.OutputLen*d.config.InputLen + d.config.OutputLen)
	return c * perChannel
}

// seasonalParamOffset returns the flat parameter offset for seasonal weights+bias of channel c.
func (d *DLinear) seasonalParamOffset(c int) int {
	perChannel := 2 * (d.config.OutputLen*d.config.InputLen + d.config.OutputLen)
	return c*perChannel + d.config.OutputLen*d.config.InputLen + d.config.OutputLen
}

// flatParams returns pointers to all trainable parameters in a flat slice.
// Order: for each channel: trendW, trendB, seasonalW, seasonalB.
func (d *DLinear) flatParams() []*float64 {
	n := d.paramCount()
	params := make([]*float64, 0, n)
	for c := 0; c < d.config.Channels; c++ {
		for i := range d.trendW[c] {
			params = append(params, &d.trendW[c][i])
		}
		for i := range d.trendB[c] {
			params = append(params, &d.trendB[c][i])
		}
		for i := range d.seasonalW[c] {
			params = append(params, &d.seasonalW[c][i])
		}
		for i := range d.seasonalB[c] {
			params = append(params, &d.seasonalB[c][i])
		}
	}
	return params
}

// dlinearWeights is the JSON-serializable form of DLinear parameters.
type dlinearWeights struct {
	Config     DLinearConfig   `json:"config"`
	TrendW     [][]float64     `json:"trend_w"`
	TrendB     [][]float64     `json:"trend_b"`
	SeasonalW  [][]float64     `json:"seasonal_w"`
	SeasonalB  [][]float64     `json:"seasonal_b"`
}

// SaveWeights writes the model weights to a JSON file.
func (d *DLinear) SaveWeights(path string) error {
	w := dlinearWeights{
		Config:    d.config,
		TrendW:    d.trendW,
		TrendB:    d.trendB,
		SeasonalW: d.seasonalW,
		SeasonalB: d.seasonalB,
	}
	data, err := json.Marshal(w)
	if err != nil {
		return fmt.Errorf("dlinear: marshal weights: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// loadWeights reads model weights from a JSON file.
func (d *DLinear) loadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var w dlinearWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Config != d.config {
		return fmt.Errorf("dlinear: config mismatch: file has %+v, model has %+v", w.Config, d.config)
	}
	d.trendW = w.TrendW
	d.trendB = w.TrendB
	d.seasonalW = w.SeasonalW
	d.seasonalB = w.SeasonalB
	return nil
}
