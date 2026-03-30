package timeseries

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/ztensor/compute"
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
	config    DLinearConfig
	trendW    [][]float64            // [channels][outputLen * inputLen]
	trendB    [][]float64            // [channels][outputLen]
	seasonalW [][]float64            // [channels][outputLen * inputLen]
	seasonalB [][]float64            // [channels][outputLen]
	engine    compute.Engine[float32] // optional; enables GPU-accelerated training
	normMeans [][]float64            // per-channel normalization means from training
	normStds  [][]float64            // per-channel normalization stds from training
	grads     []float64              // gradient accumulator for TrainableBackend
}

// dlinearCache holds activations from a forward pass needed for backpropagation.
type dlinearCache struct {
	trend    [][]float64 // [channels][inputLen] — trend component
	seasonal [][]float64 // [channels][inputLen] — seasonal component
	output   [][]float64 // [channels][outputLen] — model output
}

// DLinearOption configures a DLinear model.
type DLinearOption func(*DLinear)

// WithEngine sets the compute engine for GPU-accelerated training.
// When nil (the default), DLinear uses the pure-Go CPU training path.
func WithEngine(engine compute.Engine[float32]) DLinearOption {
	return func(d *DLinear) {
		d.engine = engine
	}
}

// NewDLinear creates a new DLinear model with the given configuration.
func NewDLinear(inputLen, outputLen, channels, kernelSize int, opts ...DLinearOption) (*DLinear, error) {
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

	for _, opt := range opts {
		opt(d)
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

	if d.engine != nil {
		return d.trainWindowedEngine(windows, labels, config)
	}

	// Z-score normalize inputs to prevent gradient explosion on multi-scale data.
	windows, d.normMeans, d.normStds = normalizeWindows(windows)

	return TrainLoop(d, windows, labels, config)
}

// applyNormalization normalizes windows using stored means and stds from training.
func applyNormalization(windows [][][]float64, means, stds [][]float64) [][][]float64 {
	nSamples := len(windows)
	if nSamples == 0 {
		return windows
	}
	nChannels := len(windows[0])
	inputLen := len(windows[0][0])
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
	return out
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

	// Apply normalization from training if available.
	if d.normMeans != nil {
		windows = applyNormalization(windows, d.normMeans, d.normStds)
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

// ForwardSample runs the DLinear forward pass on a single sample and returns
// a flat output with cached activations for BackwardSample.
func (d *DLinear) ForwardSample(input [][]float64) ([]float64, interface{}, error) {
	trend, seasonal := d.decompose(input)
	output := d.forward(input)

	flat := make([]float64, 0, d.config.Channels*d.config.OutputLen)
	for c := 0; c < d.config.Channels; c++ {
		flat = append(flat, output[c]...)
	}

	cache := &dlinearCache{
		trend:    trend,
		seasonal: seasonal,
		output:   output,
	}
	return flat, cache, nil
}

// BackwardSample accumulates parameter gradients for a single sample.
func (d *DLinear) BackwardSample(dOutput []float64, cacheIface interface{}) error {
	cache, ok := cacheIface.(*dlinearCache)
	if !ok {
		return fmt.Errorf("dlinear: invalid cache type")
	}

	if d.grads == nil {
		d.grads = make([]float64, d.paramCount())
	}

	for c := 0; c < d.config.Channels; c++ {
		for o := 0; o < d.config.OutputLen; o++ {
			dOut := dOutput[c*d.config.OutputLen+o]

			// Gradients for trend linear.
			tOff := d.trendParamOffset(c)
			for i := 0; i < d.config.InputLen; i++ {
				d.grads[tOff+o*d.config.InputLen+i] += dOut * cache.trend[c][i]
			}
			d.grads[tOff+d.config.OutputLen*d.config.InputLen+o] += dOut

			// Gradients for seasonal linear.
			sOff := d.seasonalParamOffset(c)
			for i := 0; i < d.config.InputLen; i++ {
				d.grads[sOff+o*d.config.InputLen+i] += dOut * cache.seasonal[c][i]
			}
			d.grads[sOff+d.config.OutputLen*d.config.InputLen+o] += dOut
		}
	}
	return nil
}

// FlatGrads returns the internal gradient accumulator.
func (d *DLinear) FlatGrads() []float64 {
	if d.grads == nil {
		d.grads = make([]float64, d.paramCount())
	}
	return d.grads
}

// ZeroGrads resets all accumulated gradients to zero.
func (d *DLinear) ZeroGrads() {
	if d.grads == nil {
		d.grads = make([]float64, d.paramCount())
		return
	}
	for i := range d.grads {
		d.grads[i] = 0
	}
}

// FlatParams returns pointers to all trainable parameters (exported for TrainableBackend).
func (d *DLinear) FlatParams() []*float64 {
	return d.flatParams()
}

// ParamCount returns the total number of trainable parameters (exported for TrainableBackend).
func (d *DLinear) ParamCount() int {
	return d.paramCount()
}

// Compile-time check that DLinear implements TrainableBackend.
var _ TrainableBackend = (*DLinear)(nil)

// dlinearWeights is the JSON-serializable form of DLinear parameters.
type dlinearWeights struct {
	Config     DLinearConfig   `json:"config"`
	TrendW     [][]float64     `json:"trend_w"`
	TrendB     [][]float64     `json:"trend_b"`
	SeasonalW  [][]float64     `json:"seasonal_w"`
	SeasonalB  [][]float64     `json:"seasonal_b"`
	NormMeans  [][]float64     `json:"norm_means,omitempty"`
	NormStds   [][]float64     `json:"norm_stds,omitempty"`
}

// SaveWeights writes the model weights to a JSON file.
func (d *DLinear) SaveWeights(path string) error {
	w := dlinearWeights{
		Config:    d.config,
		TrendW:    d.trendW,
		TrendB:    d.trendB,
		SeasonalW: d.seasonalW,
		SeasonalB: d.seasonalB,
		NormMeans: d.normMeans,
		NormStds:  d.normStds,
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
	d.normMeans = w.NormMeans
	d.normStds = w.NormStds
	return nil
}
