// Package timeseries provides time-series forecasting models built on ztensor.
//
// Stability: alpha
package timeseries

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// StackType identifies the type of basis expansion used in an N-BEATS stack.
type StackType int

const (
	// StackTrend uses polynomial basis expansion for trend modeling.
	StackTrend StackType = iota
	// StackSeasonality uses Fourier basis expansion for seasonal patterns.
	StackSeasonality
	// StackGeneric uses learned linear projections as basis functions.
	StackGeneric
)

// String returns the string representation of a StackType.
func (s StackType) String() string {
	switch s {
	case StackTrend:
		return "trend"
	case StackSeasonality:
		return "seasonality"
	case StackGeneric:
		return "generic"
	default:
		return fmt.Sprintf("StackType(%d)", int(s))
	}
}

// NBEATSConfig holds the configuration for an NBEATS model.
type NBEATSConfig struct {
	InputLength     int         // length of the lookback window
	OutputLength    int         // forecast horizon
	StackTypes      []StackType // types of stacks (e.g., [StackTrend, StackSeasonality])
	NBlocksPerStack int         // number of blocks in each stack
	HiddenDim       int         // hidden dimension of FC layers in each block
	NHarmonics      int         // number of Fourier harmonics for seasonality stacks
}

// nbeatsBlock is a single N-BEATS block with FC layers and basis expansion.
type nbeatsBlock struct {
	stackType   StackType
	inputLen    int
	outputLen   int
	hiddenDim   int
	nHarmonics  int
	fcLayers    []mlpLayer // 4 FC layers
	thetaBLayer mlpLayer   // projects to backcast theta
	thetaFLayer mlpLayer   // projects to forecast theta

	// Precomputed basis matrices for trend/seasonality blocks.
	backcastBasis  *tensor.TensorNumeric[float32] // [thetaDim, inputLen]
	forecastBasis  *tensor.TensorNumeric[float32] // [thetaDim, outputLen]
}

// mlpLayer holds a single linear layer's weights and biases.
type mlpLayer struct {
	weights *tensor.TensorNumeric[float32]
	biases  *tensor.TensorNumeric[float32]
}

// nbeatsStack holds a sequence of blocks of the same type.
type nbeatsStack struct {
	stackType StackType
	blocks    []nbeatsBlock
}

// NBEATSOutput holds the result of an N-BEATS forward pass.
type NBEATSOutput struct {
	Forecast       *tensor.TensorNumeric[float32]   // [batch, outputLen]
	StackForecasts []*tensor.TensorNumeric[float32]  // per-stack forecasts for decomposition
	StackBackcasts []*tensor.TensorNumeric[float32]  // per-stack backcasts
}

// NBEATS implements the N-BEATS (Neural Basis Expansion Analysis for
// Interpretable Time Series Forecasting) architecture.
//
// The model consists of stacks of blocks. Each block applies FC layers to
// produce theta parameters, which are expanded through a basis function
// (polynomial for trend, Fourier for seasonality, learned for generic)
// to produce backcast and forecast signals. Double residual stacking
// subtracts the backcast from the input and accumulates forecasts.
type NBEATS struct {
	config NBEATSConfig
	engine compute.Engine[float32]
	ops    numeric.Arithmetic[float32]
	stacks []nbeatsStack
}

// NewNBEATS creates a new N-BEATS model with the given configuration.
func NewNBEATS(config NBEATSConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*NBEATS, error) {
	if config.InputLength <= 0 {
		return nil, fmt.Errorf("nbeats: InputLength must be positive, got %d", config.InputLength)
	}
	if config.OutputLength <= 0 {
		return nil, fmt.Errorf("nbeats: OutputLength must be positive, got %d", config.OutputLength)
	}
	if len(config.StackTypes) == 0 {
		return nil, fmt.Errorf("nbeats: StackTypes must have at least one element")
	}
	if config.NBlocksPerStack <= 0 {
		return nil, fmt.Errorf("nbeats: NBlocksPerStack must be positive, got %d", config.NBlocksPerStack)
	}
	if config.HiddenDim <= 0 {
		return nil, fmt.Errorf("nbeats: HiddenDim must be positive, got %d", config.HiddenDim)
	}
	if config.NHarmonics <= 0 {
		return nil, fmt.Errorf("nbeats: NHarmonics must be positive, got %d", config.NHarmonics)
	}

	m := &NBEATS{
		config: config,
		engine: engine,
		ops:    ops,
	}

	m.stacks = make([]nbeatsStack, len(config.StackTypes))
	for i, st := range config.StackTypes {
		blocks := make([]nbeatsBlock, config.NBlocksPerStack)
		for j := 0; j < config.NBlocksPerStack; j++ {
			b, err := newNBEATSBlock(st, config.InputLength, config.OutputLength, config.HiddenDim, config.NHarmonics)
			if err != nil {
				return nil, fmt.Errorf("nbeats: stack %d block %d: %w", i, j, err)
			}
			blocks[j] = b
		}
		m.stacks[i] = nbeatsStack{stackType: st, blocks: blocks}
	}

	return m, nil
}

// thetaDim returns the number of theta parameters for the given stack type.
func thetaDim(st StackType, inputLen, outputLen, nHarmonics int) int {
	switch st {
	case StackTrend:
		// Polynomial degree: use a small polynomial (degree 2 -> 3 coefficients).
		return 3
	case StackSeasonality:
		// 2 * nHarmonics coefficients (sin + cos for each harmonic).
		return 2 * nHarmonics
	case StackGeneric:
		// For generic blocks, theta directly maps to the output lengths.
		// Backcast theta = inputLen, forecast theta = outputLen.
		// We use max(inputLen, outputLen) for a unified theta dimension,
		// then project separately.
		return inputLen + outputLen
	default:
		return inputLen + outputLen
	}
}

// newNBEATSBlock creates a single N-BEATS block.
func newNBEATSBlock(st StackType, inputLen, outputLen, hiddenDim, nHarmonics int) (nbeatsBlock, error) {
	b := nbeatsBlock{
		stackType:  st,
		inputLen:   inputLen,
		outputLen:  outputLen,
		hiddenDim:  hiddenDim,
		nHarmonics: nHarmonics,
	}

	// 4 FC layers: inputLen -> hidden -> hidden -> hidden -> hidden.
	dims := []int{inputLen, hiddenDim, hiddenDim, hiddenDim, hiddenDim}
	b.fcLayers = make([]mlpLayer, 4)
	for i := 0; i < 4; i++ {
		l, err := newMLPLayer(dims[i], dims[i+1])
		if err != nil {
			return nbeatsBlock{}, fmt.Errorf("fc layer %d: %w", i, err)
		}
		b.fcLayers[i] = l
	}

	td := thetaDim(st, inputLen, outputLen, nHarmonics)

	switch st {
	case StackTrend, StackSeasonality:
		// Shared theta dimension for backcast and forecast.
		thetaB, err := newMLPLayer(hiddenDim, td)
		if err != nil {
			return nbeatsBlock{}, fmt.Errorf("theta_b: %w", err)
		}
		b.thetaBLayer = thetaB

		thetaF, err := newMLPLayer(hiddenDim, td)
		if err != nil {
			return nbeatsBlock{}, fmt.Errorf("theta_f: %w", err)
		}
		b.thetaFLayer = thetaF

		// Precompute basis matrices.
		var basisB, basisF *tensor.TensorNumeric[float32]
		var bErr error
		switch st {
		case StackTrend:
			basisB, bErr = polynomialBasis(td, inputLen)
			if bErr != nil {
				return nbeatsBlock{}, bErr
			}
			basisF, bErr = polynomialBasis(td, outputLen)
			if bErr != nil {
				return nbeatsBlock{}, bErr
			}
		case StackSeasonality:
			basisB, bErr = fourierBasis(nHarmonics, inputLen)
			if bErr != nil {
				return nbeatsBlock{}, bErr
			}
			basisF, bErr = fourierBasis(nHarmonics, outputLen)
			if bErr != nil {
				return nbeatsBlock{}, bErr
			}
		}
		b.backcastBasis = basisB
		b.forecastBasis = basisF

	case StackGeneric:
		// Generic: separate theta for backcast (inputLen) and forecast (outputLen).
		thetaB, err := newMLPLayer(hiddenDim, inputLen)
		if err != nil {
			return nbeatsBlock{}, fmt.Errorf("theta_b: %w", err)
		}
		b.thetaBLayer = thetaB

		thetaF, err := newMLPLayer(hiddenDim, outputLen)
		if err != nil {
			return nbeatsBlock{}, fmt.Errorf("theta_f: %w", err)
		}
		b.thetaFLayer = thetaF
	}

	return b, nil
}

// polynomialBasis creates a polynomial basis matrix of shape [degree, length].
// Row i contains (t/T)^i for t = 0, 1, ..., length-1, where T = length-1.
func polynomialBasis(degree, length int) (*tensor.TensorNumeric[float32], error) {
	data := make([]float32, degree*length)
	T := float64(length - 1)
	if T == 0 {
		T = 1
	}
	for i := 0; i < degree; i++ {
		for t := 0; t < length; t++ {
			normalized := float64(t) / T
			data[i*length+t] = float32(math.Pow(normalized, float64(i)))
		}
	}
	return tensor.New[float32]([]int{degree, length}, data)
}

// fourierBasis creates a Fourier basis matrix of shape [2*nHarmonics, length].
// Rows 0..nHarmonics-1 are cos(2*pi*k*t/T), rows nHarmonics..2*nHarmonics-1
// are sin(2*pi*k*t/T), for k = 1, ..., nHarmonics.
func fourierBasis(nHarmonics, length int) (*tensor.TensorNumeric[float32], error) {
	data := make([]float32, 2*nHarmonics*length)
	T := float64(length)
	for k := 0; k < nHarmonics; k++ {
		freq := 2.0 * math.Pi * float64(k+1) / T
		for t := 0; t < length; t++ {
			data[k*length+t] = float32(math.Cos(freq * float64(t)))
			data[(nHarmonics+k)*length+t] = float32(math.Sin(freq * float64(t)))
		}
	}
	return tensor.New[float32]([]int{2 * nHarmonics, length}, data)
}

// newMLPLayer creates a layer with He (Kaiming) weight initialization.
// Weights are stored as [out, in] for compatibility with functional.Linear.
func newMLPLayer(in, out int) (mlpLayer, error) {
	scale := float32(math.Sqrt(2.0 / float64(in)))
	wData := make([]float32, out*in)
	for i := range wData {
		wData[i] = float32(rand.NormFloat64()) * scale
	}
	w, err := tensor.New[float32]([]int{out, in}, wData)
	if err != nil {
		return mlpLayer{}, err
	}

	bData := make([]float32, out)
	b, err := tensor.New[float32]([]int{out}, bData)
	if err != nil {
		return mlpLayer{}, err
	}

	return mlpLayer{weights: w, biases: b}, nil
}

// Forward runs the N-BEATS forward pass.
// Input x has shape [batch, inputLen]. Returns NBEATSOutput with forecast
// of shape [batch, outputLen] and per-stack decomposition.
func (m *NBEATS) Forward(ctx context.Context, x *tensor.TensorNumeric[float32]) (*NBEATSOutput, error) {
	shape := x.Shape()
	if len(shape) != 2 || shape[1] != m.config.InputLength {
		return nil, fmt.Errorf("nbeats: expected input shape [batch, %d], got %v", m.config.InputLength, shape)
	}

	batch := shape[0]

	// Initialize forecast accumulator to zeros.
	forecastData := make([]float32, batch*m.config.OutputLength)
	forecast, err := tensor.New[float32]([]int{batch, m.config.OutputLength}, forecastData)
	if err != nil {
		return nil, err
	}

	residual := x
	stackForecasts := make([]*tensor.TensorNumeric[float32], len(m.stacks))
	stackBackcasts := make([]*tensor.TensorNumeric[float32], len(m.stacks))

	for si, stack := range m.stacks {
		// Accumulate stack-level forecast for decomposition.
		sfData := make([]float32, batch*m.config.OutputLength)
		stackForecast, err := tensor.New[float32]([]int{batch, m.config.OutputLength}, sfData)
		if err != nil {
			return nil, err
		}
		sbData := make([]float32, batch*m.config.InputLength)
		stackBackcast, err := tensor.New[float32]([]int{batch, m.config.InputLength}, sbData)
		if err != nil {
			return nil, err
		}

		for _, block := range stack.blocks {
			backcast, blockForecast, err := m.blockForward(ctx, block, residual)
			if err != nil {
				return nil, fmt.Errorf("nbeats: stack %d: %w", si, err)
			}

			// Double residual stacking: subtract backcast, add forecast.
			residual, err = m.engine.Sub(ctx, residual, backcast)
			if err != nil {
				return nil, err
			}
			forecast, err = m.engine.Add(ctx, forecast, blockForecast)
			if err != nil {
				return nil, err
			}

			stackForecast, err = m.engine.Add(ctx, stackForecast, blockForecast)
			if err != nil {
				return nil, err
			}
			stackBackcast, err = m.engine.Add(ctx, stackBackcast, backcast)
			if err != nil {
				return nil, err
			}
		}

		stackForecasts[si] = stackForecast
		stackBackcasts[si] = stackBackcast
	}

	return &NBEATSOutput{
		Forecast:       forecast,
		StackForecasts: stackForecasts,
		StackBackcasts: stackBackcasts,
	}, nil
}

// blockForward runs a single N-BEATS block.
// Returns (backcast [batch, inputLen], forecast [batch, outputLen]).
func (m *NBEATS) blockForward(ctx context.Context, block nbeatsBlock, x *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], error) {
	// FC layers with ReLU activation.
	h := x
	var err error
	for _, l := range block.fcLayers {
		h, err = m.linearForward(ctx, h, l)
		if err != nil {
			return nil, nil, err
		}
		h, err = m.engine.UnaryOp(ctx, h, m.ops.ReLU)
		if err != nil {
			return nil, nil, err
		}
	}

	// Compute theta parameters.
	thetaB, err := m.linearForward(ctx, h, block.thetaBLayer)
	if err != nil {
		return nil, nil, fmt.Errorf("theta_b: %w", err)
	}
	thetaF, err := m.linearForward(ctx, h, block.thetaFLayer)
	if err != nil {
		return nil, nil, fmt.Errorf("theta_f: %w", err)
	}

	// Basis expansion.
	var backcast, forecast *tensor.TensorNumeric[float32]
	switch block.stackType {
	case StackTrend, StackSeasonality:
		// backcast = thetaB @ basis_b: [batch, thetaDim] @ [thetaDim, inputLen] -> [batch, inputLen]
		backcast, err = m.engine.MatMul(ctx, thetaB, block.backcastBasis)
		if err != nil {
			return nil, nil, fmt.Errorf("backcast basis: %w", err)
		}
		// forecast = thetaF @ basis_f: [batch, thetaDim] @ [thetaDim, outputLen] -> [batch, outputLen]
		forecast, err = m.engine.MatMul(ctx, thetaF, block.forecastBasis)
		if err != nil {
			return nil, nil, fmt.Errorf("forecast basis: %w", err)
		}

	case StackGeneric:
		// Generic blocks: theta IS the output (direct linear projection).
		backcast = thetaB
		forecast = thetaF
	}

	return backcast, forecast, nil
}

// linearForward computes y = x @ W^T + b via functional.Linear.
func (m *NBEATS) linearForward(ctx context.Context, x *tensor.TensorNumeric[float32], l mlpLayer) (*tensor.TensorNumeric[float32], error) {
	return functional.Linear(ctx, m.engine, x, l.weights, l.biases)
}

// FlatParams returns pointers to all trainable parameters in a deterministic
// order: for each stack, for each block: 4 FC layers (weight, bias each),
// then thetaB (weight, bias), then thetaF (weight, bias).
func (m *NBEATS) FlatParams() []*float32 {
	var params []*float32
	for si := range m.stacks {
		for bi := range m.stacks[si].blocks {
			b := &m.stacks[si].blocks[bi]
			for li := range b.fcLayers {
				wData := b.fcLayers[li].weights.Data()
				for i := range wData {
					params = append(params, &wData[i])
				}
				bData := b.fcLayers[li].biases.Data()
				for i := range bData {
					params = append(params, &bData[i])
				}
			}
			wData := b.thetaBLayer.weights.Data()
			for i := range wData {
				params = append(params, &wData[i])
			}
			bData := b.thetaBLayer.biases.Data()
			for i := range bData {
				params = append(params, &bData[i])
			}
			wData = b.thetaFLayer.weights.Data()
			for i := range wData {
				params = append(params, &wData[i])
			}
			bData = b.thetaFLayer.biases.Data()
			for i := range bData {
				params = append(params, &bData[i])
			}
		}
	}
	return params
}

// ForwardBatched runs the N-BEATS forward pass on a 3D input tensor.
// Input x has shape [batch, channels, inputLen]. The channels are averaged
// to produce a [batch, inputLen] tensor which is then passed through the
// standard Forward path. Returns NBEATSOutput with forecast of shape
// [batch, outputLen].
func (m *NBEATS) ForwardBatched(ctx context.Context, x *tensor.TensorNumeric[float32]) (*NBEATSOutput, error) {
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("nbeats: ForwardBatched expects 3D input [batch, channels, inputLen], got shape %v", shape)
	}
	batch, channels, inputLen := shape[0], shape[1], shape[2]
	if inputLen != m.config.InputLength {
		return nil, fmt.Errorf("nbeats: expected inputLen %d, got %d", m.config.InputLength, inputLen)
	}

	// Average across channels: [batch, channels, inputLen] -> [batch, inputLen].
	var flat *tensor.TensorNumeric[float32]
	var err error
	if channels == 1 {
		flat, err = m.engine.Reshape(ctx, x, []int{batch, inputLen})
		if err != nil {
			return nil, fmt.Errorf("nbeats: reshape single channel: %w", err)
		}
	} else {
		flat, err = m.engine.ReduceMean(ctx, x, 1, false)
		if err != nil {
			return nil, fmt.Errorf("nbeats: reduce channels: %w", err)
		}
	}

	return m.Forward(ctx, flat)
}

// Decompose runs forward and returns the per-stack decomposition of the forecast.
// This is useful for interpretable forecasting: separating trend from seasonality.
func (m *NBEATS) Decompose(ctx context.Context, x *tensor.TensorNumeric[float32]) (map[StackType]*tensor.TensorNumeric[float32], error) {
	out, err := m.Forward(ctx, x)
	if err != nil {
		return nil, err
	}

	decomp := make(map[StackType]*tensor.TensorNumeric[float32], len(m.stacks))
	for i, stack := range m.stacks {
		decomp[stack.stackType] = out.StackForecasts[i]
	}
	return decomp, nil
}
