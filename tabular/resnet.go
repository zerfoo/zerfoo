package tabular

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

// NormMode selects the normalization applied after each residual block.
type NormMode int

const (
	// NormLayer applies layer normalization (normalize across features).
	NormLayer NormMode = iota
	// NormBatch applies batch normalization. For single-sample inference
	// this falls back to layer normalization.
	NormBatch
)

// TabResNetConfig holds the configuration for a TabResNet model.
type TabResNetConfig struct {
	InputDim    int
	OutputDim   int // number of output classes (default 3: Long/Short/Flat)
	HiddenDims  []int
	DropoutRate float64
	Activation  Activation
	Norm        NormMode
}

// resBlock holds a single residual block: linear + optional projection shortcut.
type resBlock struct {
	linear   mlpLayer
	shortcut *mlpLayer // non-nil when input/output dims differ
	// Layer norm parameters (learnable scale and shift).
	gamma *tensor.TensorNumeric[float32]
	beta  *tensor.TensorNumeric[float32]
}

// TabResNet is an MLP with skip connections between hidden layers.
// It is a simple but surprisingly strong baseline for tabular data.
type TabResNet struct {
	config TabResNetConfig
	engine compute.Engine[float32]
	ops    numeric.Arithmetic[float32]
	input  mlpLayer   // input projection: input_dim -> first hidden dim
	blocks []resBlock // residual blocks
	head   mlpLayer   // output head: last hidden -> output classes
}

// NewTabResNet creates a new TabResNet model with the given configuration.
func NewTabResNet(config TabResNetConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*TabResNet, error) {
	if config.InputDim <= 0 {
		return nil, fmt.Errorf("tabular: InputDim must be positive, got %d", config.InputDim)
	}
	if len(config.HiddenDims) == 0 {
		return nil, fmt.Errorf("tabular: HiddenDims must have at least one element")
	}
	for i, h := range config.HiddenDims {
		if h <= 0 {
			return nil, fmt.Errorf("tabular: HiddenDims[%d] must be positive, got %d", i, h)
		}
	}
	if config.DropoutRate < 0 || config.DropoutRate >= 1 {
		return nil, fmt.Errorf("tabular: DropoutRate must be in [0, 1), got %f", config.DropoutRate)
	}
	if config.OutputDim <= 0 {
		config.OutputDim = 3
	}

	m := &TabResNet{
		config: config,
		engine: engine,
		ops:    ops,
	}

	// Input projection: input_dim -> first hidden dim.
	inputLayer, err := newMLPLayer(config.InputDim, config.HiddenDims[0])
	if err != nil {
		return nil, fmt.Errorf("tabular: input projection: %w", err)
	}
	m.input = inputLayer

	// Residual blocks between consecutive hidden dims.
	m.blocks = make([]resBlock, len(config.HiddenDims))
	for i := 0; i < len(config.HiddenDims); i++ {
		dim := config.HiddenDims[i]
		var prevDim int
		if i == 0 {
			prevDim = config.HiddenDims[0] // after input projection
		} else {
			prevDim = config.HiddenDims[i-1]
		}

		linear, err := newMLPLayer(prevDim, dim)
		if err != nil {
			return nil, fmt.Errorf("tabular: block %d: %w", i, err)
		}

		block := resBlock{linear: linear}

		// If dimensions differ, add a projection shortcut.
		if prevDim != dim {
			sc, err := newMLPLayer(prevDim, dim)
			if err != nil {
				return nil, fmt.Errorf("tabular: block %d shortcut: %w", i, err)
			}
			block.shortcut = &sc
		}

		// Layer norm parameters: gamma (scale) = 1, beta (shift) = 0.
		gammaData := make([]float32, dim)
		for j := range gammaData {
			gammaData[j] = 1.0
		}
		block.gamma, err = tensor.New[float32]([]int{1, dim}, gammaData)
		if err != nil {
			return nil, fmt.Errorf("tabular: block %d gamma: %w", i, err)
		}
		betaData := make([]float32, dim)
		block.beta, err = tensor.New[float32]([]int{1, dim}, betaData)
		if err != nil {
			return nil, fmt.Errorf("tabular: block %d beta: %w", i, err)
		}

		m.blocks[i] = block
	}

	// Output head: last hidden dim -> output classes.
	head, err := newMLPLayer(config.HiddenDims[len(config.HiddenDims)-1], config.OutputDim)
	if err != nil {
		return nil, fmt.Errorf("tabular: output head: %w", err)
	}
	m.head = head

	return m, nil
}

// Predict runs inference on the given features and returns a Direction and
// confidence score.
func (m *TabResNet) Predict(features []float64) (Direction, float64, error) {
	if len(features) != m.config.InputDim {
		return Flat, 0, fmt.Errorf("tabular: expected %d features, got %d", m.config.InputDim, len(features))
	}

	ctx := context.Background()

	// Convert float64 features to float32 tensor [1, InputDim].
	f32 := make([]float32, len(features))
	for i, v := range features {
		f32[i] = float32(v)
	}
	input, err := tensor.New[float32]([]int{1, m.config.InputDim}, f32)
	if err != nil {
		return Flat, 0, err
	}

	// Forward pass.
	logits, err := m.forward(ctx, input)
	if err != nil {
		return Flat, 0, err
	}

	// Softmax to get probabilities.
	probs, err := m.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return Flat, 0, err
	}

	probData := probs.Data()
	dir, conf := argmax(probData)
	return dir, conf, nil
}

// forward runs the full forward pass: input projection -> residual blocks -> output head.
func (m *TabResNet) forward(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Input projection.
	x, err := m.linearForward(ctx, input, m.input)
	if err != nil {
		return nil, err
	}
	x, err = m.applyActivation(ctx, x)
	if err != nil {
		return nil, err
	}

	// Residual blocks.
	for _, block := range m.blocks {
		residual := x

		// Linear transform + activation.
		h, err := m.linearForward(ctx, x, block.linear)
		if err != nil {
			return nil, err
		}
		h, err = m.applyActivation(ctx, h)
		if err != nil {
			return nil, err
		}

		// Apply shortcut projection if dimensions differ.
		if block.shortcut != nil {
			residual, err = m.linearForward(ctx, residual, *block.shortcut)
			if err != nil {
				return nil, err
			}
		}

		// Skip connection: h = h + residual.
		x, err = m.engine.Add(ctx, h, residual)
		if err != nil {
			return nil, err
		}

		// Layer normalization.
		x, err = m.layerNorm(ctx, x, block.gamma, block.beta)
		if err != nil {
			return nil, err
		}
	}

	// Output head (no activation — raw logits).
	return m.linearForward(ctx, x, m.head)
}

// layerNorm applies layer normalization via the functional package.
func (m *TabResNet) layerNorm(ctx context.Context, x, gamma, beta *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return functional.LayerNorm(ctx, m.engine, x, gamma, beta, 1e-5)
}

// linearForward computes a linear transformation via functional.Linear.
// mlpLayer stores weights as [in, out], so we transpose to [out, in] for
// functional.Linear which expects [out_features, in_features].
func (m *TabResNet) linearForward(ctx context.Context, x *tensor.TensorNumeric[float32], l mlpLayer) (*tensor.TensorNumeric[float32], error) {
	wT, err := m.engine.Transpose(ctx, l.weights, []int{1, 0})
	if err != nil {
		return nil, err
	}
	return functional.Linear(ctx, m.engine, x, wT, l.biases)
}

// applyActivation applies the configured activation function.
func (m *TabResNet) applyActivation(ctx context.Context, x *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	switch m.config.Activation {
	case ActivationGELU:
		return functional.GELU(ctx, m.engine, m.ops, x)
	default:
		return functional.ReLU(ctx, m.engine, m.ops, x)
	}
}

// TabResNetParams returns model parameters as mlpLayer slices for training integration.
// Returns input layer, block layers (linear + optional shortcut), and head.
func (m *TabResNet) TabResNetParams() (inputLayer mlpLayer, blocks []resBlock, head mlpLayer) {
	return m.input, m.blocks, m.head
}

// initResNetWeights reinitializes all weights with He/Kaiming initialization.
// This is useful when you want deterministic weight initialization for testing.
func initResNetWeights(m *TabResNet, seed uint64) {
	src := rand.New(rand.NewPCG(seed, 0))
	initLayer := func(l *mlpLayer, fanIn int) {
		scale := float32(math.Sqrt(2.0 / float64(fanIn)))
		wData := l.weights.Data()
		for i := range wData {
			wData[i] = float32(src.NormFloat64()) * scale
		}
		bData := l.biases.Data()
		for i := range bData {
			bData[i] = 0
		}
	}

	initLayer(&m.input, m.config.InputDim)
	for i := range m.blocks {
		var prevDim int
		if i == 0 {
			prevDim = m.config.HiddenDims[0]
		} else {
			prevDim = m.config.HiddenDims[i-1]
		}
		initLayer(&m.blocks[i].linear, prevDim)
		if m.blocks[i].shortcut != nil {
			initLayer(m.blocks[i].shortcut, prevDim)
		}
	}
	lastHidden := m.config.HiddenDims[len(m.config.HiddenDims)-1]
	initLayer(&m.head, lastHidden)
}
