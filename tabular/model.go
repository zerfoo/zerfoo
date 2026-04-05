package tabular

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/layers/functional"
)

// Direction represents a trading signal direction.
type Direction int

const (
	// Long indicates a buy signal.
	Long Direction = iota
	// Short indicates a sell signal.
	Short
	// Flat indicates no signal.
	Flat
)

// String returns the string representation of a Direction.
func (d Direction) String() string {
	switch d {
	case Long:
		return "Long"
	case Short:
		return "Short"
	case Flat:
		return "Flat"
	default:
		return fmt.Sprintf("Direction(%d)", int(d))
	}
}

// Activation selects the activation function used between hidden layers.
type Activation int

const (
	// ActivationReLU uses the ReLU activation function.
	ActivationReLU Activation = iota
	// ActivationGELU uses the GELU activation function.
	ActivationGELU
)

// ModelConfig holds the configuration for a tabular Model.
type ModelConfig struct {
	InputDim    int
	HiddenDims  []int
	DropoutRate float64
	Activation  Activation
}

// mlpLayer holds a single linear layer's weights and biases.
type mlpLayer struct {
	weights *tensor.TensorNumeric[float32]
	biases  *tensor.TensorNumeric[float32]
}

// Model is a configurable MLP for tabular prediction built on ztensor.
type Model struct {
	config ModelConfig
	engine compute.Engine[float32]
	ops    numeric.Arithmetic[float32]
	layers []mlpLayer
	head   mlpLayer // output head: last hidden -> 3 classes
}

// NewModel creates a new tabular Model with the given configuration.
func NewModel(config ModelConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Model, error) {
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

	m := &Model{
		config: config,
		engine: engine,
		ops:    ops,
	}

	// Build hidden layers.
	dims := append([]int{config.InputDim}, config.HiddenDims...)
	m.layers = make([]mlpLayer, len(config.HiddenDims))
	for i := 0; i < len(config.HiddenDims); i++ {
		l, err := newMLPLayer(dims[i], dims[i+1])
		if err != nil {
			return nil, fmt.Errorf("tabular: layer %d: %w", i, err)
		}
		m.layers[i] = l
	}

	// Output head: last hidden dim -> 3 classes (Long, Short, Flat).
	head, err := newMLPLayer(config.HiddenDims[len(config.HiddenDims)-1], 3)
	if err != nil {
		return nil, fmt.Errorf("tabular: output head: %w", err)
	}
	m.head = head

	return m, nil
}

// newMLPLayer creates a layer with He (Kaiming) weight initialization,
// which is better suited for ReLU networks.
func newMLPLayer(in, out int) (mlpLayer, error) {
	scale := float32(math.Sqrt(2.0 / float64(in)))
	wData := make([]float32, in*out)
	for i := range wData {
		wData[i] = float32(rand.NormFloat64()) * scale
	}
	w, err := tensor.New[float32]([]int{in, out}, wData)
	if err != nil {
		return mlpLayer{}, err
	}

	bData := make([]float32, out)
	b, err := tensor.New[float32]([]int{1, out}, bData)
	if err != nil {
		return mlpLayer{}, err
	}

	return mlpLayer{weights: w, biases: b}, nil
}

// Predict runs inference on the given features and returns a Direction and
// confidence score. The features slice must have length equal to InputDim.
func (m *Model) Predict(features []float64) (Direction, float64, error) {
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

	// Forward through hidden layers.
	x := input
	for _, l := range m.layers {
		x, err = m.linearForward(ctx, x, l)
		if err != nil {
			return Flat, 0, err
		}
		x, err = m.applyActivation(ctx, x)
		if err != nil {
			return Flat, 0, err
		}
	}

	// Output head (no activation — raw logits).
	logits, err := m.linearForward(ctx, x, m.head)
	if err != nil {
		return Flat, 0, err
	}

	// Softmax to get probabilities.
	probs, err := m.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return Flat, 0, err
	}

	// Find argmax and confidence.
	probData := probs.Data()
	dir, conf := argmax(probData)

	return dir, conf, nil
}

// linearForward computes a linear transformation via functional.Linear.
// Weights are stored as [in, out] so we transpose to [out, in] for the
// canonical functional.Linear which computes x @ W^T + b.
func (m *Model) linearForward(ctx context.Context, x *tensor.TensorNumeric[float32], l mlpLayer) (*tensor.TensorNumeric[float32], error) {
	wT, err := m.engine.Transpose(ctx, l.weights, []int{1, 0})
	if err != nil {
		return nil, err
	}
	return functional.Linear(ctx, m.engine, x, wT, l.biases)
}

// applyActivation applies the configured activation function via the engine.
func (m *Model) applyActivation(ctx context.Context, x *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	switch m.config.Activation {
	case ActivationReLU:
		return functional.ReLU(ctx, m.engine, m.ops, x)
	case ActivationGELU:
		return functional.GELU(ctx, m.engine, m.ops, x)
	default:
		return functional.ReLU(ctx, m.engine, m.ops, x)
	}
}

// argmax returns the Direction corresponding to the highest probability
// and the confidence (probability value).
func argmax(probs []float32) (Direction, float64) {
	if len(probs) < 3 {
		return Flat, 0
	}
	best := 0
	for i := 1; i < 3; i++ {
		if probs[i] > probs[best] {
			best = i
		}
	}
	return Direction(best), float64(probs[best])
}
