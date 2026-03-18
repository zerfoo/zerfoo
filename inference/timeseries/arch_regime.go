package timeseries

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// RegimeConfig holds configuration for the regime detection model.
type RegimeConfig struct {
	InputDim   int // number of input features per timestep
	HiddenDim  int // GRU hidden size (e.g., 128)
	NumLayers  int // GRU depth (e.g., 2)
	SeqLen     int // input sequence length (e.g., 60 days)
	NumClasses int // number of regime classes (default: 4)
}

// RegimeDetector is a GRU-based regime classification model.
// It processes sequential input through stacked GRU layers and classifies
// the final hidden state into one of NumClasses regimes
// (bull/bear/sideways/volatile).
type RegimeDetector[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	gruLayers  []*gruLayer[T]
	classifier *classifierHead[T]
	cfg        RegimeConfig
}

// gruLayer implements a single GRU cell applied across a sequence.
type gruLayer[T tensor.Numeric] struct {
	name   string
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	// Gate weights: update (z), reset (r), candidate (n)
	wz *graph.Parameter[T] // [inputDim, hiddenDim]
	uz *graph.Parameter[T] // [hiddenDim, hiddenDim]
	wr *graph.Parameter[T] // [inputDim, hiddenDim]
	ur *graph.Parameter[T] // [hiddenDim, hiddenDim]
	wn *graph.Parameter[T] // [inputDim, hiddenDim]
	un *graph.Parameter[T] // [hiddenDim, hiddenDim]

	inputDim  int
	hiddenDim int
}

// classifierHead is a linear layer followed by softmax.
type classifierHead[T tensor.Numeric] struct {
	name    string
	engine  compute.Engine[T]
	weights *graph.Parameter[T] // [hiddenDim, numClasses]
	bias    *graph.Parameter[T] // [1, numClasses]
}

func newParam[T tensor.Numeric](name string, rows, cols int) (*graph.Parameter[T], error) {
	data := make([]T, rows*cols)
	for i := range data {
		data[i] = T(rand.Float64()*0.1 - 0.05)
	}
	t, err := tensor.New[T]([]int{rows, cols}, data)
	if err != nil {
		return nil, err
	}
	return graph.NewParameter[T](name, t, tensor.New[T])
}

func newGRULayer[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputDim, hiddenDim int,
) (*gruLayer[T], error) {
	wz, err := newParam[T](name+"_wz", inputDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	uz, err := newParam[T](name+"_uz", hiddenDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	wr, err := newParam[T](name+"_wr", inputDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	ur, err := newParam[T](name+"_ur", hiddenDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	wn, err := newParam[T](name+"_wn", inputDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	un, err := newParam[T](name+"_un", hiddenDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	return &gruLayer[T]{
		name:      name,
		engine:    engine,
		ops:       ops,
		wz:        wz,
		uz:        uz,
		wr:        wr,
		ur:        ur,
		wn:        wn,
		un:        un,
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
	}, nil
}

// forward processes a sequence through the GRU layer and returns the last hidden state.
// input: [batch, seqLen, inputDim], output: [batch, hiddenDim]
func (g *gruLayer[T]) forward(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// Initialize hidden state to zeros: [batch, hiddenDim]
	h, err := tensor.New[T]([]int{batch, g.hiddenDim}, make([]T, batch*g.hiddenDim))
	if err != nil {
		return nil, err
	}

	for t := 0; t < seqLen; t++ {
		// Extract timestep t: [batch, inputDim]
		xt, err := extractTimestep(input, t, batch, g.inputDim)
		if err != nil {
			return nil, fmt.Errorf("extract timestep %d: %w", t, err)
		}

		// Update gate: z = sigmoid(xt @ Wz + h @ Uz)
		xWz, err := g.engine.MatMul(ctx, xt, g.wz.Value)
		if err != nil {
			return nil, err
		}
		hUz, err := g.engine.MatMul(ctx, h, g.uz.Value)
		if err != nil {
			return nil, err
		}
		zPre, err := g.engine.Add(ctx, xWz, hUz)
		if err != nil {
			return nil, err
		}
		z, err := sigmoid(ctx, g.engine, g.ops, zPre)
		if err != nil {
			return nil, err
		}

		// Reset gate: r = sigmoid(xt @ Wr + h @ Ur)
		xWr, err := g.engine.MatMul(ctx, xt, g.wr.Value)
		if err != nil {
			return nil, err
		}
		hUr, err := g.engine.MatMul(ctx, h, g.ur.Value)
		if err != nil {
			return nil, err
		}
		rPre, err := g.engine.Add(ctx, xWr, hUr)
		if err != nil {
			return nil, err
		}
		r, err := sigmoid(ctx, g.engine, g.ops, rPre)
		if err != nil {
			return nil, err
		}

		// Candidate: n = tanh(xt @ Wn + (r * h) @ Un)
		xWn, err := g.engine.MatMul(ctx, xt, g.wn.Value)
		if err != nil {
			return nil, err
		}
		rh, err := g.engine.Mul(ctx, r, h)
		if err != nil {
			return nil, err
		}
		rhUn, err := g.engine.MatMul(ctx, rh, g.un.Value)
		if err != nil {
			return nil, err
		}
		nPre, err := g.engine.Add(ctx, xWn, rhUn)
		if err != nil {
			return nil, err
		}
		n, err := g.engine.UnaryOp(ctx, nPre, g.ops.Tanh)
		if err != nil {
			return nil, err
		}

		// h = (1 - z) * n + z * h_prev
		ones, err := tensor.New[T](z.Shape(), makeOnes[T](batch*g.hiddenDim))
		if err != nil {
			return nil, err
		}
		oneMinusZ, err := g.engine.Sub(ctx, ones, z)
		if err != nil {
			return nil, err
		}
		term1, err := g.engine.Mul(ctx, oneMinusZ, n)
		if err != nil {
			return nil, err
		}
		term2, err := g.engine.Mul(ctx, z, h)
		if err != nil {
			return nil, err
		}
		h, err = g.engine.Add(ctx, term1, term2)
		if err != nil {
			return nil, err
		}
	}

	return h, nil
}

func (g *gruLayer[T]) parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{g.wz, g.uz, g.wr, g.ur, g.wn, g.un}
}

func newClassifierHead[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	hiddenDim, numClasses int,
) (*classifierHead[T], error) {
	weights, err := newParam[T](name+"_weights", hiddenDim, numClasses)
	if err != nil {
		return nil, err
	}
	biasData := make([]T, numClasses)
	biasTensor, err := tensor.New[T]([]int{1, numClasses}, biasData)
	if err != nil {
		return nil, err
	}
	bias, err := graph.NewParameter[T](name+"_bias", biasTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}
	return &classifierHead[T]{
		name:    name,
		engine:  engine,
		weights: weights,
		bias:    bias,
	}, nil
}

// BuildRegimeDetector constructs a regime detection model.
// The model consists of stacked GRU layers followed by a linear classifier
// with softmax output producing 4-class probabilities
// (bull/bear/sideways/volatile).
func BuildRegimeDetector[T tensor.Numeric](
	cfg RegimeConfig,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*RegimeDetector[T], error) {
	if cfg.InputDim <= 0 {
		return nil, fmt.Errorf("InputDim must be positive, got %d", cfg.InputDim)
	}
	if cfg.HiddenDim <= 0 {
		return nil, fmt.Errorf("HiddenDim must be positive, got %d", cfg.HiddenDim)
	}
	if cfg.NumLayers <= 0 {
		return nil, fmt.Errorf("NumLayers must be positive, got %d", cfg.NumLayers)
	}
	if cfg.SeqLen <= 0 {
		return nil, fmt.Errorf("SeqLen must be positive, got %d", cfg.SeqLen)
	}
	if cfg.NumClasses <= 0 {
		cfg.NumClasses = 4
	}

	gruLayers := make([]*gruLayer[T], cfg.NumLayers)
	for i := range gruLayers {
		inDim := cfg.InputDim
		if i > 0 {
			inDim = cfg.HiddenDim
		}
		layer, err := newGRULayer[T](
			fmt.Sprintf("regime_gru_%d", i),
			engine, ops,
			inDim, cfg.HiddenDim,
		)
		if err != nil {
			return nil, fmt.Errorf("create GRU layer %d: %w", i, err)
		}
		gruLayers[i] = layer
	}

	classifier, err := newClassifierHead[T](
		"regime_classifier",
		engine,
		cfg.HiddenDim, cfg.NumClasses,
	)
	if err != nil {
		return nil, fmt.Errorf("create classifier: %w", err)
	}

	return &RegimeDetector[T]{
		engine:     engine,
		ops:        ops,
		gruLayers:  gruLayers,
		classifier: classifier,
		cfg:        cfg,
	}, nil
}

// Forward runs the regime detection model.
// Input shape: [batch, seqLen, inputDim]
// Output shape: [batch, numClasses] with softmax probabilities.
func (rd *RegimeDetector[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RegimeDetector requires exactly one input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("RegimeDetector input must be 3D [batch, seqLen, inputDim], got shape %v", shape)
	}

	// Process through stacked GRU layers.
	// First layer takes [batch, seqLen, inputDim].
	// Subsequent layers take [batch, seqLen, hiddenDim] —
	// we expand the hidden output back to 3D for chaining.
	h := x
	for i, gru := range rd.gruLayers {
		hidden, err := gru.forward(ctx, h)
		if err != nil {
			return nil, fmt.Errorf("GRU layer %d: %w", i, err)
		}
		if i < len(rd.gruLayers)-1 {
			// Expand [batch, hiddenDim] -> [batch, seqLen, hiddenDim]
			// by repeating the hidden state across the sequence dimension.
			batch := hidden.Shape()[0]
			seqLen := h.Shape()[1]
			expanded, err := expandHidden(hidden, batch, seqLen, rd.cfg.HiddenDim)
			if err != nil {
				return nil, fmt.Errorf("expand hidden layer %d: %w", i, err)
			}
			h = expanded
		} else {
			// Last layer: use the final hidden state [batch, hiddenDim]
			h = hidden
		}
	}

	// Classifier: [batch, hiddenDim] @ [hiddenDim, numClasses] + bias -> [batch, numClasses]
	logits, err := rd.engine.MatMul(ctx, h, rd.classifier.weights.Value)
	if err != nil {
		return nil, fmt.Errorf("classifier matmul: %w", err)
	}

	// Broadcast add bias
	logits, err = rd.engine.Add(ctx, logits, rd.classifier.bias.Value)
	if err != nil {
		return nil, fmt.Errorf("classifier bias: %w", err)
	}

	// Softmax along the class dimension (axis -1)
	probs, err := rd.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return nil, fmt.Errorf("softmax: %w", err)
	}

	return probs, nil
}

// OpType returns the operation type.
func (rd *RegimeDetector[T]) OpType() string { return "RegimeDetector" }

// Attributes returns the model configuration.
func (rd *RegimeDetector[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_dim":   rd.cfg.InputDim,
		"hidden_dim":  rd.cfg.HiddenDim,
		"num_layers":  rd.cfg.NumLayers,
		"seq_len":     rd.cfg.SeqLen,
		"num_classes": rd.cfg.NumClasses,
	}
}

// OutputShape returns [batch, numClasses].
func (rd *RegimeDetector[T]) OutputShape() []int {
	return []int{-1, rd.cfg.NumClasses}
}

// Parameters returns all trainable parameters.
func (rd *RegimeDetector[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	for _, gru := range rd.gruLayers {
		params = append(params, gru.parameters()...)
	}
	params = append(params, rd.classifier.weights, rd.classifier.bias)
	return params
}

// Backward is not implemented for inference-only use.
func (rd *RegimeDetector[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// sigmoid computes sigmoid(x) = exp(x) / (1 + exp(x)) using composed engine
// primitives (same approach as layers/activations.Sigmoid).
func sigmoid[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T], x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	expX, err := engine.Exp(ctx, x)
	if err != nil {
		return nil, err
	}
	onePlusExpX, err := engine.AddScalar(ctx, expX, ops.One())
	if err != nil {
		return nil, err
	}
	return engine.Div(ctx, expX, onePlusExpX)
}

// extractTimestep extracts a single timestep from a 3D tensor.
// input: [batch, seqLen, dim], output: [batch, dim]
func extractTimestep[T tensor.Numeric](input *tensor.TensorNumeric[T], t, batch, dim int) (*tensor.TensorNumeric[T], error) {
	data := input.Data()
	seqLen := input.Shape()[1]
	out := make([]T, batch*dim)
	for b := 0; b < batch; b++ {
		srcOff := b*seqLen*dim + t*dim
		copy(out[b*dim:(b+1)*dim], data[srcOff:srcOff+dim])
	}
	return tensor.New[T]([]int{batch, dim}, out)
}

// expandHidden repeats a [batch, hiddenDim] tensor across seqLen to produce
// [batch, seqLen, hiddenDim].
func expandHidden[T tensor.Numeric](hidden *tensor.TensorNumeric[T], batch, seqLen, hiddenDim int) (*tensor.TensorNumeric[T], error) {
	data := hidden.Data()
	out := make([]T, batch*seqLen*hiddenDim)
	for b := 0; b < batch; b++ {
		row := data[b*hiddenDim : (b+1)*hiddenDim]
		for s := 0; s < seqLen; s++ {
			copy(out[b*seqLen*hiddenDim+s*hiddenDim:], row)
		}
	}
	return tensor.New[T]([]int{batch, seqLen, hiddenDim}, out)
}

// makeOnes creates a slice of ones with the given length.
func makeOnes[T tensor.Numeric](n int) []T {
	data := make([]T, n)
	for i := range data {
		data[i] = 1
	}
	return data
}

// Statically assert that RegimeDetector implements graph.Node.
var _ graph.Node[float32] = (*RegimeDetector[float32])(nil)
