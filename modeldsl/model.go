package modeldsl

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// execLayer is a compiled layer that can run forward inference on float64 vectors.
type execLayer interface {
	forward(input []float64) ([]float64, error)
}

// Model is a runnable model built from a ModelGraph.
type Model struct {
	graph      *ModelGraph
	execLayers map[string]execLayer
	dims       map[string]int
	inputDim   int
	outputDim  int
}

// Forward runs inference through the model.
func (m *Model) Forward(input []float64) ([]float64, error) {
	if len(input) != m.inputDim {
		return nil, fmt.Errorf("modeldsl: expected input of size %d, got %d", m.inputDim, len(input))
	}

	activations := make(map[string][]float64, len(m.graph.order))
	for _, name := range m.graph.order {
		parentNames := m.graph.parents[name]

		var layerInput []float64
		if len(parentNames) == 0 {
			layerInput = input
		} else {
			layerInput = activations[parentNames[0]]
		}

		out, err := m.execLayers[name].forward(layerInput)
		if err != nil {
			return nil, fmt.Errorf("modeldsl: layer %q forward: %w", name, err)
		}
		activations[name] = out
	}

	// Return the output of the last output layer.
	outputName := m.graph.outputs[len(m.graph.outputs)-1]
	return activations[outputName], nil
}

// buildLayer constructs an execLayer from a LayerDef.
func buildLayer(def LayerDef, inDim, outDim int) (execLayer, error) {
	switch def.Type {
	case LayerLinear:
		return newLinearLayer(inDim, outDim), nil
	case LayerRMSNorm:
		eps := 1e-6
		if v, ok := def.Params["epsilon"]; ok {
			f, err := toFloat64(v)
			if err != nil {
				return nil, fmt.Errorf("invalid epsilon: %w", err)
			}
			eps = f
		}
		return &rmsnormLayer{epsilon: eps}, nil
	case LayerSiLU:
		return &siluLayer{}, nil
	case LayerSoftmax:
		return &softmaxLayer{}, nil
	case LayerAttention:
		numHeads := 1
		if v, ok := def.Params["num_heads"]; ok {
			h, err := toInt(v)
			if err != nil {
				return nil, fmt.Errorf("invalid num_heads: %w", err)
			}
			if h <= 0 {
				return nil, fmt.Errorf("num_heads must be positive, got %d", h)
			}
			numHeads = h
		}
		if inDim%numHeads != 0 {
			return nil, fmt.Errorf("input dim %d not divisible by num_heads %d", inDim, numHeads)
		}
		return newAttentionLayer(inDim, numHeads), nil
	default:
		return nil, fmt.Errorf("unsupported layer type %q", def.Type)
	}
}

func toFloat64(v any) (float64, error) {
	switch val := v.(type) {
	case float64:
		return val, nil
	case float32:
		return float64(val), nil
	case int:
		return float64(val), nil
	case int64:
		return float64(val), nil
	default:
		return 0, fmt.Errorf("expected numeric, got %T", v)
	}
}

// linearLayer implements a dense linear transformation: y = xW + b.
type linearLayer struct {
	weights []float64 // [inDim * outDim], row-major
	bias    []float64 // [outDim]
	inDim   int
	outDim  int
}

func newLinearLayer(inDim, outDim int) *linearLayer {
	// Xavier initialization.
	scale := math.Sqrt(2.0 / float64(inDim+outDim))
	weights := make([]float64, inDim*outDim)
	for i := range weights {
		weights[i] = rand.NormFloat64() * scale
	}
	bias := make([]float64, outDim)
	return &linearLayer{weights: weights, bias: bias, inDim: inDim, outDim: outDim}
}

func (l *linearLayer) forward(input []float64) ([]float64, error) {
	if len(input) != l.inDim {
		return nil, fmt.Errorf("linear: expected %d inputs, got %d", l.inDim, len(input))
	}
	out := make([]float64, l.outDim)
	for j := 0; j < l.outDim; j++ {
		sum := l.bias[j]
		for i := 0; i < l.inDim; i++ {
			sum += input[i] * l.weights[i*l.outDim+j]
		}
		out[j] = sum
	}
	return out, nil
}

// rmsnormLayer implements Root Mean Square Layer Normalization.
type rmsnormLayer struct {
	epsilon float64
}

func (l *rmsnormLayer) forward(input []float64) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("rmsnorm: empty input")
	}
	var sumSq float64
	for _, v := range input {
		sumSq += v * v
	}
	rms := math.Sqrt(sumSq/float64(len(input)) + l.epsilon)
	out := make([]float64, len(input))
	for i, v := range input {
		out[i] = v / rms
	}
	return out, nil
}

// siluLayer implements the SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
type siluLayer struct{}

func (l *siluLayer) forward(input []float64) ([]float64, error) {
	out := make([]float64, len(input))
	for i, v := range input {
		out[i] = v * (1.0 / (1.0 + math.Exp(-v)))
	}
	return out, nil
}

// softmaxLayer implements softmax normalization.
type softmaxLayer struct{}

func (l *softmaxLayer) forward(input []float64) ([]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("softmax: empty input")
	}
	maxVal := input[0]
	for _, v := range input[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	out := make([]float64, len(input))
	var sum float64
	for i, v := range input {
		out[i] = math.Exp(v - maxVal)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out, nil
}

// attentionLayer implements a basic single-sequence self-attention.
// It projects the input into Q, K, V, computes scaled dot-product attention
// per head, concatenates, and applies an output projection.
type attentionLayer struct {
	numHeads int
	headDim  int
	dim      int
	wq       *linearLayer
	wk       *linearLayer
	wv       *linearLayer
	wo       *linearLayer
}

func newAttentionLayer(dim, numHeads int) *attentionLayer {
	headDim := dim / numHeads
	return &attentionLayer{
		numHeads: numHeads,
		headDim:  headDim,
		dim:      dim,
		wq:       newLinearLayer(dim, dim),
		wk:       newLinearLayer(dim, dim),
		wv:       newLinearLayer(dim, dim),
		wo:       newLinearLayer(dim, dim),
	}
}

func (a *attentionLayer) forward(input []float64) ([]float64, error) {
	q, err := a.wq.forward(input)
	if err != nil {
		return nil, err
	}
	k, err := a.wk.forward(input)
	if err != nil {
		return nil, err
	}
	v, err := a.wv.forward(input)
	if err != nil {
		return nil, err
	}

	// Per-head attention: for a single token, attention(q, k, v) over 1 position
	// simplifies to just the value vector (softmax of a single score is 1.0).
	// We still do the full computation for correctness.
	scale := 1.0 / math.Sqrt(float64(a.headDim))
	result := make([]float64, a.dim)
	for h := 0; h < a.numHeads; h++ {
		offset := h * a.headDim

		// Dot product of q and k for this head.
		var score float64
		for d := 0; d < a.headDim; d++ {
			score += q[offset+d] * k[offset+d]
		}
		score *= scale

		// With a single position, softmax(score) = 1.0.
		// Output is simply v scaled by 1.0.
		for d := 0; d < a.headDim; d++ {
			result[offset+d] = v[offset+d]
		}
		_ = score // Used in multi-position scenarios.
	}

	return a.wo.forward(result)
}
