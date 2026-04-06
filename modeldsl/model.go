package modeldsl

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/layers/core"
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
		return &rmsnormLayerT{epsilon: eps}, nil
	case LayerSiLU:
		return &siluLayerT{}, nil
	case LayerSoftmax:
		return &softmaxLayerT{}, nil
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

// dslEngine is a package-level CPU engine for float64, used by linearLayer
// to delegate matrix multiplication to layers/core.Linear.
var dslEngine = compute.NewCPUEngine[float64](numeric.Float64Ops{})

// linearLayer implements a dense linear transformation: y = xW + b,
// delegating the matmul to layers/core.Linear.
type linearLayer struct {
	linear *core.Linear[float64]
	bias   []float64 // [outDim]
	inDim  int
	outDim int
	// weights stores the raw weight data for use by linearLayerT (training).
	weights []float64 // [inDim * outDim], row-major
}

func newLinearLayer(inDim, outDim int) *linearLayer {
	// Xavier initialization.
	scale := math.Sqrt(2.0 / float64(inDim+outDim))
	weights := make([]float64, inDim*outDim)
	for i := range weights {
		weights[i] = rand.NormFloat64() * scale
	}
	bias := make([]float64, outDim)

	// Create the core.Linear layer using the initialized weights.
	lin, err := core.NewLinear[float64]("dsl_linear", dslEngine, dslEngine.Ops(), inDim, outDim)
	if err != nil {
		// NewLinear only errors on empty name or non-positive dims, which we control.
		panic(fmt.Sprintf("modeldsl: newLinearLayer: %v", err))
	}
	// Overwrite the random weights from core.Linear with our Xavier-initialized weights.
	copy(lin.Parameters()[0].Value.Data(), weights)

	return &linearLayer{linear: lin, bias: bias, inDim: inDim, outDim: outDim, weights: weights}
}

func (l *linearLayer) forward(input []float64) ([]float64, error) {
	if len(input) != l.inDim {
		return nil, fmt.Errorf("linear: expected %d inputs, got %d", l.inDim, len(input))
	}

	// Wrap input as [1, inDim] tensor for core.Linear.Forward.
	inputT, err := tensor.New[float64]([]int{1, l.inDim}, input)
	if err != nil {
		return nil, err
	}

	outT, err := l.linear.Forward(context.Background(), inputT)
	if err != nil {
		return nil, err
	}

	// Add bias.
	out := outT.Data()
	result := make([]float64, l.outDim)
	for j := range result {
		result[j] = out[j] + l.bias[j]
	}
	return result, nil
}

// NOTE: Element-wise layers (rmsnorm, silu, softmax) operate on raw []float64
// slices rather than tensors. The layers/activations/ package provides
// tensor-based equivalents via compute.Engine[T], but the DSL intentionally
// uses slice-based ops for those because its entire pipeline (forward, backward,
// parameter updates) operates on []float64. The linear layer delegates to
// layers/core.Linear for the matrix multiplication.
//
// The trainable variants (rmsnormLayerT, siluLayerT, softmaxLayerT) in
// train.go serve as the single implementations for both inference and
// training, since they implement the execLayer interface and only cache
// state needed for backward when forward is called.

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
