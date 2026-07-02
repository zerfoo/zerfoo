package dsl

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/components"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
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
		return newRMSNormLayer(inDim, eps)
	case LayerSiLU:
		return newSiLULayer(), nil
	case LayerSoftmax:
		return newSoftmaxLayer(), nil
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
	// Xavier initialization via layers/components.XavierInitializer.
	xavier := components.NewXavierInitializer[float64](numeric.Float64Ops{})
	weights, err := xavier.Initialize(inDim, outDim)
	if err != nil {
		panic(fmt.Sprintf("modeldsl: newLinearLayer xavier init: %v", err))
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

	// Add bias via engine to avoid raw .Data() loop.
	biasT, err := tensor.New[float64]([]int{1, l.outDim}, l.bias)
	if err != nil {
		return nil, err
	}
	sumT, err := dslEngine.Add(context.Background(), outT, biasT)
	if err != nil {
		return nil, err
	}
	return sumT.Data(), nil
}

// rmsnormLayer wraps layers/normalization.RMSNorm for inference.
type rmsnormLayer struct {
	norm *normalization.RMSNorm[float64]
	dim  int
}

func newRMSNormLayer(dim int, epsilon float64) (*rmsnormLayer, error) {
	ops := numeric.Float64Ops{}
	norm, err := normalization.NewRMSNorm[float64](
		"dsl_rmsnorm", dslEngine, ops, dim,
		normalization.WithRMSNormEpsilon[float64](epsilon),
	)
	if err != nil {
		return nil, fmt.Errorf("modeldsl: newRMSNormLayer: %w", err)
	}
	return &rmsnormLayer{norm: norm, dim: dim}, nil
}

func (l *rmsnormLayer) forward(input []float64) ([]float64, error) {
	t, err := tensor.New[float64]([]int{1, l.dim}, input)
	if err != nil {
		return nil, err
	}
	out, err := l.norm.Forward(context.Background(), t)
	if err != nil {
		return nil, err
	}
	return out.Data(), nil
}

// siluLayer wraps layers/activations.Sigmoid to compute SiLU: x * sigmoid(x).
type siluLayer struct {
	sigmoid *activations.Sigmoid[float64]
}

func newSiLULayer() *siluLayer {
	return &siluLayer{sigmoid: activations.NewSigmoid[float64](dslEngine, numeric.Float64Ops{})}
}

func (l *siluLayer) forward(input []float64) ([]float64, error) {
	t, err := tensor.New[float64]([]int{len(input)}, input)
	if err != nil {
		return nil, err
	}
	ctx := context.Background()
	sig, err := l.sigmoid.Forward(ctx, t)
	if err != nil {
		return nil, err
	}
	out, err := dslEngine.Mul(ctx, t, sig)
	if err != nil {
		return nil, err
	}
	return out.Data(), nil
}

// softmaxLayer wraps layers/activations.Softmax for inference.
type softmaxLayer struct {
	sm *activations.Softmax[float64]
}

func newSoftmaxLayer() *softmaxLayer {
	return &softmaxLayer{sm: activations.NewSoftmax[float64](dslEngine, -1)}
}

func (l *softmaxLayer) forward(input []float64) ([]float64, error) {
	t, err := tensor.New[float64]([]int{1, len(input)}, input)
	if err != nil {
		return nil, err
	}
	out, err := l.sm.Forward(context.Background(), t)
	if err != nil {
		return nil, err
	}
	return out.Data(), nil
}

// attentionLayer implements self-attention using core.Linear projections and
// layers/attention.ScaledDotProductAttention for the score computation.
type attentionLayer struct {
	numHeads int
	headDim  int
	dim      int
	wq       *linearLayer
	wk       *linearLayer
	wv       *linearLayer
	wo       *linearLayer
	sdpa     *attention.ScaledDotProductAttention[float64]
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
		sdpa:     attention.NewScaledDotProductAttention[float64](dslEngine, headDim),
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

	ctx := context.Background()

	// Reshape Q, K, V from [dim] to [numHeads, 1, headDim] for SDPA.
	qT, err := tensor.New[float64]([]int{a.numHeads, 1, a.headDim}, q)
	if err != nil {
		return nil, err
	}
	kT, err := tensor.New[float64]([]int{a.numHeads, 1, a.headDim}, k)
	if err != nil {
		return nil, err
	}
	vT, err := tensor.New[float64]([]int{a.numHeads, 1, a.headDim}, v)
	if err != nil {
		return nil, err
	}

	// ScaledDotProductAttention: for seq_len=1, output = V.
	outT, err := a.sdpa.Forward(ctx, qT, kT, vT, nil)
	if err != nil {
		return nil, err
	}

	// Reshape back to [dim].
	flat, err := outT.Reshape([]int{a.dim})
	if err != nil {
		return nil, err
	}

	return a.wo.forward(flat.Data())
}
