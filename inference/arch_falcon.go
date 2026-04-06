package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func init() {
	RegisterArchitecture("falcon", buildFalconGraph)
}

// buildFalconGraph constructs a computation graph for the Falcon architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// Falcon differs from Llama in three key ways:
//
//  1. Parallel attention: attention and FFN share the same pre-norm input and
//     their outputs are summed together into the residual in a single step:
//     hidden = x + attn(norm(x)) + ffn(norm(x))
//
//  2. LayerNorm (not RMSNorm): standard Layer Normalization with learned
//     gamma (scale) and beta (shift) parameters.
//
//  3. GELU activation in the FFN (not SwiGLU): simple 2-layer dense FFN:
//     ffn(x) = down(gelu(up(x)))
//
// Falcon supports both MQA (NumKVHeads=1) and GQA (NumKVHeads > 1).
//
// The architecture is:
//
//	Embed -> [LayerNorm -> GQA + FFN(GELU) -> Add(x + attn + ffn)] x N -> LayerNorm -> LMHead
func buildFalconGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	layerNormEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		layerNormEps = cfg.RMSNormEps
	}

	pw := newParamWrapper[float32]()

	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	finalNormW, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}
	finalNormB, _ := tl.Optional("model.norm.bias")

	_, isGPUEngine := engine.(compute.WeightUploader)

	transposeWeight := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return transposeWeight2D(engine, isGPUEngine, name, t)
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	embNode := newEmbeddingNode(proxy, embedWeight, 0)
	hidden := builder.AddNode(embNode, input)

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Input LayerNorm (shared by attention and FFN in parallel) ---
		lnW, err := tl.Lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		lnB, _ := tl.Optional(prefix + "input_layernorm.bias")

		lnBParam, err := falconBetaParam(pw, prefix+"input_layernorm.bias", lnB, lnW)
		if err != nil {
			return nil, nil, err
		}
		ln := normalization.NewLayerNormalizationFromParams[float32](
			proxy, ops.FromFloat64(float64(layerNormEps)),
			pw.Wrap(prefix+"input_layernorm.weight", lnW),
			lnBParam,
		)
		// Wrap in residual-caching node so falconParallelAddNode can
		// retrieve the pre-norm input without an extra graph edge.
		inputNorm := &falconResidualLayerNorm[float32]{norm: ln}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Self Attention (GQA / MQA) ---
		qW, err := tl.Lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := tl.Lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := tl.Lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := tl.Lookup(prefix + "self_attn.o_proj.weight")
		if err != nil {
			return nil, nil, err
		}

		qWT, err := transposeWeight(prefix+"self_attn.q_proj.weight", qW)
		if err != nil {
			return nil, nil, err
		}
		kWT, err := transposeWeight(prefix+"self_attn.k_proj.weight", kW)
		if err != nil {
			return nil, nil, err
		}
		vWT, err := transposeWeight(prefix+"self_attn.v_proj.weight", vW)
		if err != nil {
			return nil, nil, err
		}
		oWT, err := transposeWeight(prefix+"self_attn.o_proj.weight", oW)
		if err != nil {
			return nil, nil, err
		}

		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)),
			nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.k_proj.weight", kWT)),
			nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.v_proj.weight", vWT)),
			nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)),
			nil,
		)

		ropeOpts := []embeddings.RotaryPositionalEmbeddingOption{
			embeddings.WithRotaryBase(cfg.RopeTheta),
		}
		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), proxy, headDim, cfg.MaxSeqLen, ropeOpts...,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
			proxy, ops, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads,
			wq, wk, wv, wo, rope, headDim,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d gqa: %w", i, err)
		}
		gqa.LayerIndex = i

		attnOut := builder.AddNode(gqa, normed)

		// --- FFN (GELU, no gating) ---
		// Falcon FFN: ffn(x) = down(gelu(up(x)))
		upW, err := tl.Lookup(prefix + "mlp.dense_h_to_4h.weight")
		if err != nil {
			return nil, nil, err
		}
		downW, err := tl.Lookup(prefix + "mlp.dense_4h_to_h.weight")
		if err != nil {
			return nil, nil, err
		}

		upWT, err := transposeWeight(prefix+"mlp.dense_h_to_4h.weight", upW)
		if err != nil {
			return nil, nil, err
		}
		downWT, err := transposeWeight(prefix+"mlp.dense_4h_to_h.weight", downW)
		if err != nil {
			return nil, nil, err
		}

		// Optional FFN biases (Falcon 7B has bias; 40B+ drops it).
		upB, _ := tl.Optional(prefix + "mlp.dense_h_to_4h.bias")
		downB, _ := tl.Optional(prefix + "mlp.dense_4h_to_h.bias")

		var upBias, downBias *core.Bias[float32]
		if upB != nil {
			upBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"mlp.dense_h_to_4h.bias", upB))
		}
		if downB != nil {
			downBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"mlp.dense_4h_to_h.bias", downB))
		}

		ffnUp := core.NewDenseFromParams(core.NewLinearFromParam(proxy, pw.Wrap(prefix+"mlp.dense_h_to_4h.weight", upWT)), upBias)
		ffnGelu := activations.NewGelu[float32](proxy, ops)
		ffnDown := core.NewDenseFromParams(core.NewLinearFromParam(proxy, pw.Wrap(prefix+"mlp.dense_4h_to_h.weight", downWT)), downBias)

		ffnUpOut := builder.AddNode(ffnUp, normed)
		ffnGeluOut := builder.AddNode(ffnGelu, ffnUpOut)
		ffnOut := builder.AddNode(ffnDown, ffnGeluOut)

		// --- Parallel Residual Add: hidden = residual + attn(norm(x)) + ffn(norm(x)) ---
		// The residual is retrieved from inputNorm's cached pre-norm input.
		parallelAdd := &falconParallelAddNode[float32]{
			engine: proxy,
			norm:   inputNorm,
		}
		hidden = builder.AddNode(parallelAdd, attnOut, ffnOut)
	}

	// --- Final LayerNorm ---
	finalBetaParam, err := falconBetaParam(pw, "model.norm.bias", finalNormB, finalNormW)
	if err != nil {
		return nil, nil, err
	}
	finalNorm := normalization.NewLayerNormalizationFromParams[float32](
		proxy, ops.FromFloat64(float64(layerNormEps)),
		pw.Wrap("model.norm.weight", finalNormW),
		finalBetaParam,
	)
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	if s := lmHeadWeight.GetStorage(); s != nil {
		if qs, ok := any(s).(*tensor.Q8Storage); ok {
			f32 := make([]float32, qs.Len())
			qs.Dequantize(f32)
			q4 := tensor.QuantizeQ4(f32)
			lmHeadWeight, err = tensor.NewWithStorage[float32](lmHeadWeight.Shape(), q4)
			if err != nil {
				return nil, nil, fmt.Errorf("transpose lm_head weight: %w", err)
			}
		}
	}
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// falconBetaParam returns a beta (bias) parameter for LayerNorm. If the model
// provides a bias tensor, it wraps it in a parameter. Otherwise it creates a
// zero-filled tensor matching gamma's shape, since
// normalization.LayerNormalization always applies beta.
func falconBetaParam(
	pw paramWrapper[float32],
	name string,
	bias *tensor.TensorNumeric[float32],
	gamma *tensor.TensorNumeric[float32],
) (*graph.Parameter[float32], error) {
	if bias != nil {
		return pw.Wrap(name, bias), nil
	}
	zeroBeta, err := tensor.New[float32](gamma.Shape(), nil)
	if err != nil {
		return nil, fmt.Errorf("create zero beta for %s: %w", name, err)
	}
	return pw.Wrap(name, zeroBeta), nil
}

// falconResidualLayerNorm wraps normalization.LayerNormalization and caches the
// pre-norm input (the residual) so that falconParallelAddNode can retrieve it
// without an extra graph edge. This is required by Falcon's parallel attention
// pattern: hidden = residual + attn(norm(x)) + ffn(norm(x)).
type falconResidualLayerNorm[T tensor.Numeric] struct {
	norm     *normalization.LayerNormalization[T]
	residual *tensor.TensorNumeric[T]
}

func (n *falconResidualLayerNorm[T]) OpType() string              { return "FalconLayerNorm" }
func (n *falconResidualLayerNorm[T]) Attributes() map[string]any  { return n.norm.Attributes() }
func (n *falconResidualLayerNorm[T]) OutputShape() []int           { return n.norm.OutputShape() }
func (n *falconResidualLayerNorm[T]) Parameters() []*graph.Parameter[T] { return n.norm.Parameters() }

func (n *falconResidualLayerNorm[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FalconLayerNorm: expected 1 input, got %d", len(inputs))
	}
	n.residual = inputs[0]
	return n.norm.Forward(ctx, inputs...)
}

func (n *falconResidualLayerNorm[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return n.norm.Backward(ctx, mode, dOut, inputs...)
}

// Residual returns the pre-norm input cached during the most recent Forward call.
func (n *falconResidualLayerNorm[T]) Residual() *tensor.TensorNumeric[T] {
	return n.residual
}

// falconParallelAddNode implements Falcon's parallel residual update:
//
//	hidden = residual + attnOut + ffnOut
//
// The residual is retrieved from the layer's falconLayerNormNode which caches
// its input (the pre-norm hidden state) during Forward.
type falconParallelAddNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	norm   *falconResidualLayerNorm[T] // supplies the cached residual
}

func (n *falconParallelAddNode[T]) OpType() string                  { return "FalconParallelAdd" }
func (n *falconParallelAddNode[T]) Attributes() map[string]any       { return nil }
func (n *falconParallelAddNode[T]) OutputShape() []int               { return nil }
func (n *falconParallelAddNode[T]) Parameters() []*graph.Parameter[T] { return nil }

// Forward adds attnOut and ffnOut to the residual stored in the layer norm node.
// inputs[0] = attnOut, inputs[1] = ffnOut
func (n *falconParallelAddNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("FalconParallelAdd: expected 2 inputs (attnOut, ffnOut), got %d", len(inputs))
	}
	res := n.norm.Residual()
	if res == nil {
		return nil, fmt.Errorf("FalconParallelAdd: layer norm has no cached residual")
	}
	// hidden = res + attnOut + ffnOut
	sum1, err := n.engine.Add(ctx, res, inputs[0])
	if err != nil {
		return nil, err
	}
	return n.engine.Add(ctx, sum1, inputs[1])
}

func (n *falconParallelAddNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("FalconParallelAdd: backward not implemented")
}
