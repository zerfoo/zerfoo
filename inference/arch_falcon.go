package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
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

	lookup := func(name string) (*tensor.TensorNumeric[float32], error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: t}
	}

	embedWeight, ok := tensors["model.embed_tokens.weight"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %q", "model.embed_tokens.weight")
	}

	lmHeadWeight, ok := tensors["lm_head.weight"]
	if !ok {
		lmHeadWeight = embedWeight
	}

	finalNormW, err := lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}
	finalNormB, _ := tensors["model.norm.bias"]

	_, isGPUEngine := engine.(compute.WeightUploader)

	transposeWeight := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		s := t.GetStorage()
		if isGPUEngine {
			if _, ok := any(s).(*tensor.Q4Storage); ok {
				shape := t.Shape()
				if len(shape) == 2 {
					return tensor.NewWithStorage[float32]([]int{shape[1], shape[0]}, s)
				}
			}
			shape := t.Shape()
			if len(shape) == 2 {
				if qs, ok := any(s).(*tensor.Q8Storage); ok {
					f32 := make([]float32, qs.Len())
					qs.Dequantize(f32)
					rows, cols := shape[0], shape[1]
					transposed := make([]float32, len(f32))
					for r := range rows {
						for c := range cols {
							transposed[c*rows+r] = f32[r*cols+c]
						}
					}
					return tensor.New([]int{cols, rows}, transposed)
				}
				if fs, ok := any(s).(*tensor.Float16Storage); ok {
					f32 := fs.Slice()
					rows, cols := shape[0], shape[1]
					transposed := make([]float32, len(f32))
					for r := range rows {
						for c := range cols {
							transposed[c*rows+r] = f32[r*cols+c]
						}
					}
					fp16 := tensor.NewFloat16StorageFromF32(transposed)
					return tensor.NewWithStorage[float32]([]int{cols, rows}, fp16)
				}
				if fs, ok := any(s).(*tensor.FP8E4M3Storage); ok {
					f32 := fs.Slice()
					rows, cols := shape[0], shape[1]
					transposed := make([]float32, len(f32))
					for r := range rows {
						for c := range cols {
							transposed[c*rows+r] = f32[r*cols+c]
						}
					}
					fp8 := tensor.NewFP8E4M3Storage(transposed)
					return tensor.NewWithStorage[float32]([]int{cols, rows}, fp8)
				}
			}
			tr, terr := engine.Transpose(context.Background(), t, []int{1, 0})
			if terr != nil {
				return nil, fmt.Errorf("transpose %s: %w", name, terr)
			}
			return tr, nil
		}

		// CPU path.
		if _, ok := any(s).(*tensor.Q4Storage); ok {
			shape := t.Shape()
			if len(shape) == 2 {
				return tensor.NewWithStorage[float32]([]int{shape[1], shape[0]}, s)
			}
		}
		if _, ok := any(s).(*tensor.Q8Storage); ok {
			shape := t.Shape()
			if len(shape) == 2 {
				return tensor.NewWithStorage[float32]([]int{shape[1], shape[0]}, s)
			}
		}
		if fs, ok := any(s).(*tensor.Float16Storage); ok {
			shape := t.Shape()
			if len(shape) == 2 {
				f32 := fs.Slice()
				rows, cols := shape[0], shape[1]
				transposed := make([]float32, len(f32))
				for r := range rows {
					for c := range cols {
						transposed[c*rows+r] = f32[r*cols+c]
					}
				}
				fp16 := tensor.NewFloat16StorageFromF32(transposed)
				return tensor.NewWithStorage[float32]([]int{cols, rows}, fp16)
			}
		}
		if fs, ok := any(s).(*tensor.FP8E4M3Storage); ok {
			shape := t.Shape()
			if len(shape) == 2 {
				f32 := fs.Slice()
				rows, cols := shape[0], shape[1]
				transposed := make([]float32, len(f32))
				for r := range rows {
					for c := range cols {
						transposed[c*rows+r] = f32[r*cols+c]
					}
				}
				fp8 := tensor.NewFP8E4M3Storage(transposed)
				return tensor.NewWithStorage[float32]([]int{cols, rows}, fp8)
			}
		}
		tr, terr := engine.Transpose(context.Background(), t, []int{1, 0})
		if terr != nil {
			return nil, fmt.Errorf("transpose %s: %w", name, terr)
		}
		return tr, nil
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	embNode := &embeddingLookupNode[float32]{engine: proxy, weight: embedWeight}
	hidden := builder.AddNode(embNode, input)

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Input LayerNorm (shared by attention and FFN in parallel) ---
		lnW, err := lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		lnB, _ := tensors[prefix+"input_layernorm.bias"]

		// falconLayerNormNode caches the pre-norm input (residual) so that
		// falconParallelAddNode can retrieve it without an extra graph input.
		inputNorm := &falconLayerNormNode[float32]{
			engine: proxy,
			weight: lnW,
			bias:   lnB,
			eps:    layerNormEps,
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Self Attention (GQA / MQA) ---
		qW, err := lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := lookup(prefix + "self_attn.o_proj.weight")
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
			core.NewLinearFromParam(proxy, param(prefix+"self_attn.q_proj.weight", qWT)),
			nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, param(prefix+"self_attn.k_proj.weight", kWT)),
			nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, param(prefix+"self_attn.v_proj.weight", vWT)),
			nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, param(prefix+"self_attn.o_proj.weight", oWT)),
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
		upW, err := lookup(prefix + "mlp.dense_h_to_4h.weight")
		if err != nil {
			return nil, nil, err
		}
		downW, err := lookup(prefix + "mlp.dense_4h_to_h.weight")
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
		upB, _ := tensors[prefix+"mlp.dense_h_to_4h.bias"]
		downB, _ := tensors[prefix+"mlp.dense_4h_to_h.bias"]

		ffnNode := &falconFFNNode[float32]{
			engine: proxy,
			ops:    ops,
			upW:    upWT,
			upB:    upB,
			downW:  downWT,
			downB:  downB,
		}
		ffnOut := builder.AddNode(ffnNode, normed)

		// --- Parallel Residual Add: hidden = residual + attn(norm(x)) + ffn(norm(x)) ---
		// The residual is retrieved from inputNorm's cached pre-norm input.
		parallelAdd := &falconParallelAddNode[float32]{
			engine: proxy,
			norm:   inputNorm,
		}
		hidden = builder.AddNode(parallelAdd, attnOut, ffnOut)
	}

	// --- Final LayerNorm ---
	finalNorm := &falconLayerNormNode[float32]{
		engine: proxy,
		weight: finalNormW,
		bias:   finalNormB,
		eps:    layerNormEps,
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	if s := lmHeadWeight.GetStorage(); s != nil {
		if qs, ok := any(s).(*tensor.Q8Storage); ok {
			f32 := make([]float32, qs.Len())
			qs.Dequantize(f32)
			q4 := tensor.QuantizeQ4(f32)
			lmHeadWeight, _ = tensor.NewWithStorage[float32](lmHeadWeight.Shape(), q4)
		}
	}
	lmHead := &lmHeadNode[float32]{engine: proxy, weight: lmHeadWeight}
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// falconLayerNormNode applies Layer Normalization with pre-loaded gamma and beta.
// It also caches the pre-norm input (the residual) so that falconParallelAddNode
// can retrieve it without re-introducing it as a graph input.
//
// Implements: (x - mean) / sqrt(var + eps) * gamma [+ beta]
type falconLayerNormNode[T tensor.Numeric] struct {
	engine   compute.Engine[T]
	weight   *tensor.TensorNumeric[T] // gamma (scale)
	bias     *tensor.TensorNumeric[T] // beta (shift), may be nil
	eps      float32
	residual *tensor.TensorNumeric[T] // pre-norm input, cached during Forward
}

func (n *falconLayerNormNode[T]) OpType() string { return "FalconLayerNorm" }
func (n *falconLayerNormNode[T]) Attributes() map[string]any {
	return map[string]any{"eps": n.eps}
}
func (n *falconLayerNormNode[T]) OutputShape() []int { return nil }
func (n *falconLayerNormNode[T]) Parameters() []*graph.Parameter[T] {
	params := []*graph.Parameter[T]{{Name: "weight", Value: n.weight}}
	if n.bias != nil {
		params = append(params, &graph.Parameter[T]{Name: "bias", Value: n.bias})
	}
	return params
}

func (n *falconLayerNormNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FalconLayerNorm: expected 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	// Cache the pre-norm input for use by falconParallelAddNode.
	n.residual = x

	ops := n.engine.Ops()
	eps := ops.FromFloat64(float64(n.eps))

	shape := x.Shape()
	lastDim := len(shape) - 1

	// Mean along the last dimension.
	sum, err := n.engine.ReduceSum(ctx, x, lastDim, true)
	if err != nil {
		return nil, err
	}
	featureSize := ops.FromFloat64(float64(shape[lastDim]))
	mean, err := n.engine.DivScalar(ctx, sum, featureSize)
	if err != nil {
		return nil, err
	}

	// Variance = mean((x - mean)^2).
	diff, err := n.engine.Sub(ctx, x, mean)
	if err != nil {
		return nil, err
	}
	diff2, err := n.engine.Mul(ctx, diff, diff)
	if err != nil {
		return nil, err
	}
	varSum, err := n.engine.ReduceSum(ctx, diff2, lastDim, true)
	if err != nil {
		return nil, err
	}
	variance, err := n.engine.DivScalar(ctx, varSum, featureSize)
	if err != nil {
		return nil, err
	}

	// Normalize: (x - mean) / sqrt(var + eps).
	varPlusEps, err := n.engine.AddScalar(ctx, variance, eps)
	if err != nil {
		return nil, err
	}
	stdDev, err := n.engine.Sqrt(ctx, varPlusEps)
	if err != nil {
		return nil, err
	}
	normed, err := n.engine.Div(ctx, diff, stdDev)
	if err != nil {
		return nil, err
	}

	// Scale by gamma.
	scaled, err := n.engine.Mul(ctx, normed, n.weight)
	if err != nil {
		return nil, err
	}

	// Shift by beta if present.
	if n.bias != nil {
		return n.engine.Add(ctx, scaled, n.bias)
	}
	return scaled, nil
}

func (n *falconLayerNormNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("FalconLayerNorm: backward not implemented")
}

// Residual returns the pre-norm input cached during the most recent Forward call.
func (n *falconLayerNormNode[T]) Residual() *tensor.TensorNumeric[T] {
	return n.residual
}

// falconFFNNode implements Falcon's 2-layer GELU FFN:
//
//	ffn(x) = down(gelu(up(x)))
//
// Unlike the SwiGLU FFN used in Llama, there is no gate projection.
type falconFFNNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	upW    *tensor.TensorNumeric[T]
	upB    *tensor.TensorNumeric[T] // optional
	downW  *tensor.TensorNumeric[T]
	downB  *tensor.TensorNumeric[T] // optional
}

func (n *falconFFNNode[T]) OpType() string         { return "FalconFFN" }
func (n *falconFFNNode[T]) Attributes() map[string]any { return nil }
func (n *falconFFNNode[T]) OutputShape() []int         { return nil }
func (n *falconFFNNode[T]) Parameters() []*graph.Parameter[T] {
	params := []*graph.Parameter[T]{
		{Name: "up_proj", Value: n.upW},
		{Name: "down_proj", Value: n.downW},
	}
	if n.upB != nil {
		params = append(params, &graph.Parameter[T]{Name: "up_bias", Value: n.upB})
	}
	if n.downB != nil {
		params = append(params, &graph.Parameter[T]{Name: "down_bias", Value: n.downB})
	}
	return params
}

func (n *falconFFNNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FalconFFN: expected 1 input, got %d", len(inputs))
	}
	x := inputs[0]

	// Up projection: [batch, seq, hidden] x [hidden, 4h] -> [batch, seq, 4h]
	up, err := n.engine.MatMul(ctx, x, n.upW)
	if err != nil {
		return nil, err
	}
	if n.upB != nil {
		up, err = n.engine.Add(ctx, up, n.upB)
		if err != nil {
			return nil, err
		}
	}

	// GELU activation.
	activated, err := falconGELU(ctx, n.engine, n.ops, up)
	if err != nil {
		return nil, err
	}

	// Down projection: [batch, seq, 4h] x [4h, hidden] -> [batch, seq, hidden]
	down, err := n.engine.MatMul(ctx, activated, n.downW)
	if err != nil {
		return nil, err
	}
	if n.downB != nil {
		down, err = n.engine.Add(ctx, down, n.downB)
		if err != nil {
			return nil, err
		}
	}

	return down, nil
}

func (n *falconFFNNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("FalconFFN: backward not implemented")
}

// falconGELU computes the GELU activation using engine primitives.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func falconGELU[T tensor.Numeric](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T], x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// x^2
	x2, err := engine.Mul(ctx, x, x)
	if err != nil {
		return nil, err
	}
	// x^3
	x3, err := engine.Mul(ctx, x2, x)
	if err != nil {
		return nil, err
	}
	// 0.044715 * x^3
	term1, err := engine.MulScalar(ctx, x3, ops.FromFloat64(0.044715))
	if err != nil {
		return nil, err
	}
	// x + 0.044715 * x^3
	term2, err := engine.Add(ctx, x, term1)
	if err != nil {
		return nil, err
	}
	// sqrt(2/pi) * (x + 0.044715 * x^3)
	const sqrtTwoPi = 0.7978845608028654 // math.Sqrt(2 / math.Pi)
	term3, err := engine.MulScalar(ctx, term2, ops.FromFloat64(sqrtTwoPi))
	if err != nil {
		return nil, err
	}
	// tanh(...)
	tanhResult, err := engine.Tanh(ctx, term3)
	if err != nil {
		return nil, err
	}
	// 1 + tanh(...)
	term4, err := engine.AddScalar(ctx, tanhResult, ops.One())
	if err != nil {
		return nil, err
	}
	// x * (1 + tanh(...))
	term5, err := engine.Mul(ctx, x, term4)
	if err != nil {
		return nil, err
	}
	// 0.5 * x * (1 + tanh(...))
	return engine.MulScalar(ctx, term5, ops.FromFloat64(0.5))
}

// falconParallelAddNode implements Falcon's parallel residual update:
//
//	hidden = residual + attnOut + ffnOut
//
// The residual is retrieved from the layer's falconLayerNormNode which caches
// its input (the pre-norm hidden state) during Forward.
type falconParallelAddNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	norm   *falconLayerNormNode[T] // supplies the cached residual
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
