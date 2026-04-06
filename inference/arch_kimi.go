package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func init() {
	RegisterArchitecture("kimi-linear", buildKimiLinearGraph)
}

// buildKimiLinearGraph constructs a computation graph for the Kimi K2/K2.5
// linear-attention MoE architecture from pre-loaded GGUF tensors.
//
// Kimi uses linear attention: phi(Q) * (phi(K)^T * V) where phi(x) = ELU(x) + 1,
// replacing the softmax(QK^T/sqrt(d))V formulation. Combined with MoE routing,
// this enables efficient long-context inference.
//
// The architecture is:
//
//	Embed -> [RMSNorm -> LinearAttn -> Add -> RMSNorm -> MoE -> Add] x N -> RMSNorm -> LMHead
func buildKimiLinearGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
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

	finalNormWeight, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}

	numExperts := cfg.NumExperts
	if numExperts == 0 {
		numExperts = 8
	}
	topK := cfg.NumExpertsPerToken
	if topK == 0 {
		topK = 2
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
		blkPrefix := fmt.Sprintf("blk.%d.", i)

		// --- Input RMSNorm ---
		inputNormW, err := tl.Lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		inputNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"input_layernorm.weight", inputNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Linear Attention ---
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

		ctx := context.Background()
		qWT, err := engine.Transpose(ctx, qW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose q: %w", i, err)
		}
		kWT, err := engine.Transpose(ctx, kW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose k: %w", i, err)
		}
		vWT, err := engine.Transpose(ctx, vW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose v: %w", i, err)
		}
		oWT, err := engine.Transpose(ctx, oW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose o: %w", i, err)
		}

		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)), nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.k_proj.weight", kWT)), nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.v_proj.weight", vWT)), nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)), nil,
		)

		kvHeads := cfg.NumKVHeads
		if kvHeads == 0 {
			kvHeads = cfg.NumHeads
		}

		linearAttn := &kimiLinearAttentionNode[float32]{
			engine:   proxy,
			ops:      ops,
			numHeads: cfg.NumHeads,
			kvHeads:  kvHeads,
			headDim:  headDim,
			wQ:       wq,
			wK:       wk,
			wV:       wv,
			wO:       wo,
		}

		attnOut := builder.AddNode(linearAttn, normed)

		// --- Fused Residual Add + Pre-MoE RMSNorm ---
		postNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- MoE ---
		ffnOut, err := buildKimiMoE(
			tensors, cfg, proxy, ops, builder, normed2,
			i, blkPrefix, numExperts, topK,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d moe: %w", i, err)
		}

		// --- Residual Add ---
		resAdd := &residualAddNode[float32]{engine: proxy, source: fusedNode}
		hidden = builder.AddNode(resAdd, ffnOut)
	}

	// --- Final RMSNorm ---
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, rmsEps, pw.Wrap("model.norm.weight", finalNormWeight),
	)
	if err != nil {
		return nil, nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// buildKimiMoE constructs the MoE sub-graph for a single Kimi layer.
func buildKimiMoE(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	proxy *compute.EngineProxy[float32],
	ops numeric.Float32Ops,
	builder *graph.Builder[float32],
	normed graph.Node[float32],
	layerIdx int,
	blkPrefix string,
	numExperts, topK int,
) (graph.Node[float32], error) {
	routerW, ok := tensors[blkPrefix+"ffn_gate_inp.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blkPrefix+"ffn_gate_inp.weight")
	}

	gateExpsW, ok := tensors[blkPrefix+"ffn_gate_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blkPrefix+"ffn_gate_exps.weight")
	}
	upExpsW, ok := tensors[blkPrefix+"ffn_up_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blkPrefix+"ffn_up_exps.weight")
	}
	downExpsW, ok := tensors[blkPrefix+"ffn_down_exps.weight"]
	if !ok {
		return nil, fmt.Errorf("missing tensor %q", blkPrefix+"ffn_down_exps.weight")
	}

	experts := make([]graph.Node[float32], numExperts)
	for e := 0; e < numExperts; e++ {
		expertFFN, err := buildExpertFFN(
			proxy, ops, gateExpsW, upExpsW, downExpsW,
			e, numExperts, cfg.HiddenSize, cfg.IntermediateSize,
			fmt.Sprintf("layer%d_expert%d", layerIdx, e),
		)
		if err != nil {
			return nil, fmt.Errorf("expert %d: %w", e, err)
		}
		experts[e] = expertFFN
	}

	gate := core.NewMoEGate[float32](proxy, ops, topK)
	moe := core.NewMixtureOfExperts[float32](proxy, ops, gate, experts, numExperts, topK)

	reshapeNode := &deepSeekReshapeNode[float32]{engine: proxy, flatten: true}
	flat := builder.AddNode(reshapeNode, normed)

	routerNode := &deepSeekConstNode[float32]{value: routerW}
	routerOut := builder.AddNode(routerNode)

	moeOut := builder.AddNode(moe, flat, routerOut)

	unreshapeNode := &deepSeekReshapeNode[float32]{engine: proxy, flatten: false}
	return builder.AddNode(unreshapeNode, moeOut, normed), nil
}

// kimiLinearAttentionNode implements linear attention for Kimi models.
// Instead of softmax(QK^T/sqrt(d))V, it computes phi(Q)(phi(K)^T V) where
// phi(x) = ELU(x) + 1. This yields O(n*d^2) complexity instead of O(n^2*d).
type kimiLinearAttentionNode[T tensor.Numeric] struct {
	engine   compute.Engine[T]
	ops      numeric.Float32Ops
	numHeads int
	kvHeads  int
	headDim  int
	wQ       *core.Dense[T]
	wK       *core.Dense[T]
	wV       *core.Dense[T]
	wO       *core.Dense[T]
}

func (n *kimiLinearAttentionNode[T]) OpType() string                  { return "KimiLinearAttention" }
func (n *kimiLinearAttentionNode[T]) Attributes() map[string]any       { return nil }
func (n *kimiLinearAttentionNode[T]) OutputShape() []int               { return nil }
func (n *kimiLinearAttentionNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *kimiLinearAttentionNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0]
	shape := x.Shape()
	batch, seqLen, _ := shape[0], shape[1], shape[2]

	// Project Q, K, V.
	q, err := n.wQ.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("linear attn q_proj: %w", err)
	}
	k, err := n.wK.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("linear attn k_proj: %w", err)
	}
	v, err := n.wV.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("linear attn v_proj: %w", err)
	}

	// Reshape to [batch, heads, seqLen, headDim].
	qData := q.Data()
	kData := k.Data()
	vData := v.Data()

	kvHeadsPerGroup := n.numHeads / n.kvHeads
	outData := make([]T, batch*seqLen*n.numHeads*n.headDim)

	for b := 0; b < batch; b++ {
		for h := 0; h < n.numHeads; h++ {
			kvH := h / kvHeadsPerGroup

			// Apply phi (ELU+1) to Q and K, then compute linear attention:
			// O = phi(Q) * (phi(K)^T * V)
			//
			// For each position, accumulate KV state: S += phi(k) * v^T
			// Then output = phi(q) * S / (phi(q) . sum_phi_k)

			// Accumulate KV state: S[headDim x headDim] and normalization z[headDim].
			s := make([]float64, n.headDim*n.headDim)
			z := make([]float64, n.headDim)

			for t := 0; t < seqLen; t++ {
				// Extract phi(k) and v for this position.
				phiK := make([]float64, n.headDim)
				vVec := make([]float64, n.headDim)
				for d := 0; d < n.headDim; d++ {
					kIdx := b*seqLen*n.kvHeads*n.headDim + t*n.kvHeads*n.headDim + kvH*n.headDim + d
					kVal := float64(kData[kIdx])
					phiK[d] = eluPlus1(kVal)

					vIdx := b*seqLen*n.kvHeads*n.headDim + t*n.kvHeads*n.headDim + kvH*n.headDim + d
					vVec[d] = float64(vData[vIdx])
				}

				// S += phi(k) * v^T (outer product).
				for di := 0; di < n.headDim; di++ {
					for dj := 0; dj < n.headDim; dj++ {
						s[di*n.headDim+dj] += phiK[di] * vVec[dj]
					}
					z[di] += phiK[di]
				}

				// Compute output: o = phi(q) * S, normalized by phi(q) . z.
				phiQ := make([]float64, n.headDim)
				for d := 0; d < n.headDim; d++ {
					qIdx := b*seqLen*n.numHeads*n.headDim + t*n.numHeads*n.headDim + h*n.headDim + d
					qVal := float64(qData[qIdx])
					phiQ[d] = eluPlus1(qVal)
				}

				// o_d = sum_i phi(q)_i * S[i,d]
				norm := float64(0)
				for d := 0; d < n.headDim; d++ {
					norm += phiQ[d] * z[d]
				}
				if norm < 1e-6 {
					norm = 1e-6
				}

				for d := 0; d < n.headDim; d++ {
					val := float64(0)
					for di := 0; di < n.headDim; di++ {
						val += phiQ[di] * s[di*n.headDim+d]
					}
					oIdx := b*seqLen*n.numHeads*n.headDim + t*n.numHeads*n.headDim + h*n.headDim + d
					outData[oIdx] = T(val / norm)
				}
			}
		}
	}

	attnOut, err := tensor.New([]int{batch, seqLen, n.numHeads * n.headDim}, outData)
	if err != nil {
		return nil, err
	}

	// Output projection.
	return n.wO.Forward(ctx, attnOut)
}

func (n *kimiLinearAttentionNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// eluPlus1 computes the ELU+1 feature map: phi(x) = ELU(x) + 1.
// For x >= 0: phi(x) = x + 1. For x < 0: phi(x) = exp(x).
func eluPlus1(x float64) float64 {
	if x >= 0 {
		return x + 1
	}
	return math.Exp(x)
}
