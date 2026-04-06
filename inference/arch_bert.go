package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// buildBertGraph constructs a computation graph for the BERT encoder-only
// architecture from pre-loaded GGUF tensors. BERT differs from decoder-only
// models in several ways:
//
//   - Three embeddings summed: token + position + token_type
//   - LayerNorm (not RMSNorm) with bias
//   - Bidirectional self-attention (no causal mask)
//   - GELU activation in FFN (not SwiGLU)
//   - Post-norm residual connections
//   - Sequence classification head instead of LM head
//   - No KV cache
//
// The graph accepts token IDs as input [1, seqLen] and produces classification
// logits of shape [batch, numClasses].
func buildBertGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	pw := newParamWrapper[float32]()

	lnEps := float32(1e-12)
	if cfg.LayerNormEps > 0 {
		lnEps = cfg.LayerNormEps
	}
	if cfg.RMSNormEps > 0 && cfg.LayerNormEps == 0 {
		lnEps = cfg.RMSNormEps
	}

	tl := newTensorLookup(tensors)

	// Load global embedding tensors.
	tokenEmbdW, err := tl.Lookup("token_embd.weight")
	if err != nil {
		return nil, nil, err
	}
	posEmbdW, err := tl.Lookup("position_embd.weight")
	if err != nil {
		return nil, nil, err
	}
	tokenTypeEmbdW, err := tl.Lookup("token_type_embd.weight")
	if err != nil {
		return nil, nil, err
	}

	// Embedding LayerNorm.
	embNormW, err := tl.Lookup("token_embd_norm.weight")
	if err != nil {
		return nil, nil, err
	}
	embNormB, err := tl.Lookup("token_embd_norm.bias")
	if err != nil {
		return nil, nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)

	// Input: token IDs as [1, seqLen].
	input := builder.Input([]int{1, 1})

	// Embedding: token + position + token_type.
	embNode := &bertEmbeddingNode[float32]{
		engine:      proxy,
		tokenWeight: tokenEmbdW,
		posWeight:   posEmbdW,
		typeWeight:  tokenTypeEmbdW,
	}
	embedded := builder.AddNode(embNode, input)

	// Embedding LayerNorm as a separate standard graph node.
	embLN := normalization.NewLayerNormalizationFromParams[float32](
		proxy, float32(lnEps),
		pw.Wrap("token_embd_norm.weight", embNormW),
		pw.Wrap("token_embd_norm.bias", embNormB),
	)
	hidden := builder.AddNode(embLN, embedded)

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)

		// --- Self-Attention (bidirectional, using GQA from layers/attention) ---
		qW, err := tl.Lookup(prefix + "attn_q.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := tl.Lookup(prefix + "attn_k.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := tl.Lookup(prefix + "attn_v.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := tl.Lookup(prefix + "attn_output.weight")
		if err != nil {
			return nil, nil, err
		}

		// Transpose weights for matmul: [out, in] -> [in, out].
		qWT, err := engine.Transpose(context.Background(), qW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("transpose %sattn_q: %w", prefix, err)
		}
		kWT, err := engine.Transpose(context.Background(), kW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("transpose %sattn_k: %w", prefix, err)
		}
		vWT, err := engine.Transpose(context.Background(), vW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("transpose %sattn_v: %w", prefix, err)
		}
		oWT, err := engine.Transpose(context.Background(), oW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("transpose %sattn_output: %w", prefix, err)
		}

		// Build Q/K/V/O Dense layers with optional bias.
		var qBias, kBias, vBias, oBias *core.Bias[float32]
		if b := tensors[prefix+"attn_q.bias"]; b != nil {
			qBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"attn_q.bias", b))
		}
		if b := tensors[prefix+"attn_k.bias"]; b != nil {
			kBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"attn_k.bias", b))
		}
		if b := tensors[prefix+"attn_v.bias"]; b != nil {
			vBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"attn_v.bias", b))
		}
		if b := tensors[prefix+"attn_output.bias"]; b != nil {
			oBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"attn_output.bias", b))
		}
		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_q.weight", qWT)), qBias,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_k.weight", kWT)), kBias,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_v.weight", vWT)), vBias,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_output.weight", oWT)), oBias,
		)

		gqa, gqaErr := attention.NewGroupedQueryAttentionFromParams[float32](
			proxy, ops, cfg.HiddenSize, cfg.NumHeads, cfg.NumHeads,
			wq, wk, wv, wo, nil, headDim,
		)
		if gqaErr != nil {
			return nil, nil, fmt.Errorf("layer %d gqa: %w", i, gqaErr)
		}
		gqa.SetBidirectional(true)
		gqa.LayerIndex = i
		attnOut := builder.AddNode(gqa, hidden)

		// --- Post-attention residual + LayerNorm (BERT post-norm) ---
		attnResAdd := &elementwiseAddNode[float32]{engine: proxy}
		attnSum := builder.AddNode(attnResAdd, attnOut, hidden)

		attnLN := normalization.NewLayerNormalizationFromParams[float32](
			proxy, float32(lnEps),
			pw.Wrap(prefix+"attn_norm.weight", mustLookup(tensors, prefix+"attn_norm.weight")),
			pw.Wrap(prefix+"attn_norm.bias", mustLookup(tensors, prefix+"attn_norm.bias")),
		)
		normed := builder.AddNode(attnLN, attnSum)

		// --- FFN: Dense(hidden->inter) + GELU + Dense(inter->hidden) ---
		ffnUpW, err := tl.Lookup(prefix + "ffn_up.weight")
		if err != nil {
			return nil, nil, err
		}
		ffnDownW, err := tl.Lookup(prefix + "ffn_down.weight")
		if err != nil {
			return nil, nil, err
		}

		ffnUpWT, err := engine.Transpose(context.Background(), ffnUpW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("transpose %sffn_up: %w", prefix, err)
		}
		ffnDownWT, err := engine.Transpose(context.Background(), ffnDownW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("transpose %sffn_down: %w", prefix, err)
		}

		var upBias, downBias *core.Bias[float32]
		if b := tensors[prefix+"ffn_up.bias"]; b != nil {
			upBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"ffn_up.bias", b))
		}
		if b := tensors[prefix+"ffn_down.bias"]; b != nil {
			downBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"ffn_down.bias", b))
		}
		ffnUp := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"ffn_up.weight", ffnUpWT)), upBias,
		)
		ffnDown := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"ffn_down.weight", ffnDownWT)), downBias,
		)
		gelu := activations.NewGelu[float32](proxy, ops)

		ffnUpOut := builder.AddNode(ffnUp, normed)
		geluOut := builder.AddNode(gelu, ffnUpOut)
		ffnOut := builder.AddNode(ffnDown, geluOut)

		// --- Post-FFN residual + LayerNorm ---
		ffnResAdd := &elementwiseAddNode[float32]{engine: proxy}
		ffnSum := builder.AddNode(ffnResAdd, ffnOut, normed)

		ffnLN := normalization.NewLayerNormalizationFromParams[float32](
			proxy, float32(lnEps),
			pw.Wrap(prefix+"ffn_norm.weight", mustLookup(tensors, prefix+"ffn_norm.weight")),
			pw.Wrap(prefix+"ffn_norm.bias", mustLookup(tensors, prefix+"ffn_norm.bias")),
		)
		hidden = builder.AddNode(ffnLN, ffnSum)
	}

	// --- Pooler: extract CLS token, linear projection + tanh ---
	var poolerDense *core.Dense[float32]
	if poolerW := tensors["cls_pooler.weight"]; poolerW != nil {
		poolerWT, tErr := engine.Transpose(context.Background(), poolerW, []int{1, 0})
		if tErr != nil {
			return nil, nil, fmt.Errorf("transpose cls_pooler.weight: %w", tErr)
		}
		var poolerBias *core.Bias[float32]
		if b := tensors["cls_pooler.bias"]; b != nil {
			poolerBias = core.NewBiasFromParam(proxy, ops, pw.Wrap("cls_pooler.bias", b))
		}
		poolerDense = core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap("cls_pooler.weight", poolerWT)), poolerBias,
		)
	}
	pooler := &bertPoolerNode[float32]{
		engine: proxy,
		dense:  poolerDense,
		tanh:   activations.NewTanh[float32](proxy, ops),
	}
	pooled := builder.AddNode(pooler, hidden)

	// --- Classifier: linear projection to numLabels ---
	var clsDense *core.Dense[float32]
	if clsW, ok := tensors["cls.weight"]; ok {
		clsWT, tErr := engine.Transpose(context.Background(), clsW, []int{1, 0})
		if tErr != nil {
			return nil, nil, fmt.Errorf("transpose cls.weight: %w", tErr)
		}
		var clsBias *core.Bias[float32]
		if b := tensors["cls.bias"]; b != nil {
			clsBias = core.NewBiasFromParam(proxy, ops, pw.Wrap("cls.bias", b))
		}
		clsDense = core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap("cls.weight", clsWT)), clsBias,
		)
	}
	classifier := &bertClassifierNode[float32]{
		dense: clsDense,
	}
	output := builder.AddNode(classifier, pooled)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, tokenEmbdW, nil
}

// mustLookup returns the tensor for name, or nil if not found.
func mustLookup(tensors map[string]*tensor.TensorNumeric[float32], name string) *tensor.TensorNumeric[float32] {
	return tensors[name]
}

// bertEmbeddingNode computes BERT-style embeddings: token + position + token_type.
// LayerNorm is applied as a separate graph node after this one.
type bertEmbeddingNode[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	tokenWeight *tensor.TensorNumeric[T] // [vocabSize, hiddenDim]
	posWeight   *tensor.TensorNumeric[T] // [maxPos, hiddenDim]
	typeWeight  *tensor.TensorNumeric[T] // [2, hiddenDim]
}

func (e *bertEmbeddingNode[T]) OpType() string                   { return "BertEmbedding" }
func (e *bertEmbeddingNode[T]) Attributes() map[string]any        { return nil }
func (e *bertEmbeddingNode[T]) OutputShape() []int                { return nil }
func (e *bertEmbeddingNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (e *bertEmbeddingNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{e.tokenWeight, e.posWeight, e.typeWeight}
}

func (e *bertEmbeddingNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	ids := input.Data()
	hiddenDim := e.tokenWeight.Shape()[1]

	seqLen := 1
	for _, d := range shape {
		seqLen *= d
	}

	tokenData := e.tokenWeight.Data()
	posData := e.posWeight.Data()
	typeData := e.typeWeight.Data()

	result := make([]T, seqLen*hiddenDim)
	for i := 0; i < seqLen; i++ {
		tokenID := int(ids[i])
		posID := i
		typeID := 0 // token type 0 for single-segment

		tokenOff := tokenID * hiddenDim
		posOff := posID * hiddenDim
		typeOff := typeID * hiddenDim

		for d := 0; d < hiddenDim; d++ {
			result[i*hiddenDim+d] = tokenData[tokenOff+d] + posData[posOff+d] + typeData[typeOff+d]
		}
	}

	return tensor.New[T]([]int{1, seqLen, hiddenDim}, result)
}

func (e *bertEmbeddingNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// bertPoolerNode extracts the CLS token (first position), applies a Dense
// projection and Tanh activation from layers/.
type bertPoolerNode[T tensor.Float] struct {
	engine compute.Engine[T]
	dense  *core.Dense[T]                 // [hidden, hidden], may be nil
	tanh   *activations.BaseActivation[T] // Tanh activation
}

func (p *bertPoolerNode[T]) OpType() string                    { return "BertPooler" }
func (p *bertPoolerNode[T]) Attributes() map[string]any         { return nil }
func (p *bertPoolerNode[T]) OutputShape() []int                 { return nil }
func (p *bertPoolerNode[T]) Parameters() []*graph.Parameter[T]  { return nil }

func (p *bertPoolerNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0] // [batch, seqLen, hidden]
	shape := x.Shape()
	batch := shape[0]
	hidden := shape[2]
	data := x.Data()

	// Extract CLS token (position 0) for each batch.
	clsData := make([]T, batch*hidden)
	for b := 0; b < batch; b++ {
		copy(clsData[b*hidden:(b+1)*hidden], data[b*shape[1]*hidden:b*shape[1]*hidden+hidden])
	}
	cls, err := tensor.New[T]([]int{batch, hidden}, clsData)
	if err != nil {
		return nil, fmt.Errorf("BertPooler CLS extract: %w", err)
	}

	// If no pooler weight, just return CLS (fallback for models without pooler).
	if p.dense == nil {
		return cls, nil
	}

	// Dense projection: [batch, hidden] @ [hidden, hidden] + bias
	projected, err := p.dense.Forward(ctx, cls)
	if err != nil {
		return nil, fmt.Errorf("BertPooler dense: %w", err)
	}

	// Tanh activation via layers/activations.
	return p.tanh.Forward(ctx, projected)
}

func (p *bertPoolerNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// bertClassifierNode applies a Dense projection to produce classification logits.
type bertClassifierNode[T tensor.Float] struct {
	dense *core.Dense[T] // [hidden, numLabels], may be nil
}

func (c *bertClassifierNode[T]) OpType() string                    { return "BertClassifier" }
func (c *bertClassifierNode[T]) Attributes() map[string]any         { return nil }
func (c *bertClassifierNode[T]) OutputShape() []int                 { return nil }
func (c *bertClassifierNode[T]) Parameters() []*graph.Parameter[T]  { return nil }

func (c *bertClassifierNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0] // [batch, hidden]

	if c.dense == nil {
		return x, nil
	}

	return c.dense.Forward(ctx, x)
}

func (c *bertClassifierNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
