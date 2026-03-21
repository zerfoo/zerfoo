package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/layers/activations"
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

	lnEps := float32(1e-12)
	if cfg.LayerNormEps > 0 {
		lnEps = cfg.LayerNormEps
	}
	if cfg.RMSNormEps > 0 && cfg.LayerNormEps == 0 {
		lnEps = cfg.RMSNormEps
	}

	lookup := func(name string) (*tensor.TensorNumeric[float32], error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	// Load global embedding tensors.
	tokenEmbdW, err := lookup("token_embd.weight")
	if err != nil {
		return nil, nil, err
	}
	posEmbdW, err := lookup("position_embd.weight")
	if err != nil {
		return nil, nil, err
	}
	tokenTypeEmbdW, err := lookup("token_type_embd.weight")
	if err != nil {
		return nil, nil, err
	}

	// Embedding LayerNorm.
	embNormW, err := lookup("token_embd_norm.weight")
	if err != nil {
		return nil, nil, err
	}
	embNormB, err := lookup("token_embd_norm.bias")
	if err != nil {
		return nil, nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)

	// Input: token IDs as [1, seqLen].
	input := builder.Input([]int{1, 1})

	// Embedding: token + position + token_type, then LayerNorm.
	embNode := &bertEmbeddingNode[float32]{
		engine:       proxy,
		tokenWeight:  tokenEmbdW,
		posWeight:    posEmbdW,
		typeWeight:   tokenTypeEmbdW,
		normWeight:   embNormW,
		normBias:     embNormB,
		normEps:      lnEps,
	}
	hidden := builder.AddNode(embNode, input)

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)

		// --- Self-Attention (bidirectional, direct multi-head) ---
		qW, err := lookup(prefix + "attn_q.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := lookup(prefix + "attn_k.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := lookup(prefix + "attn_v.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := lookup(prefix + "attn_output.weight")
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

		selfAttn := &bertSelfAttentionNode[float32]{
			engine:   proxy,
			numHeads: cfg.NumHeads,
			headDim:  headDim,
			qWeight:  qWT,
			kWeight:  kWT,
			vWeight:  vWT,
			oWeight:  oWT,
			qBias:    tensors[prefix+"attn_q.bias"],
			kBias:    tensors[prefix+"attn_k.bias"],
			vBias:    tensors[prefix+"attn_v.bias"],
			oBias:    tensors[prefix+"attn_output.bias"],
			layerIdx: i,
		}
		attnOut := builder.AddNode(selfAttn, hidden)

		// --- Post-attention residual + LayerNorm (BERT post-norm) ---
		attnResNorm := &bertResidualLayerNormNode[float32]{
			engine: proxy,
			weight: mustLookup(tensors, prefix+"attn_norm.weight"),
			bias:   mustLookup(tensors, prefix+"attn_norm.bias"),
			eps:    lnEps,
		}
		normed := builder.AddNode(attnResNorm, attnOut, hidden)

		// --- FFN: Linear(hidden->inter) + GELU + Linear(inter->hidden) ---
		ffnUpW, err := lookup(prefix + "ffn_up.weight")
		if err != nil {
			return nil, nil, err
		}
		ffnDownW, err := lookup(prefix + "ffn_down.weight")
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

		bertFFN := &bertFFNNode[float32]{
			engine:     proxy,
			ops:        ops,
			upWeight:   ffnUpWT,
			upBias:     tensors[prefix+"ffn_up.bias"],
			downWeight: ffnDownWT,
			downBias:   tensors[prefix+"ffn_down.bias"],
		}
		ffnOut := builder.AddNode(bertFFN, normed)

		// --- Post-FFN residual + LayerNorm ---
		ffnResNorm := &bertResidualLayerNormNode[float32]{
			engine: proxy,
			weight: mustLookup(tensors, prefix+"ffn_norm.weight"),
			bias:   mustLookup(tensors, prefix+"ffn_norm.bias"),
			eps:    lnEps,
		}
		hidden = builder.AddNode(ffnResNorm, ffnOut, normed)
	}

	// --- Pooler: extract CLS token, linear projection + tanh ---
	poolerW := tensors["cls_pooler.weight"]
	poolerB := tensors["cls_pooler.bias"]
	if poolerW != nil {
		poolerWT, tErr := engine.Transpose(context.Background(), poolerW, []int{1, 0})
		if tErr != nil {
			return nil, nil, fmt.Errorf("transpose cls_pooler.weight: %w", tErr)
		}
		poolerW = poolerWT
	}
	pooler := &bertPoolerNode[float32]{
		engine: proxy,
		weight: poolerW,
		bias:   poolerB,
	}
	pooled := builder.AddNode(pooler, hidden)

	// --- Classifier: linear projection to numLabels ---
	var clsWT *tensor.TensorNumeric[float32]
	if clsW, ok := tensors["cls.weight"]; ok {
		var tErr error
		clsWT, tErr = engine.Transpose(context.Background(), clsW, []int{1, 0})
		if tErr != nil {
			return nil, nil, fmt.Errorf("transpose cls.weight: %w", tErr)
		}
	}
	classifier := &bertClassifierNode[float32]{
		engine: proxy,
		weight: clsWT,
		bias:   tensors["cls.bias"],
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

// bertEmbeddingNode computes BERT-style embeddings: token + position + token_type,
// followed by LayerNorm.
type bertEmbeddingNode[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	tokenWeight *tensor.TensorNumeric[T] // [vocabSize, hiddenDim]
	posWeight   *tensor.TensorNumeric[T] // [maxPos, hiddenDim]
	typeWeight  *tensor.TensorNumeric[T] // [2, hiddenDim]
	normWeight  *tensor.TensorNumeric[T] // [hiddenDim]
	normBias    *tensor.TensorNumeric[T] // [hiddenDim]
	normEps     float32
}

func (e *bertEmbeddingNode[T]) OpType() string                  { return "BertEmbedding" }
func (e *bertEmbeddingNode[T]) Attributes() map[string]any       { return nil }
func (e *bertEmbeddingNode[T]) OutputShape() []int               { return nil }
func (e *bertEmbeddingNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (e *bertEmbeddingNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{e.tokenWeight, e.posWeight, e.typeWeight, e.normWeight, e.normBias}
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

	embedded, err := tensor.New[T]([]int{1, seqLen, hiddenDim}, result)
	if err != nil {
		return nil, err
	}

	// Apply LayerNorm.
	ln := normalization.NewLayerNormalizationFromParams[T](
		e.engine,
		T(e.normEps),
		&graph.Parameter[T]{Name: "emb_norm_gamma", Value: e.normWeight},
		&graph.Parameter[T]{Name: "emb_norm_beta", Value: e.normBias},
	)

	return ln.Forward(ctx, embedded)
}

func (e *bertEmbeddingNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// bertBiasAddNode adds a bias vector to the last dimension of the input.
type bertBiasAddNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	bias   *tensor.TensorNumeric[T]
}

func (b *bertBiasAddNode[T]) OpType() string                  { return "BertBiasAdd" }
func (b *bertBiasAddNode[T]) Attributes() map[string]any       { return nil }
func (b *bertBiasAddNode[T]) OutputShape() []int               { return nil }
func (b *bertBiasAddNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (b *bertBiasAddNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return b.engine.Add(ctx, inputs[0], b.bias, nil)
}

func (b *bertBiasAddNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// bertResidualLayerNormNode computes: LayerNorm(x + residual) with weight and bias.
// This implements BERT's post-norm pattern.
type bertResidualLayerNormNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	weight *tensor.TensorNumeric[T] // LayerNorm gamma
	bias   *tensor.TensorNumeric[T] // LayerNorm beta
	eps    float32
}

func (n *bertResidualLayerNormNode[T]) OpType() string                  { return "BertResidualLayerNorm" }
func (n *bertResidualLayerNormNode[T]) Attributes() map[string]any       { return nil }
func (n *bertResidualLayerNormNode[T]) OutputShape() []int               { return nil }
func (n *bertResidualLayerNormNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *bertResidualLayerNormNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("BertResidualLayerNorm: expected 2 inputs (x, residual), got %d", len(inputs))
	}

	// residual add: x + residual
	sum, err := n.engine.Add(ctx, inputs[0], inputs[1], nil)
	if err != nil {
		return nil, err
	}

	// LayerNorm
	ln := normalization.NewLayerNormalizationFromParams[T](
		n.engine,
		T(n.eps),
		&graph.Parameter[T]{Name: "ln_gamma", Value: n.weight},
		&graph.Parameter[T]{Name: "ln_beta", Value: n.bias},
	)

	return ln.Forward(ctx, sum)
}

func (n *bertResidualLayerNormNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// bertFFNNode computes the BERT FFN: Linear + GELU + Linear.
// BERT uses GELU activation (not SwiGLU) and has bias on both linear layers.
type bertFFNNode[T tensor.Float] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	upWeight   *tensor.TensorNumeric[T] // [hiddenDim, interDim] (transposed)
	upBias     *tensor.TensorNumeric[T] // [interDim], may be nil
	downWeight *tensor.TensorNumeric[T] // [interDim, hiddenDim] (transposed)
	downBias   *tensor.TensorNumeric[T] // [hiddenDim], may be nil
}

func (f *bertFFNNode[T]) OpType() string                  { return "BertFFN" }
func (f *bertFFNNode[T]) Attributes() map[string]any       { return nil }
func (f *bertFFNNode[T]) OutputShape() []int               { return nil }
func (f *bertFFNNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (f *bertFFNNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0]

	// Up projection: [batch, seqLen, hidden] @ [hidden, inter] = [batch, seqLen, inter]
	up, err := f.engine.MatMul(ctx, x, f.upWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("BertFFN up: %w", err)
	}
	if f.upBias != nil {
		up, err = f.engine.Add(ctx, up, f.upBias, nil)
		if err != nil {
			return nil, fmt.Errorf("BertFFN up bias: %w", err)
		}
	}

	// GELU activation.
	gelu := activations.NewGelu[T](f.engine, f.ops)
	activated, err := gelu.Forward(ctx, up)
	if err != nil {
		return nil, fmt.Errorf("BertFFN gelu: %w", err)
	}

	// Down projection: [batch, seqLen, inter] @ [inter, hidden] = [batch, seqLen, hidden]
	down, err := f.engine.MatMul(ctx, activated, f.downWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("BertFFN down: %w", err)
	}
	if f.downBias != nil {
		down, err = f.engine.Add(ctx, down, f.downBias, nil)
		if err != nil {
			return nil, fmt.Errorf("BertFFN down bias: %w", err)
		}
	}

	return down, nil
}

func (f *bertFFNNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// bertSelfAttentionNode computes bidirectional multi-head self-attention
// directly, without RoPE or KV-cache. This replaces the GQA+RoPE hack that
// caused float precision drift over multiple layers.
type bertSelfAttentionNode[T tensor.Float] struct {
	engine   compute.Engine[T]
	numHeads int
	headDim  int
	qWeight  *tensor.TensorNumeric[T] // transposed [hidden, hidden]
	kWeight  *tensor.TensorNumeric[T]
	vWeight  *tensor.TensorNumeric[T]
	oWeight  *tensor.TensorNumeric[T]
	qBias    *tensor.TensorNumeric[T] // [hidden], may be nil
	kBias    *tensor.TensorNumeric[T]
	vBias    *tensor.TensorNumeric[T]
	oBias    *tensor.TensorNumeric[T]
	layerIdx int
}

func (a *bertSelfAttentionNode[T]) OpType() string                    { return "BertSelfAttention" }
func (a *bertSelfAttentionNode[T]) Attributes() map[string]any         { return nil }
func (a *bertSelfAttentionNode[T]) OutputShape() []int                 { return nil }
func (a *bertSelfAttentionNode[T]) Parameters() []*graph.Parameter[T]  { return nil }

func (a *bertSelfAttentionNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0] // [batch, seqLen, hidden]
	shape := x.Shape()
	batch := shape[0]
	seqLen := shape[1]
	hidden := shape[2]

	// Q/K/V projections: [batch, seqLen, hidden] @ [hidden, hidden]
	q, err := a.engine.MatMul(ctx, x, a.qWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("BertSelfAttention Q matmul: %w", err)
	}
	if a.qBias != nil {
		q, err = a.engine.Add(ctx, q, a.qBias, nil)
		if err != nil {
			return nil, fmt.Errorf("BertSelfAttention Q bias: %w", err)
		}
	}

	k, err := a.engine.MatMul(ctx, x, a.kWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("BertSelfAttention K matmul: %w", err)
	}
	if a.kBias != nil {
		k, err = a.engine.Add(ctx, k, a.kBias, nil)
		if err != nil {
			return nil, fmt.Errorf("BertSelfAttention K bias: %w", err)
		}
	}

	v, err := a.engine.MatMul(ctx, x, a.vWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("BertSelfAttention V matmul: %w", err)
	}
	if a.vBias != nil {
		v, err = a.engine.Add(ctx, v, a.vBias, nil)
		if err != nil {
			return nil, fmt.Errorf("BertSelfAttention V bias: %w", err)
		}
	}

	// Manual multi-head attention on CPU.
	qData := q.Data()
	kData := k.Data()
	vData := v.Data()
	scale := T(1.0 / math.Sqrt(float64(a.headDim)))
	numHeads := a.numHeads
	headDim := a.headDim

	output := make([]T, batch*seqLen*hidden)

	for b := 0; b < batch; b++ {
		bOff := b * seqLen * hidden
		for h := 0; h < numHeads; h++ {
			// Compute scores = Q @ K^T / sqrt(headDim) for this head.
			scores := make([]T, seqLen*seqLen)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					var dot T
					for d := 0; d < headDim; d++ {
						qi := qData[bOff+i*hidden+h*headDim+d]
						kj := kData[bOff+j*hidden+h*headDim+d]
						dot += qi * kj
					}
					scores[i*seqLen+j] = dot * scale
				}
			}

			// Softmax per row (no causal mask).
			for i := 0; i < seqLen; i++ {
				// Find max for numerical stability.
				maxVal := scores[i*seqLen]
				for j := 1; j < seqLen; j++ {
					if scores[i*seqLen+j] > maxVal {
						maxVal = scores[i*seqLen+j]
					}
				}
				var sumExp T
				for j := 0; j < seqLen; j++ {
					scores[i*seqLen+j] = T(math.Exp(float64(scores[i*seqLen+j] - maxVal)))
					sumExp += scores[i*seqLen+j]
				}
				for j := 0; j < seqLen; j++ {
					scores[i*seqLen+j] /= sumExp
				}
			}

			// Weighted sum: output = scores @ V for this head.
			for i := 0; i < seqLen; i++ {
				for d := 0; d < headDim; d++ {
					var sum T
					for j := 0; j < seqLen; j++ {
						sum += scores[i*seqLen+j] * vData[bOff+j*hidden+h*headDim+d]
					}
					output[bOff+i*hidden+h*headDim+d] = sum
				}
			}
		}
	}

	attnOut, err := tensor.New[T]([]int{batch, seqLen, hidden}, output)
	if err != nil {
		return nil, fmt.Errorf("BertSelfAttention output tensor: %w", err)
	}

	// Output projection: [batch, seqLen, hidden] @ [hidden, hidden]
	result, err := a.engine.MatMul(ctx, attnOut, a.oWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("BertSelfAttention O matmul: %w", err)
	}
	if a.oBias != nil {
		result, err = a.engine.Add(ctx, result, a.oBias, nil)
		if err != nil {
			return nil, fmt.Errorf("BertSelfAttention O bias: %w", err)
		}
	}

	return result, nil
}

func (a *bertSelfAttentionNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// bertPoolerNode extracts the CLS token (first position), applies a linear
// projection and tanh activation.
type bertPoolerNode[T tensor.Float] struct {
	engine compute.Engine[T]
	weight *tensor.TensorNumeric[T] // [hidden, hidden] transposed, may be nil
	bias   *tensor.TensorNumeric[T] // [hidden], may be nil
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
	if p.weight == nil {
		return cls, nil
	}

	// Linear projection: [batch, hidden] @ [hidden, hidden]
	projected, err := p.engine.MatMul(ctx, cls, p.weight, nil)
	if err != nil {
		return nil, fmt.Errorf("BertPooler matmul: %w", err)
	}
	if p.bias != nil {
		projected, err = p.engine.Add(ctx, projected, p.bias, nil)
		if err != nil {
			return nil, fmt.Errorf("BertPooler bias: %w", err)
		}
	}

	// Tanh activation.
	projData := projected.Data()
	tanhData := make([]T, len(projData))
	for i, v := range projData {
		tanhData[i] = T(math.Tanh(float64(v)))
	}
	return tensor.New[T](projected.Shape(), tanhData)
}

func (p *bertPoolerNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// bertClassifierNode applies a linear projection to produce classification logits.
type bertClassifierNode[T tensor.Float] struct {
	engine compute.Engine[T]
	weight *tensor.TensorNumeric[T] // [hidden, numLabels] transposed, may be nil
	bias   *tensor.TensorNumeric[T] // [numLabels], may be nil
}

func (c *bertClassifierNode[T]) OpType() string                    { return "BertClassifier" }
func (c *bertClassifierNode[T]) Attributes() map[string]any         { return nil }
func (c *bertClassifierNode[T]) OutputShape() []int                 { return nil }
func (c *bertClassifierNode[T]) Parameters() []*graph.Parameter[T]  { return nil }

func (c *bertClassifierNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0] // [batch, hidden]

	if c.weight == nil {
		return x, nil
	}

	// Linear: [batch, hidden] @ [hidden, numLabels]
	logits, err := c.engine.MatMul(ctx, x, c.weight, nil)
	if err != nil {
		return nil, fmt.Errorf("BertClassifier matmul: %w", err)
	}
	if c.bias != nil {
		logits, err = c.engine.Add(ctx, logits, c.bias, nil)
		if err != nil {
			return nil, fmt.Errorf("BertClassifier bias: %w", err)
		}
	}
	return logits, nil
}

func (c *bertClassifierNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
