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

// buildGPT2Graph constructs a computation graph for the GPT-2 architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// GPT-2 is a decoder-only transformer with:
//   - Learned absolute position embeddings (not RoPE)
//   - Pre-norm LayerNorm with bias (not RMSNorm)
//   - GELU activation (not SiLU)
//   - 2-matrix FFN: up_proj + down_proj (not gated SwiGLU)
//   - MHA (num_kv_heads == num_heads)
//   - Tied LM head to token_embd.weight when output.weight absent
//
// Graph structure:
//
//	TokenEmbed+PosEmbed -> [LN -> GQA(noRoPE) -> ResAdd -> LN -> FFN(GELU) -> ResAdd] x N -> LN -> LMHead
func buildGPT2Graph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	lnEps := float32(1e-5)
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

	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: t}
	}

	// Load global tensors.
	tokenEmbdW, err := lookup("token_embd.weight")
	if err != nil {
		return nil, nil, err
	}
	posEmbdW, err := lookup("position_embd.weight")
	if err != nil {
		return nil, nil, err
	}

	// Tied LM head: use output.weight if present, else token_embd.weight.
	lmHeadWeight, ok := tensors["output.weight"]
	if !ok {
		lmHeadWeight = tokenEmbdW
	}

	// Final LayerNorm.
	outputNormW, err := lookup("output_norm.weight")
	if err != nil {
		return nil, nil, err
	}
	outputNormB, err := lookup("output_norm.bias")
	if err != nil {
		return nil, nil, err
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)

	// Input: token IDs as [1, seqLen].
	input := builder.Input([]int{1, 1})

	// Embedding: token lookup + position embedding addition.
	embNode := &gpt2EmbeddingNode[float32]{
		engine:      proxy,
		tokenWeight: tokenEmbdW,
		posWeight:   posEmbdW,
	}
	hidden := builder.AddNode(embNode, input)

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	numKVHeads := cfg.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = cfg.NumHeads
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)

		// --- Pre-attention LayerNorm ---
		attnNormW, err := lookup(prefix + "attn_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		attnNormB, err := lookup(prefix + "attn_norm.bias")
		if err != nil {
			return nil, nil, err
		}
		attnNorm := normalization.NewLayerNormalizationFromParams[float32](
			proxy, float32(lnEps),
			param(prefix+"attn_norm.weight", attnNormW),
			param(prefix+"attn_norm.bias", attnNormB),
		)
		normed := builder.AddNode(attnNorm, hidden)

		// --- Self-Attention ---
		// Split merged QKV tensor into separate Q, K, V.
		qkvW, err := lookup(prefix + "attn_qkv.weight")
		if err != nil {
			return nil, nil, err
		}
		qW, kW, vW, err := splitQKV(qkvW, cfg.NumHeads, numKVHeads, headDim)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d split QKV weight: %w", i, err)
		}

		var qB, kB, vB *tensor.TensorNumeric[float32]
		if qkvB, ok := tensors[prefix+"attn_qkv.bias"]; ok {
			qB, kB, vB, err = splitQKVBias(qkvB, cfg.NumHeads, numKVHeads, headDim)
			if err != nil {
				return nil, nil, fmt.Errorf("layer %d split QKV bias: %w", i, err)
			}
		}

		oW, err := lookup(prefix + "attn_output.weight")
		if err != nil {
			return nil, nil, err
		}

		// Transpose weights from [outDim, inDim] -> [inDim, outDim] for matmul.
		qWT, err := engine.Transpose(context.Background(), qW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose Q: %w", i, err)
		}
		kWT, err := engine.Transpose(context.Background(), kW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose K: %w", i, err)
		}
		vWT, err := engine.Transpose(context.Background(), vW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose V: %w", i, err)
		}
		oWT, err := engine.Transpose(context.Background(), oW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose O: %w", i, err)
		}

		oBias := tensors[prefix+"attn_output.bias"]

		attnNode := &gpt2SelfAttentionNode[float32]{
			engine:   proxy,
			numHeads: cfg.NumHeads,
			headDim:  headDim,
			qWeight:  qWT,
			kWeight:  kWT,
			vWeight:  vWT,
			oWeight:  oWT,
			qBias:    qB,
			kBias:    kB,
			vBias:    vB,
			oBias:    oBias,
			layerIdx: i,
		}
		attnOut := builder.AddNode(attnNode, normed)

		// --- Post-attention residual add ---
		resAdd1 := &gpt2ResidualAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd1, attnOut, hidden)

		// --- Pre-FFN LayerNorm ---
		ffnNormW, err := lookup(prefix + "ffn_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		ffnNormB, err := lookup(prefix + "ffn_norm.bias")
		if err != nil {
			return nil, nil, err
		}
		ffnNorm := normalization.NewLayerNormalizationFromParams[float32](
			proxy, float32(lnEps),
			param(prefix+"ffn_norm.weight", ffnNormW),
			param(prefix+"ffn_norm.bias", ffnNormB),
		)
		normed2 := builder.AddNode(ffnNorm, hidden)

		// --- FFN: Linear(GELU) ---
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
			return nil, nil, fmt.Errorf("layer %d transpose ffn_up: %w", i, err)
		}
		ffnDownWT, err := engine.Transpose(context.Background(), ffnDownW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose ffn_down: %w", i, err)
		}

		ffnNode := &gpt2FFNNode[float32]{
			engine:     proxy,
			ops:        ops,
			upWeight:   ffnUpWT,
			upBias:     tensors[prefix+"ffn_up.bias"],
			downWeight: ffnDownWT,
			downBias:   tensors[prefix+"ffn_down.bias"],
		}
		ffnOut := builder.AddNode(ffnNode, normed2)

		// --- Post-FFN residual add ---
		resAdd2 := &gpt2ResidualAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd2, ffnOut, hidden)
	}

	// --- Final LayerNorm ---
	finalNorm := normalization.NewLayerNormalizationFromParams[float32](
		proxy, float32(lnEps),
		param("output_norm.weight", outputNormW),
		param("output_norm.bias", outputNormB),
	)
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	lmHead := &lmHeadNode[float32]{engine: proxy, weight: lmHeadWeight}
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, tokenEmbdW, nil
}

// splitQKV splits a merged QKV weight tensor [qRows+kRows+vRows, hiddenDim]
// into separate Q, K, V weight tensors.
func splitQKV(
	qkv *tensor.TensorNumeric[float32],
	numHeads, numKVHeads, headDim int,
) (q, k, v *tensor.TensorNumeric[float32], err error) {
	qRows := numHeads * headDim
	kRows := numKVHeads * headDim
	vRows := numKVHeads * headDim

	shape := qkv.Shape()
	if len(shape) != 2 {
		return nil, nil, nil, fmt.Errorf("QKV tensor must be 2D, got %v", shape)
	}
	totalRows := shape[0]
	cols := shape[1]

	expected := qRows + kRows + vRows
	if totalRows != expected {
		return nil, nil, nil, fmt.Errorf("QKV tensor has %d rows, expected %d (Q=%d + K=%d + V=%d)",
			totalRows, expected, qRows, kRows, vRows)
	}

	data := qkv.Data()
	qData := make([]float32, qRows*cols)
	kData := make([]float32, kRows*cols)
	vData := make([]float32, vRows*cols)

	copy(qData, data[:qRows*cols])
	copy(kData, data[qRows*cols:(qRows+kRows)*cols])
	copy(vData, data[(qRows+kRows)*cols:])

	q, err = tensor.New([]int{qRows, cols}, qData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("create Q tensor: %w", err)
	}
	k, err = tensor.New([]int{kRows, cols}, kData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("create K tensor: %w", err)
	}
	v, err = tensor.New([]int{vRows, cols}, vData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("create V tensor: %w", err)
	}
	return q, k, v, nil
}

// splitQKVBias splits a merged QKV bias vector [qDim+kDim+vDim] into
// separate Q, K, V bias vectors.
func splitQKVBias(
	qkvBias *tensor.TensorNumeric[float32],
	numHeads, numKVHeads, headDim int,
) (qB, kB, vB *tensor.TensorNumeric[float32], err error) {
	qDim := numHeads * headDim
	kDim := numKVHeads * headDim
	vDim := numKVHeads * headDim

	data := qkvBias.Data()
	expected := qDim + kDim + vDim
	if len(data) != expected {
		return nil, nil, nil, fmt.Errorf("QKV bias has %d elements, expected %d", len(data), expected)
	}

	qBData := make([]float32, qDim)
	kBData := make([]float32, kDim)
	vBData := make([]float32, vDim)

	copy(qBData, data[:qDim])
	copy(kBData, data[qDim:qDim+kDim])
	copy(vBData, data[qDim+kDim:])

	qB, err = tensor.New([]int{qDim}, qBData)
	if err != nil {
		return nil, nil, nil, err
	}
	kB, err = tensor.New([]int{kDim}, kBData)
	if err != nil {
		return nil, nil, nil, err
	}
	vB, err = tensor.New([]int{vDim}, vBData)
	if err != nil {
		return nil, nil, nil, err
	}
	return qB, kB, vB, nil
}

// gpt2EmbeddingNode computes GPT-2-style embeddings: token lookup + learned
// position embedding addition. It tracks a position offset for autoregressive
// decode and implements graph.Resettable to reset between sequences.
type gpt2EmbeddingNode[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	tokenWeight *tensor.TensorNumeric[T] // [vocabSize, hiddenDim]
	posWeight   *tensor.TensorNumeric[T] // [maxPos, hiddenDim]
	posOffset   int                      // current position offset for autoregressive decode
}

func (e *gpt2EmbeddingNode[T]) OpType() string                  { return "GPT2Embedding" }
func (e *gpt2EmbeddingNode[T]) Attributes() map[string]any       { return nil }
func (e *gpt2EmbeddingNode[T]) OutputShape() []int               { return nil }
func (e *gpt2EmbeddingNode[T]) Parameters() []*graph.Parameter[T] { return nil }

// EmbeddedFrozen returns the token and position embedding weights so the
// compiler registers them as frozen slots during graph compilation.
func (e *gpt2EmbeddingNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{e.tokenWeight, e.posWeight}
}

// Reset resets the position offset between generation sequences.
// This implements the graph.Resettable interface.
func (e *gpt2EmbeddingNode[T]) Reset() {
	e.posOffset = 0
}

func (e *gpt2EmbeddingNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	ids := input.Data()
	hiddenDim := e.tokenWeight.Shape()[1]
	vocabSize := e.tokenWeight.Shape()[0]
	maxPos := e.posWeight.Shape()[0]

	seqLen := 1
	for _, d := range shape {
		seqLen *= d
	}
	batch := shape[0]
	sl := seqLen / batch

	tokenData := e.tokenWeight.Data()
	posData := e.posWeight.Data()

	result := make([]T, seqLen*hiddenDim)
	for i := 0; i < seqLen; i++ {
		tokenID := int(ids[i])
		if tokenID < 0 || tokenID >= vocabSize {
			return nil, fmt.Errorf("token ID %d out of range [0, %d)", tokenID, vocabSize)
		}

		// Position ID: offset within the batch element + global offset.
		posInBatch := i % sl
		posID := e.posOffset + posInBatch
		if posID >= maxPos {
			return nil, fmt.Errorf("position %d exceeds max position %d", posID, maxPos)
		}

		tokenOff := tokenID * hiddenDim
		posOff := posID * hiddenDim

		for d := 0; d < hiddenDim; d++ {
			result[i*hiddenDim+d] = tokenData[tokenOff+d] + posData[posOff+d]
		}
	}

	// Advance position offset for next autoregressive decode step.
	e.posOffset += sl

	return tensor.New[T]([]int{batch, sl, hiddenDim}, result)
}

func (e *gpt2EmbeddingNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// gpt2SelfAttentionNode computes causal multi-head self-attention for GPT-2.
// GPT-2 uses standard MHA (num_kv_heads == num_heads) without RoPE.
type gpt2SelfAttentionNode[T tensor.Float] struct {
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

func (a *gpt2SelfAttentionNode[T]) OpType() string                    { return "GPT2SelfAttention" }
func (a *gpt2SelfAttentionNode[T]) Attributes() map[string]any         { return nil }
func (a *gpt2SelfAttentionNode[T]) OutputShape() []int                 { return nil }
func (a *gpt2SelfAttentionNode[T]) Parameters() []*graph.Parameter[T]  { return nil }

func (a *gpt2SelfAttentionNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0] // [batch, seqLen, hidden]
	shape := x.Shape()
	batch := shape[0]
	seqLen := shape[1]
	hidden := shape[2]

	// Q/K/V projections: [batch, seqLen, hidden] @ [hidden, hidden]
	q, err := a.engine.MatMul(ctx, x, a.qWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("GPT2SelfAttention Q matmul: %w", err)
	}
	if a.qBias != nil {
		q, err = a.engine.Add(ctx, q, a.qBias, nil)
		if err != nil {
			return nil, fmt.Errorf("GPT2SelfAttention Q bias: %w", err)
		}
	}

	k, err := a.engine.MatMul(ctx, x, a.kWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("GPT2SelfAttention K matmul: %w", err)
	}
	if a.kBias != nil {
		k, err = a.engine.Add(ctx, k, a.kBias, nil)
		if err != nil {
			return nil, fmt.Errorf("GPT2SelfAttention K bias: %w", err)
		}
	}

	v, err := a.engine.MatMul(ctx, x, a.vWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("GPT2SelfAttention V matmul: %w", err)
	}
	if a.vBias != nil {
		v, err = a.engine.Add(ctx, v, a.vBias, nil)
		if err != nil {
			return nil, fmt.Errorf("GPT2SelfAttention V bias: %w", err)
		}
	}

	// Manual causal multi-head attention on CPU.
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
			// Compute scores = Q @ K^T / sqrt(headDim) with causal mask.
			scores := make([]T, seqLen*seqLen)
			for i := 0; i < seqLen; i++ {
				for j := 0; j <= i; j++ { // causal: j <= i
					var dot T
					for d := 0; d < headDim; d++ {
						qi := qData[bOff+i*hidden+h*headDim+d]
						kj := kData[bOff+j*hidden+h*headDim+d]
						dot += qi * kj
					}
					scores[i*seqLen+j] = dot * scale
				}
				// Fill masked positions with -inf.
				for j := i + 1; j < seqLen; j++ {
					scores[i*seqLen+j] = T(math.Inf(-1))
				}
			}

			// Softmax per row.
			for i := 0; i < seqLen; i++ {
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

			// Weighted sum: output = scores @ V.
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
		return nil, fmt.Errorf("GPT2SelfAttention output tensor: %w", err)
	}

	// Output projection: [batch, seqLen, hidden] @ [hidden, hidden]
	result, err := a.engine.MatMul(ctx, attnOut, a.oWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("GPT2SelfAttention O matmul: %w", err)
	}
	if a.oBias != nil {
		result, err = a.engine.Add(ctx, result, a.oBias, nil)
		if err != nil {
			return nil, fmt.Errorf("GPT2SelfAttention O bias: %w", err)
		}
	}

	return result, nil
}

func (a *gpt2SelfAttentionNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// gpt2FFNNode computes the GPT-2 FFN: Linear + GELU + Linear.
// GPT-2 uses GELU activation (not SwiGLU) with a 2-matrix FFN (up + down).
type gpt2FFNNode[T tensor.Float] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	upWeight   *tensor.TensorNumeric[T] // [hiddenDim, interDim] (transposed)
	upBias     *tensor.TensorNumeric[T] // [interDim], may be nil
	downWeight *tensor.TensorNumeric[T] // [interDim, hiddenDim] (transposed)
	downBias   *tensor.TensorNumeric[T] // [hiddenDim], may be nil
}

func (f *gpt2FFNNode[T]) OpType() string                  { return "GPT2FFN" }
func (f *gpt2FFNNode[T]) Attributes() map[string]any       { return nil }
func (f *gpt2FFNNode[T]) OutputShape() []int               { return nil }
func (f *gpt2FFNNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (f *gpt2FFNNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	x := inputs[0]

	// Up projection: [batch, seqLen, hidden] @ [hidden, inter] = [batch, seqLen, inter]
	up, err := f.engine.MatMul(ctx, x, f.upWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("GPT2FFN up: %w", err)
	}
	if f.upBias != nil {
		up, err = f.engine.Add(ctx, up, f.upBias, nil)
		if err != nil {
			return nil, fmt.Errorf("GPT2FFN up bias: %w", err)
		}
	}

	// GELU activation.
	gelu := activations.NewGelu[T](f.engine, f.ops)
	activated, err := gelu.Forward(ctx, up)
	if err != nil {
		return nil, fmt.Errorf("GPT2FFN gelu: %w", err)
	}

	// Down projection: [batch, seqLen, inter] @ [inter, hidden] = [batch, seqLen, hidden]
	down, err := f.engine.MatMul(ctx, activated, f.downWeight, nil)
	if err != nil {
		return nil, fmt.Errorf("GPT2FFN down: %w", err)
	}
	if f.downBias != nil {
		down, err = f.engine.Add(ctx, down, f.downBias, nil)
		if err != nil {
			return nil, fmt.Errorf("GPT2FFN down bias: %w", err)
		}
	}

	return down, nil
}

func (f *gpt2FFNNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// gpt2ResidualAddNode computes x + residual for GPT-2 pre-norm residual connections.
type gpt2ResidualAddNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (n *gpt2ResidualAddNode[T]) OpType() string                  { return "GPT2ResidualAdd" }
func (n *gpt2ResidualAddNode[T]) Attributes() map[string]any       { return nil }
func (n *gpt2ResidualAddNode[T]) OutputShape() []int               { return nil }
func (n *gpt2ResidualAddNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *gpt2ResidualAddNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("GPT2ResidualAdd: expected 2 inputs (x, residual), got %d", len(inputs))
	}
	return n.engine.Add(ctx, inputs[0], inputs[1], nil)
}

func (n *gpt2ResidualAddNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Static interface assertions.
var _ graph.EmbeddedFrozenProvider[float32] = (*gpt2EmbeddingNode[float32])(nil)
