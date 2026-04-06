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

	tl := newTensorLookup(tensors)

	pw := newParamWrapper[float32]()

	// Load global tensors.
	tokenEmbdW, err := tl.Lookup("token_embd.weight")
	if err != nil {
		return nil, nil, err
	}
	posEmbdW, err := tl.Lookup("position_embd.weight")
	if err != nil {
		return nil, nil, err
	}

	// Tied LM head: use output.weight if present, else token_embd.weight.
	lmHeadWeight, ok := tensors["output.weight"]
	if !ok {
		lmHeadWeight = tokenEmbdW
	}

	// Final LayerNorm.
	outputNormW, err := tl.Lookup("output_norm.weight")
	if err != nil {
		return nil, nil, err
	}
	outputNormB, err := tl.Lookup("output_norm.bias")
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
		attnNormW, err := tl.Lookup(prefix + "attn_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		attnNormB, err := tl.Lookup(prefix + "attn_norm.bias")
		if err != nil {
			return nil, nil, err
		}
		attnNorm := normalization.NewLayerNormalizationFromParams[float32](
			proxy, float32(lnEps),
			pw.Wrap(prefix+"attn_norm.weight", attnNormW),
			pw.Wrap(prefix+"attn_norm.bias", attnNormB),
		)
		normed := builder.AddNode(attnNorm, hidden)

		// --- Self-Attention (GQA with no RoPE) ---
		// Split merged QKV tensor into separate Q, K, V.
		qkvW, err := tl.Lookup(prefix + "attn_qkv.weight")
		if err != nil {
			return nil, nil, err
		}
		qW, kW, vW, err := splitQKV(qkvW, cfg.NumHeads, numKVHeads, headDim)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d split QKV weight: %w", i, err)
		}

		var qBiasLayer, kBiasLayer, vBiasLayer *core.Bias[float32]
		if qkvB, ok := tensors[prefix+"attn_qkv.bias"]; ok {
			qB, kB, vB, splitErr := splitQKVBias(qkvB, cfg.NumHeads, numKVHeads, headDim)
			if splitErr != nil {
				return nil, nil, fmt.Errorf("layer %d split QKV bias: %w", i, splitErr)
			}
			qBiasLayer = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"attn_q.bias", qB))
			kBiasLayer = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"attn_k.bias", kB))
			vBiasLayer = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"attn_v.bias", vB))
		}

		oW, err := tl.Lookup(prefix + "attn_output.weight")
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

		var oBiasLayer *core.Bias[float32]
		if oB := tensors[prefix+"attn_output.bias"]; oB != nil {
			oBiasLayer = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"attn_output.bias", oB))
		}

		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_q.weight", qWT)),
			qBiasLayer,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_k.weight", kWT)),
			kBiasLayer,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_v.weight", vWT)),
			vBiasLayer,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"attn_output.weight", oWT)),
			oBiasLayer,
		)

		gqa, gqaErr := attention.NewGroupedQueryAttentionFromParams[float32](
			proxy, ops, cfg.HiddenSize, cfg.NumHeads, numKVHeads,
			wq, wk, wv, wo, nil, headDim,
		)
		if gqaErr != nil {
			return nil, nil, fmt.Errorf("layer %d gqa: %w", i, gqaErr)
		}
		gqa.LayerIndex = i
		attnOut := builder.AddNode(gqa, normed)

		// --- Post-attention residual add ---
		resAdd1 := &elementwiseAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd1, attnOut, hidden)

		// --- Pre-FFN LayerNorm ---
		ffnNormW, err := tl.Lookup(prefix + "ffn_norm.weight")
		if err != nil {
			return nil, nil, err
		}
		ffnNormB, err := tl.Lookup(prefix + "ffn_norm.bias")
		if err != nil {
			return nil, nil, err
		}
		ffnNorm := normalization.NewLayerNormalizationFromParams[float32](
			proxy, float32(lnEps),
			pw.Wrap(prefix+"ffn_norm.weight", ffnNormW),
			pw.Wrap(prefix+"ffn_norm.bias", ffnNormB),
		)
		normed2 := builder.AddNode(ffnNorm, hidden)

		// --- FFN: Dense(GELU) + Dense ---
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
			return nil, nil, fmt.Errorf("layer %d transpose ffn_up: %w", i, err)
		}
		ffnDownWT, err := engine.Transpose(context.Background(), ffnDownW, []int{1, 0})
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d transpose ffn_down: %w", i, err)
		}

		var ffnUpBias *core.Bias[float32]
		if upB := tensors[prefix+"ffn_up.bias"]; upB != nil {
			ffnUpBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"ffn_up.bias", upB))
		}
		var ffnDownBias *core.Bias[float32]
		if downB := tensors[prefix+"ffn_down.bias"]; downB != nil {
			ffnDownBias = core.NewBiasFromParam(proxy, ops, pw.Wrap(prefix+"ffn_down.bias", downB))
		}

		ffnUp := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"ffn_up.weight", ffnUpWT)),
			ffnUpBias,
		)
		geluNode := activations.NewGelu[float32](proxy, ops)
		ffnDown := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"ffn_down.weight", ffnDownWT)),
			ffnDownBias,
		)

		ffnUpOut := builder.AddNode(ffnUp, normed2)
		geluOut := builder.AddNode(geluNode, ffnUpOut)
		ffnOut := builder.AddNode(ffnDown, geluOut)

		// --- Post-FFN residual add ---
		resAdd2 := &elementwiseAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd2, ffnOut, hidden)
	}

	// --- Final LayerNorm ---
	finalNorm := normalization.NewLayerNormalizationFromParams[float32](
		proxy, float32(lnEps),
		pw.Wrap("output_norm.weight", outputNormW),
		pw.Wrap("output_norm.bias", outputNormB),
	)
	normedFinal := builder.AddNode(finalNorm, hidden)

	// --- LM Head ---
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
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

// Static interface assertions.
var _ graph.EmbeddedFrozenProvider[float32] = (*gpt2EmbeddingNode[float32])(nil)
