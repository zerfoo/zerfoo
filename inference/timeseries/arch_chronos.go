// Package timeseries implements time-series model builders.
package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ChronosConfig holds configuration for building a Chronos-2 T5 graph.
type ChronosConfig struct {
	// NumEncoderLayers is the number of encoder transformer blocks.
	NumEncoderLayers int
	// NumDecoderLayers is the number of decoder transformer blocks.
	NumDecoderLayers int
	// DModel is the model hidden dimension.
	DModel int
	// NumHeads is the number of attention heads.
	NumHeads int
	// DFF is the feed-forward intermediate dimension.
	DFF int
	// VocabSize is the size of the bin vocabulary.
	VocabSize int
	// Horizon is the prediction horizon (number of decoder steps).
	Horizon int
}

// validateChronosConfig validates that the ChronosConfig has all required fields.
func validateChronosConfig(cfg *ChronosConfig) error {
	if cfg.NumEncoderLayers <= 0 {
		return fmt.Errorf("NumEncoderLayers must be positive, got %d", cfg.NumEncoderLayers)
	}
	if cfg.NumDecoderLayers <= 0 {
		return fmt.Errorf("NumDecoderLayers must be positive, got %d", cfg.NumDecoderLayers)
	}
	if cfg.DModel <= 0 {
		return fmt.Errorf("DModel must be positive, got %d", cfg.DModel)
	}
	if cfg.NumHeads <= 0 {
		return fmt.Errorf("NumHeads must be positive, got %d", cfg.NumHeads)
	}
	if cfg.DModel%cfg.NumHeads != 0 {
		return fmt.Errorf("DModel (%d) must be divisible by NumHeads (%d)", cfg.DModel, cfg.NumHeads)
	}
	if cfg.DFF <= 0 {
		return fmt.Errorf("DFF must be positive, got %d", cfg.DFF)
	}
	if cfg.VocabSize <= 0 {
		return fmt.Errorf("VocabSize must be positive, got %d", cfg.VocabSize)
	}
	if cfg.Horizon <= 0 {
		return fmt.Errorf("Horizon must be positive, got %d", cfg.Horizon)
	}
	return nil
}

// BuildChronos constructs a Chronos-2 T5 encoder-decoder computation graph.
//
// The Chronos-2 architecture is a T5 encoder-decoder model operating on
// tokenized time-series values. The pipeline is:
//
//  1. Token embedding: [batch, seq_len] token indices -> [batch, seq_len, d_model]
//  2. Encoder: stack of self-attention blocks (bidirectional)
//  3. Decoder: stack of self-attention (causal) + cross-attention blocks
//  4. LM head: [batch, horizon, d_model] -> [batch, horizon, vocab_size]
//
// tensors is a map of GGUF tensor name -> tensor data for loading pre-trained weights.
func BuildChronos[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *ChronosConfig,
	engine compute.Engine[T],
) (*graph.Graph[T], error) {
	if err := validateChronosConfig(cfg); err != nil {
		return nil, fmt.Errorf("invalid Chronos config: %w", err)
	}

	ops := engine.Ops()
	node, err := newChronosNode[T](tensors, cfg, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("create Chronos node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	input := builder.Input([]int{-1, -1})
	builder.AddNode(node, input)

	return builder.Build(node)
}

// chronosEncoderBlock holds layers for one T5 encoder block.
type chronosEncoderBlock[T tensor.Float] struct {
	attnNorm *normalization.LayerNormalization[T]
	attnQ    *core.Linear[T]
	attnK    *core.Linear[T]
	attnV    *core.Linear[T]
	attnO    *core.Linear[T]
	sdpa     *attention.ScaledDotProductAttention[T]
	ffnNorm  *normalization.LayerNormalization[T]
	ffnWi    *core.Linear[T]
	ffnWo    *core.Linear[T]
	gelu     *activations.Gelu[T]
}

// chronosDecoderBlock holds layers for one T5 decoder block.
type chronosDecoderBlock[T tensor.Float] struct {
	selfAttnNorm  *normalization.LayerNormalization[T]
	selfAttnQ     *core.Linear[T]
	selfAttnK     *core.Linear[T]
	selfAttnV     *core.Linear[T]
	selfAttnO     *core.Linear[T]
	selfAttnSDPA  *attention.ScaledDotProductAttention[T]
	crossAttnNorm *normalization.LayerNormalization[T]
	crossAttnQ    *core.Linear[T]
	crossAttnK    *core.Linear[T]
	crossAttnV    *core.Linear[T]
	crossAttnO    *core.Linear[T]
	crossAttnSDPA *attention.ScaledDotProductAttention[T]
	ffnNorm       *normalization.LayerNormalization[T]
	ffnWi         *core.Linear[T]
	ffnWo         *core.Linear[T]
	gelu          *activations.Gelu[T]
}

// chronosNode implements the full Chronos T5 forward pass as a single graph node.
type chronosNode[T tensor.Float] struct {
	cfg    *ChronosConfig
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Token embedding: [vocab_size, d_model]
	tokenEmbd *core.Linear[T]

	// Encoder blocks.
	encBlocks []chronosEncoderBlock[T]

	// Encoder final norm.
	encFinalNorm *normalization.LayerNormalization[T]

	// Decoder token embedding (may share weights with encoder).
	decTokenEmbd *core.Linear[T]

	// Decoder blocks.
	decBlocks []chronosDecoderBlock[T]

	// Decoder final norm.
	decFinalNorm *normalization.LayerNormalization[T]

	// LM head: [d_model, vocab_size]
	lmHead *core.Linear[T]
}

func newChronosNode[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *ChronosConfig,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*chronosNode[T], error) {
	headDim := cfg.DModel / cfg.NumHeads

	// Token embedding (used for encoder input).
	tokenEmbd, err := core.NewLinear[T]("chronos_token_embd", engine, ops, cfg.VocabSize, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create token embedding: %w", err)
	}
	loadLinearWeights(tensors, tokenEmbd, "chronos.token_embd.weight")

	// Build encoder blocks.
	encBlocks := make([]chronosEncoderBlock[T], cfg.NumEncoderLayers)
	for i := range cfg.NumEncoderLayers {
		prefix := fmt.Sprintf("chronos.enc.block.%d", i)
		block, bErr := newChronosEncoderBlock[T](prefix, cfg, headDim, engine, ops, tensors)
		if bErr != nil {
			return nil, fmt.Errorf("create encoder block %d: %w", i, bErr)
		}
		encBlocks[i] = block
	}

	// Encoder final norm.
	encFinalNorm, err := normalization.NewLayerNormalization[T](engine, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create encoder final norm: %w", err)
	}
	loadLayerNormWeights(tensors, encFinalNorm, "chronos.enc.final_norm")

	// Decoder token embedding.
	decTokenEmbd, err := core.NewLinear[T]("chronos_dec_token_embd", engine, ops, cfg.VocabSize, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create decoder token embedding: %w", err)
	}
	loadLinearWeights(tensors, decTokenEmbd, "chronos.dec.token_embd.weight")

	// Build decoder blocks.
	decBlocks := make([]chronosDecoderBlock[T], cfg.NumDecoderLayers)
	for i := range cfg.NumDecoderLayers {
		prefix := fmt.Sprintf("chronos.dec.block.%d", i)
		block, bErr := newChronosDecoderBlock[T](prefix, cfg, headDim, engine, ops, tensors)
		if bErr != nil {
			return nil, fmt.Errorf("create decoder block %d: %w", i, bErr)
		}
		decBlocks[i] = block
	}

	// Decoder final norm.
	decFinalNorm, err := normalization.NewLayerNormalization[T](engine, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create decoder final norm: %w", err)
	}
	loadLayerNormWeights(tensors, decFinalNorm, "chronos.dec.final_norm")

	// LM head.
	lmHead, err := core.NewLinear[T]("chronos_lm_head", engine, ops, cfg.DModel, cfg.VocabSize)
	if err != nil {
		return nil, fmt.Errorf("create LM head: %w", err)
	}
	loadLinearWeights(tensors, lmHead, "chronos.lm_head.weight")

	return &chronosNode[T]{
		cfg:          cfg,
		engine:       engine,
		ops:          ops,
		tokenEmbd:    tokenEmbd,
		encBlocks:    encBlocks,
		encFinalNorm: encFinalNorm,
		decTokenEmbd: decTokenEmbd,
		decBlocks:    decBlocks,
		decFinalNorm: decFinalNorm,
		lmHead:       lmHead,
	}, nil
}

func newChronosEncoderBlock[T tensor.Float](
	prefix string,
	cfg *ChronosConfig,
	headDim int,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	tensors map[string]*tensor.TensorNumeric[T],
) (chronosEncoderBlock[T], error) {
	var block chronosEncoderBlock[T]
	var err error

	// Self-attention norm.
	block.attnNorm, err = normalization.NewLayerNormalization[T](engine, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create attn norm: %w", err)
	}
	loadLayerNormWeights(tensors, block.attnNorm, prefix+".attn_norm")

	// Q, K, V, O projections.
	block.attnQ, err = core.NewLinear[T](prefix+"_attn_q", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create attn Q: %w", err)
	}
	loadLinearWeights(tensors, block.attnQ, prefix+".attn.q.weight")

	block.attnK, err = core.NewLinear[T](prefix+"_attn_k", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create attn K: %w", err)
	}
	loadLinearWeights(tensors, block.attnK, prefix+".attn.k.weight")

	block.attnV, err = core.NewLinear[T](prefix+"_attn_v", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create attn V: %w", err)
	}
	loadLinearWeights(tensors, block.attnV, prefix+".attn.v.weight")

	block.attnO, err = core.NewLinear[T](prefix+"_attn_o", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create attn O: %w", err)
	}
	loadLinearWeights(tensors, block.attnO, prefix+".attn.o.weight")

	// Bidirectional SDPA for encoder.
	block.sdpa = attention.NewBidirectionalSDPA[T](engine, headDim)

	// FFN norm.
	block.ffnNorm, err = normalization.NewLayerNormalization[T](engine, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create ffn norm: %w", err)
	}
	loadLayerNormWeights(tensors, block.ffnNorm, prefix+".ffn_norm")

	// FFN: wi (d_model -> d_ff), wo (d_ff -> d_model).
	block.ffnWi, err = core.NewLinear[T](prefix+"_ffn_wi", engine, ops, cfg.DModel, cfg.DFF)
	if err != nil {
		return block, fmt.Errorf("create ffn wi: %w", err)
	}
	loadLinearWeights(tensors, block.ffnWi, prefix+".ffn.wi.weight")

	block.ffnWo, err = core.NewLinear[T](prefix+"_ffn_wo", engine, ops, cfg.DFF, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create ffn wo: %w", err)
	}
	loadLinearWeights(tensors, block.ffnWo, prefix+".ffn.wo.weight")

	block.gelu = activations.NewGelu[T](engine, ops)

	return block, nil
}

func newChronosDecoderBlock[T tensor.Float](
	prefix string,
	cfg *ChronosConfig,
	headDim int,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	tensors map[string]*tensor.TensorNumeric[T],
) (chronosDecoderBlock[T], error) {
	var block chronosDecoderBlock[T]
	var err error

	// Self-attention norm.
	block.selfAttnNorm, err = normalization.NewLayerNormalization[T](engine, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create self attn norm: %w", err)
	}
	loadLayerNormWeights(tensors, block.selfAttnNorm, prefix+".self_attn_norm")

	// Self-attention Q, K, V, O.
	block.selfAttnQ, err = core.NewLinear[T](prefix+"_self_attn_q", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create self attn Q: %w", err)
	}
	loadLinearWeights(tensors, block.selfAttnQ, prefix+".self_attn.q.weight")

	block.selfAttnK, err = core.NewLinear[T](prefix+"_self_attn_k", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create self attn K: %w", err)
	}
	loadLinearWeights(tensors, block.selfAttnK, prefix+".self_attn.k.weight")

	block.selfAttnV, err = core.NewLinear[T](prefix+"_self_attn_v", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create self attn V: %w", err)
	}
	loadLinearWeights(tensors, block.selfAttnV, prefix+".self_attn.v.weight")

	block.selfAttnO, err = core.NewLinear[T](prefix+"_self_attn_o", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create self attn O: %w", err)
	}
	loadLinearWeights(tensors, block.selfAttnO, prefix+".self_attn.o.weight")

	// Causal SDPA for decoder self-attention.
	block.selfAttnSDPA = attention.NewScaledDotProductAttention[T](engine, headDim)

	// Cross-attention norm.
	block.crossAttnNorm, err = normalization.NewLayerNormalization[T](engine, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create cross attn norm: %w", err)
	}
	loadLayerNormWeights(tensors, block.crossAttnNorm, prefix+".cross_attn_norm")

	// Cross-attention Q, K, V, O.
	block.crossAttnQ, err = core.NewLinear[T](prefix+"_cross_attn_q", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create cross attn Q: %w", err)
	}
	loadLinearWeights(tensors, block.crossAttnQ, prefix+".cross_attn.q.weight")

	block.crossAttnK, err = core.NewLinear[T](prefix+"_cross_attn_k", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create cross attn K: %w", err)
	}
	loadLinearWeights(tensors, block.crossAttnK, prefix+".cross_attn.k.weight")

	block.crossAttnV, err = core.NewLinear[T](prefix+"_cross_attn_v", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create cross attn V: %w", err)
	}
	loadLinearWeights(tensors, block.crossAttnV, prefix+".cross_attn.v.weight")

	block.crossAttnO, err = core.NewLinear[T](prefix+"_cross_attn_o", engine, ops, cfg.DModel, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create cross attn O: %w", err)
	}
	loadLinearWeights(tensors, block.crossAttnO, prefix+".cross_attn.o.weight")

	// Bidirectional SDPA for cross-attention (attends to all encoder positions).
	block.crossAttnSDPA = attention.NewBidirectionalSDPA[T](engine, headDim)

	// FFN norm.
	block.ffnNorm, err = normalization.NewLayerNormalization[T](engine, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create ffn norm: %w", err)
	}
	loadLayerNormWeights(tensors, block.ffnNorm, prefix+".ffn_norm")

	// FFN: wi (d_model -> d_ff), wo (d_ff -> d_model).
	block.ffnWi, err = core.NewLinear[T](prefix+"_ffn_wi", engine, ops, cfg.DModel, cfg.DFF)
	if err != nil {
		return block, fmt.Errorf("create ffn wi: %w", err)
	}
	loadLinearWeights(tensors, block.ffnWi, prefix+".ffn.wi.weight")

	block.ffnWo, err = core.NewLinear[T](prefix+"_ffn_wo", engine, ops, cfg.DFF, cfg.DModel)
	if err != nil {
		return block, fmt.Errorf("create ffn wo: %w", err)
	}
	loadLinearWeights(tensors, block.ffnWo, prefix+".ffn.wo.weight")

	block.gelu = activations.NewGelu[T](engine, ops)

	return block, nil
}

// loadLayerNormWeights loads weight and bias tensors into a LayerNormalization layer.
func loadLayerNormWeights[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	ln *normalization.LayerNormalization[T],
	prefix string,
) {
	params := ln.Parameters()
	if len(params) < 2 {
		return
	}
	if w, ok := tensors[prefix+".weight"]; ok {
		params[0].Value = w // gamma
	}
	if b, ok := tensors[prefix+".bias"]; ok {
		params[1].Value = b // beta
	}
}

func (n *chronosNode[T]) OpType() string { return "Chronos" }

func (n *chronosNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_encoder_layers": n.cfg.NumEncoderLayers,
		"num_decoder_layers": n.cfg.NumDecoderLayers,
		"d_model":            n.cfg.DModel,
		"num_heads":          n.cfg.NumHeads,
		"d_ff":               n.cfg.DFF,
		"vocab_size":         n.cfg.VocabSize,
		"horizon":            n.cfg.Horizon,
	}
}

func (n *chronosNode[T]) OutputShape() []int {
	return []int{-1, n.cfg.Horizon, n.cfg.VocabSize}
}

func (n *chronosNode[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, n.tokenEmbd.Parameters()...)
	for _, block := range n.encBlocks {
		params = append(params, block.attnNorm.Parameters()...)
		params = append(params, block.attnQ.Parameters()...)
		params = append(params, block.attnK.Parameters()...)
		params = append(params, block.attnV.Parameters()...)
		params = append(params, block.attnO.Parameters()...)
		params = append(params, block.ffnNorm.Parameters()...)
		params = append(params, block.ffnWi.Parameters()...)
		params = append(params, block.ffnWo.Parameters()...)
	}
	params = append(params, n.encFinalNorm.Parameters()...)
	params = append(params, n.decTokenEmbd.Parameters()...)
	for _, block := range n.decBlocks {
		params = append(params, block.selfAttnNorm.Parameters()...)
		params = append(params, block.selfAttnQ.Parameters()...)
		params = append(params, block.selfAttnK.Parameters()...)
		params = append(params, block.selfAttnV.Parameters()...)
		params = append(params, block.selfAttnO.Parameters()...)
		params = append(params, block.crossAttnNorm.Parameters()...)
		params = append(params, block.crossAttnQ.Parameters()...)
		params = append(params, block.crossAttnK.Parameters()...)
		params = append(params, block.crossAttnV.Parameters()...)
		params = append(params, block.crossAttnO.Parameters()...)
		params = append(params, block.ffnNorm.Parameters()...)
		params = append(params, block.ffnWi.Parameters()...)
		params = append(params, block.ffnWo.Parameters()...)
	}
	params = append(params, n.decFinalNorm.Parameters()...)
	params = append(params, n.lmHead.Parameters()...)
	return params
}

// Forward processes [batch, seq_len] tokenized input and produces [batch, horizon, vocab_size].
//
// The forward pass:
//  1. Embeds encoder input tokens via one-hot lookup
//  2. Runs through encoder self-attention blocks (bidirectional)
//  3. Creates decoder input (zeros for autoregressive start)
//  4. Runs through decoder self-attention + cross-attention blocks
//  5. Projects to vocabulary logits via LM head
func (n *chronosNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Chronos expects 1 input, got %d", len(inputs))
	}
	tokenIDs := inputs[0]
	shape := tokenIDs.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("Chronos input must be 2D [batch, seq_len], got shape %v", shape)
	}

	batch, seqLen := shape[0], shape[1]
	d := n.cfg.DModel
	numHeads := n.cfg.NumHeads
	headDim := d / numHeads
	horizon := n.cfg.Horizon

	// 1. Embed encoder input: one-hot [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size] -> linear -> [batch, seq_len, d_model]
	oneHot, err := oneHotEncode[T](tokenIDs, n.cfg.VocabSize, n.ops)
	if err != nil {
		return nil, fmt.Errorf("one-hot encode: %w", err)
	}
	flat, err := n.engine.Reshape(ctx, oneHot, []int{batch * seqLen, n.cfg.VocabSize})
	if err != nil {
		return nil, fmt.Errorf("reshape one-hot: %w", err)
	}
	encHidden, err := n.tokenEmbd.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("token embedding: %w", err)
	}
	encHidden, err = n.engine.Reshape(ctx, encHidden, []int{batch, seqLen, d})
	if err != nil {
		return nil, fmt.Errorf("reshape embedded: %w", err)
	}

	// T5 scales embeddings by sqrt(d_model).
	scale := n.ops.FromFloat64(math.Sqrt(float64(d)))
	encHidden, err = n.engine.MulScalar(ctx, encHidden, scale)
	if err != nil {
		return nil, fmt.Errorf("scale embeddings: %w", err)
	}

	// 2. Encoder blocks.
	for i, block := range n.encBlocks {
		encHidden, err = chronosEncoderBlockForward[T](ctx, n.engine, block, encHidden, batch, seqLen, d, numHeads, headDim)
		if err != nil {
			return nil, fmt.Errorf("encoder block %d: %w", i, err)
		}
	}

	// Encoder final norm.
	encHidden, err = n.encFinalNorm.Forward(ctx, encHidden)
	if err != nil {
		return nil, fmt.Errorf("encoder final norm: %w", err)
	}

	// 3. Decoder input: start with zero embeddings for all horizon positions.
	decData := make([]T, batch*horizon*d)
	decHidden, err := tensor.New[T]([]int{batch, horizon, d}, decData)
	if err != nil {
		return nil, fmt.Errorf("create decoder input: %w", err)
	}

	// 4. Decoder blocks.
	for i, block := range n.decBlocks {
		decHidden, err = chronosDecoderBlockForward[T](ctx, n.engine, block, decHidden, encHidden, batch, horizon, seqLen, d, numHeads, headDim)
		if err != nil {
			return nil, fmt.Errorf("decoder block %d: %w", i, err)
		}
	}

	// Decoder final norm.
	decHidden, err = n.decFinalNorm.Forward(ctx, decHidden)
	if err != nil {
		return nil, fmt.Errorf("decoder final norm: %w", err)
	}

	// 5. LM head: [batch, horizon, d_model] -> [batch, horizon, vocab_size]
	flatDec, err := n.engine.Reshape(ctx, decHidden, []int{batch * horizon, d})
	if err != nil {
		return nil, fmt.Errorf("reshape for lm head: %w", err)
	}
	logits, err := n.lmHead.Forward(ctx, flatDec)
	if err != nil {
		return nil, fmt.Errorf("lm head: %w", err)
	}
	logits, err = n.engine.Reshape(ctx, logits, []int{batch, horizon, n.cfg.VocabSize})
	if err != nil {
		return nil, fmt.Errorf("reshape logits: %w", err)
	}

	return logits, nil
}

// chronosEncoderBlockForward runs one encoder block: pre-norm self-attention + pre-norm FFN.
func chronosEncoderBlockForward[T tensor.Float](
	ctx context.Context,
	engine compute.Engine[T],
	block chronosEncoderBlock[T],
	x *tensor.TensorNumeric[T],
	batch, seqLen, d, numHeads, headDim int,
) (*tensor.TensorNumeric[T], error) {
	// Pre-norm for self-attention.
	normed, err := block.attnNorm.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("attn norm: %w", err)
	}

	// Self-attention.
	attnOut, err := multiHeadAttention[T](ctx, engine, normed, normed, normed,
		block.attnQ, block.attnK, block.attnV, block.attnO,
		block.sdpa, batch, seqLen, seqLen, d, numHeads, headDim)
	if err != nil {
		return nil, fmt.Errorf("self attention: %w", err)
	}

	// Residual.
	x, err = engine.Add(ctx, x, attnOut)
	if err != nil {
		return nil, fmt.Errorf("attn residual: %w", err)
	}

	// Pre-norm for FFN.
	normed, err = block.ffnNorm.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ffn norm: %w", err)
	}

	// FFN: wi -> gelu -> wo.
	ffnOut, err := chronosFFNForward[T](ctx, engine, block.ffnWi, block.ffnWo, block.gelu, normed, batch, seqLen, d)
	if err != nil {
		return nil, fmt.Errorf("ffn: %w", err)
	}

	// Residual.
	x, err = engine.Add(ctx, x, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("ffn residual: %w", err)
	}

	return x, nil
}

// chronosDecoderBlockForward runs one decoder block: self-attention + cross-attention + FFN.
func chronosDecoderBlockForward[T tensor.Float](
	ctx context.Context,
	engine compute.Engine[T],
	block chronosDecoderBlock[T],
	x, encOut *tensor.TensorNumeric[T],
	batch, decLen, encLen, d, numHeads, headDim int,
) (*tensor.TensorNumeric[T], error) {
	// Pre-norm for self-attention.
	normed, err := block.selfAttnNorm.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("self attn norm: %w", err)
	}

	// Causal self-attention.
	selfAttnOut, err := multiHeadAttention[T](ctx, engine, normed, normed, normed,
		block.selfAttnQ, block.selfAttnK, block.selfAttnV, block.selfAttnO,
		block.selfAttnSDPA, batch, decLen, decLen, d, numHeads, headDim)
	if err != nil {
		return nil, fmt.Errorf("self attention: %w", err)
	}

	// Residual.
	x, err = engine.Add(ctx, x, selfAttnOut)
	if err != nil {
		return nil, fmt.Errorf("self attn residual: %w", err)
	}

	// Pre-norm for cross-attention.
	normed, err = block.crossAttnNorm.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("cross attn norm: %w", err)
	}

	// Cross-attention: Q from decoder, K/V from encoder.
	crossAttnOut, err := multiHeadAttention[T](ctx, engine, normed, encOut, encOut,
		block.crossAttnQ, block.crossAttnK, block.crossAttnV, block.crossAttnO,
		block.crossAttnSDPA, batch, decLen, encLen, d, numHeads, headDim)
	if err != nil {
		return nil, fmt.Errorf("cross attention: %w", err)
	}

	// Residual.
	x, err = engine.Add(ctx, x, crossAttnOut)
	if err != nil {
		return nil, fmt.Errorf("cross attn residual: %w", err)
	}

	// Pre-norm for FFN.
	normed, err = block.ffnNorm.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ffn norm: %w", err)
	}

	// FFN.
	ffnOut, err := chronosFFNForward[T](ctx, engine, block.ffnWi, block.ffnWo, block.gelu, normed, batch, decLen, d)
	if err != nil {
		return nil, fmt.Errorf("ffn: %w", err)
	}

	// Residual.
	x, err = engine.Add(ctx, x, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("ffn residual: %w", err)
	}

	return x, nil
}

// multiHeadAttention computes multi-head attention given Q/K/V source tensors
// and projection layers. The SDPA layer handles masking (causal vs bidirectional).
func multiHeadAttention[T tensor.Float](
	ctx context.Context,
	engine compute.Engine[T],
	qSrc, kSrc, vSrc *tensor.TensorNumeric[T],
	projQ, projK, projV, projO *core.Linear[T],
	sdpa *attention.ScaledDotProductAttention[T],
	batch, qLen, kvLen, d, numHeads, headDim int,
) (*tensor.TensorNumeric[T], error) {
	// Project Q, K, V: [batch, seq, d] -> [batch*seq, d] -> linear -> [batch, seq, d]
	flatQ, err := engine.Reshape(ctx, qSrc, []int{batch * qLen, d})
	if err != nil {
		return nil, fmt.Errorf("reshape Q: %w", err)
	}
	q, err := projQ.Forward(ctx, flatQ)
	if err != nil {
		return nil, fmt.Errorf("project Q: %w", err)
	}

	flatK, err := engine.Reshape(ctx, kSrc, []int{batch * kvLen, d})
	if err != nil {
		return nil, fmt.Errorf("reshape K: %w", err)
	}
	k, err := projK.Forward(ctx, flatK)
	if err != nil {
		return nil, fmt.Errorf("project K: %w", err)
	}

	flatV, err := engine.Reshape(ctx, vSrc, []int{batch * kvLen, d})
	if err != nil {
		return nil, fmt.Errorf("reshape V: %w", err)
	}
	v, err := projV.Forward(ctx, flatV)
	if err != nil {
		return nil, fmt.Errorf("project V: %w", err)
	}

	// Reshape to multi-head: [batch*numHeads, seq, headDim]
	q, err = engine.Reshape(ctx, q, []int{batch, qLen, numHeads, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape Q heads: %w", err)
	}
	q, err = engine.Transpose(ctx, q, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("transpose Q: %w", err)
	}
	q, err = engine.Reshape(ctx, q, []int{batch * numHeads, qLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("merge batch-heads Q: %w", err)
	}

	k, err = engine.Reshape(ctx, k, []int{batch, kvLen, numHeads, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape K heads: %w", err)
	}
	k, err = engine.Transpose(ctx, k, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("transpose K: %w", err)
	}
	k, err = engine.Reshape(ctx, k, []int{batch * numHeads, kvLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("merge batch-heads K: %w", err)
	}

	v, err = engine.Reshape(ctx, v, []int{batch, kvLen, numHeads, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape V heads: %w", err)
	}
	v, err = engine.Transpose(ctx, v, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("transpose V: %w", err)
	}
	v, err = engine.Reshape(ctx, v, []int{batch * numHeads, kvLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("merge batch-heads V: %w", err)
	}

	// SDPA: [batch*numHeads, qLen, headDim] -> [batch*numHeads, qLen, headDim]
	attnOut, err := sdpa.Forward(ctx, q, k, v, nil)
	if err != nil {
		return nil, fmt.Errorf("sdpa: %w", err)
	}

	// Reshape back: [batch, numHeads, qLen, headDim] -> [batch, qLen, d]
	attnOut, err = engine.Reshape(ctx, attnOut, []int{batch, numHeads, qLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape attn output: %w", err)
	}
	attnOut, err = engine.Transpose(ctx, attnOut, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("transpose attn output: %w", err)
	}
	attnOut, err = engine.Reshape(ctx, attnOut, []int{batch * qLen, d})
	if err != nil {
		return nil, fmt.Errorf("flatten attn output: %w", err)
	}

	// Output projection.
	out, err := projO.Forward(ctx, attnOut)
	if err != nil {
		return nil, fmt.Errorf("output projection: %w", err)
	}

	out, err = engine.Reshape(ctx, out, []int{batch, qLen, d})
	if err != nil {
		return nil, fmt.Errorf("reshape output: %w", err)
	}

	return out, nil
}

// chronosFFNForward runs the T5 FFN: wi -> gelu -> wo.
func chronosFFNForward[T tensor.Float](
	ctx context.Context,
	engine compute.Engine[T],
	wi, wo *core.Linear[T],
	gelu *activations.Gelu[T],
	x *tensor.TensorNumeric[T],
	batch, seqLen, d int,
) (*tensor.TensorNumeric[T], error) {
	// Flatten: [batch, seq_len, d] -> [batch*seq_len, d]
	flat, err := engine.Reshape(ctx, x, []int{batch * seqLen, d})
	if err != nil {
		return nil, fmt.Errorf("flatten for ffn: %w", err)
	}

	// wi: [batch*seq_len, d] -> [batch*seq_len, d_ff]
	h, err := wi.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("ffn wi: %w", err)
	}

	// GELU activation.
	h, err = gelu.Forward(ctx, h)
	if err != nil {
		return nil, fmt.Errorf("ffn gelu: %w", err)
	}

	// wo: [batch*seq_len, d_ff] -> [batch*seq_len, d]
	h, err = wo.Forward(ctx, h)
	if err != nil {
		return nil, fmt.Errorf("ffn wo: %w", err)
	}

	// Reshape back: [batch, seq_len, d]
	h, err = engine.Reshape(ctx, h, []int{batch, seqLen, d})
	if err != nil {
		return nil, fmt.Errorf("reshape ffn output: %w", err)
	}

	return h, nil
}

// oneHotEncode converts a 2D tensor of integer token IDs [batch, seq_len] to
// one-hot vectors [batch, seq_len, vocab_size].
func oneHotEncode[T tensor.Float](
	tokenIDs *tensor.TensorNumeric[T],
	vocabSize int,
	ops numeric.Arithmetic[T],
) (*tensor.TensorNumeric[T], error) {
	shape := tokenIDs.Shape()
	batch, seqLen := shape[0], shape[1]
	data := tokenIDs.Data()

	out := make([]T, batch*seqLen*vocabSize)
	one := ops.FromFloat64(1.0)

	for b := range batch {
		for s := range seqLen {
			idx := int(data[b*seqLen+s])
			if idx < 0 {
				idx = 0
			}
			if idx >= vocabSize {
				idx = vocabSize - 1
			}
			out[b*seqLen*vocabSize+s*vocabSize+idx] = one
		}
	}

	return tensor.New[T]([]int{batch, seqLen, vocabSize}, out)
}

func (n *chronosNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
