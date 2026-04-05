// Package vision provides vision-related neural network layers.
package vision

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/layers/normalization"
)

// CLIPEncoderConfig holds configuration for a CLIP vision encoder.
type CLIPEncoderConfig struct {
	ImageSize   int // Input image size (square, e.g. 224).
	PatchSize   int // Patch size for patch embedding (e.g. 14).
	HiddenDim   int // Hidden dimension throughout the encoder.
	NumHeads    int // Number of attention heads per transformer block.
	NumLayers   int // Number of transformer encoder blocks.
	NumChannels int // Number of input channels (default 3 for RGB).
}

// NumPatches returns the number of patches (excluding class token).
func (c CLIPEncoderConfig) NumPatches() int {
	return (c.ImageSize / c.PatchSize) * (c.ImageSize / c.PatchSize)
}

// visionBlock holds the layers for a single CLIP transformer encoder block.
type visionBlock[T tensor.Numeric] struct {
	ln1   *normalization.LayerNormalization[T]
	qProj *core.Linear[T]
	kProj *core.Linear[T]
	vProj *core.Linear[T]
	oProj *core.Linear[T]
	sdpa  *attention.ScaledDotProductAttention[T]
	ln2   *normalization.LayerNormalization[T]
	ffn1  *core.Dense[T]
	ffn2  *core.Dense[T]
}

// CLIPEncoder implements a CLIP ViT (Vision Transformer) encoder.
//
// Architecture:
//
//	PatchEmbed -> [CLS] + PatchEmbeddings + PosEmbed -> [LN -> SelfAttn -> Add -> LN -> FFN(QuickGELU) -> Add] x N -> LN
//
// Input shape:  [batch, channels, height, width] (pixel values normalized to [-1, 1])
// Output shape: [batch, numPatches+1, hiddenDim]
type CLIPEncoder[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	cfg    CLIPEncoderConfig

	// Patch embedding: projects [batch, C, H, W] -> [batch, numPatches, hiddenDim]
	patchEmbedWeight *graph.Parameter[T] // [hiddenDim, C*patchSize*patchSize]
	patchEmbedBias   *graph.Parameter[T] // [hiddenDim]

	classEmbedding    *graph.Parameter[T] // [1, 1, hiddenDim]
	positionEmbedding *graph.Parameter[T] // [1, numPatches+1, hiddenDim]

	blocks []visionBlock[T]

	lnPost *normalization.LayerNormalization[T]
}

// NewCLIPEncoder creates a new CLIP ViT encoder.
func NewCLIPEncoder[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	cfg CLIPEncoderConfig,
) (*CLIPEncoder[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if cfg.ImageSize <= 0 {
		return nil, fmt.Errorf("ImageSize must be positive, got %d", cfg.ImageSize)
	}
	if cfg.PatchSize <= 0 {
		return nil, fmt.Errorf("PatchSize must be positive, got %d", cfg.PatchSize)
	}
	if cfg.HiddenDim <= 0 {
		return nil, fmt.Errorf("HiddenDim must be positive, got %d", cfg.HiddenDim)
	}
	if cfg.NumHeads <= 0 {
		return nil, fmt.Errorf("NumHeads must be positive, got %d", cfg.NumHeads)
	}
	if cfg.NumLayers <= 0 {
		return nil, fmt.Errorf("NumLayers must be positive, got %d", cfg.NumLayers)
	}
	if cfg.ImageSize%cfg.PatchSize != 0 {
		return nil, fmt.Errorf("ImageSize (%d) must be divisible by PatchSize (%d)", cfg.ImageSize, cfg.PatchSize)
	}
	if cfg.HiddenDim%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("HiddenDim (%d) must be divisible by NumHeads (%d)", cfg.HiddenDim, cfg.NumHeads)
	}
	if cfg.NumChannels <= 0 {
		cfg.NumChannels = 3
	}

	numPatches := cfg.NumPatches()
	patchDim := cfg.NumChannels * cfg.PatchSize * cfg.PatchSize

	// Patch embedding projection weight and bias.
	peW, err := tensor.New[T]([]int{cfg.HiddenDim, patchDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("patch_embed weight: %w", err)
	}
	peB, err := tensor.New[T]([]int{cfg.HiddenDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("patch_embed bias: %w", err)
	}

	// Class embedding [1, 1, hiddenDim].
	clsEmb, err := tensor.New[T]([]int{1, 1, cfg.HiddenDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("class_embedding: %w", err)
	}

	// Position embedding [1, numPatches+1, hiddenDim].
	posEmb, err := tensor.New[T]([]int{1, numPatches + 1, cfg.HiddenDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("position_embedding: %w", err)
	}

	// Build transformer blocks.
	blocks := make([]visionBlock[T], cfg.NumLayers)
	ffnDim := cfg.HiddenDim * 4

	for i := range blocks {
		prefix := fmt.Sprintf("%s_block%d", name, i)

		ln1, err := normalization.NewLayerNormalization[T](engine, cfg.HiddenDim)
		if err != nil {
			return nil, fmt.Errorf("block %d ln1: %w", i, err)
		}
		qProj, err := core.NewLinear[T](prefix+"_q", engine, ops, cfg.HiddenDim, cfg.HiddenDim)
		if err != nil {
			return nil, fmt.Errorf("block %d q_proj: %w", i, err)
		}
		kProj, err := core.NewLinear[T](prefix+"_k", engine, ops, cfg.HiddenDim, cfg.HiddenDim)
		if err != nil {
			return nil, fmt.Errorf("block %d k_proj: %w", i, err)
		}
		vProj, err := core.NewLinear[T](prefix+"_v", engine, ops, cfg.HiddenDim, cfg.HiddenDim)
		if err != nil {
			return nil, fmt.Errorf("block %d v_proj: %w", i, err)
		}
		oProj, err := core.NewLinear[T](prefix+"_o", engine, ops, cfg.HiddenDim, cfg.HiddenDim)
		if err != nil {
			return nil, fmt.Errorf("block %d o_proj: %w", i, err)
		}
		ln2, err := normalization.NewLayerNormalization[T](engine, cfg.HiddenDim)
		if err != nil {
			return nil, fmt.Errorf("block %d ln2: %w", i, err)
		}
		ffn1, err := core.NewDense[T](prefix+"_ffn1", engine, ops, cfg.HiddenDim, ffnDim)
		if err != nil {
			return nil, fmt.Errorf("block %d ffn1: %w", i, err)
		}
		ffn2, err := core.NewDense[T](prefix+"_ffn2", engine, ops, ffnDim, cfg.HiddenDim)
		if err != nil {
			return nil, fmt.Errorf("block %d ffn2: %w", i, err)
		}

		sdpa := attention.NewBidirectionalSDPA[T](engine, cfg.HiddenDim/cfg.NumHeads)

		blocks[i] = visionBlock[T]{
			ln1: ln1, qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
			sdpa: sdpa, ln2: ln2, ffn1: ffn1, ffn2: ffn2,
		}
	}

	lnPost, err := normalization.NewLayerNormalization[T](engine, cfg.HiddenDim)
	if err != nil {
		return nil, fmt.Errorf("ln_post: %w", err)
	}

	return &CLIPEncoder[T]{
		engine:            engine,
		ops:               ops,
		cfg:               cfg,
		patchEmbedWeight:  &graph.Parameter[T]{Name: name + "_patch_embed.weight", Value: peW},
		patchEmbedBias:    &graph.Parameter[T]{Name: name + "_patch_embed.bias", Value: peB},
		classEmbedding:    &graph.Parameter[T]{Name: name + "_class_embedding", Value: clsEmb},
		positionEmbedding: &graph.Parameter[T]{Name: name + "_position_embedding", Value: posEmb},
		blocks:            blocks,
		lnPost:            lnPost,
	}, nil
}

func (e *CLIPEncoder[T]) OpType() string { return "CLIPEncoder" }

func (e *CLIPEncoder[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"image_size": e.cfg.ImageSize,
		"patch_size": e.cfg.PatchSize,
		"hidden_dim": e.cfg.HiddenDim,
		"num_heads":  e.cfg.NumHeads,
		"num_layers": e.cfg.NumLayers,
	}
}

func (e *CLIPEncoder[T]) OutputShape() []int {
	return []int{-1, e.cfg.NumPatches() + 1, e.cfg.HiddenDim}
}

// Forward runs the CLIP vision encoder.
// Input: [batch, channels, height, width] pixel values.
// Output: [batch, numPatches+1, hiddenDim] vision embeddings.
func (e *CLIPEncoder[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("CLIPEncoder requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("CLIPEncoder input must be 4D [batch, C, H, W], got %v", shape)
	}

	batch := shape[0]
	numPatches := e.cfg.NumPatches()
	patchSize := e.cfg.PatchSize
	channels := e.cfg.NumChannels
	hiddenDim := e.cfg.HiddenDim
	gridSize := e.cfg.ImageSize / patchSize
	patchDim := channels * patchSize * patchSize

	// Extract patches using engine ops:
	// [batch, C, H, W] -> [batch, C, gridY, P, gridX, P]
	reshaped, err := e.engine.Reshape(ctx, input, []int{batch, channels, gridSize, patchSize, gridSize, patchSize})
	if err != nil {
		return nil, fmt.Errorf("reshape to grid: %w", err)
	}
	// -> [batch, gridY, gridX, C, P, P]
	permuted, err := e.engine.Transpose(ctx, reshaped, []int{0, 2, 4, 1, 3, 5})
	if err != nil {
		return nil, fmt.Errorf("transpose patches: %w", err)
	}
	// -> [batch*numPatches, C*P*P]
	patchesTensor, err := e.engine.Reshape(ctx, permuted, []int{batch * numPatches, patchDim})
	if err != nil {
		return nil, fmt.Errorf("reshape patches flat: %w", err)
	}

	// Linear projection: patches [batch*numPatches, patchDim] @ weight^T [patchDim, hiddenDim] + bias
	weightT, err := e.engine.Transpose(ctx, e.patchEmbedWeight.Value, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("transpose patch_embed weight: %w", err)
	}
	embeddingsTensor, err := e.engine.MatMul(ctx, patchesTensor, weightT)
	if err != nil {
		return nil, fmt.Errorf("patch_embed matmul: %w", err)
	}
	embeddingsTensor, err = e.engine.Add(ctx, embeddingsTensor, e.patchEmbedBias.Value)
	if err != nil {
		return nil, fmt.Errorf("patch_embed bias: %w", err)
	}

	// Reshape embeddings to [batch, numPatches, hiddenDim] for concatenation.
	embeddingsTensor, err = e.engine.Reshape(ctx, embeddingsTensor, []int{batch, numPatches, hiddenDim})
	if err != nil {
		return nil, fmt.Errorf("reshape embeddings: %w", err)
	}

	// Prepend class token: expand [1, 1, hiddenDim] to [batch, 1, hiddenDim] via Repeat, then Concat.
	seqLen := numPatches + 1
	clsTokens := e.classEmbedding.Value
	if batch > 1 {
		clsTokens, err = e.engine.Repeat(ctx, clsTokens, 0, batch)
		if err != nil {
			return nil, fmt.Errorf("repeat class token: %w", err)
		}
	}
	withCls, err := e.engine.Concat(ctx, []*tensor.TensorNumeric[T]{clsTokens, embeddingsTensor}, 1)
	if err != nil {
		return nil, fmt.Errorf("concat class token: %w", err)
	}

	// Add position embedding: [1, numPatches+1, hiddenDim] broadcasts over batch.
	x, err := e.engine.Add(ctx, withCls, e.positionEmbedding.Value)
	if err != nil {
		return nil, fmt.Errorf("add position embedding: %w", err)
	}

	// Transformer encoder blocks.
	for i := range e.blocks {
		x, err = e.forwardBlock(ctx, &e.blocks[i], x, batch, seqLen)
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", i, err)
		}
	}

	// Post layer norm.
	x, err = e.lnPost.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ln_post: %w", err)
	}

	return x, nil
}

// forwardBlock runs a single CLIP transformer encoder block.
func (e *CLIPEncoder[T]) forwardBlock(
	ctx context.Context,
	block *visionBlock[T],
	x *tensor.TensorNumeric[T],
	batch, seqLen int,
) (*tensor.TensorNumeric[T], error) {
	hiddenDim := e.cfg.HiddenDim
	numHeads := e.cfg.NumHeads
	headDim := hiddenDim / numHeads

	// Pre-norm.
	residual := x
	normed, err := block.ln1.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ln1: %w", err)
	}

	// Reshape to 2D for linear projections.
	normed2D, err := e.engine.Reshape(ctx, normed, []int{batch * seqLen, hiddenDim})
	if err != nil {
		return nil, err
	}

	q, err := block.qProj.Forward(ctx, normed2D)
	if err != nil {
		return nil, fmt.Errorf("q_proj: %w", err)
	}
	k, err := block.kProj.Forward(ctx, normed2D)
	if err != nil {
		return nil, fmt.Errorf("k_proj: %w", err)
	}
	v, err := block.vProj.Forward(ctx, normed2D)
	if err != nil {
		return nil, fmt.Errorf("v_proj: %w", err)
	}

	// Multi-head self-attention via ScaledDotProductAttention.
	// Reshape Q, K, V from [batch*seqLen, hiddenDim] to [batch*numHeads, seqLen, headDim]
	// which is the 3D format expected by SDPA.
	q, err = e.engine.Reshape(ctx, q, []int{batch, seqLen, numHeads, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape q: %w", err)
	}
	q, err = e.engine.Transpose(ctx, q, []int{0, 2, 1, 3}) // [batch, numHeads, seqLen, headDim]
	if err != nil {
		return nil, fmt.Errorf("transpose q: %w", err)
	}
	q, err = e.engine.Reshape(ctx, q, []int{batch * numHeads, seqLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape q 3d: %w", err)
	}

	k, err = e.engine.Reshape(ctx, k, []int{batch, seqLen, numHeads, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape k: %w", err)
	}
	k, err = e.engine.Transpose(ctx, k, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("transpose k: %w", err)
	}
	k, err = e.engine.Reshape(ctx, k, []int{batch * numHeads, seqLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape k 3d: %w", err)
	}

	v, err = e.engine.Reshape(ctx, v, []int{batch, seqLen, numHeads, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape v: %w", err)
	}
	v, err = e.engine.Transpose(ctx, v, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("transpose v: %w", err)
	}
	v, err = e.engine.Reshape(ctx, v, []int{batch * numHeads, seqLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape v 3d: %w", err)
	}

	// SDPA: [batch*numHeads, seqLen, headDim] -> [batch*numHeads, seqLen, headDim]
	attnResult, err := block.sdpa.Forward(ctx, q, k, v, nil)
	if err != nil {
		return nil, fmt.Errorf("sdpa: %w", err)
	}

	// Reshape back: [batch*numHeads, seqLen, headDim] -> [batch, numHeads, seqLen, headDim]
	// -> transpose to [batch, seqLen, numHeads, headDim] -> [batch*seqLen, hiddenDim]
	attnResult, err = e.engine.Reshape(ctx, attnResult, []int{batch, numHeads, seqLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("reshape attn 4d: %w", err)
	}
	attnResult, err = e.engine.Transpose(ctx, attnResult, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("transpose attn out: %w", err)
	}
	attnResult, err = e.engine.Reshape(ctx, attnResult, []int{batch * seqLen, hiddenDim})
	if err != nil {
		return nil, fmt.Errorf("reshape attn out: %w", err)
	}

	// Output projection.
	projected, err := block.oProj.Forward(ctx, attnResult)
	if err != nil {
		return nil, fmt.Errorf("o_proj: %w", err)
	}

	// Residual connection.
	projected3D, err := e.engine.Reshape(ctx, projected, []int{batch, seqLen, hiddenDim})
	if err != nil {
		return nil, fmt.Errorf("reshape projected: %w", err)
	}
	x, err = e.engine.Add(ctx, residual, projected3D)
	if err != nil {
		return nil, fmt.Errorf("residual1: %w", err)
	}

	// Pre-norm FFN.
	residual = x
	normed, err = block.ln2.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ln2: %w", err)
	}

	normed2D, err = e.engine.Reshape(ctx, normed, []int{batch * seqLen, hiddenDim})
	if err != nil {
		return nil, err
	}
	ffnOut, err := block.ffn1.Forward(ctx, normed2D)
	if err != nil {
		return nil, fmt.Errorf("ffn1: %w", err)
	}
	ffnOut, err = e.quickGELU(ctx, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("quickgelu: %w", err)
	}

	ffnOut, err = block.ffn2.Forward(ctx, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("ffn2: %w", err)
	}

	ffn3D, err := e.engine.Reshape(ctx, ffnOut, []int{batch, seqLen, hiddenDim})
	if err != nil {
		return nil, err
	}
	x, err = e.engine.Add(ctx, residual, ffn3D)
	if err != nil {
		return nil, fmt.Errorf("residual2: %w", err)
	}

	return x, nil
}

// Parameters returns all trainable parameters from the CLIP encoder.
func (e *CLIPEncoder[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, e.patchEmbedWeight, e.patchEmbedBias)
	params = append(params, e.classEmbedding, e.positionEmbedding)
	for i := range e.blocks {
		b := &e.blocks[i]
		params = append(params, b.ln1.Parameters()...)
		params = append(params, b.qProj.Parameters()...)
		params = append(params, b.kProj.Parameters()...)
		params = append(params, b.vProj.Parameters()...)
		params = append(params, b.oProj.Parameters()...)
		params = append(params, b.ln2.Parameters()...)
		params = append(params, b.ffn1.Parameters()...)
		params = append(params, b.ffn2.Parameters()...)
	}
	params = append(params, e.lnPost.Parameters()...)
	return params
}

func (e *CLIPEncoder[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// quickGELU applies QuickGELU(x) = x * sigmoid(1.702 * x) using engine ops.
func (e *CLIPEncoder[T]) quickGELU(ctx context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	scaled, err := e.engine.MulScalar(ctx, x, e.ops.FromFloat64(1.702))
	if err != nil {
		return nil, fmt.Errorf("quickgelu mul coeff: %w", err)
	}
	sig, err := functional.Sigmoid(ctx, e.engine, e.ops, scaled)
	if err != nil {
		return nil, fmt.Errorf("quickgelu sigmoid: %w", err)
	}
	result, err := e.engine.Mul(ctx, x, sig)
	if err != nil {
		return nil, fmt.Errorf("quickgelu mul: %w", err)
	}
	return result, nil
}
