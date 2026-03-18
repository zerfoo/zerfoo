// Package vision provides vision-related neural network layers.
package vision

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
)

// CLIPEncoderConfig holds configuration for a CLIP vision encoder.
type CLIPEncoderConfig struct {
	ImageSize  int // Input image size (square, e.g. 224).
	PatchSize  int // Patch size for patch embedding (e.g. 14).
	HiddenDim  int // Hidden dimension throughout the encoder.
	NumHeads   int // Number of attention heads per transformer block.
	NumLayers  int // Number of transformer encoder blocks.
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

		blocks[i] = visionBlock[T]{
			ln1: ln1, qProj: qProj, kProj: kProj, vProj: vProj, oProj: oProj,
			ln2: ln2, ffn1: ffn1, ffn2: ffn2,
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

	// Extract patches: [batch, C, H, W] -> [batch, numPatches, C*P*P]
	inputData := input.Data()
	patches := make([]T, batch*numPatches*patchDim)
	for b := 0; b < batch; b++ {
		for gy := 0; gy < gridSize; gy++ {
			for gx := 0; gx < gridSize; gx++ {
				patchIdx := gy*gridSize + gx
				for c := 0; c < channels; c++ {
					for py := 0; py < patchSize; py++ {
						for px := 0; px < patchSize; px++ {
							srcY := gy*patchSize + py
							srcX := gx*patchSize + px
							srcIdx := b*channels*e.cfg.ImageSize*e.cfg.ImageSize + c*e.cfg.ImageSize*e.cfg.ImageSize + srcY*e.cfg.ImageSize + srcX
							dstIdx := b*numPatches*patchDim + patchIdx*patchDim + c*patchSize*patchSize + py*patchSize + px
							patches[dstIdx] = inputData[srcIdx]
						}
					}
				}
			}
		}
	}

	// Linear projection: patches [batch*numPatches, patchDim] * weight^T [patchDim, hiddenDim] + bias
	peW := e.patchEmbedWeight.Value.Data()
	peB := e.patchEmbedBias.Value.Data()
	embeddings := make([]T, batch*numPatches*hiddenDim)
	for b := 0; b < batch; b++ {
		for p := 0; p < numPatches; p++ {
			for h := 0; h < hiddenDim; h++ {
				var sum T
				for d := 0; d < patchDim; d++ {
					pIdx := b*numPatches*patchDim + p*patchDim + d
					wIdx := h*patchDim + d // weight is [hiddenDim, patchDim]
					sum = e.ops.Add(sum, e.ops.Mul(patches[pIdx], peW[wIdx]))
				}
				eIdx := b*numPatches*hiddenDim + p*hiddenDim + h
				embeddings[eIdx] = e.ops.Add(sum, peB[h])
			}
		}
	}

	// Prepend class token: [batch, numPatches+1, hiddenDim]
	seqLen := numPatches + 1
	clsData := e.classEmbedding.Value.Data()
	withCls := make([]T, batch*seqLen*hiddenDim)
	for b := 0; b < batch; b++ {
		// Copy class token for this batch element.
		for h := 0; h < hiddenDim; h++ {
			withCls[b*seqLen*hiddenDim+h] = clsData[h]
		}
		// Copy patch embeddings.
		copy(
			withCls[b*seqLen*hiddenDim+hiddenDim:b*seqLen*hiddenDim+seqLen*hiddenDim],
			embeddings[b*numPatches*hiddenDim:(b+1)*numPatches*hiddenDim],
		)
	}

	// Add position embedding.
	posData := e.positionEmbedding.Value.Data()
	for b := 0; b < batch; b++ {
		for i := 0; i < seqLen*hiddenDim; i++ {
			withCls[b*seqLen*hiddenDim+i] = e.ops.Add(withCls[b*seqLen*hiddenDim+i], posData[i])
		}
	}

	x, err := tensor.New[T]([]int{batch, seqLen, hiddenDim}, withCls)
	if err != nil {
		return nil, fmt.Errorf("create embeddings tensor: %w", err)
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
	normedData := normed.Data()
	normed2D, err := tensor.New[T]([]int{batch * seqLen, hiddenDim}, normedData)
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

	// Multi-head self-attention.
	qData := q.Data()
	kData := k.Data()
	vData := v.Data()
	scale := T(1.0 / math.Sqrt(float64(headDim)))

	attnOut := make([]T, batch*seqLen*hiddenDim)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			scores := make([]T, seqLen*seqLen)
			for qi := 0; qi < seqLen; qi++ {
				for ki := 0; ki < seqLen; ki++ {
					var dot T
					for d := 0; d < headDim; d++ {
						qIdx := b*seqLen*hiddenDim + qi*hiddenDim + h*headDim + d
						kIdx := b*seqLen*hiddenDim + ki*hiddenDim + h*headDim + d
						dot = e.ops.Add(dot, e.ops.Mul(qData[qIdx], kData[kIdx]))
					}
					scores[qi*seqLen+ki] = e.ops.Mul(dot, scale)
				}
			}

			// Softmax per query position.
			for qi := 0; qi < seqLen; qi++ {
				maxVal := scores[qi*seqLen]
				for ki := 1; ki < seqLen; ki++ {
					if e.ops.GreaterThan(scores[qi*seqLen+ki], maxVal) {
						maxVal = scores[qi*seqLen+ki]
					}
				}
				var sumExp T
				for ki := 0; ki < seqLen; ki++ {
					diff := e.ops.Sub(scores[qi*seqLen+ki], maxVal)
					expVal := e.ops.Exp(diff)
					scores[qi*seqLen+ki] = expVal
					sumExp = e.ops.Add(sumExp, expVal)
				}
				for ki := 0; ki < seqLen; ki++ {
					scores[qi*seqLen+ki] = e.ops.Div(scores[qi*seqLen+ki], sumExp)
				}

				// Weighted sum of V.
				for d := 0; d < headDim; d++ {
					var val T
					for ki := 0; ki < seqLen; ki++ {
						vIdx := b*seqLen*hiddenDim + ki*hiddenDim + h*headDim + d
						val = e.ops.Add(val, e.ops.Mul(scores[qi*seqLen+ki], vData[vIdx]))
					}
					outIdx := b*seqLen*hiddenDim + qi*hiddenDim + h*headDim + d
					attnOut[outIdx] = val
				}
			}
		}
	}

	// Output projection.
	attnTensor, err := tensor.New[T]([]int{batch * seqLen, hiddenDim}, attnOut)
	if err != nil {
		return nil, err
	}
	projected, err := block.oProj.Forward(ctx, attnTensor)
	if err != nil {
		return nil, fmt.Errorf("o_proj: %w", err)
	}

	// Residual connection.
	projected3D, err := tensor.New[T]([]int{batch, seqLen, hiddenDim}, projected.Data())
	if err != nil {
		return nil, err
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

	normed2D, err = tensor.New[T]([]int{batch * seqLen, hiddenDim}, normed.Data())
	if err != nil {
		return nil, err
	}
	ffnOut, err := block.ffn1.Forward(ctx, normed2D)
	if err != nil {
		return nil, fmt.Errorf("ffn1: %w", err)
	}
	applyQuickGELU(ffnOut, e.ops)

	ffnOut, err = block.ffn2.Forward(ctx, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("ffn2: %w", err)
	}

	ffn3D, err := tensor.New[T]([]int{batch, seqLen, hiddenDim}, ffnOut.Data())
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

// applyQuickGELU applies the QuickGELU activation function in-place.
// QuickGELU(x) = x * sigmoid(1.702 * x)
func applyQuickGELU[T tensor.Numeric](t *tensor.TensorNumeric[T], ops numeric.Arithmetic[T]) {
	data := t.Data()
	coeff := ops.FromFloat64(1.702)
	one := ops.One()
	for i, v := range data {
		// sigmoid(1.702 * x) = 1 / (1 + exp(-1.702 * x))
		negScaled := ops.Mul(ops.FromFloat64(-1.0), ops.Mul(coeff, v))
		sigmoid := ops.Div(one, ops.Add(one, ops.Exp(negScaled)))
		data[i] = ops.Mul(v, sigmoid)
	}
	t.SetData(data)
}
