// Package audio provides audio-related neural network layers.
package audio

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// WhisperEncoderConfig holds configuration for a WhisperEncoder.
type WhisperEncoderConfig struct {
	NumMels          int  // Number of mel channels (input channels for conv frontend).
	HiddenDim        int  // Hidden dimension throughout the encoder.
	NumHeads         int  // Number of attention heads per transformer block.
	NumLayers        int  // Number of transformer encoder blocks.
	KernelSize       int  // Kernel size for the conv1d frontend layers.
	IntermediateSize int  // FFN intermediate size (0 = 4*HiddenDim for backward compatibility).
	AttentionBias    bool // If true, Q/K/V projections include bias terms.
}

// transformerBlock holds the layers for a single transformer encoder block.
type transformerBlock[T tensor.Numeric] struct {
	ln1   *normalization.LayerNormalization[T]
	qProj *core.Linear[T]
	kProj *core.Linear[T]
	vProj *core.Linear[T]
	oProj *core.Linear[T]
	ln2   *normalization.LayerNormalization[T]
	ffn1  *core.Dense[T]
	ffn2  *core.Dense[T]
	qBias *graph.Parameter[T] // optional Q bias (nil when AttentionBias is false)
	kBias *graph.Parameter[T] // optional K bias
	vBias *graph.Parameter[T] // optional V bias
}

// WhisperEncoder implements a Whisper-style audio encoder with a 2-layer Conv1D
// frontend (stride 2 for temporal downsampling) followed by N transformer
// encoder blocks (self-attention + FFN + layer norm).
//
// Input shape:  [batch, num_mels, T_frames]
// Output shape: [T_downsampled, hidden_dim]
type WhisperEncoder[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	cfg    WhisperEncoderConfig

	conv1  *core.Conv1D[T]
	conv2  *core.Conv1D[T]
	posEnc *graph.Parameter[T] // Learned positional encoding

	blocks []transformerBlock[T]

	lnPost *normalization.LayerNormalization[T]
}

// NewWhisperEncoder creates a new WhisperEncoder.
func NewWhisperEncoder[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	cfg WhisperEncoderConfig,
) (*WhisperEncoder[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if cfg.NumMels <= 0 {
		return nil, fmt.Errorf("NumMels must be positive, got %d", cfg.NumMels)
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
	if cfg.KernelSize <= 0 {
		return nil, fmt.Errorf("KernelSize must be positive, got %d", cfg.KernelSize)
	}
	if cfg.HiddenDim%cfg.NumHeads != 0 {
		return nil, fmt.Errorf("HiddenDim (%d) must be divisible by NumHeads (%d)", cfg.HiddenDim, cfg.NumHeads)
	}

	padding := (cfg.KernelSize - 1) / 2

	// Conv1D frontend: 2 layers with stride=2 for 4x temporal downsampling.
	conv1, err := core.NewConv1D[T](
		name+"_conv1", engine, ops,
		cfg.NumMels, cfg.HiddenDim, cfg.KernelSize,
		core.Conv1DStride(2), core.Conv1DPadding(padding),
	)
	if err != nil {
		return nil, fmt.Errorf("conv1: %w", err)
	}

	conv2, err := core.NewConv1D[T](
		name+"_conv2", engine, ops,
		cfg.HiddenDim, cfg.HiddenDim, cfg.KernelSize,
		core.Conv1DStride(2), core.Conv1DPadding(padding),
	)
	if err != nil {
		return nil, fmt.Errorf("conv2: %w", err)
	}

	// Build transformer blocks.
	blocks := make([]transformerBlock[T], cfg.NumLayers)
	ffnDim := cfg.HiddenDim * 4
	if cfg.IntermediateSize > 0 {
		ffnDim = cfg.IntermediateSize
	}

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

		blk := transformerBlock[T]{
			ln1:   ln1,
			qProj: qProj,
			kProj: kProj,
			vProj: vProj,
			oProj: oProj,
			ln2:   ln2,
			ffn1:  ffn1,
			ffn2:  ffn2,
		}

		if cfg.AttentionBias {
			biasTensor := func(pname string) (*graph.Parameter[T], error) {
				zeros := make([]T, cfg.HiddenDim)
				t, err := tensor.New[T]([]int{cfg.HiddenDim}, zeros)
				if err != nil {
					return nil, err
				}
				return graph.NewParameter[T](pname, t, tensor.New[T])
			}
			blk.qBias, err = biasTensor(prefix + "_q_bias")
			if err != nil {
				return nil, fmt.Errorf("block %d q_bias: %w", i, err)
			}
			blk.kBias, err = biasTensor(prefix + "_k_bias")
			if err != nil {
				return nil, fmt.Errorf("block %d k_bias: %w", i, err)
			}
			blk.vBias, err = biasTensor(prefix + "_v_bias")
			if err != nil {
				return nil, fmt.Errorf("block %d v_bias: %w", i, err)
			}
		}

		blocks[i] = blk
	}

	lnPost, err := normalization.NewLayerNormalization[T](engine, cfg.HiddenDim)
	if err != nil {
		return nil, fmt.Errorf("ln_post: %w", err)
	}

	return &WhisperEncoder[T]{
		engine: engine,
		ops:    ops,
		cfg:    cfg,
		conv1:  conv1,
		conv2:  conv2,
		blocks: blocks,
		lnPost: lnPost,
	}, nil
}

func (e *WhisperEncoder[T]) OpType() string { return "WhisperEncoder" }

func (e *WhisperEncoder[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_mels":   e.cfg.NumMels,
		"hidden_dim": e.cfg.HiddenDim,
		"num_heads":  e.cfg.NumHeads,
		"num_layers": e.cfg.NumLayers,
	}
}

func (e *WhisperEncoder[T]) OutputShape() []int {
	return []int{-1, e.cfg.HiddenDim}
}

// Forward runs the Whisper encoder.
// Input: [batch, num_mels, T_frames]
// Output: [T_downsampled, hidden_dim]
func (e *WhisperEncoder[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("WhisperEncoder requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("WhisperEncoder input must be 3D [batch, num_mels, T_frames], got %v", shape)
	}
	if shape[1] != e.cfg.NumMels {
		return nil, fmt.Errorf("WhisperEncoder: input channels %d != expected num_mels %d", shape[1], e.cfg.NumMels)
	}

	// Conv1 + GELU
	x, err := e.conv1.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("conv1: %w", err)
	}
	applyGELU(x, e.ops)

	// Conv2 + GELU
	x, err = e.conv2.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("conv2: %w", err)
	}
	applyGELU(x, e.ops)

	// x is now [batch, hidden_dim, T_downsampled]
	// Transpose to [batch, T_downsampled, hidden_dim] for transformer blocks
	convShape := x.Shape()
	batch := convShape[0]
	hiddenDim := convShape[1]
	seqLen := convShape[2]

	x, err = transposeLastTwo(x, batch, hiddenDim, seqLen)
	if err != nil {
		return nil, fmt.Errorf("transpose: %w", err)
	}

	// Add sinusoidal positional encoding
	addSinusoidalPosEnc(x, seqLen, hiddenDim, e.ops)

	// Transformer encoder blocks
	for i := range e.blocks {
		x, err = e.forwardBlock(ctx, &e.blocks[i], x, batch, seqLen)
		if err != nil {
			return nil, fmt.Errorf("block %d: %w", i, err)
		}
	}

	// Post layer norm
	x, err = e.lnPost.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ln_post: %w", err)
	}

	// Squeeze batch dimension: output [T_downsampled, hidden_dim]
	// Take first batch element
	outData := x.Data()[:seqLen*hiddenDim]
	return tensor.New[T]([]int{seqLen, hiddenDim}, outData)
}

// forwardBlock runs a single transformer encoder block.
// x shape: [batch, seqLen, hiddenDim]
func (e *WhisperEncoder[T]) forwardBlock(
	ctx context.Context,
	block *transformerBlock[T],
	x *tensor.TensorNumeric[T],
	batch, seqLen int,
) (*tensor.TensorNumeric[T], error) {
	hiddenDim := e.cfg.HiddenDim
	numHeads := e.cfg.NumHeads
	headDim := hiddenDim / numHeads

	// Pre-norm self-attention
	residual := x

	normed, err := block.ln1.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ln1: %w", err)
	}

	// Project Q, K, V: each [batch, seqLen, hiddenDim]
	// Reshape normed to 2D [batch*seqLen, hiddenDim] for linear projection
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

	// Apply attention biases if present.
	if block.qBias != nil {
		addBiasInPlace(q, block.qBias.Value, e.ops)
	}
	if block.kBias != nil {
		addBiasInPlace(k, block.kBias.Value, e.ops)
	}
	if block.vBias != nil {
		addBiasInPlace(v, block.vBias.Value, e.ops)
	}

	// Compute multi-head self-attention manually
	// Q, K, V are [batch*seqLen, hiddenDim]
	qData := q.Data()
	kData := k.Data()
	vData := v.Data()
	scale := T(1.0 / math.Sqrt(float64(headDim)))

	attnOut := make([]T, batch*seqLen*hiddenDim)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			// Compute attention scores for this head
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

			// Softmax per query
			for qi := 0; qi < seqLen; qi++ {
				// Find max for numerical stability
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

				// Weighted sum of V
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

	// Output projection
	attnTensor, err := tensor.New[T]([]int{batch * seqLen, hiddenDim}, attnOut)
	if err != nil {
		return nil, err
	}
	projected, err := block.oProj.Forward(ctx, attnTensor)
	if err != nil {
		return nil, fmt.Errorf("o_proj: %w", err)
	}

	// Residual connection
	projected3D, err := tensor.New[T]([]int{batch, seqLen, hiddenDim}, projected.Data())
	if err != nil {
		return nil, err
	}
	x, err = e.engine.Add(ctx, residual, projected3D)
	if err != nil {
		return nil, fmt.Errorf("residual1: %w", err)
	}

	// Pre-norm FFN
	residual = x
	normed, err = block.ln2.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("ln2: %w", err)
	}

	// FFN: Dense(hiddenDim -> 4*hiddenDim) + GELU + Dense(4*hiddenDim -> hiddenDim)
	normed2D, err = tensor.New[T]([]int{batch * seqLen, hiddenDim}, normed.Data())
	if err != nil {
		return nil, err
	}
	ffnOut, err := block.ffn1.Forward(ctx, normed2D)
	if err != nil {
		return nil, fmt.Errorf("ffn1: %w", err)
	}
	applyGELU(ffnOut, e.ops)

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

// Parameters returns all trainable parameters from the encoder.
// The order is: conv1 params, conv2 params, [posEnc],
// then per block: ln1, qProj, kProj, vProj, oProj, [qBias, kBias, vBias], ln2, ffn1, ffn2,
// then lnPost params.
func (e *WhisperEncoder[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, e.conv1.Parameters()...)
	params = append(params, e.conv2.Parameters()...)
	if e.posEnc != nil {
		params = append(params, e.posEnc)
	}
	for i := range e.blocks {
		b := &e.blocks[i]
		params = append(params, b.ln1.Parameters()...)
		params = append(params, b.qProj.Parameters()...)
		params = append(params, b.kProj.Parameters()...)
		params = append(params, b.vProj.Parameters()...)
		params = append(params, b.oProj.Parameters()...)
		if b.qBias != nil {
			params = append(params, b.qBias)
		}
		if b.kBias != nil {
			params = append(params, b.kBias)
		}
		if b.vBias != nil {
			params = append(params, b.vBias)
		}
		params = append(params, b.ln2.Parameters()...)
		params = append(params, b.ffn1.Parameters()...)
		params = append(params, b.ffn2.Parameters()...)
	}
	params = append(params, e.lnPost.Parameters()...)
	return params
}

// HasAttentionBias returns true if the encoder was configured with attention biases.
func (e *WhisperEncoder[T]) HasAttentionBias() bool {
	return e.cfg.AttentionBias
}

func (e *WhisperEncoder[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// addBiasInPlace adds a 1D bias vector to a 2D tensor [rows, dim] in-place.
func addBiasInPlace[T tensor.Numeric](t *tensor.TensorNumeric[T], bias *tensor.TensorNumeric[T], ops numeric.Arithmetic[T]) {
	data := t.Data()
	biasData := bias.Data()
	dim := len(biasData)
	for i, v := range data {
		data[i] = ops.Add(v, biasData[i%dim])
	}
	t.SetData(data)
}

// applyGELU applies the GELU activation function in-place.
// Uses the approximation: GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
//
// TODO(T124.2.3): replace with a call to layers/activations.NewGelu.
// Blocked because (a) canonical Gelu requires tensor.Float while this
// type is tensor.Numeric, and (b) callers depend on the in-place
// SetData semantics, while the canonical Node returns a new tensor.
// Switching requires both a constraint relaxation and a storage-
// semantics audit at every call site.
func applyGELU[T tensor.Numeric](t *tensor.TensorNumeric[T], ops numeric.Arithmetic[T]) {
	data := t.Data()
	half := ops.FromFloat64(0.5)
	one := ops.One()
	coeff := ops.FromFloat64(0.044715)
	sqrtTwoOverPi := ops.FromFloat64(math.Sqrt(2.0 / math.Pi))
	for i, v := range data {
		// x^3
		x3 := ops.Mul(v, ops.Mul(v, v))
		// sqrt(2/pi) * (x + 0.044715 * x^3)
		inner := ops.Mul(sqrtTwoOverPi, ops.Add(v, ops.Mul(coeff, x3)))
		// 0.5 * x * (1 + tanh(inner))
		data[i] = ops.Mul(half, ops.Mul(v, ops.Add(one, ops.Tanh(inner))))
	}
	t.SetData(data)
}

// transposeLastTwo transposes a 3D tensor from [batch, dim1, dim2] to [batch, dim2, dim1].
func transposeLastTwo[T tensor.Numeric](t *tensor.TensorNumeric[T], batch, dim1, dim2 int) (*tensor.TensorNumeric[T], error) {
	data := t.Data()
	out := make([]T, len(data))
	for b := 0; b < batch; b++ {
		for i := 0; i < dim1; i++ {
			for j := 0; j < dim2; j++ {
				srcIdx := b*dim1*dim2 + i*dim2 + j
				dstIdx := b*dim2*dim1 + j*dim1 + i
				out[dstIdx] = data[srcIdx]
			}
		}
	}
	return tensor.New[T]([]int{batch, dim2, dim1}, out)
}

// addSinusoidalPosEnc adds sinusoidal positional encoding in-place.
func addSinusoidalPosEnc[T tensor.Numeric](t *tensor.TensorNumeric[T], seqLen, hiddenDim int, ops numeric.Arithmetic[T]) {
	data := t.Data()
	shape := t.Shape()
	batch := shape[0]

	for pos := 0; pos < seqLen; pos++ {
		for d := 0; d < hiddenDim; d++ {
			angle := float64(pos) / math.Pow(10000, float64(2*(d/2))/float64(hiddenDim))
			var enc float64
			if d%2 == 0 {
				enc = math.Sin(angle)
			} else {
				enc = math.Cos(angle)
			}
			posVal := ops.FromFloat64(enc)
			for b := 0; b < batch; b++ {
				idx := b*seqLen*hiddenDim + pos*hiddenDim + d
				data[idx] = ops.Add(data[idx], posVal)
			}
		}
	}
	t.SetData(data)
}
