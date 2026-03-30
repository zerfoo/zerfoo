package timeseries

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// PatchTSTConfig holds configuration for a PatchTST model.
type PatchTSTConfig struct {
	InputLength        int  // length of input time series
	PatchLength        int  // length of each patch
	Stride             int  // stride between patches
	DModel             int  // transformer hidden dimension
	NHeads             int  // number of attention heads
	NLayers            int  // number of transformer encoder layers
	OutputDim          int  // output prediction dimension
	ChannelIndependent bool // process each channel independently through same transformer
}

// NumPatches returns the number of patches produced by the patching configuration.
func (c PatchTSTConfig) NumPatches() int {
	return (c.InputLength-c.PatchLength)/c.Stride + 1
}

// encoderLayer holds the weights for one transformer encoder layer.
type encoderLayer struct {
	qProj linearLayer
	kProj linearLayer
	vProj linearLayer
	oProj linearLayer
	ffn1  linearLayer
	ffn2  linearLayer
	norm1 *tensor.TensorNumeric[float32] // layer norm weights (scale)
	bias1 *tensor.TensorNumeric[float32] // layer norm bias
	norm2 *tensor.TensorNumeric[float32] // layer norm weights (scale)
	bias2 *tensor.TensorNumeric[float32] // layer norm bias
}

// PatchTST implements the Patch Time-Series Transformer.
type PatchTST struct {
	config    PatchTSTConfig
	engine    compute.Engine[float32]
	ops       numeric.Arithmetic[float32]
	patchEmb  linearLayer                    // patch_length -> d_model
	posEmb    *tensor.TensorNumeric[float32] // [1, num_patches, d_model]
	layers    []encoderLayer
	head      linearLayer // num_patches * d_model -> output_dim
	normMeans [][]float64 // per-channel normalization means from training
	normStds  [][]float64 // per-channel normalization stds from training

	// Training state for TrainableBackend (CPU path only).
	trainParams *patchTSTParamsF64 // extracted f64 params, set during TrainWindowed
	grads       []float64          // gradient accumulator for TrainableBackend
}

// NewPatchTST creates a new PatchTST model with the given configuration.
func NewPatchTST(config PatchTSTConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*PatchTST, error) {
	if config.InputLength <= 0 {
		return nil, fmt.Errorf("patchtst: InputLength must be positive, got %d", config.InputLength)
	}
	if config.PatchLength <= 0 {
		return nil, fmt.Errorf("patchtst: PatchLength must be positive, got %d", config.PatchLength)
	}
	if config.Stride <= 0 {
		return nil, fmt.Errorf("patchtst: Stride must be positive, got %d", config.Stride)
	}
	if config.PatchLength > config.InputLength {
		return nil, fmt.Errorf("patchtst: PatchLength (%d) must be <= InputLength (%d)", config.PatchLength, config.InputLength)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("patchtst: DModel must be positive, got %d", config.DModel)
	}
	if config.NHeads <= 0 {
		return nil, fmt.Errorf("patchtst: NHeads must be positive, got %d", config.NHeads)
	}
	if config.DModel%config.NHeads != 0 {
		return nil, fmt.Errorf("patchtst: DModel (%d) must be divisible by NHeads (%d)", config.DModel, config.NHeads)
	}
	if config.NLayers <= 0 {
		return nil, fmt.Errorf("patchtst: NLayers must be positive, got %d", config.NLayers)
	}
	if config.OutputDim <= 0 {
		return nil, fmt.Errorf("patchtst: OutputDim must be positive, got %d", config.OutputDim)
	}

	numPatches := config.NumPatches()
	if numPatches <= 0 {
		return nil, fmt.Errorf("patchtst: configuration yields %d patches (need at least 1)", numPatches)
	}

	m := &PatchTST{
		config: config,
		engine: engine,
		ops:    ops,
	}

	var err error

	// Patch embedding: patch_length -> d_model.
	m.patchEmb, err = newLinearXavier(config.PatchLength, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("patchtst: patch embedding: %w", err)
	}

	// Learnable positional embeddings: [1, num_patches, d_model].
	posData := make([]float32, numPatches*config.DModel)
	scale := float32(math.Sqrt(2.0 / float64(config.DModel)))
	for i := range posData {
		posData[i] = float32(rand.NormFloat64()) * scale * 0.02
	}
	m.posEmb, err = tensor.New[float32]([]int{1, numPatches, config.DModel}, posData)
	if err != nil {
		return nil, fmt.Errorf("patchtst: positional embedding: %w", err)
	}

	// Transformer encoder layers.
	m.layers = make([]encoderLayer, config.NLayers)
	for i := range config.NLayers {
		m.layers[i], err = newEncoderLayer(config.DModel, config.NHeads)
		if err != nil {
			return nil, fmt.Errorf("patchtst: encoder layer %d: %w", i, err)
		}
	}

	// Output head: flatten num_patches * d_model -> output_dim.
	m.head, err = newLinearXavier(numPatches*config.DModel, config.OutputDim)
	if err != nil {
		return nil, fmt.Errorf("patchtst: output head: %w", err)
	}

	return m, nil
}

// Forward runs the PatchTST forward pass on input time series data.
// input shape: [batch, channels, input_length] for multivariate, or
// [batch, input_length] for univariate (treated as 1 channel).
// Returns predictions of shape [batch, channels, output_dim] if channel_independent,
// or [batch, output_dim] if not channel_independent.
func (m *PatchTST) Forward(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	var batch, channels, length int

	switch len(shape) {
	case 2:
		batch, length = shape[0], shape[1]
		channels = 1
	case 3:
		batch, channels, length = shape[0], shape[1], shape[2]
	default:
		return nil, fmt.Errorf("patchtst: input must be 2D [batch, length] or 3D [batch, channels, length], got shape %v", shape)
	}

	if length != m.config.InputLength {
		return nil, fmt.Errorf("patchtst: expected input length %d, got %d", m.config.InputLength, length)
	}

	// Reshape to [batch*channels, input_length] for uniform processing.
	flat, err := m.engine.Reshape(ctx, input, []int{batch * channels, length})
	if err != nil {
		return nil, fmt.Errorf("patchtst: reshape input: %w", err)
	}

	// Patching: extract overlapping patches.
	patches, err := m.extractPatches(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("patchtst: extract patches: %w", err)
	}
	// patches shape: [batch*channels, num_patches, patch_length]

	numPatches := m.config.NumPatches()

	// Patch embedding: project each patch to d_model.
	// Reshape to [batch*channels*num_patches, patch_length] for linear.
	pFlat, err := m.engine.Reshape(ctx, patches, []int{batch * channels * numPatches, m.config.PatchLength})
	if err != nil {
		return nil, fmt.Errorf("patchtst: reshape patches: %w", err)
	}

	embedded, err := m.linear(ctx, pFlat, m.patchEmb)
	if err != nil {
		return nil, fmt.Errorf("patchtst: patch embed: %w", err)
	}
	// embedded shape: [batch*channels*num_patches, d_model]

	// Reshape to [batch*channels, num_patches, d_model].
	x, err := m.engine.Reshape(ctx, embedded, []int{batch * channels, numPatches, m.config.DModel})
	if err != nil {
		return nil, fmt.Errorf("patchtst: reshape embedded: %w", err)
	}

	// Add positional embeddings (broadcast over batch*channels).
	x, err = m.engine.Add(ctx, x, m.posEmb)
	if err != nil {
		return nil, fmt.Errorf("patchtst: add position: %w", err)
	}

	// Transformer encoder layers.
	for i, layer := range m.layers {
		x, err = m.encoderForward(ctx, x, layer)
		if err != nil {
			return nil, fmt.Errorf("patchtst: encoder layer %d: %w", i, err)
		}
	}
	// x shape: [batch*channels, num_patches, d_model]

	// Flatten: [batch*channels, num_patches * d_model].
	x, err = m.engine.Reshape(ctx, x, []int{batch * channels, numPatches * m.config.DModel})
	if err != nil {
		return nil, fmt.Errorf("patchtst: flatten: %w", err)
	}

	// Output head.
	x, err = m.linear(ctx, x, m.head)
	if err != nil {
		return nil, fmt.Errorf("patchtst: head: %w", err)
	}
	// x shape: [batch*channels, output_dim]

	if m.config.ChannelIndependent && channels > 1 {
		// Reshape to [batch, channels, output_dim].
		x, err = m.engine.Reshape(ctx, x, []int{batch, channels, m.config.OutputDim})
		if err != nil {
			return nil, fmt.Errorf("patchtst: reshape output: %w", err)
		}
	} else if channels == 1 {
		// Keep [batch, output_dim].
	} else {
		// Non-channel-independent with multiple channels: reshape to [batch, channels * output_dim]
		// then project — but for simplicity, just reshape to [batch, channels, output_dim].
		x, err = m.engine.Reshape(ctx, x, []int{batch, channels, m.config.OutputDim})
		if err != nil {
			return nil, fmt.Errorf("patchtst: reshape output: %w", err)
		}
	}

	return x, nil
}

// extractPatches splits the input time series into overlapping patches.
// input shape: [N, input_length], returns [N, num_patches, patch_length].
func (m *PatchTST) extractPatches(ctx context.Context, input *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	n := shape[0]
	numPatches := m.config.NumPatches()
	data := input.Data()

	outData := make([]float32, n*numPatches*m.config.PatchLength)
	for i := range n {
		rowOffset := i * m.config.InputLength
		for p := range numPatches {
			patchStart := p * m.config.Stride
			dstOffset := (i*numPatches + p) * m.config.PatchLength
			copy(outData[dstOffset:dstOffset+m.config.PatchLength], data[rowOffset+patchStart:rowOffset+patchStart+m.config.PatchLength])
		}
	}

	return tensor.New[float32]([]int{n, numPatches, m.config.PatchLength}, outData)
}

// encoderForward runs one transformer encoder layer.
// x shape: [N, seq, d_model], returns same shape.
func (m *PatchTST) encoderForward(ctx context.Context, x *tensor.TensorNumeric[float32], layer encoderLayer) (*tensor.TensorNumeric[float32], error) {
	shape := x.Shape()
	n, seq, dModel := shape[0], shape[1], shape[2]

	// Pre-norm: layer norm before attention.
	normed, err := m.layerNorm(ctx, x, layer.norm1, layer.bias1)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}

	// Multi-head self-attention.
	attnOut, err := m.multiHeadAttention(ctx, normed, layer, n, seq, dModel)
	if err != nil {
		return nil, fmt.Errorf("attention: %w", err)
	}

	// Residual connection.
	x, err = m.engine.Add(ctx, x, attnOut)
	if err != nil {
		return nil, fmt.Errorf("residual1: %w", err)
	}

	// Pre-norm: layer norm before FFN.
	normed, err = m.layerNorm(ctx, x, layer.norm2, layer.bias2)
	if err != nil {
		return nil, fmt.Errorf("norm2: %w", err)
	}

	// FFN: two linear layers with GELU activation.
	ffnOut, err := m.ffnForward(ctx, normed, layer, n, seq, dModel)
	if err != nil {
		return nil, fmt.Errorf("ffn: %w", err)
	}

	// Residual connection.
	x, err = m.engine.Add(ctx, x, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("residual2: %w", err)
	}

	return x, nil
}

// multiHeadAttention computes multi-head self-attention.
// input shape: [N, seq, d_model], output shape: [N, seq, d_model].
func (m *PatchTST) multiHeadAttention(ctx context.Context, x *tensor.TensorNumeric[float32], layer encoderLayer, n, seq, dModel int) (*tensor.TensorNumeric[float32], error) {
	headDim := dModel / m.config.NHeads

	// Reshape to [N*seq, d_model] for linear projections.
	xFlat, err := m.engine.Reshape(ctx, x, []int{n * seq, dModel})
	if err != nil {
		return nil, err
	}

	q, err := m.linear(ctx, xFlat, layer.qProj)
	if err != nil {
		return nil, fmt.Errorf("q proj: %w", err)
	}
	k, err := m.linear(ctx, xFlat, layer.kProj)
	if err != nil {
		return nil, fmt.Errorf("k proj: %w", err)
	}
	v, err := m.linear(ctx, xFlat, layer.vProj)
	if err != nil {
		return nil, fmt.Errorf("v proj: %w", err)
	}

	// Reshape to [N, seq, n_heads, head_dim] then transpose to [N*n_heads, seq, head_dim].
	// First reshape to [N, seq, n_heads, head_dim].
	q, err = m.engine.Reshape(ctx, q, []int{n, seq, m.config.NHeads, headDim})
	if err != nil {
		return nil, err
	}
	k, err = m.engine.Reshape(ctx, k, []int{n, seq, m.config.NHeads, headDim})
	if err != nil {
		return nil, err
	}
	v, err = m.engine.Reshape(ctx, v, []int{n, seq, m.config.NHeads, headDim})
	if err != nil {
		return nil, err
	}

	// Transpose to [N, n_heads, seq, head_dim].
	q, err = m.engine.Transpose(ctx, q, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	k, err = m.engine.Transpose(ctx, k, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	v, err = m.engine.Transpose(ctx, v, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	// Reshape to [N*n_heads, seq, head_dim] for batched matmul.
	q, err = m.engine.Reshape(ctx, q, []int{n * m.config.NHeads, seq, headDim})
	if err != nil {
		return nil, err
	}
	k, err = m.engine.Reshape(ctx, k, []int{n * m.config.NHeads, seq, headDim})
	if err != nil {
		return nil, err
	}
	v, err = m.engine.Reshape(ctx, v, []int{n * m.config.NHeads, seq, headDim})
	if err != nil {
		return nil, err
	}

	// Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
	// Q @ K^T: [N*n_heads, seq, head_dim] @ [N*n_heads, head_dim, seq] = [N*n_heads, seq, seq].
	kT, err := m.engine.Transpose(ctx, k, []int{0, 2, 1})
	if err != nil {
		return nil, err
	}

	scores, err := m.batchedMatMul(ctx, q, kT, n*m.config.NHeads, seq, headDim, seq)
	if err != nil {
		return nil, fmt.Errorf("qk matmul: %w", err)
	}

	// Scale by 1/sqrt(head_dim).
	scaleFactor := float32(1.0 / math.Sqrt(float64(headDim)))
	scores, err = m.engine.MulScalar(ctx, scores, scaleFactor)
	if err != nil {
		return nil, err
	}

	// Softmax along last axis.
	attnWeights, err := m.engine.Softmax(ctx, scores, -1)
	if err != nil {
		return nil, err
	}

	// Attention output: weights @ V.
	// [N*n_heads, seq, seq] @ [N*n_heads, seq, head_dim] = [N*n_heads, seq, head_dim].
	attnOut, err := m.batchedMatMul(ctx, attnWeights, v, n*m.config.NHeads, seq, seq, headDim)
	if err != nil {
		return nil, fmt.Errorf("attn v matmul: %w", err)
	}

	// Reshape back to [N, n_heads, seq, head_dim].
	attnOut, err = m.engine.Reshape(ctx, attnOut, []int{n, m.config.NHeads, seq, headDim})
	if err != nil {
		return nil, err
	}

	// Transpose to [N, seq, n_heads, head_dim].
	attnOut, err = m.engine.Transpose(ctx, attnOut, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	// Reshape to [N*seq, d_model] for output projection.
	attnOut, err = m.engine.Reshape(ctx, attnOut, []int{n * seq, dModel})
	if err != nil {
		return nil, err
	}

	// Output projection.
	out, err := m.linear(ctx, attnOut, layer.oProj)
	if err != nil {
		return nil, fmt.Errorf("o proj: %w", err)
	}

	// Reshape to [N, seq, d_model].
	return m.engine.Reshape(ctx, out, []int{n, seq, dModel})
}

// batchedMatMul performs matrix multiplication on batched 3D tensors by iterating
// over the batch dimension and using 2D MatMul.
// a shape: [batch, m, k], b shape: [batch, k, n], result: [batch, m, n].
func (m *PatchTST) batchedMatMul(ctx context.Context, a, b *tensor.TensorNumeric[float32], batch, rows, inner, cols int) (*tensor.TensorNumeric[float32], error) {
	aData := a.Data()
	bData := b.Data()
	outData := make([]float32, batch*rows*cols)

	for i := range batch {
		aSlice := aData[i*rows*inner : (i+1)*rows*inner]
		bSlice := bData[i*inner*cols : (i+1)*inner*cols]

		aMat, err := tensor.New[float32]([]int{rows, inner}, aSlice)
		if err != nil {
			return nil, err
		}
		bMat, err := tensor.New[float32]([]int{inner, cols}, bSlice)
		if err != nil {
			return nil, err
		}

		result, err := m.engine.MatMul(ctx, aMat, bMat)
		if err != nil {
			return nil, err
		}

		copy(outData[i*rows*cols:(i+1)*rows*cols], result.Data())
	}

	return tensor.New[float32]([]int{batch, rows, cols}, outData)
}

// ffnForward runs the feed-forward network sub-layer.
// input shape: [N, seq, d_model], output shape: [N, seq, d_model].
func (m *PatchTST) ffnForward(ctx context.Context, x *tensor.TensorNumeric[float32], layer encoderLayer, n, seq, dModel int) (*tensor.TensorNumeric[float32], error) {
	ffnDim := layer.ffn1.weights.Shape()[1] // d_model * 4

	// Reshape to [N*seq, d_model].
	xFlat, err := m.engine.Reshape(ctx, x, []int{n * seq, dModel})
	if err != nil {
		return nil, err
	}

	// First linear: d_model -> 4*d_model.
	h, err := m.linear(ctx, xFlat, layer.ffn1)
	if err != nil {
		return nil, err
	}

	// GELU activation.
	h, err = m.engine.UnaryOp(ctx, h, geluScalar)
	if err != nil {
		return nil, err
	}

	// Second linear: 4*d_model -> d_model.
	// Reshape h to [N*seq, ffnDim] (already is).
	_ = ffnDim
	out, err := m.linear(ctx, h, layer.ffn2)
	if err != nil {
		return nil, err
	}

	// Reshape to [N, seq, d_model].
	return m.engine.Reshape(ctx, out, []int{n, seq, dModel})
}

// layerNorm applies layer normalization over the last dimension.
// x shape: [..., d_model], scale/bias shape: [d_model].
func (m *PatchTST) layerNorm(ctx context.Context, x *tensor.TensorNumeric[float32], scale, bias *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	shape := x.Shape()
	dModel := shape[len(shape)-1]
	outerSize := 1
	for _, s := range shape[:len(shape)-1] {
		outerSize *= s
	}

	// Reshape to [outer, d_model].
	flat, err := m.engine.Reshape(ctx, x, []int{outerSize, dModel})
	if err != nil {
		return nil, err
	}

	// Compute mean per row.
	mean, err := m.engine.ReduceMean(ctx, flat, 1, true)
	if err != nil {
		return nil, err
	}

	// Subtract mean.
	centered, err := m.engine.Sub(ctx, flat, mean)
	if err != nil {
		return nil, err
	}

	// Compute variance: mean(centered^2).
	sq, err := m.engine.Mul(ctx, centered, centered)
	if err != nil {
		return nil, err
	}
	variance, err := m.engine.ReduceMean(ctx, sq, 1, true)
	if err != nil {
		return nil, err
	}

	// Add epsilon and compute rsqrt.
	variance, err = m.engine.AddScalar(ctx, variance, 1e-5)
	if err != nil {
		return nil, err
	}
	invStd, err := m.engine.Rsqrt(ctx, variance)
	if err != nil {
		return nil, err
	}

	// Normalize.
	normed, err := m.engine.Mul(ctx, centered, invStd)
	if err != nil {
		return nil, err
	}

	// Scale and shift: normed * scale + bias (broadcast over rows).
	// scale/bias are [1, d_model].
	normed, err = m.engine.Mul(ctx, normed, scale)
	if err != nil {
		return nil, err
	}
	normed, err = m.engine.Add(ctx, normed, bias)
	if err != nil {
		return nil, err
	}

	// Reshape back to original shape.
	return m.engine.Reshape(ctx, normed, shape)
}

// linear computes x @ W + b.
func (m *PatchTST) linear(ctx context.Context, x *tensor.TensorNumeric[float32], l linearLayer) (*tensor.TensorNumeric[float32], error) {
	out, err := m.engine.MatMul(ctx, x, l.weights)
	if err != nil {
		return nil, err
	}
	return m.engine.Add(ctx, out, l.biases)
}

// newEncoderLayer creates a single transformer encoder layer.
func newEncoderLayer(dModel, nHeads int) (encoderLayer, error) {
	var l encoderLayer
	var err error

	ffnDim := dModel * 4

	l.qProj, err = newLinearXavier(dModel, dModel)
	if err != nil {
		return l, fmt.Errorf("q proj: %w", err)
	}
	l.kProj, err = newLinearXavier(dModel, dModel)
	if err != nil {
		return l, fmt.Errorf("k proj: %w", err)
	}
	l.vProj, err = newLinearXavier(dModel, dModel)
	if err != nil {
		return l, fmt.Errorf("v proj: %w", err)
	}
	l.oProj, err = newLinearXavier(dModel, dModel)
	if err != nil {
		return l, fmt.Errorf("o proj: %w", err)
	}
	l.ffn1, err = newLinearXavier(dModel, ffnDim)
	if err != nil {
		return l, fmt.Errorf("ffn1: %w", err)
	}
	l.ffn2, err = newLinearXavier(ffnDim, dModel)
	if err != nil {
		return l, fmt.Errorf("ffn2: %w", err)
	}

	// Layer norm parameters (initialized to scale=1, bias=0).
	ones := make([]float32, dModel)
	for i := range ones {
		ones[i] = 1.0
	}
	zeros := make([]float32, dModel)

	l.norm1, err = tensor.New[float32]([]int{1, dModel}, append([]float32(nil), ones...))
	if err != nil {
		return l, fmt.Errorf("norm1 scale: %w", err)
	}
	l.bias1, err = tensor.New[float32]([]int{1, dModel}, append([]float32(nil), zeros...))
	if err != nil {
		return l, fmt.Errorf("norm1 bias: %w", err)
	}
	l.norm2, err = tensor.New[float32]([]int{1, dModel}, append([]float32(nil), ones...))
	if err != nil {
		return l, fmt.Errorf("norm2 scale: %w", err)
	}
	l.bias2, err = tensor.New[float32]([]int{1, dModel}, append([]float32(nil), zeros...))
	if err != nil {
		return l, fmt.Errorf("norm2 bias: %w", err)
	}

	return l, nil
}

// Predict runs inference on multivariate time series data using float64 slices.
// input[channel][time] has one sub-slice per channel, each of length InputLength.
// Returns output[channel][horizon] with OutputDim predictions per channel.
// For single-channel input, pass a single sub-slice.
func (m *PatchTST) Predict(input [][]float64) ([][]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("patchtst: input must have at least one channel")
	}
	channels := len(input)
	for c, ch := range input {
		if len(ch) != m.config.InputLength {
			return nil, fmt.Errorf("patchtst: channel %d length %d, want %d", c, len(ch), m.config.InputLength)
		}
	}

	// Build float32 tensor [1, channels, input_length].
	data := make([]float32, channels*m.config.InputLength)
	for c, ch := range input {
		for i, v := range ch {
			data[c*m.config.InputLength+i] = float32(v)
		}
	}

	var shape []int
	if channels == 1 {
		shape = []int{1, m.config.InputLength}
	} else {
		shape = []int{1, channels, m.config.InputLength}
	}
	t, err := tensor.New[float32](shape, data)
	if err != nil {
		return nil, fmt.Errorf("patchtst: create input tensor: %w", err)
	}

	ctx := context.Background()
	out, err := m.Forward(ctx, t)
	if err != nil {
		return nil, err
	}

	outData := out.Data()
	result := make([][]float64, channels)
	for c := range channels {
		result[c] = make([]float64, m.config.OutputDim)
		for i := range m.config.OutputDim {
			result[c][i] = float64(outData[c*m.config.OutputDim+i])
		}
	}
	return result, nil
}

// geluScalar computes the GELU approximation for a single float32 value.
func geluScalar(x float32) float32 {
	xf := float64(x)
	inner := math.Sqrt(2/math.Pi) * (xf + 0.044715*xf*xf*xf)
	return float32(0.5 * xf * (1 + math.Tanh(inner)))
}

// TrainWindowed trains the PatchTST model on windowed data using AdamW.
// windows: [nSamples][channels][inputLen] input windows.
// labels: flat slice of length nSamples * outputDim.
func (m *PatchTST) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("patchtst: empty training set")
	}

	expectedLabels := nSamples * m.config.OutputDim
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("patchtst: expected %d labels, got %d", expectedLabels, len(labels))
	}

	for i, w := range windows {
		if len(w) == 0 {
			return nil, fmt.Errorf("patchtst: window %d has 0 channels", i)
		}
		for c, ch := range w {
			if len(ch) != m.config.InputLength {
				return nil, fmt.Errorf("patchtst: window %d channel %d has length %d, expected %d",
					i, c, len(ch), m.config.InputLength)
			}
		}
	}

	if config.Epochs <= 0 {
		config.Epochs = 100
	}
	if config.LR <= 0 {
		config.LR = 1e-3
	}
	if config.Beta1 <= 0 {
		config.Beta1 = 0.9
	}
	if config.Beta2 <= 0 {
		config.Beta2 = 0.999
	}
	if config.Epsilon <= 0 {
		config.Epsilon = 1e-8
	}

	// Z-score normalize inputs to prevent gradient explosion on multi-scale data.
	windows, m.normMeans, m.normStds = normalizeWindows(windows)

	if m.engine != nil {
		return m.trainWindowedGPU(windows, labels, config)
	}

	// Extract float64 params for CPU training via TrainLoop.
	m.trainParams = m.extractParamsF64()
	m.grads = nil

	result, err := TrainLoop(m, windows, labels, config)

	// Write trained params back to float32 tensors.
	m.writeBackF32(m.trainParams)
	m.trainParams = nil

	return result, err
}

// ForwardSample runs the PatchTST forward pass on a single sample and returns
// a flat output with cached activations for BackwardSample.
func (m *PatchTST) ForwardSample(input [][]float64) ([]float64, interface{}, error) {
	if m.trainParams == nil {
		return nil, nil, fmt.Errorf("patchtst: ForwardSample requires active training state (call TrainWindowed)")
	}
	output, cache := m.forwardF64WithCache(input, m.trainParams)
	return output, cache, nil
}

// BackwardSample accumulates parameter gradients for a single sample.
func (m *PatchTST) BackwardSample(dOutput []float64, cacheIface interface{}) error {
	cache, ok := cacheIface.(*patchTSTCacheF64)
	if !ok {
		return fmt.Errorf("patchtst: invalid cache type")
	}
	sampleGrads := m.backwardF64(dOutput, m.trainParams, cache)
	grads := m.FlatGrads()
	for i := range grads {
		grads[i] += sampleGrads[i]
	}
	return nil
}

// FlatGrads returns the internal gradient accumulator.
func (m *PatchTST) FlatGrads() []float64 {
	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
	}
	return m.grads
}

// ZeroGrads resets all accumulated gradients to zero.
func (m *PatchTST) ZeroGrads() {
	if m.grads == nil {
		m.grads = make([]float64, m.ParamCount())
		return
	}
	for i := range m.grads {
		m.grads[i] = 0
	}
}

// FlatParams returns pointers to all trainable parameters (exported for TrainableBackend).
func (m *PatchTST) FlatParams() []*float64 {
	if m.trainParams == nil {
		return nil
	}
	return m.trainParams.flatParams()
}

// ParamCount returns the total number of trainable parameters (exported for TrainableBackend).
func (m *PatchTST) ParamCount() int {
	if m.trainParams != nil {
		return m.trainParams.paramCount()
	}
	// Compute from config without extracting params.
	p := m.extractParamsF64()
	return p.paramCount()
}

// Compile-time check that PatchTST implements TrainableBackend.
var _ TrainableBackend = (*PatchTST)(nil)

// PredictWindowed runs inference on windowed data.
// windows: [nSamples][channels][inputLen].
// Returns flat predictions of length nSamples * outputDim.
func (m *PatchTST) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	if modelPath != "" {
		if err := m.loadWeights(modelPath); err != nil {
			return nil, fmt.Errorf("patchtst: load weights: %w", err)
		}
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("patchtst: empty input")
	}

	// Apply normalization from training if available.
	if m.normMeans != nil {
		windows = applyNormalization(windows, m.normMeans, m.normStds)
	}

	params := m.extractParamsF64()
	out := make([]float64, 0, nSamples*m.config.OutputDim)
	for _, w := range windows {
		pred := m.forwardF64(w, params)
		out = append(out, pred...)
	}
	return out, nil
}

// patchTSTWeights is the JSON-serializable form of PatchTST parameters.
type patchTSTWeights struct {
	Config    PatchTSTConfig     `json:"config"`
	PatchEmbW []float64         `json:"patch_emb_w"`
	PatchEmbB []float64         `json:"patch_emb_b"`
	PosEmb    []float64         `json:"pos_emb"`
	Layers    []encoderLayerJSON `json:"layers"`
	HeadW     []float64         `json:"head_w"`
	HeadB     []float64         `json:"head_b"`
	NormMeans [][]float64       `json:"norm_means,omitempty"`
	NormStds  [][]float64       `json:"norm_stds,omitempty"`
}

type encoderLayerJSON struct {
	QW    []float64 `json:"q_w"`
	QB    []float64 `json:"q_b"`
	KW    []float64 `json:"k_w"`
	KB    []float64 `json:"k_b"`
	VW    []float64 `json:"v_w"`
	VB    []float64 `json:"v_b"`
	OW    []float64 `json:"o_w"`
	OB    []float64 `json:"o_b"`
	FFN1W []float64 `json:"ffn1_w"`
	FFN1B []float64 `json:"ffn1_b"`
	FFN2W []float64 `json:"ffn2_w"`
	FFN2B []float64 `json:"ffn2_b"`
	Norm1 []float64 `json:"norm1"`
	Bias1 []float64 `json:"bias1"`
	Norm2 []float64 `json:"norm2"`
	Bias2 []float64 `json:"bias2"`
}

// SaveWeights writes the model weights to a JSON file.
func (m *PatchTST) SaveWeights(path string) error {
	p := m.extractParamsF64()
	w := patchTSTWeights{
		Config:    m.config,
		PatchEmbW: p.patchEmbW,
		PatchEmbB: p.patchEmbB,
		PosEmb:    p.posEmb,
		HeadW:     p.headW,
		HeadB:     p.headB,
		NormMeans: m.normMeans,
		NormStds:  m.normStds,
	}
	for _, l := range p.layers {
		w.Layers = append(w.Layers, encoderLayerJSON{
			QW: l.qW, QB: l.qB, KW: l.kW, KB: l.kB,
			VW: l.vW, VB: l.vB, OW: l.oW, OB: l.oB,
			FFN1W: l.ffn1W, FFN1B: l.ffn1B,
			FFN2W: l.ffn2W, FFN2B: l.ffn2B,
			Norm1: l.norm1, Bias1: l.bias1,
			Norm2: l.norm2, Bias2: l.bias2,
		})
	}
	data, err := json.Marshal(w)
	if err != nil {
		return fmt.Errorf("patchtst: marshal weights: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// loadWeights reads model weights from a JSON file.
func (m *PatchTST) loadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var w patchTSTWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Config != m.config {
		return fmt.Errorf("patchtst: config mismatch: file has %+v, model has %+v", w.Config, m.config)
	}
	if len(w.Layers) != len(m.layers) {
		return fmt.Errorf("patchtst: layer count mismatch: file=%d, model=%d", len(w.Layers), len(m.layers))
	}

	p := &patchTSTParamsF64{
		patchEmbW: w.PatchEmbW,
		patchEmbB: w.PatchEmbB,
		posEmb:    w.PosEmb,
		headW:     w.HeadW,
		headB:     w.HeadB,
	}
	p.layers = make([]encoderLayerF64, len(w.Layers))
	for i, l := range w.Layers {
		p.layers[i] = encoderLayerF64{
			qW: l.QW, qB: l.QB, kW: l.KW, kB: l.KB,
			vW: l.VW, vB: l.VB, oW: l.OW, oB: l.OB,
			ffn1W: l.FFN1W, ffn1B: l.FFN1B,
			ffn2W: l.FFN2W, ffn2B: l.FFN2B,
			norm1: l.Norm1, bias1: l.Bias1,
			norm2: l.Norm2, bias2: l.Bias2,
		}
	}
	m.writeBackF32(p)
	m.normMeans = w.NormMeans
	m.normStds = w.NormStds
	return nil
}
