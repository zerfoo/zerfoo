package timeseries

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
)

// transformerBlock is a simplified single-head self-attention + FFN block
// with residual connections and layer normalization.
type transformerBlock[T tensor.Float] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Self-attention projections.
	qProj *core.Linear[T] // [dModel, dModel]
	kProj *core.Linear[T] // [dModel, dModel]
	vProj *core.Linear[T] // [dModel, dModel]
	norm1 *normalization.LayerNormalization[T]

	// FFN.
	ffn1  *core.Linear[T] // [dModel, 4*dModel]
	ffn2  *core.Linear[T] // [4*dModel, dModel]
	gelu  *activations.Gelu[T]
	norm2 *normalization.LayerNormalization[T]

	dModel int
	scale  T
}

// newTransformerBlock creates a simplified transformer encoder block.
func newTransformerBlock[T tensor.Float](
	prefix string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	dModel int,
) (*transformerBlock[T], error) {
	qProj, err := core.NewLinear[T](prefix+"_q", engine, ops, dModel, dModel)
	if err != nil {
		return nil, fmt.Errorf("create Q projection: %w", err)
	}
	kProj, err := core.NewLinear[T](prefix+"_k", engine, ops, dModel, dModel)
	if err != nil {
		return nil, fmt.Errorf("create K projection: %w", err)
	}
	vProj, err := core.NewLinear[T](prefix+"_v", engine, ops, dModel, dModel)
	if err != nil {
		return nil, fmt.Errorf("create V projection: %w", err)
	}
	norm1, err := normalization.NewLayerNormalization[T](engine, dModel)
	if err != nil {
		return nil, fmt.Errorf("create norm1: %w", err)
	}

	hiddenDim := 4 * dModel
	ffn1, err := core.NewLinear[T](prefix+"_ffn1", engine, ops, dModel, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("create FFN1: %w", err)
	}
	ffn2, err := core.NewLinear[T](prefix+"_ffn2", engine, ops, hiddenDim, dModel)
	if err != nil {
		return nil, fmt.Errorf("create FFN2: %w", err)
	}
	norm2, err := normalization.NewLayerNormalization[T](engine, dModel)
	if err != nil {
		return nil, fmt.Errorf("create norm2: %w", err)
	}

	return &transformerBlock[T]{
		engine: engine,
		ops:    ops,
		qProj:  qProj,
		kProj:  kProj,
		vProj:  vProj,
		norm1:  norm1,
		ffn1:   ffn1,
		ffn2:   ffn2,
		gelu:   activations.NewGelu[T](engine, ops),
		norm2:  norm2,
		dModel: dModel,
		scale:  T(1.0 / math.Sqrt(float64(dModel))),
	}, nil
}

// Forward applies self-attention + FFN with residual connections.
// Input shape: [batch, seqLen, dModel]
// Output shape: [batch, seqLen, dModel]
func (tb *transformerBlock[T]) Forward(ctx context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := x.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// --- Self-attention with residual ---
	residual := x

	// Layer norm before attention (pre-norm).
	normed, err := tb.norm1.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}

	// Reshape [batch, seqLen, dModel] -> [batch*seqLen, dModel] for linear projections.
	flat, err := tb.engine.Reshape(ctx, normed, []int{batch * seqLen, tb.dModel})
	if err != nil {
		return nil, fmt.Errorf("flatten for QKV: %w", err)
	}

	q, err := tb.qProj.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("Q projection: %w", err)
	}
	k, err := tb.kProj.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("K projection: %w", err)
	}
	v, err := tb.vProj.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("V projection: %w", err)
	}

	// Reshape back to [batch, seqLen, dModel].
	q, err = tb.engine.Reshape(ctx, q, []int{batch, seqLen, tb.dModel})
	if err != nil {
		return nil, fmt.Errorf("reshape Q: %w", err)
	}
	k, err = tb.engine.Reshape(ctx, k, []int{batch, seqLen, tb.dModel})
	if err != nil {
		return nil, fmt.Errorf("reshape K: %w", err)
	}
	v, err = tb.engine.Reshape(ctx, v, []int{batch, seqLen, tb.dModel})
	if err != nil {
		return nil, fmt.Errorf("reshape V: %w", err)
	}

	// Compute attention scores: Q @ K^T / sqrt(dModel).
	// K^T: [batch, dModel, seqLen]
	kT, err := tb.engine.Transpose(ctx, k, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("transpose K: %w", err)
	}

	// scores: [batch, seqLen, seqLen]
	scores, err := tb.engine.MatMul(ctx, q, kT)
	if err != nil {
		return nil, fmt.Errorf("attention scores: %w", err)
	}
	scores, err = tb.engine.MulScalar(ctx, scores, tb.scale)
	if err != nil {
		return nil, fmt.Errorf("scale scores: %w", err)
	}

	// Softmax along last dimension.
	attnWeights, err := tb.engine.Softmax(ctx, scores, -1)
	if err != nil {
		return nil, fmt.Errorf("softmax: %w", err)
	}

	// Weighted sum: [batch, seqLen, seqLen] @ [batch, seqLen, dModel] -> [batch, seqLen, dModel]
	attnOut, err := tb.engine.MatMul(ctx, attnWeights, v)
	if err != nil {
		return nil, fmt.Errorf("attention output: %w", err)
	}

	// Residual connection.
	x, err = tb.engine.Add(ctx, attnOut, residual)
	if err != nil {
		return nil, fmt.Errorf("attention residual: %w", err)
	}

	// --- FFN with residual ---
	residual = x

	normed, err = tb.norm2.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("norm2: %w", err)
	}

	flat, err = tb.engine.Reshape(ctx, normed, []int{batch * seqLen, tb.dModel})
	if err != nil {
		return nil, fmt.Errorf("flatten for FFN: %w", err)
	}

	ffnOut, err := tb.ffn1.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("FFN1: %w", err)
	}
	ffnOut, err = tb.gelu.Forward(ctx, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("FFN GELU: %w", err)
	}
	ffnOut, err = tb.ffn2.Forward(ctx, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("FFN2: %w", err)
	}

	// Reshape back to [batch, seqLen, dModel].
	ffnOut, err = tb.engine.Reshape(ctx, ffnOut, []int{batch, seqLen, tb.dModel})
	if err != nil {
		return nil, fmt.Errorf("reshape FFN output: %w", err)
	}

	// Residual connection.
	x, err = tb.engine.Add(ctx, ffnOut, residual)
	if err != nil {
		return nil, fmt.Errorf("FFN residual: %w", err)
	}

	return x, nil
}

// Parameters returns all trainable parameters.
func (tb *transformerBlock[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, tb.qProj.Parameters()...)
	params = append(params, tb.kProj.Parameters()...)
	params = append(params, tb.vProj.Parameters()...)
	params = append(params, tb.norm1.Parameters()...)
	params = append(params, tb.ffn1.Parameters()...)
	params = append(params, tb.ffn2.Parameters()...)
	params = append(params, tb.norm2.Parameters()...)
	return params
}

// DualSpaceOutput contains both fine-grained and semantic embeddings
// produced by the DualSpaceEncoder.
type DualSpaceOutput[T tensor.Float] struct {
	// FineGrained is the full patch-level representation [batch, numPatches, dModel].
	// Used for anomaly detection and imputation where per-timestep detail is needed.
	FineGrained *tensor.TensorNumeric[T]

	// Semantic is the mean-pooled series-level representation [batch, dModel].
	// Used for classification and similarity search.
	Semantic *tensor.TensorNumeric[T]
}

// DualSpaceEncoder processes time series in both time and frequency domains,
// following the dual-space masked reconstruction approach from IBM Granite TSPulse.
//
// The encoder splits input into patches, then processes them through two parallel
// paths — one in the time domain and one in the frequency domain (via DFT). The
// results are fused via concatenation and linear projection to produce embeddings
// that capture both temporal patterns and spectral characteristics.
type DualSpaceEncoder[T tensor.Float] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Time-domain path.
	timePatchEmbed *PatchEmbed[T]
	timeEncoder    []*transformerBlock[T]

	// Frequency-domain path.
	freqLinear1    *core.Linear[T] // process real part of DFT
	freqLinear2    *core.Linear[T] // process imag part of DFT
	freqPatchEmbed *PatchEmbed[T]
	freqEncoder    []*transformerBlock[T]

	// Fusion projection: [2*dModel] -> [dModel].
	fusionProj *core.Linear[T]

	dModel    int
	patchLen  int
	numLayers int

	// Precomputed DFT matrices for the given patch length.
	cosTable []T // [patchLen * patchLen]
	sinTable []T // [patchLen * patchLen]
}

// NewDualSpaceEncoder creates a new dual-space encoder.
//
// Parameters:
//   - engine: compute engine for tensor operations
//   - ops: arithmetic operations for the numeric type
//   - dModel: model embedding dimension
//   - patchLen: length of each time-domain patch
//   - numLayers: number of transformer encoder layers in each path
func NewDualSpaceEncoder[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	dModel, patchLen, numLayers int,
) (*DualSpaceEncoder[T], error) {
	if dModel <= 0 {
		return nil, fmt.Errorf("dModel must be positive, got %d", dModel)
	}
	if patchLen <= 0 {
		return nil, fmt.Errorf("patchLen must be positive, got %d", patchLen)
	}
	if numLayers <= 0 {
		return nil, fmt.Errorf("numLayers must be positive, got %d", numLayers)
	}

	// Time-domain path.
	timePatchEmbed, err := NewPatchEmbed[T]("dse_time_patch", engine, ops, patchLen, dModel)
	if err != nil {
		return nil, fmt.Errorf("create time patch embed: %w", err)
	}

	timeEnc := make([]*transformerBlock[T], numLayers)
	for i := range numLayers {
		timeEnc[i], err = newTransformerBlock[T](
			fmt.Sprintf("dse_time_enc_%d", i), engine, ops, dModel,
		)
		if err != nil {
			return nil, fmt.Errorf("create time encoder block %d: %w", i, err)
		}
	}

	// Frequency-domain path: linear layers for DFT coefficients.
	// DFT produces patchLen real + patchLen imag coefficients per patch.
	freqLinear1, err := core.NewLinear[T]("dse_freq_real", engine, ops, patchLen, patchLen)
	if err != nil {
		return nil, fmt.Errorf("create freq linear1: %w", err)
	}
	freqLinear2, err := core.NewLinear[T]("dse_freq_imag", engine, ops, patchLen, patchLen)
	if err != nil {
		return nil, fmt.Errorf("create freq linear2: %w", err)
	}

	freqPatchEmbed, err := NewPatchEmbed[T]("dse_freq_patch", engine, ops, patchLen, dModel)
	if err != nil {
		return nil, fmt.Errorf("create freq patch embed: %w", err)
	}

	freqEnc := make([]*transformerBlock[T], numLayers)
	for i := range numLayers {
		freqEnc[i], err = newTransformerBlock[T](
			fmt.Sprintf("dse_freq_enc_%d", i), engine, ops, dModel,
		)
		if err != nil {
			return nil, fmt.Errorf("create freq encoder block %d: %w", i, err)
		}
	}

	// Fusion: project concatenated [2*dModel] -> [dModel].
	fusionProj, err := core.NewLinear[T]("dse_fusion", engine, ops, 2*dModel, dModel)
	if err != nil {
		return nil, fmt.Errorf("create fusion projection: %w", err)
	}

	// Precompute DFT cosine and sine tables for the given patch length.
	cosTable := make([]T, patchLen*patchLen)
	sinTable := make([]T, patchLen*patchLen)
	for k := range patchLen {
		for n := range patchLen {
			angle := 2.0 * math.Pi * float64(k) * float64(n) / float64(patchLen)
			cosTable[k*patchLen+n] = T(math.Cos(angle))
			sinTable[k*patchLen+n] = T(math.Sin(angle))
		}
	}

	return &DualSpaceEncoder[T]{
		engine:         engine,
		ops:            ops,
		timePatchEmbed: timePatchEmbed,
		timeEncoder:    timeEnc,
		freqLinear1:    freqLinear1,
		freqLinear2:    freqLinear2,
		freqPatchEmbed: freqPatchEmbed,
		freqEncoder:    freqEnc,
		fusionProj:     fusionProj,
		dModel:         dModel,
		patchLen:       patchLen,
		numLayers:      numLayers,
		cosTable:       cosTable,
		sinTable:       sinTable,
	}, nil
}

// Forward processes the input through both time and frequency domain paths,
// fuses the results, and returns fine-grained and semantic embeddings.
//
// Input shape: [batch, seqLen] where seqLen is divisible by patchLen (or will be padded).
// Output: DualSpaceOutput with FineGrained [batch, numPatches, dModel] and Semantic [batch, dModel].
func (e *DualSpaceEncoder[T]) Forward(ctx context.Context, input *tensor.TensorNumeric[T]) (*DualSpaceOutput[T], error) {
	shape := input.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("DualSpaceEncoder input must be 2D [batch, seqLen], got shape %v", shape)
	}

	batch := shape[0]
	seqLen := shape[1]

	// Pad input if needed.
	paddedInput := input
	paddedLen := seqLen
	if seqLen%e.patchLen != 0 {
		paddedLen = ((seqLen + e.patchLen - 1) / e.patchLen) * e.patchLen
		padded, err := tensor.New[T]([]int{batch, paddedLen}, make([]T, batch*paddedLen))
		if err != nil {
			return nil, err
		}
		srcData := input.Data()
		dstData := padded.Data()
		for b := range batch {
			copy(dstData[b*paddedLen:b*paddedLen+seqLen], srcData[b*seqLen:(b+1)*seqLen])
		}
		paddedInput = padded
	}
	numPatches := paddedLen / e.patchLen

	// === Time-domain path ===
	timeEmb, err := e.timePatchEmbed.Forward(ctx, paddedInput)
	if err != nil {
		return nil, fmt.Errorf("time patch embed: %w", err)
	}
	for i, block := range e.timeEncoder {
		timeEmb, err = block.Forward(ctx, timeEmb)
		if err != nil {
			return nil, fmt.Errorf("time encoder block %d: %w", i, err)
		}
	}

	// === Frequency-domain path ===
	// Apply DFT to each patch, process in frequency space, then IDFT back.
	freqPatches, err := e.applyFrequencyPath(ctx, paddedInput, batch, numPatches)
	if err != nil {
		return nil, fmt.Errorf("frequency path: %w", err)
	}

	// Embed frequency-filtered patches.
	freqEmb, err := e.freqPatchEmbed.Forward(ctx, freqPatches)
	if err != nil {
		return nil, fmt.Errorf("freq patch embed: %w", err)
	}
	for i, block := range e.freqEncoder {
		freqEmb, err = block.Forward(ctx, freqEmb)
		if err != nil {
			return nil, fmt.Errorf("freq encoder block %d: %w", i, err)
		}
	}

	// === Fusion ===
	// Concatenate time and freq embeddings along feature dim.
	// timeEmb: [batch, numPatches, dModel], freqEmb: [batch, numPatches, dModel]
	// -> concat: [batch, numPatches, 2*dModel]
	concatData := make([]T, batch*numPatches*2*e.dModel)
	timeData := timeEmb.Data()
	freqData := freqEmb.Data()
	for b := range batch {
		for p := range numPatches {
			baseConcat := (b*numPatches + p) * 2 * e.dModel
			baseTime := (b*numPatches + p) * e.dModel
			baseFreq := (b*numPatches + p) * e.dModel
			copy(concatData[baseConcat:baseConcat+e.dModel], timeData[baseTime:baseTime+e.dModel])
			copy(concatData[baseConcat+e.dModel:baseConcat+2*e.dModel], freqData[baseFreq:baseFreq+e.dModel])
		}
	}
	concat, err := tensor.New[T]([]int{batch * numPatches, 2 * e.dModel}, concatData)
	if err != nil {
		return nil, fmt.Errorf("create concat tensor: %w", err)
	}

	// Project [batch*numPatches, 2*dModel] -> [batch*numPatches, dModel].
	fused, err := e.fusionProj.Forward(ctx, concat)
	if err != nil {
		return nil, fmt.Errorf("fusion projection: %w", err)
	}

	// Reshape to [batch, numPatches, dModel].
	fineGrained, err := e.engine.Reshape(ctx, fused, []int{batch, numPatches, e.dModel})
	if err != nil {
		return nil, fmt.Errorf("reshape fused: %w", err)
	}

	// === Semantic embedding: mean pool over patch dimension ===
	// ReduceMean over axis 1 (patch dim), keepDims=false -> [batch, dModel].
	semantic, err := e.engine.ReduceMean(ctx, fineGrained, 1, false)
	if err != nil {
		return nil, fmt.Errorf("mean pool: %w", err)
	}

	return &DualSpaceOutput[T]{
		FineGrained: fineGrained,
		Semantic:    semantic,
	}, nil
}

// applyFrequencyPath applies DFT to each patch, processes real and imaginary
// components through linear layers, then applies IDFT to recover time-domain
// patches filtered in frequency space.
//
// Input: [batch, paddedLen], Output: [batch, paddedLen] (frequency-filtered).
func (e *DualSpaceEncoder[T]) applyFrequencyPath(
	ctx context.Context,
	input *tensor.TensorNumeric[T],
	batch, numPatches int,
) (*tensor.TensorNumeric[T], error) {
	data := input.Data()
	L := e.patchLen
	totalPatches := batch * numPatches

	// DFT: compute real and imaginary parts for each patch.
	realParts := make([]T, totalPatches*L)
	imagParts := make([]T, totalPatches*L)

	for p := range totalPatches {
		patchStart := p * L
		for k := range L {
			var re, im T
			for n := range L {
				x := data[patchStart+n]
				re += x * e.cosTable[k*L+n]
				im -= x * e.sinTable[k*L+n]
			}
			realParts[p*L+k] = re
			imagParts[p*L+k] = im
		}
	}

	// Create tensors for linear processing: [totalPatches, patchLen].
	realTensor, err := tensor.New[T]([]int{totalPatches, L}, realParts)
	if err != nil {
		return nil, fmt.Errorf("create real tensor: %w", err)
	}
	imagTensor, err := tensor.New[T]([]int{totalPatches, L}, imagParts)
	if err != nil {
		return nil, fmt.Errorf("create imag tensor: %w", err)
	}

	// Process through linear layers in frequency space.
	realProc, err := e.freqLinear1.Forward(ctx, realTensor)
	if err != nil {
		return nil, fmt.Errorf("freq linear1 (real): %w", err)
	}
	imagProc, err := e.freqLinear2.Forward(ctx, imagTensor)
	if err != nil {
		return nil, fmt.Errorf("freq linear2 (imag): %w", err)
	}

	// IDFT: reconstruct time-domain signal from processed frequency components.
	// x[n] = (1/L) * sum_{k=0}^{L-1} (Re[k]*cos(2*pi*k*n/L) - Im[k]*sin(2*pi*k*n/L))
	realData := realProc.Data()
	imagData := imagProc.Data()
	outputData := make([]T, totalPatches*L)
	invL := T(1.0 / float64(L))

	for p := range totalPatches {
		for n := range L {
			var val T
			for k := range L {
				re := realData[p*L+k]
				im := imagData[p*L+k]
				val += re*e.cosTable[k*L+n] - im*e.sinTable[k*L+n]
			}
			outputData[p*L+n] = val * invL
		}
	}

	// Reshape back to [batch, paddedLen].
	result, err := tensor.New[T]([]int{batch, numPatches * L}, outputData)
	if err != nil {
		return nil, fmt.Errorf("create IDFT output: %w", err)
	}

	return result, nil
}

// Parameters returns all trainable parameters of the dual-space encoder.
func (e *DualSpaceEncoder[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]

	// Time-domain path.
	params = append(params, e.timePatchEmbed.Parameters()...)
	for _, block := range e.timeEncoder {
		params = append(params, block.Parameters()...)
	}

	// Frequency-domain path.
	params = append(params, e.freqLinear1.Parameters()...)
	params = append(params, e.freqLinear2.Parameters()...)
	params = append(params, e.freqPatchEmbed.Parameters()...)
	for _, block := range e.freqEncoder {
		params = append(params, block.Parameters()...)
	}

	// Fusion.
	params = append(params, e.fusionProj.Parameters()...)

	return params
}

// dftRoundTrip computes DFT followed by IDFT on raw patch data to verify
// the transform pair preserves the input. Exported for testing.
func dftRoundTrip[T tensor.Float](input []T, patchLen int) []T {
	L := patchLen

	// Precompute tables.
	cosTab := make([]float64, L*L)
	sinTab := make([]float64, L*L)
	for k := range L {
		for n := range L {
			angle := 2.0 * math.Pi * float64(k) * float64(n) / float64(L)
			cosTab[k*L+n] = math.Cos(angle)
			sinTab[k*L+n] = math.Sin(angle)
		}
	}

	numPatches := len(input) / L
	output := make([]T, len(input))

	for p := range numPatches {
		// Forward DFT.
		realParts := make([]float64, L)
		imagParts := make([]float64, L)
		for k := range L {
			for n := range L {
				x := float64(input[p*L+n])
				realParts[k] += x * cosTab[k*L+n]
				imagParts[k] -= x * sinTab[k*L+n]
			}
		}

		// Inverse DFT.
		invL := 1.0 / float64(L)
		for n := range L {
			var val float64
			for k := range L {
				val += realParts[k]*cosTab[k*L+n] - imagParts[k]*sinTab[k*L+n]
			}
			output[p*L+n] = T(val * invL)
		}
	}

	return output
}

// randomSlice generates a slice of random T values for testing and initialization.
func randomSlice[T tensor.Float](n int) []T {
	data := make([]T, n)
	for i := range data {
		data[i] = T(rand.Float64()*2 - 1)
	}
	return data
}
