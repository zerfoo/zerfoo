package embeddings

import (
	"context"
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// RotaryPositionalEmbedding applies Rotary Positional Embedding to a tensor.
type RotaryPositionalEmbedding[T tensor.Numeric] struct {
	engine    compute.Engine[T]
	headDim   int
	rotaryDim int // number of dimensions that receive rotation (<= headDim)
	cosAngles *tensor.TensorNumeric[T]
	sinAngles *tensor.TensorNumeric[T]
	// gpuUploaded tracks whether cos/sin have been uploaded to GPU.
	gpuUploaded bool
	// Cached input for backward pass
	inputShape  []int
	xRot0Slice  *tensor.TensorNumeric[T]
	xRot1Slice  *tensor.TensorNumeric[T]
	outputShape []int
	// attnScaleFactor is the YaRN attention scaling factor (1.0 when no YaRN).
	attnScaleFactor float64
	// posOffset shifts the cos/sin angle slicing during Forward.
	// During autoregressive decode, set this to the current cache sequence
	// length so the new token gets the correct positional rotation.
	posOffset int
	// documentBoundaries holds sequence positions where document boundaries
	// occur. When set, position IDs reset to 0 at each boundary so that each
	// document receives independent positional encoding. Boundaries must be
	// sorted in ascending order and refer to positions within the sequence.
	documentBoundaries []int
}

// RotaryPositionalEmbeddingOptions holds configuration options for RotaryPositionalEmbedding layers.
type RotaryPositionalEmbeddingOptions struct {
	Base            float64 // Base for the inverse frequency calculation (theta parameter)
	YaRN            bool    // Whether to apply YaRN scaling
	YaRNFactor      float64 // YaRN scaling factor (e.g. 4.0 for 4x context extension)
	YaRNOrigML      int     // Original max sequence length before scaling
	RotaryDimFraction float64 // Fraction of head dims to rotate (default 1.0 = all)
}

// RotaryPositionalEmbeddingOption is a functional option for configuring RotaryPositionalEmbedding layers.
type RotaryPositionalEmbeddingOption func(*RotaryPositionalEmbeddingOptions)

// WithRotaryBase sets the base (theta) parameter for the inverse frequency calculation.
func WithRotaryBase(base float64) RotaryPositionalEmbeddingOption {
	return func(opts *RotaryPositionalEmbeddingOptions) {
		opts.Base = base
	}
}

// WithRotaryDimFraction sets the fraction of head dimensions that receive rotation.
// Default is 1.0 (all dimensions rotated). Phi-4 uses 0.75 for partial RoPE.
func WithRotaryDimFraction(fraction float64) RotaryPositionalEmbeddingOption {
	return func(opts *RotaryPositionalEmbeddingOptions) {
		opts.RotaryDimFraction = fraction
	}
}

// WithYaRNScaling enables YaRN (Yet another RoPE extensioN) scaling.
// factor is the context extension factor (e.g. 4.0 for 4x).
// origMaxLen is the original maximum sequence length before scaling.
func WithYaRNScaling(factor float64, origMaxLen int) RotaryPositionalEmbeddingOption {
	return func(opts *RotaryPositionalEmbeddingOptions) {
		opts.YaRN = true
		opts.YaRNFactor = factor
		opts.YaRNOrigML = origMaxLen
	}
}

// NewRotaryPositionalEmbedding creates a new RotaryPositionalEmbedding layer.
// headDim: The dimension of the head. Must be even.
// seqLen: The maximum sequence length this embedding will be applied to.
// engine: The compute engine to use for tensor operations.
func NewRotaryPositionalEmbedding[T tensor.Numeric](
	ctx context.Context,
	engine compute.Engine[T],
	headDim int,
	seqLen int,
	options ...RotaryPositionalEmbeddingOption,
) (*RotaryPositionalEmbedding[T], error) {
	if headDim%2 != 0 {
		return nil, fmt.Errorf("head dimension (%d) must be even for RoPE", headDim)
	}

	// Apply functional options
	opts := &RotaryPositionalEmbeddingOptions{
		Base:              10000.0, // Default base value (theta)
		RotaryDimFraction: 1.0,
	}
	for _, option := range options {
		option(opts)
	}

	// Compute the number of dimensions that receive rotation.
	rotaryDim := headDim
	if opts.RotaryDimFraction > 0 && opts.RotaryDimFraction < 1.0 {
		rotaryDim = int(float64(headDim) * opts.RotaryDimFraction)
		// Ensure rotaryDim is even.
		rotaryDim &^= 1
	}

	// Create position indices: [0, 1, ..., seq_len-1]
	positions := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		positions[i] = i
	}

	// Create inverse frequencies: 1 / (base^(2i/rotaryDim))
	ops := engine.Ops()
	halfDim := rotaryDim / 2
	invFreqs64 := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		invFreqs64[i] = 1.0 / math.Pow(opts.Base, float64(2*i)/float64(rotaryDim))
	}

	// Apply YaRN scaling to inverse frequencies if enabled.
	attnScaleFactor := 1.0
	if opts.YaRN {
		attnScaleFactor = math.Sqrt(1 + math.Log(opts.YaRNFactor)/math.Log(float64(opts.YaRNOrigML)))
		origML := float64(opts.YaRNOrigML)
		for i := 0; i < halfDim; i++ {
			wavelength := 2 * math.Pi / invFreqs64[i]
			if wavelength > opts.YaRNFactor*origML {
				// Low frequency: scale by 1/factor
				invFreqs64[i] /= opts.YaRNFactor
			} else if wavelength >= origML {
				// Intermediate frequency: linearly interpolate
				ratio := (wavelength - origML) / (opts.YaRNFactor*origML - origML)
				invFreqs64[i] = invFreqs64[i] * (1 - ratio) + invFreqs64[i]/opts.YaRNFactor*ratio
			}
			// High frequency (wavelength < origMaxLen): keep unchanged
		}
	}

	// Precompute cos and sin of angles using float64 and convert to T
	size := seqLen * halfDim
	cosData := make([]T, size)
	sinData := make([]T, size)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < halfDim; j++ {
			angle := float64(positions[i]) * invFreqs64[j]
			idx := i*halfDim + j
			cosData[idx] = ops.FromFloat64(math.Cos(angle))
			sinData[idx] = ops.FromFloat64(math.Sin(angle))
		}
	}

	cosAngles, err := tensor.New[T]([]int{seqLen, halfDim}, cosData)
	if err != nil {
		return nil, err
	}

	sinAngles, err := tensor.New[T]([]int{seqLen, halfDim}, sinData)
	if err != nil {
		return nil, err
	}

	return &RotaryPositionalEmbedding[T]{
		engine:          engine,
		headDim:         headDim,
		rotaryDim:       rotaryDim,
		cosAngles:       cosAngles,
		sinAngles:       sinAngles,
		attnScaleFactor: attnScaleFactor,
	}, nil
}

// SetDocumentBoundaries sets document boundary positions for document-wise
// RoPE. When boundaries are set, position IDs reset to 0 at each boundary
// so each document gets independent positional encoding. Boundaries are
// sequence positions (0-indexed) where new documents begin.
// Pass nil to disable document-wise mode.
func (rpe *RotaryPositionalEmbedding[T]) SetDocumentBoundaries(boundaries []int) {
	rpe.documentBoundaries = boundaries
}

// gatherDocumentWiseAngles builds cos/sin tensors with document-local
// positions. Position IDs reset to 0 at each boundary in documentBoundaries.
// The result has shape [seqLen, halfRotary].
func (rpe *RotaryPositionalEmbedding[T]) gatherDocumentWiseAngles(seqLen, halfRotary int) (
	cos, sin *tensor.TensorNumeric[T], err error,
) {
	// Compute document-local position for each sequence position.
	localPositions := make([]int, seqLen)
	boundaryIdx := 0
	localPos := 0
	for i := 0; i < seqLen; i++ {
		// Check if this position is a document boundary.
		if boundaryIdx < len(rpe.documentBoundaries) && i == rpe.documentBoundaries[boundaryIdx] {
			localPos = 0
			boundaryIdx++
		}
		localPositions[i] = localPos
		localPos++
	}

	// Gather cos/sin rows from the precomputed tables using local positions.
	cosData := rpe.cosAngles.Data()
	sinData := rpe.sinAngles.Data()
	cosTableStride := rpe.cosAngles.Shape()[1] // halfRotary (or rotaryDim/2)

	gatheredCos := make([]T, seqLen*halfRotary)
	gatheredSin := make([]T, seqLen*halfRotary)
	for i := 0; i < seqLen; i++ {
		srcOff := localPositions[i] * cosTableStride
		dstOff := i * halfRotary
		copy(gatheredCos[dstOff:dstOff+halfRotary], cosData[srcOff:srcOff+halfRotary])
		copy(gatheredSin[dstOff:dstOff+halfRotary], sinData[srcOff:srcOff+halfRotary])
	}

	cos, err = tensor.New[T]([]int{seqLen, halfRotary}, gatheredCos)
	if err != nil {
		return nil, nil, err
	}
	sin, err = tensor.New[T]([]int{seqLen, halfRotary}, gatheredSin)
	if err != nil {
		return nil, nil, err
	}
	return cos, sin, nil
}

// SetPositionOffset sets the position offset for the next Forward call.
// During autoregressive decode, call this with the current cache sequence
// length so that the new token is rotated at the correct absolute position
// instead of always position 0.
func (rpe *RotaryPositionalEmbedding[T]) SetPositionOffset(offset int) {
	rpe.posOffset = offset
}

// OutputShape returns the output shape of the RoPE layer.
func (rpe *RotaryPositionalEmbedding[T]) OutputShape() []int {
	return rpe.outputShape
}

// Parameters returns no trainable parameters for RoPE.
func (rpe *RotaryPositionalEmbedding[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Forward applies Rotary Positional Embedding to the input tensor.
func (rpe *RotaryPositionalEmbedding[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RotaryPositionalEmbedding expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	rpe.inputShape = input.Shape()

	rpe.outputShape = input.Shape()
	if len(rpe.inputShape) < 2 {
		return nil, fmt.Errorf("input tensor must have at least 2 dimensions, got %d", len(rpe.inputShape))
	}

	seqLen := rpe.inputShape[1]
	halfRotary := rpe.rotaryDim / 2

	// Lazily upload cos/sin tables to GPU on first forward pass when
	// the engine supports GPU. This eliminates per-token H2D copies
	// that dominated getDevicePtr overhead in the decode loop.
	// Unwrap EngineProxy so the check works even when the engine is wrapped.
	if !rpe.gpuUploaded {
		checkEngine := compute.Engine[T](rpe.engine)
		if proxy, ok := rpe.engine.(*compute.EngineProxy[T]); ok {
			checkEngine = proxy.Real()
		}
		if _, ok := checkEngine.(compute.WeightUploader); ok {
			if gpuCos, err := tensor.ToGPU(rpe.cosAngles); err == nil {
				rpe.cosAngles = gpuCos
			}
			if gpuSin, err := tensor.ToGPU(rpe.sinAngles); err == nil {
				rpe.sinAngles = gpuSin
			}
		}
		rpe.gpuUploaded = true
	}

	// Slice cos/sin angles using posOffset so decode tokens get the correct
	// absolute position rotation instead of always position 0.
	off := rpe.posOffset
	var cosSliced, sinSliced *tensor.TensorNumeric[T]
	var err error

	// Document-wise RoPE: gather cos/sin rows using document-local positions
	// instead of a contiguous slice. Position IDs reset to 0 at each
	// document boundary so each document gets independent positional encoding.
	if len(rpe.documentBoundaries) > 0 {
		cosSliced, sinSliced, err = rpe.gatherDocumentWiseAngles(seqLen, halfRotary)
		if err != nil {
			return nil, err
		}
	} else if cosGS, ok := rpe.cosAngles.GetStorage().(*tensor.GPUStorage[T]); ok {
		// GPU path: create non-owning views into GPU-resident cos/sin tables
		// to avoid tensor.Slice() which unconditionally creates CPUStorage.
		cosView := tensor.NewGPUStorageView(cosGS, off*halfRotary, seqLen*halfRotary)
		cosSliced, err = tensor.NewWithStorage[T]([]int{seqLen, halfRotary}, cosView)
		if err != nil {
			return nil, err
		}
		sinGS := rpe.sinAngles.GetStorage().(*tensor.GPUStorage[T])
		sinView := tensor.NewGPUStorageView(sinGS, off*halfRotary, seqLen*halfRotary)
		sinSliced, err = tensor.NewWithStorage[T]([]int{seqLen, halfRotary}, sinView)
		if err != nil {
			return nil, err
		}
	} else {
		cosSliced, err = rpe.cosAngles.Slice([2]int{off, off + seqLen}, [2]int{0, halfRotary})
		if err != nil {
			return nil, err
		}
		sinSliced, err = rpe.sinAngles.Slice([2]int{off, off + seqLen}, [2]int{0, halfRotary})
		if err != nil {
			return nil, err
		}
	}

	// Fused single-pass kernel (inference hot path).
	// Unwrap EngineProxy to detect the real engine type, so the fused path
	// is taken even when the engine is wrapped.
	realEngine := compute.Engine[T](rpe.engine)
	if proxy, ok := rpe.engine.(*compute.EngineProxy[T]); ok {
		realEngine = proxy.Real()
	}
	// GPU fused RoPE: one kernel launch replaces Split + 4 Mul + Sub + Add + Concat.
	if provider, ok := realEngine.(compute.FusedRoPEProvider[T]); ok {
		out, err := provider.GPUFusedRoPE(input, cosSliced, sinSliced, rpe.rotaryDim)
		if err == nil {
			rpe.outputShape = input.Shape()
			// Skip backward cache (xRot0Slice/xRot1Slice) on GPU path.
			// Inference never calls Backward, and input.Slice() on GPU
			// tensors triggers expensive D2H copies.
			return out, nil
		}
		// Fall through to unfused path on error.
	}
	// CPU fused RoPE.
	if _, isCPU := realEngine.(*compute.CPUEngine[T]); isCPU {
		if f32Input, ok := any(input).(*tensor.TensorNumeric[float32]); ok {
			f32Cos, cOk := any(cosSliced).(*tensor.TensorNumeric[float32])
			f32Sin, sOk := any(sinSliced).(*tensor.TensorNumeric[float32])
			if cOk && sOk {
				out, err := compute.FusedRoPE(f32Input, f32Cos, f32Sin, rpe.rotaryDim)
				if err == nil {
					rpe.outputShape = input.Shape()
					rpe.xRot0Slice, _ = input.Slice([2]int{0, rpe.inputShape[0]}, [2]int{0, seqLen}, [2]int{0, halfRotary})
					rpe.xRot1Slice, _ = input.Slice([2]int{0, rpe.inputShape[0]}, [2]int{0, seqLen}, [2]int{halfRotary, rpe.rotaryDim})
					return any(out).(*tensor.TensorNumeric[T]), nil
				}
				// Fall through to unfused path on error (e.g. non-3D input).
			}
		}
	}

	cosAngles := cosSliced
	sinAngles := sinSliced

	// Split rotary portion into two halves: x_rot0, x_rot1.
	// When rotaryDim equals the full last dimension, use engine.Split to
	// keep data on the GPU and avoid costly GPU→CPU copies via tensor.Slice.
	if rpe.rotaryDim == rpe.inputShape[len(rpe.inputShape)-1] {
		halves, splitErr := rpe.engine.Split(ctx, input, 2, len(rpe.inputShape)-1)
		if splitErr != nil {
			return nil, splitErr
		}
		rpe.xRot0Slice = halves[0]
		rpe.xRot1Slice = halves[1]
	} else {
		// Partial RoPE (rotaryDim < headDim): fall back to tensor.Slice.
		rpe.xRot0Slice, err = input.Slice([2]int{0, rpe.inputShape[0]}, [2]int{0, seqLen}, [2]int{0, halfRotary})
		if err != nil {
			return nil, err
		}

		rpe.xRot1Slice, err = input.Slice([2]int{0, rpe.inputShape[0]}, [2]int{0, seqLen}, [2]int{halfRotary, rpe.rotaryDim})
		if err != nil {
			return nil, err
		}
	}

	// Apply rotation:
	// x_rot0 * cos(angles) - x_rot1 * sin(angles)
	// x_rot1 * cos(angles) + x_rot0 * sin(angles)
	term1, err := rpe.engine.Mul(ctx, rpe.xRot0Slice, cosAngles)
	if err != nil {
		return nil, err
	}

	term2, err := rpe.engine.Mul(ctx, rpe.xRot1Slice, sinAngles)
	if err != nil {
		return nil, err
	}

	rotatedX0, err := rpe.engine.Sub(ctx, term1, term2)
	if err != nil {
		return nil, err
	}

	mul1, err := rpe.engine.Mul(ctx, rpe.xRot1Slice, cosAngles)
	if err != nil {
		return nil, err
	}

	mul2, err := rpe.engine.Mul(ctx, rpe.xRot0Slice, sinAngles)
	if err != nil {
		return nil, err
	}

	rotatedX1, err := rpe.engine.Add(ctx, mul1, mul2)
	if err != nil {
		return nil, err
	}

	// Concatenate rotated halves, plus pass-through if partial.
	parts := []*tensor.TensorNumeric[T]{rotatedX0, rotatedX1}
	if rpe.rotaryDim < rpe.headDim {
		passThrough, err2 := input.Slice([2]int{0, rpe.inputShape[0]}, [2]int{0, seqLen}, [2]int{rpe.rotaryDim, rpe.headDim})
		if err2 != nil {
			return nil, err2
		}
		parts = append(parts, passThrough)
	}

	output, err := rpe.engine.Concat(ctx, parts, 2)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for RoPE.
// Shapes are derived from dOut so that a single RoPE instance can be shared
// across Q and K paths whose batch dimensions differ.
func (rpe *RotaryPositionalEmbedding[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	dShape := dOut.Shape()
	batchDim := dShape[0]
	seqLen := dShape[1]
	halfRotary := rpe.rotaryDim / 2

	// Slice cos and sin angles to match the input sequence length
	cosAngles, err := rpe.cosAngles.Slice([2]int{0, seqLen}, [2]int{0, halfRotary})
	if err != nil {
		return nil, err
	}

	sinAngles, err := rpe.sinAngles.Slice([2]int{0, seqLen}, [2]int{0, halfRotary})
	if err != nil {
		return nil, err
	}

	// Split dOut rotary portion into d_rotated_x0, d_rotated_x1
	dRotatedX0, err := dOut.Slice([2]int{0, batchDim}, [2]int{0, seqLen}, [2]int{0, halfRotary})
	if err != nil {
		return nil, err
	}

	dRotatedX1, err := dOut.Slice([2]int{0, batchDim}, [2]int{0, seqLen}, [2]int{halfRotary, rpe.rotaryDim})
	if err != nil {
		return nil, err
	}

	// dL/dx_rot0 = d_rotated_x0 * cos(angles) + d_rotated_x1 * sin(angles)
	mul1, err := rpe.engine.Mul(ctx, dRotatedX0, cosAngles)
	if err != nil {
		return nil, err
	}

	mul2, err := rpe.engine.Mul(ctx, dRotatedX1, sinAngles)
	if err != nil {
		return nil, err
	}

	dLdxRot0, err := rpe.engine.Add(ctx, mul1, mul2)
	if err != nil {
		return nil, err
	}

	// dL/dx_rot1 = d_rotated_x1 * cos(angles) - d_rotated_x0 * sin(angles)
	mul3, err := rpe.engine.Mul(ctx, dRotatedX1, cosAngles)
	if err != nil {
		return nil, err
	}

	mul4, err := rpe.engine.Mul(ctx, dRotatedX0, sinAngles)
	if err != nil {
		return nil, err
	}

	dLdxRot1, err := rpe.engine.Sub(ctx, mul3, mul4)
	if err != nil {
		return nil, err
	}

	// Concatenate rotary gradients, plus pass-through if partial.
	parts := []*tensor.TensorNumeric[T]{dLdxRot0, dLdxRot1}
	if rpe.rotaryDim < rpe.headDim {
		dPassThrough, err2 := dOut.Slice([2]int{0, batchDim}, [2]int{0, seqLen}, [2]int{rpe.rotaryDim, rpe.headDim})
		if err2 != nil {
			return nil, err2
		}
		parts = append(parts, dPassThrough)
	}

	dInput, err := rpe.engine.Concat(ctx, parts, 2)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// OpType returns the operation type of the RotaryPositionalEmbedding layer.
func (rpe *RotaryPositionalEmbedding[T]) OpType() string {
	return "RotaryPositionalEmbedding"
}

// Attributes returns the attributes of the RotaryPositionalEmbedding layer.
func (rpe *RotaryPositionalEmbedding[T]) Attributes() map[string]interface{} {
	return nil
}

// GetAngles returns the cos/sin angle tensors for the given position range,
// along with halfRotary. For GPU-resident tables, returns non-owning views.
// This is used by the fused QK norm+RoPE kernel during decode.
func (rpe *RotaryPositionalEmbedding[T]) GetAngles(offset, seqLen int) (cos, sin *tensor.TensorNumeric[T], halfRotary int, err error) {
	halfRotary = rpe.rotaryDim / 2

	// Lazily upload cos/sin tables to GPU if not done yet.
	if !rpe.gpuUploaded {
		checkEngine := compute.Engine[T](rpe.engine)
		if proxy, ok := rpe.engine.(*compute.EngineProxy[T]); ok {
			checkEngine = proxy.Real()
		}
		if _, ok := checkEngine.(compute.WeightUploader); ok {
			if gpuCos, uploadErr := tensor.ToGPU(rpe.cosAngles); uploadErr == nil {
				rpe.cosAngles = gpuCos
			}
			if gpuSin, uploadErr := tensor.ToGPU(rpe.sinAngles); uploadErr == nil {
				rpe.sinAngles = gpuSin
			}
		}
		rpe.gpuUploaded = true
	}

	// GPU path: create non-owning views.
	if cosGS, ok := rpe.cosAngles.GetStorage().(*tensor.GPUStorage[T]); ok {
		cosView := tensor.NewGPUStorageView(cosGS, offset*halfRotary, seqLen*halfRotary)
		cos, err = tensor.NewWithStorage[T]([]int{seqLen, halfRotary}, cosView)
		if err != nil {
			return nil, nil, 0, err
		}
		sinGS := rpe.sinAngles.GetStorage().(*tensor.GPUStorage[T])
		sinView := tensor.NewGPUStorageView(sinGS, offset*halfRotary, seqLen*halfRotary)
		sin, err = tensor.NewWithStorage[T]([]int{seqLen, halfRotary}, sinView)
		if err != nil {
			return nil, nil, 0, err
		}
		return cos, sin, halfRotary, nil
	}

	// CPU path.
	cos, err = rpe.cosAngles.Slice([2]int{offset, offset + seqLen}, [2]int{0, halfRotary})
	if err != nil {
		return nil, nil, 0, err
	}
	sin, err = rpe.sinAngles.Slice([2]int{offset, offset + seqLen}, [2]int{0, halfRotary})
	if err != nil {
		return nil, nil, 0, err
	}
	return cos, sin, halfRotary, nil
}

// GetAnglesGPU returns cos/sin angle tensors selected by a GPU-resident
// counter, avoiding CPU-side offset computation. This enables CUDA graph
// capture of the decode loop by keeping all position-dependent reads on GPU.
// counterPtr is a device pointer to an int32 position counter (from GPUKVCache).
// stream is the CUDA stream (unsafe.Pointer to cudaStream_t) for kernel launch.
// seqLen is the number of positions to select (1 for decode).
func (rpe *RotaryPositionalEmbedding[T]) GetAnglesGPU(counterPtr unsafe.Pointer, seqLen int, stream unsafe.Pointer) (
	cos, sin *tensor.TensorNumeric[T], halfRotary int, err error,
) {
	halfRotary = rpe.rotaryDim / 2

	// Lazily upload cos/sin tables to GPU if not done yet.
	if !rpe.gpuUploaded {
		checkEngine := compute.Engine[T](rpe.engine)
		if proxy, ok := rpe.engine.(*compute.EngineProxy[T]); ok {
			checkEngine = proxy.Real()
		}
		if _, ok := checkEngine.(compute.WeightUploader); ok {
			if gpuCos, uploadErr := tensor.ToGPU(rpe.cosAngles); uploadErr == nil {
				rpe.cosAngles = gpuCos
			}
			if gpuSin, uploadErr := tensor.ToGPU(rpe.sinAngles); uploadErr == nil {
				rpe.sinAngles = gpuSin
			}
		}
		rpe.gpuUploaded = true
	}

	cosGS, ok := rpe.cosAngles.GetStorage().(*tensor.GPUStorage[T])
	if !ok {
		return nil, nil, 0, fmt.Errorf("GetAnglesGPU: cos table not on GPU")
	}
	sinGS, ok := rpe.sinAngles.GetStorage().(*tensor.GPUStorage[T])
	if !ok {
		return nil, nil, 0, fmt.Errorf("GetAnglesGPU: sin table not on GPU")
	}

	// Allocate GPU output buffers for the selected cos/sin slice.
	outLen := seqLen * halfRotary
	cosOut, err := tensor.NewGPUStorage[T](outLen)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("GetAnglesGPU: alloc cos output: %w", err)
	}
	sinOut, err := tensor.NewGPUStorage[T](outLen)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("GetAnglesGPU: alloc sin output: %w", err)
	}

	// Launch rope_select kernel: reads counter[0] on GPU to index into the table.
	if err := kernels.RoPESelect(
		cosGS.Ptr(), sinGS.Ptr(),
		cosOut.Ptr(), sinOut.Ptr(),
		counterPtr, halfRotary, stream,
	); err != nil {
		return nil, nil, 0, fmt.Errorf("GetAnglesGPU: rope_select kernel: %w", err)
	}

	cos, err = tensor.NewWithStorage[T]([]int{seqLen, halfRotary}, cosOut)
	if err != nil {
		return nil, nil, 0, err
	}
	sin, err = tensor.NewWithStorage[T]([]int{seqLen, halfRotary}, sinOut)
	if err != nil {
		return nil, nil, 0, err
	}

	return cos, sin, halfRotary, nil
}

// RotaryDim returns the number of dimensions that receive rotation.
func (rpe *RotaryPositionalEmbedding[T]) RotaryDim() int {
	return rpe.rotaryDim
}

// AttentionScaleFactor returns the YaRN attention scaling factor.
// Returns 1.0 when YaRN is not enabled.
func (rpe *RotaryPositionalEmbedding[T]) AttentionScaleFactor() float64 {
	if rpe.attnScaleFactor == 0 {
		return 1.0
	}
	return rpe.attnScaleFactor
}

// Scale scales the positional embeddings by a given factor.
func (rpe *RotaryPositionalEmbedding[T]) Scale(ctx context.Context, factor float64) error {
	ops := rpe.engine.Ops()
	scaledCos, err := rpe.engine.MulScalar(ctx, rpe.cosAngles, ops.FromFloat64(factor), nil)
	if err != nil {
		return err
	}
	rpe.cosAngles = scaledCos

	scaledSin, err := rpe.engine.MulScalar(ctx, rpe.sinAngles, ops.FromFloat64(factor), nil)
	if err != nil {
		return err
	}
	rpe.sinAngles = scaledSin

	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*RotaryPositionalEmbedding[float32])(nil)
