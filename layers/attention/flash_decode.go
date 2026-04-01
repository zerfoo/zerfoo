//go:build !(rocm && cutlass)

package attention

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/tensor"
)

// tryFlashDecode attempts to use the split-KV flash decode CUDA kernel for
// single-query autoregressive decode. Returns (result, nil) on success,
// (nil, nil) when not applicable, or (nil, err) on kernel failure.
//
// This is the DECODE path (seqLen_Q == 1). For prefill (seqLen_Q > 1),
// use tryFlashForward instead.
//
// Q: [batch*numQueryHeads, 1, headDim]
// K: [batch*numKVHeads, seqKV, headDim]  (full KV cache)
// V: [batch*numKVHeads, seqKV, headDim]
func tryFlashDecode[T tensor.Numeric](
	q, k, v *tensor.TensorNumeric[T],
	headDim int,
	numQueryHeads, numKVHeads int,
) (*tensor.TensorNumeric[T], error) {
	if !cuda.Available() {
		return nil, nil
	}

	if !kernels.IsFlashDecodeSplitKVSupported() {
		return nil, nil
	}

	// Only applicable for decode mode (single query token).
	if q.Shape()[1] != 1 {
		return nil, nil
	}

	// Register pressure limit: head_dim up to 128.
	if headDim > 128 {
		return nil, nil
	}

	// Q, K, V must all be on GPU.
	if q.GetStorage().DeviceType() != device.CUDA {
		return nil, nil
	}
	if k.GetStorage().DeviceType() != device.CUDA {
		return nil, nil
	}
	if v.GetStorage().DeviceType() != device.CUDA {
		return nil, nil
	}

	// Extract GPU storage pointers.
	qGPU := q.GetStorage().(*tensor.GPUStorage[T])
	kGPU := k.GetStorage().(*tensor.GPUStorage[T])
	vGPU := v.GetStorage().(*tensor.GPUStorage[T])

	// Q shape: [batch*numQueryHeads, 1, headDim]
	numBH := q.Shape()[0] // batch * numQueryHeads
	kvLen := k.Shape()[1] // KV sequence length

	// Allocate output GPU storage: [batch*numQueryHeads, headDim].
	oGPU, err := tensor.NewGPUStorage[T](numBH*headDim, qGPU.DeviceID())
	if err != nil {
		return nil, err
	}

	// Split-KV scratch buffers.
	const chunkSize = 256
	numSplits := (kvLen + chunkSize - 1) / chunkSize

	partialOGPU, err := tensor.NewGPUStorage[T](numBH*numSplits*headDim, qGPU.DeviceID())
	if err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}
	defer partialOGPU.Free() //nolint:errcheck

	// partialLSE stores 2 floats (max, sum) per (bh, split) pair.
	// Allocate as float32 regardless of T — the kernel always uses f32 for LSE.
	partialLSEGPU, err := tensor.NewGPUStorage[float32](2*numBH*numSplits, qGPU.DeviceID())
	if err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}
	defer partialLSEGPU.Free() //nolint:errcheck

	stream, err := cuda.CreateStream()
	if err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}
	defer func() { _ = stream.Destroy() }()

	if err := kernels.FlashDecodeSplitKV(
		qGPU.Ptr(), kGPU.Ptr(), vGPU.Ptr(), oGPU.Ptr(),
		partialOGPU.Ptr(), partialLSEGPU.Ptr(),
		numBH, kvLen, headDim, kvLen,
		unsafe.Pointer(nil), // kvLenPtr: not using CUDA graph capture here
		numQueryHeads, numKVHeads,
		chunkSize,
		stream.Ptr(),
	); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	if err := stream.Synchronize(); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	// Return with shape [batch*numQueryHeads, 1, headDim] to match expected SDPA output.
	return tensor.NewWithStorage([]int{numBH, 1, headDim}, oGPU)
}
