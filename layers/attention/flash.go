//go:build !(rocm && cutlass)

package attention

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
	"github.com/zerfoo/zerfoo/tensor"
)

// tryFlashForward attempts to use the fused flash attention CUDA kernel.
// It returns (result, nil) on success, (nil, nil) when flash attention is not
// applicable (CPU data, head_dim > 128, or CUDA not available), or (nil, err)
// on kernel failure.
//
// Q, K, V must be 3D tensors with shape [batch*heads, seq_len, head_dim].
// The kernel treats each element of the first dimension as an independent
// (batch, head) pair (i.e., batch=batch*heads, heads=1).
func tryFlashForward[T tensor.Numeric](
	q, k, v *tensor.TensorNumeric[T],
	headDim int,
	causal bool,
) (*tensor.TensorNumeric[T], error) {
	if !cuda.Available() {
		return nil, nil
	}

	// Flash attention prefill kernel supports head_dim up to 128 (limited by
	// register pressure from per-thread q_row[MAX_HEAD_DIM] and acc[MAX_HEAD_DIM]).
	// For head_dim > 128 during decode, use tryFlashDecode instead.
	if headDim > 128 {
		return nil, nil
	}

	// Check that Q is on GPU.
	if q.GetStorage().DeviceType() != device.CUDA {
		return nil, nil
	}

	// Extract GPU storage pointers. The type assertion is safe because
	// DeviceType() == CUDA implies GPUStorage.
	qGPU := q.GetStorage().(*tensor.GPUStorage[T])
	kGPU := k.GetStorage().(*tensor.GPUStorage[T])
	vGPU := v.GetStorage().(*tensor.GPUStorage[T])

	shape := q.Shape() // [batchHeads, seqLen, headDim]
	batchHeads := shape[0]
	seqLen := shape[1]

	// Allocate output on the same GPU device.
	n := batchHeads * seqLen * headDim
	oGPU, err := tensor.NewGPUStorage[T](n, qGPU.DeviceID())
	if err != nil {
		return nil, err
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}
	defer stream.Destroy()

	// Call the kernel with batch=batchHeads, heads=1 so each element of the
	// first dimension maps to one thread block row in the kernel.
	if err := kernels.FlashAttentionForward(
		qGPU.Ptr(), kGPU.Ptr(), vGPU.Ptr(), oGPU.Ptr(),
		batchHeads, 1, seqLen, headDim, causal, stream.Ptr(),
	); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	if err := stream.Synchronize(); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	return tensor.NewWithStorage(shape, oGPU)
}

// tryFlashDecode attempts to use the decode-specific flash attention kernel.
// This handles the autoregressive decode case where Q has 1 token but K/V
// have kvSeqLen tokens (from the KV cache). Supports GQA where numQueryHeads
// may differ from numKVHeads.
//
// Q:  [batch*numQueryHeads, 1, headDim]
// K:  [batch*numKVHeads, maxKVLen, headDim]  -- pre-allocated KV buffer
// V:  [batch*numKVHeads, maxKVLen, headDim]
//
// kvSeqLen:       actual KV sequence length (used when kvLenPtr is nil).
// kvLenPtr:       GPU-resident int32 pointer; when non-nil, the kernel reads
//                 the KV length from GPU memory, making it CUDA graph compatible.
// maxKVLen:       stride of the KV buffer (allocated capacity).
// numQueryHeads:  number of query heads per batch element.
// numKVHeads:     number of KV heads per batch element.
// stream:         CUDA stream for kernel launch.
//
// Returns (result, nil) on success, (nil, nil) when not applicable.
func tryFlashDecode[T tensor.Numeric](
	q, k, v *tensor.TensorNumeric[T],
	headDim, kvSeqLen, maxKVLen int,
	kvLenPtr unsafe.Pointer,
	numQueryHeads, numKVHeads int,
	stream unsafe.Pointer,
) (*tensor.TensorNumeric[T], error) {
	if !cuda.Available() {
		return nil, nil
	}

	if headDim > 256 {
		return nil, nil
	}

	// Q must have seqLen=1 for decode.
	qShape := q.Shape()
	if len(qShape) < 3 || qShape[1] != 1 {
		return nil, nil
	}

	if q.GetStorage().DeviceType() != device.CUDA {
		return nil, nil
	}

	qGPU := q.GetStorage().(*tensor.GPUStorage[T])
	kGPU := k.GetStorage().(*tensor.GPUStorage[T])
	vGPU := v.GetStorage().(*tensor.GPUStorage[T])

	batchHeads := qShape[0]

	// Output: [batchHeads, 1, headDim]
	n := batchHeads * headDim
	oGPU, err := tensor.NewGPUStorage[T](n, qGPU.DeviceID())
	if err != nil {
		return nil, err
	}

	if err := kernels.FlashAttentionDecode(
		qGPU.Ptr(), kGPU.Ptr(), vGPU.Ptr(), oGPU.Ptr(),
		batchHeads, maxKVLen, headDim, kvSeqLen,
		kvLenPtr, numQueryHeads, numKVHeads, stream,
	); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	outShape := []int{batchHeads, 1, headDim}
	return tensor.NewWithStorage(outShape, oGPU)
}
