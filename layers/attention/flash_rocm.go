//go:build rocm && cutlass

package attention

import (
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/zerfoo/internal/hip"
	"github.com/zerfoo/zerfoo/internal/hip/kernels"
	"github.com/zerfoo/ztensor/tensor"
)

// tryFlashForward attempts to use the fused flash attention HIP kernel.
// It returns (result, nil) on success, (nil, nil) when flash attention is not
// applicable (CPU data, head_dim > 128), or (nil, err) on kernel failure.
//
// Q, K, V must be 3D tensors with shape [batch*heads, seq_len, head_dim].
// The kernel treats each element of the first dimension as an independent
// (batch, head) pair (i.e., batch=batch*heads, heads=1).
//
// engStream is the producing engine's stream (compute.StreamProvider), or
// nil. When non-nil the kernel launches on it without a host synchronize so
// the launch is ordered after the Q/K/V producers (see the CUDA variant in
// flash.go for the cross-stream race this prevents). When nil, the legacy
// temporary-stream + host-sync behavior is kept.
func tryFlashForward[T tensor.Numeric](
	q, k, v *tensor.TensorNumeric[T],
	headDim int,
	causal bool,
	engStream unsafe.Pointer,
) (*tensor.TensorNumeric[T], error) {
	// Flash attention supports head_dim up to 128.
	if headDim > 128 {
		return nil, nil
	}

	// The flash prefill kernel assumes Q and K/V share the same sequence
	// length (it uses a single seq_len for all pointer arithmetic). During
	// autoregressive decode with KV cache, Q has seqLen=1 while K/V have
	// seqLen=cachedSeqLen. Using Q's seqLen to index into K/V produces wrong
	// offsets and garbage output. Bail out so the standard SDPA path handles it.
	if q.Shape()[1] != k.Shape()[1] {
		return nil, nil
	}

	// The kernel uses Q's batch*heads dimension to index into K and V.
	// If K/V have a different batch dimension (e.g., single KV head before
	// repeat), the kernel would read out of bounds. Bail out.
	if q.Shape()[0] != k.Shape()[0] {
		return nil, nil
	}

	// Check that Q is on GPU.
	if q.GetStorage().DeviceType() != device.ROCm {
		return nil, nil
	}

	// Extract GPU storage pointers. The type assertion is safe because
	// DeviceType() == ROCm implies GPUStorage.
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

	// Engine-stream path: launch stream-ordered after the Q/K/V producers.
	if engStream != nil {
		if err := kernels.FlashAttentionForward(
			qGPU.Ptr(), kGPU.Ptr(), vGPU.Ptr(), oGPU.Ptr(),
			batchHeads, 1, seqLen, headDim, causal, engStream,
		); err != nil {
			oGPU.Free() //nolint:errcheck
			return nil, err
		}
		return tensor.NewWithStorage(shape, oGPU)
	}

	stream, err := hip.CreateStream()
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
