//go:build !(rocm && cutlass)

package attention

import (
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
	"github.com/zerfoo/ztensor/tensor"
)

// tryFlashForward attempts to use the fused flash attention CUDA kernel.
// It returns (result, nil) on success, (nil, nil) when flash attention is not
// applicable (CPU data, head_dim > 128, or CUDA not available), or (nil, err)
// on kernel failure.
//
// Q, K, V must be 3D tensors with shape [batch*heads, seq_len, head_dim].
// The kernel treats each element of the first dimension as an independent
// (batch, head) pair (i.e., batch=batch*heads, heads=1).
//
// engStream is the producing engine's CUDA stream (compute.StreamProvider),
// or nil when the caller has no engine stream. When non-nil the kernel is
// launched on that stream WITHOUT a host synchronize: Q/K/V were enqueued by
// engine ops on the same stream, so the launch is stream-ordered after their
// producers, and downstream engine ops (or the engine's host-access sync
// hooks, ztensor#137) order the output. When nil, the legacy behavior is
// kept: a temporary stream is created and host-synchronized before return.
// Launching on a private temporary stream while the inputs are still being
// produced on the engine stream is a cross-stream race -- the kernel can read
// in-flight Q/K/V (observed as a silently-wrong training trajectory in Wolf's
// CrossAsset GB10 run, fold-0 acc 0.7042 -> 0.4948 at identical seed).
func tryFlashForward[T tensor.Numeric](
	q, k, v *tensor.TensorNumeric[T],
	headDim int,
	causal bool,
	engStream unsafe.Pointer,
) (*tensor.TensorNumeric[T], error) {
	if !cuda.Available() {
		return nil, nil
	}

	// Flash attention prefill kernel supports head_dim up to 128 (limited by
	// register pressure from per-thread q_row[MAX_HEAD_DIM] and acc[MAX_HEAD_DIM]).
	// For head_dim > 128, the kernel is not applicable.
	if headDim > 128 {
		return nil, nil
	}

	// The flash prefill kernel assumes Q and K/V share the same sequence
	// length (it uses a single seq_len for all pointer arithmetic). During
	// autoregressive decode with KV cache, Q has seqLen=1 while K/V have
	// seqLen=cachedSeqLen. Using Q's seqLen to index into K/V produces wrong
	// offsets and garbage output. Bail out so the cuBLAS SDPA path handles it.
	if q.Shape()[1] != k.Shape()[1] {
		return nil, nil
	}

	// The kernel uses Q's batch*heads dimension to index into K and V.
	// If K/V have a different batch dimension (e.g., single KV head before
	// repeat), the kernel would read out of bounds. Bail out.
	if q.Shape()[0] != k.Shape()[0] {
		return nil, nil
	}

	// Check that Q, K, V are all on GPU.
	if q.GetStorage().DeviceType() != device.CUDA {
		return nil, nil
	}
	if k.GetStorage().DeviceType() != device.CUDA {
		return nil, nil
	}
	if v.GetStorage().DeviceType() != device.CUDA {
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

	// Engine-stream path: launch stream-ordered after the Q/K/V producers.
	// No host synchronize -- consumers on the same stream are ordered, and
	// host access goes through the engine's host-access sync hooks.
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

	// Legacy path (no engine stream known): temporary stream + host sync.
	// Only safe when the caller guarantees Q/K/V are not in-flight on
	// another stream (e.g. standalone use with synchronous uploads).
	stream, err := cuda.CreateStream()
	if err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}
	defer func() { _ = stream.Destroy() }()

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
