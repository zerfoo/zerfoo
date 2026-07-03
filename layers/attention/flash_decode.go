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
//
// engStream is the producing engine's CUDA stream (compute.StreamProvider,
// proxy-unwrapped by the caller), or nil when the caller has no engine stream.
// When non-nil the kernel launches on that stream, so it is ordered AFTER the
// engine ops that produced Q/K/V. Launching on a private stream instead has no
// ordering against those producers and races the still-in-flight Q/K/V -- the
// same cross-stream race fixed for prefill in #866, tracked for decode in
// zerfoo#865 (docs/lore.md L-0006). When nil, a temporary stream is used
// (standalone use with synchronous inputs).
//
// out, partialO, and partialLSE are the caller-owned persistent scratch
// caches (zerfoo#870, docs/lore.md L-0006): the output and split-KV scratch
// buffers are allocated once (lazily, on first use) and reused across calls
// instead of being malloc'd and defer-freed per call, which is what made
// replay under training.CaptureReplayRunner dereference freed memory. Callers
// pass nil only when they know CUDA is unavailable (the function bails before
// touching any of them in that case); production callers (SDPA.Forward) pass
// buffers cached on the owning node so their lifetime spans the whole
// capture-replay life of the graph.
func tryFlashDecode[T tensor.Numeric](
	q, k, v *tensor.TensorNumeric[T],
	headDim int,
	numQueryHeads, numKVHeads int,
	engStream unsafe.Pointer,
	out, partialO *gpuScratchBuffer[T],
	partialLSE *gpuScratchBuffer[float32],
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

	// Output GPU storage: [batch*numQueryHeads, headDim]. Persistent (cached on
	// the caller-owned scratch buffer) so its device address survives across
	// CUDA-graph replays -- see the gpuScratchBuffer doc comment and zerfoo#870.
	oGPU, err := out.view(numBH*headDim, qGPU.DeviceID())
	if err != nil {
		return nil, err
	}

	// Split-KV scratch buffers. Also persistent: these were previously
	// malloc'd and defer-freed per call, which is exactly the "per-call
	// scratch with defer-frees inside the captured region" landmine
	// (docs/lore.md L-0006) that crashed replay ~#141/511 under
	// training.CaptureReplayRunner (zerfoo#870).
	const chunkSize = 256
	numSplits := (kvLen + chunkSize - 1) / chunkSize

	partialOGPU, err := partialO.view(numBH*numSplits*headDim, qGPU.DeviceID())
	if err != nil {
		return nil, err
	}

	// partialLSE stores 2 floats (max, sum) per (bh, split) pair.
	// Allocate as float32 regardless of T — the kernel always uses f32 for LSE.
	partialLSEGPU, err := partialLSE.view(2*numBH*numSplits, qGPU.DeviceID())
	if err != nil {
		return nil, err
	}

	// Resolve the stream to launch on. With an engine stream, launch
	// stream-ordered after the Q/K/V producers (fixes the cross-stream race,
	// zerfoo#865). Without one, create a temporary stream for standalone use.
	launchStream := engStream
	if launchStream == nil {
		tmp, err := cuda.CreateStream()
		if err != nil {
			oGPU.Free() //nolint:errcheck
			return nil, err
		}
		defer func() { _ = tmp.Destroy() }()
		launchStream = tmp.Ptr()
	}

	if err := kernels.FlashDecodeSplitKV(
		qGPU.Ptr(), kGPU.Ptr(), vGPU.Ptr(), oGPU.Ptr(),
		partialOGPU.Ptr(), partialLSEGPU.Ptr(),
		numBH, kvLen, headDim, kvLen,
		unsafe.Pointer(nil), // kvLenPtr: not using CUDA graph capture here
		numQueryHeads, numKVHeads,
		chunkSize,
		launchStream,
	); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	// Synchronize before returning so the output is ready for the synchronous
	// caller (single-token / latency-bound; matches the legacy cost). The
	// output and split-KV scratch are now persistent (zerfoo#870) so this sync
	// is no longer protecting a defer-free race -- but it is still a genuine
	// host-blocking call on the launch stream, which is illegal mid-capture.
	// Capture-replay decode therefore remains out of scope (tracked in the
	// #865->#870->#878 cluster, docs/lore.md L-0006): decode is an
	// inference-time (KV-cache) path CaptureReplayRunner's training walk does
	// not exercise, so this is a pre-existing, documented limitation, not a
	// regression -- callers that DO reach this under capture get a loud
	// EndCapture error instead of a silent replay corruption.
	if err := cuda.StreamFromPtr(launchStream).Synchronize(); err != nil {
		oGPU.Free() //nolint:errcheck
		return nil, err
	}

	// Return with shape [batch*numQueryHeads, 1, headDim] to match expected SDPA output.
	return tensor.NewWithStorage([]int{numBH, 1, headDim}, oGPU)
}
