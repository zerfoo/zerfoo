package attention

import "github.com/zerfoo/ztensor/tensor"

// gpuScratchBuffer caches a single GPU device allocation across calls so a
// flash-attention kernel's scratch/output buffer is replay-stable under
// CUDA-graph capture (docs/lore.md L-0006, zerfoo#870).
//
// tryFlashForward and tryFlashDecode bypass the compute.Engine arena and
// allocate their scratch/output buffers directly via tensor.NewGPUStorage
// (raw cudaMalloc), because they operate below the engine abstraction on
// bare device pointers. Doing that per call and releasing it before
// returning (the previous behavior: a defer'd Free) is safe in eager
// execution but fatal under training.CaptureReplayRunner: a CUDA graph
// bakes in the literal device pointer used by the kernel launch it records,
// so freeing that allocation makes it available for reuse by any later,
// unrelated cudaMalloc -- and the graph keeps replaying kernels against the
// stale address. zerfoo#870 crashed with an illegal memory access around
// replay #141/511 once something else's allocation happened to land on the
// freed split-KV scratch.
//
// The fix: allocate once, lazily, and reuse the SAME buffer on every call
// instead of freeing it. A gpuScratchBuffer is meant to be a field on a
// graph node (e.g. ScaledDotProductAttention) so its lifetime matches the
// node's -- which spans the whole capture-replay life of the graph it
// belongs to. CaptureReplayRunner's own contract (static shapes across every
// captured step) guarantees the requested size is unchanged from the eager
// warmup steps (which is when growth, i.e. a real free+malloc, actually
// happens) through every subsequent captured/replayed call.
type gpuScratchBuffer[T tensor.Numeric] struct {
	storage  *tensor.GPUStorage[T]
	capacity int
	deviceID int
}

// view returns a non-owning GPUStorage view of exactly elems elements backed
// by the cached allocation, growing (freeing the old allocation and making a
// new one) first if the cache is empty, undersized, or on a different
// device. The returned view's Free() is a no-op, so callers on error paths
// may call Free() on it without releasing the persistent buffer early; the
// buffer itself is only released via release (or its GC finalizer).
func (b *gpuScratchBuffer[T]) view(elems, deviceID int) (*tensor.GPUStorage[T], error) {
	if b.storage == nil || b.capacity < elems || b.deviceID != deviceID {
		if b.storage != nil {
			_ = b.storage.Free()
		}

		s, err := tensor.NewGPUStorage[T](elems, deviceID)
		if err != nil {
			b.storage = nil
			b.capacity = 0

			return nil, err
		}

		b.storage = s
		b.capacity = elems
		b.deviceID = deviceID
	}

	return b.storage.View(elems), nil
}

// release frees the cached allocation, if any. Intended for tests and any
// explicit node-teardown path; ordinary operation relies on the buffer
// living for the node's lifetime and the GC finalizer on the underlying
// GPUStorage as the backstop.
func (b *gpuScratchBuffer[T]) release() {
	if b.storage != nil {
		_ = b.storage.Free()
		b.storage = nil
		b.capacity = 0
	}
}
