package generate

import (
	"fmt"
	"log/slog"
	"unsafe"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/ztensor/tensor"
)

// tensorLayerBuf holds pre-allocated K/V buffers for a single layer.
// For GPU-resident tensors, buffers are backed by GPUStorage with direct D2D
// memcpy appends. For CPU tensors, flat Go slices are used with copy.
type tensorLayerBuf[T tensor.Numeric] struct {
	// GPU path: pre-allocated persistent GPU memory (F32 mode).
	kStorage *tensor.GPUStorage[T]
	vStorage *tensor.GPUStorage[T]
	// CPU path: pre-allocated flat buffers.
	kBuf []T
	vBuf []T

	// FP16 KV cache: when kvFP16 is true, these hold the half-precision
	// cache buffers (2 bytes/element instead of 4). The F32 kStorage/vStorage
	// are used as scratch space for F32<->FP16 conversion on Get.
	kFP16 *tensor.GPUStorage[float16.Float16]
	vFP16 *tensor.GPUStorage[float16.Float16]

	batch  int
	dim    int
	seqLen int
	isGPU  bool
}

// TensorCache is a KV cache that keeps tensors in pre-allocated buffers.
// On the first Update for a layer, it allocates [batch, maxSeqLen, dim] memory
// (GPU or CPU depending on the source tensor). Subsequent Updates append new
// K/V data via direct memcpy at the correct offset, avoiding per-token
// allocation overhead.
type TensorCache[T tensor.Numeric] struct {
	engine    compute.Engine[T]
	layers    []tensorLayerBuf[T]
	maxSeqLen int
	stream    gpuapi.Stream // GPU stream for async D2D/H2D copies (nil for CPU)
	kvFP16    bool          // when true, store KV cache in FP16 (GPU-only)

	// Shared F32 scratch buffers for FP16→F32 readback in Get.
	// Sized to batch*maxSeqLen*dim on first use. Reused across layers
	// (Get is called sequentially per layer).
	fp16ScratchK *tensor.GPUStorage[T]
	fp16ScratchV *tensor.GPUStorage[T]

	// GPU-resident int32 position counter for CUDA graph capture.
	// When non-nil, Update uses offset_memcpy kernel (reads counter on GPU)
	// instead of CPU-computed offsets, making the KV append capturable.
	gpuCounter *tensor.GPUStorage[int32]

	// GPU-resident int32 KV sequence length counter. Tracks the total number
	// of tokens in the KV cache. The flash_attention_decode kernel reads this
	// from GPU memory so the KV length is not frozen by CUDA graph capture.
	//
	// Incremented by 1 in Update() for the first layer (layer 0) during
	// single-token decode, before attention runs for any layer.
	kvSeqLenCounter *tensor.GPUStorage[int32]
}

// TensorCacheOption configures a TensorCache.
type TensorCacheOption func(*tensorCacheOptions)

type tensorCacheOptions struct {
	kvFP16 bool
}

// WithKVDtype sets the KV cache storage dtype. Supported values: "fp32" (default), "fp16".
// FP16 mode halves KV memory bandwidth but requires GPU and CUDA conversion kernels.
func WithKVDtype(dtype string) TensorCacheOption {
	return func(o *tensorCacheOptions) {
		o.kvFP16 = dtype == "fp16"
	}
}

// NewTensorCache creates a TensorCache backed by the given engine.
// numLayers should match the model's transformer layer count.
// maxSeqLen limits the total cached sequence length.
// If the engine implements GPUStreamAccessor, async memcpy is used for
// KV cache updates (required for CUDA graph capture compatibility).
func NewTensorCache[T tensor.Numeric](engine compute.Engine[T], numLayers, maxSeqLen int, opts ...TensorCacheOption) *TensorCache[T] {
	var o tensorCacheOptions
	for _, opt := range opts {
		opt(&o)
	}
	tc := &TensorCache[T]{
		engine:    engine,
		layers:    make([]tensorLayerBuf[T], numLayers),
		maxSeqLen: maxSeqLen,
		kvFP16:    o.kvFP16,
	}
	if sa, ok := any(engine).(compute.GPUStreamAccessor); ok {
		tc.stream = sa.GPUStream()
	}
	// Allocate a GPU-resident int32 counter for CUDA graph capture.
	// The counter tracks the current sequence position on the GPU so that
	// offset_memcpy and rope_select kernels can read it without D2H copies.
	if tc.stream != nil {
		if gs, err := tensor.NewGPUStorageFromSlice([]int32{0}); err == nil {
			tc.gpuCounter = gs
		}
		// KV sequence length counter: tracks total tokens in cache.
		// Used by flash_attention_decode to read kv_len from GPU memory.
		if gs, err := tensor.NewGPUStorageFromSlice([]int32{0}); err == nil {
			tc.kvSeqLenCounter = gs
		}
	}
	return tc
}

// Update appends new key and value tensors to the cache for the given layer.
// Tensors must be 3D with shape [batch, seqLen, dim]. On the first call for
// a layer, pre-allocated buffers are created. Subsequent calls copy new data
// directly into the buffers at the current sequence offset.
func (c *TensorCache[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	if layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, len(c.layers))
	}
	shape := newK.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("expected 3D tensor [batch, seq, dim], got %dD", len(shape))
	}
	batch, seqLen, dim := shape[0], shape[1], shape[2]
	lb := &c.layers[layer]

	if lb.seqLen+seqLen > c.maxSeqLen {
		return fmt.Errorf("cache overflow: cursor=%d + seq=%d > maxSeqLen=%d", lb.seqLen, seqLen, c.maxSeqLen)
	}

	// Pre-allocate buffers on first call.
	if lb.batch == 0 {
		lb.batch = batch
		lb.dim = dim
		totalElems := batch * c.maxSeqLen * dim

		// Try GPU allocation first; fall back to CPU buffers.
		_, isGPU := newK.GetStorage().(*tensor.GPUStorage[T])
		if isGPU && c.kvFP16 {
			// FP16 mode: allocate half-size FP16 buffers for the cache,
			// plus F32 scratch buffers for readback conversion in Get.
			if err := allocFP16Buffers(lb, totalElems); err != nil {
				isGPU = false
			}
		}
		if isGPU && !c.kvFP16 {
			kSt, err := tensor.NewGPUStorage[T](totalElems)
			if err != nil {
				isGPU = false
			} else {
				vSt, err2 := tensor.NewGPUStorage[T](totalElems)
				if err2 != nil {
					_ = kSt.Free()
					isGPU = false
				} else {
					lb.kStorage = kSt
					lb.vStorage = vSt
				}
			}
		}
		if !isGPU {
			lb.kBuf = make([]T, totalElems)
			lb.vBuf = make([]T, totalElems)
		}
		lb.isGPU = isGPU
	}

	// If the cache layer is CPU-backed but the incoming tensor is GPU-resident,
	// promote the cache to GPU to avoid D2H copies on .Data() calls.
	if !lb.isGPU {
		if _, srcIsGPU := newK.GetStorage().(*tensor.GPUStorage[T]); srcIsGPU {
			if err := promoteToGPU(lb, batch, c.maxSeqLen, dim); err != nil {
				slog.Warn("KV cache GPU promotion failed, D2H fallback", "layer", layer, "error", err)
			}
		}
	}

	// GPU counter path: use offset_memcpy kernel for single-token decode.
	// The kernel reads the position from GPU memory, so the offset is not
	// baked into captured CUDA graphs — it stays live across replays.
	// The GPU counter is synced to the CPU seqLen at the end of prefill
	// (see below), so by decode time it already holds the correct position.
	//
	// Increment kvSeqLenCounter at the first layer (layer 0) so that the
	// flash_attention_decode kernel sees the correct KV length including
	// the token being appended in this decode step. The increment is a
	// GPU kernel (graph-capturable), so it stays live across replays.
	if c.kvSeqLenCounter != nil && seqLen == 1 && layer == 0 {
		if err := kernels.IncrementCounter(c.kvSeqLenCounter.Ptr(), 1, c.stream.Ptr()); err != nil {
			return fmt.Errorf("increment kv_seq_len counter: %w", err)
		}
	}
	if c.gpuCounter != nil && seqLen == 1 && lb.isGPU {
		kGS, kOK := newK.GetStorage().(*tensor.GPUStorage[T])
		vGS, vOK := newV.GetStorage().(*tensor.GPUStorage[T])
		if kOK && vOK {
			counterPtr := c.gpuCounter.Ptr()
			streamPtr := c.stream.Ptr()
			tokenDim := dim * batch

			if c.kvFP16 && lb.kFP16 != nil {
				// FP16 path: offset_memcpy_fp16 converts F32 src to FP16
				// and writes at the GPU-counter offset. Graph-capturable.
				if err := kernels.OffsetMemcpyFP16(lb.kFP16.Ptr(), kGS.Ptr(), counterPtr, tokenDim, c.maxSeqLen, streamPtr); err != nil {
					return fmt.Errorf("offset_memcpy_fp16 K layer %d: %w", layer, err)
				}
				if err := kernels.OffsetMemcpyFP16(lb.vFP16.Ptr(), vGS.Ptr(), counterPtr, tokenDim, c.maxSeqLen, streamPtr); err != nil {
					return fmt.Errorf("offset_memcpy_fp16 V layer %d: %w", layer, err)
				}
			} else {
				if err := kernels.OffsetMemcpy(lb.kStorage.Ptr(), kGS.Ptr(), counterPtr, tokenDim, c.maxSeqLen, streamPtr); err != nil {
					return fmt.Errorf("offset_memcpy K layer %d: %w", layer, err)
				}
				if err := kernels.OffsetMemcpy(lb.vStorage.Ptr(), vGS.Ptr(), counterPtr, tokenDim, c.maxSeqLen, streamPtr); err != nil {
					return fmt.Errorf("offset_memcpy V layer %d: %w", layer, err)
				}
			}

			// Advance GPU counter and CPU seqLen after the last layer.
			if layer == len(c.layers)-1 {
				if err := kernels.IncrementCounter(counterPtr, 1, streamPtr); err != nil {
					return fmt.Errorf("increment_counter layer %d: %w", layer, err)
				}
			}
			lb.seqLen += seqLen
			return nil
		}
	}

	// Append data at current offset.
	offset := lb.seqLen * dim * batch
	numElems := seqLen * dim * batch

	switch {
	case lb.isGPU && c.kvFP16 && lb.kFP16 != nil:
		// FP16 path: convert F32 source to FP16, then copy into FP16 cache.
		if err := appendFP16(lb.kFP16, offset, numElems, newK, c.stream); err != nil {
			return fmt.Errorf("append K fp16 layer %d: %w", layer, err)
		}
		if err := appendFP16(lb.vFP16, offset, numElems, newV, c.stream); err != nil {
			return fmt.Errorf("append V fp16 layer %d: %w", layer, err)
		}
	case lb.isGPU:
		if err := appendGPU(lb.kStorage, offset, numElems, newK, c.stream); err != nil {
			return fmt.Errorf("append K layer %d: %w", layer, err)
		}
		if err := appendGPU(lb.vStorage, offset, numElems, newV, c.stream); err != nil {
			return fmt.Errorf("append V layer %d: %w", layer, err)
		}
	default:
		if _, srcIsGPU := newK.GetStorage().(*tensor.GPUStorage[T]); srcIsGPU {
			slog.Warn("KV cache CPU fallback with GPU tensor, D2H copy triggered", "layer", layer)
		}
		copy(lb.kBuf[offset:offset+numElems], newK.Data())
		copy(lb.vBuf[offset:offset+numElems], newV.Data())
	}

	lb.seqLen += seqLen

	// After prefill (seqLen > 1) completes for the last layer, sync the GPU
	// counters to match the CPU seqLen. This ensures that when decode starts,
	// offset_memcpy, rope_select, and flash_attention_decode kernels read
	// the correct position from GPU memory. We use H2D memcpy here (outside
	// CUDA graph capture) since prefill always runs before capture.
	if c.gpuCounter != nil && seqLen > 1 && layer == len(c.layers)-1 {
		val := int32(lb.seqLen)
		if err := c.gpuCounter.CopyFromHost([]int32{val}, 0); err != nil {
			return fmt.Errorf("sync GPU counter to %d after prefill: %w", val, err)
		}
	}
	if c.kvSeqLenCounter != nil && seqLen > 1 && layer == len(c.layers)-1 {
		val := int32(lb.seqLen)
		if err := c.kvSeqLenCounter.CopyFromHost([]int32{val}, 0); err != nil {
			return fmt.Errorf("sync KV seqLen counter to %d after prefill: %w", val, err)
		}
	}

	return nil
}

// promoteToGPU migrates a CPU-backed cache layer to GPU storage. Existing
// cached data (if any) is uploaded via H2D copy. On failure the layer remains
// CPU-backed and the caller should log a warning.
func promoteToGPU[T tensor.Numeric](lb *tensorLayerBuf[T], batch, maxSeqLen, dim int) error {
	totalElems := batch * maxSeqLen * dim
	kSt, err := tensor.NewGPUStorage[T](totalElems)
	if err != nil {
		return fmt.Errorf("alloc K GPU storage: %w", err)
	}
	vSt, err := tensor.NewGPUStorage[T](totalElems)
	if err != nil {
		_ = kSt.Free()
		return fmt.Errorf("alloc V GPU storage: %w", err)
	}

	// Upload any existing CPU-cached data.
	if lb.seqLen > 0 {
		cached := lb.seqLen * dim * batch
		if err := kSt.CopyFromHost(lb.kBuf[:cached], 0); err != nil {
			_ = kSt.Free()
			_ = vSt.Free()
			return fmt.Errorf("upload K: %w", err)
		}
		if err := vSt.CopyFromHost(lb.vBuf[:cached], 0); err != nil {
			_ = kSt.Free()
			_ = vSt.Free()
			return fmt.Errorf("upload V: %w", err)
		}
	}

	lb.kStorage = kSt
	lb.vStorage = vSt
	lb.kBuf = nil
	lb.vBuf = nil
	lb.isGPU = true
	return nil
}

// appendGPU copies tensor data into the pre-allocated GPU buffer at the given
// element offset, using D2D memcpy for GPU sources or H2D for CPU sources.
// When stream is non-nil, async memcpy is used (required for CUDA graph
// capture compatibility).
func appendGPU[T tensor.Numeric](dst *tensor.GPUStorage[T], offset, numElems int, src *tensor.TensorNumeric[T], stream gpuapi.Stream) error {
	if gs, ok := src.GetStorage().(*tensor.GPUStorage[T]); ok {
		if stream != nil {
			return dst.CopyFromDeviceAsync(gs, offset, 0, numElems, stream)
		}
		return dst.CopyFromDevice(gs, offset, 0, numElems)
	}
	if stream != nil {
		return dst.CopyFromHostAsync(src.Data(), offset, stream)
	}
	return dst.CopyFromHost(src.Data(), offset)
}

// allocFP16Buffers allocates half-precision GPU buffers for the KV cache layer.
func allocFP16Buffers[T tensor.Numeric](lb *tensorLayerBuf[T], totalElems int) error {
	kFP16, err := tensor.NewGPUStorage[float16.Float16](totalElems)
	if err != nil {
		return fmt.Errorf("alloc K FP16 storage: %w", err)
	}
	vFP16, err := tensor.NewGPUStorage[float16.Float16](totalElems)
	if err != nil {
		_ = kFP16.Free()
		return fmt.Errorf("alloc V FP16 storage: %w", err)
	}
	lb.kFP16 = kFP16
	lb.vFP16 = vFP16
	return nil
}

// appendFP16 converts F32 source tensor data to FP16 and writes directly into
// the FP16 cache buffer at the given element offset. No temporary buffer is
// allocated — the F32→FP16 kernel writes directly to dst + offset, which is
// safe because the kernel is stream-ordered (async) and touches no other memory.
func appendFP16[T tensor.Numeric](dst *tensor.GPUStorage[float16.Float16], offset, numElems int, src *tensor.TensorNumeric[T], stream gpuapi.Stream) error {
	srcGS, isGPU := src.GetStorage().(*tensor.GPUStorage[T])
	if !isGPU {
		return fmt.Errorf("FP16 KV cache requires GPU-resident source tensors")
	}

	// Convert F32 → FP16 directly into the cache at the correct offset.
	// Pointer arithmetic: offset is in elements, each FP16 element is 2 bytes.
	dstPtr := unsafe.Add(dst.Ptr(), offset*2)

	streamPtr := unsafe.Pointer(nil)
	if stream != nil {
		streamPtr = stream.Ptr()
	}
	if err := kernels.F32ToFP16(srcGS.Ptr(), dstPtr, numElems, streamPtr); err != nil {
		return fmt.Errorf("f32_to_fp16: %w", err)
	}

	return nil
}

// ensureFP16Scratch allocates or grows the shared F32 scratch buffers for
// FP16→F32 readback in Get. The scratch is reused across layers.
func (c *TensorCache[T]) ensureFP16Scratch(minElems int) error {
	if c.fp16ScratchK != nil && c.fp16ScratchK.Len() >= minElems {
		return nil
	}
	if c.fp16ScratchK != nil {
		_ = c.fp16ScratchK.Free()
	}
	if c.fp16ScratchV != nil {
		_ = c.fp16ScratchV.Free()
	}
	var err error
	c.fp16ScratchK, err = tensor.NewGPUStorage[T](minElems)
	if err != nil {
		return fmt.Errorf("alloc FP16 scratch K: %w", err)
	}
	c.fp16ScratchV, err = tensor.NewGPUStorage[T](minElems)
	if err != nil {
		_ = c.fp16ScratchK.Free()
		c.fp16ScratchK = nil
		return fmt.Errorf("alloc FP16 scratch V: %w", err)
	}
	return nil
}

// Get returns the cached key-value pair for the given layer.
// For GPU-backed layers, returns a view into the pre-allocated buffer.
// For CPU-backed layers, returns a tensor wrapping the buffer slice.
// Returns false if the layer index is out of range or the layer is empty.
func (c *TensorCache[T]) Get(layer int) (*LayerKV[T], bool) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, false
	}
	lb := &c.layers[layer]
	if lb.seqLen == 0 || lb.batch == 0 {
		return nil, false
	}

	viewElems := lb.batch * lb.seqLen * lb.dim
	viewShape := []int{lb.batch, lb.seqLen, lb.dim}

	if lb.isGPU && c.kvFP16 && lb.kFP16 != nil {
		// FP16 readback: convert cached FP16 data to F32 scratch buffers.
		if err := c.ensureFP16Scratch(viewElems); err != nil {
			return nil, false
		}
		streamPtr := unsafe.Pointer(nil)
		if c.stream != nil {
			streamPtr = c.stream.Ptr()
		}
		kView := tensor.NewGPUStorageView(c.fp16ScratchK, 0, viewElems)
		if err := kernels.FP16ToF32(lb.kFP16.Ptr(), kView.Ptr(), viewElems, streamPtr); err != nil {
			return nil, false
		}
		kTensor, _ := tensor.NewWithStorage(viewShape, kView)

		vView := tensor.NewGPUStorageView(c.fp16ScratchV, 0, viewElems)
		if err := kernels.FP16ToF32(lb.vFP16.Ptr(), vView.Ptr(), viewElems, streamPtr); err != nil {
			return nil, false
		}
		vTensor, _ := tensor.NewWithStorage(viewShape, vView)
		return &LayerKV[T]{Key: kTensor, Value: vTensor}, true
	}

	if lb.isGPU {
		kView := tensor.NewGPUStorageView(lb.kStorage, 0, viewElems)
		vView := tensor.NewGPUStorageView(lb.vStorage, 0, viewElems)
		kTensor, _ := tensor.NewWithStorage(viewShape, kView)
		vTensor, _ := tensor.NewWithStorage(viewShape, vView)
		return &LayerKV[T]{Key: kTensor, Value: vTensor}, true
	}

	// CPU path: wrap the pre-allocated slice.
	kTensor, _ := tensor.New(viewShape, lb.kBuf[:viewElems])
	vTensor, _ := tensor.New(viewShape, lb.vBuf[:viewElems])
	return &LayerKV[T]{Key: kTensor, Value: vTensor}, true
}

// GetFullBuffer returns GPU-backed KV tensors spanning the full pre-allocated
// buffer (maxSeqLen capacity) for the given layer. The shape is
// [batch, maxSeqLen, dim]. This is used by flash_attention_decode which reads
// the actual KV length from a GPU-resident counter, so it needs the buffer
// with its full stride rather than a seqLen-trimmed view.
// Returns nil if the layer is CPU-backed or not yet initialized.
func (c *TensorCache[T]) GetFullBuffer(layer int) (k, v *tensor.TensorNumeric[T]) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, nil
	}
	lb := &c.layers[layer]
	if lb.batch == 0 || !lb.isGPU {
		return nil, nil
	}

	fullElems := lb.batch * c.maxSeqLen * lb.dim
	fullShape := []int{lb.batch, c.maxSeqLen, lb.dim}

	if c.kvFP16 && lb.kFP16 != nil {
		// FP16 path: convert full buffer to F32 scratch.
		if err := c.ensureFP16Scratch(fullElems); err != nil {
			return nil, nil
		}
		streamPtr := unsafe.Pointer(nil)
		if c.stream != nil {
			streamPtr = c.stream.Ptr()
		}
		kView := tensor.NewGPUStorageView(c.fp16ScratchK, 0, fullElems)
		if err := kernels.FP16ToF32(lb.kFP16.Ptr(), kView.Ptr(), fullElems, streamPtr); err != nil {
			return nil, nil
		}
		vView := tensor.NewGPUStorageView(c.fp16ScratchV, 0, fullElems)
		if err := kernels.FP16ToF32(lb.vFP16.Ptr(), vView.Ptr(), fullElems, streamPtr); err != nil {
			return nil, nil
		}
		kT, _ := tensor.NewWithStorage(fullShape, kView)
		vT, _ := tensor.NewWithStorage(fullShape, vView)
		return kT, vT
	}

	kView := tensor.NewGPUStorageView(lb.kStorage, 0, fullElems)
	vView := tensor.NewGPUStorageView(lb.vStorage, 0, fullElems)
	kT, _ := tensor.NewWithStorage(fullShape, kView)
	vT, _ := tensor.NewWithStorage(fullShape, vView)
	return kT, vT
}

// MaxSeqLen returns the maximum sequence length (buffer capacity).
func (c *TensorCache[T]) MaxSeqLen() int {
	return c.maxSeqLen
}

// SeqLen returns the current cached sequence length (from layer 0).
func (c *TensorCache[T]) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	return c.layers[0].seqLen
}

// Reset clears sequence lengths to zero. Pre-allocated buffers are kept
// for reuse; only data pointers are logically invalidated. The GPU counters
// are also zeroed so that GPU-side kernels see the reset position.
func (c *TensorCache[T]) Reset() {
	for i := range c.layers {
		c.layers[i].seqLen = 0
	}
	if c.gpuCounter != nil {
		_ = c.gpuCounter.CopyFromHost([]int32{0}, 0)
	}
	if c.kvSeqLenCounter != nil {
		_ = c.kvSeqLenCounter.CopyFromHost([]int32{0}, 0)
	}
}

// Truncate rolls back the cache to the given sequence length.
// Pre-allocated buffers are kept; the data beyond newSeqLen is simply ignored.
// GPU-resident counters (gpuCounter and kvSeqLenCounter) are also reset to
// match newSeqLen so that GPU-side kernels (offset_memcpy, rope_select,
// flash_attention_decode) see the correct position after rollback.
func (c *TensorCache[T]) Truncate(newSeqLen int) {
	for i := range c.layers {
		if c.layers[i].seqLen > newSeqLen {
			c.layers[i].seqLen = newSeqLen
		}
	}
	val := int32(newSeqLen)
	if c.gpuCounter != nil {
		_ = c.gpuCounter.CopyFromHost([]int32{val}, 0)
	}
	if c.kvSeqLenCounter != nil {
		_ = c.kvSeqLenCounter.CopyFromHost([]int32{val}, 0)
	}
}

// GPUCounterPtr returns the device pointer to the GPU-resident int32 position
// counter. Returns nil if no GPU counter is allocated (CPU-only cache).
// Kernels (offset_memcpy, rope_select, increment_counter) use this pointer
// to read/write the current sequence position on the GPU, enabling CUDA graph
// capture of the decode loop.
func (c *TensorCache[T]) GPUCounterPtr() unsafe.Pointer {
	if c.gpuCounter == nil {
		return nil
	}
	return c.gpuCounter.Ptr()
}

// KVSeqLenPtr returns the device pointer to the GPU-resident int32 KV
// sequence length counter. Returns nil if not allocated (CPU-only cache).
// The flash_attention_decode kernel reads this pointer at runtime so the
// KV length is not frozen by CUDA graph capture.
func (c *TensorCache[T]) KVSeqLenPtr() unsafe.Pointer {
	if c.kvSeqLenCounter == nil {
		return nil
	}
	return c.kvSeqLenCounter.Ptr()
}

// SyncCounterFromGPU performs a D2H copy of the GPU counter to update the CPU
// seqLen across all layers. Call this after the decode loop completes to bring
// the CPU-side cursor back in sync with the GPU counter.
func (c *TensorCache[T]) SyncCounterFromGPU() error {
	if c.gpuCounter == nil {
		return fmt.Errorf("tensor_cache: GPU counter not allocated")
	}
	buf := []int32{0}
	if err := c.gpuCounter.CopyTo(buf); err != nil {
		return fmt.Errorf("tensor_cache: sync counter D2H: %w", err)
	}
	pos := int(buf[0])
	for i := range c.layers {
		c.layers[i].seqLen = pos
	}
	return nil
}

// Free releases all pre-allocated GPU buffers. CPU buffers are left to GC.
func (c *TensorCache[T]) Free() {
	for i := range c.layers {
		if c.layers[i].kStorage != nil {
			_ = c.layers[i].kStorage.Free()
			c.layers[i].kStorage = nil
		}
		if c.layers[i].vStorage != nil {
			_ = c.layers[i].vStorage.Free()
			c.layers[i].vStorage = nil
		}
		if c.layers[i].kFP16 != nil {
			_ = c.layers[i].kFP16.Free()
			c.layers[i].kFP16 = nil
		}
		if c.layers[i].vFP16 != nil {
			_ = c.layers[i].vFP16.Free()
			c.layers[i].vFP16 = nil
		}
		c.layers[i].kBuf = nil
		c.layers[i].vBuf = nil
		c.layers[i].seqLen = 0
		c.layers[i].batch = 0
	}
	if c.fp16ScratchK != nil {
		_ = c.fp16ScratchK.Free()
		c.fp16ScratchK = nil
	}
	if c.fp16ScratchV != nil {
		_ = c.fp16ScratchV.Free()
		c.fp16ScratchV = nil
	}
	if c.gpuCounter != nil {
		_ = c.gpuCounter.Free()
		c.gpuCounter = nil
	}
	if c.kvSeqLenCounter != nil {
		_ = c.kvSeqLenCounter.Free()
		c.kvSeqLenCounter = nil
	}
}
