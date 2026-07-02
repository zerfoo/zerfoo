package generate

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
)

// GPUAllocator abstracts GPU memory operations so that GPUKVCache can be
// tested without a real GPU device. Production code passes a thin wrapper
// around gpuapi.Runtime; tests supply a mock.
type GPUAllocator interface {
	// Alloc allocates size bytes of device memory and returns a device pointer.
	Alloc(size int) (unsafe.Pointer, error)
	// Free releases device memory previously returned by Alloc.
	Free(ptr unsafe.Pointer) error
	// Memcpy copies size bytes between host and device memory.
	// kind follows the gpuapi convention: 0 = HostToDevice, 1 = DeviceToHost.
	Memcpy(dst, src unsafe.Pointer, size int, kind int) error
}

// gpuMemcpyHostToDevice is the Memcpy direction constant matching gpuapi.MemcpyKind.
const gpuMemcpyHostToDevice = 0

// gpuLayerBuf holds a pair of device pointers for one layer's K and V buffers.
type gpuLayerBuf struct {
	kPtr unsafe.Pointer
	vPtr unsafe.Pointer
}

// GPUKVCache manages GPU-resident key/value buffers for all attention layers
// during megakernel inference. Memory is allocated once at construction and
// reused across generation steps.
type GPUKVCache struct {
	alloc      GPUAllocator
	layers     []gpuLayerBuf
	numLayers  int
	maxSeqLen  int
	headDim    int
	numHeads   int
	seqLen     int
	bufBytes   int            // per-layer buffer size in bytes (maxSeqLen * numHeads * headDim * 4)
	gpuCounter unsafe.Pointer // GPU-allocated int32 position counter for CUDA graph capture
}

// NewGPUKVCache allocates GPU buffers for numLayers attention layers.
// Each layer gets two buffers (K and V) of size maxSeqLen * numHeads * headDim
// float32 elements.
func NewGPUKVCache(alloc GPUAllocator, numLayers, maxSeqLen, numHeads, headDim int) (*GPUKVCache, error) {
	if alloc == nil {
		return nil, fmt.Errorf("gpu_kv_cache: allocator must not be nil")
	}
	if numLayers <= 0 {
		return nil, fmt.Errorf("gpu_kv_cache: numLayers must be > 0, got %d", numLayers)
	}
	if maxSeqLen <= 0 {
		return nil, fmt.Errorf("gpu_kv_cache: maxSeqLen must be > 0, got %d", maxSeqLen)
	}
	if numHeads <= 0 {
		return nil, fmt.Errorf("gpu_kv_cache: numHeads must be > 0, got %d", numHeads)
	}
	if headDim <= 0 {
		return nil, fmt.Errorf("gpu_kv_cache: headDim must be > 0, got %d", headDim)
	}

	bufBytes := maxSeqLen * numHeads * headDim * 4 // float32 = 4 bytes

	layers := make([]gpuLayerBuf, numLayers)
	for i := range numLayers {
		kPtr, err := alloc.Alloc(bufBytes)
		if err != nil {
			// Free any already-allocated buffers.
			for j := range i {
				_ = alloc.Free(layers[j].kPtr)
				_ = alloc.Free(layers[j].vPtr)
			}
			return nil, fmt.Errorf("gpu_kv_cache: alloc K buffer for layer %d: %w", i, err)
		}
		vPtr, err := alloc.Alloc(bufBytes)
		if err != nil {
			_ = alloc.Free(kPtr)
			for j := range i {
				_ = alloc.Free(layers[j].kPtr)
				_ = alloc.Free(layers[j].vPtr)
			}
			return nil, fmt.Errorf("gpu_kv_cache: alloc V buffer for layer %d: %w", i, err)
		}
		layers[i] = gpuLayerBuf{kPtr: kPtr, vPtr: vPtr}
	}

	// Allocate a GPU-resident int32 counter for position tracking.
	// This counter is used by CUDA graph-captured kernels (offset_memcpy,
	// rope_select, increment_counter) instead of CPU-side seqLen.
	counterPtr, err := alloc.Alloc(4) // sizeof(int32)
	if err != nil {
		for i := range numLayers {
			_ = alloc.Free(layers[i].kPtr)
			_ = alloc.Free(layers[i].vPtr)
		}
		return nil, fmt.Errorf("gpu_kv_cache: alloc GPU counter: %w", err)
	}
	// Zero-initialize the counter via H2D copy.
	zero := int32(0)
	if err := alloc.Memcpy(counterPtr, unsafe.Pointer(&zero), 4, gpuMemcpyHostToDevice); err != nil {
		_ = alloc.Free(counterPtr)
		for i := range numLayers {
			_ = alloc.Free(layers[i].kPtr)
			_ = alloc.Free(layers[i].vPtr)
		}
		return nil, fmt.Errorf("gpu_kv_cache: zero GPU counter: %w", err)
	}

	return &GPUKVCache{
		alloc:      alloc,
		layers:     layers,
		numLayers:  numLayers,
		maxSeqLen:  maxSeqLen,
		headDim:    headDim,
		numHeads:   numHeads,
		bufBytes:   bufBytes,
		gpuCounter: counterPtr,
	}, nil
}

// Append copies new K/V float32 data to the correct position in the GPU buffer
// for the given layer. k and v must each have length numHeads * headDim (one
// token's worth of data). seqPos is the sequence position to write at; it must
// equal the current seqLen (enforcing sequential append).
func (c *GPUKVCache) Append(layerIdx int, k, v []float32, seqPos int) error {
	if layerIdx < 0 || layerIdx >= c.numLayers {
		return fmt.Errorf("gpu_kv_cache: layer %d out of range [0, %d)", layerIdx, c.numLayers)
	}
	if seqPos != c.seqLen {
		return fmt.Errorf("gpu_kv_cache: seqPos %d != current seqLen %d", seqPos, c.seqLen)
	}
	if seqPos >= c.maxSeqLen {
		return fmt.Errorf("gpu_kv_cache: seqPos %d >= maxSeqLen %d", seqPos, c.maxSeqLen)
	}

	tokenElems := c.numHeads * c.headDim
	if len(k) != tokenElems {
		return fmt.Errorf("gpu_kv_cache: k length %d != expected %d", len(k), tokenElems)
	}
	if len(v) != tokenElems {
		return fmt.Errorf("gpu_kv_cache: v length %d != expected %d", len(v), tokenElems)
	}

	byteOffset := seqPos * tokenElems * 4
	copyBytes := tokenElems * 4

	lb := &c.layers[layerIdx]

	kDst := unsafe.Add(lb.kPtr, byteOffset)
	if err := c.alloc.Memcpy(kDst, unsafe.Pointer(&k[0]), copyBytes, gpuMemcpyHostToDevice); err != nil {
		return fmt.Errorf("gpu_kv_cache: memcpy K layer %d pos %d: %w", layerIdx, seqPos, err)
	}

	vDst := unsafe.Add(lb.vPtr, byteOffset)
	if err := c.alloc.Memcpy(vDst, unsafe.Pointer(&v[0]), copyBytes, gpuMemcpyHostToDevice); err != nil {
		return fmt.Errorf("gpu_kv_cache: memcpy V layer %d pos %d: %w", layerIdx, seqPos, err)
	}

	// Advance seqLen only after the last layer's append for this position.
	if layerIdx == c.numLayers-1 {
		c.seqLen++
	}

	return nil
}

// gpuMemcpyDeviceToHost is the Memcpy direction constant matching gpuapi.MemcpyKind.
const gpuMemcpyDeviceToHost = 1

// AppendGPU copies one token's K/V data from GPU-resident src pointers into the
// KV cache using the offset_memcpy kernel. The kernel reads gpuCounter on the
// GPU to compute the write offset, eliminating any D2H copy per token.
// After writing K and V for the last layer, it increments the GPU counter via
// the increment_counter kernel and advances the CPU seqLen for compatibility.
//
// kSrc and vSrc must each point to numHeads*headDim float32 values on the GPU.
// stream is the CUDA stream for async execution.
func (c *GPUKVCache) AppendGPU(layerIdx int, kSrc, vSrc unsafe.Pointer, stream unsafe.Pointer) error {
	if layerIdx < 0 || layerIdx >= c.numLayers {
		return fmt.Errorf("gpu_kv_cache: layer %d out of range [0, %d)", layerIdx, c.numLayers)
	}
	if c.seqLen >= c.maxSeqLen {
		return fmt.Errorf("gpu_kv_cache: seqLen %d >= maxSeqLen %d", c.seqLen, c.maxSeqLen)
	}

	dim := c.numHeads * c.headDim
	lb := &c.layers[layerIdx]

	if err := kernels.OffsetMemcpy(lb.kPtr, kSrc, c.gpuCounter, dim, c.maxSeqLen, stream); err != nil {
		return fmt.Errorf("gpu_kv_cache: offset_memcpy K layer %d: %w", layerIdx, err)
	}
	if err := kernels.OffsetMemcpy(lb.vPtr, vSrc, c.gpuCounter, dim, c.maxSeqLen, stream); err != nil {
		return fmt.Errorf("gpu_kv_cache: offset_memcpy V layer %d: %w", layerIdx, err)
	}

	// Advance GPU counter and CPU seqLen after the last layer's append.
	if layerIdx == c.numLayers-1 {
		if err := kernels.IncrementCounter(c.gpuCounter, 1, stream); err != nil {
			return fmt.Errorf("gpu_kv_cache: increment_counter: %w", err)
		}
		c.seqLen++
	}

	return nil
}

// SyncCounterFromGPU performs a D2H copy of the GPU counter to update the CPU
// seqLen. Call this after the decode loop completes, not per token.
func (c *GPUKVCache) SyncCounterFromGPU() error {
	if c.gpuCounter == nil {
		return fmt.Errorf("gpu_kv_cache: GPU counter not allocated")
	}
	var val int32
	if err := c.alloc.Memcpy(unsafe.Pointer(&val), c.gpuCounter, 4, gpuMemcpyDeviceToHost); err != nil {
		return fmt.Errorf("gpu_kv_cache: sync counter D2H: %w", err)
	}
	c.seqLen = int(val)
	return nil
}

// Pointers returns the device pointers for the K and V buffers of the given
// layer, along with the current sequence length. The megakernel reads from
// these pointers directly.
func (c *GPUKVCache) Pointers(layerIdx int) (kPtr, vPtr unsafe.Pointer, seqLen int) {
	if layerIdx < 0 || layerIdx >= c.numLayers {
		return nil, nil, 0
	}
	lb := &c.layers[layerIdx]
	return lb.kPtr, lb.vPtr, c.seqLen
}

// SeqLen returns the current cached sequence length.
func (c *GPUKVCache) SeqLen() int {
	return c.seqLen
}

// GPUCounterPtr returns the device pointer to the GPU-resident int32 position
// counter. Kernels (offset_memcpy, rope_select, increment_counter) use this
// pointer to read/write the current sequence position on the GPU, enabling
// CUDA graph capture of the decode loop.
func (c *GPUKVCache) GPUCounterPtr() unsafe.Pointer {
	return c.gpuCounter
}

// Reset resets the sequence position to zero without freeing GPU memory.
// Buffers are reused for the next generation. The GPU counter is also
// zeroed so that GPU-side kernels see the reset position.
func (c *GPUKVCache) Reset() {
	c.seqLen = 0
	if c.gpuCounter != nil {
		zero := int32(0)
		// Best-effort: GPU counter reset failure is non-fatal since the CPU
		// seqLen is the authoritative source during non-graph execution.
		_ = c.alloc.Memcpy(c.gpuCounter, unsafe.Pointer(&zero), 4, gpuMemcpyHostToDevice)
	}
}

// DevicePointerArrays returns GPU-resident arrays of float* pointers for K
// and V buffers across all layers. These can be passed directly to the
// megakernel. The arrays are allocated once and cached.
func (c *GPUKVCache) DevicePointerArrays() (kPtrs, vPtrs unsafe.Pointer, err error) {
	kHostPtrs := make([]uintptr, c.numLayers)
	vHostPtrs := make([]uintptr, c.numLayers)
	for i := range c.numLayers {
		kHostPtrs[i] = uintptr(c.layers[i].kPtr)
		vHostPtrs[i] = uintptr(c.layers[i].vPtr)
	}
	ptrArrayBytes := c.numLayers * 8 // 8 bytes per pointer on 64-bit

	kPtrs, err = c.alloc.Alloc(ptrArrayBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("gpu_kv_cache: alloc K ptr array: %w", err)
	}
	if err := c.alloc.Memcpy(kPtrs, unsafe.Pointer(&kHostPtrs[0]), ptrArrayBytes, gpuMemcpyHostToDevice); err != nil {
		_ = c.alloc.Free(kPtrs)
		return nil, nil, fmt.Errorf("gpu_kv_cache: upload K ptr array: %w", err)
	}

	vPtrs, err = c.alloc.Alloc(ptrArrayBytes)
	if err != nil {
		_ = c.alloc.Free(kPtrs)
		return nil, nil, fmt.Errorf("gpu_kv_cache: alloc V ptr array: %w", err)
	}
	if err := c.alloc.Memcpy(vPtrs, unsafe.Pointer(&vHostPtrs[0]), ptrArrayBytes, gpuMemcpyHostToDevice); err != nil {
		_ = c.alloc.Free(kPtrs)
		_ = c.alloc.Free(vPtrs)
		return nil, nil, fmt.Errorf("gpu_kv_cache: upload V ptr array: %w", err)
	}

	return kPtrs, vPtrs, nil
}

// Close frees all GPU memory held by the cache. The cache must not be used
// after Close is called.
func (c *GPUKVCache) Close() error {
	var firstErr error
	for i := range c.layers {
		if c.layers[i].kPtr != nil {
			if err := c.alloc.Free(c.layers[i].kPtr); err != nil && firstErr == nil {
				firstErr = err
			}
			c.layers[i].kPtr = nil
		}
		if c.layers[i].vPtr != nil {
			if err := c.alloc.Free(c.layers[i].vPtr); err != nil && firstErr == nil {
				firstErr = err
			}
			c.layers[i].vPtr = nil
		}
	}
	if c.gpuCounter != nil {
		if err := c.alloc.Free(c.gpuCounter); err != nil && firstErr == nil {
			firstErr = err
		}
		c.gpuCounter = nil
	}
	return firstErr
}
