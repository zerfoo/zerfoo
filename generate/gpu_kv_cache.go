package generate

import (
	"fmt"
	"unsafe"
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
	alloc     GPUAllocator
	layers    []gpuLayerBuf
	numLayers int
	maxSeqLen int
	headDim   int
	numHeads  int
	seqLen    int
	bufBytes  int // per-layer buffer size in bytes (maxSeqLen * numHeads * headDim * 4)
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

	return &GPUKVCache{
		alloc:     alloc,
		layers:    layers,
		numLayers: numLayers,
		maxSeqLen: maxSeqLen,
		headDim:   headDim,
		numHeads:  numHeads,
		bufBytes:  bufBytes,
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

// Reset resets the sequence position to zero without freeing GPU memory.
// Buffers are reused for the next generation.
func (c *GPUKVCache) Reset() {
	c.seqLen = 0
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
	return firstErr
}
