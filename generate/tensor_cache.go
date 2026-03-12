package generate

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// tensorLayerBuf holds pre-allocated K/V buffers for a single layer.
// For GPU-resident tensors, buffers are backed by GPUStorage with direct D2D
// memcpy appends. For CPU tensors, flat Go slices are used with copy.
type tensorLayerBuf[T tensor.Numeric] struct {
	// GPU path: pre-allocated persistent GPU memory.
	kStorage *tensor.GPUStorage[T]
	vStorage *tensor.GPUStorage[T]
	// CPU path: pre-allocated flat buffers.
	kBuf []T
	vBuf []T

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
}

// NewTensorCache creates a TensorCache backed by the given engine.
// numLayers should match the model's transformer layer count.
// maxSeqLen limits the total cached sequence length.
func NewTensorCache[T tensor.Numeric](engine compute.Engine[T], numLayers, maxSeqLen int) *TensorCache[T] {
	return &TensorCache[T]{
		engine:    engine,
		layers:    make([]tensorLayerBuf[T], numLayers),
		maxSeqLen: maxSeqLen,
	}
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
		if isGPU {
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

	// Append data at current offset.
	offset := lb.seqLen * dim * batch
	numElems := seqLen * dim * batch

	if lb.isGPU {
		if err := appendGPU(lb.kStorage, offset, numElems, newK); err != nil {
			return fmt.Errorf("append K layer %d: %w", layer, err)
		}
		if err := appendGPU(lb.vStorage, offset, numElems, newV); err != nil {
			return fmt.Errorf("append V layer %d: %w", layer, err)
		}
	} else {
		copy(lb.kBuf[offset:offset+numElems], newK.Data())
		copy(lb.vBuf[offset:offset+numElems], newV.Data())
	}

	lb.seqLen += seqLen
	return nil
}

// appendGPU copies tensor data into the pre-allocated GPU buffer at the given
// element offset, using D2D memcpy for GPU sources or H2D for CPU sources.
func appendGPU[T tensor.Numeric](dst *tensor.GPUStorage[T], offset, numElems int, src *tensor.TensorNumeric[T]) error {
	if gs, ok := src.GetStorage().(*tensor.GPUStorage[T]); ok {
		return dst.CopyFromDevice(gs, offset, 0, numElems)
	}
	return dst.CopyFromHost(src.Data(), offset)
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

// SeqLen returns the current cached sequence length (from layer 0).
func (c *TensorCache[T]) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	return c.layers[0].seqLen
}

// Reset clears sequence lengths to zero. Pre-allocated buffers are kept
// for reuse; only data pointers are logically invalidated.
func (c *TensorCache[T]) Reset() {
	for i := range c.layers {
		c.layers[i].seqLen = 0
	}
}

// Truncate rolls back the cache to the given sequence length.
// Pre-allocated buffers are kept; the data beyond newSeqLen is simply ignored.
func (c *TensorCache[T]) Truncate(newSeqLen int) {
	for i := range c.layers {
		if c.layers[i].seqLen > newSeqLen {
			c.layers[i].seqLen = newSeqLen
		}
	}
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
		c.layers[i].kBuf = nil
		c.layers[i].vBuf = nil
		c.layers[i].seqLen = 0
		c.layers[i].batch = 0
	}
}
