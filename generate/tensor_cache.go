package generate

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// freeGPUStorage releases the GPU memory backing a tensor immediately,
// rather than waiting for GC finalization which may lag behind allocation.
func freeGPUStorage[T tensor.Numeric](t *tensor.TensorNumeric[T]) {
	if t == nil {
		return
	}
	if gs, ok := t.GetStorage().(*tensor.GPUStorage[T]); ok {
		_ = gs.Free()
	}
}

// tensorLayerBuf holds GPU-resident cached K/V tensors for a single layer.
type tensorLayerBuf[T tensor.Numeric] struct {
	cachedK *tensor.TensorNumeric[T]
	cachedV *tensor.TensorNumeric[T]
	seqLen  int
}

// TensorCache is a GPU-resident KV cache that keeps tensors on-device.
// Instead of copying data to CPU via .Data(), it uses engine.Concat to
// append new K/V tensors along the sequence dimension (axis 1).
// This avoids 52 synchronous GPU→CPU memcpys per token (26 layers × 2).
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
// a layer the tensors are stored directly; subsequent calls use engine.Concat
// to append along axis 1 without copying data off-device.
func (c *TensorCache[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	if layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, len(c.layers))
	}
	shape := newK.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("expected 3D tensor [batch, seq, dim], got %dD", len(shape))
	}
	seqLen := shape[1]
	lb := &c.layers[layer]

	if lb.seqLen+seqLen > c.maxSeqLen {
		return fmt.Errorf("cache overflow: cursor=%d + seq=%d > maxSeqLen=%d", lb.seqLen, seqLen, c.maxSeqLen)
	}

	ctx := context.Background()
	if lb.cachedK == nil {
		lb.cachedK = newK
		lb.cachedV = newV
	} else {
		oldK := lb.cachedK
		oldV := lb.cachedV
		var err error
		lb.cachedK, err = c.engine.Concat(ctx, []*tensor.TensorNumeric[T]{oldK, newK}, 1)
		if err != nil {
			return fmt.Errorf("concat K layer %d: %w", layer, err)
		}
		lb.cachedV, err = c.engine.Concat(ctx, []*tensor.TensorNumeric[T]{oldV, newV}, 1)
		if err != nil {
			return fmt.Errorf("concat V layer %d: %w", layer, err)
		}
		// Free old GPU buffers immediately to prevent memory accumulation.
		freeGPUStorage(oldK)
		freeGPUStorage(oldV)
	}
	lb.seqLen += seqLen
	return nil
}

// Get returns the cached key-value pair for the given layer.
// Returns false if the layer index is out of range or the layer is empty.
func (c *TensorCache[T]) Get(layer int) (*LayerKV[T], bool) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, false
	}
	lb := &c.layers[layer]
	if lb.seqLen == 0 || lb.cachedK == nil {
		return nil, false
	}
	return &LayerKV[T]{Key: lb.cachedK, Value: lb.cachedV}, true
}

// SeqLen returns the current cached sequence length (from layer 0).
func (c *TensorCache[T]) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	return c.layers[0].seqLen
}

// Reset clears all cached tensors and resets sequence lengths to zero.
func (c *TensorCache[T]) Reset() {
	for i := range c.layers {
		c.layers[i].cachedK = nil
		c.layers[i].cachedV = nil
		c.layers[i].seqLen = 0
	}
}

// Truncate rolls back the cache to the given sequence length.
// Because GPU tensors cannot be sliced in-place, layers that exceed
// newSeqLen are fully cleared. A newSeqLen of 0 is equivalent to Reset.
func (c *TensorCache[T]) Truncate(newSeqLen int) {
	for i := range c.layers {
		if c.layers[i].seqLen > newSeqLen {
			c.layers[i].seqLen = newSeqLen
			c.layers[i].cachedK = nil
			c.layers[i].cachedV = nil
		}
	}
}
