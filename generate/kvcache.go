package generate

import (
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// LayerKV holds the cached key and value tensors for a single attention layer.
type LayerKV[T tensor.Numeric] struct {
	Key   *tensor.TensorNumeric[T]
	Value *tensor.TensorNumeric[T]
}

// layerBuf holds the pre-allocated backing buffer for one layer's KV cache.
type layerBuf[T tensor.Numeric] struct {
	keyBuf []T // pre-allocated [batch * maxSeqLen * dim]
	valBuf []T // pre-allocated [batch * maxSeqLen * dim]
	cursor int // number of sequence positions written
	batch  int // detected on first Update
	dim    int // detected on first Update
}

// KVCache stores key-value tensors for all attention layers during
// autoregressive generation. Buffers are pre-allocated to maxSeqLen on first
// Update, and subsequent Updates copy data at the cursor position with zero
// allocation.
type KVCache[T tensor.Numeric] struct {
	layers    []layerBuf[T]
	maxSeqLen int
}

// NewKVCache creates a KVCache for the specified number of layers and maximum
// sequence length. Backing buffers are lazily allocated on the first Update
// call for each layer (when batch and dim become known).
func NewKVCache[T tensor.Numeric](numLayers, maxSeqLen int) *KVCache[T] {
	return &KVCache[T]{
		layers:    make([]layerBuf[T], numLayers),
		maxSeqLen: maxSeqLen,
	}
}

// NumLayers returns the number of layers in the cache.
func (c *KVCache[T]) NumLayers() int {
	return len(c.layers)
}

// Get returns the cached key-value pair for the given layer as tensors
// covering [0:cursor] on the sequence axis. For batch=1, the returned
// tensors are zero-copy views over the pre-allocated buffer. For batch>1,
// data is compacted into a contiguous slice.
// Returns false if the layer has not been populated yet.
func (c *KVCache[T]) Get(layer int) (*LayerKV[T], bool) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, false
	}
	lb := &c.layers[layer]
	if lb.cursor == 0 {
		return nil, false
	}

	shape := []int{lb.batch, lb.cursor, lb.dim}
	size := lb.batch * lb.cursor * lb.dim

	var keyData, valData []T
	if lb.batch == 1 || lb.cursor == c.maxSeqLen {
		// Contiguous: use sub-slice directly (zero copy).
		keyData = lb.keyBuf[:size]
		valData = lb.valBuf[:size]
	} else {
		// Multi-batch with partial fill: compact the valid data.
		keyData = make([]T, size)
		valData = make([]T, size)
		seqDim := lb.cursor * lb.dim
		for bi := range lb.batch {
			srcOff := bi * c.maxSeqLen * lb.dim
			dstOff := bi * seqDim
			copy(keyData[dstOff:dstOff+seqDim], lb.keyBuf[srcOff:srcOff+seqDim])
			copy(valData[dstOff:dstOff+seqDim], lb.valBuf[srcOff:srcOff+seqDim])
		}
	}

	keyT, err := tensor.New(shape, keyData)
	if err != nil {
		return nil, false
	}
	valT, err := tensor.New(shape, valData)
	if err != nil {
		return nil, false
	}

	return &LayerKV[T]{Key: keyT, Value: valT}, true
}

// Update appends new key and value tensors to the cache for the given layer.
// Tensors are expected to have shape [batch, seq_len, dim]. Data is copied
// into the pre-allocated buffer at the current cursor position. After the
// initial allocation, Update performs zero heap allocations.
func (c *KVCache[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	if layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, len(c.layers))
	}

	shape := newK.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("expected 3D tensor [batch, seq, dim], got %dD", len(shape))
	}

	batch, seqLen, dim := shape[0], shape[1], shape[2]
	lb := &c.layers[layer]

	// Lazy allocation on first Update.
	if lb.keyBuf == nil {
		lb.batch = batch
		lb.dim = dim
		total := batch * c.maxSeqLen * dim
		lb.keyBuf = make([]T, total)
		lb.valBuf = make([]T, total)
	}

	if batch != lb.batch {
		return fmt.Errorf("batch mismatch: cache has %d, got %d", lb.batch, batch)
	}
	if dim != lb.dim {
		return fmt.Errorf("dim mismatch: cache has %d, got %d", lb.dim, dim)
	}
	if lb.cursor+seqLen > c.maxSeqLen {
		return fmt.Errorf("cache overflow: cursor=%d + seq=%d > maxSeqLen=%d", lb.cursor, seqLen, c.maxSeqLen)
	}

	// Copy new data into the buffer at the cursor position.
	// Layout: [batch, maxSeqLen, dim] — copy seqLen*dim elements per batch
	// at offset cursor*dim within each batch's maxSeqLen*dim region.
	kData := newK.Data()
	vData := newV.Data()
	for bi := range batch {
		srcOff := bi * seqLen * dim
		dstOff := bi*c.maxSeqLen*dim + lb.cursor*dim
		copy(lb.keyBuf[dstOff:dstOff+seqLen*dim], kData[srcOff:srcOff+seqLen*dim])
		copy(lb.valBuf[dstOff:dstOff+seqLen*dim], vData[srcOff:srcOff+seqLen*dim])
	}

	lb.cursor += seqLen
	return nil
}

// SeqLen returns the current cached sequence length.
// Returns 0 if the cache is empty.
func (c *KVCache[T]) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	return c.layers[0].cursor
}

// Reset clears all cached data and resets cursors to zero.
// The pre-allocated buffers are retained for reuse.
func (c *KVCache[T]) Reset() {
	for i := range c.layers {
		c.layers[i].cursor = 0
	}
}

// Truncate rolls back the cache to the given sequence length.
// If newSeqLen >= current SeqLen, this is a no-op.
func (c *KVCache[T]) Truncate(newSeqLen int) {
	for i := range c.layers {
		if c.layers[i].cursor > newSeqLen {
			c.layers[i].cursor = newSeqLen
		}
	}
}
