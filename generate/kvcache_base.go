package generate

import (
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// QuantStorage is the strategy interface for quantized KV cache storage.
// Each implementation handles allocation, encoding (float32 -> compressed),
// and decoding (compressed -> float32) for a specific quantization format.
type QuantStorage interface {
	// Decode returns all elements as float32.
	Decode() []float32
	// EncodeRegion writes float32 values into the storage at the given offset
	// within each batch's region. Parameters:
	//   batch, seqLen, dim: dimensions of the new data
	//   maxSeqLen: total sequence capacity per batch
	//   cursor: current write position on the sequence axis
	//   src: source float32 data [batch * seqLen * dim]
	EncodeRegion(batch, seqLen, dim, maxSeqLen, cursor int, src []float32)
}

// quantLayerBuf holds the pre-allocated backing buffers for one layer's
// quantized KV cache. The QuantStorage handles format-specific operations.
type quantLayerBuf struct {
	keyBuf QuantStorage
	valBuf QuantStorage
	cursor int
	batch  int
	dim    int
}

// quantKVCache is the shared base for all quantized KV cache implementations
// (FP16, FP8, Q4, Q3). It handles buffer management, cursor tracking, and
// the Get/Update/SeqLen/Reset/Truncate logic. Format-specific behavior is
// delegated to the QuantStorage strategy via the allocFn callback.
type quantKVCache struct {
	layers    []quantLayerBuf
	maxSeqLen int
	// allocFn creates a new QuantStorage for the given total number of elements.
	allocFn func(total int) QuantStorage
}

// newQuantKVCache creates a quantKVCache with the given layer count, max
// sequence length, and storage allocation function.
func newQuantKVCache(numLayers, maxSeqLen int, allocFn func(int) QuantStorage) *quantKVCache {
	return &quantKVCache{
		layers:    make([]quantLayerBuf, numLayers),
		maxSeqLen: maxSeqLen,
		allocFn:   allocFn,
	}
}

// NumLayers returns the number of layers in the cache.
func (c *quantKVCache) NumLayers() int {
	return len(c.layers)
}

// Get returns the cached key-value pair for the given layer as float32 tensors
// covering [0:cursor] on the sequence axis. Compressed data is decoded to
// float32 on the fly. Returns false if the layer has not been populated yet.
func (c *quantKVCache) Get(layer int) (*LayerKV[float32], bool) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, false
	}
	lb := &c.layers[layer]
	if lb.cursor == 0 {
		return nil, false
	}

	shape := []int{lb.batch, lb.cursor, lb.dim}
	size := lb.batch * lb.cursor * lb.dim

	allKey := lb.keyBuf.Decode()
	allVal := lb.valBuf.Decode()

	var keyData, valData []float32
	if lb.batch == 1 || lb.cursor == c.maxSeqLen {
		keyData = allKey[:size]
		valData = allVal[:size]
	} else {
		keyData = make([]float32, size)
		valData = make([]float32, size)
		seqDim := lb.cursor * lb.dim
		for bi := range lb.batch {
			srcOff := bi * c.maxSeqLen * lb.dim
			dstOff := bi * seqDim
			copy(keyData[dstOff:dstOff+seqDim], allKey[srcOff:srcOff+seqDim])
			copy(valData[dstOff:dstOff+seqDim], allVal[srcOff:srcOff+seqDim])
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

	return &LayerKV[float32]{Key: keyT, Value: valT}, true
}

// Update appends new key and value float32 tensors to the quantized cache for
// the given layer. Tensors are expected to have shape [batch, seq_len, dim].
func (c *quantKVCache) Update(layer int, newK, newV *tensor.TensorNumeric[float32]) error {
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
		lb.keyBuf = c.allocFn(total)
		lb.valBuf = c.allocFn(total)
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

	kData := newK.Data()
	vData := newV.Data()

	lb.keyBuf.EncodeRegion(batch, seqLen, dim, c.maxSeqLen, lb.cursor, kData)
	lb.valBuf.EncodeRegion(batch, seqLen, dim, c.maxSeqLen, lb.cursor, vData)

	lb.cursor += seqLen
	return nil
}

// SeqLen returns the current cached sequence length. Returns 0 if empty.
func (c *quantKVCache) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	return c.layers[0].cursor
}

// Reset clears all cached data and resets cursors to zero.
// The pre-allocated buffers are retained for reuse.
func (c *quantKVCache) Reset() {
	for i := range c.layers {
		c.layers[i].cursor = 0
	}
}

// Truncate rolls back the cache to the given sequence length.
// If newSeqLen >= current SeqLen, this is a no-op.
func (c *quantKVCache) Truncate(newSeqLen int) {
	for i := range c.layers {
		if c.layers[i].cursor > newSeqLen {
			c.layers[i].cursor = newSeqLen
		}
	}
}
