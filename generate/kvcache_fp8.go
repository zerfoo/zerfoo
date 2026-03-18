package generate

import (
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// fp8LayerBuf holds the pre-allocated FP8 backing buffer for one layer's KV cache.
// Storing in FP8 (E4M3 with per-tensor absmax scaling) quarters the memory
// bandwidth vs float32 (1 byte vs 4 bytes per element).
type fp8LayerBuf struct {
	keyBuf *tensor.FP8E4M3Storage // pre-allocated [batch * maxSeqLen * dim] in FP8
	valBuf *tensor.FP8E4M3Storage // pre-allocated [batch * maxSeqLen * dim] in FP8
	cursor int                    // number of sequence positions written
	batch  int                    // detected on first Update
	dim    int                    // detected on first Update
}

// KVCacheFP8 stores key-value tensors for all attention layers using FP8 E4M3
// storage, reducing memory by ~4x compared to float32 and ~2x compared to FP16.
// On Update, float32 values are quantized to FP8; on Get, FP8 values are
// dequantized back to float32.
//
// FP8 E4M3 has lower precision than FP16 (~1.5 decimal digits vs ~3) but the
// perplexity impact is typically within 0.5 for attention KV values.
type KVCacheFP8 struct {
	layers    []fp8LayerBuf
	maxSeqLen int
}

// NewKVCacheFP8 creates a KVCacheFP8 for the specified number of layers and
// maximum sequence length. FP8 backing buffers are lazily allocated on the
// first Update call for each layer.
func NewKVCacheFP8(numLayers, maxSeqLen int) *KVCacheFP8 {
	return &KVCacheFP8{
		layers:    make([]fp8LayerBuf, numLayers),
		maxSeqLen: maxSeqLen,
	}
}

// NumLayers returns the number of layers in the cache.
func (c *KVCacheFP8) NumLayers() int {
	return len(c.layers)
}

// Get returns the cached key-value pair for the given layer as float32 tensors
// covering [0:cursor] on the sequence axis. FP8 data is dequantized to float32
// on the fly. Returns false if the layer has not been populated yet.
func (c *KVCacheFP8) Get(layer int) (*LayerKV[float32], bool) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, false
	}
	lb := &c.layers[layer]
	if lb.cursor == 0 {
		return nil, false
	}

	shape := []int{lb.batch, lb.cursor, lb.dim}
	size := lb.batch * lb.cursor * lb.dim

	var keyData, valData []float32
	if lb.batch == 1 || lb.cursor == c.maxSeqLen {
		// Contiguous: decode the sub-region directly.
		allKey := lb.keyBuf.Slice()
		allVal := lb.valBuf.Slice()
		keyData = allKey[:size]
		valData = allVal[:size]
	} else {
		// Multi-batch with partial fill: compact the valid data.
		allKey := lb.keyBuf.Slice()
		allVal := lb.valBuf.Slice()
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

// Update appends new key and value float32 tensors to the FP8 cache for the
// given layer. Tensors are expected to have shape [batch, seq_len, dim]. Data
// is converted from float32 to FP8 and copied into the pre-allocated buffer
// at the current cursor position.
func (c *KVCacheFP8) Update(layer int, newK, newV *tensor.TensorNumeric[float32]) error {
	if layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, len(c.layers))
	}

	shape := newK.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("expected 3D tensor [batch, seq, dim], got %dD", len(shape))
	}

	batch, seqLen, dim := shape[0], shape[1], shape[2]
	lb := &c.layers[layer]

	// Lazy allocation on first Update — pre-allocate the full FP8 buffer.
	if lb.keyBuf == nil {
		lb.batch = batch
		lb.dim = dim
		total := batch * c.maxSeqLen * dim
		zeros := make([]float32, total)
		lb.keyBuf = tensor.NewFP8E4M3Storage(zeros)
		lb.valBuf = tensor.NewFP8E4M3Storage(zeros)
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

	// Encode float32 → FP8 and write into the buffer at cursor position.
	// FP8E4M3Storage uses per-tensor absmax scaling, so we decode the full
	// buffer, patch the target region, and re-encode. This is O(bufferSize)
	// per update — same strategy as the FP16 cache.
	fp8EncodeInto(lb.keyBuf, batch, seqLen, dim, c.maxSeqLen, lb.cursor, kData)
	fp8EncodeInto(lb.valBuf, batch, seqLen, dim, c.maxSeqLen, lb.cursor, vData)

	lb.cursor += seqLen
	return nil
}

// SeqLen returns the current cached sequence length. Returns 0 if the cache is empty.
func (c *KVCacheFP8) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	return c.layers[0].cursor
}

// Reset clears all cached data and resets cursors to zero.
// The pre-allocated FP8 buffers are retained for reuse.
func (c *KVCacheFP8) Reset() {
	for i := range c.layers {
		c.layers[i].cursor = 0
	}
}

// Truncate rolls back the cache to the given sequence length.
// If newSeqLen >= current SeqLen, this is a no-op.
func (c *KVCacheFP8) Truncate(newSeqLen int) {
	for i := range c.layers {
		if c.layers[i].cursor > newSeqLen {
			c.layers[i].cursor = newSeqLen
		}
	}
}

// fp8EncodeInto decodes the FP8 buffer to float32, patches the target region
// with new data, and re-encodes. Buffer layout: [batch, maxSeqLen, dim].
func fp8EncodeInto(s *tensor.FP8E4M3Storage, batch, seqLen, dim, maxSeqLen, cursor int, src []float32) {
	all := s.Slice()
	for bi := range batch {
		srcOff := bi * seqLen * dim
		dstOff := bi*maxSeqLen*dim + cursor*dim
		copy(all[dstOff:dstOff+seqLen*dim], src[srcOff:srcOff+seqLen*dim])
	}
	s.Set(all)
}
