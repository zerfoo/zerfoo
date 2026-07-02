package generate

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// compressedLayerBuf holds per-layer state for a CompressedKVCache.
type compressedLayerBuf[T tensor.Numeric] struct {
	// compressedKeys stores mean-pooled chunks: [batch, numChunks, dim].
	compressedKeys []T
	compressedVals []T
	numChunks      int

	// recentKeys stores the current (incomplete) chunk: [batch, chunkCursor, dim].
	recentKeys []T
	recentVals []T
	chunkCursor int // positions written in the current chunk

	batch int
	dim   int
}

// CompressedKVCache stores key-value tensors with chunk-wise mean pooling
// compression. When a chunk of chunkSize tokens is full, it is compressed
// into a single vector by averaging (ReduceMean over the sequence axis).
// Recent tokens within the current chunk are stored uncompressed.
// Get() returns the compressed chunks concatenated with recent tokens.
type CompressedKVCache[T tensor.Numeric] struct {
	engine    compute.Engine[T]
	layers    []compressedLayerBuf[T]
	chunkSize int
}

// NewCompressedKVCache creates a CompressedKVCache.
// layers: number of attention layers.
// heads: number of attention heads (reserved for future use).
// dim: feature dimension per token.
// chunkSize: number of tokens per chunk before compression.
func NewCompressedKVCache[T tensor.Numeric](engine compute.Engine[T], layers, heads, dim, chunkSize int) *CompressedKVCache[T] {
	return &CompressedKVCache[T]{
		engine:    engine,
		layers:    make([]compressedLayerBuf[T], layers),
		chunkSize: chunkSize,
	}
}

// NumLayers returns the number of layers in the cache.
func (c *CompressedKVCache[T]) NumLayers() int {
	return len(c.layers)
}

// SeqLen returns the total number of tokens stored (compressed + recent).
// Compressed chunks each represent chunkSize tokens; recent tokens are counted directly.
func (c *CompressedKVCache[T]) SeqLen() int {
	if len(c.layers) == 0 {
		return 0
	}
	lb := &c.layers[0]
	return lb.numChunks*c.chunkSize + lb.chunkCursor
}

// Update appends new key and value tensors for the given layer.
// Tensors must have shape [batch, seq_len, dim]. When the current chunk
// fills up, it is compressed via mean pooling and moved to the compressed store.
func (c *CompressedKVCache[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	if layer < 0 || layer >= len(c.layers) {
		return fmt.Errorf("layer index %d out of range [0, %d)", layer, len(c.layers))
	}

	shape := newK.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("expected 3D tensor [batch, seq, dim], got %dD", len(shape))
	}

	batch, seqLen, dim := shape[0], shape[1], shape[2]
	lb := &c.layers[layer]

	// Lazy init on first update.
	if lb.recentKeys == nil {
		lb.batch = batch
		lb.dim = dim
		lb.recentKeys = make([]T, batch*c.chunkSize*dim)
		lb.recentVals = make([]T, batch*c.chunkSize*dim)
	}

	if batch != lb.batch {
		return fmt.Errorf("batch mismatch: cache has %d, got %d", lb.batch, batch)
	}
	if dim != lb.dim {
		return fmt.Errorf("dim mismatch: cache has %d, got %d", lb.dim, dim)
	}

	kData := newK.Data()
	vData := newV.Data()

	// Append token by token, compressing when a chunk fills.
	for pos := range seqLen {
		// Copy one token [batch, 1, dim] into recent buffer at chunkCursor.
		for bi := range batch {
			srcOff := bi*seqLen*dim + pos*dim
			dstOff := bi*c.chunkSize*dim + lb.chunkCursor*dim
			copy(lb.recentKeys[dstOff:dstOff+dim], kData[srcOff:srcOff+dim])
			copy(lb.recentVals[dstOff:dstOff+dim], vData[srcOff:srcOff+dim])
		}
		lb.chunkCursor++

		// When chunk is full, compress it.
		if lb.chunkCursor == c.chunkSize {
			if err := c.compressChunk(layer); err != nil {
				return fmt.Errorf("compressing chunk: %w", err)
			}
		}
	}

	return nil
}

// compressChunk mean-pools the current chunk [batch, chunkSize, dim] into
// [batch, 1, dim] and appends it to the compressed store.
func (c *CompressedKVCache[T]) compressChunk(layer int) error {
	lb := &c.layers[layer]
	batch, dim, chunkSize := lb.batch, lb.dim, c.chunkSize

	chunkShape := []int{batch, chunkSize, dim}

	// Build contiguous chunk data for keys and values.
	chunkK := make([]T, batch*chunkSize*dim)
	chunkV := make([]T, batch*chunkSize*dim)
	for bi := range batch {
		srcOff := bi * chunkSize * dim
		copy(chunkK[srcOff:srcOff+chunkSize*dim], lb.recentKeys[srcOff:srcOff+chunkSize*dim])
		copy(chunkV[srcOff:srcOff+chunkSize*dim], lb.recentVals[srcOff:srcOff+chunkSize*dim])
	}

	kTensor, err := tensor.New(chunkShape, chunkK)
	if err != nil {
		return err
	}
	vTensor, err := tensor.New(chunkShape, chunkV)
	if err != nil {
		return err
	}

	// ReduceMean along axis 1 (sequence), keepDims=true → [batch, 1, dim].
	ctx := context.Background()
	kMean, err := c.engine.ReduceMean(ctx, kTensor, 1, true)
	if err != nil {
		return fmt.Errorf("ReduceMean keys: %w", err)
	}
	vMean, err := c.engine.ReduceMean(ctx, vTensor, 1, true)
	if err != nil {
		return fmt.Errorf("ReduceMean values: %w", err)
	}

	// Append compressed [batch, 1, dim] data to compressed store.
	lb.compressedKeys = append(lb.compressedKeys, kMean.Data()...)
	lb.compressedVals = append(lb.compressedVals, vMean.Data()...)
	lb.numChunks++

	// Reset chunk cursor.
	lb.chunkCursor = 0

	return nil
}

// Get returns the cached key-value pair for the given layer. The returned
// tensors have shape [batch, numCompressedChunks + recentTokens, dim],
// with compressed chunks first followed by uncompressed recent tokens.
func (c *CompressedKVCache[T]) Get(layer int) (*LayerKV[T], bool) {
	if layer < 0 || layer >= len(c.layers) {
		return nil, false
	}
	lb := &c.layers[layer]
	totalSeq := lb.numChunks + lb.chunkCursor
	if totalSeq == 0 {
		return nil, false
	}

	batch, dim := lb.batch, lb.dim
	size := batch * totalSeq * dim
	keyData := make([]T, size)
	valData := make([]T, size)

	// Copy compressed chunks then recent tokens per batch.
	for bi := range batch {
		dstOff := bi * totalSeq * dim

		// Compressed chunks: stored as [batch*numChunks*dim] contiguously
		// with layout [b0_c0, b0_c1, ..., b1_c0, b1_c1, ...].
		if lb.numChunks > 0 {
			compSrcOff := bi * lb.numChunks * dim
			copy(keyData[dstOff:dstOff+lb.numChunks*dim], lb.compressedKeys[compSrcOff:compSrcOff+lb.numChunks*dim])
			copy(valData[dstOff:dstOff+lb.numChunks*dim], lb.compressedVals[compSrcOff:compSrcOff+lb.numChunks*dim])
		}

		// Recent (uncompressed) tokens.
		if lb.chunkCursor > 0 {
			recentDstOff := dstOff + lb.numChunks*dim
			recentSrcOff := bi * c.chunkSize * dim
			recentSize := lb.chunkCursor * dim
			copy(keyData[recentDstOff:recentDstOff+recentSize], lb.recentKeys[recentSrcOff:recentSrcOff+recentSize])
			copy(valData[recentDstOff:recentDstOff+recentSize], lb.recentVals[recentSrcOff:recentSrcOff+recentSize])
		}
	}

	shape := []int{batch, totalSeq, dim}
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

// Reset clears all cached data.
func (c *CompressedKVCache[T]) Reset() {
	for i := range c.layers {
		c.layers[i].compressedKeys = c.layers[i].compressedKeys[:0]
		c.layers[i].compressedVals = c.layers[i].compressedVals[:0]
		c.layers[i].numChunks = 0
		c.layers[i].chunkCursor = 0
	}
}

// Truncate reduces the cache to newSeqLen original tokens. Compressed chunks
// that fall entirely within newSeqLen are kept; the recent buffer is trimmed
// to cover the remainder. If newSeqLen falls in the middle of a compressed
// chunk, that chunk and all subsequent data are discarded (lossy truncation).
func (c *CompressedKVCache[T]) Truncate(newSeqLen int) {
	chunksToKeep := newSeqLen / c.chunkSize
	recentToKeep := newSeqLen % c.chunkSize

	for i := range c.layers {
		lb := &c.layers[i]
		if lb.recentKeys == nil {
			continue
		}

		if chunksToKeep < lb.numChunks {
			lb.numChunks = chunksToKeep
			lb.compressedKeys = lb.compressedKeys[:lb.batch*chunksToKeep*lb.dim]
			lb.compressedVals = lb.compressedVals[:lb.batch*chunksToKeep*lb.dim]
		}

		if recentToKeep < lb.chunkCursor {
			lb.chunkCursor = recentToKeep
		}
	}
}
