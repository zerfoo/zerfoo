package generate

import (
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// PagedKVCache stores key-value tensors for autoregressive generation using
// block-level allocation from a BlockPool. Instead of pre-allocating the full
// maxSeqLen per sequence, blocks of blockSize tokens are allocated on demand,
// reducing memory waste for concurrent sequences of varying length.
//
// Each sequence gets its own PagedKVCache. The cache accepts tensors with
// arbitrary first dimensions (channels). GQA attention stores KV as
// [batchSize*numKVHeads, seqLen, headDim]; the pool's headDim must equal
// channels * dim to accommodate the full per-position data.
type PagedKVCache[T tensor.Numeric] struct {
	pool      *BlockPool[T]
	numLayers int
	blockSize int
	headDim   int // pool's headDim = channels * perPosDim

	// channels and perPosDim are lazily detected from the first Append call.
	// channels is the first dimension of the KV tensors (e.g., numKVHeads for GQA).
	// perPosDim is the last dimension (actual head dim per channel).
	channels  int
	perPosDim int

	// blockTable holds the allocated blocks in order.
	// Blocks are shared across layers: block.K and block.V are laid out as
	// [numLayers * blockSize * headDim], so layer L at position P within
	// a block is at offset L*blockSize*headDim + P*headDim.
	blockTable []*Block[T]

	// layerCursors tracks the number of tokens appended per layer.
	layerCursors []int
}

// NewPagedKVCache creates a paged KV cache backed by the given block pool.
func NewPagedKVCache[T tensor.Numeric](pool *BlockPool[T], numLayers int) *PagedKVCache[T] {
	return &PagedKVCache[T]{
		pool:         pool,
		numLayers:    numLayers,
		blockSize:    pool.blockSize,
		headDim:      pool.headDim,
		layerCursors: make([]int, numLayers),
	}
}

// SeqLen returns the number of token positions stored in the cache,
// based on layer 0's cursor. Returns 0 if the cache is empty.
func (c *PagedKVCache[T]) SeqLen() int {
	if c.numLayers == 0 {
		return 0
	}
	return c.layerCursors[0]
}

// Append writes new key and value data for the given layer. The tensors must
// have shape [channels, seqLen, dim] where channels*dim equals the pool's
// headDim. For standard caching channels=1; for GQA caching channels equals
// batchSize*numKVHeads. Data is written into the current block; a new block
// is allocated from the pool when the current one fills up.
func (c *PagedKVCache[T]) Append(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	if layer < 0 || layer >= c.numLayers {
		return fmt.Errorf("layer %d out of range [0, %d)", layer, c.numLayers)
	}

	shape := newK.Shape()
	if len(shape) != 3 {
		return fmt.Errorf("expected 3D tensor [channels, seq, dim], got %dD", len(shape))
	}
	channels, seqLen, dim := shape[0], shape[1], shape[2]
	perPosSize := channels * dim
	if perPosSize != c.headDim {
		return fmt.Errorf("channels*dim mismatch: pool has headDim=%d, tensor has %d*%d=%d",
			c.headDim, channels, dim, perPosSize)
	}

	// Lazily detect channel layout from first Append.
	if c.channels == 0 {
		c.channels = channels
		c.perPosDim = dim
	} else if c.channels != channels || c.perPosDim != dim {
		return fmt.Errorf("channel layout mismatch: expected [%d, *, %d], got [%d, *, %d]",
			c.channels, c.perPosDim, channels, dim)
	}

	cursor := c.layerCursors[layer]
	kData := newK.Data()
	vData := newV.Data()

	for pos := range seqLen {
		globalPos := cursor + pos
		blockIdx := globalPos / c.blockSize
		posInBlock := globalPos % c.blockSize

		// Allocate new block if needed.
		for blockIdx >= len(c.blockTable) {
			b, err := c.pool.Alloc()
			if err != nil {
				return fmt.Errorf("alloc block: %w", err)
			}
			c.blockTable = append(c.blockTable, b)
		}

		block := c.blockTable[blockIdx]

		// Write K and V at the layer's region within the block.
		// Layout: [numLayers][blockSize][headDim] where headDim = channels*dim.
		dstBase := layer*c.blockSize*c.headDim + posInBlock*c.headDim
		for ch := range channels {
			srcOff := ch*seqLen*dim + pos*dim
			dstOff := dstBase + ch*dim
			copy(block.K[dstOff:dstOff+dim], kData[srcOff:srcOff+dim])
			copy(block.V[dstOff:dstOff+dim], vData[srcOff:srcOff+dim])
		}

		if posInBlock+1 > block.Used {
			block.Used = posInBlock + 1
		}
	}

	c.layerCursors[layer] = cursor + seqLen
	return nil
}

// GetKV returns the cached key and value tensors for the given layer,
// gathered into contiguous [channels, seqLen, dim] tensors. Returns false if
// the layer is out of range or the cache is empty for that layer.
func (c *PagedKVCache[T]) GetKV(layer int) (*LayerKV[T], bool) {
	if layer < 0 || layer >= c.numLayers {
		return nil, false
	}
	seqLen := c.layerCursors[layer]
	if seqLen == 0 {
		return nil, false
	}

	channels := c.channels
	dim := c.perPosDim
	if channels == 0 {
		// Never appended; shouldn't happen since seqLen > 0.
		return nil, false
	}

	totalElems := channels * seqLen * dim
	kOut := make([]T, totalElems)
	vOut := make([]T, totalElems)

	for pos := range seqLen {
		blockIdx := pos / c.blockSize
		posInBlock := pos % c.blockSize
		block := c.blockTable[blockIdx]

		srcBase := layer*c.blockSize*c.headDim + posInBlock*c.headDim
		for ch := range channels {
			srcOff := srcBase + ch*dim
			dstOff := ch*seqLen*dim + pos*dim
			copy(kOut[dstOff:dstOff+dim], block.K[srcOff:srcOff+dim])
			copy(vOut[dstOff:dstOff+dim], block.V[srcOff:srcOff+dim])
		}
	}

	kTensor, err := tensor.New([]int{channels, seqLen, dim}, kOut)
	if err != nil {
		return nil, false
	}
	vTensor, err := tensor.New([]int{channels, seqLen, dim}, vOut)
	if err != nil {
		return nil, false
	}

	return &LayerKV[T]{Key: kTensor, Value: vTensor}, true
}

// Update appends new key and value data for the given layer. This is an
// alias for Append that satisfies the CacheProvider interface.
func (c *PagedKVCache[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	return c.Append(layer, newK, newV)
}

// Get returns the cached KV for the given layer. This is an alias for
// GetKV that satisfies the CacheProvider interface.
func (c *PagedKVCache[T]) Get(layer int) (*LayerKV[T], bool) {
	return c.GetKV(layer)
}

// Reset clears the cache and returns all blocks to the pool.
func (c *PagedKVCache[T]) Reset() {
	c.Free()
}

// Free returns all allocated blocks to the pool and resets the cache.
func (c *PagedKVCache[T]) Free() {
	for _, b := range c.blockTable {
		c.pool.Free(b)
	}
	c.blockTable = c.blockTable[:0]
	c.channels = 0
	c.perPosDim = 0
	for i := range c.layerCursors {
		c.layerCursors[i] = 0
	}
}

// Truncate rolls back the cache to the given sequence length.
// Blocks beyond the new length are returned to the pool.
func (c *PagedKVCache[T]) Truncate(newSeqLen int) {
	if newSeqLen < 0 {
		newSeqLen = 0
	}
	for i := range c.layerCursors {
		if c.layerCursors[i] > newSeqLen {
			c.layerCursors[i] = newSeqLen
		}
	}

	// Free blocks beyond the new length.
	neededBlocks := 0
	if newSeqLen > 0 {
		neededBlocks = (newSeqLen-1)/c.blockSize + 1
	}
	for i := neededBlocks; i < len(c.blockTable); i++ {
		c.pool.Free(c.blockTable[i])
	}
	c.blockTable = c.blockTable[:neededBlocks]

	if newSeqLen == 0 {
		c.channels = 0
		c.perPosDim = 0
	}
}

// NumLayers returns the number of layers in the cache.
func (c *PagedKVCache[T]) NumLayers() int {
	return c.numLayers
}

// InjectBlocks sets the cache's block table to the given pre-populated blocks
// and advances all layer cursors to seqLen. This is used by the prefix cache
// to inject cached KV data without running a forward pass.
func (c *PagedKVCache[T]) InjectBlocks(blocks []*Block[T], seqLen int) {
	c.blockTable = append(c.blockTable[:0], blocks...)
	for i := range c.layerCursors {
		c.layerCursors[i] = seqLen
	}
}

// BlockTable returns the cache's current block table. This is used by the
// prefix cache to snapshot blocks after prefill for caching.
func (c *PagedKVCache[T]) BlockTable() []*Block[T] {
	return c.blockTable
}
