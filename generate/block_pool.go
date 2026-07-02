package generate

import (
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/tensor"
)

// Block holds pre-allocated key and value data for a fixed number of token
// positions across all layers. K and V each have
// numLayers * blockSize * headDim elements laid out as
// [layer][position][headDim] in row-major order.
type Block[T tensor.Numeric] struct {
	K    []T
	V    []T
	Used int // number of token positions written (0..blockSize)
}

// BlockPool manages a fixed-size pool of pre-allocated KV cache blocks.
// Blocks are allocated at startup and recycled via Alloc/Free. All methods
// are safe for concurrent use.
type BlockPool[T tensor.Numeric] struct {
	blocks    []Block[T]
	free      []*Block[T] // stack of available block pointers
	numLayers int
	blockSize int // tokens per block
	headDim   int
	mu        sync.Mutex
}

// NewBlockPool creates a pool of blocks sized to fit within maxMemoryMB.
// Each block holds K and V data for blockSize token positions across
// numLayers, with headDim elements per position per layer. The element
// size is assumed to be 4 bytes (float32).
func NewBlockPool[T tensor.Numeric](numLayers, blockSize, headDim, maxMemoryMB int) (*BlockPool[T], error) {
	if numLayers <= 0 || blockSize <= 0 || headDim <= 0 || maxMemoryMB <= 0 {
		return nil, fmt.Errorf("all parameters must be positive: layers=%d blockSize=%d headDim=%d maxMB=%d",
			numLayers, blockSize, headDim, maxMemoryMB)
	}

	const elemBytes = 4 // sizeof(float32)
	elemsPerSide := numLayers * blockSize * headDim
	blockBytes := 2 * elemsPerSide * elemBytes // K + V
	maxBytes := maxMemoryMB * 1024 * 1024
	numBlocks := maxBytes / blockBytes
	if numBlocks == 0 {
		return nil, fmt.Errorf("maxMemoryMB %d too small for one block (%d bytes required)",
			maxMemoryMB, blockBytes)
	}

	blocks := make([]Block[T], numBlocks)
	free := make([]*Block[T], numBlocks)
	for i := range blocks {
		blocks[i].K = make([]T, elemsPerSide)
		blocks[i].V = make([]T, elemsPerSide)
		free[i] = &blocks[numBlocks-1-i] // stack order: first alloc returns blocks[0]
	}

	return &BlockPool[T]{
		blocks:    blocks,
		free:      free,
		numLayers: numLayers,
		blockSize: blockSize,
		headDim:   headDim,
	}, nil
}

// Alloc returns a free block from the pool. Returns an error if the pool
// is exhausted. The returned block has Used reset to 0.
func (p *BlockPool[T]) Alloc() (*Block[T], error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.free) == 0 {
		return nil, fmt.Errorf("block pool exhausted: all %d blocks allocated", len(p.blocks))
	}

	b := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	return b, nil
}

// Free returns a block to the pool. The block's Used counter is reset to 0.
func (p *BlockPool[T]) Free(b *Block[T]) {
	b.Used = 0
	p.mu.Lock()
	p.free = append(p.free, b)
	p.mu.Unlock()
}

// Cap returns the total number of blocks in the pool.
func (p *BlockPool[T]) Cap() int {
	return len(p.blocks)
}

// Available returns the number of free blocks.
func (p *BlockPool[T]) Available() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.free)
}

// BlockSize returns the number of token positions per block.
func (p *BlockPool[T]) BlockSize() int {
	return p.blockSize
}

// FragmentationRatio returns the fraction of allocated block capacity that is
// wasted (allocated but unused token positions). A value of 0 means every
// allocated block is fully used; higher values indicate internal fragmentation
// from partially-filled blocks.
func (p *BlockPool[T]) FragmentationRatio() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()

	allocated := len(p.blocks) - len(p.free)
	if allocated == 0 {
		return 0
	}

	totalCapacity := allocated * p.blockSize
	var totalUsed int
	for i := range p.blocks {
		if p.blocks[i].Used > 0 {
			totalUsed += p.blocks[i].Used
		}
	}

	if totalCapacity == 0 {
		return 0
	}
	return 1 - float64(totalUsed)/float64(totalCapacity)
}
