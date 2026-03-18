package generate

import (
	"sync"

	"github.com/zerfoo/ztensor/graph/kv"
	"github.com/zerfoo/ztensor/tensor"
)

// PrefixCache wraps a radix tree to cache KV blocks for shared prompt prefixes.
// When multiple sessions share the same system prompt, the second session can
// skip prefill for the prefix by copying cached block data instead of running
// the forward pass. PrefixCache is safe for concurrent use.
type PrefixCache[T tensor.Numeric] struct {
	tree *kv.RadixTree[T]
	pool *BlockPool[T]
	mu   sync.Mutex
}

// NewPrefixCache creates a prefix cache that stores up to capacity KV blocks
// in a radix tree. The pool is used to allocate blocks when inserting cached
// prefix data.
func NewPrefixCache[T tensor.Numeric](capacity int, pool *BlockPool[T]) *PrefixCache[T] {
	return &PrefixCache[T]{
		tree: kv.NewRadixTree[T](capacity),
		pool: pool,
	}
}

// Insert stores the KV blocks associated with a token prefix in the cache.
// The block data is copied into kv.Block[T] instances owned by the radix tree.
func (pc *PrefixCache[T]) Insert(tokenIDs []int32, blocks []*Block[T]) {
	if len(tokenIDs) == 0 || len(blocks) == 0 {
		return
	}

	// Convert generate.Block[T] to kv.Block[T] by copying data.
	kvBlocks := make([]*kv.Block[T], len(blocks))
	for i, b := range blocks {
		kvBlocks[i] = &kv.Block[T]{
			K:    make([]T, len(b.K)),
			V:    make([]T, len(b.V)),
			Used: b.Used,
		}
		copy(kvBlocks[i].K, b.K)
		copy(kvBlocks[i].V, b.V)
	}

	pc.tree.Insert(tokenIDs, kvBlocks)
}

// Match returns the cached blocks for the longest matching prefix and the
// number of tokens matched. The returned blocks are freshly allocated from
// the pool with data copied from the cache, so the caller owns them.
// Returns nil, 0 if no prefix matches or block allocation fails.
func (pc *PrefixCache[T]) Match(prefix []int32) ([]*Block[T], int) {
	kvBlocks, matchedLen := pc.tree.Match(prefix)
	if matchedLen == 0 {
		return nil, 0
	}

	// Convert kv.Block[T] back to generate.Block[T] by allocating from pool
	// and copying data.
	genBlocks := make([]*Block[T], len(kvBlocks))
	for i, kvb := range kvBlocks {
		b, err := pc.pool.Alloc()
		if err != nil {
			// Pool exhausted — free already-allocated blocks and bail.
			for j := 0; j < i; j++ {
				pc.pool.Free(genBlocks[j])
			}
			return nil, 0
		}
		copy(b.K, kvb.K)
		copy(b.V, kvb.V)
		b.Used = kvb.Used
		genBlocks[i] = b
	}

	return genBlocks, matchedLen
}

// Size returns the number of blocks currently cached in the tree.
func (pc *PrefixCache[T]) Size() int {
	return pc.tree.Size()
}
