package generate

import (
	"hash/fnv"
	"sync"
	"sync/atomic"
	"time"
)

// RadixNode is a node in the hash-based radix tree. Each node stores a hash
// of the token block it represents, enabling O(1) comparison per block rather
// than per-token matching.
type RadixNode struct {
	hash       uint64
	children   map[uint64]*RadixNode
	blockID    int
	lastAccess time.Time
	isLeaf     bool
}

// RadixCache implements a hash-based radix tree for KV block prefix matching.
// Token sequences are divided into fixed-size blocks, each hashed with FNV-1a.
// Tree traversal matches one block hash per level, giving O(prefix_length / blockSize)
// matching complexity. LRU eviction removes the coldest leaf when the block pool
// is exhausted.
//
// RadixCache is safe for concurrent use.
type RadixCache struct {
	root       *RadixNode
	blockSize  int
	maxBlocks  int
	usedBlocks int
	mu         sync.RWMutex

	hits      atomic.Int64
	misses    atomic.Int64
	evictions atomic.Int64
}

// NewRadixCache creates a radix cache that hashes token sequences in chunks
// of blockSize tokens and stores up to maxBlocks block entries.
func NewRadixCache(blockSize, maxBlocks int) *RadixCache {
	return &RadixCache{
		root: &RadixNode{
			children: make(map[uint64]*RadixNode),
		},
		blockSize: blockSize,
		maxBlocks: maxBlocks,
	}
}

// Insert divides tokens into blocks of blockSize, hashes each block, and
// inserts them into the tree. Returns the block IDs assigned to each block.
// Partial trailing blocks (len < blockSize) are included. When the cache is
// full, LRU eviction frees space before allocating new blocks.
func (rc *RadixCache) Insert(tokens []int) []int {
	if len(tokens) == 0 {
		return nil
	}

	blocks := rc.chunkTokens(tokens)
	hashes := make([]uint64, len(blocks))
	for i, block := range blocks {
		hashes[i] = hashBlock(block)
	}

	rc.mu.Lock()
	defer rc.mu.Unlock()

	blockIDs := make([]int, 0, len(hashes))
	node := rc.root

	for i, h := range hashes {
		child, exists := node.children[h]
		if exists {
			child.lastAccess = time.Now()
			blockIDs = append(blockIDs, child.blockID)
			node = child
			continue
		}

		// Need a new node — evict if full.
		for rc.usedBlocks >= rc.maxBlocks {
			if !rc.evictLRU() {
				break
			}
		}

		id := rc.usedBlocks
		rc.usedBlocks++

		child = &RadixNode{
			hash:       h,
			children:   make(map[uint64]*RadixNode),
			blockID:    id,
			lastAccess: time.Now(),
			isLeaf:     true,
		}
		node.children[h] = child

		// The previous node is no longer a leaf if it was one.
		if node != rc.root {
			node.isLeaf = false
		}

		blockIDs = append(blockIDs, id)
		node = child

		// Mark remaining descendants as leaves if this is the last block.
		if i == len(hashes)-1 {
			child.isLeaf = true
		}
	}

	return blockIDs
}

// Match finds the longest prefix match for the given token sequence.
// Returns the number of tokens matched (a multiple of blockSize, or the
// full length if the last block is partial) and the block IDs for each
// matched block.
func (rc *RadixCache) Match(tokens []int) (matchLen int, blockIDs []int) {
	if len(tokens) == 0 {
		rc.misses.Add(1)
		return 0, nil
	}

	blocks := rc.chunkTokens(tokens)
	hashes := make([]uint64, len(blocks))
	for i, block := range blocks {
		hashes[i] = hashBlock(block)
	}

	rc.mu.RLock()
	defer rc.mu.RUnlock()

	node := rc.root
	now := time.Now()

	for i, h := range hashes {
		child, exists := node.children[h]
		if !exists {
			break
		}

		blockIDs = append(blockIDs, child.blockID)
		child.lastAccess = now

		// Calculate how many tokens this block covers.
		blockTokens := len(blocks[i])
		matchLen += blockTokens

		node = child
	}

	if matchLen == 0 {
		rc.misses.Add(1)
	} else {
		rc.hits.Add(1)
	}

	return matchLen, blockIDs
}

// Evict removes the least-recently-used leaf node and frees its block.
func (rc *RadixCache) Evict() {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.evictLRU()
}

// evictLRU finds and removes the LRU leaf. Must be called with rc.mu held.
func (rc *RadixCache) evictLRU() bool {
	type leafInfo struct {
		parent *RadixNode
		hash   uint64
		access time.Time
	}

	var coldest *leafInfo

	var walk func(parent *RadixNode)
	walk = func(parent *RadixNode) {
		for h, child := range parent.children {
			if len(child.children) == 0 {
				if coldest == nil || child.lastAccess.Before(coldest.access) {
					coldest = &leafInfo{
						parent: parent,
						hash:   h,
						access: child.lastAccess,
					}
				}
			} else {
				walk(child)
			}
		}
	}
	walk(rc.root)

	if coldest == nil {
		return false
	}

	delete(coldest.parent.children, coldest.hash)
	rc.usedBlocks--
	rc.evictions.Add(1)

	// If the parent (non-root) now has no children, mark it as a leaf.
	if coldest.parent != rc.root && len(coldest.parent.children) == 0 {
		coldest.parent.isLeaf = true
	}

	return true
}

// Stats returns cumulative hit, miss, and eviction counts.
func (rc *RadixCache) Stats() (hits, misses, evictions int) {
	return int(rc.hits.Load()), int(rc.misses.Load()), int(rc.evictions.Load())
}

// hashBlock computes an FNV-1a hash over a slice of token IDs.
func hashBlock(tokens []int) uint64 {
	h := fnv.New64a()
	// Write each token as 8 bytes (little-endian).
	var buf [8]byte
	for _, t := range tokens {
		buf[0] = byte(t)
		buf[1] = byte(t >> 8)
		buf[2] = byte(t >> 16)
		buf[3] = byte(t >> 24)
		buf[4] = byte(t >> 32)
		buf[5] = byte(t >> 40)
		buf[6] = byte(t >> 48)
		buf[7] = byte(t >> 56)
		h.Write(buf[:])
	}
	return h.Sum64()
}

// chunkTokens splits tokens into blocks of rc.blockSize. The last block
// may be smaller than blockSize.
func (rc *RadixCache) chunkTokens(tokens []int) [][]int {
	n := len(tokens)
	numBlocks := (n + rc.blockSize - 1) / rc.blockSize
	blocks := make([][]int, numBlocks)
	for i := range blocks {
		start := i * rc.blockSize
		end := start + rc.blockSize
		if end > n {
			end = n
		}
		blocks[i] = tokens[start:end]
	}
	return blocks
}
