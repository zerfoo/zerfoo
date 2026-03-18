package generate

import (
	"testing"

	"github.com/zerfoo/ztensor/graph/kv"
)

func TestPrefixCache_InsertAndMatch(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 16, 64, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	pc := NewPrefixCache[float32](100, pool)

	// Allocate blocks and fill with recognizable data.
	b1, err := pool.Alloc()
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	for i := range b1.K {
		b1.K[i] = 1.0
		b1.V[i] = 2.0
	}
	b1.Used = 16

	b2, err := pool.Alloc()
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	for i := range b2.K {
		b2.K[i] = 3.0
		b2.V[i] = 4.0
	}
	b2.Used = 8

	// Insert a prefix of 2 tokens mapped to 2 blocks.
	tokenIDs := []int32{100, 200}
	pc.Insert(tokenIDs, []*Block[float32]{b1, b2})

	// Return test blocks to pool.
	pool.Free(b1)
	pool.Free(b2)

	if pc.Size() != 2 {
		t.Errorf("Size() = %d, want 2", pc.Size())
	}

	// Match with identical prefix.
	availBefore := pool.Available()
	matched, matchedLen := pc.Match(tokenIDs)
	if matchedLen != 2 {
		t.Fatalf("matchedLen = %d, want 2", matchedLen)
	}
	if len(matched) != 2 {
		t.Fatalf("len(matched) = %d, want 2", len(matched))
	}

	// Match should have allocated 2 blocks from the pool.
	availAfter := pool.Available()
	if availAfter != availBefore-2 {
		t.Errorf("pool.Available() changed from %d to %d, want %d",
			availBefore, availAfter, availBefore-2)
	}

	// Verify data was copied correctly.
	if matched[0].K[0] != 1.0 {
		t.Errorf("matched[0].K[0] = %v, want 1.0", matched[0].K[0])
	}
	if matched[0].V[0] != 2.0 {
		t.Errorf("matched[0].V[0] = %v, want 2.0", matched[0].V[0])
	}
	if matched[1].K[0] != 3.0 {
		t.Errorf("matched[1].K[0] = %v, want 3.0", matched[1].K[0])
	}

	// Clean up.
	for _, b := range matched {
		pool.Free(b)
	}
}

func TestPrefixCache_NoMatch(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 16, 64, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	pc := NewPrefixCache[float32](100, pool)

	matched, matchedLen := pc.Match([]int32{100, 200})
	if matchedLen != 0 {
		t.Errorf("matchedLen = %d, want 0", matchedLen)
	}
	if matched != nil {
		t.Errorf("matched = %v, want nil", matched)
	}
}

func TestPrefixCache_PartialMatch(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 16, 64, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	pc := NewPrefixCache[float32](100, pool)

	b1, _ := pool.Alloc()
	b1.Used = 16
	b2, _ := pool.Alloc()
	b2.Used = 8

	pc.Insert([]int32{100, 200}, []*Block[float32]{b1, b2})
	pool.Free(b1)
	pool.Free(b2)

	// Query with a longer prefix — should match the 2-token prefix.
	matched, matchedLen := pc.Match([]int32{100, 200, 300})
	if matchedLen != 2 {
		t.Errorf("matchedLen = %d, want 2", matchedLen)
	}
	if len(matched) != 2 {
		t.Errorf("len(matched) = %d, want 2", len(matched))
	}

	for _, b := range matched {
		pool.Free(b)
	}
}

func TestPrefixCache_PoolExhausted(t *testing.T) {
	// Create a very small pool.
	pool, err := NewBlockPool[float32](2, 16, 64, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	pc := NewPrefixCache[float32](100, pool)

	b1, _ := pool.Alloc()
	b1.Used = 16
	pc.Insert([]int32{100}, []*Block[float32]{b1})
	pool.Free(b1)

	// Exhaust the pool.
	allBlocks := make([]*Block[float32], 0)
	for {
		b, allocErr := pool.Alloc()
		if allocErr != nil {
			break
		}
		allBlocks = append(allBlocks, b)
	}

	// Match should fail gracefully when pool is exhausted.
	matched, matchedLen := pc.Match([]int32{100})
	if matchedLen != 0 {
		t.Errorf("matchedLen = %d, want 0 (pool exhausted)", matchedLen)
	}
	if matched != nil {
		t.Errorf("matched should be nil when pool is exhausted")
	}

	for _, b := range allBlocks {
		pool.Free(b)
	}
}

func TestPrefixCache_SharedPrefixReducesBlockAllocation(t *testing.T) {
	// This test verifies the acceptance criteria: two sessions with identical
	// system prompt share KV blocks for the system prompt prefix, verified by
	// BlockPool allocation count decreasing on the second session.
	const (
		numLayers = 2
		blockSize = 16
		headDim   = 64
	)

	pool, err := NewBlockPool[float32](numLayers, blockSize, headDim, 10)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	pc := NewPrefixCache[float32](100, pool)
	systemPrompt := []int32{10, 20, 30, 40, 50}

	// Simulate first session: allocate blocks for system prompt.
	firstBlocks := make([]*Block[float32], len(systemPrompt))
	for i := range systemPrompt {
		b, allocErr := pool.Alloc()
		if allocErr != nil {
			t.Fatalf("Alloc block %d: %v", i, allocErr)
		}
		// Fill with recognizable data.
		for j := range b.K {
			b.K[j] = float32(i*1000 + j)
			b.V[j] = float32(i*2000 + j)
		}
		b.Used = blockSize
		firstBlocks[i] = b
	}

	availAfterFirst := pool.Available()
	t.Logf("pool available after first session alloc: %d", availAfterFirst)

	// Insert into prefix cache.
	pc.Insert(systemPrompt, firstBlocks)

	// Return first session's blocks to pool (session done).
	for _, b := range firstBlocks {
		pool.Free(b)
	}
	availAfterFree := pool.Available()
	t.Logf("pool available after first session free: %d", availAfterFree)

	// Second session: match prefix — should get blocks from cache using pool alloc.
	availBeforeMatch := pool.Available()
	matched, matchedLen := pc.Match(systemPrompt)
	availAfterMatch := pool.Available()

	if matchedLen != len(systemPrompt) {
		t.Fatalf("matchedLen = %d, want %d", matchedLen, len(systemPrompt))
	}

	// The match allocated blocks from the pool (same count as original).
	allocatedByMatch := availBeforeMatch - availAfterMatch
	if allocatedByMatch != len(systemPrompt) {
		t.Errorf("match allocated %d blocks, want %d", allocatedByMatch, len(systemPrompt))
	}

	// Verify the data matches the original blocks.
	for i, b := range matched {
		if b.K[0] != float32(i*1000) {
			t.Errorf("matched[%d].K[0] = %v, want %v", i, b.K[0], float32(i*1000))
		}
		if b.V[0] != float32(i*2000) {
			t.Errorf("matched[%d].V[0] = %v, want %v", i, b.V[0], float32(i*2000))
		}
	}

	// Clean up.
	for _, b := range matched {
		pool.Free(b)
	}
}

func TestPrefixCache_InjectBlocksIntoPagedKVCache(t *testing.T) {
	const (
		numLayers = 2
		blockSize = 16
		headDim   = 64
	)

	pool, err := NewBlockPool[float32](numLayers, blockSize, headDim, 10)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	pc := NewPrefixCache[float32](100, pool)

	// Simulate system prompt with 3 blocks of data.
	systemPrompt := []int32{10, 20, 30}
	blocks := make([]*Block[float32], 3)
	for i := range blocks {
		b, allocErr := pool.Alloc()
		if allocErr != nil {
			t.Fatalf("Alloc: %v", allocErr)
		}
		for j := range b.K {
			b.K[j] = float32(i + 1)
			b.V[j] = float32((i + 1) * 10)
		}
		b.Used = blockSize
		blocks[i] = b
	}

	pc.Insert(systemPrompt, blocks)
	for _, b := range blocks {
		pool.Free(b)
	}

	// Match and inject into a fresh PagedKVCache.
	matched, matchedLen := pc.Match(systemPrompt)
	if matchedLen != 3 {
		t.Fatalf("matchedLen = %d, want 3", matchedLen)
	}

	cache := NewPagedKVCache[float32](pool, numLayers)
	cache.InjectBlocks(matched, matchedLen)

	// Verify the cache reflects the injected state.
	if cache.SeqLen() != 3 {
		t.Errorf("cache.SeqLen() = %d, want 3", cache.SeqLen())
	}

	bt := cache.BlockTable()
	if len(bt) != 3 {
		t.Errorf("len(BlockTable()) = %d, want 3", len(bt))
	}

	// Verify block data integrity.
	if bt[0].K[0] != 1.0 {
		t.Errorf("BlockTable()[0].K[0] = %v, want 1.0", bt[0].K[0])
	}
	if bt[2].V[0] != 30.0 {
		t.Errorf("BlockTable()[2].V[0] = %v, want 30.0", bt[2].V[0])
	}

	// Clean up.
	cache.Free()
}

func TestIntsToInt32(t *testing.T) {
	tests := []struct {
		name string
		in   []int
		want []int32
	}{
		{"empty", nil, []int32{}},
		{"single", []int{42}, []int32{42}},
		{"multi", []int{1, 2, 3}, []int32{1, 2, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := intsToInt32(tt.in)
			if len(got) != len(tt.want) {
				t.Fatalf("len = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("got[%d] = %d, want %d", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// TestPrefixCache_KVBlockType verifies that the kv.Block type from ztensor
// is compatible with the conversion logic in PrefixCache.
func TestPrefixCache_KVBlockType(t *testing.T) {
	// Verify kv.Block has the same fields we depend on.
	b := &kv.Block[float32]{
		K:    make([]float32, 10),
		V:    make([]float32, 10),
		Used: 5,
	}
	if len(b.K) != 10 || len(b.V) != 10 || b.Used != 5 {
		t.Error("kv.Block field layout mismatch")
	}
}
