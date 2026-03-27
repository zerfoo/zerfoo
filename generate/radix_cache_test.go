package generate

import (
	"math/rand"
	"sync"
	"testing"
)

func TestRadixCache_InsertAndMatch(t *testing.T) {
	rc := NewRadixCache(4, 100)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}
	ids := rc.Insert(tokens)

	if len(ids) != 2 { // 8 tokens / blockSize 4 = 2 blocks
		t.Fatalf("Insert returned %d block IDs, want 2", len(ids))
	}

	matchLen, matchIDs := rc.Match(tokens)
	if matchLen != 8 {
		t.Errorf("matchLen = %d, want 8", matchLen)
	}
	if len(matchIDs) != 2 {
		t.Fatalf("Match returned %d block IDs, want 2", len(matchIDs))
	}
	for i := range ids {
		if matchIDs[i] != ids[i] {
			t.Errorf("matchIDs[%d] = %d, want %d", i, matchIDs[i], ids[i])
		}
	}
}

func TestRadixCache_PartialMatch(t *testing.T) {
	rc := NewRadixCache(4, 100)

	rc.Insert([]int{1, 2, 3, 4, 5, 6, 7, 8})

	// Query with first block matching, second diverging.
	matchLen, matchIDs := rc.Match([]int{1, 2, 3, 4, 9, 10, 11, 12})
	if matchLen != 4 {
		t.Errorf("matchLen = %d, want 4", matchLen)
	}
	if len(matchIDs) != 1 {
		t.Errorf("len(matchIDs) = %d, want 1", len(matchIDs))
	}
}

func TestRadixCache_NoMatch(t *testing.T) {
	rc := NewRadixCache(4, 100)

	rc.Insert([]int{1, 2, 3, 4})

	matchLen, matchIDs := rc.Match([]int{5, 6, 7, 8})
	if matchLen != 0 {
		t.Errorf("matchLen = %d, want 0", matchLen)
	}
	if len(matchIDs) != 0 {
		t.Errorf("len(matchIDs) = %d, want 0", len(matchIDs))
	}
}

func TestRadixCache_EmptyInput(t *testing.T) {
	rc := NewRadixCache(4, 100)

	ids := rc.Insert(nil)
	if ids != nil {
		t.Errorf("Insert(nil) = %v, want nil", ids)
	}

	matchLen, matchIDs := rc.Match(nil)
	if matchLen != 0 || matchIDs != nil {
		t.Errorf("Match(nil) = (%d, %v), want (0, nil)", matchLen, matchIDs)
	}
}

func TestRadixCache_PartialBlock(t *testing.T) {
	rc := NewRadixCache(4, 100)

	// 6 tokens = 1 full block + 1 partial block.
	tokens := []int{1, 2, 3, 4, 5, 6}
	ids := rc.Insert(tokens)
	if len(ids) != 2 {
		t.Fatalf("Insert returned %d block IDs, want 2", len(ids))
	}

	matchLen, matchIDs := rc.Match(tokens)
	if matchLen != 6 {
		t.Errorf("matchLen = %d, want 6", matchLen)
	}
	if len(matchIDs) != 2 {
		t.Errorf("len(matchIDs) = %d, want 2", len(matchIDs))
	}
}

func TestRadixCache_LRUEviction(t *testing.T) {
	// maxBlocks=3, blockSize=2. Insert 3 blocks then insert a 4th to trigger eviction.
	rc := NewRadixCache(2, 3)

	// Insert prefix A: tokens [1,2,3,4] = 2 blocks.
	rc.Insert([]int{1, 2, 3, 4})

	// Insert prefix B: tokens [5,6] = 1 block. Cache now full (3 blocks).
	rc.Insert([]int{5, 6})

	// Access prefix A to refresh its timestamps.
	rc.Match([]int{1, 2, 3, 4})

	// Insert prefix C: tokens [7,8] = 1 block. Must evict the LRU leaf (prefix B).
	rc.Insert([]int{7, 8})

	// Prefix B should be evicted.
	matchLen, _ := rc.Match([]int{5, 6})
	if matchLen != 0 {
		t.Errorf("prefix B matchLen = %d, want 0 (should be evicted)", matchLen)
	}

	// Prefix A should still be present.
	matchLen, _ = rc.Match([]int{1, 2, 3, 4})
	if matchLen != 4 {
		t.Errorf("prefix A matchLen = %d, want 4", matchLen)
	}

	_, _, evictions := rc.Stats()
	if evictions != 1 {
		t.Errorf("evictions = %d, want 1", evictions)
	}
}

func TestRadixCache_ExplicitEvict(t *testing.T) {
	rc := NewRadixCache(2, 100)

	rc.Insert([]int{1, 2})
	rc.Evict()

	matchLen, _ := rc.Match([]int{1, 2})
	if matchLen != 0 {
		t.Errorf("matchLen = %d, want 0 after eviction", matchLen)
	}
}

func TestRadixCache_Stats(t *testing.T) {
	rc := NewRadixCache(4, 100)

	rc.Insert([]int{1, 2, 3, 4})

	// Hit.
	rc.Match([]int{1, 2, 3, 4})
	// Miss.
	rc.Match([]int{5, 6, 7, 8})

	hits, misses, _ := rc.Stats()
	if hits != 1 {
		t.Errorf("hits = %d, want 1", hits)
	}
	if misses != 1 {
		t.Errorf("misses = %d, want 1", misses)
	}
}

func TestRadixCache_SharedPrefix(t *testing.T) {
	rc := NewRadixCache(4, 100)

	// Two sequences sharing a 4-token prefix.
	rc.Insert([]int{1, 2, 3, 4, 10, 11, 12, 13})
	rc.Insert([]int{1, 2, 3, 4, 20, 21, 22, 23})

	// Both should match the shared first block.
	matchLen1, ids1 := rc.Match([]int{1, 2, 3, 4, 10, 11, 12, 13})
	matchLen2, ids2 := rc.Match([]int{1, 2, 3, 4, 20, 21, 22, 23})

	if matchLen1 != 8 {
		t.Errorf("matchLen1 = %d, want 8", matchLen1)
	}
	if matchLen2 != 8 {
		t.Errorf("matchLen2 = %d, want 8", matchLen2)
	}

	// First block ID should be shared.
	if ids1[0] != ids2[0] {
		t.Errorf("shared block IDs differ: %d vs %d", ids1[0], ids2[0])
	}

	// Second block IDs should differ.
	if ids1[1] == ids2[1] {
		t.Errorf("divergent block IDs should differ, both = %d", ids1[1])
	}
}

func TestRadixCache_1000Prefixes(t *testing.T) {
	const (
		blockSize  = 16
		maxBlocks  = 10000
		numInserts = 1000
		seqLen     = 64 // 4 blocks per sequence
	)

	rc := NewRadixCache(blockSize, maxBlocks)
	rng := rand.New(rand.NewSource(42))

	// Generate and insert 1000 unique prefixes.
	prefixes := make([][]int, numInserts)
	allIDs := make([][]int, numInserts)
	for i := range prefixes {
		seq := make([]int, seqLen)
		for j := range seq {
			seq[j] = rng.Intn(50000)
		}
		prefixes[i] = seq
		allIDs[i] = rc.Insert(seq)

		if len(allIDs[i]) != seqLen/blockSize {
			t.Fatalf("insert %d: got %d block IDs, want %d",
				i, len(allIDs[i]), seqLen/blockSize)
		}
	}

	// Verify all 1000 prefixes match correctly.
	for i, prefix := range prefixes {
		matchLen, matchIDs := rc.Match(prefix)
		if matchLen != seqLen {
			t.Errorf("prefix %d: matchLen = %d, want %d", i, matchLen, seqLen)
		}
		if len(matchIDs) != len(allIDs[i]) {
			t.Errorf("prefix %d: got %d block IDs, want %d",
				i, len(matchIDs), len(allIDs[i]))
			continue
		}
		for j := range matchIDs {
			if matchIDs[j] != allIDs[i][j] {
				t.Errorf("prefix %d block %d: got ID %d, want %d",
					i, j, matchIDs[j], allIDs[i][j])
			}
		}
	}

	hits, _, _ := rc.Stats()
	if hits != numInserts {
		t.Errorf("hits = %d, want %d", hits, numInserts)
	}
}

func TestRadixCache_EvictionUnderPressure(t *testing.T) {
	// Small cache: blockSize=4, maxBlocks=8. Insert sequences until eviction
	// must occur, then verify the cache still functions.
	rc := NewRadixCache(4, 8)

	// Insert 4 sequences of 2 blocks each = 8 blocks (fills cache).
	for i := 0; i < 4; i++ {
		base := i * 100
		rc.Insert([]int{base, base + 1, base + 2, base + 3, base + 4, base + 5, base + 6, base + 7})
	}

	// Insert a 5th — forces eviction.
	rc.Insert([]int{500, 501, 502, 503})

	_, _, evictions := rc.Stats()
	if evictions == 0 {
		t.Error("expected at least one eviction")
	}

	// The newly inserted prefix should be matchable.
	matchLen, _ := rc.Match([]int{500, 501, 502, 503})
	if matchLen != 4 {
		t.Errorf("matchLen = %d, want 4", matchLen)
	}
}

func TestRadixCache_HashCollisionSafety(t *testing.T) {
	// Verify that different token blocks produce different hashes (probabilistic).
	rc := NewRadixCache(4, 100)

	rc.Insert([]int{1, 2, 3, 4})
	rc.Insert([]int{4, 3, 2, 1})

	matchLen, _ := rc.Match([]int{1, 2, 3, 4})
	if matchLen != 4 {
		t.Errorf("forward matchLen = %d, want 4", matchLen)
	}

	matchLen, _ = rc.Match([]int{4, 3, 2, 1})
	if matchLen != 4 {
		t.Errorf("reverse matchLen = %d, want 4", matchLen)
	}

	// Completely different tokens should not match.
	matchLen, _ = rc.Match([]int{9, 9, 9, 9})
	if matchLen != 0 {
		t.Errorf("unrelated matchLen = %d, want 0", matchLen)
	}
}

func TestRadixCache_ConcurrentAccess(t *testing.T) {
	rc := NewRadixCache(4, 1000)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			base := id * 1000
			tokens := []int{base, base + 1, base + 2, base + 3}
			rc.Insert(tokens)
			rc.Match(tokens)
		}(i)
	}
	wg.Wait()

	// All 10 prefixes should be findable.
	for i := 0; i < 10; i++ {
		base := i * 1000
		matchLen, _ := rc.Match([]int{base, base + 1, base + 2, base + 3})
		if matchLen != 4 {
			t.Errorf("goroutine %d: matchLen = %d, want 4", i, matchLen)
		}
	}
}

func TestHashBlock_Deterministic(t *testing.T) {
	tokens := []int{42, 100, 200, 300}
	h1 := hashBlock(tokens)
	h2 := hashBlock(tokens)
	if h1 != h2 {
		t.Errorf("hashBlock not deterministic: %d != %d", h1, h2)
	}

	// Different tokens should produce different hash.
	h3 := hashBlock([]int{42, 100, 200, 301})
	if h1 == h3 {
		t.Errorf("different tokens produced same hash: %d", h1)
	}
}

func TestRadixCache_SingleToken(t *testing.T) {
	rc := NewRadixCache(4, 100)

	// Single token = 1 partial block.
	ids := rc.Insert([]int{42})
	if len(ids) != 1 {
		t.Fatalf("Insert returned %d IDs, want 1", len(ids))
	}

	matchLen, matchIDs := rc.Match([]int{42})
	if matchLen != 1 {
		t.Errorf("matchLen = %d, want 1", matchLen)
	}
	if len(matchIDs) != 1 || matchIDs[0] != ids[0] {
		t.Errorf("matchIDs = %v, want %v", matchIDs, ids)
	}
}
