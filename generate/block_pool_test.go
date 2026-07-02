package generate

import (
	"sync"
	"testing"
)

func TestNewBlockPool(t *testing.T) {
	tests := []struct {
		name      string
		layers    int
		blockSize int
		headDim   int
		maxMB     int
		wantErr   bool
		minBlocks int // minimum expected blocks (0 means don't check)
	}{
		{
			name:      "valid small pool",
			layers:    2,
			blockSize: 16,
			headDim:   64,
			maxMB:     1,
			wantErr:   false,
			minBlocks: 1,
		},
		{
			name:      "larger pool",
			layers:    32,
			blockSize: 16,
			headDim:   128,
			maxMB:     256,
			wantErr:   false,
			minBlocks: 1,
		},
		{
			name:    "zero layers",
			layers:  0,
			maxMB:   1,
			wantErr: true,
		},
		{
			name:      "zero blockSize",
			layers:    2,
			blockSize: 0,
			headDim:   64,
			maxMB:     1,
			wantErr:   true,
		},
		{
			name:      "zero headDim",
			layers:    2,
			blockSize: 16,
			headDim:   0,
			maxMB:     1,
			wantErr:   true,
		},
		{
			name:      "zero maxMB",
			layers:    2,
			blockSize: 16,
			headDim:   64,
			maxMB:     0,
			wantErr:   true,
		},
		{
			name:      "negative layers",
			layers:    -1,
			blockSize: 16,
			headDim:   64,
			maxMB:     1,
			wantErr:   true,
		},
		{
			name:      "memory too small for one block",
			layers:    32,
			blockSize: 16,
			headDim:   128,
			maxMB:     0,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pool, err := NewBlockPool[float32](tt.layers, tt.blockSize, tt.headDim, tt.maxMB)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if pool == nil {
				t.Fatal("pool is nil")
			}
			if got := pool.Cap(); got < tt.minBlocks {
				t.Errorf("Cap() = %d, want >= %d", got, tt.minBlocks)
			}
			if got := pool.Available(); got != pool.Cap() {
				t.Errorf("Available() = %d, want %d (all free)", got, pool.Cap())
			}
		})
	}
}

func TestBlockPool_AllocFree(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 16, 64, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	cap := pool.Cap()
	if cap == 0 {
		t.Fatal("pool has 0 capacity")
	}

	// Allocate all blocks.
	blocks := make([]*Block[float32], 0, cap)
	for range cap {
		b, allocErr := pool.Alloc()
		if allocErr != nil {
			t.Fatalf("Alloc error: %v", allocErr)
		}
		if b == nil {
			t.Fatal("Alloc returned nil block")
		}
		blocks = append(blocks, b)
	}

	if got := pool.Available(); got != 0 {
		t.Errorf("Available() after full alloc = %d, want 0", got)
	}

	// Next alloc should fail.
	_, allocErr := pool.Alloc()
	if allocErr == nil {
		t.Error("Alloc on exhausted pool should return error")
	}

	// Free all blocks.
	for _, b := range blocks {
		pool.Free(b)
	}

	if got := pool.Available(); got != cap {
		t.Errorf("Available() after free all = %d, want %d", got, cap)
	}

	// Should be able to allocate again.
	b, allocErr := pool.Alloc()
	if allocErr != nil {
		t.Fatalf("Alloc after free: %v", allocErr)
	}
	if b == nil {
		t.Fatal("Alloc after free returned nil")
	}
}

func TestBlockPool_BlockDataIntegrity(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 4, 8, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	b, err := pool.Alloc()
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}

	// Verify block dimensions.
	// Each side (K, V) has numLayers * blockSize * headDim elements = 2*4*8 = 64.
	wantLen := 2 * 4 * 8
	if got := len(b.K); got != wantLen {
		t.Errorf("len(K) = %d, want %d", got, wantLen)
	}
	if got := len(b.V); got != wantLen {
		t.Errorf("len(V) = %d, want %d", got, wantLen)
	}

	// Write data and verify it persists.
	for i := range b.K {
		b.K[i] = float32(i)
		b.V[i] = float32(i + 100)
	}

	// Verify data.
	for i := range b.K {
		if b.K[i] != float32(i) {
			t.Errorf("K[%d] = %v, want %v", i, b.K[i], float32(i))
			break
		}
		if b.V[i] != float32(i+100) {
			t.Errorf("V[%d] = %v, want %v", i, b.V[i], float32(i+100))
			break
		}
	}
}

func TestBlockPool_BlockUsedField(t *testing.T) {
	pool, err := NewBlockPool[float32](1, 16, 4, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	b, err := pool.Alloc()
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}

	if b.Used != 0 {
		t.Errorf("new block Used = %d, want 0", b.Used)
	}

	b.Used = 5
	pool.Free(b)

	// After free and re-alloc, Used should be reset to 0.
	b2, err := pool.Alloc()
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	if b2.Used != 0 {
		t.Errorf("re-allocated block Used = %d, want 0", b2.Used)
	}
}

func TestBlockPool_ConcurrentAccess(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 16, 64, 10)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	cap := pool.Cap()
	if cap < 10 {
		t.Skipf("pool too small for concurrency test: %d blocks", cap)
	}

	const goroutines = 8
	const opsPerGoroutine = 50

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for range goroutines {
		go func() {
			defer wg.Done()
			var held []*Block[float32]
			for range opsPerGoroutine {
				if len(held) > 0 && len(held) > cap/goroutines {
					// Free some blocks.
					pool.Free(held[0])
					held = held[1:]
				}
				b, allocErr := pool.Alloc()
				if allocErr != nil {
					// Pool exhausted, free one and retry.
					if len(held) > 0 {
						pool.Free(held[0])
						held = held[1:]
					}
					continue
				}
				held = append(held, b)
			}
			for _, b := range held {
				pool.Free(b)
			}
		}()
	}

	wg.Wait()

	if got := pool.Available(); got != cap {
		t.Errorf("Available() after concurrent test = %d, want %d", got, cap)
	}
}

func TestBlockPool_MemoryComputation(t *testing.T) {
	// 2 layers, blockSize=16, headDim=64
	// Block bytes = 2 (K+V) * 2 * 16 * 64 * 4 bytes = 16384 bytes = 16 KB
	// 1 MB = 1048576 bytes -> 1048576 / 16384 = 64 blocks
	pool, err := NewBlockPool[float32](2, 16, 64, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	if got := pool.Cap(); got != 64 {
		t.Errorf("Cap() = %d, want 64 (1MB / 16KB per block)", got)
	}
}

func TestBlockPool_BlockSize(t *testing.T) {
	pool, err := NewBlockPool[float32](1, 16, 4, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	if got := pool.BlockSize(); got != 16 {
		t.Errorf("BlockSize() = %d, want 16", got)
	}
}

func BenchmarkBlockPool_AllocFree(b *testing.B) {
	pool, err := NewBlockPool[float32](32, 16, 128, 256)
	if err != nil {
		b.Fatalf("NewBlockPool: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		block, allocErr := pool.Alloc()
		if allocErr != nil {
			b.Fatal(allocErr)
		}
		pool.Free(block)
	}
}
