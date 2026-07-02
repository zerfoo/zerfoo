//go:build integration

package paged_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/tensor"
)

// TestConcurrent8Sessions verifies that 8 concurrent PagedAttention sessions
// can allocate, write, and read KV cache blocks without corruption, and that
// the KV memory waste (fragmentation) stays below 4% while all sessions are
// active.
func TestConcurrent8Sessions(t *testing.T) {
	const (
		numSessions      = 8
		numLayers        = 16
		blockSize        = 4
		headDim          = 64
		tokensPerSession = 20
		maxMemoryMB      = 64
	)

	pool, err := generate.NewBlockPool[float32](numLayers, blockSize, headDim, maxMemoryMB)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	type sessionResult struct {
		idx int
		err error
	}

	// Each goroutine owns its cache and signals when generation is done.
	// Caches are kept alive until fragmentation is measured.
	caches := make([]*generate.PagedKVCache[float32], numSessions)
	results := make([]sessionResult, numSessions)
	var wg sync.WaitGroup

	for s := range numSessions {
		wg.Add(1)
		go func(sessionIdx int) {
			defer wg.Done()

			cache := generate.NewPagedKVCache[float32](pool, numLayers)
			caches[sessionIdx] = cache

			for tok := range tokensPerSession {
				for layer := range numLayers {
					kData := make([]float32, headDim)
					vData := make([]float32, headDim)
					for i := range headDim {
						val := float32(sessionIdx*10000+tok*100+layer) + float32(i)*0.01
						kData[i] = val
						vData[i] = val + 0.5
					}

					k, kErr := tensor.New([]int{1, 1, headDim}, kData)
					if kErr != nil {
						results[sessionIdx] = sessionResult{sessionIdx, kErr}
						return
					}
					v, vErr := tensor.New([]int{1, 1, headDim}, vData)
					if vErr != nil {
						results[sessionIdx] = sessionResult{sessionIdx, vErr}
						return
					}

					if appendErr := cache.Append(layer, k, v); appendErr != nil {
						results[sessionIdx] = sessionResult{sessionIdx, appendErr}
						return
					}
				}
			}

			// Verify token count.
			if got := cache.SeqLen(); got != tokensPerSession {
				results[sessionIdx] = sessionResult{sessionIdx, fmt.Errorf("session %d: SeqLen() = %d, want %d", sessionIdx, got, tokensPerSession)}
				return
			}

			// Verify data integrity for layer 0.
			lkv, ok := cache.GetKV(0)
			if !ok {
				results[sessionIdx] = sessionResult{sessionIdx, fmt.Errorf("session %d: GetKV(0) returned false", sessionIdx)}
				return
			}

			kd := lkv.Key.Data()
			wantFirst := float32(sessionIdx*10000) + 0
			if kd[0] != wantFirst {
				results[sessionIdx] = sessionResult{sessionIdx, fmt.Errorf("session %d: Key[0][0] = %v, want %v", sessionIdx, kd[0], wantFirst)}
				return
			}

			lastOff := (tokensPerSession - 1) * headDim
			wantLast := float32(sessionIdx*10000 + (tokensPerSession-1)*100)
			if kd[lastOff] != wantLast {
				results[sessionIdx] = sessionResult{sessionIdx, fmt.Errorf("session %d: Key[last][0] = %v, want %v", sessionIdx, kd[lastOff], wantLast)}
				return
			}
		}(s)
	}

	wg.Wait()

	// Check for session errors.
	for _, r := range results {
		if r.err != nil {
			t.Errorf("session %d failed: %v", r.idx, r.err)
		}
	}

	// Measure fragmentation while all 8 sessions still hold their blocks.
	frag := pool.FragmentationRatio()
	t.Logf("FragmentationRatio with %d active sessions (%d tokens each, blockSize=%d): %.4f",
		numSessions, tokensPerSession, blockSize, frag)
	if frag >= 0.04 {
		t.Errorf("FragmentationRatio = %.4f, want < 0.04", frag)
	}

	// Clean up: free all caches.
	for _, cache := range caches {
		if cache != nil {
			cache.Free()
		}
	}

	// Verify all blocks returned to pool.
	if got, want := pool.Available(), pool.Cap(); got != want {
		t.Errorf("Available() after cleanup = %d, want %d", got, want)
	}
}
