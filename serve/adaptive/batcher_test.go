package adaptive

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestAdaptiveBatcher_ScalesUp(t *testing.T) {
	// When queue is deep and latency is under target, batch size should increase.
	cfg := Config{
		MinBatchSize:    1,
		MaxBatchSize:    16,
		TargetLatencyMS: 100,
		QueueTimeoutMS:  10,
	}

	var maxBatchSeen atomic.Int32

	handler := func(_ context.Context, reqs []Request) []Result {
		n := int32(len(reqs))
		for {
			old := maxBatchSeen.Load()
			if n <= old || maxBatchSeen.CompareAndSwap(old, n) {
				break
			}
		}
		// Fast handler — well under latency target.
		time.Sleep(1 * time.Millisecond)
		results := make([]Result, len(reqs))
		for i, r := range reqs {
			results[i] = Result{RequestID: r.ID, Output: []int{42}}
		}
		return results
	}

	b := New(cfg, handler)
	b.Start()
	defer b.Stop()

	// Flood the queue with many concurrent requests to create queue pressure.
	const numRequests = 64
	var wg sync.WaitGroup
	wg.Add(numRequests)
	for i := range numRequests {
		go func(idx int) {
			defer wg.Done()
			req := Request{ID: idStr(idx), Tokens: []int{1, 2, 3}}
			_, _ = b.Submit(context.Background(), req)
		}(i)
	}

	wg.Wait()

	// The batcher should have scaled up beyond the initial MinBatchSize of 1.
	finalSize := b.BatchSize()
	if finalSize <= cfg.MinBatchSize {
		t.Errorf("batch size did not scale up: got %d, want > %d", finalSize, cfg.MinBatchSize)
	}
	if got := maxBatchSeen.Load(); got <= 1 {
		t.Errorf("max batch seen = %d, want > 1 (should have formed larger batches)", got)
	}
}

func TestAdaptiveBatcher_ScalesDown(t *testing.T) {
	// When latency exceeds target, batch size should decrease.
	cfg := Config{
		MinBatchSize:    1,
		MaxBatchSize:    16,
		TargetLatencyMS: 5, // Very tight latency target.
		QueueTimeoutMS:  5,
	}

	handler := func(_ context.Context, reqs []Request) []Result {
		// Slow handler — exceeds the 5ms target.
		time.Sleep(20 * time.Millisecond)
		results := make([]Result, len(reqs))
		for i, r := range reqs {
			results[i] = Result{RequestID: r.ID, Output: []int{1}}
		}
		return results
	}

	b := New(cfg, handler)
	// Start with a high batch size to observe decrease.
	b.currentBatchSize.Store(int32(cfg.MaxBatchSize))
	b.Start()
	defer b.Stop()

	// Submit enough requests to trigger multiple adaptation cycles.
	const numRequests = 32
	var wg sync.WaitGroup
	wg.Add(numRequests)
	for i := range numRequests {
		go func(idx int) {
			defer wg.Done()
			req := Request{ID: idStr(idx), Tokens: []int{1}}
			_, _ = b.Submit(context.Background(), req)
		}(i)
	}

	wg.Wait()

	finalSize := b.BatchSize()
	if finalSize >= cfg.MaxBatchSize {
		t.Errorf("batch size did not scale down: got %d, want < %d", finalSize, cfg.MaxBatchSize)
	}
}

func TestAdaptiveBatcher_LatencyTarget(t *testing.T) {
	// Verify that the EMA tracks latency and the batcher respects the target.
	cfg := Config{
		MinBatchSize:    1,
		MaxBatchSize:    8,
		TargetLatencyMS: 50,
		QueueTimeoutMS:  10,
	}

	var callCount atomic.Int32

	handler := func(_ context.Context, reqs []Request) []Result {
		n := callCount.Add(1)
		// First few calls are fast, then get slow.
		if n > 3 {
			time.Sleep(80 * time.Millisecond)
		} else {
			time.Sleep(2 * time.Millisecond)
		}
		results := make([]Result, len(reqs))
		for i, r := range reqs {
			results[i] = Result{RequestID: r.ID, Output: []int{int(n)}}
		}
		return results
	}

	b := New(cfg, handler)
	b.currentBatchSize.Store(int32(cfg.MaxBatchSize))
	b.Start()
	defer b.Stop()

	// Submit requests in waves.
	for wave := range 6 {
		var wg sync.WaitGroup
		const perWave = 8
		wg.Add(perWave)
		for i := range perWave {
			go func(idx int) {
				defer wg.Done()
				req := Request{ID: idStr(wave*perWave + idx), Tokens: []int{1}}
				_, _ = b.Submit(context.Background(), req)
			}(i)
		}
		wg.Wait()
	}

	// After slow batches, the EMA should have risen and batch size should have decreased.
	ema := b.LatencyEMA()
	if ema <= 0 {
		t.Errorf("latency EMA = %f, want > 0", ema)
	}

	finalSize := b.BatchSize()
	if finalSize > cfg.MaxBatchSize {
		t.Errorf("batch size %d exceeds max %d", finalSize, cfg.MaxBatchSize)
	}
	if finalSize < cfg.MinBatchSize {
		t.Errorf("batch size %d below min %d", finalSize, cfg.MinBatchSize)
	}
}

func TestAdaptiveBatcher_ContextCancellation(t *testing.T) {
	cfg := Config{
		MinBatchSize:    1,
		MaxBatchSize:    8,
		TargetLatencyMS: 100,
		QueueTimeoutMS:  1000,
	}

	handler := func(_ context.Context, reqs []Request) []Result {
		results := make([]Result, len(reqs))
		return results
	}

	b := New(cfg, handler)
	b.Start()
	defer b.Stop()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := b.Submit(ctx, Request{ID: "canceled", Tokens: []int{1}})
	if err == nil {
		t.Error("expected error from canceled context")
	}
}

func TestAdaptiveBatcher_ConfigDefaults(t *testing.T) {
	tests := []struct {
		name   string
		input  Config
		wantFn func(t *testing.T, c Config)
	}{
		{
			name:  "zero config gets defaults",
			input: Config{},
			wantFn: func(t *testing.T, c Config) {
				if c.MinBatchSize != 1 {
					t.Errorf("MinBatchSize = %d, want 1", c.MinBatchSize)
				}
				if c.MaxBatchSize != 32 {
					t.Errorf("MaxBatchSize = %d, want 32", c.MaxBatchSize)
				}
				if c.TargetLatencyMS != 100 {
					t.Errorf("TargetLatencyMS = %f, want 100", c.TargetLatencyMS)
				}
				if c.QueueTimeoutMS != 50 {
					t.Errorf("QueueTimeoutMS = %f, want 50", c.QueueTimeoutMS)
				}
			},
		},
		{
			name:  "min clamped to max",
			input: Config{MinBatchSize: 10, MaxBatchSize: 4},
			wantFn: func(t *testing.T, c Config) {
				if c.MinBatchSize != c.MaxBatchSize {
					t.Errorf("MinBatchSize = %d, want %d (clamped to MaxBatchSize)", c.MinBatchSize, c.MaxBatchSize)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.input
			c.defaults()
			tt.wantFn(t, c)
		})
	}
}

func idStr(n int) string {
	// Simple int-to-string without importing strconv.
	if n == 0 {
		return "req-0"
	}
	s := "req-"
	digits := []byte{}
	for v := n; v > 0; v /= 10 {
		digits = append(digits, byte('0'+v%10))
	}
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}
	return s + string(digits)
}
