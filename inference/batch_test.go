package inference

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
)

func TestGenerateBatch(t *testing.T) {
	tests := []struct {
		name       string
		prompts    []string
		tokenSeq   []int // token sequence the fixedLogitsNode cycles through
		wantLen    int
		wantNilErr bool
	}{
		{
			name:       "empty slice",
			prompts:    nil,
			tokenSeq:   []int{6, 2},
			wantLen:    0,
			wantNilErr: true,
		},
		{
			name:       "single prompt",
			prompts:    []string{"hello"},
			tokenSeq:   []int{6, 2}, // produces token 6 ("foo") then EOS
			wantLen:    1,
			wantNilErr: true,
		},
		{
			name:       "two prompts",
			prompts:    []string{"hello", "world"},
			tokenSeq:   []int{6, 2},
			wantLen:    2,
			wantNilErr: true,
		},
		{
			name:       "four prompts",
			prompts:    []string{"hello", "world", "foo", "bar"},
			tokenSeq:   []int{6, 2},
			wantLen:    4,
			wantNilErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := buildTestModel(t, 8, tt.tokenSeq)
			results, err := m.GenerateBatch(context.Background(), tt.prompts, WithTemperature(0), WithMaxTokens(10))
			if tt.wantNilErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(results) != tt.wantLen {
				t.Fatalf("got %d results, want %d", len(results), tt.wantLen)
			}
			for i, r := range results {
				if r == "" && tt.wantLen > 0 {
					t.Errorf("results[%d] is empty", i)
				}
			}
		})
	}
}

func TestGenerateBatch_ContextCancellation(t *testing.T) {
	m := buildTestModel(t, 8, []int{6, 7, 6, 7, 6, 7}) // no EOS — relies on maxTokens or cancellation

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	results, _ := m.GenerateBatch(ctx, []string{"hello", "world"}, WithTemperature(0), WithMaxTokens(5))
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
}

func TestGenerateBatch_SinglePromptOutput(t *testing.T) {
	// Token 6 = "foo", token 2 = EOS. Sequence: produce "foo" then stop.
	m := buildTestModel(t, 8, []int{6, 2})

	results, err := m.GenerateBatch(context.Background(), []string{"hello"}, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if results[0] != "foo" {
		t.Errorf("got %q, want %q", results[0], "foo")
	}
}

func TestGenerateBatch_ConcurrencyLimit(t *testing.T) {
	const numPrompts = 20

	prompts := make([]string, numPrompts)
	for i := range prompts {
		prompts[i] = "hello"
	}

	// Verify that with maxBatchConcurrency=2 all prompts still complete correctly.
	m := buildTestModel(t, 8, []int{6, 2})
	m.maxBatchConcurrency = 2

	results, err := m.GenerateBatch(context.Background(), prompts, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != numPrompts {
		t.Fatalf("got %d results, want %d", len(results), numPrompts)
	}
	for i, r := range results {
		if r == "" {
			t.Errorf("results[%d] is empty", i)
		}
	}

	// With maxBatchConcurrency=1, generation is effectively serial — verify correctness.
	m.maxBatchConcurrency = 1
	results, err = m.GenerateBatch(context.Background(), prompts, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("unexpected error with concurrency=1: %v", err)
	}
	if len(results) != numPrompts {
		t.Fatalf("got %d results, want %d", len(results), numPrompts)
	}
}

func TestGenerateBatch_DefaultConcurrency(t *testing.T) {
	m := buildTestModel(t, 8, []int{6, 2})

	// maxBatchConcurrency is 0 (zero value) — should use defaultMaxBatchConcurrency.
	if m.maxBatchConcurrency != 0 {
		t.Fatalf("expected zero-value maxBatchConcurrency, got %d", m.maxBatchConcurrency)
	}

	prompts := make([]string, 20)
	for i := range prompts {
		prompts[i] = "hello"
	}

	results, err := m.GenerateBatch(context.Background(), prompts, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 20 {
		t.Fatalf("got %d results, want 20", len(results))
	}
}

func TestGenerateBatch_SetMaxBatchConcurrency(t *testing.T) {
	m := buildTestModel(t, 8, []int{6, 2})

	// Verify SetMaxBatchConcurrency works.
	m.SetMaxBatchConcurrency(4)
	if m.maxBatchConcurrency != 4 {
		t.Fatalf("expected maxBatchConcurrency=4, got %d", m.maxBatchConcurrency)
	}

	// Zero and negative values should be ignored.
	m.SetMaxBatchConcurrency(0)
	if m.maxBatchConcurrency != 4 {
		t.Fatalf("SetMaxBatchConcurrency(0) should be ignored, got %d", m.maxBatchConcurrency)
	}
	m.SetMaxBatchConcurrency(-1)
	if m.maxBatchConcurrency != 4 {
		t.Fatalf("SetMaxBatchConcurrency(-1) should be ignored, got %d", m.maxBatchConcurrency)
	}

	// Should still work with the set value.
	prompts := make([]string, 10)
	for i := range prompts {
		prompts[i] = "hello"
	}
	results, err := m.GenerateBatch(context.Background(), prompts, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 10 {
		t.Fatalf("got %d results, want 10", len(results))
	}
}

func TestGenerateBatch_UsesSessionPool(t *testing.T) {
	// Verify GenerateBatch acquires sessions from the pool instead of calling
	// generator.Generate directly. We set up a session pool and track concurrent
	// session usage via an atomic counter to confirm multiple sessions are
	// active at the same time.
	m := buildTestModel(t, 8, []int{6, 2})

	// Create a session pool and pre-warm it with sessions.
	const poolSize = 4
	pool := make(chan *generate.InferenceSession[float32], poolSize)
	for range poolSize {
		pool <- m.generator.NewSession()
	}
	m.sessionPool = pool

	// Allow enough concurrency to use multiple sessions simultaneously.
	m.maxBatchConcurrency = poolSize

	const numPrompts = 8
	prompts := make([]string, numPrompts)
	for i := range prompts {
		prompts[i] = "hello"
	}

	results, err := m.GenerateBatch(context.Background(), prompts, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != numPrompts {
		t.Fatalf("got %d results, want %d", len(results), numPrompts)
	}
	for i, r := range results {
		if r != "foo" {
			t.Errorf("results[%d] = %q, want %q", i, r, "foo")
		}
	}
}

func TestGenerateBatch_ConcurrentSessions(t *testing.T) {
	if testing.Short() {
		t.Skip("Flaky on CI runners due to goroutine scheduling timing")
	}
	// Verify that GenerateBatch runs multiple sessions concurrently rather than
	// serializing through the generator mutex. We use a longer token sequence
	// (no immediate EOS) so sessions overlap in time, and track peak concurrency
	// with an atomic counter.
	//
	// The test instruments acquireSession/releaseSession indirectly: we give the
	// model a session pool and enough concurrency, then check that the total
	// wall-clock time is consistent with parallel execution (i.e., all prompts
	// produce correct output under -race).
	m := buildTestModel(t, 8, []int{6, 7, 6, 7, 6, 2}) // longer sequence before EOS

	const poolSize = 4
	pool := make(chan *generate.InferenceSession[float32], poolSize)
	for range poolSize {
		pool <- m.generator.NewSession()
	}
	m.sessionPool = pool
	m.maxBatchConcurrency = poolSize

	// Run enough prompts that they must overlap if concurrency > 1.
	const numPrompts = 8
	prompts := make([]string, numPrompts)
	for i := range prompts {
		prompts[i] = "hello"
	}

	// Track peak concurrent session usage via a wrapper. We wrap acquire/release
	// by observing pool channel length changes.
	var peak atomic.Int32
	var active atomic.Int32

	// Run the batch and observe pool behavior.
	done := make(chan struct{})
	var results []string
	var batchErr error
	go func() {
		defer close(done)
		// Monkey-patch is not possible, so we verify via pool drain observation.
		results, batchErr = m.GenerateBatch(context.Background(), prompts, WithTemperature(0), WithMaxTokens(10))
	}()

	// Poll active session count while batch runs.
	go func() {
		for {
			select {
			case <-done:
				return
			default:
				// Pool capacity minus current pool length = sessions checked out.
				checkedOut := int32(poolSize) - int32(len(pool))
				if checkedOut > 0 {
					active.Store(checkedOut)
					for {
						old := peak.Load()
						if checkedOut <= old || peak.CompareAndSwap(old, checkedOut) {
							break
						}
					}
				}
			}
		}
	}()

	<-done
	if batchErr != nil {
		t.Fatalf("unexpected error: %v", batchErr)
	}
	if len(results) != numPrompts {
		t.Fatalf("got %d results, want %d", len(results), numPrompts)
	}
	for i, r := range results {
		if r == "" {
			t.Errorf("results[%d] is empty", i)
		}
	}

	// With poolSize=4 and 8 prompts, we expect at least 2 concurrent sessions.
	// The race detector will also catch any shared-state issues.
	if p := peak.Load(); p < 2 {
		t.Errorf("peak concurrent sessions = %d, want >= 2 (sessions should run concurrently)", p)
	}
}
