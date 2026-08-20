package inference

import (
	"context"
	"sync"
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

// TestGenerateBatch_ConcurrentSessions verifies the invariant batch generation
// actually depends on: every simultaneously in-flight prompt holds its OWN
// generate.InferenceSession, drawn from and returned to the model's session
// pool. Session isolation is what keeps one prompt's KV cache and position
// state out of another's; aliasing two live prompts onto one session, or
// bypassing the pool entirely, silently corrupts output.
//
// It deliberately does NOT assert that generations overlap in wall-clock time.
// InferenceSession.Generate holds the Generator's shared graph mutex for the
// whole call because *graph.Graph is not concurrency-safe (see CONC-H1 in
// speculative_concurrency_test.go), so exactly one generation is inside graph
// Forward at any instant no matter how many sessions are checked out. An
// earlier version of this test sampled len(sessionPool) from a spinning
// goroutine and required a "peak concurrent sessions >= 2" reading; that
// reading is a scheduling artifact of the moment sampling happened to land,
// not a property the API provides, and it failed ~70% of runs on main. See
// docs/lore.md L-0019.
func TestGenerateBatch_ConcurrentSessions(t *testing.T) {
	// Token 6 = "foo", token 2 = EOS: every generation is exactly two graph
	// Forward calls, so each result is independent of interleaving.
	m := buildTestModel(t, 8, []int{6, 2})

	const poolSize = 4
	pool := make(chan *generate.InferenceSession[float32], poolSize)
	for range poolSize {
		pool <- m.generator.NewSession()
	}
	m.sessionPool = pool
	m.maxBatchConcurrency = poolSize

	// Part 1: acquireSession never aliases. poolSize holders acquire before any
	// of them releases -- the arrival barrier makes that ordering deterministic
	// rather than a scheduling race -- so all poolSize sessions are live at once
	// and must be distinct.
	sessions := make([]*generate.InferenceSession[float32], poolSize)
	var arrived, finished sync.WaitGroup
	arrived.Add(poolSize)
	finished.Add(poolSize)
	release := make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	// A t.Fatalf below would abandon the holders on <-release; make their exit
	// path unconditional.
	t.Cleanup(func() {
		unblock()
		finished.Wait()
	})
	for i := range poolSize {
		go func(idx int) {
			defer finished.Done()
			sess := m.acquireSession()
			sessions[idx] = sess
			arrived.Done()
			<-release
			m.releaseSession(sess)
		}(i)
	}
	arrived.Wait()

	seen := make(map[*generate.InferenceSession[float32]]int, poolSize)
	for i, sess := range sessions {
		if sess == nil {
			t.Fatalf("holder %d acquired a nil session", i)
		}
		if prev, dup := seen[sess]; dup {
			t.Fatalf("acquireSession handed the same session to live holders %d and %d", prev, i)
		}
		seen[sess] = i
	}
	unblock()
	finished.Wait()

	if got := len(pool); got != poolSize {
		t.Fatalf("after releasing every holder, pool holds %d sessions, want %d", got, poolSize)
	}

	// Part 2: GenerateBatch must route every prompt through
	// acquireSession/releaseSession rather than driving the Generator directly.
	// Draining the pool first makes that a deterministic observation: a batch
	// that never touches the pool leaves it empty, a batch that releases
	// sessions refills it.
	for range poolSize {
		<-pool
	}
	if got := len(pool); got != 0 {
		t.Fatalf("pool not drained: %d sessions left", got)
	}

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
	// Cross-session state contamination shows up here as a wrong or empty
	// string; -race additionally catches unsynchronized access to the shared
	// graph node.
	for i, r := range results {
		if r != "foo" {
			t.Errorf("results[%d] = %q, want %q", i, r, "foo")
		}
	}
	if len(pool) == 0 {
		t.Error("session pool still empty after GenerateBatch: no session was returned via releaseSession, so the batch bypassed the session pool")
	}
}
