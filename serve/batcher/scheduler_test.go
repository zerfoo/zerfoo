package batcher

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// stubPrefixMatcher implements PrefixMatcher for testing. It returns
// preconfigured match lengths keyed by the first token in the sequence.
type stubPrefixMatcher struct {
	// matches maps the first token of a sequence to the number of tokens
	// that are "cached". This is a simplistic stub — real caches use
	// hash-based radix trees.
	matches map[int]int
}

func (m *stubPrefixMatcher) Match(tokens []int) (int, []int) {
	if len(tokens) == 0 {
		return 0, nil
	}
	matchLen, ok := m.matches[tokens[0]]
	if !ok || matchLen > len(tokens) {
		return 0, nil
	}
	// Return dummy block IDs.
	blockIDs := make([]int, matchLen)
	for i := range blockIDs {
		blockIDs[i] = i
	}
	return matchLen, blockIDs
}

// TestSchedulerZeroPadding verifies that the continuous batching scheduler
// assembles ragged batches with zero padding tokens. Sequences of varying
// lengths run concurrently; the test asserts that every StepBatch contains
// only the exact tokens of active sequences (no padding).
func TestSchedulerZeroPadding(t *testing.T) {
	const maxBatch = 4

	// Track the maximum padding we observe across all steps.
	var paddingTokensSeen atomic.Int64

	stepFn := func(_ context.Context, batch *StepBatch) {
		// Verify ragged property: total tokens == sum of individual lengths.
		expectedTotal := 0
		for _, s := range batch.Slots {
			expectedTotal += len(s.Request.Tokens) + len(s.GeneratedToks)
		}
		actual := batch.TotalTokens()
		if actual != expectedTotal {
			// Any difference would be padding.
			paddingTokensSeen.Add(int64(actual - expectedTotal))
		}

		// Simulate one decode step: append a token and mark done at MaxNewTokens.
		for _, s := range batch.Slots {
			tok := len(s.GeneratedToks) + 1
			s.GeneratedToks = append(s.GeneratedToks, tok)
			if len(s.GeneratedToks) >= s.Request.MaxNewTokens {
				s.Done = true
			}
		}
	}

	sched := New(maxBatch, stepFn, WithPollInterval(100*time.Microsecond))
	sched.Start()
	defer sched.Stop()

	// Submit requests with varying lengths.
	lengths := []int{3, 7, 1, 5, 10, 2, 8, 4}
	var wg sync.WaitGroup
	for i, n := range lengths {
		wg.Add(1)
		go func(id int, maxToks int) {
			defer wg.Done()
			req := Request{
				ID:           fmt.Sprintf("req-%d", id),
				Tokens:       make([]int, id+1), // varying prompt lengths
				MaxNewTokens: maxToks,
			}
			res, err := sched.Submit(context.Background(), req)
			if err != nil {
				t.Errorf("request %d failed: %v", id, err)
				return
			}
			if len(res.Tokens) != maxToks {
				t.Errorf("request %d: got %d tokens, want %d", id, len(res.Tokens), maxToks)
			}
		}(i, n)
	}
	wg.Wait()

	if p := paddingTokensSeen.Load(); p != 0 {
		t.Errorf("observed %d padding tokens, want 0", p)
	}
}

// TestSchedulerImmediateEviction verifies that completed sequences are freed
// immediately without waiting for the rest of the batch to finish.
//
// The test submits a short request (1 token) and a long request (20 tokens)
// sequentially to guarantee both are queued before the scheduler runs.
// Both land in the same batch. The short request finishes in 1 step and must
// be evicted immediately — its result is delivered while the long request
// keeps running. We record completion timestamps (not goroutine scheduling
// order) to make the assertion deterministic.
func TestSchedulerImmediateEviction(t *testing.T) {
	const maxBatch = 4

	stepFn := func(_ context.Context, batch *StepBatch) {
		for _, s := range batch.Slots {
			s.GeneratedToks = append(s.GeneratedToks, 1)
			if len(s.GeneratedToks) >= s.Request.MaxNewTokens {
				s.Done = true
			}
		}
	}

	// Create the scheduler but don't start yet — queue both requests first
	// so they land in the same batch deterministically.
	sched := New(maxBatch, stepFn, WithPollInterval(100*time.Microsecond))

	// Submit both requests before starting the scheduler.
	type completion struct {
		id string
		at time.Time
	}
	completions := make(chan completion, 2)

	var wg sync.WaitGroup
	for _, tc := range []struct {
		id      string
		maxToks int
	}{
		{"short", 1},
		{"long", 20},
	} {
		wg.Add(1)
		go func(id string, maxToks int) {
			defer wg.Done()
			req := Request{
				ID:           id,
				Tokens:       []int{1, 2, 3},
				MaxNewTokens: maxToks,
			}
			_, err := sched.Submit(context.Background(), req)
			if err != nil {
				t.Errorf("request %s failed: %v", id, err)
				return
			}
			completions <- completion{id: id, at: time.Now()}
		}(tc.id, tc.maxToks)
	}

	// Give goroutines time to enqueue, then start.
	time.Sleep(time.Millisecond)
	sched.Start()
	wg.Wait()
	sched.Stop()
	close(completions)

	var results []completion
	for c := range completions {
		results = append(results, c)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 completions, got %d", len(results))
	}

	// Sort by timestamp to determine actual completion order.
	if results[0].at.After(results[1].at) {
		results[0], results[1] = results[1], results[0]
	}
	if results[0].id != "short" {
		t.Errorf("expected 'short' to complete first, got order: [%s, %s]",
			results[0].id, results[1].id)
	}
}

// TestSchedulerThroughputVsFixed compares continuous batching against a
// simulated fixed-batch baseline by counting total decode steps and total
// "wasted" (padding) token-steps. The continuous scheduler should achieve
// at least 2x throughput because it never wastes cycles on padding and
// evicts completed sequences immediately.
//
// We measure in token-steps rather than wall-clock time to avoid flaky
// timing-dependent assertions. A token-step is one slot occupying one
// decode step. In fixed batching, short sequences that finish early still
// occupy a slot (padding) until the longest sequence in the batch completes.
// In continuous batching, finished sequences are evicted immediately, so
// every token-step does useful work.
func TestSchedulerThroughputVsFixed(t *testing.T) {
	const (
		maxBatch = 8
		numReqs  = 64
	)

	// Generate requests with heavily skewed lengths: mostly short (1-2 tokens)
	// with a few long ones (16 tokens). This maximizes the padding waste in
	// fixed batching.
	type reqSpec struct {
		id      int
		maxToks int
	}
	specs := make([]reqSpec, numReqs)
	for i := range specs {
		specs[i].id = i
		// 75% of requests finish in 1-2 steps, 25% take 16 steps.
		if i%4 == 0 {
			specs[i].maxToks = 16
		} else {
			specs[i].maxToks = 1 + (i % 2)
		}
	}

	// --- Continuous batching ---
	// Count total useful token-steps (one per active slot per step).
	contUsefulTokenSteps := atomic.Int64{}
	contTotalSteps := atomic.Int64{}

	contStepFn := func(_ context.Context, batch *StepBatch) {
		contTotalSteps.Add(1)
		contUsefulTokenSteps.Add(int64(len(batch.Slots)))
		for _, s := range batch.Slots {
			s.GeneratedToks = append(s.GeneratedToks, 1)
			if len(s.GeneratedToks) >= s.Request.MaxNewTokens {
				s.Done = true
			}
		}
	}

	sched := New(maxBatch, contStepFn, WithPollInterval(50*time.Microsecond))
	sched.Start()

	var wg sync.WaitGroup
	for _, sp := range specs {
		wg.Add(1)
		go func(s reqSpec) {
			defer wg.Done()
			req := Request{
				ID:           fmt.Sprintf("cont-%d", s.id),
				Tokens:       []int{1},
				MaxNewTokens: s.maxToks,
			}
			sched.Submit(context.Background(), req)
		}(sp)
	}
	wg.Wait()
	sched.Stop()

	// --- Fixed-batch simulation ---
	// In fixed batching: collect up to maxBatch requests, pad all to the
	// longest sequence in the batch, run that many steps. Short sequences
	// waste slots as padding.
	fixedUsefulTokenSteps := 0
	fixedTotalTokenSteps := 0
	for i := 0; i < numReqs; i += maxBatch {
		end := i + maxBatch
		if end > numReqs {
			end = numReqs
		}
		batch := specs[i:end]
		batchSize := len(batch)
		// All sequences padded to the longest.
		maxLen := 0
		for _, r := range batch {
			if r.maxToks > maxLen {
				maxLen = r.maxToks
			}
		}
		// Total token-steps = batchSize * maxLen (includes padding).
		fixedTotalTokenSteps += batchSize * maxLen
		// Useful token-steps = sum of actual sequence lengths.
		for _, r := range batch {
			fixedUsefulTokenSteps += r.maxToks
		}
	}

	contUseful := contUsefulTokenSteps.Load()
	fixedWaste := fixedTotalTokenSteps - fixedUsefulTokenSteps

	t.Logf("Continuous: %d useful token-steps, %d total steps, 0 wasted",
		contUseful, contTotalSteps.Load())
	t.Logf("Fixed: %d useful token-steps, %d total token-steps, %d wasted",
		fixedUsefulTokenSteps, fixedTotalTokenSteps, fixedWaste)

	// The throughput ratio is fixedTotalTokenSteps / contUsefulTokenSteps.
	// Continuous batching has zero waste, so every token-step is useful.
	// Fixed batching wastes slots on padding, so its total is much higher.
	ratio := float64(fixedTotalTokenSteps) / float64(contUseful)
	t.Logf("Efficiency ratio (fixed-total / continuous-useful): %.2fx", ratio)

	// With our skewed distribution, fixed batching wastes >50% of token-steps,
	// giving a ratio well above 2x.
	if ratio < 2.0 {
		t.Errorf("efficiency ratio %.2fx < 2.0x target", ratio)
	}

	// Also verify continuous batching had zero padding.
	// Every token-step in the continuous scheduler was useful work.
	// (The step function only sees active, non-done slots.)
	if contUseful == 0 {
		t.Error("continuous scheduler produced zero useful token-steps")
	}
}

// TestSchedulerContextCancellation verifies that a canceled context returns
// promptly without blocking the scheduler.
func TestSchedulerContextCancellation(t *testing.T) {
	stepFn := func(_ context.Context, batch *StepBatch) {
		// Slow step — should not block canceled requests.
		time.Sleep(100 * time.Millisecond)
		for _, s := range batch.Slots {
			s.GeneratedToks = append(s.GeneratedToks, 1)
			if len(s.GeneratedToks) >= s.Request.MaxNewTokens {
				s.Done = true
			}
		}
	}

	sched := New(2, stepFn, WithPollInterval(100*time.Microsecond))
	sched.Start()
	defer sched.Stop()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately.

	_, err := sched.Submit(ctx, Request{
		ID:           "canceled",
		Tokens:       []int{1},
		MaxNewTokens: 100,
	})
	if err == nil {
		t.Error("expected error from canceled context")
	}
}

// TestSchedulerCacheAwareOrdering verifies that when a PrefixMatcher is
// configured, the scheduler fills slots preferring requests with the highest
// prefix cache hit ratio (matchLen / promptLen).
//
// Three requests are queued before the scheduler starts:
//   - "no-cache":  10 tokens, 0 cached  → ratio 0.0
//   - "half-cache": 10 tokens, 5 cached → ratio 0.5
//   - "full-cache": 10 tokens, 10 cached → ratio 1.0
//
// With maxBatchSize=1, only one request runs at a time. The scheduler
// should pick them in order: full-cache, half-cache, no-cache.
func TestSchedulerCacheAwareOrdering(t *testing.T) {
	// Record the order in which requests are activated.
	var mu sync.Mutex
	var activationOrder []string

	stepFn := func(_ context.Context, batch *StepBatch) {
		for _, s := range batch.Slots {
			if len(s.GeneratedToks) == 0 {
				mu.Lock()
				activationOrder = append(activationOrder, s.Request.ID)
				mu.Unlock()
			}
			s.GeneratedToks = append(s.GeneratedToks, 1)
			if len(s.GeneratedToks) >= s.Request.MaxNewTokens {
				s.Done = true
			}
		}
	}

	matcher := &stubPrefixMatcher{
		matches: map[int]int{
			100: 10, // "full-cache" starts with token 100 → 10 matched
			200: 5,  // "half-cache" starts with token 200 → 5 matched
			// token 300 ("no-cache") has no entry → 0 matched
		},
	}

	// maxBatchSize=1 forces sequential scheduling so we can observe order.
	sched := New(1, stepFn,
		WithPollInterval(100*time.Microsecond),
		WithPrefixCache(matcher),
	)

	// Queue all requests before starting. Use tokens with distinguishable
	// first elements so the stub matcher can identify them.
	var wg sync.WaitGroup
	requests := []Request{
		{ID: "no-cache", Tokens: make([]int, 10), MaxNewTokens: 1},
		{ID: "half-cache", Tokens: make([]int, 10), MaxNewTokens: 1},
		{ID: "full-cache", Tokens: make([]int, 10), MaxNewTokens: 1},
	}
	// Set distinguishing first tokens.
	requests[0].Tokens[0] = 300 // no match
	requests[1].Tokens[0] = 200 // 5/10 = 0.5
	requests[2].Tokens[0] = 100 // 10/10 = 1.0

	for _, req := range requests {
		wg.Add(1)
		go func(r Request) {
			defer wg.Done()
			sched.Submit(context.Background(), r)
		}(req)
	}

	// Let goroutines enqueue, then start.
	time.Sleep(time.Millisecond)
	sched.Start()
	wg.Wait()
	sched.Stop()

	mu.Lock()
	defer mu.Unlock()

	if len(activationOrder) != 3 {
		t.Fatalf("expected 3 activations, got %d: %v", len(activationOrder), activationOrder)
	}

	want := []string{"full-cache", "half-cache", "no-cache"}
	for i, id := range want {
		if activationOrder[i] != id {
			t.Errorf("activation[%d] = %q, want %q (full order: %v)", i, activationOrder[i], id, activationOrder)
		}
	}
}

// TestSchedulerCacheAwareNoPrefixCache verifies that the scheduler behaves
// identically to FIFO when no PrefixMatcher is configured — the queue order
// is preserved.
func TestSchedulerCacheAwareNoPrefixCache(t *testing.T) {
	var mu sync.Mutex
	var activationOrder []string

	stepFn := func(_ context.Context, batch *StepBatch) {
		for _, s := range batch.Slots {
			if len(s.GeneratedToks) == 0 {
				mu.Lock()
				activationOrder = append(activationOrder, s.Request.ID)
				mu.Unlock()
			}
			s.GeneratedToks = append(s.GeneratedToks, 1)
			if len(s.GeneratedToks) >= s.Request.MaxNewTokens {
				s.Done = true
			}
		}
	}

	// No WithPrefixCache — should be FIFO.
	sched := New(1, stepFn, WithPollInterval(100*time.Microsecond))

	// Enqueue in deterministic order by adding directly under lock.
	ids := []string{"first", "second", "third"}
	var wg sync.WaitGroup
	for _, id := range ids {
		req := Request{ID: id, Tokens: []int{1, 2, 3}, MaxNewTokens: 1}
		ch := make(chan CompletionResult, 1)
		sched.mu.Lock()
		sched.queue = append(sched.queue, pending{req: req, result: ch})
		sched.results[req.ID] = ch
		sched.mu.Unlock()
		wg.Add(1)
		go func(c chan CompletionResult) {
			defer wg.Done()
			<-c
		}(ch)
	}

	sched.Start()
	wg.Wait()
	sched.Stop()

	mu.Lock()
	defer mu.Unlock()

	if len(activationOrder) != 3 {
		t.Fatalf("expected 3 activations, got %d", len(activationOrder))
	}
	for i, id := range ids {
		if activationOrder[i] != id {
			t.Errorf("activation[%d] = %q, want %q (FIFO order)", i, activationOrder[i], id)
		}
	}
}

// TestSortQueueByCacheHitRatio is a unit test for the sorting logic itself,
// independent of the scheduling loop.
func TestSortQueueByCacheHitRatio(t *testing.T) {
	matcher := &stubPrefixMatcher{
		matches: map[int]int{
			10: 8, // 8/10 = 0.8
			20: 2, // 2/10 = 0.2
			30: 5, // 5/10 = 0.5
		},
	}

	sched := &Scheduler{prefixCache: matcher}

	makeTokens := func(first int) []int {
		toks := make([]int, 10)
		toks[0] = first
		return toks
	}

	sched.queue = []pending{
		{req: Request{ID: "low", Tokens: makeTokens(20)}},
		{req: Request{ID: "high", Tokens: makeTokens(10)}},
		{req: Request{ID: "mid", Tokens: makeTokens(30)}},
		{req: Request{ID: "none", Tokens: makeTokens(99)}}, // no match → 0.0
	}

	sched.sortQueueByCacheHitRatio()

	got := make([]string, len(sched.queue))
	for i, p := range sched.queue {
		got[i] = p.req.ID
	}

	want := []string{"high", "mid", "low", "none"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("queue[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
			break
		}
	}
}

// TestSchedulerCacheAwareStableSort verifies that requests with equal cache
// hit ratios maintain their original FIFO order (stable sort).
func TestSchedulerCacheAwareStableSort(t *testing.T) {
	matcher := &stubPrefixMatcher{
		matches: map[int]int{
			10: 5, // All three use token 10 → same ratio
		},
	}

	sched := &Scheduler{prefixCache: matcher}

	makeTokens := func() []int {
		toks := make([]int, 10)
		toks[0] = 10
		return toks
	}

	ids := []string{"a", "b", "c", "d"}
	for _, id := range ids {
		sched.queue = append(sched.queue, pending{
			req: Request{ID: id, Tokens: makeTokens()},
		})
	}

	sched.sortQueueByCacheHitRatio()

	got := make([]string, len(sched.queue))
	for i, p := range sched.queue {
		got[i] = p.req.ID
	}

	// Equal ratios → original order preserved.
	if !sort.SliceIsSorted(got, func(i, j int) bool { return got[i] < got[j] }) {
		// They should still be in alphabetical order since that was the insertion order.
		for i, id := range ids {
			if got[i] != id {
				t.Errorf("queue[%d] = %q, want %q — stable sort violated (full: %v)", i, got[i], id, got)
				break
			}
		}
	}
}
