// Package batcher implements a continuous batching scheduler for inference serving.
//
// Unlike fixed-batch schedulers that pad all sequences to the same length and
// wait for the entire batch to finish, a continuous batching scheduler:
//   - Assembles variable-length (ragged) batches each step with zero padding.
//   - Evicts completed sequences immediately, freeing the slot for a new request.
//   - Fills vacated slots from the pending queue without stalling active sequences.
//
// This approach typically achieves 2x+ throughput over fixed batching at the same
// concurrency because GPU cycles are never wasted on padding tokens and slots are
// never blocked by the slowest sequence in the batch.
package batcher

import (
	"context"
	"sort"
	"sync"
	"time"
)

// Request represents an incoming inference request.
type Request struct {
	ID     string // Caller-assigned identifier.
	Tokens []int  // Input token IDs (prompt).

	// MaxNewTokens is the maximum number of decode steps for this request.
	MaxNewTokens int
}

// Slot tracks the state of one active sequence inside the scheduler.
type Slot struct {
	Request       Request
	GeneratedToks []int // Tokens produced so far (decode output).
	Done          bool  // True once the sequence is finished.
}

// StepBatch is the ragged batch handed to the caller each step.
// It contains only active (non-done) slots with their actual token counts —
// no padding is ever added.
type StepBatch struct {
	Slots []*Slot
}

// TotalTokens returns the total number of tokens across all slots (prompt + generated).
// Because the batch is ragged this is a simple sum, not maxLen * batchSize.
func (b *StepBatch) TotalTokens() int {
	n := 0
	for _, s := range b.Slots {
		n += len(s.Request.Tokens) + len(s.GeneratedToks)
	}
	return n
}

// StepFunc is called once per decode step with the current ragged batch.
// The implementation should run one forward pass and append exactly one new
// token to each Slot.GeneratedToks. It must also set Slot.Done = true for
// any sequence that has finished (EOS or max tokens reached).
type StepFunc func(ctx context.Context, batch *StepBatch)

// CompletionResult is delivered to the caller when a request finishes.
type CompletionResult struct {
	RequestID string
	Tokens    []int // All generated tokens.
	Err       error
}

type pending struct {
	req    Request
	result chan CompletionResult
}

// Scheduler implements continuous batching.
type Scheduler struct {
	maxBatchSize int
	stepFn       StepFunc
	pollInterval time.Duration
	prefixCache  PrefixMatcher

	mu      sync.Mutex
	queue   []pending
	active  []*Slot
	results map[string]chan CompletionResult // requestID → result channel

	stop chan struct{}
	wg   sync.WaitGroup
}

// PrefixMatcher reports how many leading tokens of a sequence are already
// cached. The scheduler uses this to prioritize requests with high cache
// hit ratios (matchLen / totalPromptLen), reducing redundant prefill work.
type PrefixMatcher interface {
	// Match returns the number of leading tokens in the sequence that are
	// already present in the prefix cache.
	Match(tokens []int) (matchLen int, blockIDs []int)
}

// Option configures a Scheduler.
type Option func(*Scheduler)

// WithPollInterval sets how often the scheduler checks for new work when idle.
func WithPollInterval(d time.Duration) Option {
	return func(s *Scheduler) {
		s.pollInterval = d
	}
}

// WithPrefixCache enables cache-aware scheduling. When set, the scheduler
// sorts queued requests by prefix cache hit ratio (matchLen / promptLen)
// in descending order so that requests with the most cached prefix tokens
// are promoted to active slots first, reducing redundant prefill computation.
func WithPrefixCache(pm PrefixMatcher) Option {
	return func(s *Scheduler) {
		s.prefixCache = pm
	}
}

// New creates a continuous batching scheduler.
//
// maxBatchSize controls the maximum number of concurrent active slots.
// stepFn is invoked once per decode step with the current ragged batch.
func New(maxBatchSize int, stepFn StepFunc, opts ...Option) *Scheduler {
	if maxBatchSize <= 0 {
		maxBatchSize = 8
	}
	s := &Scheduler{
		maxBatchSize: maxBatchSize,
		stepFn:       stepFn,
		pollInterval: 1 * time.Millisecond,
		results:      make(map[string]chan CompletionResult),
		stop:         make(chan struct{}),
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// Start begins the scheduling loop in a background goroutine.
func (s *Scheduler) Start() {
	s.wg.Add(1)
	go s.loop()
}

// Stop gracefully shuts down the scheduler, draining active work.
func (s *Scheduler) Stop() {
	close(s.stop)
	s.wg.Wait()
}

// Submit enqueues a request and blocks until it completes or the context is canceled.
func (s *Scheduler) Submit(ctx context.Context, req Request) (CompletionResult, error) {
	ch := make(chan CompletionResult, 1)
	p := pending{req: req, result: ch}

	s.mu.Lock()
	s.queue = append(s.queue, p)
	s.results[req.ID] = ch
	s.mu.Unlock()

	select {
	case res := <-ch:
		return res, res.Err
	case <-ctx.Done():
		return CompletionResult{RequestID: req.ID, Err: ctx.Err()}, ctx.Err()
	case <-s.stop:
		return CompletionResult{RequestID: req.ID, Err: context.Canceled}, context.Canceled
	}
}

// loop is the main scheduling loop. Each iteration:
//  1. Evict completed slots and deliver results.
//  2. Fill vacated slots from the queue.
//  3. Run one decode step on the ragged batch.
func (s *Scheduler) loop() {
	defer s.wg.Done()

	for {
		select {
		case <-s.stop:
			s.drainActive(context.Canceled)
			return
		default:
		}

		s.mu.Lock()
		// 1. Evict done slots.
		s.evictCompleted()

		// 2. Fill from queue.
		s.fillSlots()

		if len(s.active) == 0 {
			s.mu.Unlock()
			// Nothing to do — poll for new work.
			select {
			case <-s.stop:
				return
			case <-time.After(s.pollInterval):
			}
			continue
		}

		// Build step batch (snapshot under lock).
		batch := &StepBatch{Slots: make([]*Slot, len(s.active))}
		copy(batch.Slots, s.active)
		s.mu.Unlock()

		// 3. Run one decode step.
		s.stepFn(context.Background(), batch)

		// After the step, check for newly completed sequences.
		s.mu.Lock()
		s.evictCompleted()
		s.mu.Unlock()
	}
}

// evictCompleted removes done slots and delivers results. Caller must hold s.mu.
func (s *Scheduler) evictCompleted() {
	alive := s.active[:0]
	for _, slot := range s.active {
		if slot.Done {
			if ch, ok := s.results[slot.Request.ID]; ok {
				ch <- CompletionResult{
					RequestID: slot.Request.ID,
					Tokens:    slot.GeneratedToks,
				}
				delete(s.results, slot.Request.ID)
			}
		} else {
			alive = append(alive, slot)
		}
	}
	s.active = alive
}

// fillSlots moves pending requests into active slots. When a PrefixMatcher
// is configured, the queue is sorted by prefix cache hit ratio (descending)
// so that requests whose prompts are mostly cached get scheduled first.
// Caller must hold s.mu.
func (s *Scheduler) fillSlots() {
	if s.prefixCache != nil && len(s.queue) > 1 {
		s.sortQueueByCacheHitRatio()
	}
	for len(s.active) < s.maxBatchSize && len(s.queue) > 0 {
		p := s.queue[0]
		s.queue = s.queue[1:]
		slot := &Slot{
			Request: p.req,
		}
		s.active = append(s.active, slot)
	}
}

// sortQueueByCacheHitRatio sorts the pending queue in descending order of
// prefix cache hit ratio (matchLen / len(tokens)). Requests with a higher
// fraction of cached prefix tokens are placed first. Caller must hold s.mu
// and s.prefixCache must be non-nil.
func (s *Scheduler) sortQueueByCacheHitRatio() {
	type scored struct {
		p     pending
		ratio float64
	}
	scored_ := make([]scored, len(s.queue))
	for i, p := range s.queue {
		promptLen := len(p.req.Tokens)
		if promptLen == 0 {
			scored_[i] = scored{p: p, ratio: 0}
			continue
		}
		matchLen, _ := s.prefixCache.Match(p.req.Tokens)
		scored_[i] = scored{p: p, ratio: float64(matchLen) / float64(promptLen)}
	}
	sort.SliceStable(scored_, func(i, j int) bool {
		return scored_[i].ratio > scored_[j].ratio
	})
	for i, sc := range scored_ {
		s.queue[i] = sc.p
	}
}

// drainActive cancels all active and queued requests with the given error.
func (s *Scheduler) drainActive(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, slot := range s.active {
		if ch, ok := s.results[slot.Request.ID]; ok {
			ch <- CompletionResult{RequestID: slot.Request.ID, Err: err}
			delete(s.results, slot.Request.ID)
		}
	}
	s.active = nil

	for _, p := range s.queue {
		p.result <- CompletionResult{RequestID: p.req.ID, Err: err}
	}
	s.queue = nil
}
