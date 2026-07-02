package serve

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// BatchRequest represents a single inference request in a batch.
type BatchRequest struct {
	Prompt string
	Phase  string // "prefill" or "decode"
}

// BatchResult holds the result for a single request in a batch.
type BatchResult struct {
	Value string
	Err   error
}

// BatchHandler processes a batch of requests and returns results.
// The results slice must have the same length as the requests slice.
type BatchHandler func(ctx context.Context, reqs []BatchRequest) []BatchResult

// BatchConfig configures the batch scheduler.
type BatchConfig struct {
	MaxBatchSize int
	BatchTimeout time.Duration
	Handler      BatchHandler
}

type pendingRequest struct {
	req    BatchRequest
	ctx    context.Context
	result chan BatchResult
}

// BatchScheduler collects incoming requests into batches for efficient processing.
type BatchScheduler struct {
	config  BatchConfig
	pending chan pendingRequest
	stop    chan struct{}
	wg      sync.WaitGroup
}

// NewBatchScheduler creates a new batch scheduler.
func NewBatchScheduler(config BatchConfig) *BatchScheduler {
	if config.MaxBatchSize <= 0 {
		config.MaxBatchSize = 8
	}
	if config.BatchTimeout <= 0 {
		config.BatchTimeout = 10 * time.Millisecond
	}
	return &BatchScheduler{
		config:  config,
		pending: make(chan pendingRequest, config.MaxBatchSize*4),
		stop:    make(chan struct{}),
	}
}

// Start begins the batch collection loop.
func (s *BatchScheduler) Start() {
	s.wg.Add(1)
	go s.loop()
}

// Stop gracefully shuts down the scheduler.
func (s *BatchScheduler) Stop() {
	close(s.stop)
	s.wg.Wait()
}

// Submit adds a request to the next batch and waits for the result.
func (s *BatchScheduler) Submit(ctx context.Context, req BatchRequest) (BatchResult, error) {
	if ctx.Err() != nil {
		return BatchResult{}, ctx.Err()
	}

	pr := pendingRequest{
		req:    req,
		ctx:    ctx,
		result: make(chan BatchResult, 1),
	}

	select {
	case s.pending <- pr:
	case <-ctx.Done():
		return BatchResult{}, ctx.Err()
	case <-s.stop:
		return BatchResult{}, context.Canceled
	}

	select {
	case r := <-pr.result:
		return r, r.Err
	case <-ctx.Done():
		return BatchResult{}, ctx.Err()
	}
}

func (s *BatchScheduler) loop() {
	defer s.wg.Done()

	for {
		// Wait for the first request.
		var batch []pendingRequest
		select {
		case pr := <-s.pending:
			batch = append(batch, pr)
		case <-s.stop:
			return
		}

		// Collect more requests up to max batch size or timeout.
		timer := time.NewTimer(s.config.BatchTimeout)
	collect:
		for len(batch) < s.config.MaxBatchSize {
			select {
			case pr := <-s.pending:
				batch = append(batch, pr)
			case <-timer.C:
				break collect
			case <-s.stop:
				timer.Stop()
				// Drain remaining with error.
				for _, pr := range batch {
					pr.result <- BatchResult{Err: context.Canceled}
				}
				return
			}
		}
		timer.Stop()

		// Execute the batch.
		s.executeBatch(batch)
	}
}

func (s *BatchScheduler) executeBatch(batch []pendingRequest) {
	// Filter out canceled requests.
	var live []pendingRequest
	for _, pr := range batch {
		if pr.ctx.Err() == nil {
			live = append(live, pr)
		} else {
			pr.result <- BatchResult{Err: pr.ctx.Err()}
		}
	}

	if len(live) == 0 {
		return
	}

	// Build request slice.
	reqs := make([]BatchRequest, len(live))
	for i, pr := range live {
		reqs[i] = pr.req
	}

	// Build a merged context that cancels only when ALL live requests cancel.
	batchCtx, batchCancel := context.WithCancel(context.Background())
	var once sync.Once
	var remaining atomic.Int32
	remaining.Store(int32(len(live)))
	for _, pr := range live {
		go func(ctx context.Context) {
			<-ctx.Done()
			if remaining.Add(-1) <= 0 {
				once.Do(batchCancel)
			}
		}(pr.ctx)
	}

	results := s.config.Handler(batchCtx, reqs)
	once.Do(batchCancel) // ensure merged context is always cleaned up

	// Deliver results.
	for i, pr := range live {
		if i < len(results) {
			pr.result <- results[i]
		} else {
			pr.result <- BatchResult{Err: context.Canceled}
		}
	}
}
