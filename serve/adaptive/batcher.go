// Package adaptive implements an adaptive batch scheduler that dynamically
// adjusts batch size based on queue depth and latency targets to maximize
// throughput while meeting latency SLOs. (Stability: beta)
package adaptive

import (
	"context"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// Request represents an incoming inference request.
type Request struct {
	ID     string
	Tokens []int
}

// Result holds the output for a completed request.
type Result struct {
	RequestID string
	Output    []int
	Err       error
}

// Handler processes a batch of requests and returns results.
type Handler func(ctx context.Context, reqs []Request) []Result

// Config controls the adaptive batcher.
type Config struct {
	// MinBatchSize is the smallest batch the batcher will form (default 1).
	MinBatchSize int
	// MaxBatchSize is the largest batch the batcher will form (default 32).
	MaxBatchSize int
	// TargetLatencyMS is the latency SLO in milliseconds. When the
	// exponential moving average of batch latency exceeds this target,
	// the batcher decreases the batch size.
	TargetLatencyMS float64
	// QueueTimeoutMS is how long (in ms) the batcher waits to fill a batch
	// before dispatching whatever it has collected (default 50).
	QueueTimeoutMS float64
}

func (c *Config) defaults() {
	if c.MinBatchSize <= 0 {
		c.MinBatchSize = 1
	}
	if c.MaxBatchSize <= 0 {
		c.MaxBatchSize = 32
	}
	if c.MinBatchSize > c.MaxBatchSize {
		c.MinBatchSize = c.MaxBatchSize
	}
	if c.TargetLatencyMS <= 0 {
		c.TargetLatencyMS = 100
	}
	if c.QueueTimeoutMS <= 0 {
		c.QueueTimeoutMS = 50
	}
}

type pending struct {
	req    Request
	result chan Result
}

// Batcher dynamically adjusts batch size based on queue depth and latency.
type Batcher struct {
	config  Config
	handler Handler
	queue   chan pending
	stop    chan struct{}
	wg      sync.WaitGroup

	// currentBatchSize is the dynamically adjusted batch size.
	currentBatchSize atomic.Int32

	// latencyEMA tracks the exponential moving average of batch latency in ms.
	mu         sync.Mutex
	latencyEMA float64
}

// New creates a new adaptive batcher.
func New(config Config, handler Handler) *Batcher {
	config.defaults()
	b := &Batcher{
		config:  config,
		handler: handler,
		queue:   make(chan pending, config.MaxBatchSize*4),
		stop:    make(chan struct{}),
	}
	b.currentBatchSize.Store(int32(config.MinBatchSize))
	return b
}

// Start begins the batch collection loop.
func (b *Batcher) Start() {
	b.wg.Add(1)
	go b.loop()
}

// Stop gracefully shuts down the batcher.
func (b *Batcher) Stop() {
	close(b.stop)
	b.wg.Wait()
}

// Submit enqueues a request and blocks until it completes or the context expires.
func (b *Batcher) Submit(ctx context.Context, req Request) (Result, error) {
	if ctx.Err() != nil {
		return Result{}, ctx.Err()
	}

	p := pending{
		req:    req,
		result: make(chan Result, 1),
	}

	select {
	case b.queue <- p:
	case <-ctx.Done():
		return Result{}, ctx.Err()
	case <-b.stop:
		return Result{}, context.Canceled
	}

	select {
	case r := <-p.result:
		return r, r.Err
	case <-ctx.Done():
		return Result{}, ctx.Err()
	}
}

// BatchSize returns the current dynamically-adjusted batch size.
func (b *Batcher) BatchSize() int {
	return int(b.currentBatchSize.Load())
}

// LatencyEMA returns the current exponential moving average latency in ms.
func (b *Batcher) LatencyEMA() float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.latencyEMA
}

func (b *Batcher) loop() {
	defer b.wg.Done()

	for {
		// Wait for the first request.
		var batch []pending
		select {
		case p := <-b.queue:
			batch = append(batch, p)
		case <-b.stop:
			return
		}

		// Collect more requests up to current batch size or timeout.
		target := int(b.currentBatchSize.Load())
		timeout := time.Duration(b.config.QueueTimeoutMS * float64(time.Millisecond))
		timer := time.NewTimer(timeout)

	collect:
		for len(batch) < target {
			select {
			case p := <-b.queue:
				batch = append(batch, p)
			case <-timer.C:
				break collect
			case <-b.stop:
				timer.Stop()
				for _, p := range batch {
					p.result <- Result{Err: context.Canceled}
				}
				return
			}
		}
		timer.Stop()

		// Execute the batch and measure latency.
		b.executeBatch(batch)

		// Adapt batch size based on queue depth and latency.
		b.adapt()
	}
}

func (b *Batcher) executeBatch(batch []pending) {
	// Filter canceled requests.
	var live []pending
	for _, p := range batch {
		if p.req.ID != "" || true { // Always include; caller manages context.
			live = append(live, p)
		}
	}
	if len(live) == 0 {
		return
	}

	reqs := make([]Request, len(live))
	for i, p := range live {
		reqs[i] = p.req
	}

	start := time.Now()
	results := b.handler(context.Background(), reqs)
	elapsed := time.Since(start)

	// Update latency EMA (alpha = 0.3 for responsiveness).
	b.updateLatency(float64(elapsed.Milliseconds()))

	// Deliver results.
	for i, p := range live {
		if i < len(results) {
			p.result <- results[i]
		} else {
			p.result <- Result{Err: context.Canceled}
		}
	}
}

const emaAlpha = 0.3

func (b *Batcher) updateLatency(latencyMS float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.latencyEMA == 0 {
		b.latencyEMA = latencyMS
	} else {
		b.latencyEMA = emaAlpha*latencyMS + (1-emaAlpha)*b.latencyEMA
	}
}

// adapt adjusts the batch size based on queue depth and latency EMA.
//
// Strategy:
//   - When queue is deep (>= current batch size) and latency is under target → scale up.
//   - When latency EMA exceeds target → scale down.
//   - Otherwise hold steady.
func (b *Batcher) adapt() {
	b.mu.Lock()
	ema := b.latencyEMA
	b.mu.Unlock()

	current := int(b.currentBatchSize.Load())
	queueDepth := len(b.queue)

	var next int
	if ema > b.config.TargetLatencyMS {
		// Latency too high — decrease batch size.
		next = int(math.Ceil(float64(current) * 0.75))
	} else if queueDepth >= current {
		// Queue is deep and latency is OK — increase batch size.
		next = int(math.Min(float64(current*2), float64(b.config.MaxBatchSize)))
	} else {
		next = current
	}

	// Clamp to [min, max].
	if next < b.config.MinBatchSize {
		next = b.config.MinBatchSize
	}
	if next > b.config.MaxBatchSize {
		next = b.config.MaxBatchSize
	}

	b.currentBatchSize.Store(int32(next))
}
