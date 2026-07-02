package registry

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

var (
	errCanaryAlreadyStarted = errors.New("canary: already started")
	errCanaryNotStarted     = errors.New("canary: not started")
)

// CanaryConfig configures a canary release rollout.
type CanaryConfig struct {
	// ModelID identifies the model version being canaried.
	ModelID string
	// InitialWeight is the starting traffic weight (0.0–1.0).
	InitialWeight float64
	// MaxWeight is the maximum weight before requiring explicit promotion.
	MaxWeight float64
	// StepSize is the weight increment applied each step.
	StepSize float64
	// StepInterval is the duration between automatic weight increases.
	StepInterval time.Duration
	// SuccessThreshold is the minimum success rate (0.0–1.0) required to step up.
	SuccessThreshold float64
}

// CanaryController manages gradual traffic ramp-up for a model version.
// It automatically increases the traffic weight when the observed success rate
// meets or exceeds the configured threshold.
type CanaryController struct {
	cfg CanaryConfig

	mu      sync.RWMutex
	weight  float64
	running bool
	cancel  context.CancelFunc
	done    chan struct{}

	successes atomic.Int64
	failures  atomic.Int64
}

// NewCanaryController creates a CanaryController with the given configuration.
func NewCanaryController(cfg CanaryConfig) *CanaryController {
	return &CanaryController{
		cfg:    cfg,
		weight: cfg.InitialWeight,
	}
}

// Start begins a background goroutine that periodically steps up the canary
// weight when the success rate meets the threshold.
func (c *CanaryController) Start(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.running {
		return errCanaryAlreadyStarted
	}

	ctx, cancel := context.WithCancel(ctx)
	c.cancel = cancel
	c.running = true
	c.done = make(chan struct{})

	go c.loop(ctx)
	return nil
}

// Stop halts the background stepping goroutine.
func (c *CanaryController) Stop() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.running {
		return errCanaryNotStarted
	}

	c.cancel()
	c.mu.Unlock()
	<-c.done
	c.mu.Lock()

	c.running = false
	c.cancel = nil
	return nil
}

// CurrentWeight returns the current traffic weight.
func (c *CanaryController) CurrentWeight() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.weight
}

// RecordSuccess increments the success counter (thread-safe).
func (c *CanaryController) RecordSuccess() {
	c.successes.Add(1)
}

// RecordFailure increments the failure counter (thread-safe).
func (c *CanaryController) RecordFailure() {
	c.failures.Add(1)
}

// Promote sets the canary weight to 1.0, directing all traffic to this version.
func (c *CanaryController) Promote() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.weight = 1.0
	return nil
}

// Rollback sets the canary weight to 0.0, removing all traffic from this version.
func (c *CanaryController) Rollback() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.weight = 0.0
	return nil
}

// SuccessRate returns the ratio of successes to total requests.
// Returns 0 if no requests have been recorded.
func (c *CanaryController) SuccessRate() float64 {
	s := c.successes.Load()
	f := c.failures.Load()
	total := s + f
	if total == 0 {
		return 0
	}
	return float64(s) / float64(total)
}

// Step performs a single canary step: if the success rate meets the threshold
// and weight is below max, it increases weight by StepSize (capped at MaxWeight).
func (c *CanaryController) Step() {
	if c.SuccessRate() < c.cfg.SuccessThreshold {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.weight >= c.cfg.MaxWeight {
		return
	}
	c.weight += c.cfg.StepSize
	if c.weight > c.cfg.MaxWeight {
		c.weight = c.cfg.MaxWeight
	}
}

func (c *CanaryController) loop(ctx context.Context) {
	defer close(c.done)
	ticker := time.NewTicker(c.cfg.StepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.Step()
		}
	}
}
