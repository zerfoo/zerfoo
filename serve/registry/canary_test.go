package registry

import (
	"context"
	"math"
	"testing"
	"time"
)

const floatTol = 1e-9

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < floatTol
}

func TestCanaryStep(t *testing.T) {
	c := NewCanaryController(CanaryConfig{
		ModelID:          "llama-3-8b-v2",
		InitialWeight:    0.1,
		MaxWeight:        0.5,
		StepSize:         0.1,
		StepInterval:     time.Second, // not used in manual step
		SuccessThreshold: 0.9,
	})

	// No requests recorded — success rate is 0, step should not increase weight.
	c.Step()
	if got := c.CurrentWeight(); !approxEqual(got, 0.1) {
		t.Fatalf("expected weight 0.1 after step with no requests, got %f", got)
	}

	// Record enough successes to meet threshold.
	for i := 0; i < 10; i++ {
		c.RecordSuccess()
	}

	c.Step()
	if got := c.CurrentWeight(); !approxEqual(got, 0.2) {
		t.Fatalf("expected weight 0.2 after first step, got %f", got)
	}

	// Step again.
	c.Step()
	if got := c.CurrentWeight(); !approxEqual(got, 0.3) {
		t.Fatalf("expected weight 0.3 after second step, got %f", got)
	}

	// Step until max.
	c.Step() // 0.4
	c.Step() // 0.5
	c.Step() // should stay at 0.5
	if got := c.CurrentWeight(); !approxEqual(got, 0.5) {
		t.Fatalf("expected weight capped at 0.5, got %f", got)
	}
}

func TestCanaryPromote(t *testing.T) {
	c := NewCanaryController(CanaryConfig{
		ModelID:       "llama-3-8b-v2",
		InitialWeight: 0.1,
		MaxWeight:     0.5,
		StepSize:      0.1,
		StepInterval:  time.Second,
	})

	if err := c.Promote(); err != nil {
		t.Fatalf("Promote: %v", err)
	}
	if got := c.CurrentWeight(); got != 1.0 {
		t.Fatalf("expected weight 1.0 after Promote, got %f", got)
	}
}

func TestCanaryRollback(t *testing.T) {
	c := NewCanaryController(CanaryConfig{
		ModelID:       "llama-3-8b-v2",
		InitialWeight: 0.5,
		MaxWeight:     1.0,
		StepSize:      0.1,
		StepInterval:  time.Second,
	})

	if err := c.Rollback(); err != nil {
		t.Fatalf("Rollback: %v", err)
	}
	if got := c.CurrentWeight(); got != 0.0 {
		t.Fatalf("expected weight 0.0 after Rollback, got %f", got)
	}
}

func TestCanarySuccessRate(t *testing.T) {
	c := NewCanaryController(CanaryConfig{
		ModelID:      "llama-3-8b-v2",
		StepInterval: time.Second,
	})

	// No requests — rate is 0.
	if got := c.SuccessRate(); got != 0 {
		t.Fatalf("expected 0 with no requests, got %f", got)
	}

	// 7 successes, 3 failures → 0.7.
	for i := 0; i < 7; i++ {
		c.RecordSuccess()
	}
	for i := 0; i < 3; i++ {
		c.RecordFailure()
	}
	if got := c.SuccessRate(); got != 0.7 {
		t.Fatalf("expected 0.7, got %f", got)
	}
}

func TestCanaryAutoStep(t *testing.T) {
	c := NewCanaryController(CanaryConfig{
		ModelID:          "llama-3-8b-v2",
		InitialWeight:    0.0,
		MaxWeight:        1.0,
		StepSize:         0.1,
		StepInterval:     time.Millisecond,
		SuccessThreshold: 0.5,
	})

	// Seed enough successes so threshold is met.
	for i := 0; i < 10; i++ {
		c.RecordSuccess()
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := c.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Wait for a few steps to fire.
	time.Sleep(50 * time.Millisecond)

	if err := c.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	got := c.CurrentWeight()
	if got <= 0.0 {
		t.Fatalf("expected weight > 0 after auto-stepping, got %f", got)
	}
}
