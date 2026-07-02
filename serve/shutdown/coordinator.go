// Package shutdown provides orderly shutdown coordination using context
// cancellation and cleanup callbacks.
//
// Register Closer instances in initialization order. On Shutdown, they are
// closed in reverse registration order. Each Closer receives the parent
// context so it can respect deadlines. Shutdown is idempotent.
package shutdown

import (
	"context"
	"fmt"
	"sync"
)

// Closer is implemented by any component that needs cleanup on shutdown.
type Closer interface {
	// Close performs cleanup. It should respect context cancellation for
	// deadline enforcement.
	Close(ctx context.Context) error
}

// Coordinator manages orderly shutdown of registered Closer instances.
type Coordinator struct {
	mu      sync.Mutex
	closers []Closer
	done    bool
}

// New creates a new Coordinator.
func New() *Coordinator {
	return &Coordinator{}
}

// Register adds a Closer to the coordinator. On Shutdown, closers are
// called in reverse registration order.
func (c *Coordinator) Register(cl Closer) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.done {
		c.closers = append(c.closers, cl)
	}
}

// Shutdown closes all registered Closer instances in reverse order.
// It collects and returns all errors. If the context is canceled,
// remaining closers still receive the canceled context so they can
// choose to abort quickly. Shutdown is idempotent — subsequent calls
// return nil immediately.
func (c *Coordinator) Shutdown(ctx context.Context) []error {
	c.mu.Lock()
	if c.done {
		c.mu.Unlock()
		return nil
	}
	c.done = true
	// Copy to release lock during close calls.
	closers := make([]Closer, len(c.closers))
	copy(closers, c.closers)
	c.mu.Unlock()

	var errs []error
	for i := len(closers) - 1; i >= 0; i-- {
		if err := closers[i].Close(ctx); err != nil {
			errs = append(errs, fmt.Errorf("shutdown closer %d: %w", i, err))
		}
	}

	return errs
}
