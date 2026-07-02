package shutdown

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// mockCloser records whether Close was called and its order.
type mockCloser struct {
	name    string
	order   *[]string
	mu      *sync.Mutex
	delay   time.Duration
	err     error
	closeCh chan struct{} // optional: blocks until closed
}

func (m *mockCloser) Close(ctx context.Context) error {
	if m.closeCh != nil {
		select {
		case <-m.closeCh:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	m.mu.Lock()
	*m.order = append(*m.order, m.name)
	m.mu.Unlock()
	return m.err
}

func newMock(name string, order *[]string, mu *sync.Mutex) *mockCloser {
	return &mockCloser{name: name, order: order, mu: mu}
}

func TestCoordinator_EmptyShutdown(t *testing.T) {
	c := New()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	errs := c.Shutdown(ctx)
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %v", errs)
	}
}

func TestCoordinator_ReverseOrder(t *testing.T) {
	var order []string
	var mu sync.Mutex
	c := New()

	c.Register(newMock("first", &order, &mu))
	c.Register(newMock("second", &order, &mu))
	c.Register(newMock("third", &order, &mu))

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	errs := c.Shutdown(ctx)
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %v", errs)
	}

	want := []string{"third", "second", "first"}
	if len(order) != len(want) {
		t.Fatalf("order = %v, want %v", order, want)
	}
	for i, name := range want {
		if order[i] != name {
			t.Errorf("order[%d] = %q, want %q", i, order[i], name)
		}
	}
}

func TestCoordinator_TimeoutSkipsSlowCloser(t *testing.T) {
	var order []string
	var mu sync.Mutex
	c := New()

	c.Register(newMock("first", &order, &mu))

	slow := &mockCloser{
		name:  "slow",
		order: &order,
		mu:    &mu,
		delay: 5 * time.Second,
	}
	c.Register(slow)

	c.Register(newMock("third", &order, &mu))

	// Short timeout — slow closer should be skipped.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	errs := c.Shutdown(ctx)

	// Should have at least one error (the slow closer timing out).
	hasTimeout := false
	for _, err := range errs {
		if errors.Is(err, context.DeadlineExceeded) {
			hasTimeout = true
		}
	}
	if !hasTimeout {
		t.Errorf("expected timeout error, got %v", errs)
	}
}

func TestCoordinator_CloserError(t *testing.T) {
	var order []string
	var mu sync.Mutex
	c := New()

	errBoom := errors.New("boom")

	c.Register(newMock("first", &order, &mu))

	failing := &mockCloser{
		name:  "failing",
		order: &order,
		mu:    &mu,
		err:   errBoom,
	}
	c.Register(failing)

	c.Register(newMock("third", &order, &mu))

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	errs := c.Shutdown(ctx)

	// Should report the error but still close others.
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if !errors.Is(errs[0], errBoom) {
		t.Errorf("expected boom error, got %v", errs[0])
	}

	// All three should have been called (reverse order).
	want := []string{"third", "failing", "first"}
	if len(order) != len(want) {
		t.Fatalf("order = %v, want %v", order, want)
	}
	for i, name := range want {
		if order[i] != name {
			t.Errorf("order[%d] = %q, want %q", i, order[i], name)
		}
	}
}

func TestCoordinator_ShutdownIdempotent(t *testing.T) {
	var order []string
	var mu sync.Mutex
	c := New()
	c.Register(newMock("only", &order, &mu))

	ctx := context.Background()

	errs1 := c.Shutdown(ctx)
	errs2 := c.Shutdown(ctx)

	if len(errs1) != 0 {
		t.Errorf("first shutdown: expected 0 errors, got %v", errs1)
	}
	if len(errs2) != 0 {
		t.Errorf("second shutdown: expected 0 errors, got %v", errs2)
	}

	// Close should only be called once.
	if len(order) != 1 {
		t.Errorf("expected 1 close call, got %d: %v", len(order), order)
	}
}

func TestCoordinator_ConcurrentRegisterAndShutdown(t *testing.T) {
	var order []string
	var mu sync.Mutex
	c := New()

	// Pre-register some closers.
	for i := 0; i < 5; i++ {
		c.Register(newMock("pre", &order, &mu))
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	// Concurrent registration should not race with shutdown.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		c.Shutdown(ctx)
	}()

	// Try to register during shutdown — should not panic.
	c.Register(newMock("late", &order, &mu))
	wg.Wait()
}
