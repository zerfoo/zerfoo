package cli

import (
	"context"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/shutdown"
)

func TestSignalContext_CancelOnSIGINT(t *testing.T) {
	coord := shutdown.New()

	var closed bool
	var mu sync.Mutex
	coord.Register(shutdownFunc(func(ctx context.Context) error {
		mu.Lock()
		closed = true
		mu.Unlock()
		return nil
	}))

	ctx, cancel := SignalContext(context.Background(), coord)
	defer cancel()

	// Send ourselves SIGINT.
	if err := syscall.Kill(syscall.Getpid(), syscall.SIGINT); err != nil {
		t.Fatalf("failed to send SIGINT: %v", err)
	}

	// Wait for context to be canceled.
	select {
	case <-ctx.Done():
		// expected
	case <-time.After(2 * time.Second):
		t.Fatal("context was not canceled after SIGINT")
	}

	mu.Lock()
	defer mu.Unlock()
	if !closed {
		t.Error("shutdown coordinator was not triggered")
	}
}

func TestSignalContext_NilCoordinator(t *testing.T) {
	ctx, cancel := SignalContext(context.Background(), nil)
	defer cancel()

	// Send ourselves SIGINT.
	if err := syscall.Kill(syscall.Getpid(), syscall.SIGINT); err != nil {
		t.Fatalf("failed to send SIGINT: %v", err)
	}

	select {
	case <-ctx.Done():
		// expected -- no panic with nil coordinator
	case <-time.After(2 * time.Second):
		t.Fatal("context was not canceled after SIGINT")
	}
}

func TestSignalContext_ParentCancel(t *testing.T) {
	parent, parentCancel := context.WithCancel(context.Background())
	ctx, cancel := SignalContext(parent, nil)
	defer cancel()

	parentCancel()

	select {
	case <-ctx.Done():
		// expected -- parent cancel propagates
	case <-time.After(2 * time.Second):
		t.Fatal("context was not canceled when parent was canceled")
	}
}

// shutdownFunc adapts a function to the shutdown.Closer interface.
type shutdownFunc func(ctx context.Context) error

func (f shutdownFunc) Close(ctx context.Context) error {
	return f(ctx)
}
