package cli

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/zerfoo/zerfoo/shutdown"
)

// SignalContext returns a context that is canceled when SIGINT or SIGTERM
// is received. If a non-nil shutdown.Coordinator is provided, its Shutdown
// method is called before the context is canceled. The returned cancel
// function should be deferred by the caller to release signal resources.
func SignalContext(parent context.Context, coord *shutdown.Coordinator) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(parent)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		select {
		case <-sigCh:
			if coord != nil {
				_ = coord.Shutdown(ctx)
			}
			cancel()
		case <-ctx.Done():
		}
		signal.Stop(sigCh)
	}()

	return ctx, cancel
}
