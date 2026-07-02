package marketplace

import (
	"context"
	"log/slog"
	"math/rand/v2"
	"time"
)

// RetryConfig controls exponential backoff retry behavior for metering calls.
type RetryConfig struct {
	// MaxAttempts is the total number of attempts (including the first).
	MaxAttempts int
	// BaseDelay is the initial delay before the first retry.
	BaseDelay time.Duration
	// MaxJitter is the maximum random jitter added to each delay.
	MaxJitter time.Duration
}

// DefaultRetryConfig returns the standard retry configuration for marketplace
// metering: 3 attempts with 1s base delay and 500ms max jitter.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Second,
		MaxJitter:   500 * time.Millisecond,
	}
}

// RetryFunc executes fn with exponential backoff retry. It logs each retry
// attempt at warn level and logs at error level if all attempts fail.
// The operation parameter is used in log messages to identify the call.
func RetryFunc(ctx context.Context, cfg RetryConfig, operation string, fn func() error) error {
	var lastErr error
	for attempt := 0; attempt < cfg.MaxAttempts; attempt++ {
		lastErr = fn()
		if lastErr == nil {
			return nil
		}

		if attempt < cfg.MaxAttempts-1 {
			delay := cfg.BaseDelay << uint(attempt)
			jitter := time.Duration(rand.Int64N(int64(cfg.MaxJitter)))
			delay += jitter

			slog.Warn("marketplace metering retry",
				"operation", operation,
				"attempt", attempt+1,
				"max_attempts", cfg.MaxAttempts,
				"error", lastErr,
				"next_delay", delay,
			)

			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}
	}

	slog.Error("marketplace metering failed after all retries",
		"operation", operation,
		"attempts", cfg.MaxAttempts,
		"error", lastErr,
	)
	return lastErr
}
