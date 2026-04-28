package marketplace

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestRetryFunc_SucceedsFirstAttempt(t *testing.T) {
	calls := 0
	err := RetryFunc(context.Background(), RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Millisecond,
		MaxJitter:   1 * time.Millisecond,
	}, "test.op", func() error {
		calls++
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 1 {
		t.Errorf("got %d calls, want 1", calls)
	}
}

func TestRetryFunc_SucceedsOnRetry(t *testing.T) {
	calls := 0
	err := RetryFunc(context.Background(), RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Millisecond,
		MaxJitter:   1 * time.Millisecond,
	}, "test.op", func() error {
		calls++
		if calls < 3 {
			return errors.New("transient error")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 3 {
		t.Errorf("got %d calls, want 3", calls)
	}
}

func TestRetryFunc_AllAttemptsFail(t *testing.T) {
	calls := 0
	testErr := errors.New("persistent error")
	err := RetryFunc(context.Background(), RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Millisecond,
		MaxJitter:   1 * time.Millisecond,
	}, "test.op", func() error {
		calls++
		return testErr
	})
	if !errors.Is(err, testErr) {
		t.Fatalf("got error %v, want %v", err, testErr)
	}
	if calls != 3 {
		t.Errorf("got %d calls, want 3", calls)
	}
}

func TestRetryFunc_RespectsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	err := RetryFunc(ctx, RetryConfig{
		MaxAttempts: 3,
		BaseDelay:   1 * time.Second,
		MaxJitter:   1 * time.Millisecond,
	}, "test.op", func() error {
		calls++
		cancel()
		return errors.New("fail")
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got error %v, want context.Canceled", err)
	}
	if calls != 1 {
		t.Errorf("got %d calls, want 1", calls)
	}
}

func TestRetryFunc_ZeroConfig(t *testing.T) {
	calls := 0
	err := RetryFunc(context.Background(), RetryConfig{}, "test.op", func() error {
		calls++
		return errors.New("should not be called")
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 0 {
		t.Errorf("got %d calls, want 0", calls)
	}
}
