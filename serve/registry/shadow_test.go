package registry

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
)

func TestShadowRunnerSampling(t *testing.T) {
	runner := NewShadowRunner(ShadowConfig{
		ChampionID:   "champion-v1",
		ChallengerID: "challenger-v2",
		SampleRate:   1.0,
	})

	input := []float32{1.0, 2.0, 3.0}
	inferFn := func(modelID string, in []float32) ([]float32, error) {
		switch modelID {
		case "champion-v1":
			return []float32{10.0}, nil
		case "challenger-v2":
			return []float32{20.0}, nil
		default:
			return nil, errors.New("unknown model")
		}
	}

	result, err := runner.RunShadow(context.Background(), input, inferFn)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Sampled {
		t.Fatal("expected Sampled=true with SampleRate=1.0")
	}
	if len(result.ChampionOutput) != 1 || result.ChampionOutput[0] != 10.0 {
		t.Fatalf("unexpected champion output: %v", result.ChampionOutput)
	}
	if len(result.ChallengerOutput) != 1 || result.ChallengerOutput[0] != 20.0 {
		t.Fatalf("unexpected challenger output: %v", result.ChallengerOutput)
	}
}

func TestShadowRunnerNoSampling(t *testing.T) {
	runner := NewShadowRunner(ShadowConfig{
		ChampionID:   "champion-v1",
		ChallengerID: "challenger-v2",
		SampleRate:   0.0,
	})

	var challengerCalled atomic.Bool
	inferFn := func(modelID string, in []float32) ([]float32, error) {
		if modelID == "challenger-v2" {
			challengerCalled.Store(true)
		}
		return []float32{1.0}, nil
	}

	result, err := runner.RunShadow(context.Background(), []float32{1.0}, inferFn)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Sampled {
		t.Fatal("expected Sampled=false with SampleRate=0.0")
	}
	if challengerCalled.Load() {
		t.Fatal("challenger should not have been called with SampleRate=0.0")
	}
	if result.ChallengerOutput != nil {
		t.Fatalf("expected nil challenger output, got %v", result.ChallengerOutput)
	}
}

func TestShadowRunnerChallengerError(t *testing.T) {
	runner := NewShadowRunner(ShadowConfig{
		ChampionID:   "champion-v1",
		ChallengerID: "challenger-v2",
		SampleRate:   1.0,
	})

	inferFn := func(modelID string, in []float32) ([]float32, error) {
		if modelID == "challenger-v2" {
			return nil, errors.New("challenger failed")
		}
		return []float32{42.0}, nil
	}

	result, err := runner.RunShadow(context.Background(), []float32{1.0}, inferFn)
	if err != nil {
		t.Fatalf("unexpected error: champion should still succeed: %v", err)
	}
	if len(result.ChampionOutput) != 1 || result.ChampionOutput[0] != 42.0 {
		t.Fatalf("unexpected champion output: %v", result.ChampionOutput)
	}
	if result.ChallengerOutput != nil {
		t.Fatalf("expected nil challenger output on error, got %v", result.ChallengerOutput)
	}

	m := runner.Metrics()
	if m.ChallengerErrors != 1 {
		t.Fatalf("expected 1 challenger error, got %d", m.ChallengerErrors)
	}
}

func TestShadowMetrics(t *testing.T) {
	runner := NewShadowRunner(ShadowConfig{
		ChampionID:   "champion-v1",
		ChallengerID: "challenger-v2",
		SampleRate:   1.0,
	})

	successFn := func(modelID string, in []float32) ([]float32, error) {
		return []float32{1.0}, nil
	}
	errorFn := func(modelID string, in []float32) ([]float32, error) {
		if modelID == "challenger-v2" {
			return nil, errors.New("fail")
		}
		return []float32{1.0}, nil
	}

	ctx := context.Background()
	input := []float32{1.0}

	// Two successful runs.
	if _, err := runner.RunShadow(ctx, input, successFn); err != nil {
		t.Fatal(err)
	}
	if _, err := runner.RunShadow(ctx, input, successFn); err != nil {
		t.Fatal(err)
	}
	// One run with challenger error.
	if _, err := runner.RunShadow(ctx, input, errorFn); err != nil {
		t.Fatal(err)
	}

	m := runner.Metrics()
	if m.TotalRequests != 3 {
		t.Fatalf("expected TotalRequests=3, got %d", m.TotalRequests)
	}
	if m.SampledRequests != 3 {
		t.Fatalf("expected SampledRequests=3, got %d", m.SampledRequests)
	}
	if m.ChallengerErrors != 1 {
		t.Fatalf("expected ChallengerErrors=1, got %d", m.ChallengerErrors)
	}
}

func TestShadowRunnerContextCancelled(t *testing.T) {
	runner := NewShadowRunner(ShadowConfig{
		ChampionID:   "champion-v1",
		ChallengerID: "challenger-v2",
		SampleRate:   1.0,
	})

	ctx, cancel := context.WithCancel(context.Background())

	inferFn := func(modelID string, in []float32) ([]float32, error) {
		if modelID == "challenger-v2" {
			// Simulate a slow challenger — cancel before it returns.
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return []float32{5.0}, nil
	}

	// Cancel the context after champion completes but before challenger.
	go func() {
		// Give enough time for champion to finish and goroutine to launch.
		for runner.Metrics().SampledRequests == 0 {
			// spin until sampled
		}
		cancel()
	}()

	result, err := runner.RunShadow(ctx, []float32{1.0}, inferFn)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.ChampionOutput) != 1 || result.ChampionOutput[0] != 5.0 {
		t.Fatalf("unexpected champion output: %v", result.ChampionOutput)
	}
	// Challenger output should be nil due to context cancellation.
	if result.ChallengerOutput != nil {
		t.Fatalf("expected nil challenger output on ctx cancel, got %v", result.ChallengerOutput)
	}
}
