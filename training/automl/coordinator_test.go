package automl

import (
	"errors"
	"math"
	"testing"
)

// mockWorker implements Worker for testing.
type mockWorker struct {
	objective func(Config) (Metric, error)
	calls     int
}

func (m *mockWorker) RunTrial(config Config) (Metric, error) {
	m.calls++
	return m.objective(config)
}

func TestCoordinator(t *testing.T) {
	params := []HParam{
		{Name: "lr", Min: 0.001, Max: 1.0, IsLog: true},
		{Name: "batch_size", Min: 8, Max: 128},
	}

	worker := &mockWorker{
		objective: func(cfg Config) (Metric, error) {
			// Score is higher when lr is closer to 0.01 and batch_size closer to 32.
			lr := cfg.Params["lr"]
			bs := cfg.Params["batch_size"]
			score := -math.Pow(math.Log(lr)-math.Log(0.01), 2) - math.Pow((bs-32)/100, 2)
			return Metric{Score: score}, nil
		},
	}

	strategy := NewBayesianOptimizer(params, 42)
	coord, err := NewCoordinator(CoordinatorConfig{
		Space:     SearchSpace{Params: params},
		Strategy:  strategy,
		MaxTrials: 20,
	}, worker)
	if err != nil {
		t.Fatal(err)
	}

	best, err := coord.Run()
	if err != nil {
		t.Fatal(err)
	}

	if best.Err != nil {
		t.Errorf("best trial had error: %v", best.Err)
	}

	results := coord.Results()
	if len(results) != 20 {
		t.Errorf("expected 20 results, got %d", len(results))
	}

	if worker.calls != 20 {
		t.Errorf("expected 20 worker calls, got %d", worker.calls)
	}

	// Verify best is actually the best among results.
	for _, r := range results {
		if r.Err == nil && r.Metric.Score > best.Metric.Score {
			t.Errorf("found result with score %f > best score %f", r.Metric.Score, best.Metric.Score)
		}
	}

	// Verify Best() returns the same.
	bestFromMethod, ok := coord.Best()
	if !ok {
		t.Fatal("Best() returned ok=false")
	}
	if bestFromMethod.TrialID != best.TrialID {
		t.Errorf("Best() trial ID %d != Run() best trial ID %d", bestFromMethod.TrialID, best.TrialID)
	}
}

func TestCoordinatorEarlyStopping(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}

	callCount := 0
	worker := &mockWorker{
		objective: func(cfg Config) (Metric, error) {
			callCount++
			// Return a constant score so no improvement is ever made after first trial.
			return Metric{Score: 1.0}, nil
		},
	}

	strategy := NewBayesianOptimizer(params, 7)
	coord, err := NewCoordinator(CoordinatorConfig{
		Space:             SearchSpace{Params: params},
		Strategy:          strategy,
		MaxTrials:         100,
		EarlyStopPatience: 5,
	}, worker)
	if err != nil {
		t.Fatal(err)
	}

	best, err := coord.Run()
	if err != nil {
		t.Fatal(err)
	}

	// Should have stopped after 1 (improvement) + 5 (patience) = 6 trials.
	if callCount != 6 {
		t.Errorf("expected 6 trials with early stopping, got %d", callCount)
	}

	if best.Metric.Score != 1.0 {
		t.Errorf("expected best score 1.0, got %f", best.Metric.Score)
	}
}

func TestCoordinatorWorkerErrors(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}

	callCount := 0
	worker := &mockWorker{
		objective: func(cfg Config) (Metric, error) {
			callCount++
			if callCount <= 3 {
				return Metric{}, errors.New("simulated failure")
			}
			return Metric{Score: 0.5}, nil
		},
	}

	strategy := NewBayesianOptimizer(params, 11)
	coord, err := NewCoordinator(CoordinatorConfig{
		Space:     SearchSpace{Params: params},
		Strategy:  strategy,
		MaxTrials: 10,
	}, worker)
	if err != nil {
		t.Fatal(err)
	}

	best, err := coord.Run()
	if err != nil {
		t.Fatal(err)
	}

	results := coord.Results()
	if len(results) != 10 {
		t.Errorf("expected 10 results, got %d", len(results))
	}

	// First 3 should have errors.
	errCount := 0
	for _, r := range results {
		if r.Err != nil {
			errCount++
		}
	}
	if errCount != 3 {
		t.Errorf("expected 3 error results, got %d", errCount)
	}

	if best.Metric.Score != 0.5 {
		t.Errorf("expected best score 0.5, got %f", best.Metric.Score)
	}
}

func TestCoordinatorAllFailures(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}

	worker := &mockWorker{
		objective: func(cfg Config) (Metric, error) {
			return Metric{}, errors.New("always fails")
		},
	}

	strategy := NewBayesianOptimizer(params, 3)
	coord, err := NewCoordinator(CoordinatorConfig{
		Space:     SearchSpace{Params: params},
		Strategy:  strategy,
		MaxTrials: 5,
	}, worker)
	if err != nil {
		t.Fatal(err)
	}

	_, err = coord.Run()
	if err == nil {
		t.Error("expected error when all trials fail")
	}

	_, ok := coord.Best()
	if ok {
		t.Error("Best() should return false when all trials fail")
	}
}

func TestCoordinatorValidation(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}
	strategy := NewBayesianOptimizer(params, 1)
	worker := &mockWorker{objective: func(cfg Config) (Metric, error) {
		return Metric{Score: 1.0}, nil
	}}

	// nil worker
	_, err := NewCoordinator(CoordinatorConfig{
		Strategy:  strategy,
		MaxTrials: 10,
	}, nil)
	if err == nil {
		t.Error("expected error for nil worker")
	}

	// nil strategy
	_, err = NewCoordinator(CoordinatorConfig{
		MaxTrials: 10,
	}, worker)
	if err == nil {
		t.Error("expected error for nil strategy")
	}

	// zero max_trials
	_, err = NewCoordinator(CoordinatorConfig{
		Strategy:  strategy,
		MaxTrials: 0,
	}, worker)
	if err == nil {
		t.Error("expected error for zero max_trials")
	}
}

func TestCoordinatorDoubleRun(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}
	worker := &mockWorker{objective: func(cfg Config) (Metric, error) {
		return Metric{Score: 1.0}, nil
	}}

	coord, err := NewCoordinator(CoordinatorConfig{
		Space:     SearchSpace{Params: params},
		Strategy:  NewBayesianOptimizer(params, 1),
		MaxTrials: 3,
	}, worker)
	if err != nil {
		t.Fatal(err)
	}

	_, err = coord.Run()
	if err != nil {
		t.Fatal(err)
	}

	_, err = coord.Run()
	if err == nil {
		t.Error("expected error on second Run()")
	}
}

func TestCoordinatorRandomStrategy(t *testing.T) {
	params := []HParam{
		{Name: "lr", Min: 0.001, Max: 1.0, IsLog: true},
		{Name: "rank", Min: 4, Max: 64},
	}

	worker := &mockWorker{
		objective: func(cfg Config) (Metric, error) {
			return Metric{Score: cfg.Params["rank"]}, nil
		},
	}

	strategy := NewRandomStrategy(params, 99)
	coord, err := NewCoordinator(CoordinatorConfig{
		Space:     SearchSpace{Params: params},
		Strategy:  strategy,
		MaxTrials: 15,
	}, worker)
	if err != nil {
		t.Fatal(err)
	}

	best, err := coord.Run()
	if err != nil {
		t.Fatal(err)
	}

	// Best should have the highest rank value.
	for _, r := range coord.Results() {
		if r.Err == nil && r.Metric.Score > best.Metric.Score {
			t.Errorf("found result with score %f > best %f", r.Metric.Score, best.Metric.Score)
		}
	}
}

func TestCoordinatorString(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}
	worker := &mockWorker{objective: func(cfg Config) (Metric, error) {
		return Metric{Score: 0.42}, nil
	}}

	coord, err := NewCoordinator(CoordinatorConfig{
		Space:     SearchSpace{Params: params},
		Strategy:  NewBayesianOptimizer(params, 1),
		MaxTrials: 2,
	}, worker)
	if err != nil {
		t.Fatal(err)
	}

	s := coord.String()
	if s != "Coordinator: 0 trials, no best yet" {
		t.Errorf("unexpected initial String(): %s", s)
	}

	coord.Run()
	s = coord.String()
	if s == "Coordinator: 0 trials, no best yet" {
		t.Errorf("String() unchanged after Run()")
	}
}
