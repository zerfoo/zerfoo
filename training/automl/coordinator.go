package automl

import (
	"errors"
	"fmt"
	"sync"
)

// Config holds a hyperparameter configuration for a single trial.
type Config struct {
	Params map[string]float64
}

// Metric holds the result of evaluating a trial configuration.
type Metric struct {
	Score float64
}

// Worker executes a trial with a given configuration and returns the metric.
type Worker interface {
	RunTrial(config Config) (Metric, error)
}

// Strategy suggests and reports trials. Both BayesianOptimizer and random
// search implement this interface.
type Strategy interface {
	Suggest() (int, map[string]float64)
	Report(trialID int, score float64) error
	BestTrial() (Trial, bool)
}

// SearchSpace defines the hyperparameter ranges to explore.
type SearchSpace struct {
	Params []HParam
}

// CoordinatorConfig configures the AutoML loop coordinator.
type CoordinatorConfig struct {
	// Space defines the hyperparameter search space.
	Space SearchSpace

	// Strategy is the search strategy (Bayesian, random, etc.).
	Strategy Strategy

	// MaxTrials is the maximum number of trials to run.
	MaxTrials int

	// EarlyStopPatience is the number of trials without improvement
	// before stopping early. Zero disables early stopping.
	EarlyStopPatience int
}

// TrialResult records the outcome of a single trial.
type TrialResult struct {
	TrialID int
	Config  Config
	Metric  Metric
	Err     error
}

// Coordinator orchestrates the AutoML search loop, dispatching trials
// to a Worker and collecting results.
type Coordinator struct {
	mu       sync.Mutex
	config   CoordinatorConfig
	worker   Worker
	results  []TrialResult
	best     *TrialResult
	finished bool
}

// NewCoordinator creates a coordinator with the given configuration and worker.
func NewCoordinator(config CoordinatorConfig, worker Worker) (*Coordinator, error) {
	if worker == nil {
		return nil, errors.New("automl: worker must not be nil")
	}
	if config.Strategy == nil {
		return nil, errors.New("automl: strategy must not be nil")
	}
	if config.MaxTrials <= 0 {
		return nil, errors.New("automl: max_trials must be positive")
	}

	return &Coordinator{
		config:  config,
		worker:  worker,
		results: make([]TrialResult, 0, config.MaxTrials),
	}, nil
}

// Run executes the AutoML loop, running up to MaxTrials trials.
// It returns the best trial result found, or an error if no trials succeeded.
func (c *Coordinator) Run() (TrialResult, error) {
	c.mu.Lock()
	if c.finished {
		c.mu.Unlock()
		return TrialResult{}, errors.New("automl: coordinator already ran")
	}
	c.mu.Unlock()

	sinceImproved := 0

	for i := 0; i < c.config.MaxTrials; i++ {
		trialID, params := c.config.Strategy.Suggest()

		cfg := Config{Params: params}
		metric, err := c.worker.RunTrial(cfg)

		result := TrialResult{
			TrialID: trialID,
			Config:  cfg,
			Metric:  metric,
			Err:     err,
		}

		c.mu.Lock()
		c.results = append(c.results, result)
		c.mu.Unlock()

		if err != nil {
			continue
		}

		// Report score to strategy.
		if reportErr := c.config.Strategy.Report(trialID, metric.Score); reportErr != nil {
			continue
		}

		c.mu.Lock()
		improved := false
		if c.best == nil || metric.Score > c.best.Metric.Score {
			c.best = &result
			improved = true
		}
		c.mu.Unlock()

		if improved {
			sinceImproved = 0
		} else {
			sinceImproved++
		}

		// Early stopping check.
		if c.config.EarlyStopPatience > 0 && sinceImproved >= c.config.EarlyStopPatience {
			break
		}
	}

	c.mu.Lock()
	c.finished = true
	best := c.best
	c.mu.Unlock()

	if best == nil {
		return TrialResult{}, errors.New("automl: no successful trials")
	}
	return *best, nil
}

// Results returns all trial results collected so far.
func (c *Coordinator) Results() []TrialResult {
	c.mu.Lock()
	defer c.mu.Unlock()

	out := make([]TrialResult, len(c.results))
	copy(out, c.results)
	return out
}

// Best returns the best trial result, or false if none succeeded.
func (c *Coordinator) Best() (TrialResult, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.best == nil {
		return TrialResult{}, false
	}
	return *c.best, true
}

// RandomStrategy implements a simple random search strategy.
type RandomStrategy struct {
	bo *BayesianOptimizer
}

// NewRandomStrategy creates a random search strategy (uses BayesianOptimizer
// with forced exploration by never exceeding the exploration threshold).
func NewRandomStrategy(params []HParam, seed int64) *RandomStrategy {
	return &RandomStrategy{
		bo: NewBayesianOptimizer(params, seed),
	}
}

// Suggest returns the next random parameter configuration.
func (rs *RandomStrategy) Suggest() (int, map[string]float64) {
	return rs.bo.Suggest()
}

// Report records a trial score.
func (rs *RandomStrategy) Report(trialID int, score float64) error {
	return rs.bo.Report(trialID, score)
}

// BestTrial returns the best completed trial.
func (rs *RandomStrategy) BestTrial() (Trial, bool) {
	return rs.bo.BestTrial()
}

// compile-time interface checks
var _ Strategy = (*BayesianOptimizer)(nil)
var _ Strategy = (*RandomStrategy)(nil)

// String returns a summary of the coordinator state.
func (c *Coordinator) String() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.best == nil {
		return fmt.Sprintf("Coordinator: %d trials, no best yet", len(c.results))
	}
	return fmt.Sprintf("Coordinator: %d trials, best score=%.6f (trial %d)",
		len(c.results), c.best.Metric.Score, c.best.TrialID)
}
