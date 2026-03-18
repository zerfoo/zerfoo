package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/zerfoo/zerfoo/training/automl"
)

// AutoMLCommand implements the "automl" CLI command for automated
// hyperparameter optimization.
type AutoMLCommand struct {
	out io.Writer
	// workerFactory creates a Worker for the given model/dataset/metric.
	// Injected for testing; production code uses a real training worker.
	workerFactory func(cfg autoMLRunConfig) automl.Worker
}

// autoMLRunConfig holds parsed CLI flags for an AutoML run.
type autoMLRunConfig struct {
	Model    string
	Dataset  string
	Trials   int
	Metric   string
	Output   string
	Strategy string
	Patience int
	Seed     int64
}

// trialLog is written as NDJSON for each completed trial.
type trialLog struct {
	TrialID int                `json:"trial_id"`
	Params  map[string]float64 `json:"params"`
	Score   float64            `json:"score"`
	Error   string             `json:"error,omitempty"`
}

// bestConfigOutput is the JSON saved for the best configuration.
type bestConfigOutput struct {
	TrialID int                `json:"trial_id"`
	Params  map[string]float64 `json:"params"`
	Score   float64            `json:"score"`
	Model   string             `json:"model"`
	Dataset string             `json:"dataset"`
	Metric  string             `json:"metric"`
}

// NewAutoMLCommand creates a new automl command.
func NewAutoMLCommand(out io.Writer) *AutoMLCommand {
	if out == nil {
		out = os.Stdout
	}
	return &AutoMLCommand{out: out}
}

// Name implements Command.Name.
func (c *AutoMLCommand) Name() string { return "automl" }

// Description implements Command.Description.
func (c *AutoMLCommand) Description() string {
	return "Run automated hyperparameter optimization"
}

// Run implements Command.Run.
func (c *AutoMLCommand) Run(ctx context.Context, args []string) error {
	cfg, err := c.parseArgs(args)
	if err != nil {
		return err
	}

	// Default search space: learning rate, LoRA rank, batch size, patch size.
	space := automl.SearchSpace{
		Params: []automl.HParam{
			{Name: "lr", Min: 1e-5, Max: 1e-2, IsLog: true},
			{Name: "rank", Min: 4, Max: 64, IsLog: false},
			{Name: "batch_size", Min: 1, Max: 64, IsLog: false},
			{Name: "patch_size", Min: 8, Max: 64, IsLog: false},
		},
	}

	var strategy automl.Strategy
	switch cfg.Strategy {
	case "random":
		strategy = automl.NewRandomStrategy(space.Params, cfg.Seed)
	default:
		strategy = automl.NewBayesianOptimizer(space.Params, cfg.Seed)
	}

	var worker automl.Worker
	if c.workerFactory != nil {
		worker = c.workerFactory(cfg)
	} else {
		worker = &placeholderWorker{metric: cfg.Metric}
	}

	coord, err := automl.NewCoordinator(automl.CoordinatorConfig{
		Space:             space,
		Strategy:          strategy,
		MaxTrials:         cfg.Trials,
		EarlyStopPatience: cfg.Patience,
	}, worker)
	if err != nil {
		return fmt.Errorf("automl: create coordinator: %w", err)
	}

	// Run trials, logging each result as NDJSON.
	enc := json.NewEncoder(c.out)
	best, runErr := coord.Run()

	for _, r := range coord.Results() {
		entry := trialLog{
			TrialID: r.TrialID,
			Params:  r.Config.Params,
			Score:   r.Metric.Score,
		}
		if r.Err != nil {
			entry.Error = r.Err.Error()
		}
		_ = enc.Encode(entry)
	}

	if runErr != nil {
		return fmt.Errorf("automl: %w", runErr)
	}

	// Save best config as JSON.
	if cfg.Output != "" {
		out := bestConfigOutput{
			TrialID: best.TrialID,
			Params:  best.Config.Params,
			Score:   best.Metric.Score,
			Model:   cfg.Model,
			Dataset: cfg.Dataset,
			Metric:  cfg.Metric,
		}
		data, err := json.MarshalIndent(out, "", "  ")
		if err != nil {
			return fmt.Errorf("automl: marshal best config: %w", err)
		}
		if err := os.WriteFile(cfg.Output, data, 0600); err != nil {
			return fmt.Errorf("automl: write best config: %w", err)
		}
		fmt.Fprintf(c.out, "Best config saved to %s\n", cfg.Output)
	}

	return nil
}

func (c *AutoMLCommand) parseArgs(args []string) (autoMLRunConfig, error) {
	cfg := autoMLRunConfig{
		Trials:   50,
		Metric:   "loss",
		Strategy: "bayesian",
		Patience: 10,
		Seed:     42,
	}

	for i := 0; i < len(args); i++ {
		arg := args[i]
		var eqVal string
		var hasEq bool
		if flag, val, ok := splitFlag(arg); ok {
			arg = flag
			eqVal = val
			hasEq = true
		}
		nextVal := func(flagName string) (string, error) {
			if hasEq {
				return eqVal, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", flagName)
			}
			i++
			return args[i], nil
		}
		switch arg {
		case "--model":
			v, err := nextVal("--model")
			if err != nil {
				return cfg, err
			}
			cfg.Model = v
		case "--dataset":
			v, err := nextVal("--dataset")
			if err != nil {
				return cfg, err
			}
			cfg.Dataset = v
		case "--trials":
			v, err := nextVal("--trials")
			if err != nil {
				return cfg, err
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				return cfg, fmt.Errorf("--trials: %w", err)
			}
			cfg.Trials = n
		case "--metric":
			v, err := nextVal("--metric")
			if err != nil {
				return cfg, err
			}
			cfg.Metric = v
		case "--output":
			v, err := nextVal("--output")
			if err != nil {
				return cfg, err
			}
			cfg.Output = v
		case "--strategy":
			v, err := nextVal("--strategy")
			if err != nil {
				return cfg, err
			}
			cfg.Strategy = v
		case "--patience":
			v, err := nextVal("--patience")
			if err != nil {
				return cfg, err
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				return cfg, fmt.Errorf("--patience: %w", err)
			}
			cfg.Patience = n
		case "--seed":
			v, err := nextVal("--seed")
			if err != nil {
				return cfg, err
			}
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil {
				return cfg, fmt.Errorf("--seed: %w", err)
			}
			cfg.Seed = n
		default:
			return cfg, fmt.Errorf("unknown flag: %s", arg)
		}
	}

	if cfg.Model == "" {
		return cfg, fmt.Errorf("--model is required")
	}
	if cfg.Dataset == "" {
		return cfg, fmt.Errorf("--dataset is required")
	}

	return cfg, nil
}

// Usage implements Command.Usage.
func (c *AutoMLCommand) Usage() string {
	return `automl [OPTIONS]

Run automated hyperparameter optimization.

OPTIONS:
  --model <path>        Path to model file (required)
  --dataset <path>      Path to dataset in JSONL format (required)
  --trials <int>        Number of trials to run (default: 50)
  --metric <name>       Optimization metric (default: loss)
  --output <path>       Path to save best config JSON
  --strategy <name>     Search strategy: bayesian, random (default: bayesian)
  --patience <int>      Early stopping patience (default: 10)
  --seed <int>          Random seed (default: 42)`
}

// Examples implements Command.Examples.
func (c *AutoMLCommand) Examples() []string {
	return []string{
		"automl --model path/to/model.gguf --dataset data.jsonl --trials 50 --metric sharpe",
		"automl --model model.gguf --dataset train.jsonl --strategy random --output best.json",
	}
}

// placeholderWorker is a stub worker used when no real training worker is
// available. In production, this would be replaced by a worker that actually
// loads the model, trains on the dataset, and evaluates the metric.
type placeholderWorker struct {
	metric string
}

func (w *placeholderWorker) RunTrial(config automl.Config) (automl.Metric, error) {
	return automl.Metric{}, fmt.Errorf("no training worker configured for metric %q; provide a worker implementation", w.metric)
}

// Static interface assertion.
var _ Command = (*AutoMLCommand)(nil)
