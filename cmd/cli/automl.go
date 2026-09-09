package cli

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/zerfoo/training/automl"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
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
	Model           string
	Dataset         string
	Trials          int
	Metric          string
	Output          string
	Strategy        string
	Patience        int
	Seed            int64
	PriorExperiment string
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
	TrialID        int                `json:"trial_id"`
	Params         map[string]float64 `json:"params"`
	Score          float64            `json:"score"`
	Model          string             `json:"model"`
	Dataset        string             `json:"dataset"`
	Metric         string             `json:"metric"`
	ArtifactPath   string             `json:"artifact_path,omitempty"`
	ArtifactSHA256 string             `json:"artifact_sha256,omitempty"`
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

	if cfg.Model == "tabular" && c.workerFactory == nil {
		return c.runTabularExperiment(ctx, cfg)
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
	if err := ctx.Err(); err != nil {
		return err
	}

	for _, r := range coord.Results() {
		entry := trialLog{
			TrialID: r.TrialID,
			Params:  r.Config.Params,
			Score:   r.Metric.Score,
		}
		if r.Err != nil {
			entry.Error = r.Err.Error()
		}
		if err := enc.Encode(entry); err != nil {
			return fmt.Errorf("automl: write trial result: %w", err)
		}
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
		if _, err := fmt.Fprintf(c.out, "Best config saved to %s\n", cfg.Output); err != nil {
			return fmt.Errorf("automl: write result: %w", err)
		}

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
		case "--prior-experiment":
			v, err := nextVal("--prior-experiment")
			if err != nil {
				return cfg, err
			}
			cfg.PriorExperiment = v
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
  --model <name>        Use tabular for numeric CSV training (required)
  --dataset <path>      CSV with target in the final column (required)
  --trials <int>        Search trials after three baselines (default: 50)
  --metric <name>       Optimization metric (default: loss)
  --output <path>       New best-config JSON; trials live in <path>.run
  --prior-experiment <dir>  Link prior test exposure when retuning
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

// tabularWorker trains a tabular model for each trial configuration and
// returns the validation metric score.
type tabularWorker struct {
	dataset        *tabular.Dataset
	metric         string
	models         map[string]*tabular.Classifier[float32]
	lastEvaluation tabular.Evaluation
	seed           uint64
}

func newTabularWorker(datasetPath, metric string) (*tabularWorker, error) {
	return newTabularWorkerContext(context.Background(), datasetPath, metric, 42)
}

func newTabularWorkerContext(ctx context.Context, datasetPath, metric string, seed uint64) (*tabularWorker, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch metric {
	case "accuracy", "macro_f1", "loss", "cross_entropy":
	default:
		return nil, fmt.Errorf("automl: unsupported metric %q", metric)
	}
	file, err := os.Open(datasetPath)
	if err != nil {
		return nil, fmt.Errorf("automl: open dataset: %w", err)
	}
	header, err := csv.NewReader(file).Read()
	if err != nil {
		closeErr := file.Close()
		return nil, errors.Join(fmt.Errorf("automl: CSV header: %w", err), closeErr)
	}
	if len(header) < 2 {
		closeErr := file.Close()
		return nil, errors.Join(fmt.Errorf("automl: need features and target column"), closeErr)
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		closeErr := file.Close()
		return nil, errors.Join(err, closeErr)
	}
	dataset, err := tabular.InspectCSV(ctx, file, tabular.DatasetOptions{Target: header[len(header)-1], Split: "stratified", Seed: seed})
	if err := errors.Join(err, file.Close()); err != nil {
		return nil, err
	}
	return &tabularWorker{dataset: dataset, metric: metric, models: make(map[string]*tabular.Classifier[float32]), seed: seed}, nil
}

func tabularTrialKey(config automl.Config) (string, error) {
	raw, err := json.Marshal(config.Params)
	if err != nil {
		return "", fmt.Errorf("automl: invalid trial parameters: %w", err)
	}
	return string(raw), nil
}

// RunTrial implements the legacy Worker interface. The public command supplies
// its request context through runTrial instead of this compatibility wrapper.
func (w *tabularWorker) RunTrial(config automl.Config) (automl.Metric, error) {
	return w.runTrial(context.Background(), config)
}

func (w *tabularWorker) runTrial(ctx context.Context, config automl.Config) (automl.Metric, error) {
	options := tabular.FitOptions{Epochs: 10, BatchSize: 32, LearningRate: 0.01, WeightDecay: 0.0001, Seed: w.seed}
	for name, value := range config.Params {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return automl.Metric{}, fmt.Errorf("automl: nonfinite parameter %q", name)
		}
		switch name {
		case "lr":
			options.LearningRate = value
		case "batch_size":
			if value < 1 || value > 64 {
				return automl.Metric{}, fmt.Errorf("automl: batch_size must be in [1,64]")
			}
			options.BatchSize = int(math.Round(value))
		default:
			return automl.Metric{}, fmt.Errorf("automl: unsupported parameter %q", name)
		}
	}
	manifest := w.dataset.Manifest()
	modelConfig := tabular.ClassifierConfig{InputDim: len(manifest.Options.Features), ClassCount: len(manifest.Labels), Labels: manifest.Labels, HiddenDims: []int{16}, Seed: w.seed}
	result, err := tabular.FitClassifier(ctx, w.dataset, modelConfig, options, compute.NewCPUEngine(numeric.Float32Ops{}), nil)
	if err != nil {
		return automl.Metric{}, fmt.Errorf("automl: trial training: %w", err)
	}
	rows, labels, err := w.dataset.Partition("validation")
	if err != nil {
		return automl.Metric{}, err
	}
	report, err := result.Model.Evaluate(ctx, rows, labels)
	if err != nil {
		return automl.Metric{}, fmt.Errorf("automl: validation: %w", err)
	}
	w.lastEvaluation = report
	key, err := tabularTrialKey(config)
	if err != nil {
		return automl.Metric{}, err
	}
	w.models[key] = result.Model
	score := report.Accuracy
	switch w.metric {
	case "macro_f1":
		score = report.MacroF1
	case "loss", "cross_entropy":
		score = -report.CrossEntropy
	}
	// The existing coordinator maximizes utility. Negate loss internally only;
	// public trial/config output restores the positive measured loss.
	return automl.Metric{Score: score}, nil
}

// readTabularCSV reads a CSV file where all columns except the last are
// numeric features and the last column is an integer label.
func readTabularCSV(path string) ([][]float64, []int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open dataset: %w", err)
	}

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	err = errors.Join(err, f.Close())
	if err != nil {
		return nil, nil, fmt.Errorf("read dataset: %w", err)
	}
	if len(records) < 2 {
		return nil, nil, fmt.Errorf("dataset must have a header and at least one data row")
	}

	// Skip header row.
	rows := records[1:]
	numCols := len(records[0])
	if numCols < 2 {
		return nil, nil, fmt.Errorf("dataset must have at least one feature column and one label column")
	}

	data := make([][]float64, len(rows))
	labels := make([]int, len(rows))
	for i, row := range rows {
		if len(row) != numCols {
			return nil, nil, fmt.Errorf("row %d has %d columns, expected %d", i+1, len(row), numCols)
		}
		features := make([]float64, numCols-1)
		for j := 0; j < numCols-1; j++ {
			v, err := strconv.ParseFloat(row[j], 64)
			if err != nil {
				return nil, nil, fmt.Errorf("row %d col %d: %w", i+1, j, err)
			}
			features[j] = v
		}
		data[i] = features

		label, err := strconv.Atoi(row[numCols-1])
		if err != nil {
			return nil, nil, fmt.Errorf("row %d label: %w", i+1, err)
		}
		labels[i] = label
	}

	return data, labels, nil
}

// Static interface assertion.
var _ Command = (*AutoMLCommand)(nil)
