package cli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/zerfoo/training/automl"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// experimentTrial is committed before dispatch, then replaced on completion.
// A process crash leaves an honest running record, never a fabricated success.
type experimentTrial struct {
	Version        int                 `json:"version"`
	Type           string              `json:"type"`
	TrialID        int                 `json:"trial_id"`
	Recipe         string              `json:"recipe"`
	Params         map[string]float64  `json:"params"`
	Status         string              `json:"status"`
	Score          *float64            `json:"score,omitempty"`
	Validation     *tabular.Evaluation `json:"validation,omitempty"`
	Error          string              `json:"error,omitempty"`
	ElapsedNS      int64               `json:"elapsed_ns"`
	AllocatedBytes uint64              `json:"process_allocated_bytes"`
	Steps          int                 `json:"steps"`
	ArtifactPath   string              `json:"artifact_path,omitempty"`
	ArtifactSHA256 string              `json:"artifact_sha256,omitempty"`
}

type experimentRecord struct {
	Version            int                 `json:"version"`
	Protocol           string              `json:"protocol"`
	DatasetHash        string              `json:"dataset_hash"`
	Seed               int64               `json:"seed"`
	Metric             string              `json:"metric"`
	Status             string              `json:"status"`
	PriorExperiment    string              `json:"prior_experiment,omitempty"`
	PriorTestExposures int                 `json:"prior_test_exposures"`
	TestExposures      int                 `json:"test_exposures"`
	Winner             *experimentTrial    `json:"winner,omitempty"`
	Test               *tabular.Evaluation `json:"test,omitempty"`
}

// writeExperimentJSON replaces one complete record and syncs it before return.
func writeExperimentJSON(dir, name string, value any) (err error) {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("automl: encode record: %w", err)
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		return fmt.Errorf("automl: open experiment: %w", err)
	}
	defer func() { err = errors.Join(err, root.Close()) }()
	file, err := root.OpenFile(name+".tmp", os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return fmt.Errorf("automl: stage record: %w", err)
	}
	_, writeErr := file.Write(append(raw, '\n'))
	syncErr := file.Sync()
	if err := errors.Join(writeErr, syncErr, file.Close()); err != nil {
		return err
	}
	if err := root.Rename(name+".tmp", name); err != nil {
		return fmt.Errorf("automl: publish record: %w", err)
	}
	directory, err := root.Open(".")
	if err != nil {
		return err
	}
	return errors.Join(directory.Sync(), directory.Close())
}

func experimentScore(report tabular.Evaluation, metric string) float64 {
	switch metric {
	case "loss", "cross_entropy":
		return report.CrossEntropy
	case "macro_f1":
		return report.MacroF1
	default:
		return report.Accuracy
	}
}

func betterExperimentTrial(candidate, best *experimentTrial, metric string) bool {
	if candidate.Status != "succeeded" || candidate.Score == nil {
		return false
	}
	if best == nil {
		return true
	}
	if metric == "loss" || metric == "cross_entropy" {
		return *candidate.Score < *best.Score
	}
	return *candidate.Score > *best.Score
}

func (c *AutoMLCommand) runTabularExperiment(ctx context.Context, cfg autoMLRunConfig) error {
	if cfg.Trials <= 0 || cfg.Patience < 0 {
		return fmt.Errorf("automl: trials must be positive and patience nonnegative")
	}
	if cfg.Strategy != "random" && cfg.Strategy != "bayesian" {
		return fmt.Errorf("automl: unsupported strategy %q", cfg.Strategy)
	}
	worker, err := newTabularWorkerContext(ctx, cfg.Dataset, cfg.Metric, uint64(cfg.Seed))
	if err != nil {
		return err
	}
	hash, err := worker.dataset.Hash()
	if err != nil {
		return err
	}
	record := experimentRecord{Version: 1, Protocol: "tabular-heldout-v1", DatasetHash: hash, Seed: cfg.Seed, Metric: cfg.Metric, Status: "tuning", PriorExperiment: cfg.PriorExperiment}
	if cfg.PriorExperiment != "" {
		root, err := os.OpenRoot(cfg.PriorExperiment)
		if err != nil {
			return fmt.Errorf("automl: prior experiment: %w", err)
		}
		raw, readErr := root.ReadFile("experiment.json")
		if err := errors.Join(readErr, root.Close()); err != nil {
			return err
		}
		var prior experimentRecord
		if err := json.Unmarshal(raw, &prior); err != nil {
			return err
		}
		if prior.Version != 1 || prior.Protocol != record.Protocol || prior.DatasetHash != hash {
			return fmt.Errorf("automl: incompatible prior experiment")
		}
		record.PriorTestExposures = prior.PriorTestExposures + prior.TestExposures
	}
	dir := cfg.Output + ".run"
	if cfg.Output == "" {
		dir, err = os.MkdirTemp("", "zerfoo-automl-")
	} else {
		if _, statErr := os.Lstat(cfg.Output); statErr == nil {
			return fmt.Errorf("automl: output already exists")
		} else if !errors.Is(statErr, os.ErrNotExist) {
			return statErr
		}
		err = os.Mkdir(dir, 0700)
	}
	if err != nil {
		return fmt.Errorf("automl: create new experiment: %w", err)
	}
	dir, err = filepath.Abs(dir)
	if err != nil {
		return err
	}
	if err := writeExperimentJSON(dir, "dataset.json", worker.dataset.Manifest()); err != nil {
		return err
	}
	if err := writeExperimentJSON(dir, "experiment.json", record); err != nil {
		return err
	}
	enc := json.NewEncoder(c.out)
	if err := enc.Encode(map[string]any{"version": 1, "type": "experiment", "path": dir}); err != nil {
		return err
	}
	finishError := func(cause error) error {
		record.Status = "failed"
		if errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) {
			record.Status = "canceled"
		}
		return errors.Join(cause, writeExperimentJSON(dir, "experiment.json", record))
	}
	var winner *experimentTrial
	id := 0
	run := func(recipe string, params map[string]float64) (*experimentTrial, error) {
		id++
		trial := &experimentTrial{Version: 1, Type: "trial", TrialID: id, Recipe: recipe, Params: params, Status: "running"}
		name := fmt.Sprintf("trial-%04d.json", id)
		if err := writeExperimentJSON(dir, name, trial); err != nil {
			return nil, err
		}
		started := time.Now()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		model, steps, trialErr := fitExperimentCandidate(ctx, worker.dataset, recipe, params, uint64(cfg.Seed))
		trial.Steps = steps
		if trialErr == nil {
			rows, labels, err := worker.dataset.Partition("validation")
			if err != nil {
				trialErr = err
			} else {
				report, err := model.Evaluate(ctx, rows, labels)
				trialErr = err
				if err == nil {
					trial.Validation = &report
					score := experimentScore(report, cfg.Metric)
					trial.Score = &score
				}
			}
		}
		// Retain partial models too. Cancellation stops compute, not terminal evidence.
		if model != nil {
			trial.ArtifactPath = filepath.Join(dir, fmt.Sprintf("trial-%04d.bundle", id))
			artifact, err := tabular.SaveClassifierBundle(context.WithoutCancel(ctx), trial.ArtifactPath, model, worker.dataset, name)
			trial.ArtifactSHA256 = artifact
			trialErr = errors.Join(trialErr, err)
		}
		runtime.ReadMemStats(&after)
		trial.ElapsedNS = time.Since(started).Nanoseconds()
		trial.AllocatedBytes = after.TotalAlloc - before.TotalAlloc
		trial.Status = "succeeded"
		if trialErr != nil {
			trial.Status = "failed"
			if errors.Is(trialErr, context.Canceled) || errors.Is(trialErr, context.DeadlineExceeded) {
				trial.Status = "canceled"
			}
			trial.Error = trialErr.Error()
		}
		if err := writeExperimentJSON(dir, name, trial); err != nil {
			return nil, err
		}
		if err := enc.Encode(trial); err != nil {
			return nil, err
		}
		if betterExperimentTrial(trial, winner, cfg.Metric) {
			winner = trial
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return trial, nil
	}
	// Ties retain the simpler, earlier candidate; a majority win is valid.
	defaults := map[string]float64{"lr": 0.01, "batch_size": 32}
	for _, recipe := range []string{"majority", "linear", "mlp"} {
		params := defaults
		if recipe == "majority" {
			params = map[string]float64{}
		}
		if _, err := run(recipe, params); err != nil {
			return finishError(err)
		}
	}
	space := []automl.HParam{{Name: "lr", Min: 1e-5, Max: 1e-2, IsLog: true}, {Name: "batch_size", Min: 1, Max: 64}}
	var strategy automl.Strategy = automl.NewBayesianOptimizer(space, cfg.Seed)
	if cfg.Strategy == "random" {
		strategy = automl.NewRandomStrategy(space, cfg.Seed)
	}
	stale := 0
	for i := 0; i < cfg.Trials; i++ {
		suggestion, params := strategy.Suggest()
		previous := winner
		trial, err := run("mlp", params)
		if err != nil {
			return finishError(err)
		}
		if trial.Status == "succeeded" {
			utility := *trial.Score
			if cfg.Metric == "loss" || cfg.Metric == "cross_entropy" {
				utility = -utility
			}
			if err := strategy.Report(suggestion, utility); err != nil {
				return finishError(err)
			}
		}
		if winner == previous {
			stale++
		} else {
			stale = 0
		}
		if cfg.Patience > 0 && stale >= cfg.Patience {
			break
		}
	}
	if winner == nil {
		return finishError(fmt.Errorf("automl: no successful candidates"))
	}
	record.Winner = winner
	record.Status = "frozen"
	if err := writeExperimentJSON(dir, "experiment.json", record); err != nil {
		return err
	}
	// Reload the exact scored artifact, never retrain the winner.
	bundle, err := tabular.LoadClassifierBundle(ctx, winner.ArtifactPath, winner.ArtifactSHA256, compute.NewCPUEngine(numeric.Float32Ops{}))
	if err != nil {
		return finishError(err)
	}
	// Commit exposure BEFORE reading test rows. Interrupted evaluations are not retried.
	record.TestExposures = 1
	record.Status = "evaluating"
	if err := writeExperimentJSON(dir, "experiment.json", record); err != nil {
		return err
	}
	rows, labels, err := worker.dataset.Partition("test")
	if err != nil {
		return finishError(err)
	}
	report, err := bundle.Model.Evaluate(ctx, rows, labels)
	if err != nil {
		return finishError(err)
	}
	record.Test = &report
	record.Status = "succeeded"
	if err := writeExperimentJSON(dir, "experiment.json", record); err != nil {
		return err
	}
	if cfg.Output != "" {
		out := bestConfigOutput{TrialID: winner.TrialID, Params: winner.Params, Score: *winner.Score, Model: cfg.Model, Dataset: cfg.Dataset, Metric: cfg.Metric, ArtifactPath: winner.ArtifactPath, ArtifactSHA256: winner.ArtifactSHA256}
		if err := writeExperimentJSON(filepath.Dir(cfg.Output), filepath.Base(cfg.Output), out); err != nil {
			return err
		}
	}
	return enc.Encode(map[string]any{"version": 1, "type": "result", "experiment": dir, "artifact_path": winner.ArtifactPath, "artifact_sha256": winner.ArtifactSHA256, "recipe": winner.Recipe, "test": report})
}

func fitExperimentCandidate(ctx context.Context, dataset *tabular.Dataset, recipe string, params map[string]float64, seed uint64) (*tabular.Classifier[float32], int, error) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	if recipe == "majority" {
		model, err := tabular.NewMajorityClassifier(ctx, dataset, engine)
		return model, 0, err
	}
	manifest := dataset.Manifest()
	config := tabular.ClassifierConfig{InputDim: len(manifest.Options.Features), ClassCount: len(manifest.Labels), Labels: manifest.Labels, Seed: seed}
	if recipe == "mlp" {
		config.HiddenDims = []int{16}
	} else if recipe != "linear" {
		return nil, 0, fmt.Errorf("automl: unsupported recipe %q", recipe)
	}
	options := tabular.FitOptions{Epochs: 10, BatchSize: int(params["batch_size"] + 0.5), LearningRate: params["lr"], WeightDecay: 0.0001, Seed: seed}
	result, err := tabular.FitClassifier(ctx, dataset, config, options, engine, nil)
	if result == nil {
		return nil, 0, err
	}
	return result.Model, result.Steps, err
}
