package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

type tabularTrainEvent struct {
	Version     int     `json:"version"`
	Type        string  `json:"type"`
	Status      string  `json:"status,omitempty"`
	Steps       int     `json:"steps,omitempty"`
	Epoch       int     `json:"epoch,omitempty"`
	Loss        float64 `json:"loss,omitempty"`
	Bundle      string  `json:"bundle,omitempty"`
	BundleID    string  `json:"bundle_id,omitempty"`
	DatasetHash string  `json:"dataset_hash,omitempty"`
	Error       string  `json:"error,omitempty"`
}

func (c *TrainCommand) runTabularTrain(ctx context.Context, args []string) (runErr error) {
	encoder := json.NewEncoder(c.out)
	defer func() {
		if runErr != nil {
			status := "failed"
			if errors.Is(runErr, context.Canceled) || errors.Is(runErr, context.DeadlineExceeded) {
				status = "canceled"
			}
			if errors.Is(runErr, tabular.ErrTrainingLimit) {
				status = "budget_exhausted"
			}
			runErr = errors.Join(runErr, encoder.Encode(tabularTrainEvent{Version: 1, Type: "error", Status: status, Error: runErr.Error()}))
		}
	}()
	flags := flag.NewFlagSet("train tabular", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	dataPath := flags.String("data", "", "numeric CSV file")
	target := flags.String("target", "", "target column")
	output := flags.String("output", "", "new bundle directory")
	recipe := flags.String("recipe", "mlp", "linear or mlp")
	epochs := flags.Int("epochs", 200, "epochs")
	batch := flags.Int("batch-size", 15, "batch size")
	lr := flags.Float64("lr", 0.01, "learning rate")
	seed := flags.Uint64("seed", 42, "seed")
	split := flags.String("split", "stratified", "split policy")
	group := flags.String("group", "", "group column")
	timestamp := flags.String("time", "", "timestamp column")
	pinned := flags.String("dataset-manifest", "", "previously written DatasetManifest JSON")
	device := flags.String("device", "cpu", "cpu or cuda")
	maxSteps := flags.Int("max-steps", 0, "step limit")
	duration := flags.Duration("max-duration", 2*time.Minute, "time limit")
	if err := flags.Parse(args); err != nil {
		return fmt.Errorf("train tabular: flags: %w", err)
	}
	if flags.NArg() != 0 || *dataPath == "" || *target == "" || *output == "" {
		return fmt.Errorf("train tabular requires --data, --target and --output with no positional arguments")
	}
	if *recipe != "linear" && *recipe != "mlp" {
		return fmt.Errorf("train tabular: recipe must be linear or mlp")
	}
	file, err := os.Open(*dataPath)
	if err != nil {
		return fmt.Errorf("train tabular: dataset: %w", err)
	}
	var dataset *tabular.Dataset
	if *pinned != "" {
		raw, readErr := os.ReadFile(*pinned)
		if readErr != nil {
			return errors.Join(fmt.Errorf("train tabular: manifest: %w", readErr), file.Close())
		}
		var manifest tabular.DatasetManifest
		if decodeErr := json.Unmarshal(raw, &manifest); decodeErr != nil {
			return errors.Join(fmt.Errorf("train tabular: manifest JSON: %w", decodeErr), file.Close())
		}
		if manifest.Options.Target != *target {
			return errors.Join(fmt.Errorf("train tabular: target differs from pinned manifest"), file.Close())
		}
		dataset, err = tabular.ReplayCSV(ctx, file, manifest)
	} else {
		dataset, err = tabular.InspectCSV(ctx, file, tabular.DatasetOptions{Target: *target, Split: *split, Group: *group, Time: *timestamp, Seed: *seed})
	}
	if err := errors.Join(err, file.Close()); err != nil {
		return err
	}
	var engine compute.Engine[float32]
	switch *device {
	case "cpu":
		engine = compute.NewCPUEngine(numeric.Float32Ops{})
	case "cuda":
		gpu, gpuErr := compute.NewGPUEngine[float32](numeric.Float32Ops{})
		if gpuErr != nil {
			return fmt.Errorf("train tabular: CUDA: %w", gpuErr)
		}
		engine = gpu
		defer func() { runErr = errors.Join(runErr, gpu.Close()) }()
	default:
		return fmt.Errorf("train tabular: device must be cpu or cuda")
	}
	manifest := dataset.Manifest()
	config := tabular.ClassifierConfig{InputDim: len(manifest.Options.Features), ClassCount: len(manifest.Labels), Labels: manifest.Labels, Seed: *seed}
	if *recipe == "mlp" {
		config.HiddenDims = []int{16}
	}
	result, err := tabular.FitClassifier(ctx, dataset, config, tabular.FitOptions{Epochs: *epochs, BatchSize: *batch, LearningRate: *lr, WeightDecay: 0.0001, Seed: *seed, MaxSteps: *maxSteps, MaxDuration: *duration}, engine, func(p tabular.TrainingProgress) error {
		return encoder.Encode(tabularTrainEvent{Version: 1, Type: "progress", Steps: p.Steps, Epoch: p.Epoch, Loss: p.Loss})
	})
	if err != nil {
		return err
	}
	id, err := tabular.SaveClassifierBundle(ctx, *output, result.Model, dataset, "")
	if err != nil {
		return err
	}
	return encoder.Encode(tabularTrainEvent{Version: 1, Type: "result", Status: result.Status, Steps: result.Steps, Epoch: result.Epoch, Loss: result.Loss, Bundle: *output, BundleID: id, DatasetHash: result.DatasetHash})
}
