package tabular

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"time"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ErrTrainingLimit distinguishes exhausted execution limits from success.
var ErrTrainingLimit = errors.New("tabular: training limit exhausted")

// FitOptions controls bounded, seeded classifier training. Zero MaxSteps means
// the requested complete epoch count; zero MaxDuration defaults to two minutes.
type FitOptions struct {
	Epochs       int           `json:"epochs"`
	BatchSize    int           `json:"batch_size"`
	LearningRate float64       `json:"learning_rate"`
	WeightDecay  float64       `json:"weight_decay"`
	Seed         uint64        `json:"seed"`
	MaxSteps     int           `json:"max_steps"`
	MaxDuration  time.Duration `json:"max_duration_ns"`
}

// TrainingProgress reports completed optimizer steps and actual mean batch loss.
type TrainingProgress struct {
	Epoch int     `json:"epoch"`
	Steps int     `json:"steps"`
	Loss  float64 `json:"loss"`
}

// FitResult retains the actual trained model, including on interruption.
// Status is succeeded, canceled, budget_exhausted, or failed.
type FitResult[T tensor.Float] struct {
	Model       *Classifier[T]
	DatasetHash string
	Status      string
	TrainingProgress
}

// FitClassifier trains on the manifest's training rows only. Progress callbacks
// run synchronously and their errors stop training; no diagnostics are printed.
// This does not evaluate the held-out test split or qualify a deployment.
func FitClassifier[T tensor.Float](ctx context.Context, dataset *Dataset, config ClassifierConfig,
	options FitOptions, engine compute.Engine[T], progress func(TrainingProgress) error) (*FitResult[T], error) {
	if err := validateFitOptions(options); err != nil {
		return nil, err
	}
	if dataset == nil {
		return nil, fmt.Errorf("tabular: dataset is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	manifest := dataset.Manifest()
	if config.InputDim != len(manifest.Options.Features) || config.ClassCount != len(manifest.Labels) || !slices.Equal(config.Labels, manifest.Labels) {
		return nil, fmt.Errorf("tabular: classifier schema does not match dataset")
	}
	model, err := NewClassifier(config, engine)
	if err != nil {
		return nil, err
	}
	hash, err := dataset.Hash()
	if err != nil {
		return nil, err
	}
	rows, labels, err := dataset.Partition("train")
	if err != nil {
		return nil, err
	}
	result := &FitResult[T]{Model: model, DatasetHash: hash, Status: "failed"}
	duration := options.MaxDuration
	if duration == 0 {
		duration = 2 * time.Minute
	}
	runCtx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()
	opt := optimizer.NewAdamWFromFloat64[T](engine, options.LearningRate, 0.9, 0.999, 1e-8, options.WeightDecay)
	rng := rand.New(rand.NewPCG(options.Seed, options.Seed^0xd1b54a32d192ed03))
	for epoch := 0; epoch < options.Epochs; epoch++ {
		permutation := rng.Perm(len(rows))
		for start := 0; start < len(rows); start += options.BatchSize {
			if err := fitContextError(ctx, runCtx); err != nil {
				return stopFit(result, err)
			}
			if options.MaxSteps > 0 && result.Steps >= options.MaxSteps {
				return stopFit(result, ErrTrainingLimit)
			}
			end := min(start+options.BatchSize, len(rows))
			batch := make([][]float64, end-start)
			targets := make([]int, end-start)
			for i, index := range permutation[start:end] {
				batch[i] = rows[index]
				targets[i] = labels[index]
			}
			objective, err := model.computeGradients(runCtx, batch, targets)
			if err != nil {
				if contextErr := fitContextError(ctx, runCtx); contextErr != nil {
					return stopFit(result, contextErr)
				}
				return stopFit(result, err)
			}
			if err := opt.Step(runCtx, model.params); err != nil {
				if contextErr := fitContextError(ctx, runCtx); contextErr != nil {
					return stopFit(result, contextErr)
				}
				return stopFit(result, fmt.Errorf("tabular: optimizer: %w", err))
			}
			result.TrainingProgress = TrainingProgress{Epoch: epoch + 1, Steps: result.Steps + 1, Loss: objective}
			if progress != nil {
				if err := progress(result.TrainingProgress); err != nil {
					return stopFit(result, fmt.Errorf("tabular: progress: %w", err))
				}
			}
		}
	}
	if err := fitContextError(ctx, runCtx); err != nil {
		return stopFit(result, err)
	}
	result.Status = "succeeded"
	return result, nil
}

func validateFitOptions(options FitOptions) error {
	if options.Epochs <= 0 || options.BatchSize <= 0 || options.MaxSteps < 0 || options.MaxDuration < 0 {
		return fmt.Errorf("tabular: epochs/batch size must be positive and limits nonnegative")
	}
	if options.LearningRate <= 0 || math.IsNaN(options.LearningRate) || math.IsInf(options.LearningRate, 0) || options.WeightDecay < 0 || math.IsNaN(options.WeightDecay) || math.IsInf(options.WeightDecay, 0) {
		return fmt.Errorf("tabular: learning rate and weight decay must be finite and valid")
	}
	return nil
}

func fitContextError(parent, run context.Context) error {
	if err := parent.Err(); err != nil {
		return err
	}
	if err := run.Err(); err != nil {
		return fmt.Errorf("%w: %w", ErrTrainingLimit, err)
	}
	return nil
}

func stopFit[T tensor.Float](result *FitResult[T], err error) (*FitResult[T], error) {
	switch {
	case errors.Is(err, ErrTrainingLimit):
		result.Status = "budget_exhausted"
	case errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded):
		result.Status = "canceled"
	default:
		result.Status = "failed"
	}
	return result, err
}

// computeGradients composes existing linear/loss layers and engine operations.
// Gradients are copied into persistent parameter buffers before optimizer Step.
func (c *Classifier[T]) computeGradients(ctx context.Context, rows [][]float64, targets []int) (float64, error) {
	input, err := c.inputTensor(ctx, rows)
	if err != nil {
		return 0, err
	}
	labels, err := c.targetTensor(len(rows), targets)
	if err != nil {
		return 0, err
	}
	layers := len(c.params) / 2
	inputs := make([]*tensor.TensorNumeric[T], layers)
	pre := make([]*tensor.TensorNumeric[T], layers)
	x := input
	for layer := 0; layer < layers; layer++ {
		inputs[layer] = x
		weight, err := c.engine.Transpose(ctx, c.params[2*layer].Value, []int{1, 0})
		if err != nil {
			return 0, fmt.Errorf("tabular: training weight: %w", err)
		}
		x, err = functional.Linear(ctx, c.engine, x, weight, c.params[2*layer+1].Value)
		if err != nil {
			return 0, fmt.Errorf("tabular: training linear: %w", err)
		}
		pre[layer] = x
		if layer+1 < layers {
			x, err = functional.ReLU(ctx, c.engine, c.engine.Ops(), x)
			if err != nil {
				return 0, fmt.Errorf("tabular: training ReLU: %w", err)
			}
		}
	}
	objective := loss.NewCrossEntropyLoss[T](c.engine)
	lossTensor, err := objective.Forward(ctx, x, labels)
	if err != nil {
		return 0, fmt.Errorf("tabular: training loss: %w", err)
	}
	value := float64(lossTensor.Data()[0])
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("tabular: nonfinite training loss")
	}
	one, err := tensor.New[T]([]int{1}, []T{1})
	if err != nil {
		return 0, fmt.Errorf("tabular: loss seed: %w", err)
	}
	gradients, err := objective.Backward(ctx, types.FullBackprop, one)
	if err != nil {
		return 0, fmt.Errorf("tabular: loss backward: %w", err)
	}
	upstream := gradients[0]
	for layer := layers - 1; layer >= 0; layer-- {
		weight, err := c.engine.Transpose(ctx, c.params[2*layer].Value, []int{1, 0})
		if err != nil {
			return 0, fmt.Errorf("tabular: backward weight: %w", err)
		}
		next, dw, db, err := functional.LinearBackward(ctx, c.engine, upstream, inputs[layer], weight)
		if err != nil {
			return 0, fmt.Errorf("tabular: linear backward: %w", err)
		}
		dw, err = c.engine.Transpose(ctx, dw, []int{1, 0})
		if err != nil {
			return 0, fmt.Errorf("tabular: weight gradient: %w", err)
		}
		db, err = c.engine.Reshape(ctx, db, c.params[2*layer+1].Value.Shape())
		if err != nil {
			return 0, fmt.Errorf("tabular: bias gradient: %w", err)
		}
		if err := c.engine.Copy(ctx, c.params[2*layer].Gradient, dw); err != nil {
			return 0, fmt.Errorf("tabular: copy weight gradient: %w", err)
		}
		if err := c.engine.Copy(ctx, c.params[2*layer+1].Gradient, db); err != nil {
			return 0, fmt.Errorf("tabular: copy bias gradient: %w", err)
		}
		if layer > 0 {
			mask, err := c.engine.UnaryOp(ctx, pre[layer-1], c.engine.Ops().ReLUGrad)
			if err != nil {
				return 0, fmt.Errorf("tabular: activation gradient: %w", err)
			}
			upstream, err = c.engine.Mul(ctx, next, mask)
			if err != nil {
				return 0, fmt.Errorf("tabular: multiply activation gradient: %w", err)
			}
		}
	}
	return value, nil
}
