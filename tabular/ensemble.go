package tabular

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/training/optimizer"
)

// Ensemble combines multiple MLP sub-models and optional tree predictions
// via a stacking meta-learner. The meta-learner is a small MLP that takes
// concatenated sub-model outputs and produces the final prediction.
type Ensemble struct {
	models          []*Model
	treePredictions func([]float64) []float64
	metaLearner     *Model
}

// NewEnsemble creates an Ensemble from a set of MLP sub-models and an optional
// tree predictions callback. The treePredictions function takes a feature vector
// and returns tree ensemble output scores. Pass nil if no tree models are used.
func NewEnsemble(models []*Model, treePredictions func([]float64) []float64) *Ensemble {
	return &Ensemble{
		models:          models,
		treePredictions: treePredictions,
	}
}

// EnsembleTrainConfig holds hyperparameters for training the meta-learner.
type EnsembleTrainConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	WeightDecay  float64
	HiddenDims   []int
	Activation   Activation
}

// TrainEnsemble trains the stacking meta-learner on pre-computed sub-model
// outputs and labels. subModelOutputs[i] is the concatenated output vector
// from all sub-models (and tree predictions) for sample i. labels[i] must be
// in [0, 3) corresponding to Long, Short, Flat.
func (e *Ensemble) TrainEnsemble(subModelOutputs [][]float64, labels []int, config EnsembleTrainConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) error {
	if len(subModelOutputs) == 0 {
		return fmt.Errorf("tabular: ensemble: no training data provided")
	}
	if len(subModelOutputs) != len(labels) {
		return fmt.Errorf("tabular: ensemble: data length %d != labels length %d", len(subModelOutputs), len(labels))
	}
	if config.Epochs <= 0 {
		return fmt.Errorf("tabular: ensemble: Epochs must be positive")
	}
	if config.BatchSize <= 0 {
		config.BatchSize = len(subModelOutputs)
	}
	if config.LearningRate <= 0 {
		config.LearningRate = 0.01
	}

	numClasses := 3
	inputDim := len(subModelOutputs[0])
	for i, row := range subModelOutputs {
		if len(row) != inputDim {
			return fmt.Errorf("tabular: ensemble: row %d has %d features, expected %d", i, len(row), inputDim)
		}
	}
	for i, l := range labels {
		if l < 0 || l >= numClasses {
			return fmt.Errorf("tabular: ensemble: label %d at index %d is out of range [0, %d)", l, i, numClasses)
		}
	}

	hiddenDims := config.HiddenDims
	if len(hiddenDims) == 0 {
		hiddenDims = []int{16}
	}

	mc := ModelConfig{
		InputDim:    inputDim,
		HiddenDims:  hiddenDims,
		DropoutRate: 0.0,
		Activation:  config.Activation,
	}

	model, err := NewModel(mc, engine, ops)
	if err != nil {
		return fmt.Errorf("tabular: ensemble: %w", err)
	}

	params, err := buildParams(model)
	if err != nil {
		return fmt.Errorf("tabular: ensemble: %w", err)
	}

	lr := float32(config.LearningRate)
	wd := float32(config.WeightDecay)
	opt := optimizer.NewAdamW[float32](engine, lr, 0.9, 0.999, 1e-8, wd)

	ctx := context.Background()

	for epoch := 0; epoch < config.Epochs; epoch++ {
		perm := rand.Perm(len(subModelOutputs))

		for batchStart := 0; batchStart < len(subModelOutputs); batchStart += config.BatchSize {
			batchEnd := batchStart + config.BatchSize
			if batchEnd > len(subModelOutputs) {
				batchEnd = len(subModelOutputs)
			}
			batchSize := batchEnd - batchStart

			inputData := make([]float32, batchSize*inputDim)
			labelData := make([]int, batchSize)
			for i := 0; i < batchSize; i++ {
				idx := perm[batchStart+i]
				for j := 0; j < inputDim; j++ {
					inputData[i*inputDim+j] = float32(subModelOutputs[idx][j])
				}
				labelData[i] = labels[idx]
			}

			input, err := tensor.New[float32]([]int{batchSize, inputDim}, inputData)
			if err != nil {
				return err
			}

			logits, activations, preActivations, err := forwardPass(ctx, model, input)
			if err != nil {
				return fmt.Errorf("tabular: ensemble: forward: %w", err)
			}

			_, softmaxOut, err := crossEntropyLoss(ctx, engine, logits, labelData, batchSize, numClasses)
			if err != nil {
				return fmt.Errorf("tabular: ensemble: loss: %w", err)
			}

			err = backwardPass(ctx, model, engine, ops, params, activations, preActivations, input, softmaxOut, labelData, batchSize, numClasses)
			if err != nil {
				return fmt.Errorf("tabular: ensemble: backward: %w", err)
			}

			if err := opt.Step(ctx, params); err != nil {
				return fmt.Errorf("tabular: ensemble: optimizer step: %w", err)
			}
		}
	}

	e.metaLearner = model
	return nil
}

// Predict runs all sub-models and tree predictions, concatenates their outputs,
// and feeds them through the meta-learner to produce a final Direction and
// confidence score.
func (e *Ensemble) Predict(features []float64) (Direction, float64, error) {
	if e.metaLearner == nil {
		return Flat, 0, fmt.Errorf("tabular: ensemble: meta-learner not trained")
	}

	metaInput, err := e.collectSubOutputs(features)
	if err != nil {
		return Flat, 0, err
	}

	return e.metaLearner.Predict(metaInput)
}

// collectSubOutputs runs each sub-model and tree prediction callback,
// concatenating all outputs into a single feature vector for the meta-learner.
func (e *Ensemble) collectSubOutputs(features []float64) ([]float64, error) {
	var outputs []float64

	for i, m := range e.models {
		dir, conf, err := m.Predict(features)
		if err != nil {
			return nil, fmt.Errorf("tabular: ensemble: sub-model %d: %w", i, err)
		}
		// Encode direction as one-hot + confidence.
		oneHot := make([]float64, 3)
		oneHot[int(dir)] = 1.0
		outputs = append(outputs, oneHot...)
		outputs = append(outputs, conf)
	}

	if e.treePredictions != nil {
		treePreds := e.treePredictions(features)
		outputs = append(outputs, treePreds...)
	}

	return outputs, nil
}

// GenerateSubModelOutputs generates the sub-model output matrix for a dataset.
// This is used to prepare training data for TrainEnsemble.
func (e *Ensemble) GenerateSubModelOutputs(data [][]float64) ([][]float64, error) {
	outputs := make([][]float64, len(data))
	for i, features := range data {
		row, err := e.collectSubOutputs(features)
		if err != nil {
			return nil, fmt.Errorf("tabular: ensemble: sample %d: %w", i, err)
		}
		outputs[i] = row
	}
	return outputs, nil
}

