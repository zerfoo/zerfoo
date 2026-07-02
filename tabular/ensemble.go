package tabular

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// MetaLearnerConfig holds configuration for the ensemble's meta-learner MLP.
type MetaLearnerConfig struct {
	HiddenDims []int
}

// Ensemble combines multiple tabular Models and an optional tree ensemble
// via stacking. A learned meta-learner MLP fuses sub-model softmax outputs
// and tree predictions into a final Direction prediction.
type Ensemble struct {
	models          []*Model
	treePredictions func([]float64) []float64
	metaLearner     *Model
	engine          compute.Engine[float32]
	ops             numeric.Arithmetic[float32]
}

// NewEnsemble creates an Ensemble from trained sub-models and an optional tree
// prediction callback. treePredictions may be nil if no tree ensemble is used.
// The callback receives raw features and returns tree ensemble outputs (e.g.,
// class probabilities), decoupling the ensemble from any specific tree library.
func NewEnsemble(models []*Model, treePredictions func([]float64) []float64, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Ensemble, error) {
	if len(models) == 0 {
		return nil, fmt.Errorf("tabular: ensemble: at least one model is required")
	}

	return &Ensemble{
		models:          models,
		treePredictions: treePredictions,
		engine:          engine,
		ops:             ops,
	}, nil
}

// TrainMetaLearner trains the stacking meta-learner on pre-computed sub-model
// outputs. subModelOutputs[i] is the concatenated softmax outputs from all
// sub-models and tree predictions for sample i. labels[i] is in [0, 3).
func (e *Ensemble) TrainMetaLearner(subModelOutputs [][]float64, labels []int, tc TrainConfig, mlc MetaLearnerConfig) error {
	if len(subModelOutputs) == 0 {
		return fmt.Errorf("tabular: ensemble: no training data provided")
	}
	if len(subModelOutputs) != len(labels) {
		return fmt.Errorf("tabular: ensemble: outputs length %d != labels length %d", len(subModelOutputs), len(labels))
	}
	for i, l := range labels {
		if l < 0 || l >= 3 {
			return fmt.Errorf("tabular: ensemble: label %d at index %d is out of range [0, 3)", l, i)
		}
	}

	hiddenDims := mlc.HiddenDims
	if len(hiddenDims) == 0 {
		hiddenDims = []int{16}
	}

	mc := ModelConfig{
		InputDim:   len(subModelOutputs[0]),
		HiddenDims: hiddenDims,
		Activation: ActivationReLU,
	}

	model, err := Train(subModelOutputs, labels, tc, mc, e.engine, e.ops)
	if err != nil {
		return fmt.Errorf("tabular: ensemble: train meta-learner: %w", err)
	}
	e.metaLearner = model
	return nil
}

// Predict runs all sub-models and the tree prediction callback on the given
// features, concatenates their outputs, and feeds the result through the
// trained meta-learner to produce a final Direction and confidence.
func (e *Ensemble) Predict(features []float64) (Direction, float64, error) {
	if e.metaLearner == nil {
		return Flat, 0, fmt.Errorf("tabular: ensemble: meta-learner not trained; call TrainMetaLearner first")
	}

	// Collect sub-model outputs.
	metaInput, err := e.collectOutputs(features)
	if err != nil {
		return Flat, 0, err
	}

	return e.metaLearner.Predict(metaInput)
}

// predictFromOutputs runs the meta-learner on pre-computed concatenated outputs.
func (e *Ensemble) predictFromOutputs(outputs []float64) (Direction, float64, error) {
	if e.metaLearner == nil {
		return Flat, 0, fmt.Errorf("tabular: ensemble: meta-learner not trained")
	}
	return e.metaLearner.Predict(outputs)
}

// collectOutputs runs all sub-models and tree predictions, returning the
// concatenated output vector with actual softmax probabilities.
func (e *Ensemble) collectOutputs(features []float64) ([]float64, error) {
	var metaInput []float64

	for i, m := range e.models {
		probs, err := modelSoftmaxProbs(m, features)
		if err != nil {
			return nil, fmt.Errorf("tabular: ensemble: model %d: %w", i, err)
		}
		metaInput = append(metaInput, probs...)
	}

	if e.treePredictions != nil {
		treeOut := e.treePredictions(features)
		metaInput = append(metaInput, treeOut...)
	}

	return metaInput, nil
}

// modelSoftmaxProbs runs a forward pass through the model and returns the
// full 3-class softmax probabilities as float64.
func modelSoftmaxProbs(m *Model, features []float64) ([]float64, error) {
	if len(features) != m.config.InputDim {
		return nil, fmt.Errorf("expected %d features, got %d", m.config.InputDim, len(features))
	}

	ctx := context.Background()

	f32 := make([]float32, len(features))
	for i, v := range features {
		f32[i] = float32(v)
	}
	input, err := tensor.New[float32]([]int{1, m.config.InputDim}, f32)
	if err != nil {
		return nil, err
	}

	logits, _, _, err := forwardPass(ctx, m, input)
	if err != nil {
		return nil, err
	}

	probs, err := m.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return nil, err
	}

	probData := probs.Data()
	result := make([]float64, len(probData))
	for i, v := range probData {
		result[i] = float64(v)
	}
	return result, nil
}
