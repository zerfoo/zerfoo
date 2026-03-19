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

// PreTrainConfig holds hyperparameters for pre-training a base model on
// multi-source data.
type PreTrainConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	WeightDecay  float64
	HiddenDims   []int
	DropoutRate  float64
	Activation   Activation
}

// BaseModel wraps a pre-trained Model whose weights serve as an initialisation
// point for fine-tuning on a specific data source.
type BaseModel struct {
	Model *Model
}

// PreTrain trains a tabular model on data from multiple sources so the model
// learns universal feature patterns. allData[source][sample][feature] contains
// the feature vectors; allLabels[source][sample] contains the corresponding
// labels. All sources must share the same feature dimensionality and label
// space.
func PreTrain(
	allData [][][]float64,
	allLabels [][]int,
	config PreTrainConfig,
	engine compute.Engine[float32],
	ops numeric.Arithmetic[float32],
) (*BaseModel, error) {
	if len(allData) == 0 {
		return nil, fmt.Errorf("tabular: pretrain: no data sources provided")
	}
	if len(allData) != len(allLabels) {
		return nil, fmt.Errorf("tabular: pretrain: data sources (%d) != label sources (%d)", len(allData), len(allLabels))
	}

	// Concatenate all sources into a single dataset.
	var combined [][]float64
	var combinedLabels []int
	for i, src := range allData {
		if len(src) == 0 {
			return nil, fmt.Errorf("tabular: pretrain: source %d is empty", i)
		}
		if len(src) != len(allLabels[i]) {
			return nil, fmt.Errorf("tabular: pretrain: source %d data length %d != labels length %d", i, len(src), len(allLabels[i]))
		}
		combined = append(combined, src...)
		combinedLabels = append(combinedLabels, allLabels[i]...)
	}

	mc := ModelConfig{
		HiddenDims:  config.HiddenDims,
		DropoutRate:  config.DropoutRate,
		Activation:   config.Activation,
	}

	tc := TrainConfig{
		Epochs:       config.Epochs,
		BatchSize:    config.BatchSize,
		LearningRate: config.LearningRate,
		WeightDecay:  config.WeightDecay,
	}

	model, err := Train(combined, combinedLabels, tc, mc, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("tabular: pretrain: %w", err)
	}

	return &BaseModel{Model: model}, nil
}

// FineTune creates a new Model initialised from the BaseModel's pre-trained
// weights and trains it on the given data. The fine-tuning data must have the
// same feature dimensionality as the pre-training data.
func (bm *BaseModel) FineTune(data [][]float64, labels []int, config TrainConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Model, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("tabular: finetune: no data provided")
	}
	if len(data) != len(labels) {
		return nil, fmt.Errorf("tabular: finetune: data length %d != labels length %d", len(data), len(labels))
	}

	inputDim := len(data[0])
	if inputDim != bm.Model.config.InputDim {
		return nil, fmt.Errorf("tabular: finetune: input dim %d != pretrained dim %d", inputDim, bm.Model.config.InputDim)
	}

	// Clone the pre-trained model so the base weights are not mutated.
	model, err := cloneModel(bm.Model)
	if err != nil {
		return nil, fmt.Errorf("tabular: finetune: %w", err)
	}

	// Fine-tune using the existing training loop internals.
	return fineTuneModel(model, data, labels, config, engine, ops)
}

// cloneModel creates a deep copy of a Model, duplicating all weight tensors.
func cloneModel(src *Model) (*Model, error) {
	dst := &Model{
		config: src.config,
		engine: src.engine,
		ops:    src.ops,
		layers: make([]mlpLayer, len(src.layers)),
	}

	for i, l := range src.layers {
		cl, err := cloneLayer(l)
		if err != nil {
			return nil, fmt.Errorf("clone layer %d: %w", i, err)
		}
		dst.layers[i] = cl
	}

	head, err := cloneLayer(src.head)
	if err != nil {
		return nil, fmt.Errorf("clone head: %w", err)
	}
	dst.head = head

	return dst, nil
}

// cloneLayer creates a deep copy of an mlpLayer.
func cloneLayer(l mlpLayer) (mlpLayer, error) {
	wData := make([]float32, len(l.weights.Data()))
	copy(wData, l.weights.Data())
	w, err := newTensorFromSlice(l.weights.Shape(), wData)
	if err != nil {
		return mlpLayer{}, err
	}

	bData := make([]float32, len(l.biases.Data()))
	copy(bData, l.biases.Data())
	b, err := newTensorFromSlice(l.biases.Shape(), bData)
	if err != nil {
		return mlpLayer{}, err
	}

	return mlpLayer{weights: w, biases: b}, nil
}

// newTensorFromSlice wraps tensor.New for float32.
func newTensorFromSlice(shape []int, data []float32) (*tensor.TensorNumeric[float32], error) {
	return tensor.New[float32](shape, data)
}

// fineTuneModel runs the training loop on an already-initialised model.
func fineTuneModel(model *Model, data [][]float64, labels []int, config TrainConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Model, error) {
	if config.Epochs <= 0 {
		return nil, fmt.Errorf("tabular: finetune: Epochs must be positive")
	}
	if config.BatchSize <= 0 {
		config.BatchSize = len(data)
	}
	if config.LearningRate <= 0 {
		config.LearningRate = 0.01
	}

	numClasses := 3
	inputDim := model.config.InputDim

	for i, l := range labels {
		if l < 0 || l >= numClasses {
			return nil, fmt.Errorf("tabular: finetune: label %d at index %d is out of range [0, %d)", l, i, numClasses)
		}
	}

	trainData, trainLabels, _, _ := splitData(data, labels, config.ValidationSplit)

	params, err := buildParams(model)
	if err != nil {
		return nil, fmt.Errorf("tabular: finetune: %w", err)
	}

	lr := float32(config.LearningRate)
	wd := float32(config.WeightDecay)
	opt := optimizer.NewAdamW[float32](engine, lr, 0.9, 0.999, 1e-8, wd)

	ctx := context.Background()

	for epoch := 0; epoch < config.Epochs; epoch++ {
		perm := rand.Perm(len(trainData))

		for batchStart := 0; batchStart < len(trainData); batchStart += config.BatchSize {
			batchEnd := batchStart + config.BatchSize
			if batchEnd > len(trainData) {
				batchEnd = len(trainData)
			}
			batchSize := batchEnd - batchStart

			inputSlice := make([]float32, batchSize*inputDim)
			labelSlice := make([]int, batchSize)
			for i := 0; i < batchSize; i++ {
				idx := perm[batchStart+i]
				for j := 0; j < inputDim; j++ {
					inputSlice[i*inputDim+j] = float32(trainData[idx][j])
				}
				labelSlice[i] = trainLabels[idx]
			}

			input, err := tensor.New[float32]([]int{batchSize, inputDim}, inputSlice)
			if err != nil {
				return nil, err
			}

			logits, activations, preActivations, err := forwardPass(ctx, model, input)
			if err != nil {
				return nil, err
			}

			_, softmaxOut, err := crossEntropyLoss(ctx, engine, logits, labelSlice, batchSize, numClasses)
			if err != nil {
				return nil, err
			}

			err = backwardPass(ctx, model, engine, ops, params, activations, preActivations, input, softmaxOut, labelSlice, batchSize, numClasses)
			if err != nil {
				return nil, err
			}

			if err := opt.Step(ctx, params); err != nil {
				return nil, err
			}
		}
	}

	return model, nil
}
