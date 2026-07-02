package tabular

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// TrainConfig holds hyperparameters for training.
type TrainConfig struct {
	Epochs          int
	BatchSize       int
	LearningRate    float64
	WeightDecay     float64
	ValidationSplit float64
}

// Train trains a tabular Model on the given data and labels using AdamW and
// cross-entropy loss. labels[i] must be in [0, 3) corresponding to Long,
// Short, Flat. It returns a trained Model ready for Predict.
func Train(data [][]float64, labels []int, config TrainConfig, mc ModelConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Model, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("tabular: train: no data provided")
	}
	if len(data) != len(labels) {
		return nil, fmt.Errorf("tabular: train: data length %d != labels length %d", len(data), len(labels))
	}
	if config.Epochs <= 0 {
		return nil, fmt.Errorf("tabular: train: Epochs must be positive")
	}
	if config.BatchSize <= 0 {
		config.BatchSize = len(data)
	}
	if config.LearningRate <= 0 {
		config.LearningRate = 0.01
	}
	if config.ValidationSplit < 0 || config.ValidationSplit >= 1 {
		return nil, fmt.Errorf("tabular: train: ValidationSplit must be in [0, 1)")
	}

	numClasses := 3
	inputDim := len(data[0])
	for i, row := range data {
		if len(row) != inputDim {
			return nil, fmt.Errorf("tabular: train: row %d has %d features, expected %d", i, len(row), inputDim)
		}
	}
	for i, l := range labels {
		if l < 0 || l >= numClasses {
			return nil, fmt.Errorf("tabular: train: label %d at index %d is out of range [0, %d)", l, i, numClasses)
		}
	}

	mc.InputDim = inputDim

	// Split into train/validation.
	trainData, trainLabels, valData, valLabels := splitData(data, labels, config.ValidationSplit)

	// Create model.
	model, err := NewModel(mc, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("tabular: train: %w", err)
	}

	// Wrap model weights as graph.Parameter for AdamW.
	params, err := buildParams(model)
	if err != nil {
		return nil, fmt.Errorf("tabular: train: %w", err)
	}

	// Create AdamW optimizer.
	lr := float32(config.LearningRate)
	wd := float32(config.WeightDecay)
	opt := optimizer.NewAdamW[float32](engine, lr, 0.9, 0.999, 1e-8, wd)

	ctx := context.Background()

	for epoch := 0; epoch < config.Epochs; epoch++ {
		// Shuffle training data.
		perm := rand.Perm(len(trainData))

		var epochLoss float64
		numBatches := 0

		for batchStart := 0; batchStart < len(trainData); batchStart += config.BatchSize {
			batchEnd := batchStart + config.BatchSize
			if batchEnd > len(trainData) {
				batchEnd = len(trainData)
			}

			batchSize := batchEnd - batchStart

			// Build batch tensors.
			inputData := make([]float32, batchSize*inputDim)
			labelData := make([]int, batchSize)
			for i := 0; i < batchSize; i++ {
				idx := perm[batchStart+i]
				for j := 0; j < inputDim; j++ {
					inputData[i*inputDim+j] = float32(trainData[idx][j])
				}
				labelData[i] = trainLabels[idx]
			}

			input, err := tensor.New[float32]([]int{batchSize, inputDim}, inputData)
			if err != nil {
				return nil, err
			}

			// Forward pass.
			logits, activations, preActivations, err := forwardPass(ctx, model, input)
			if err != nil {
				return nil, fmt.Errorf("tabular: train: forward: %w", err)
			}

			// Compute softmax cross-entropy loss.
			loss, softmaxOut, err := crossEntropyLoss(ctx, engine, logits, labelData, batchSize, numClasses)
			if err != nil {
				return nil, fmt.Errorf("tabular: train: loss: %w", err)
			}
			epochLoss += loss

			// Backward pass: compute gradients.
			err = backwardPass(ctx, model, engine, ops, params, activations, preActivations, input, softmaxOut, labelData, batchSize, numClasses)
			if err != nil {
				return nil, fmt.Errorf("tabular: train: backward: %w", err)
			}

			// AdamW step.
			if err := opt.Step(ctx, params); err != nil {
				return nil, fmt.Errorf("tabular: train: optimizer step: %w", err)
			}

			numBatches++
		}

		avgLoss := epochLoss / float64(numBatches)
		logLine := slog.With("epoch", epoch+1, "loss", avgLoss)

		if len(valData) > 0 {
			valLoss, valAcc, err := evaluate(ctx, model, engine, valData, valLabels, inputDim, numClasses)
			if err != nil {
				return nil, fmt.Errorf("tabular: train: validation: %w", err)
			}
			logLine = logLine.With("val_loss", valLoss, "val_accuracy", valAcc)
		}

		logLine.Info("tabular: training epoch")
	}

	return model, nil
}

// splitData splits data into train and validation sets.
func splitData(data [][]float64, labels []int, valSplit float64) ([][]float64, []int, [][]float64, []int) {
	if valSplit <= 0 {
		return data, labels, nil, nil
	}
	n := len(data)
	valSize := int(float64(n) * valSplit)
	if valSize == 0 {
		return data, labels, nil, nil
	}

	// Shuffle indices for random split.
	perm := rand.Perm(n)
	trainData := make([][]float64, 0, n-valSize)
	trainLabels := make([]int, 0, n-valSize)
	valData := make([][]float64, 0, valSize)
	valLabels := make([]int, 0, valSize)

	for i, idx := range perm {
		if i < valSize {
			valData = append(valData, data[idx])
			valLabels = append(valLabels, labels[idx])
		} else {
			trainData = append(trainData, data[idx])
			trainLabels = append(trainLabels, labels[idx])
		}
	}
	return trainData, trainLabels, valData, valLabels
}

// buildParams wraps model weights and biases as graph.Parameter for use with AdamW.
func buildParams(model *Model) ([]*graph.Parameter[float32], error) {
	var params []*graph.Parameter[float32]
	for i, l := range model.layers {
		wp, err := graph.NewParameter[float32](fmt.Sprintf("layer%d.weights", i), l.weights, tensor.New[float32])
		if err != nil {
			return nil, err
		}
		bp, err := graph.NewParameter[float32](fmt.Sprintf("layer%d.biases", i), l.biases, tensor.New[float32])
		if err != nil {
			return nil, err
		}
		params = append(params, wp, bp)
	}
	wp, err := graph.NewParameter[float32]("head.weights", model.head.weights, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	bp, err := graph.NewParameter[float32]("head.biases", model.head.biases, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	params = append(params, wp, bp)
	return params, nil
}

// forwardPass runs the forward pass and caches intermediate values for backprop.
// Returns logits, activations (post-activation outputs per hidden layer), and
// preActivations (pre-activation outputs per hidden layer).
func forwardPass(ctx context.Context, model *Model, input *tensor.TensorNumeric[float32]) (
	logits *tensor.TensorNumeric[float32],
	activations []*tensor.TensorNumeric[float32],
	preActivations []*tensor.TensorNumeric[float32],
	err error,
) {
	x := input
	activations = make([]*tensor.TensorNumeric[float32], len(model.layers))
	preActivations = make([]*tensor.TensorNumeric[float32], len(model.layers))

	for i, l := range model.layers {
		preAct, fwdErr := model.linearForward(ctx, x, l)
		if fwdErr != nil {
			return nil, nil, nil, fwdErr
		}
		preActivations[i] = preAct

		postAct, actErr := model.applyActivation(ctx, preAct)
		if actErr != nil {
			return nil, nil, nil, actErr
		}
		activations[i] = postAct
		x = postAct
	}

	logits, err = model.linearForward(ctx, x, model.head)
	if err != nil {
		return nil, nil, nil, err
	}
	return logits, activations, preActivations, nil
}

// crossEntropyLoss computes the softmax cross-entropy loss using training/loss.CrossEntropyLoss.
// Returns scalar loss, softmax output tensor, and error.
func crossEntropyLoss(ctx context.Context, engine compute.Engine[float32], logits *tensor.TensorNumeric[float32], labels []int, batchSize, numClasses int) (float64, *tensor.TensorNumeric[float32], error) {
	// Convert int labels to float32 tensor for the canonical CrossEntropyLoss.
	labelF32 := make([]float32, batchSize)
	for i, l := range labels {
		labelF32[i] = float32(l)
	}
	targetTensor, err := tensor.New[float32]([]int{batchSize}, labelF32)
	if err != nil {
		return 0, nil, err
	}

	cel := loss.NewCrossEntropyLoss[float32](engine)
	lossTensor, err := cel.Forward(ctx, logits, targetTensor)
	if err != nil {
		return 0, nil, err
	}

	// Extract scalar loss value.
	lossVal := float64(lossTensor.Data()[0])

	return lossVal, cel.SoftmaxOutput(), nil
}

// backwardPass computes gradients and accumulates them into the parameter gradient tensors.
func backwardPass(
	ctx context.Context,
	model *Model,
	engine compute.Engine[float32],
	ops numeric.Arithmetic[float32],
	params []*graph.Parameter[float32],
	activations []*tensor.TensorNumeric[float32],
	preActivations []*tensor.TensorNumeric[float32],
	input *tensor.TensorNumeric[float32],
	softmaxOut *tensor.TensorNumeric[float32],
	labels []int,
	batchSize, numClasses int,
) error {
	// dLogits = softmax - one_hot(labels), shape [batchSize, numClasses].
	dLogitsData := make([]float32, batchSize*numClasses)
	smData := softmaxOut.Data()
	copy(dLogitsData, smData)
	scale := 1.0 / float32(batchSize)
	for i := 0; i < batchSize; i++ {
		dLogitsData[i*numClasses+labels[i]] -= 1.0
		for j := 0; j < numClasses; j++ {
			dLogitsData[i*numClasses+j] *= scale
		}
	}
	dLogits, err := tensor.New[float32]([]int{batchSize, numClasses}, dLogitsData)
	if err != nil {
		return err
	}

	// Number of hidden layers.
	nLayers := len(model.layers)

	// Backprop through output head.
	// head params are the last 2 in params slice.
	headWParam := params[nLayers*2]
	headBParam := params[nLayers*2+1]

	// Input to head is the last hidden activation.
	var headInput *tensor.TensorNumeric[float32]
	if nLayers > 0 {
		headInput = activations[nLayers-1]
	} else {
		headInput = input
	}

	// dW_head = headInput^T @ dLogits, shape [lastHidden, numClasses].
	headInputT, err := engine.Transpose(ctx, headInput, []int{1, 0})
	if err != nil {
		return err
	}
	dWHead, err := engine.MatMul(ctx, headInputT, dLogits)
	if err != nil {
		return err
	}
	if err := headWParam.AddGradient(dWHead); err != nil {
		return err
	}

	// dB_head = sum(dLogits, axis=0), shape [1, numClasses].
	dBHead, err := engine.ReduceSum(ctx, dLogits, 0, true)
	if err != nil {
		return err
	}
	if err := headBParam.AddGradient(dBHead); err != nil {
		return err
	}

	// dX = dLogits @ head.weights^T, shape [batchSize, lastHidden].
	headWeightsT, err := engine.Transpose(ctx, model.head.weights, []int{1, 0})
	if err != nil {
		return err
	}
	dX, err := engine.MatMul(ctx, dLogits, headWeightsT)
	if err != nil {
		return err
	}

	// Backprop through hidden layers in reverse.
	for i := nLayers - 1; i >= 0; i-- {
		wParam := params[i*2]
		bParam := params[i*2+1]

		// Apply activation gradient.
		dX, err = activationBackward(ctx, engine, ops, model.config.Activation, dX, preActivations[i])
		if err != nil {
			return err
		}

		// Layer input.
		var layerInput *tensor.TensorNumeric[float32]
		if i > 0 {
			layerInput = activations[i-1]
		} else {
			layerInput = input
		}

		// dW = layerInput^T @ dX.
		layerInputT, err := engine.Transpose(ctx, layerInput, []int{1, 0})
		if err != nil {
			return err
		}
		dW, err := engine.MatMul(ctx, layerInputT, dX)
		if err != nil {
			return err
		}
		if err := wParam.AddGradient(dW); err != nil {
			return err
		}

		// dB = sum(dX, axis=0).
		dB, err := engine.ReduceSum(ctx, dX, 0, true)
		if err != nil {
			return err
		}
		if err := bParam.AddGradient(dB); err != nil {
			return err
		}

		// Propagate to previous layer.
		if i > 0 {
			weightsT, err := engine.Transpose(ctx, model.layers[i].weights, []int{1, 0})
			if err != nil {
				return err
			}
			dX, err = engine.MatMul(ctx, dX, weightsT)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// activationBackward computes the element-wise activation gradient.
func activationBackward(ctx context.Context, engine compute.Engine[float32], ops numeric.Arithmetic[float32], act Activation, dOut, preAct *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	switch act {
	case ActivationReLU:
		grad, err := engine.UnaryOp(ctx, preAct, ops.ReLUGrad)
		if err != nil {
			return nil, err
		}
		return engine.Mul(ctx, dOut, grad)
	case ActivationGELU:
		return functional.GELUBackward(ctx, engine, ops, dOut, preAct)
	default:
		grad, err := engine.UnaryOp(ctx, preAct, ops.ReLUGrad)
		if err != nil {
			return nil, err
		}
		return engine.Mul(ctx, dOut, grad)
	}
}

// evaluate runs a forward pass on validation data and returns loss and accuracy.
func evaluate(ctx context.Context, model *Model, engine compute.Engine[float32], data [][]float64, labels []int, inputDim, numClasses int) (float64, float64, error) {
	n := len(data)
	inputData := make([]float32, n*inputDim)
	for i, row := range data {
		for j, v := range row {
			inputData[i*inputDim+j] = float32(v)
		}
	}
	input, err := tensor.New[float32]([]int{n, inputDim}, inputData)
	if err != nil {
		return 0, 0, err
	}

	logits, _, _, err := forwardPass(ctx, model, input)
	if err != nil {
		return 0, 0, err
	}

	loss, softmaxOut, err := crossEntropyLoss(ctx, engine, logits, labels, n, numClasses)
	if err != nil {
		return 0, 0, err
	}

	// Compute accuracy.
	probs := softmaxOut.Data()
	correct := 0
	for i := 0; i < n; i++ {
		best := 0
		for j := 1; j < numClasses; j++ {
			if probs[i*numClasses+j] > probs[i*numClasses+best] {
				best = j
			}
		}
		if best == labels[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(n)

	return loss, acc, nil
}
