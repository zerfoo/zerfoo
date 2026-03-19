package tabular

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/training/optimizer"
)

// LoRAConfig holds configuration for LoRA fine-tuning of a tabular model.
type LoRAConfig struct {
	Rank         int   // Low-rank dimension (typically 2-16 for tabular).
	Alpha        float32 // LoRA scaling factor.
	TargetLayers []int // Hidden layer indices to adapt; nil means all hidden layers.
	Epochs       int
	BatchSize    int
	LearningRate float64
	WeightDecay  float64
}

// loraLayerAdapter stores the low-rank A and B matrices for a single layer.
// The LoRA output is: scale * x @ A @ B, where scale = Alpha / Rank.
type loraLayerAdapter struct {
	A *tensor.TensorNumeric[float32] // [inDim, rank]
	B *tensor.TensorNumeric[float32] // [rank, outDim]
}

// Adapter holds LoRA adapter weights produced by FineTuneLoRA.
type Adapter struct {
	Layers map[int]loraLayerAdapter // hidden layer index → adapter
	Config LoRAConfig
	// Model architecture metadata for validation during merge.
	InputDim   int
	HiddenDims []int
}

// FineTuneLoRA applies Low-Rank Adaptation to a pre-trained BaseModel.
// Only the LoRA A and B matrices are trained; base model weights are frozen.
// This enables fast adaptation on small per-source datasets.
func FineTuneLoRA(
	base *BaseModel,
	data [][]float64,
	labels []int,
	config LoRAConfig,
	engine compute.Engine[float32],
	ops numeric.Arithmetic[float32],
) (*Adapter, error) {
	if base == nil || base.Model == nil {
		return nil, fmt.Errorf("tabular: lora: base model is nil")
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("tabular: lora: no data provided")
	}
	if len(data) != len(labels) {
		return nil, fmt.Errorf("tabular: lora: data length %d != labels length %d", len(data), len(labels))
	}
	if config.Rank <= 0 {
		return nil, fmt.Errorf("tabular: lora: rank must be positive, got %d", config.Rank)
	}
	if config.Alpha <= 0 {
		return nil, fmt.Errorf("tabular: lora: alpha must be positive, got %f", config.Alpha)
	}
	if config.Epochs <= 0 {
		return nil, fmt.Errorf("tabular: lora: epochs must be positive")
	}

	model := base.Model
	inputDim := model.config.InputDim

	// Validate input dimensions.
	for i, row := range data {
		if len(row) != inputDim {
			return nil, fmt.Errorf("tabular: lora: row %d has %d features, expected %d", i, len(row), inputDim)
		}
	}
	numClasses := 3
	for i, l := range labels {
		if l < 0 || l >= numClasses {
			return nil, fmt.Errorf("tabular: lora: label %d at index %d out of range [0, %d)", l, i, numClasses)
		}
	}

	// Determine which layers to adapt.
	targetSet := make(map[int]bool)
	if config.TargetLayers == nil {
		for i := range model.layers {
			targetSet[i] = true
		}
	} else {
		for _, idx := range config.TargetLayers {
			if idx < 0 || idx >= len(model.layers) {
				return nil, fmt.Errorf("tabular: lora: target layer %d out of range [0, %d)", idx, len(model.layers))
			}
			targetSet[idx] = true
		}
	}

	// Initialise LoRA adapters: A with Kaiming, B with zeros (standard LoRA init).
	adapters := make(map[int]loraLayerAdapter)
	for idx := range targetSet {
		l := model.layers[idx]
		wShape := l.weights.Shape()
		inDim := wShape[0]
		outDim := wShape[1]

		aData := make([]float32, inDim*config.Rank)
		scale := float32(math.Sqrt(2.0 / float64(inDim)))
		for i := range aData {
			aData[i] = float32(rand.NormFloat64()) * scale
		}
		a, err := tensor.New[float32]([]int{inDim, config.Rank}, aData)
		if err != nil {
			return nil, fmt.Errorf("tabular: lora: init A layer %d: %w", idx, err)
		}

		bData := make([]float32, config.Rank*outDim) // zeros
		b, err := tensor.New[float32]([]int{config.Rank, outDim}, bData)
		if err != nil {
			return nil, fmt.Errorf("tabular: lora: init B layer %d: %w", idx, err)
		}

		adapters[idx] = loraLayerAdapter{A: a, B: b}
	}

	// Build graph parameters for LoRA matrices only, in sorted layer order
	// to ensure deterministic param indexing.
	sortedLayerIndices := make([]int, 0, len(adapters))
	for idx := range adapters {
		sortedLayerIndices = append(sortedLayerIndices, idx)
	}
	sort.Ints(sortedLayerIndices)

	var loraParams []*graph.Parameter[float32]
	loraParamIndex := make(map[int]int) // layer index → position in loraParams (A at pos, B at pos+1)
	for _, idx := range sortedLayerIndices {
		adapter := adapters[idx]
		loraParamIndex[idx] = len(loraParams)
		ap, err := graph.NewParameter[float32](fmt.Sprintf("lora.%d.A", idx), adapter.A, tensor.New[float32])
		if err != nil {
			return nil, err
		}
		bp, err := graph.NewParameter[float32](fmt.Sprintf("lora.%d.B", idx), adapter.B, tensor.New[float32])
		if err != nil {
			return nil, err
		}
		loraParams = append(loraParams, ap, bp)
	}

	// Training loop.
	if config.BatchSize <= 0 {
		config.BatchSize = len(data)
	}
	lr := float32(config.LearningRate)
	if lr <= 0 {
		lr = 0.001
	}
	wd := float32(config.WeightDecay)
	opt := optimizer.NewAdamW[float32](engine, lr, 0.9, 0.999, 1e-8, wd)
	loraScale := config.Alpha / float32(config.Rank)
	ctx := context.Background()

	for epoch := 0; epoch < config.Epochs; epoch++ {
		perm := rand.Perm(len(data))

		for batchStart := 0; batchStart < len(data); batchStart += config.BatchSize {
			batchEnd := batchStart + config.BatchSize
			if batchEnd > len(data) {
				batchEnd = len(data)
			}
			batchSize := batchEnd - batchStart

			inputSlice := make([]float32, batchSize*inputDim)
			labelSlice := make([]int, batchSize)
			for i := 0; i < batchSize; i++ {
				idx := perm[batchStart+i]
				for j := 0; j < inputDim; j++ {
					inputSlice[i*inputDim+j] = float32(data[idx][j])
				}
				labelSlice[i] = labels[idx]
			}

			input, err := tensor.New[float32]([]int{batchSize, inputDim}, inputSlice)
			if err != nil {
				return nil, err
			}

			// Forward pass with LoRA.
			logits, activations, preActivations, loraIntermediate, err := loraForwardPass(ctx, model, adapters, loraScale, input, engine)
			if err != nil {
				return nil, err
			}

			// Cross-entropy loss.
			_, softmaxOut, err := crossEntropyLoss(ctx, engine, logits, labelSlice, batchSize, numClasses)
			if err != nil {
				return nil, err
			}

			// Backward pass for LoRA params only.
			err = loraBackwardPass(ctx, model, adapters, loraScale, engine, ops, loraParams, loraParamIndex,
				activations, preActivations, loraIntermediate, input, softmaxOut, labelSlice, batchSize, numClasses)
			if err != nil {
				return nil, err
			}

			if err := opt.Step(ctx, loraParams); err != nil {
				return nil, err
			}
		}
	}

	// Copy trained adapter weights into the result.
	resultLayers := make(map[int]loraLayerAdapter)
	for idx, adapter := range adapters {
		resultLayers[idx] = loraLayerAdapter{A: adapter.A, B: adapter.B}
	}

	return &Adapter{
		Layers:     resultLayers,
		Config:     config,
		InputDim:   inputDim,
		HiddenDims: append([]int(nil), model.config.HiddenDims...),
	}, nil
}

// loraForwardIntermediate stores per-layer intermediate values for LoRA backward.
type loraForwardIntermediate struct {
	xA map[int]*tensor.TensorNumeric[float32] // x @ A per LoRA layer
}

// loraForwardPass runs the forward pass with LoRA adapters injected.
func loraForwardPass(
	ctx context.Context,
	model *Model,
	adapters map[int]loraLayerAdapter,
	scale float32,
	input *tensor.TensorNumeric[float32],
	engine compute.Engine[float32],
) (
	logits *tensor.TensorNumeric[float32],
	activations []*tensor.TensorNumeric[float32],
	preActivations []*tensor.TensorNumeric[float32],
	intermediate loraForwardIntermediate,
	err error,
) {
	x := input
	n := len(model.layers)
	activations = make([]*tensor.TensorNumeric[float32], n)
	preActivations = make([]*tensor.TensorNumeric[float32], n)
	intermediate.xA = make(map[int]*tensor.TensorNumeric[float32])

	for i, l := range model.layers {
		// Base linear: x @ W + b
		preAct, fwdErr := model.linearForward(ctx, x, l)
		if fwdErr != nil {
			return nil, nil, nil, intermediate, fwdErr
		}

		// Add LoRA contribution if this layer has an adapter.
		if adapter, ok := adapters[i]; ok {
			// lora_out = scale * x @ A @ B
			xA, fwdErr := engine.MatMul(ctx, x, adapter.A)
			if fwdErr != nil {
				return nil, nil, nil, intermediate, fwdErr
			}
			intermediate.xA[i] = xA

			xAB, fwdErr := engine.MatMul(ctx, xA, adapter.B)
			if fwdErr != nil {
				return nil, nil, nil, intermediate, fwdErr
			}

			// Scale the LoRA output.
			scaleTensor, fwdErr := scalarTensor(scale, xAB.Shape())
			if fwdErr != nil {
				return nil, nil, nil, intermediate, fwdErr
			}
			scaled, fwdErr := engine.Mul(ctx, xAB, scaleTensor)
			if fwdErr != nil {
				return nil, nil, nil, intermediate, fwdErr
			}

			preAct, fwdErr = engine.Add(ctx, preAct, scaled)
			if fwdErr != nil {
				return nil, nil, nil, intermediate, fwdErr
			}
		}

		preActivations[i] = preAct

		postAct, actErr := model.applyActivation(ctx, preAct)
		if actErr != nil {
			return nil, nil, nil, intermediate, actErr
		}
		activations[i] = postAct
		x = postAct
	}

	logits, err = model.linearForward(ctx, x, model.head)
	if err != nil {
		return nil, nil, nil, intermediate, err
	}
	return logits, activations, preActivations, intermediate, nil
}

// loraBackwardPass computes gradients for LoRA A and B matrices only.
// Base model weights are frozen.
func loraBackwardPass(
	ctx context.Context,
	model *Model,
	adapters map[int]loraLayerAdapter,
	scale float32,
	engine compute.Engine[float32],
	ops numeric.Arithmetic[float32],
	loraParams []*graph.Parameter[float32],
	loraParamIdx map[int]int,
	activations []*tensor.TensorNumeric[float32],
	preActivations []*tensor.TensorNumeric[float32],
	intermediate loraForwardIntermediate,
	input *tensor.TensorNumeric[float32],
	softmaxOut *tensor.TensorNumeric[float32],
	labels []int,
	batchSize, numClasses int,
) error {
	// dLogits = (softmax - one_hot) / batchSize
	dLogitsData := make([]float32, batchSize*numClasses)
	smData := softmaxOut.Data()
	copy(dLogitsData, smData)
	batchScale := 1.0 / float32(batchSize)
	for i := 0; i < batchSize; i++ {
		dLogitsData[i*numClasses+labels[i]] -= 1.0
		for j := 0; j < numClasses; j++ {
			dLogitsData[i*numClasses+j] *= batchScale
		}
	}
	dLogits, err := tensor.New[float32]([]int{batchSize, numClasses}, dLogitsData)
	if err != nil {
		return err
	}

	nLayers := len(model.layers)

	// Propagate through head (frozen): dX = dLogits @ head.W^T
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

		// Compute LoRA gradients if this layer has an adapter.
		if adapter, ok := adapters[i]; ok {
			paramPos := loraParamIdx[i]
			aParam := loraParams[paramPos]
			bParam := loraParams[paramPos+1]

			xA := intermediate.xA[i]

			// dB = scale * (x @ A)^T @ dX = scale * xA^T @ dX, shape [rank, outDim]
			xAT, tErr := engine.Transpose(ctx, xA, []int{1, 0})
			if tErr != nil {
				return tErr
			}
			dB, mErr := engine.MatMul(ctx, xAT, dX)
			if mErr != nil {
				return mErr
			}
			dBScaled, sErr := scaleByScalar(ctx, engine, dB, scale)
			if sErr != nil {
				return sErr
			}
			if err := bParam.AddGradient(dBScaled); err != nil {
				return err
			}

			// dA = scale * layerInput^T @ (dX @ B^T), shape [inDim, rank]
			bT, tErr := engine.Transpose(ctx, adapter.B, []int{1, 0})
			if tErr != nil {
				return tErr
			}
			dXBT, mErr := engine.MatMul(ctx, dX, bT)
			if mErr != nil {
				return mErr
			}
			layerInputT, tErr := engine.Transpose(ctx, layerInput, []int{1, 0})
			if tErr != nil {
				return tErr
			}
			dA, mErr := engine.MatMul(ctx, layerInputT, dXBT)
			if mErr != nil {
				return mErr
			}
			dAScaled, sErr := scaleByScalar(ctx, engine, dA, scale)
			if sErr != nil {
				return sErr
			}
			if err := aParam.AddGradient(dAScaled); err != nil {
				return err
			}
		}

		// Propagate gradient through frozen base weights to previous layer.
		if i > 0 {
			weightsT, tErr := engine.Transpose(ctx, model.layers[i].weights, []int{1, 0})
			if tErr != nil {
				return tErr
			}
			baseDX, mErr := engine.MatMul(ctx, dX, weightsT)
			if mErr != nil {
				return mErr
			}

			// Also propagate through LoRA path if adapter exists.
			if adapter, ok := adapters[i]; ok {
				// dX_lora = scale * dX @ B^T @ A^T
				bT, tErr := engine.Transpose(ctx, adapter.B, []int{1, 0})
				if tErr != nil {
					return tErr
				}
				aT, tErr := engine.Transpose(ctx, adapter.A, []int{1, 0})
				if tErr != nil {
					return tErr
				}
				dXBT, mErr := engine.MatMul(ctx, dX, bT)
				if mErr != nil {
					return mErr
				}
				dXLora, mErr := engine.MatMul(ctx, dXBT, aT)
				if mErr != nil {
					return mErr
				}
				loraDX, sErr := scaleByScalar(ctx, engine, dXLora, scale)
				if sErr != nil {
					return sErr
				}
				baseDX, err = engine.Add(ctx, baseDX, loraDX)
				if err != nil {
					return err
				}
			}

			dX = baseDX
		}
	}

	return nil
}

// scalarTensor creates a tensor filled with the given scalar value.
func scalarTensor(val float32, shape []int) (*tensor.TensorNumeric[float32], error) {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = val
	}
	return tensor.New[float32](shape, data)
}

// scaleByScalar multiplies a tensor by a scalar value.
func scaleByScalar(ctx context.Context, engine compute.Engine[float32], t *tensor.TensorNumeric[float32], s float32) (*tensor.TensorNumeric[float32], error) {
	st, err := scalarTensor(s, t.Shape())
	if err != nil {
		return nil, err
	}
	return engine.Mul(ctx, t, st)
}

// MergeAdapter merges LoRA adapter weights into a BaseModel to produce a
// regular Model with no LoRA overhead during inference. The merged model
// produces identical predictions to running the base model with the adapter.
func MergeAdapter(base *BaseModel, adapter *Adapter, engine compute.Engine[float32]) (*Model, error) {
	if base == nil || base.Model == nil {
		return nil, fmt.Errorf("tabular: merge: base model is nil")
	}
	if adapter == nil {
		return nil, fmt.Errorf("tabular: merge: adapter is nil")
	}

	model := base.Model

	// Validate architecture compatibility.
	if adapter.InputDim != model.config.InputDim {
		return nil, fmt.Errorf("tabular: merge: input dim mismatch: adapter %d, model %d", adapter.InputDim, model.config.InputDim)
	}
	if len(adapter.HiddenDims) != len(model.config.HiddenDims) {
		return nil, fmt.Errorf("tabular: merge: hidden dims length mismatch: adapter %d, model %d", len(adapter.HiddenDims), len(model.config.HiddenDims))
	}
	for i, d := range adapter.HiddenDims {
		if d != model.config.HiddenDims[i] {
			return nil, fmt.Errorf("tabular: merge: hidden dim %d mismatch: adapter %d, model %d", i, d, model.config.HiddenDims[i])
		}
	}

	// Clone the base model.
	merged, err := cloneModel(model)
	if err != nil {
		return nil, fmt.Errorf("tabular: merge: %w", err)
	}

	ctx := context.Background()
	loraScale := adapter.Config.Alpha / float32(adapter.Config.Rank)

	// For each adapted layer, merge: W_new = W + scale * A @ B
	for idx, la := range adapter.Layers {
		if idx < 0 || idx >= len(merged.layers) {
			return nil, fmt.Errorf("tabular: merge: adapter layer %d out of range", idx)
		}

		// Compute A @ B: [inDim, rank] @ [rank, outDim] = [inDim, outDim]
		ab, err := engine.MatMul(ctx, la.A, la.B)
		if err != nil {
			return nil, fmt.Errorf("tabular: merge: layer %d matmul: %w", idx, err)
		}

		// Scale: scale * A @ B
		scaled, err := scaleByScalar(ctx, engine, ab, loraScale)
		if err != nil {
			return nil, fmt.Errorf("tabular: merge: layer %d scale: %w", idx, err)
		}

		// Add to base weights.
		newW, err := engine.Add(ctx, merged.layers[idx].weights, scaled)
		if err != nil {
			return nil, fmt.Errorf("tabular: merge: layer %d add: %w", idx, err)
		}
		merged.layers[idx].weights = newW
	}

	return merged, nil
}
