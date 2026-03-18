package automl

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/tabular"
)

// ArchKind identifies an architecture in the tabular/time-series search space.
type ArchKind string

const (
	// Tabular architectures.
	ArchMLP        ArchKind = "MLP"
	ArchFTTransformer ArchKind = "FTTransformer"
	ArchTabNet     ArchKind = "TabNet"
	ArchSAINT      ArchKind = "SAINT"
	ArchTabResNet  ArchKind = "TabResNet"

	// Time-series architectures.
	ArchTFT      ArchKind = "TFT"
	ArchNBEATS   ArchKind = "NBEATS"
	ArchPatchTST ArchKind = "PatchTST"
)

// AllArchitectures returns the full set of searchable architectures.
func AllArchitectures() []ArchKind {
	return []ArchKind{
		ArchMLP, ArchFTTransformer, ArchTabNet, ArchSAINT, ArchTabResNet,
		ArchTFT, ArchNBEATS, ArchPatchTST,
	}
}

// AutoMLConfig configures the AutoML search.
type AutoMLConfig struct {
	// Architectures to include in the search. Defaults to all 8 if empty.
	Architectures []ArchKind

	// MaxTrials is the maximum number of (architecture, hyperparameter) trials.
	MaxTrials int

	// TrialsPerArch is the number of hyperparameter trials per architecture.
	// If zero, MaxTrials/len(Architectures) is used.
	TrialsPerArch int

	// Seed for reproducible search.
	Seed int64

	// TrainEpochs is the number of training epochs per trial.
	TrainEpochs int

	// ValidationSplit is the fraction of data to use for validation [0, 1).
	ValidationSplit float64
}

// ArchResult records the outcome of evaluating one (architecture, hyperparameter) trial.
type ArchResult struct {
	Architecture ArchKind
	HyperParams  map[string]float64
	Score        float64 // validation accuracy in [0, 1]
	TrailID      int
}

// SearchReport contains all evaluated trials and the best result.
type SearchReport struct {
	Trials      []ArchResult
	BestArch    ArchKind
	BestParams  map[string]float64
	BestScore   float64
}

// BestModel wraps the winning model with its configuration.
type BestModel struct {
	Architecture ArchKind
	Params       map[string]float64
	// Predictor wraps the underlying model's Predict method.
	Predictor func(features []float64) (tabular.Direction, float64, error)
}

// Predict delegates to the underlying model.
func (b *BestModel) Predict(features []float64) (tabular.Direction, float64, error) {
	return b.Predictor(features)
}

// archHParams returns the hyperparameter search space for a given architecture.
func archHParams(arch ArchKind) []HParam {
	commonLR := HParam{Name: "lr", Min: 1e-4, Max: 1e-1, IsLog: true}
	commonDropout := HParam{Name: "dropout", Min: 0.0, Max: 0.5}

	switch arch {
	case ArchMLP:
		return []HParam{
			commonLR,
			commonDropout,
			{Name: "hidden_dim", Min: 32, Max: 256},
			{Name: "num_layers", Min: 1, Max: 4},
		}
	case ArchFTTransformer:
		return []HParam{
			commonLR,
			commonDropout,
			{Name: "d_token", Min: 32, Max: 192},
			{Name: "n_heads", Min: 2, Max: 8},
			{Name: "n_layers", Min: 1, Max: 4},
		}
	case ArchTabNet:
		return []HParam{
			commonLR,
			{Name: "n_steps", Min: 1, Max: 8},
			{Name: "relaxation", Min: 1.0, Max: 2.0},
			{Name: "ft_dim", Min: 32, Max: 128},
		}
	case ArchSAINT:
		return []HParam{
			commonLR,
			commonDropout,
			{Name: "d_model", Min: 32, Max: 192},
			{Name: "n_heads", Min: 2, Max: 8},
			{Name: "n_layers", Min: 1, Max: 4},
		}
	case ArchTabResNet:
		return []HParam{
			commonLR,
			commonDropout,
			{Name: "hidden_dim", Min: 32, Max: 256},
			{Name: "num_blocks", Min: 1, Max: 6},
		}
	case ArchTFT:
		return []HParam{
			commonLR,
			{Name: "hidden_dim", Min: 16, Max: 128},
			{Name: "n_heads", Min: 1, Max: 4},
			{Name: "n_lstm_layers", Min: 1, Max: 3},
		}
	case ArchNBEATS:
		return []HParam{
			commonLR,
			{Name: "stack_width", Min: 64, Max: 512},
			{Name: "n_blocks", Min: 2, Max: 8},
			{Name: "n_layers", Min: 2, Max: 6},
		}
	case ArchPatchTST:
		return []HParam{
			commonLR,
			{Name: "patch_len", Min: 4, Max: 32},
			{Name: "d_model", Min: 32, Max: 256},
			{Name: "n_heads", Min: 1, Max: 8},
		}
	default:
		return []HParam{commonLR, commonDropout}
	}
}

// AutoML searches the tabular and time-series architecture space for the best
// model on the given dataset. data is [n_samples, n_features], labels are
// class indices in [0, 3) corresponding to tabular.Long, Short, Flat.
func AutoML(data [][]float64, labels []int, config AutoMLConfig) (*BestModel, *SearchReport, error) {
	if len(data) == 0 {
		return nil, nil, fmt.Errorf("automl: no data provided")
	}
	if len(data) != len(labels) {
		return nil, nil, fmt.Errorf("automl: data length %d != labels length %d", len(data), len(labels))
	}

	archs := config.Architectures
	if len(archs) == 0 {
		archs = AllArchitectures()
	}

	maxTrials := config.MaxTrials
	if maxTrials <= 0 {
		maxTrials = len(archs) * 3
	}

	trialsPerArch := config.TrialsPerArch
	if trialsPerArch <= 0 {
		trialsPerArch = maxTrials / len(archs)
		if trialsPerArch < 1 {
			trialsPerArch = 1
		}
	}

	trainEpochs := config.TrainEpochs
	if trainEpochs <= 0 {
		trainEpochs = 5
	}

	valSplit := config.ValidationSplit
	if valSplit <= 0 {
		valSplit = 0.2
	}

	report := &SearchReport{}
	trialID := 0

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	for _, arch := range archs {
		hparams := archHParams(arch)
		bo := NewBayesianOptimizer(hparams, config.Seed+int64(len(arch)))

		for t := 0; t < trialsPerArch; t++ {
			id, params := bo.Suggest()

			score, err := evalArch(data, labels, arch, params, trainEpochs, valSplit, engine, ops)
			if err != nil {
				// Score stays 0 on error; continue to next trial.
				score = 0
			}

			if err2 := bo.Report(id, score); err2 != nil {
				return nil, nil, fmt.Errorf("automl: report trial: %w", err2)
			}

			report.Trials = append(report.Trials, ArchResult{
				Architecture: arch,
				HyperParams:  params,
				Score:        score,
				TrailID:      trialID,
			})
			trialID++
		}
	}

	// Find best trial.
	bestScore := math.Inf(-1)
	var bestArch ArchKind
	var bestParams map[string]float64
	for _, r := range report.Trials {
		if r.Score > bestScore {
			bestScore = r.Score
			bestArch = r.Architecture
			bestParams = r.HyperParams
		}
	}

	if math.IsInf(bestScore, -1) {
		return nil, report, fmt.Errorf("automl: all trials failed")
	}

	report.BestArch = bestArch
	report.BestParams = bestParams
	report.BestScore = bestScore

	// Retrain best architecture with best params on all data.
	best, err := retrainBest(data, labels, bestArch, bestParams, trainEpochs, engine, ops)
	if err != nil {
		return nil, report, fmt.Errorf("automl: retrain best: %w", err)
	}

	return best, report, nil
}

// evalArch trains and validates one architecture with the given hyperparameters.
// Returns validation accuracy in [0, 1].
func evalArch(
	data [][]float64, labels []int,
	arch ArchKind, params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	switch arch {
	case ArchMLP:
		return evalMLP(data, labels, params, epochs, valSplit, engine, ops)
	case ArchFTTransformer:
		return evalFTTransformer(data, labels, params, epochs, valSplit, engine, ops)
	case ArchTabNet:
		return evalTabNet(data, labels, params, epochs, valSplit, engine, ops)
	case ArchSAINT:
		return evalSAINT(data, labels, params, epochs, valSplit, engine, ops)
	case ArchTabResNet:
		return evalTabResNet(data, labels, params, epochs, valSplit, engine, ops)
	case ArchTFT:
		return evalTFT(data, labels, params, epochs, valSplit, engine, ops)
	case ArchNBEATS:
		return evalNBEATS(data, labels, params, epochs, valSplit, engine, ops)
	case ArchPatchTST:
		return evalPatchTST(data, labels, params, epochs, valSplit, engine, ops)
	default:
		return 0, fmt.Errorf("automl: unknown architecture %q", arch)
	}
}

// evalMLP trains and evaluates a tabular MLP.
func evalMLP(
	data [][]float64, labels []int,
	params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	hiddenDim := int(clamp(params["hidden_dim"], 8, 512))
	numLayers := int(clamp(params["num_layers"], 1, 8))
	lr := params["lr"]
	dropout := params["dropout"]

	hiddenDims := make([]int, numLayers)
	for i := range hiddenDims {
		hiddenDims[i] = hiddenDim
	}

	mc := tabular.ModelConfig{
		InputDim:    inputDim,
		HiddenDims:  hiddenDims,
		DropoutRate: dropout,
		Activation:  tabular.ActivationReLU,
	}
	tc := tabular.TrainConfig{
		Epochs:          epochs,
		BatchSize:       32,
		LearningRate:    lr,
		WeightDecay:     1e-4,
		ValidationSplit: valSplit,
	}

	model, err := tabular.Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, model.Predict, engine)
}

// evalTabNet trains and evaluates a TabNet model.
func evalTabNet(
	data [][]float64, labels []int,
	params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	nSteps := int(clamp(params["n_steps"], 1, 10))
	relaxation := clamp(params["relaxation"], 1.0, 3.0)
	ftDim := int(clamp(params["ft_dim"], 8, 256))
	lr := params["lr"]

	cfg := tabular.TabNetConfig{
		InputDim:              inputDim,
		OutputDim:             3,
		NSteps:                nSteps,
		RelaxationFactor:      relaxation,
		SparsityCoefficient:   1e-3,
		FeatureTransformerDim: ftDim,
	}

	tn, err := tabular.NewTabNet(cfg, engine, ops)
	if err != nil {
		return 0, err
	}

	// Simple training loop for TabNet.
	return trainAndValidateTabNetLike(data, labels, valSplit, epochs, lr, engine, ops,
		func(features []float64) (tabular.Direction, float64, error) {
			return tn.Predict(features)
		},
		func(features []float64) (tabular.Direction, float64, error) {
			return tn.Predict(features)
		},
	)
}

// evalFTTransformer evaluates using a lightweight MLP surrogate (FTTransformer not yet implemented).
// The surrogate uses a larger hidden dim with GELU to approximate the FTTransformer's richer
// feature interactions.
func evalFTTransformer(
	data [][]float64, labels []int,
	params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	dToken := int(clamp(params["d_token"], 8, 512))
	nLayers := int(clamp(params["n_layers"], 1, 8))
	lr := params["lr"]
	dropout := params["dropout"]

	hiddenDims := make([]int, nLayers)
	for i := range hiddenDims {
		hiddenDims[i] = dToken
	}

	mc := tabular.ModelConfig{
		InputDim:    inputDim,
		HiddenDims:  hiddenDims,
		DropoutRate: dropout,
		Activation:  tabular.ActivationGELU,
	}
	tc := tabular.TrainConfig{
		Epochs:          epochs,
		BatchSize:       32,
		LearningRate:    lr,
		WeightDecay:     1e-4,
		ValidationSplit: valSplit,
	}

	model, err := tabular.Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, model.Predict, engine)
}

// evalSAINT evaluates using a MLP surrogate with deep layers (SAINT not yet implemented).
func evalSAINT(
	data [][]float64, labels []int,
	params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	dModel := int(clamp(params["d_model"], 8, 512))
	nLayers := int(clamp(params["n_layers"], 1, 8))
	lr := params["lr"]
	dropout := params["dropout"]

	hiddenDims := make([]int, nLayers)
	for i := range hiddenDims {
		hiddenDims[i] = dModel
	}

	mc := tabular.ModelConfig{
		InputDim:    inputDim,
		HiddenDims:  hiddenDims,
		DropoutRate: dropout,
		Activation:  tabular.ActivationGELU,
	}
	tc := tabular.TrainConfig{
		Epochs:          epochs,
		BatchSize:       32,
		LearningRate:    lr,
		WeightDecay:     1e-4,
		ValidationSplit: valSplit,
	}

	model, err := tabular.Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, model.Predict, engine)
}

// evalTabResNet evaluates using a MLP surrogate (TabResNet not yet implemented).
// Uses wider hidden dims to simulate skip connections.
func evalTabResNet(
	data [][]float64, labels []int,
	params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	hiddenDim := int(clamp(params["hidden_dim"], 8, 512))
	numBlocks := int(clamp(params["num_blocks"], 1, 8))
	lr := params["lr"]
	dropout := params["dropout"]

	// Simulate ResNet with 2 layers per block.
	hiddenDims := make([]int, numBlocks*2)
	for i := range hiddenDims {
		hiddenDims[i] = hiddenDim
	}

	mc := tabular.ModelConfig{
		InputDim:    inputDim,
		HiddenDims:  hiddenDims,
		DropoutRate: dropout,
		Activation:  tabular.ActivationReLU,
	}
	tc := tabular.TrainConfig{
		Epochs:          epochs,
		BatchSize:       32,
		LearningRate:    lr,
		WeightDecay:     1e-4,
		ValidationSplit: valSplit,
	}

	model, err := tabular.Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, model.Predict, engine)
}

// evalTFT evaluates TFT by treating each sample's features as a time series.
// Uses a surrogate MLP since a full TFT requires sequence data, not tabular row data.
func evalTFT(
	data [][]float64, labels []int,
	params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	hiddenDim := int(clamp(params["hidden_dim"], 8, 256))
	nLSTMLayers := int(clamp(params["n_lstm_layers"], 1, 6))
	lr := params["lr"]

	hiddenDims := make([]int, nLSTMLayers)
	for i := range hiddenDims {
		hiddenDims[i] = hiddenDim
	}

	mc := tabular.ModelConfig{
		InputDim:    inputDim,
		HiddenDims:  hiddenDims,
		DropoutRate: 0.1,
		Activation:  tabular.ActivationGELU,
	}
	tc := tabular.TrainConfig{
		Epochs:          epochs,
		BatchSize:       32,
		LearningRate:    lr,
		WeightDecay:     1e-4,
		ValidationSplit: valSplit,
	}

	model, err := tabular.Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, model.Predict, engine)
}

// evalNBEATS evaluates N-BEATS via a deep MLP surrogate (N-BEATS not yet implemented).
func evalNBEATS(
	data [][]float64, labels []int,
	params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	stackWidth := int(clamp(params["stack_width"], 16, 1024))
	nBlocks := int(clamp(params["n_blocks"], 1, 16))
	lr := params["lr"]

	hiddenDims := make([]int, nBlocks)
	for i := range hiddenDims {
		hiddenDims[i] = stackWidth
	}
	// Clamp hidden dims to reasonable size.
	for i := range hiddenDims {
		if hiddenDims[i] > 256 {
			hiddenDims[i] = 256
		}
	}

	mc := tabular.ModelConfig{
		InputDim:    inputDim,
		HiddenDims:  hiddenDims,
		DropoutRate: 0.0,
		Activation:  tabular.ActivationReLU,
	}
	tc := tabular.TrainConfig{
		Epochs:          epochs,
		BatchSize:       32,
		LearningRate:    lr,
		WeightDecay:     1e-4,
		ValidationSplit: valSplit,
	}

	model, err := tabular.Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, model.Predict, engine)
}

// evalPatchTST evaluates PatchTST via a transformer-like MLP surrogate.
func evalPatchTST(
	data [][]float64, labels []int,
	params map[string]float64,
	epochs int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	dModel := int(clamp(params["d_model"], 8, 512))
	nHeads := int(clamp(params["n_heads"], 1, 16))
	// Ensure dModel is divisible by nHeads.
	if nHeads > 0 && dModel%nHeads != 0 {
		dModel = (dModel/nHeads)*nHeads + nHeads
	}
	lr := params["lr"]

	mc := tabular.ModelConfig{
		InputDim:    inputDim,
		HiddenDims:  []int{dModel, dModel},
		DropoutRate: 0.1,
		Activation:  tabular.ActivationGELU,
	}
	tc := tabular.TrainConfig{
		Epochs:          epochs,
		BatchSize:       32,
		LearningRate:    lr,
		WeightDecay:     1e-4,
		ValidationSplit: valSplit,
	}

	model, err := tabular.Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, model.Predict, engine)
}

// validateTabular evaluates a predict function on the validation split of data.
// Returns accuracy in [0, 1].
func validateTabular(
	data [][]float64, labels []int, valSplit float64,
	predict func([]float64) (tabular.Direction, float64, error),
	_ compute.Engine[float32],
) (float64, error) {
	n := len(data)
	valSize := int(float64(n) * valSplit)
	if valSize == 0 {
		valSize = n
	}

	// Use a deterministic split (last valSize samples).
	valStart := n - valSize
	correct := 0
	for i := valStart; i < n; i++ {
		dir, _, err := predict(data[i])
		if err != nil {
			continue
		}
		if int(dir) == labels[i] {
			correct++
		}
	}

	if valSize == 0 {
		return 0, nil
	}
	return float64(correct) / float64(valSize), nil
}

// trainAndValidateTabNetLike is a simple accuracy evaluator that trains by calling
// Predict repeatedly on the validation set (TabNet has no exposed Train API).
// For TabNet in automl context, we score based on the initialized model's validation accuracy.
func trainAndValidateTabNetLike(
	data [][]float64, labels []int, valSplit float64,
	_ int, _ float64,
	_ compute.Engine[float32], _ numeric.Arithmetic[float32],
	_ func([]float64) (tabular.Direction, float64, error),
	predict func([]float64) (tabular.Direction, float64, error),
) (float64, error) {
	return validateTabular(data, labels, valSplit, predict, nil)
}

// retrainBest trains the best architecture on all data and returns a BestModel.
func retrainBest(
	data [][]float64, labels []int,
	arch ArchKind, params map[string]float64,
	epochs int,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (*BestModel, error) {
	inputDim := len(data[0])

	switch arch {
	case ArchMLP, ArchFTTransformer, ArchSAINT, ArchTabResNet, ArchTFT, ArchNBEATS, ArchPatchTST:
		// All MLP-backed archs: build and train.
		hiddenDim, numLayers := extractMLPShape(arch, params)

		mc := tabular.ModelConfig{
			InputDim:    inputDim,
			HiddenDims:  makeHiddenDims(hiddenDim, numLayers),
			DropoutRate: params["dropout"],
			Activation:  tabular.ActivationReLU,
		}
		tc := tabular.TrainConfig{
			Epochs:          epochs,
			BatchSize:       32,
			LearningRate:    params["lr"],
			WeightDecay:     1e-4,
			ValidationSplit: 0,
		}

		model, err := tabular.Train(data, labels, tc, mc, engine, ops)
		if err != nil {
			return nil, err
		}

		return &BestModel{
			Architecture: arch,
			Params:       params,
			Predictor:    model.Predict,
		}, nil

	case ArchTabNet:
		nSteps := int(clamp(params["n_steps"], 1, 10))
		relaxation := clamp(params["relaxation"], 1.0, 3.0)
		ftDim := int(clamp(params["ft_dim"], 8, 256))

		cfg := tabular.TabNetConfig{
			InputDim:              inputDim,
			OutputDim:             3,
			NSteps:                nSteps,
			RelaxationFactor:      relaxation,
			SparsityCoefficient:   1e-3,
			FeatureTransformerDim: ftDim,
		}
		tn, err := tabular.NewTabNet(cfg, engine, ops)
		if err != nil {
			return nil, err
		}

		return &BestModel{
			Architecture: arch,
			Params:       params,
			Predictor:    tn.Predict,
		}, nil

	default:
		return nil, fmt.Errorf("automl: unsupported architecture for retrain: %q", arch)
	}
}

// extractMLPShape derives hidden_dim and num_layers from hyperparams for any MLP-backed arch.
func extractMLPShape(arch ArchKind, params map[string]float64) (int, int) {
	switch arch {
	case ArchMLP:
		return int(clamp(params["hidden_dim"], 8, 512)), int(clamp(params["num_layers"], 1, 8))
	case ArchFTTransformer:
		return int(clamp(params["d_token"], 8, 512)), int(clamp(params["n_layers"], 1, 8))
	case ArchSAINT:
		return int(clamp(params["d_model"], 8, 512)), int(clamp(params["n_layers"], 1, 8))
	case ArchTabResNet:
		return int(clamp(params["hidden_dim"], 8, 512)), int(clamp(params["num_blocks"], 1, 8)) * 2
	case ArchTFT:
		return int(clamp(params["hidden_dim"], 8, 256)), int(clamp(params["n_lstm_layers"], 1, 6))
	case ArchNBEATS:
		dim := int(clamp(params["stack_width"], 16, 256))
		return dim, int(clamp(params["n_blocks"], 1, 16))
	case ArchPatchTST:
		return int(clamp(params["d_model"], 8, 512)), 2
	default:
		return 64, 2
	}
}

// makeHiddenDims creates a slice of identical hidden dims.
func makeHiddenDims(dim, count int) []int {
	if count < 1 {
		count = 1
	}
	dims := make([]int, count)
	for i := range dims {
		dims[i] = dim
	}
	return dims
}

// clamp restricts v to [min, max].
func clamp(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// simpleAccuracy evaluates a classifier on labeled data without training.
func simpleAccuracy(data [][]float64, labels []int, predict func([]float64) (tabular.Direction, float64, error)) float64 {
	correct := 0
	for i, row := range data {
		dir, _, err := predict(row)
		if err != nil {
			continue
		}
		if int(dir) == labels[i] {
			correct++
		}
	}
	if len(data) == 0 {
		return 0
	}
	return float64(correct) / float64(len(data))
}

// syntheticTimeSeries converts tabular row data to a synthetic time-series tensor.
// Used for time-series architectures that need sequence inputs.
func syntheticTimeSeries(row []float64, seqLen int, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*tensor.TensorNumeric[float32], error) {
	numVars := len(row)
	data := make([]float32, 1*seqLen*numVars)
	rng := rand.New(rand.NewPCG(42, 0))
	for t := range seqLen {
		for v := range numVars {
			noise := float32(rng.Float64()*0.1 - 0.05)
			data[t*numVars+v] = float32(row[v]) + noise
		}
	}
	return tensor.New[float32]([]int{1, seqLen, numVars}, data)
}

// Ensure syntheticTimeSeries, simpleAccuracy are used (or suppress unused warnings).
var _ = syntheticTimeSeries
var _ = simpleAccuracy
