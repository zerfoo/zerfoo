package automl

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/zerfoo/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// ArchKind identifies an architecture in the tabular/time-series search space.
type ArchKind string

const (
	// Tabular architectures.
	ArchMLP           ArchKind = "MLP"
	ArchFTTransformer ArchKind = "FTTransformer"
	ArchTabNet        ArchKind = "TabNet"
	ArchSAINT         ArchKind = "SAINT"
	ArchTabResNet     ArchKind = "TabResNet"

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
	Trials     []ArchResult
	BestArch   ArchKind
	BestParams map[string]float64
	BestScore  float64
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

// evalTabNet creates and evaluates a TabNet model.
func evalTabNet(
	data [][]float64, labels []int,
	params map[string]float64,
	_ int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
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
		return 0, err
	}

	return validateTabular(data, labels, valSplit, tn.Predict, engine)
}

// evalFTTransformer creates and evaluates an actual FTTransformer model.
func evalFTTransformer(
	data [][]float64, labels []int,
	params map[string]float64,
	_ int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	dToken := int(clamp(params["d_token"], 8, 192))
	nHeads := int(clamp(params["n_heads"], 2, 8))
	nLayers := int(clamp(params["n_layers"], 1, 4))
	dropout := params["dropout"]

	dToken = alignDim(dToken, nHeads)

	cfg := tabular.FTTransformerConfig{
		NumFeatures: inputDim,
		DToken:      dToken,
		NHeads:      nHeads,
		NLayers:     nLayers,
		DFFN:        dToken * 4,
		DropoutRate: dropout,
	}

	ft, err := tabular.NewFTTransformer(cfg, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, ft.Predict, engine)
}

// evalSAINT creates and evaluates an actual SAINT model.
func evalSAINT(
	data [][]float64, labels []int,
	params map[string]float64,
	_ int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	dModel := int(clamp(params["d_model"], 8, 192))
	nHeads := int(clamp(params["n_heads"], 2, 8))
	nLayers := int(clamp(params["n_layers"], 1, 4))

	dModel = alignDim(dModel, nHeads)

	cfg := tabular.SAINTConfig{
		NumFeatures:          inputDim,
		DModel:               dModel,
		NHeads:               nHeads,
		NLayers:              nLayers,
		InterSampleAttention: true,
	}

	s, err := tabular.NewSAINT(cfg, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, s.Predict, engine)
}

// evalTabResNet creates and evaluates an actual TabResNet model.
func evalTabResNet(
	data [][]float64, labels []int,
	params map[string]float64,
	_ int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	hiddenDim := int(clamp(params["hidden_dim"], 8, 256))
	numBlocks := int(clamp(params["num_blocks"], 1, 6))
	dropout := params["dropout"]

	hiddenDims := make([]int, numBlocks)
	for i := range hiddenDims {
		hiddenDims[i] = hiddenDim
	}

	cfg := tabular.TabResNetConfig{
		InputDim:    inputDim,
		OutputDim:   3,
		HiddenDims:  hiddenDims,
		DropoutRate: dropout,
		Activation:  tabular.ActivationReLU,
		Norm:        tabular.NormLayer,
	}

	rn, err := tabular.NewTabResNet(cfg, engine, ops)
	if err != nil {
		return 0, err
	}

	return validateTabular(data, labels, valSplit, rn.Predict, engine)
}

// evalTFT creates and evaluates an actual Temporal Fusion Transformer.
// Tabular features are split into static and time covariates for the TFT interface.
func evalTFT(
	data [][]float64, labels []int,
	params map[string]float64,
	_ int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	hiddenDim := int(clamp(params["hidden_dim"], 16, 128))
	nHeads := int(clamp(params["n_heads"], 1, 4))

	hiddenDim = alignDim(hiddenDim, nHeads)

	// Split features: first half static, second half temporal (1 time step).
	numStatic := inputDim / 2
	if numStatic < 1 {
		numStatic = 1
	}
	numTime := inputDim - numStatic
	if numTime < 1 {
		numTime = 1
		numStatic = inputDim - 1
		if numStatic < 1 {
			numStatic = 1
		}
	}

	cfg := timeseries.TFTConfig{
		NumStaticFeatures: numStatic,
		NumTimeFeatures:   numTime,
		DModel:            hiddenDim,
		NHeads:            nHeads,
		NHorizons:         3,
		Quantiles:         []float64{0.5},
	}

	tft, err := timeseries.NewTFT(cfg, engine, ops)
	if err != nil {
		return 0, err
	}

	predict := func(features []float64) (tabular.Direction, float64, error) {
		staticF := features[:numStatic]
		timeF := [][]float64{features[numStatic : numStatic+numTime]}
		out, err := tft.Predict(staticF, timeF)
		if err != nil {
			return tabular.Flat, 0, err
		}
		// out is [NHorizons][nQuantiles] = [3][1]; treat horizon values as logits.
		logits := make([]float32, len(out))
		for i, row := range out {
			if len(row) > 0 {
				logits[i] = float32(row[0])
			}
		}
		dir, conf := argmax(softmax(logits))
		return dir, conf, nil
	}

	return validateTabular(data, labels, valSplit, predict, engine)
}

// evalNBEATS creates and evaluates an actual N-BEATS model.
// Tabular features are treated as a univariate time series (features = time steps),
// and the forecast output (length 3) is interpreted as class logits.
func evalNBEATS(
	data [][]float64, labels []int,
	params map[string]float64,
	_ int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	stackWidth := int(clamp(params["stack_width"], 64, 256))
	nBlocks := int(clamp(params["n_blocks"], 2, 8))

	cfg := timeseries.NBEATSConfig{
		InputLength:     inputDim,
		OutputLength:    3, // 3 outputs → class logits
		StackTypes:      []timeseries.StackType{timeseries.StackGeneric},
		NBlocksPerStack: nBlocks,
		HiddenDim:       stackWidth,
		NHarmonics:      4,
	}

	nb, err := timeseries.NewNBEATS(cfg, engine, ops)
	if err != nil {
		return 0, err
	}

	ctx := context.Background()

	predict := func(features []float64) (tabular.Direction, float64, error) {
		f32 := make([]float32, len(features))
		for i, v := range features {
			f32[i] = float32(v)
		}
		input, err := tensor.New[float32]([]int{1, inputDim}, f32)
		if err != nil {
			return tabular.Flat, 0, err
		}
		out, err := nb.Forward(ctx, input)
		if err != nil {
			return tabular.Flat, 0, err
		}
		dir, conf := argmax(softmax(out.Forecast.Data()))
		return dir, conf, nil
	}

	return validateTabular(data, labels, valSplit, predict, engine)
}

// evalPatchTST creates and evaluates an actual PatchTST model.
// Tabular features are treated as a single-channel time series,
// and the output dimension is set to 3 for classification.
func evalPatchTST(
	data [][]float64, labels []int,
	params map[string]float64,
	_ int, valSplit float64,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (float64, error) {
	inputDim := len(data[0])
	patchLen := int(clamp(params["patch_len"], 1, float64(inputDim)))
	dModel := int(clamp(params["d_model"], 8, 256))
	nHeads := int(clamp(params["n_heads"], 1, 8))

	dModel = alignDim(dModel, nHeads)

	stride := patchLen / 2
	if stride < 1 {
		stride = 1
	}

	cfg := timeseries.PatchTSTConfig{
		InputLength:        inputDim,
		PatchLength:        patchLen,
		Stride:             stride,
		DModel:             dModel,
		NHeads:             nHeads,
		NLayers:            2,
		OutputDim:          3, // 3 outputs → class logits
		ChannelIndependent: false,
	}

	pt, err := timeseries.NewPatchTST(cfg, engine, ops)
	if err != nil {
		return 0, err
	}

	predict := func(features []float64) (tabular.Direction, float64, error) {
		// Single channel: features as time points.
		out, err := pt.Predict([][]float64{features})
		if err != nil {
			return tabular.Flat, 0, err
		}
		if len(out) == 0 || len(out[0]) < 3 {
			return tabular.Flat, 0, fmt.Errorf("automl: patchtst output too short")
		}
		logits := make([]float32, 3)
		for i := 0; i < 3; i++ {
			logits[i] = float32(out[0][i])
		}
		dir, conf := argmax(softmax(logits))
		return dir, conf, nil
	}

	return validateTabular(data, labels, valSplit, predict, engine)
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

// retrainBest creates the best architecture with the winning hyperparameters
// and returns a BestModel. For architectures with training support (MLP),
// the model is trained on all data. For others, the model is instantiated
// with the validated hyperparameters.
func retrainBest(
	data [][]float64, labels []int,
	arch ArchKind, params map[string]float64,
	epochs int,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32],
) (*BestModel, error) {
	inputDim := len(data[0])

	switch arch {
	case ArchMLP:
		hiddenDim := int(clamp(params["hidden_dim"], 8, 512))
		numLayers := int(clamp(params["num_layers"], 1, 8))

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
		return &BestModel{Architecture: arch, Params: params, Predictor: model.Predict}, nil

	case ArchFTTransformer:
		dToken := int(clamp(params["d_token"], 8, 192))
		nHeads := int(clamp(params["n_heads"], 2, 8))
		nLayers := int(clamp(params["n_layers"], 1, 4))
		dToken = alignDim(dToken, nHeads)

		ft, err := tabular.NewFTTransformer(tabular.FTTransformerConfig{
			NumFeatures: inputDim, DToken: dToken, NHeads: nHeads,
			NLayers: nLayers, DFFN: dToken * 4, DropoutRate: params["dropout"],
		}, engine, ops)
		if err != nil {
			return nil, err
		}
		return &BestModel{Architecture: arch, Params: params, Predictor: ft.Predict}, nil

	case ArchTabNet:
		nSteps := int(clamp(params["n_steps"], 1, 10))
		relaxation := clamp(params["relaxation"], 1.0, 3.0)
		ftDim := int(clamp(params["ft_dim"], 8, 256))

		tn, err := tabular.NewTabNet(tabular.TabNetConfig{
			InputDim: inputDim, OutputDim: 3, NSteps: nSteps,
			RelaxationFactor: relaxation, SparsityCoefficient: 1e-3,
			FeatureTransformerDim: ftDim,
		}, engine, ops)
		if err != nil {
			return nil, err
		}
		return &BestModel{Architecture: arch, Params: params, Predictor: tn.Predict}, nil

	case ArchSAINT:
		dModel := int(clamp(params["d_model"], 8, 192))
		nHeads := int(clamp(params["n_heads"], 2, 8))
		nLayers := int(clamp(params["n_layers"], 1, 4))
		dModel = alignDim(dModel, nHeads)

		s, err := tabular.NewSAINT(tabular.SAINTConfig{
			NumFeatures: inputDim, DModel: dModel, NHeads: nHeads,
			NLayers: nLayers, InterSampleAttention: true,
		}, engine, ops)
		if err != nil {
			return nil, err
		}
		return &BestModel{Architecture: arch, Params: params, Predictor: s.Predict}, nil

	case ArchTabResNet:
		hiddenDim := int(clamp(params["hidden_dim"], 8, 256))
		numBlocks := int(clamp(params["num_blocks"], 1, 6))

		rn, err := tabular.NewTabResNet(tabular.TabResNetConfig{
			InputDim: inputDim, OutputDim: 3,
			HiddenDims:  makeHiddenDims(hiddenDim, numBlocks),
			DropoutRate: params["dropout"], Activation: tabular.ActivationReLU,
			Norm: tabular.NormLayer,
		}, engine, ops)
		if err != nil {
			return nil, err
		}
		return &BestModel{Architecture: arch, Params: params, Predictor: rn.Predict}, nil

	case ArchTFT:
		hiddenDim := int(clamp(params["hidden_dim"], 16, 128))
		nHeads := int(clamp(params["n_heads"], 1, 4))
		hiddenDim = alignDim(hiddenDim, nHeads)
		numStatic := inputDim / 2
		if numStatic < 1 {
			numStatic = 1
		}
		numTime := inputDim - numStatic
		if numTime < 1 {
			numTime = 1
		}

		tft, err := timeseries.NewTFT(timeseries.TFTConfig{
			NumStaticFeatures: numStatic, NumTimeFeatures: numTime,
			DModel: hiddenDim, NHeads: nHeads, NHorizons: 3,
			Quantiles: []float64{0.5},
		}, engine, ops)
		if err != nil {
			return nil, err
		}
		predict := func(features []float64) (tabular.Direction, float64, error) {
			staticF := features[:numStatic]
			timeF := [][]float64{features[numStatic : numStatic+numTime]}
			out, err := tft.Predict(staticF, timeF)
			if err != nil {
				return tabular.Flat, 0, err
			}
			logits := make([]float32, len(out))
			for i, row := range out {
				if len(row) > 0 {
					logits[i] = float32(row[0])
				}
			}
			dir, conf := argmax(softmax(logits))
			return dir, conf, nil
		}
		return &BestModel{Architecture: arch, Params: params, Predictor: predict}, nil

	case ArchNBEATS:
		stackWidth := int(clamp(params["stack_width"], 64, 256))
		nBlocks := int(clamp(params["n_blocks"], 2, 8))

		nb, err := timeseries.NewNBEATS(timeseries.NBEATSConfig{
			InputLength: inputDim, OutputLength: 3,
			StackTypes:      []timeseries.StackType{timeseries.StackGeneric},
			NBlocksPerStack: nBlocks, HiddenDim: stackWidth, NHarmonics: 4,
		}, engine, ops)
		if err != nil {
			return nil, err
		}
		ctx := context.Background()
		predict := func(features []float64) (tabular.Direction, float64, error) {
			f32 := make([]float32, len(features))
			for i, v := range features {
				f32[i] = float32(v)
			}
			input, err := tensor.New[float32]([]int{1, inputDim}, f32)
			if err != nil {
				return tabular.Flat, 0, err
			}
			out, err := nb.Forward(ctx, input)
			if err != nil {
				return tabular.Flat, 0, err
			}
			dir, conf := argmax(softmax(out.Forecast.Data()))
			return dir, conf, nil
		}
		return &BestModel{Architecture: arch, Params: params, Predictor: predict}, nil

	case ArchPatchTST:
		patchLen := int(clamp(params["patch_len"], 1, float64(inputDim)))
		dModel := int(clamp(params["d_model"], 8, 256))
		nHeads := int(clamp(params["n_heads"], 1, 8))
		dModel = alignDim(dModel, nHeads)
		stride := patchLen / 2
		if stride < 1 {
			stride = 1
		}

		pt, err := timeseries.NewPatchTST(timeseries.PatchTSTConfig{
			InputLength: inputDim, PatchLength: patchLen, Stride: stride,
			DModel: dModel, NHeads: nHeads, NLayers: 2, OutputDim: 3,
			ChannelIndependent: false,
		}, engine, ops)
		if err != nil {
			return nil, err
		}
		predict := func(features []float64) (tabular.Direction, float64, error) {
			out, err := pt.Predict([][]float64{features})
			if err != nil {
				return tabular.Flat, 0, err
			}
			if len(out) == 0 || len(out[0]) < 3 {
				return tabular.Flat, 0, fmt.Errorf("automl: patchtst output too short")
			}
			logits := make([]float32, 3)
			for i := 0; i < 3; i++ {
				logits[i] = float32(out[0][i])
			}
			dir, conf := argmax(softmax(logits))
			return dir, conf, nil
		}
		return &BestModel{Architecture: arch, Params: params, Predictor: predict}, nil

	default:
		return nil, fmt.Errorf("automl: unsupported architecture for retrain: %q", arch)
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

// alignDim rounds dim up to the nearest multiple of nHeads.
// Returns at least nHeads.
func alignDim(dim, nHeads int) int {
	if nHeads <= 0 {
		return dim
	}
	if dim%nHeads == 0 {
		return dim
	}
	aligned := ((dim / nHeads) + 1) * nHeads
	if aligned <= 0 {
		return nHeads
	}
	return aligned
}

// softmax applies the softmax function to logits.
func softmax(logits []float32) []float32 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	probs := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - maxVal)))
		sum += probs[i]
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

// argmax returns the Direction and confidence for the highest-probability class.
func argmax(probs []float32) (tabular.Direction, float64) {
	if len(probs) < 3 {
		return tabular.Flat, 0
	}
	best := 0
	for i := 1; i < 3; i++ {
		if probs[i] > probs[best] {
			best = i
		}
	}
	return tabular.Direction(best), float64(probs[best])
}
