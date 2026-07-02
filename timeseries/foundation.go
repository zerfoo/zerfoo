package timeseries

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"

	its "github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// FoundationForecaster provides zero-shot time-series forecasting using a
// pre-trained foundation model (TiRex). It handles input normalization
// (instance norm), graph execution, and output denormalization.
type FoundationForecaster struct {
	graph     *graph.Graph[float32]
	engine    compute.Engine[float32]
	cfg       *its.TiRexConfig
	numVars   int
	horizon   int
	extractor *its.TiRexFeatureExtractor[float32]
}

// FineTuneConfig configures foundation model fine-tuning.
type FineTuneConfig struct {
	Epochs         int     // number of training epochs
	LearningRate   float64 // AdamW learning rate
	BatchSize      int     // mini-batch size (0 = full batch)
	FreezeBackbone bool    // if true, only train output head
}

// LoadFoundationModel loads a TiRex foundation model from a GGUF file and
// returns a forecaster ready for zero-shot inference.
func LoadFoundationModel(path string, engine compute.Engine[float32]) (*FoundationForecaster, error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("open GGUF file: %w", err)
	}
	defer func() { _ = f.Close() }()

	gf, err := gguf.Parse(f)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	cfg, err := loadTiRexConfigFromMeta(gf.Metadata)
	if err != nil {
		return nil, fmt.Errorf("load TiRex config: %w", err)
	}

	tensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		return nil, fmt.Errorf("load GGUF tensors: %w", err)
	}

	g, ext, err := its.BuildTiRexWithExtractor[float32](tensors, cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build TiRex graph: %w", err)
	}

	return &FoundationForecaster{
		graph:     g,
		engine:    engine,
		cfg:       cfg,
		numVars:   cfg.NumVars,
		horizon:   cfg.Horizon,
		extractor: ext,
	}, nil
}

// newFoundationForecasterFromConfig creates a FoundationForecaster directly
// from a config and engine, bypassing GGUF loading. Used for testing.
func newFoundationForecasterFromConfig(cfg *its.TiRexConfig, engine compute.Engine[float32]) (*FoundationForecaster, error) {
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	g, ext, err := its.BuildTiRexWithExtractor[float32](tensors, cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build TiRex graph: %w", err)
	}
	return &FoundationForecaster{
		graph:     g,
		engine:    engine,
		cfg:       cfg,
		numVars:   cfg.NumVars,
		horizon:   cfg.Horizon,
		extractor: ext,
	}, nil
}

// Forecast produces a zero-shot forecast for a single time series.
//
// Input shape: [seq_len][num_vars] as [][]float64.
// Output shape: [horizon][num_vars] as [][]float64.
//
// The pipeline:
//  1. Per-channel instance normalization: x_norm = (x - mean) / (std + eps)
//  2. Forward pass through TiRex graph
//  3. Denormalize predictions: y = y_norm * (std + eps) + mean
func (f *FoundationForecaster) Forecast(ctx context.Context, input [][]float64, horizon int) ([][]float64, error) {
	if len(input) == 0 {
		return nil, fmt.Errorf("input must not be empty")
	}
	numVars := len(input[0])
	if numVars != f.numVars {
		return nil, fmt.Errorf("input variables must be %d, got %d", f.numVars, numVars)
	}
	if horizon <= 0 {
		return nil, fmt.Errorf("horizon must be positive, got %d", horizon)
	}

	seqLen := len(input)

	// Per-channel instance normalization.
	mean, std := instanceNorm(input, seqLen, numVars)

	// Build input tensor: [1, seq_len, num_vars].
	data := make([]float32, seqLen*numVars)
	for t := range seqLen {
		for c := range numVars {
			data[t*numVars+c] = float32((input[t][c] - mean[c]) / (std[c] + 1e-8))
		}
	}

	inputTensor, err := tensor.New[float32]([]int{1, seqLen, numVars}, data)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outputTensor, err := f.graph.Forward(ctx, inputTensor)
	if err != nil {
		return nil, fmt.Errorf("forward pass: %w", err)
	}

	// The graph outputs [1, model_horizon, num_vars]. Extract and denormalize.
	outData := outputTensor.Data()
	modelHorizon := f.horizon
	effectiveHorizon := min(horizon, modelHorizon)

	result := make([][]float64, effectiveHorizon)
	for t := range effectiveHorizon {
		result[t] = make([]float64, numVars)
		for c := range numVars {
			val := float64(outData[t*numVars+c])
			result[t][c] = val*(std[c]+1e-8) + mean[c]
		}
	}

	return result, nil
}

// BatchForecast produces zero-shot forecasts for multiple time series.
//
// Input shape: [batch][seq_len][num_vars] as [][][]float64.
// Output shape: [batch][horizon][num_vars] as [][][]float64.
func (f *FoundationForecaster) BatchForecast(ctx context.Context, inputs [][][]float64, horizon int) ([][][]float64, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("batch input must not be empty")
	}
	if horizon <= 0 {
		return nil, fmt.Errorf("horizon must be positive, got %d", horizon)
	}

	batchSize := len(inputs)
	seqLen := len(inputs[0])
	numVars := f.numVars

	// Validate all samples.
	for i, input := range inputs {
		if len(input) == 0 {
			return nil, fmt.Errorf("batch[%d]: input must not be empty", i)
		}
		if len(input) != seqLen {
			return nil, fmt.Errorf("batch[%d]: sequence length must be %d, got %d", i, seqLen, len(input))
		}
		if len(input[0]) != numVars {
			return nil, fmt.Errorf("batch[%d]: input variables must be %d, got %d", i, numVars, len(input[0]))
		}
	}

	// Per-sample normalization stats.
	means := make([][]float64, batchSize)
	stds := make([][]float64, batchSize)

	// Build batched tensor: [batch, seq_len, num_vars].
	data := make([]float32, batchSize*seqLen*numVars)
	for b := range batchSize {
		mean, std := instanceNorm(inputs[b], seqLen, numVars)
		means[b] = mean
		stds[b] = std

		for t := range seqLen {
			for c := range numVars {
				data[b*seqLen*numVars+t*numVars+c] = float32((inputs[b][t][c] - mean[c]) / (std[c] + 1e-8))
			}
		}
	}

	inputTensor, err := tensor.New[float32]([]int{batchSize, seqLen, numVars}, data)
	if err != nil {
		return nil, fmt.Errorf("create batch input tensor: %w", err)
	}

	outputTensor, err := f.graph.Forward(ctx, inputTensor)
	if err != nil {
		return nil, fmt.Errorf("batch forward pass: %w", err)
	}

	// Convert output [batch, model_horizon, num_vars] and denormalize.
	outData := outputTensor.Data()
	modelHorizon := f.horizon
	effectiveHorizon := min(horizon, modelHorizon)

	results := make([][][]float64, batchSize)
	for b := range batchSize {
		results[b] = make([][]float64, effectiveHorizon)
		for t := range effectiveHorizon {
			results[b][t] = make([]float64, numVars)
			for c := range numVars {
				idx := b*modelHorizon*numVars + t*numVars + c
				val := float64(outData[idx])
				results[b][t][c] = val*(stds[b][c]+1e-8) + means[b][c]
			}
		}
	}

	return results, nil
}

// instanceNorm computes per-channel mean and standard deviation.
func instanceNorm(input [][]float64, seqLen, numVars int) ([]float64, []float64) {
	mean := make([]float64, numVars)
	std := make([]float64, numVars)

	for c := range numVars {
		var sum float64
		for t := range seqLen {
			sum += input[t][c]
		}
		mean[c] = sum / float64(seqLen)

		var sumSq float64
		for t := range seqLen {
			diff := input[t][c] - mean[c]
			sumSq += diff * diff
		}
		std[c] = math.Sqrt(sumSq / float64(seqLen))
	}

	return mean, std
}

// FineTune adapts the foundation model on task-specific data using AdamW.
// When cfg.FreezeBackbone is true, only the output head is trained (few-shot
// adaptation). data contains input sequences [sample][seq_len*num_vars] and
// labels contains target outputs [sample][horizon*num_vars], both flattened
// row-major.
//
// Input data layout: each data[i] has length seq_len * num_vars, representing
// a [seq_len][num_vars] input flattened row-major. Each labels[i] has length
// horizon * num_vars, representing [horizon][num_vars] target flattened.
func (f *FoundationForecaster) FineTune(ctx context.Context, data [][]float64, labels [][]float64, cfg FineTuneConfig) (*TrainResult, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("fine-tune: empty training data")
	}
	if len(data) != len(labels) {
		return nil, fmt.Errorf("fine-tune: data (%d) and labels (%d) length mismatch", len(data), len(labels))
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 10
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 1e-3
	}

	nSamples := len(data)
	numVars := f.numVars
	horizon := f.horizon
	hiddenDim := f.extractor.HiddenDim()

	expectedDataLen := -1 // infer seq_len from first sample
	expectedLabelLen := horizon * numVars
	for i := range data {
		if expectedDataLen < 0 {
			if len(data[i])%numVars != 0 {
				return nil, fmt.Errorf("fine-tune: data[%d] length %d not divisible by num_vars %d", i, len(data[i]), numVars)
			}
			expectedDataLen = len(data[i])
		}
		if len(data[i]) != expectedDataLen {
			return nil, fmt.Errorf("fine-tune: data[%d] length %d, expected %d", i, len(data[i]), expectedDataLen)
		}
		if len(labels[i]) != expectedLabelLen {
			return nil, fmt.Errorf("fine-tune: labels[%d] length %d, expected %d", i, len(labels[i]), expectedLabelLen)
		}
	}
	seqLen := expectedDataLen / numVars

	// Get output head parameters for AdamW.
	headParams := f.extractor.OutputHeadParams()
	nParams := 0
	for _, p := range headParams {
		nParams += len(p.Value.Data())
	}

	// AdamW state.
	beta1, beta2, eps := 0.9, 0.999, 1e-8
	weightDecay := 1e-4
	gradClip := 1.0
	mState := make([]float64, nParams)
	vState := make([]float64, nParams)

	result := &TrainResult{Metrics: make(map[string]float64)}
	step := 0

	batchSize := cfg.BatchSize
	if batchSize <= 0 || batchSize > nSamples {
		batchSize = nSamples
	}

	for epoch := range cfg.Epochs {
		epochLoss := 0.0
		epochCount := 0

		for bStart := 0; bStart < nSamples; bStart += batchSize {
			bEnd := bStart + batchSize
			if bEnd > nSamples {
				bEnd = nSamples
			}
			bSize := bEnd - bStart

			// Build batched input tensor [bSize, seqLen, numVars].
			inputData := make([]float32, bSize*seqLen*numVars)
			for b := range bSize {
				for j := range seqLen * numVars {
					inputData[b*seqLen*numVars+j] = float32(data[bStart+b][j])
				}
			}
			inputTensor, err := tensor.New[float32]([]int{bSize, seqLen, numVars}, inputData)
			if err != nil {
				return nil, fmt.Errorf("fine-tune: create input tensor: %w", err)
			}

			// Extract backbone features [bSize, hiddenDim].
			hidden, err := f.extractor.ForwardFeatures(ctx, inputTensor)
			if err != nil {
				return nil, fmt.Errorf("fine-tune: forward features: %w", err)
			}

			// Output head forward: [bSize, horizon*numVars].
			pred, err := f.extractor.OutputHeadForward(ctx, hidden)
			if err != nil {
				return nil, fmt.Errorf("fine-tune: output head forward: %w", err)
			}

			predData := pred.Data()
			hiddenData := hidden.Data()
			outSize := horizon * numVars

			// Compute MSE loss and gradient dL/dOutput.
			dOutput := make([]float64, bSize*outSize)
			batchLoss := 0.0
			for b := range bSize {
				for j := range outSize {
					idx := b*outSize + j
					diff := float64(predData[idx]) - labels[bStart+b][j]
					batchLoss += diff * diff
					dOutput[idx] = 2.0 * diff / float64(bSize*outSize)
				}
			}
			batchLoss /= float64(bSize * outSize)
			epochLoss += batchLoss * float64(bSize)
			epochCount += bSize

			// Compute dL/dW for output head: W is [hiddenDim, outSize].
			// output = hidden @ W, so dL/dW = hidden^T @ dL/dOutput.
			gradW := make([]float64, hiddenDim*outSize)
			for b := range bSize {
				for h := range hiddenDim {
					hVal := float64(hiddenData[b*hiddenDim+h])
					for o := range outSize {
						gradW[h*outSize+o] += hVal * dOutput[b*outSize+o]
					}
				}
			}

			// Gradient clipping.
			norm := 0.0
			for _, g := range gradW {
				norm += g * g
			}
			norm = math.Sqrt(norm)
			if norm > gradClip {
				scale := gradClip / norm
				for i := range gradW {
					gradW[i] *= scale
				}
			}

			// AdamW update on output head weight.
			step++
			t := float64(step)
			wData := headParams[0].Value.Data()
			for i := range wData {
				g := gradW[i]
				mState[i] = beta1*mState[i] + (1-beta1)*g
				vState[i] = beta2*vState[i] + (1-beta2)*g*g
				mHat := mState[i] / (1 - math.Pow(beta1, t))
				vHat := vState[i] / (1 - math.Pow(beta2, t))
				val := float64(wData[i])
				val -= cfg.LearningRate * (mHat/(math.Sqrt(vHat)+eps) + weightDecay*val)
				wData[i] = float32(val)
			}
		}

		avgLoss := epochLoss / float64(epochCount)
		result.LossHistory = append(result.LossHistory, avgLoss)

		_ = epoch
	}

	if len(result.LossHistory) > 0 {
		result.FinalLoss = result.LossHistory[len(result.LossHistory)-1]
	}

	return result, nil
}

// loadTiRexConfigFromMeta extracts a TiRexConfig from GGUF metadata.
func loadTiRexConfigFromMeta(meta map[string]interface{}) (*its.TiRexConfig, error) {
	getInt := func(key string) (int, error) {
		v, ok := meta[key]
		if !ok {
			return 0, fmt.Errorf("missing metadata key %q", key)
		}
		switch n := v.(type) {
		case uint32:
			return int(n), nil
		case int:
			return n, nil
		case int64:
			return int(n), nil
		case uint64:
			return int(n), nil
		default:
			return 0, fmt.Errorf("metadata key %q has unexpected type %T", key, v)
		}
	}

	numLayers, err := getInt("tirex.block_count")
	if err != nil {
		return nil, err
	}
	hiddenDim, err := getInt("tirex.hidden_dim")
	if err != nil {
		return nil, err
	}

	// InputDim defaults to 1 for univariate foundation models.
	inputDim := 1
	if v, ok := meta["tirex.input_dim"]; ok {
		switch n := v.(type) {
		case uint32:
			inputDim = int(n)
		case int:
			inputDim = n
		}
	}

	// Horizon defaults to 1.
	horizon := 1
	if v, ok := meta["tirex.horizon"]; ok {
		switch n := v.(type) {
		case uint32:
			horizon = int(n)
		case int:
			horizon = n
		}
	}

	// NumVars defaults to InputDim.
	numVars := inputDim
	if v, ok := meta["tirex.num_vars"]; ok {
		switch n := v.(type) {
		case uint32:
			numVars = int(n)
		case int:
			numVars = n
		}
	}

	// Block types from string array.
	var blockTypes []string
	if v, ok := meta["tirex.block_types"]; ok {
		switch arr := v.(type) {
		case []string:
			blockTypes = arr
		case []any:
			blockTypes = make([]string, len(arr))
			for i, s := range arr {
				str, ok := s.(string)
				if !ok {
					return nil, fmt.Errorf("tirex.block_types[%d] is not a string", i)
				}
				blockTypes[i] = str
			}
		}
	}

	return &its.TiRexConfig{
		NumLayers:  numLayers,
		InputDim:   inputDim,
		HiddenDim:  hiddenDim,
		Horizon:    horizon,
		NumVars:    numVars,
		BlockTypes: blockTypes,
	}, nil
}
