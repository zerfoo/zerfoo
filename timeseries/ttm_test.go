package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func ttmTestConfig() TTMTrainConfig {
	return TTMTrainConfig{
		ContextLen:     16,
		ForecastLen:    4,
		NumChannels:    1,
		PatchLen:       4,
		DModel:         8,
		NumMixerLayers: 1,
		ChannelMixing:  false,
		FreezeEncoder:  false,
	}
}

func TestTTM_TrainWindowed_Convergence(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	model, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}

	// Generate synthetic sinusoidal training data.
	nSamples := 20
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.ForecastLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1)
		windows[s][0] = make([]float64, config.ContextLen)
		for i := 0; i < config.ContextLen; i++ {
			windows[s][0][i] = math.Sin(float64(s+i) * 0.3)
		}
		for o := 0; o < config.ForecastLen; o++ {
			labels[s*config.ForecastLen+o] = math.Sin(float64(s+config.ContextLen+o) * 0.3)
		}
	}

	result, err := model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   50,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if len(result.LossHistory) != 50 {
		t.Fatalf("loss history length = %d, want 50", len(result.LossHistory))
	}

	// Verify loss decreases: first loss should be larger than final loss.
	if result.LossHistory[0] <= result.FinalLoss {
		t.Errorf("loss did not decrease: first=%v, final=%v", result.LossHistory[0], result.FinalLoss)
	}

	if math.IsNaN(result.FinalLoss) || math.IsInf(result.FinalLoss, 0) {
		t.Errorf("FinalLoss is not finite: %v", result.FinalLoss)
	}

	if _, ok := result.Metrics["mse"]; !ok {
		t.Error("Metrics missing 'mse' key")
	}
}

func TestTTM_PredictWindowed_Shape(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	model, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}

	nSamples := 5
	windows := make([][][]float64, nSamples)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1)
		windows[s][0] = make([]float64, config.ContextLen)
		for i := 0; i < config.ContextLen; i++ {
			windows[s][0][i] = float64(s+i) * 0.1
		}
	}

	preds, err := model.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	expected := nSamples * config.ForecastLen
	if len(preds) != expected {
		t.Fatalf("PredictWindowed returned %d values, want %d", len(preds), expected)
	}

	for i, v := range preds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("prediction[%d] is not finite: %v", i, v)
		}
	}
}

func TestTTM_FreezeEncoder(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	config.FreezeEncoder = true
	model, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}

	// Save encoder weights before training.
	paramsBefore := model.extractParamsF64()
	encoderBefore := make([][]float64, len(paramsBefore.encoder))
	for i, l := range paramsBefore.encoder {
		encoderBefore[i] = l.appendGrads(nil)
	}

	// Generate training data.
	nSamples := 10
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.ForecastLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1)
		windows[s][0] = make([]float64, config.ContextLen)
		for i := 0; i < config.ContextLen; i++ {
			windows[s][0][i] = math.Sin(float64(s+i) * 0.3)
		}
		for o := 0; o < config.ForecastLen; o++ {
			labels[s*config.ForecastLen+o] = math.Sin(float64(s+config.ContextLen+o) * 0.3)
		}
	}

	_, err = model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   20,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Check encoder weights unchanged.
	paramsAfter := model.extractParamsF64()
	for i, l := range paramsAfter.encoder {
		after := l.appendGrads(nil)
		for j := range after {
			if after[j] != encoderBefore[i][j] {
				t.Fatalf("encoder block %d param %d changed from %v to %v (should be frozen)",
					i, j, encoderBefore[i][j], after[j])
			}
		}
	}

	// Verify decoder weights DID change (at least some).
	decoderBefore := make([][]float64, len(paramsBefore.decoder))
	for i, l := range paramsBefore.decoder {
		decoderBefore[i] = l.appendGrads(nil)
	}
	decoderChanged := false
	for i, l := range paramsAfter.decoder {
		after := l.appendGrads(nil)
		for j := range after {
			if after[j] != decoderBefore[i][j] {
				decoderChanged = true
				break
			}
		}
		if decoderChanged {
			break
		}
	}
	if !decoderChanged {
		t.Error("decoder weights did not change during training (expected fine-tuning)")
	}
}

func TestTTM_Forward_Shape(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	model, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}

	// Create input: [batch=2, contextLen=16].
	batch := 2
	data := make([]float32, batch*config.ContextLen)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, config.ContextLen}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()
	out, err := model.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != batch || shape[1] != config.ForecastLen {
		t.Errorf("output shape = %v, want [%d, %d]", shape, batch, config.ForecastLen)
	}
}

func TestTTM_TrainWindowed_Empty(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	model, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}

	_, err = model.TrainWindowed(nil, nil, TrainConfig{Epochs: 1})
	if err == nil {
		t.Error("expected error for empty training set")
	}
}

func TestTTM_SaveLoadWeights(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	model, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}

	// Train briefly to get non-default weights.
	nSamples := 5
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.ForecastLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1)
		windows[s][0] = make([]float64, config.ContextLen)
		for i := 0; i < config.ContextLen; i++ {
			windows[s][0][i] = float64(s+i) * 0.1
		}
		for o := 0; o < config.ForecastLen; o++ {
			labels[s*config.ForecastLen+o] = float64(s) * 0.05
		}
	}
	_, err = model.TrainWindowed(windows, labels, TrainConfig{Epochs: 3, LR: 1e-3, GradClip: 1.0})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	path := t.TempDir() + "/ttm_weights.json"
	if err := model.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	// Create a fresh model and load weights.
	model2, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}
	if err := model2.loadWeights(path); err != nil {
		t.Fatalf("loadWeights: %v", err)
	}

	// Predictions should match.
	preds1, err := model.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed model1: %v", err)
	}
	preds2, err := model2.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed model2: %v", err)
	}
	for i := range preds1 {
		if math.Abs(preds1[i]-preds2[i]) > 1e-6 {
			t.Errorf("prediction mismatch at %d: %v vs %v", i, preds1[i], preds2[i])
		}
	}
}

func TestTTMEngine_Forward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	config := ttmTestConfig()
	ttmEngine, err := NewTTMEngine[float32](config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTMEngine: %v", err)
	}

	ctx := context.Background()
	batch := 2
	numPatches := config.NumPatches()

	// Create patched input: [batch, numPatches, patchLen].
	data := make([]float32, batch*numPatches*config.PatchLen)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, numPatches, config.PatchLen}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	out, err := ttmEngine.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != batch || shape[1] != config.ForecastLen {
		t.Errorf("output shape = %v, want [%d, %d]", shape, batch, config.ForecastLen)
	}

	// Verify outputs are finite.
	for i, v := range out.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("output[%d] is not finite: %v", i, v)
		}
	}
}

func TestTTMEngine_ExtractPatches(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	config := ttmTestConfig()
	ttmEngine, err := NewTTMEngine[float32](config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTMEngine: %v", err)
	}

	ctx := context.Background()
	batch := 3

	data := make([]float32, batch*config.ContextLen)
	for i := range data {
		data[i] = float32(i)
	}
	input, err := tensor.New[float32]([]int{batch, config.ContextLen}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	patches, err := ttmEngine.ExtractPatches(ctx, input)
	if err != nil {
		t.Fatalf("ExtractPatches: %v", err)
	}

	numPatches := config.NumPatches()
	shape := patches.Shape()
	if len(shape) != 3 || shape[0] != batch || shape[1] != numPatches || shape[2] != config.PatchLen {
		t.Errorf("patches shape = %v, want [%d, %d, %d]", shape, batch, numPatches, config.PatchLen)
	}

	// Verify first sample, first patch.
	pData := patches.Data()
	for i := 0; i < config.PatchLen; i++ {
		if pData[i] != float32(i) {
			t.Errorf("patch[0][0][%d] = %v, want %v", i, pData[i], float32(i))
		}
	}
}

func TestTTM_ChannelMixing(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	config.ChannelMixing = true
	config.NumChannels = 2
	model, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}

	nSamples := 10
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.ForecastLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 2)
		for c := 0; c < 2; c++ {
			windows[s][c] = make([]float64, config.ContextLen)
			for i := 0; i < config.ContextLen; i++ {
				windows[s][c][i] = math.Sin(float64(s+i+c*3) * 0.3)
			}
		}
		for o := 0; o < config.ForecastLen; o++ {
			labels[s*config.ForecastLen+o] = math.Sin(float64(s+config.ContextLen+o) * 0.3)
		}
	}

	result, err := model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   30,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if math.IsNaN(result.FinalLoss) || math.IsInf(result.FinalLoss, 0) {
		t.Errorf("FinalLoss is not finite: %v", result.FinalLoss)
	}
}

func TestTTM_TrainWindowed_EngineDispatch(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	modelWithEngine, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM with engine: %v", err)
	}

	// Verify engine is set (dispatch should go to trainWindowedEngine).
	if modelWithEngine.engine == nil {
		t.Fatal("engine should not be nil")
	}

	// Create model without engine (nil engine should go to trainWindowedCPU).
	modelWithoutEngine, err := NewTTM(config, nil, nil)
	if err != nil {
		t.Fatalf("NewTTM without engine: %v", err)
	}
	if modelWithoutEngine.engine != nil {
		t.Fatal("engine should be nil")
	}

	nSamples := 10
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.ForecastLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1)
		windows[s][0] = make([]float64, config.ContextLen)
		for i := 0; i < config.ContextLen; i++ {
			windows[s][0][i] = math.Sin(float64(s+i) * 0.3)
		}
		for o := 0; o < config.ForecastLen; o++ {
			labels[s*config.ForecastLen+o] = math.Sin(float64(s+config.ContextLen+o) * 0.3)
		}
	}

	// Both paths should train without error.
	_, err = modelWithEngine.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   10,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed (engine path): %v", err)
	}

	_, err = modelWithoutEngine.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   10,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed (CPU path): %v", err)
	}
}

func TestTTM_TrainWindowed_EngineParity(t *testing.T) {
	// Verify that the engine path produces the same results as the CPU path
	// (within float32 precision tolerance) by training both from identical
	// initial weights.
	engine, ops := newTestEngine()
	config := ttmTestConfig()

	modelEngine, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM (engine): %v", err)
	}

	modelCPU, err := NewTTM(config, nil, nil)
	if err != nil {
		t.Fatalf("NewTTM (cpu): %v", err)
	}

	// Copy weights from engine model to CPU model so they start identical.
	copyTTMWeights(modelEngine, modelCPU)

	nSamples := 15
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.ForecastLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1)
		windows[s][0] = make([]float64, config.ContextLen)
		for i := 0; i < config.ContextLen; i++ {
			windows[s][0][i] = math.Sin(float64(s+i) * 0.3)
		}
		for o := 0; o < config.ForecastLen; o++ {
			labels[s*config.ForecastLen+o] = math.Sin(float64(s+config.ContextLen+o) * 0.3)
		}
	}

	trainCfg := TrainConfig{
		Epochs:   5,
		LR:       1e-3,
		GradClip: 1.0,
	}

	resultEngine, err := modelEngine.TrainWindowed(windows, labels, trainCfg)
	if err != nil {
		t.Fatalf("TrainWindowed (engine): %v", err)
	}

	resultCPU, err := modelCPU.TrainWindowed(windows, labels, trainCfg)
	if err != nil {
		t.Fatalf("TrainWindowed (CPU): %v", err)
	}

	// Compare losses — the engine path uses float32 for MatMul, so allow
	// some tolerance due to reduced precision.
	const tol = 1e-3
	for epoch := range resultEngine.LossHistory {
		diff := math.Abs(resultEngine.LossHistory[epoch] - resultCPU.LossHistory[epoch])
		relDiff := diff / (math.Abs(resultCPU.LossHistory[epoch]) + 1e-10)
		if relDiff > tol {
			t.Errorf("epoch %d: engine loss=%v, cpu loss=%v, relDiff=%v > tol %v",
				epoch, resultEngine.LossHistory[epoch], resultCPU.LossHistory[epoch], relDiff, tol)
		}
	}
}

func TestTTM_TrainWindowed_EngineConvergence(t *testing.T) {
	engine, ops := newTestEngine()

	config := ttmTestConfig()
	model, err := NewTTM(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTTM: %v", err)
	}

	nSamples := 20
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.ForecastLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1)
		windows[s][0] = make([]float64, config.ContextLen)
		for i := 0; i < config.ContextLen; i++ {
			windows[s][0][i] = math.Sin(float64(s+i) * 0.3)
		}
		for o := 0; o < config.ForecastLen; o++ {
			labels[s*config.ForecastLen+o] = math.Sin(float64(s+config.ContextLen+o) * 0.3)
		}
	}

	result, err := model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   50,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if result.LossHistory[0] <= result.FinalLoss {
		t.Errorf("loss did not decrease via engine path: first=%v, final=%v",
			result.LossHistory[0], result.FinalLoss)
	}

	if math.IsNaN(result.FinalLoss) || math.IsInf(result.FinalLoss, 0) {
		t.Errorf("FinalLoss is not finite: %v", result.FinalLoss)
	}
}

// copyTTMWeights copies float32 tensor weights from src to dst.
func copyTTMWeights(src, dst *TTM) {
	copyLinearLayer(&src.patchEmb, &dst.patchEmb)
	copyLinearLayer(&src.head, &dst.head)
	for i := range src.encoder {
		copyMixerBlock(&src.encoder[i], &dst.encoder[i])
	}
	for i := range src.decoder {
		copyMixerBlock(&src.decoder[i], &dst.decoder[i])
	}
}

func copyLinearLayer(src, dst *linearLayer) {
	srcW := src.weights.Data()
	dstW := dst.weights.Data()
	copy(dstW, srcW)
	srcB := src.biases.Data()
	dstB := dst.biases.Data()
	copy(dstB, srcB)
}

func copyMixerBlock(src, dst *ttmMixerBlockF32) {
	copyLinearLayer(&src.timeMLP1, &dst.timeMLP1)
	copyLinearLayer(&src.timeMLP2, &dst.timeMLP2)
	copy(dst.timeNorm.scale.Data(), src.timeNorm.scale.Data())
	copy(dst.timeNorm.bias.Data(), src.timeNorm.bias.Data())
	if src.channelMixing {
		copyLinearLayer(&src.featMLP1, &dst.featMLP1)
		copyLinearLayer(&src.featMLP2, &dst.featMLP2)
		copy(dst.featNorm.scale.Data(), src.featNorm.scale.Data())
		copy(dst.featNorm.bias.Data(), src.featNorm.bias.Data())
	}
}

func TestTTM_BatchForwardParity(t *testing.T) {
	engine, ops := newTestEngine()

	for _, tc := range []struct {
		name          string
		channelMixing bool
		numChannels   int
	}{
		{"single_channel", false, 1},
		{"multi_channel", false, 2},
		{"channel_mixing", true, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			config := ttmTestConfig()
			config.ChannelMixing = tc.channelMixing
			config.NumChannels = tc.numChannels

			model, err := NewTTM(config, engine, ops)
			if err != nil {
				t.Fatalf("NewTTM: %v", err)
			}

			ctx := context.Background()
			params := model.extractParamsF64()
			batch := 4

			// Build synthetic windows [batch][channels][contextLen].
			windows := make([][][]float64, batch)
			for s := 0; s < batch; s++ {
				windows[s] = make([][]float64, tc.numChannels)
				for ch := 0; ch < tc.numChannels; ch++ {
					windows[s][ch] = make([]float64, config.ContextLen)
					for i := 0; i < config.ContextLen; i++ {
						windows[s][ch][i] = math.Sin(float64(s*tc.numChannels*config.ContextLen+ch*config.ContextLen+i) * 0.1)
					}
				}
			}

			// Sample-by-sample forward using forwardF64WithCacheEngine.
			sampleResults := make([][]float64, batch)
			for s := 0; s < batch; s++ {
				pred, _, err := model.forwardF64WithCacheEngine(ctx, windows[s], params)
				if err != nil {
					t.Fatalf("sample %d forward: %v", s, err)
				}
				sampleResults[s] = pred
			}

			// Batched forward.
			batchResults, err := model.batchForwardF64Engine(ctx, windows, params)
			if err != nil {
				t.Fatalf("batchForwardF64Engine: %v", err)
			}

			if len(batchResults) != batch {
				t.Fatalf("batch results length = %d, want %d", len(batchResults), batch)
			}

			// Compare outputs.
			const tol = 1e-9
			for s := 0; s < batch; s++ {
				if len(batchResults[s]) != config.ForecastLen {
					t.Errorf("sample %d: output length = %d, want %d", s, len(batchResults[s]), config.ForecastLen)
					continue
				}
				for j := 0; j < config.ForecastLen; j++ {
					diff := math.Abs(sampleResults[s][j] - batchResults[s][j])
					if diff > tol {
						t.Errorf("sample %d output[%d]: sample=%.12f batch=%.12f diff=%.2e",
							s, j, sampleResults[s][j], batchResults[s][j], diff)
					}
				}
			}
		})
	}
}
