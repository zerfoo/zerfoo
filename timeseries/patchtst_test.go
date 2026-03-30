package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestPatchTST_Forward(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name      string
		config    PatchTSTConfig
		inputDims []int // shape of input tensor
		wantShape []int // expected output shape
		wantErr   bool
	}{
		{
			name: "univariate basic",
			config: PatchTSTConfig{
				InputLength:        24,
				PatchLength:        8,
				Stride:             4,
				DModel:             16,
				NHeads:             2,
				NLayers:            1,
				OutputDim:          4,
				ChannelIndependent: false,
			},
			inputDims: []int{2, 24},
			wantShape: []int{2, 4},
		},
		{
			name: "multivariate channel independent",
			config: PatchTSTConfig{
				InputLength:        24,
				PatchLength:        8,
				Stride:             4,
				DModel:             16,
				NHeads:             2,
				NLayers:            1,
				OutputDim:          4,
				ChannelIndependent: true,
			},
			inputDims: []int{2, 3, 24},
			wantShape: []int{2, 3, 4},
		},
		{
			name: "single batch single channel",
			config: PatchTSTConfig{
				InputLength:        16,
				PatchLength:        4,
				Stride:             4,
				DModel:             8,
				NHeads:             2,
				NLayers:            1,
				OutputDim:          2,
				ChannelIndependent: false,
			},
			inputDims: []int{1, 16},
			wantShape: []int{1, 2},
		},
		{
			name: "two encoder layers",
			config: PatchTSTConfig{
				InputLength:        16,
				PatchLength:        4,
				Stride:             4,
				DModel:             8,
				NHeads:             2,
				NLayers:            2,
				OutputDim:          3,
				ChannelIndependent: false,
			},
			inputDims: []int{1, 16},
			wantShape: []int{1, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := NewPatchTST(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewPatchTST: %v", err)
			}

			totalElems := 1
			for _, d := range tt.inputDims {
				totalElems *= d
			}
			data := make([]float32, totalElems)
			for i := range data {
				data[i] = float32(i) * 0.01
			}

			input, err := tensor.New[float32](tt.inputDims, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			output, err := model.Forward(ctx, input)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			gotShape := output.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("output shape rank = %d, want %d (got %v, want %v)", len(gotShape), len(tt.wantShape), gotShape, tt.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("output shape[%d] = %d, want %d (got %v, want %v)", i, gotShape[i], tt.wantShape[i], gotShape, tt.wantShape)
				}
			}

			// Verify output contains finite values.
			outData := output.Data()
			for i, v := range outData {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v, want finite", i, v)
					break
				}
			}
		})
	}
}

func TestPatchTST_Patching(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := PatchTSTConfig{
		InputLength:        12,
		PatchLength:        4,
		Stride:             2,
		DModel:             8,
		NHeads:             2,
		NLayers:            1,
		OutputDim:          2,
		ChannelIndependent: false,
	}

	// Verify NumPatches calculation: (12 - 4) / 2 + 1 = 5
	wantPatches := 5
	if got := config.NumPatches(); got != wantPatches {
		t.Fatalf("NumPatches() = %d, want %d", got, wantPatches)
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Create input [1, 12] with sequential values.
	data := make([]float32, 12)
	for i := range data {
		data[i] = float32(i)
	}
	input, err := tensor.New[float32]([]int{1, 12}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	// Extract patches directly to verify correctness.
	patches, err := model.extractPatches(ctx, input)
	if err != nil {
		t.Fatalf("extractPatches: %v", err)
	}

	// Expected shape: [1, 5, 4]
	gotShape := patches.Shape()
	wantShape := []int{1, 5, 4}
	if len(gotShape) != len(wantShape) {
		t.Fatalf("patches shape = %v, want %v", gotShape, wantShape)
	}
	for i := range gotShape {
		if gotShape[i] != wantShape[i] {
			t.Fatalf("patches shape[%d] = %d, want %d", i, gotShape[i], wantShape[i])
		}
	}

	// Verify patch contents: patches extracted at strides 0, 2, 4, 6, 8.
	pData := patches.Data()
	expected := [][]float32{
		{0, 1, 2, 3},   // offset 0
		{2, 3, 4, 5},   // offset 2
		{4, 5, 6, 7},   // offset 4
		{6, 7, 8, 9},   // offset 6
		{8, 9, 10, 11}, // offset 8
	}
	for p := range wantPatches {
		for j := range 4 {
			idx := p*4 + j
			if pData[idx] != expected[p][j] {
				t.Errorf("patch[%d][%d] = %v, want %v", p, j, pData[idx], expected[p][j])
			}
		}
	}

	// Verify model produces output with correct shape.
	output, err := model.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	outShape := output.Shape()
	if outShape[0] != 1 || outShape[1] != 2 {
		t.Errorf("output shape = %v, want [1, 2]", outShape)
	}
}

func TestPatchTST_ChannelIndependence(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := PatchTSTConfig{
		InputLength:        16,
		PatchLength:        4,
		Stride:             4,
		DModel:             8,
		NHeads:             2,
		NLayers:            1,
		OutputDim:          2,
		ChannelIndependent: true,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Create multivariate input [1, 3, 16] where channels have different data.
	channels := 3
	data := make([]float32, channels*16)
	for c := range channels {
		for i := range 16 {
			data[c*16+i] = float32(c*100 + i)
		}
	}
	input, err := tensor.New[float32]([]int{1, channels, 16}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	output, err := model.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Output should be [1, 3, 2] — each channel processed independently.
	gotShape := output.Shape()
	wantShape := []int{1, 3, 2}
	if len(gotShape) != len(wantShape) {
		t.Fatalf("output shape = %v, want %v", gotShape, wantShape)
	}
	for i := range gotShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, gotShape[i], wantShape[i])
		}
	}

	// Verify channel independence: running each channel separately should
	// produce the same result as running all channels together.
	outData := output.Data()
	for c := range channels {
		chData := data[c*16 : (c+1)*16]
		chInput, err := tensor.New[float32]([]int{1, 16}, chData)
		if err != nil {
			t.Fatalf("tensor.New channel %d: %v", c, err)
		}

		chOutput, err := model.Forward(ctx, chInput)
		if err != nil {
			t.Fatalf("Forward channel %d: %v", c, err)
		}

		chOutData := chOutput.Data()
		for i := range chOutData {
			multiIdx := c*config.OutputDim + i
			if math.Abs(float64(chOutData[i]-outData[multiIdx])) > 1e-5 {
				t.Errorf("channel %d output[%d] = %v (independent) vs %v (batched), diff > 1e-5",
					c, i, chOutData[i], outData[multiIdx])
			}
		}
	}
}

func TestPatchTST_Predict(t *testing.T) {
	engine, ops := newTestEngine()

	config := PatchTSTConfig{
		InputLength:        16,
		PatchLength:        4,
		Stride:             4,
		DModel:             8,
		NHeads:             2,
		NLayers:            1,
		OutputDim:          3,
		ChannelIndependent: true,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Single channel.
	input1 := [][]float64{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}
	out1, err := model.Predict(input1)
	if err != nil {
		t.Fatalf("Predict single channel: %v", err)
	}
	if len(out1) != 1 || len(out1[0]) != 3 {
		t.Fatalf("Predict single channel: got %d channels with %d outputs, want 1 channel with 3", len(out1), len(out1[0]))
	}

	// Multi-channel.
	input2 := [][]float64{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115},
	}
	out2, err := model.Predict(input2)
	if err != nil {
		t.Fatalf("Predict multi-channel: %v", err)
	}
	if len(out2) != 2 || len(out2[0]) != 3 || len(out2[1]) != 3 {
		t.Fatalf("Predict multi-channel: unexpected shape")
	}

	// Verify finite values.
	for c, ch := range out2 {
		for i, v := range ch {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("Predict output[%d][%d] = %v, want finite", c, i, v)
			}
		}
	}
}

func TestPatchTST_TrainWindowed(t *testing.T) {
	engine, ops := newTestEngine()

	config := PatchTSTConfig{
		InputLength: 16,
		PatchLength: 4,
		Stride:      4,
		DModel:      8,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   2,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	nSamples := 10
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.OutputDim)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1) // 1 channel
		windows[s][0] = make([]float64, config.InputLength)
		for i := 0; i < config.InputLength; i++ {
			windows[s][0][i] = float64(s+i) * 0.1
		}
		for o := 0; o < config.OutputDim; o++ {
			labels[s*config.OutputDim+o] = float64(s) * 0.05
		}
	}

	result, err := model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:  5,
		LR:      1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if len(result.LossHistory) != 5 {
		t.Fatalf("loss history length = %d, want 5", len(result.LossHistory))
	}
	if result.FinalLoss != result.LossHistory[4] {
		t.Errorf("FinalLoss = %v, want %v", result.FinalLoss, result.LossHistory[4])
	}
	if math.IsNaN(result.FinalLoss) || math.IsInf(result.FinalLoss, 0) {
		t.Errorf("FinalLoss is not finite: %v", result.FinalLoss)
	}
	if result.Metrics == nil {
		t.Fatal("Metrics is nil")
	}
	if _, ok := result.Metrics["mse"]; !ok {
		t.Error("Metrics missing 'mse' key")
	}
}

func TestPatchTST_PredictWindowed(t *testing.T) {
	engine, ops := newTestEngine()

	config := PatchTSTConfig{
		InputLength: 16,
		PatchLength: 4,
		Stride:      4,
		DModel:      8,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   2,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	windows := make([][][]float64, 3)
	for s := 0; s < 3; s++ {
		windows[s] = [][]float64{make([]float64, config.InputLength)}
		for i := range windows[s][0] {
			windows[s][0][i] = float64(i) * 0.1
		}
	}

	preds, err := model.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	expectedLen := 3 * config.OutputDim
	if len(preds) != expectedLen {
		t.Fatalf("predictions length = %d, want %d", len(preds), expectedLen)
	}

	for i, v := range preds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("prediction[%d] = %v, want finite", i, v)
			break
		}
	}
}

func TestPatchTST_SaveLoadWeights(t *testing.T) {
	engine, ops := newTestEngine()

	config := PatchTSTConfig{
		InputLength: 16,
		PatchLength: 4,
		Stride:      4,
		DModel:      8,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   2,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	dir := t.TempDir()
	path := dir + "/patchtst.json"

	if err := model.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	windows := [][][]float64{
		{make([]float64, config.InputLength)},
	}
	for i := range windows[0][0] {
		windows[0][0][i] = float64(i)
	}

	preds1, err := model.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	model2, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	preds2, err := model2.PredictWindowed(path, windows)
	if err != nil {
		t.Fatalf("PredictWindowed with load: %v", err)
	}

	for i := range preds1 {
		if math.Abs(preds1[i]-preds2[i]) > 1e-6 {
			t.Errorf("loaded model prediction[%d] = %v, want %v", i, preds2[i], preds1[i])
		}
	}
}

func TestPatchTST_TrainWindowed_Empty(t *testing.T) {
	engine, ops := newTestEngine()
	config := PatchTSTConfig{
		InputLength: 16, PatchLength: 4, Stride: 4,
		DModel: 8, NHeads: 2, NLayers: 1, OutputDim: 2,
	}
	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	_, err = model.TrainWindowed(nil, nil, TrainConfig{Epochs: 5})
	if err == nil {
		t.Fatal("expected error for empty training set")
	}
}

func TestPatchTST_PredictWindowed_Empty(t *testing.T) {
	engine, ops := newTestEngine()
	config := PatchTSTConfig{
		InputLength: 16, PatchLength: 4, Stride: 4,
		DModel: 8, NHeads: 2, NLayers: 1, OutputDim: 2,
	}
	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	_, err = model.PredictWindowed("", nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestNewPatchTST_Validation(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  PatchTSTConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 8, Stride: 4,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
		},
		{
			name: "zero input length",
			config: PatchTSTConfig{
				InputLength: 0, PatchLength: 8, Stride: 4,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "zero patch length",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 0, Stride: 4,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "patch longer than input",
			config: PatchTSTConfig{
				InputLength: 4, PatchLength: 8, Stride: 2,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "d_model not divisible by n_heads",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 8, Stride: 4,
				DModel: 15, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "zero stride",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 8, Stride: 0,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "zero output dim",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 8, Stride: 4,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewPatchTST(tt.config, engine, ops)
			if tt.wantErr && err == nil {
				t.Fatal("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestPatchTST_TrainWindowed_MultiScale(t *testing.T) {
	// Issue #121: training on data with features spanning 10 orders of magnitude
	// previously produced NaN/Inf weights. Normalization should prevent this.
	// PatchTST uses numerical gradients (O(nParams * forward) per sample), so
	// we keep the test small to avoid long runtimes.
	engine, ops := newTestEngine()

	nChannels := 3
	inputLen := 8
	outputDim := 1
	config := PatchTSTConfig{
		InputLength: inputLen,
		PatchLength: 4,
		Stride:      4,
		DModel:      8,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   outputDim,
	}

	m, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	windows, labels := makeMultiScaleWindows(20, nChannels, inputLen, outputDim)

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       5,
		LR:           1e-3,
		GradClip:     1.0,
		WarmupEpochs: 3,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	t.Logf("multi-scale training: final_loss=%.6f (5 epochs, 3 channels, 20 samples)", result.FinalLoss)
}

func TestPatchTST_TrainWindowed_EngineConvergence(t *testing.T) {
	engine, ops := newTestEngine()

	config := PatchTSTConfig{
		InputLength: 8,
		PatchLength: 4,
		Stride:      4,
		DModel:      8,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   1,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	nSamples := 15
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.OutputDim)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, 1)
		windows[s][0] = make([]float64, config.InputLength)
		sum := 0.0
		for i := 0; i < config.InputLength; i++ {
			v := float64(s*config.InputLength+i) * 0.01
			windows[s][0][i] = v
			sum += v
		}
		labels[s] = sum / float64(config.InputLength)
	}

	result, err := model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   10,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if len(result.LossHistory) != 10 {
		t.Fatalf("loss history length = %d, want 10", len(result.LossHistory))
	}

	for i, l := range result.LossHistory {
		if !isFinite(l) {
			t.Fatalf("epoch %d: loss is not finite: %v", i, l)
		}
	}

	if result.LossHistory[9] >= result.LossHistory[0] {
		t.Errorf("loss did not decrease: epoch 0 = %v, epoch 9 = %v",
			result.LossHistory[0], result.LossHistory[9])
	}

	t.Logf("engine convergence: loss[0]=%.6f -> loss[9]=%.6f",
		result.LossHistory[0], result.LossHistory[9])
}

func TestPatchTST_PredictWindowed_NormalizationApplied(t *testing.T) {
	engine, ops := newTestEngine()

	nChannels := 3
	inputLen := 8
	outputDim := 1
	config := PatchTSTConfig{
		InputLength: inputLen,
		PatchLength: 4,
		Stride:      4,
		DModel:      8,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   outputDim,
	}

	m, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	windows, labels := makeMultiScaleWindows(20, nChannels, inputLen, outputDim)

	_, err = m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       5,
		LR:           1e-3,
		GradClip:     1.0,
		WarmupEpochs: 2,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if m.normMeans == nil || m.normStds == nil {
		t.Fatal("normMeans/normStds not stored after training")
	}

	preds, err := m.PredictWindowed("", windows[:5])
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	for i, v := range preds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("prediction[%d] = %v, want finite", i, v)
		}
	}
}

// TestPatchTST_GradientVerification checks that analytical gradients match
// finite-difference gradients for the CPU backward pass.
func TestPatchTST_GradientVerification(t *testing.T) {
	config := PatchTSTConfig{
		InputLength: 8,
		PatchLength: 4,
		Stride:      4,
		DModel:      4,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   2,
	}

	m, err := NewPatchTST(config, nil, nil)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	params := m.extractParamsF64()

	// Single-channel input.
	input := [][]float64{{0.1, -0.2, 0.3, 0.4, -0.1, 0.5, -0.3, 0.2}}
	labels := []float64{1.0, -0.5}

	// Forward with cache + analytical backward.
	pred, cache := m.forwardF64WithCache(input, params)
	outDim := config.OutputDim
	dOutput := make([]float64, outDim)
	for j := 0; j < outDim; j++ {
		diff := pred[j] - labels[j]
		dOutput[j] = 2.0 * diff / float64(outDim) // MSE gradient
	}
	analyticalGrads := m.backwardF64(dOutput, params, cache)

	// Finite-difference gradients.
	eps := 1e-5
	flatP := params.flatParams()
	nParams := len(flatP)

	maxRelErr := 0.0
	failCount := 0
	for pi := 0; pi < nParams; pi++ {
		orig := *flatP[pi]

		*flatP[pi] = orig + eps
		predPlus := m.forwardF64(input, params)
		lossPlus := 0.0
		for j := 0; j < outDim; j++ {
			diff := predPlus[j] - labels[j]
			lossPlus += diff * diff
		}
		lossPlus /= float64(outDim)

		*flatP[pi] = orig - eps
		predMinus := m.forwardF64(input, params)
		lossMinus := 0.0
		for j := 0; j < outDim; j++ {
			diff := predMinus[j] - labels[j]
			lossMinus += diff * diff
		}
		lossMinus /= float64(outDim)

		*flatP[pi] = orig
		fdGrad := (lossPlus - lossMinus) / (2 * eps)

		// Relative error with numerical stability.
		absErr := math.Abs(analyticalGrads[pi] - fdGrad)
		denom := math.Max(math.Abs(analyticalGrads[pi]), math.Abs(fdGrad))
		if denom < 1e-8 {
			denom = 1e-8
		}
		relErr := absErr / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		// Skip near-zero gradients where relative error is meaningless
		// (analytical ~ 1e-17 vs finite-diff ~ 1e-10 is numerical noise).
		if math.Abs(analyticalGrads[pi]) < 1e-12 && math.Abs(fdGrad) < 1e-6 {
			continue
		}
		if relErr > 1e-2 {
			failCount++
			if failCount <= 5 {
				t.Errorf("param[%d]: analytical=%.8e, fd=%.8e, relErr=%.4e",
					pi, analyticalGrads[pi], fdGrad, relErr)
			}
		}
	}

	if failCount > 0 {
		t.Errorf("%d/%d parameters exceed 1%% relative error", failCount, nParams)
	}
	t.Logf("gradient check: %d params, maxRelErr=%.4e, failures=%d", nParams, maxRelErr, failCount)
}

// TestPatchTST_BatchedTrainConvergence verifies that training with the batched
// forward pass produces decreasing loss, confirming end-to-end correctness.
func TestPatchTST_BatchedTrainConvergence(t *testing.T) {
	engine, ops := newTestEngine()

	config := PatchTSTConfig{
		InputLength: 8,
		PatchLength: 4,
		Stride:      4,
		DModel:      8,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   1,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	nSamples := 20
	nChannels := 2
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.OutputDim)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, nChannels)
		sum := 0.0
		for c := 0; c < nChannels; c++ {
			windows[s][c] = make([]float64, config.InputLength)
			for i := 0; i < config.InputLength; i++ {
				v := float64(s*config.InputLength+c*10+i) * 0.01
				windows[s][c][i] = v
				sum += v
			}
		}
		labels[s] = sum / float64(nChannels*config.InputLength)
	}

	result, err := model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:    10,
		LR:        1e-3,
		GradClip:  1.0,
		BatchSize: 5,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if len(result.LossHistory) != 10 {
		t.Fatalf("loss history length = %d, want 10", len(result.LossHistory))
	}

	for i, l := range result.LossHistory {
		if !isFinite(l) {
			t.Fatalf("epoch %d: loss is not finite: %v", i, l)
		}
	}

	if result.LossHistory[9] >= result.LossHistory[0] {
		t.Errorf("loss did not decrease: epoch 0 = %v, epoch 9 = %v",
			result.LossHistory[0], result.LossHistory[9])
	}

	t.Logf("batched train convergence: loss[0]=%.6f -> loss[9]=%.6f (batch_size=5, %d samples, %d channels)",
		result.LossHistory[0], result.LossHistory[9], nSamples, nChannels)
}
