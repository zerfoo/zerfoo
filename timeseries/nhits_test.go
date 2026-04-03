package timeseries

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestNHiTS_NewValidation(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  NHiTSConfig
		wantErr bool
	}{
		{
			name: "valid 3-stack",
			config: NHiTSConfig{
				InputLength:  24,
				OutputLength: 12,
				Channels:     1,
				PoolKernels:  []int{2, 4, 8},
				HiddenSize:   32,
			},
		},
		{
			name: "valid single stack multichannel",
			config: NHiTSConfig{
				InputLength:  16,
				OutputLength: 8,
				Channels:     3,
				PoolKernels:  []int{4},
				HiddenSize:   16,
			},
		},
		{
			name: "zero input length",
			config: NHiTSConfig{
				InputLength:  0,
				OutputLength: 5,
				Channels:     1,
				PoolKernels:  []int{2},
				HiddenSize:   16,
			},
			wantErr: true,
		},
		{
			name: "zero output length",
			config: NHiTSConfig{
				InputLength:  10,
				OutputLength: 0,
				Channels:     1,
				PoolKernels:  []int{2},
				HiddenSize:   16,
			},
			wantErr: true,
		},
		{
			name: "zero channels",
			config: NHiTSConfig{
				InputLength:  10,
				OutputLength: 5,
				Channels:     0,
				PoolKernels:  []int{2},
				HiddenSize:   16,
			},
			wantErr: true,
		},
		{
			name: "empty pool kernels",
			config: NHiTSConfig{
				InputLength:  10,
				OutputLength: 5,
				Channels:     1,
				PoolKernels:  []int{},
				HiddenSize:   16,
			},
			wantErr: true,
		},
		{
			name: "zero hidden size",
			config: NHiTSConfig{
				InputLength:  10,
				OutputLength: 5,
				Channels:     1,
				PoolKernels:  []int{2},
				HiddenSize:   0,
			},
			wantErr: true,
		},
		{
			name: "pool kernel exceeds input",
			config: NHiTSConfig{
				InputLength:  4,
				OutputLength: 2,
				Channels:     1,
				PoolKernels:  []int{2, 8},
				HiddenSize:   16,
			},
			wantErr: true,
		},
		{
			name: "negative pool kernel",
			config: NHiTSConfig{
				InputLength:  10,
				OutputLength: 5,
				Channels:     1,
				PoolKernels:  []int{-1},
				HiddenSize:   16,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewNHiTS(tt.config, engine, ops)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if m == nil {
				t.Fatal("expected non-nil model")
			}
			if len(m.stacks) != len(tt.config.PoolKernels) {
				t.Errorf("expected %d stacks, got %d", len(tt.config.PoolKernels), len(m.stacks))
			}
			for i, s := range m.stacks {
				if s.poolKernel != tt.config.PoolKernels[i] {
					t.Errorf("stack %d pool kernel = %d, want %d", i, s.poolKernel, tt.config.PoolKernels[i])
				}
			}
		})
	}
}

func TestMaxPool1D(t *testing.T) {
	tests := []struct {
		name   string
		data   []float32
		length int
		kernel int
		want   []float32
	}{
		{
			name:   "basic kernel=2",
			data:   []float32{1, 3, 2, 4, 5, 1},
			length: 6,
			kernel: 2,
			want:   []float32{3, 4, 5},
		},
		{
			name:   "kernel=3",
			data:   []float32{1, 5, 2, 3, 8, 4},
			length: 6,
			kernel: 3,
			want:   []float32{5, 8},
		},
		{
			name:   "kernel=1 identity",
			data:   []float32{3, 1, 4, 1, 5},
			length: 5,
			kernel: 1,
			want:   []float32{3, 1, 4, 1, 5},
		},
		{
			name:   "batch=2 kernel=2",
			data:   []float32{1, 4, 2, 3, 5, 2, 6, 1},
			length: 4,
			kernel: 2,
			want:   []float32{4, 3, 5, 6},
		},
		{
			name:   "negative values",
			data:   []float32{-5, -1, -3, -2},
			length: 4,
			kernel: 2,
			want:   []float32{-1, -2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := maxPool1D(tt.data, tt.length, tt.kernel)
			if len(got) != len(tt.want) {
				t.Fatalf("length = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("maxPool1D[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestLinearInterpolation(t *testing.T) {
	// N-HiTS uses direct MLP output for each stack (no interpolation needed
	// since output proj maps directly to outputLen). Verify the output
	// projection dimensions are correct by checking forward pass shapes.
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := NHiTSConfig{
		InputLength:  12,
		OutputLength: 6,
		Channels:     1,
		PoolKernels:  []int{2, 4},
		HiddenSize:   16,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}

	// Verify stack output dimensions match outputLen.
	for i, s := range m.stacks {
		projShape := s.outputProj.weights.Shape()
		if projShape[0] != config.OutputLength {
			t.Errorf("stack %d output proj rows = %d, want %d", i, projShape[0], config.OutputLength)
		}
	}

	// Verify forward produces correct shape.
	data := make([]float32, 2*12)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	x, err := tensor.New[float32]([]int{2, 12}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	pred, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if s := pred.Shape(); s[0] != 2 || s[1] != 6 {
		t.Errorf("output shape = %v, want [2, 6]", s)
	}
}

func TestNHiTS_Forward(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name       string
		config     NHiTSConfig
		batch      int
		wantErr    bool
		errOnInput bool
	}{
		{
			name: "single stack single batch",
			config: NHiTSConfig{
				InputLength:  12,
				OutputLength: 6,
				Channels:     1,
				PoolKernels:  []int{2},
				HiddenSize:   16,
			},
			batch: 1,
		},
		{
			name: "3-stack batch 4",
			config: NHiTSConfig{
				InputLength:  24,
				OutputLength: 12,
				Channels:     1,
				PoolKernels:  []int{2, 4, 8},
				HiddenSize:   32,
			},
			batch: 4,
		},
		{
			name: "multichannel",
			config: NHiTSConfig{
				InputLength:  16,
				OutputLength: 8,
				Channels:     3,
				PoolKernels:  []int{2, 4},
				HiddenSize:   16,
			},
			batch: 2,
		},
		{
			name: "wrong input shape",
			config: NHiTSConfig{
				InputLength:  10,
				OutputLength: 5,
				Channels:     1,
				PoolKernels:  []int{2},
				HiddenSize:   16,
			},
			batch:      1,
			wantErr:    true,
			errOnInput: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewNHiTS(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewNHiTS: %v", err)
			}

			cols := tt.config.Channels * tt.config.InputLength
			if tt.errOnInput {
				cols += 5
			}
			data := make([]float32, tt.batch*cols)
			for i := range data {
				data[i] = float32(i) * 0.01
			}
			x, err := tensor.New[float32]([]int{tt.batch, cols}, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			pred, err := m.Forward(ctx, x)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			shape := pred.Shape()
			if len(shape) != 2 || shape[0] != tt.batch || shape[1] != tt.config.OutputLength {
				t.Errorf("output shape = %v, want [%d, %d]", shape, tt.batch, tt.config.OutputLength)
			}

			pData := pred.Data()
			for i, v := range pData {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v, want finite", i, v)
					break
				}
			}
		})
	}
}

func TestNHiTS_ForwardConsistency(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := NHiTSConfig{
		InputLength:  12,
		OutputLength: 6,
		Channels:     1,
		PoolKernels:  []int{2, 4},
		HiddenSize:   16,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}

	data := make([]float32, 2*12)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	x, err := tensor.New[float32]([]int{2, 12}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	out1, err := m.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward 1: %v", err)
	}

	for i := 0; i < 5; i++ {
		out2, err := m.Forward(ctx, x)
		if err != nil {
			t.Fatalf("Forward %d: %v", i+2, err)
		}
		d1 := out1.Data()
		d2 := out2.Data()
		for j := range d1 {
			if d1[j] != d2[j] {
				t.Errorf("iteration %d: output[%d] = %v, want %v", i+2, j, d2[j], d1[j])
				break
			}
		}
	}
}

func TestNHiTS_ConvergenceSynthetic(t *testing.T) {
	engine, ops := newTestEngine()

	config := NHiTSConfig{
		InputLength:  8,
		OutputLength: 4,
		Channels:     1,
		PoolKernels:  []int{2, 4},
		HiddenSize:   16,
		NumMLPLayers: 2,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}
	m.initWeightsSmall()

	// Generate synthetic data: simple linear trend y = 0.1 * t.
	numSamples := 32
	windows := make([][][]float64, numSamples)
	labels := make([]float64, numSamples*config.OutputLength)

	for i := 0; i < numSamples; i++ {
		offset := float64(i)
		ch := make([]float64, config.InputLength)
		for t := 0; t < config.InputLength; t++ {
			ch[t] = 0.1 * (offset + float64(t))
		}
		windows[i] = [][]float64{ch}
		for t := 0; t < config.OutputLength; t++ {
			labels[i*config.OutputLength+t] = 0.1 * (offset + float64(config.InputLength+t))
		}
	}

	tc := TrainConfig{
		Epochs:       200,
		LR: 1e-3,
		BatchSize:    16,
		GradClip:     1.0,
	}

	result, err := m.TrainWindowed(windows, labels, tc)
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Loss should decrease.
	if len(result.LossHistory) < 2 {
		t.Fatal("expected at least 2 epoch losses")
	}
	firstLoss := result.LossHistory[0]
	finalLoss := result.FinalLoss
	if finalLoss >= firstLoss {
		t.Errorf("loss did not decrease: first=%v, final=%v", firstLoss, finalLoss)
	}

	// Final loss should be reasonably small.
	if finalLoss > 1.0 {
		t.Errorf("final loss %v too high (want < 1.0)", finalLoss)
	}

	t.Logf("convergence: first_loss=%.6f final_loss=%.6f", firstLoss, finalLoss)
}

func TestNHiTS_RoundTrip(t *testing.T) {
	engine, ops := newTestEngine()

	config := NHiTSConfig{
		InputLength:  8,
		OutputLength: 4,
		Channels:     1,
		PoolKernels:  []int{2, 4},
		HiddenSize:   16,
		NumMLPLayers: 2,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}
	m.initWeightsSmall()

	// Train briefly.
	numSamples := 16
	windows := make([][][]float64, numSamples)
	labels := make([]float64, numSamples*config.OutputLength)
	for i := 0; i < numSamples; i++ {
		ch := make([]float64, config.InputLength)
		for t := 0; t < config.InputLength; t++ {
			ch[t] = 0.1 * float64(t)
		}
		windows[i] = [][]float64{ch}
		for t := 0; t < config.OutputLength; t++ {
			labels[i*config.OutputLength+t] = 0.1 * float64(config.InputLength+t)
		}
	}

	tc := TrainConfig{
		Epochs:       50,
		LR: 1e-3,
		BatchSize:    16,
	}

	_, err = m.TrainWindowed(windows, labels, tc)
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Save model.
	tmpPath := t.TempDir() + "/nhits.json"
	if err := m.Save(tmpPath); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Create new model and load.
	m2, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}

	testWindows := windows[:4]
	pred, err := m2.PredictWindowed(tmpPath, testWindows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	if len(pred) != 4*config.OutputLength {
		t.Fatalf("prediction length = %d, want %d", len(pred), 4*config.OutputLength)
	}

	// Predictions should be finite.
	for i, v := range pred {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("pred[%d] = %v, want finite", i, v)
			break
		}
	}

	// Compare with original model's predictions.
	origPred, err := m.PredictWindowed("", testWindows)
	if err != nil {
		t.Fatalf("original PredictWindowed: %v", err)
	}

	for i := range pred {
		diff := math.Abs(pred[i] - origPred[i])
		if diff > 1e-4 {
			t.Errorf("pred[%d] = %v, orig = %v, diff = %v", i, pred[i], origPred[i], diff)
			break
		}
	}
}

func TestNHiTS_TrainWindowed_MultiScale(t *testing.T) {
	// Issue #121: training on data with features spanning 10 orders of magnitude
	// previously produced NaN/Inf weights. Normalization should prevent this.
	engine, ops := newTestEngine()

	nChannels := 5
	inputLen := 8
	outputLen := 4
	config := NHiTSConfig{
		InputLength:  inputLen,
		OutputLength: outputLen,
		Channels:     nChannels,
		PoolKernels:  []int{2, 4},
		HiddenSize:   16,
		NumMLPLayers: 2,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}
	m.initWeightsSmall()

	windows, labels := makeMultiScaleWindows(500, nChannels, inputLen, outputLen)

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       20,
		LR:           1e-3,
		BatchSize:    32,
		GradClip:     1.0,
		WarmupEpochs: 5,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	t.Logf("multi-scale training: final_loss=%.6f (20 epochs, 5 channels, 500 samples)", result.FinalLoss)
}

func TestNHiTS_TrainWindowed_EngineBackward(t *testing.T) {
	// Verify that the engine-accelerated backward path produces finite loss
	// and converges, matching the behavior of the CPU path.
	engine, ops := newTestEngine()

	config := NHiTSConfig{
		InputLength:  8,
		OutputLength: 4,
		Channels:     1,
		PoolKernels:  []int{2, 4},
		HiddenSize:   16,
		NumMLPLayers: 2,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}
	m.initWeightsSmall()

	// Confirm engine is set (GPU path will be used).
	if m.engine == nil {
		t.Fatal("expected engine to be set")
	}

	// Generate synthetic data: simple linear trend.
	numSamples := 32
	windows := make([][][]float64, numSamples)
	labels := make([]float64, numSamples*config.OutputLength)

	for i := 0; i < numSamples; i++ {
		offset := float64(i)
		ch := make([]float64, config.InputLength)
		for tt := 0; tt < config.InputLength; tt++ {
			ch[tt] = 0.1 * (offset + float64(tt))
		}
		windows[i] = [][]float64{ch}
		for tt := 0; tt < config.OutputLength; tt++ {
			labels[i*config.OutputLength+tt] = 0.1 * (offset + float64(config.InputLength+tt))
		}
	}

	tc := TrainConfig{
		Epochs:    200,
		LR:        1e-3,
		BatchSize: 16,
		GradClip:  1.0,
	}

	result, err := m.TrainWindowed(windows, labels, tc)
	if err != nil {
		t.Fatalf("TrainWindowed with engine backward: %v", err)
	}

	// Loss must be finite throughout.
	for i, loss := range result.LossHistory {
		if !isFinite(loss) {
			t.Fatalf("epoch %d: loss = %v, want finite", i, loss)
		}
	}

	// Loss should decrease.
	if len(result.LossHistory) < 2 {
		t.Fatal("expected at least 2 epoch losses")
	}
	firstLoss := result.LossHistory[0]
	finalLoss := result.FinalLoss
	if finalLoss >= firstLoss {
		t.Errorf("loss did not decrease: first=%v, final=%v", firstLoss, finalLoss)
	}

	// Final loss should be reasonably small.
	if finalLoss > 1.0 {
		t.Errorf("final loss %v too high (want < 1.0)", finalLoss)
	}

	t.Logf("engine backward: first_loss=%.6f final_loss=%.6f", firstLoss, finalLoss)
}

func TestNHiTS_TrainWindowed_EngineBackward_MultiChannel(t *testing.T) {
	// Verify engine backward path works with multiple channels.
	engine, ops := newTestEngine()

	config := NHiTSConfig{
		InputLength:  12,
		OutputLength: 4,
		Channels:     3,
		PoolKernels:  []int{2, 4},
		HiddenSize:   16,
		NumMLPLayers: 2,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}
	m.initWeightsSmall()

	numSamples := 24
	windows := make([][][]float64, numSamples)
	labels := make([]float64, numSamples*config.OutputLength)

	for i := 0; i < numSamples; i++ {
		w := make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			ch := make([]float64, config.InputLength)
			for tt := 0; tt < config.InputLength; tt++ {
				ch[tt] = 0.01 * float64(c+tt+i)
			}
			w[c] = ch
		}
		windows[i] = w
		for tt := 0; tt < config.OutputLength; tt++ {
			labels[i*config.OutputLength+tt] = 0.01 * float64(i+config.InputLength+tt)
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       100,
		LR:           1e-3,
		BatchSize:    12,
		GradClip:     1.0,
		WarmupEpochs: 5,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	if result.FinalLoss >= result.LossHistory[0] {
		t.Errorf("loss did not decrease: first=%v, final=%v", result.LossHistory[0], result.FinalLoss)
	}

	t.Logf("engine backward multichannel: final_loss=%.6f", result.FinalLoss)
}

func TestNHiTS_HighChannelNoNilPanic(t *testing.T) {
	// Issue #123: NHiTS panics with nil pointer dereference in linearForward
	// when called via TrainWindowed with 132 channels and various window sizes.
	engine, ops := newTestEngine()

	channels := 132
	outputLen := 4

	for _, inputLen := range []int{15, 30, 60, 120, 240} {
		t.Run(fmt.Sprintf("window_%d", inputLen), func(t *testing.T) {
			// Choose pool kernels that are valid for this input length.
			var poolKernels []int
			for _, k := range []int{2, 4, 8} {
				if k <= inputLen {
					poolKernels = append(poolKernels, k)
				}
			}

			config := NHiTSConfig{
				InputLength:  inputLen,
				OutputLength: outputLen,
				Channels:     channels,
				PoolKernels:  poolKernels,
				HiddenSize:   32,
				NumMLPLayers: 2,
			}

			m, err := NewNHiTS(config, engine, ops)
			if err != nil {
				t.Fatalf("NewNHiTS: %v", err)
			}
			m.initWeightsSmall()

			// Generate synthetic training data.
			numSamples := 16
			windows := make([][][]float64, numSamples)
			labels := make([]float64, numSamples*outputLen)
			for i := 0; i < numSamples; i++ {
				w := make([][]float64, channels)
				for c := 0; c < channels; c++ {
					ch := make([]float64, inputLen)
					for tt := 0; tt < inputLen; tt++ {
						ch[tt] = 0.01 * float64(c+tt+i)
					}
					w[c] = ch
				}
				windows[i] = w
				for tt := 0; tt < outputLen; tt++ {
					labels[i*outputLen+tt] = 0.01 * float64(i+inputLen+tt)
				}
			}

			result, err := m.TrainWindowed(windows, labels, TrainConfig{
				Epochs:       10,
				LR:           1e-4,
				BatchSize:    8,
				GradClip:     1.0,
				WarmupEpochs: 3,
			})
			if err != nil {
				t.Fatalf("TrainWindowed (window=%d): %v", inputLen, err)
			}

			if !isFinite(result.FinalLoss) {
				t.Fatalf("final loss = %v, want finite (window=%d)", result.FinalLoss, inputLen)
			}

			t.Logf("window=%d: final_loss=%.6f", inputLen, result.FinalLoss)
		})
	}
}

func TestNHiTS_SmallInputLen_NoSegfault(t *testing.T) {
	// Issue #152: NHiTS segfaults when inputLen is small relative to poolKernel.
	// This regression test verifies no panic occurs and training completes with finite loss.
	engine, ops := newTestEngine()

	config := NHiTSConfig{
		InputLength:  10,
		OutputLength: 4,
		Channels:     10,
		PoolKernels:  []int{2, 4, 8},
		HiddenSize:   16,
		NumMLPLayers: 2,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}
	m.initWeightsSmall()

	numSamples := 32
	windows := make([][][]float64, numSamples)
	labels := make([]float64, numSamples*config.OutputLength)

	for i := 0; i < numSamples; i++ {
		w := make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			ch := make([]float64, config.InputLength)
			for tt := 0; tt < config.InputLength; tt++ {
				ch[tt] = 0.01 * float64(c+tt+i)
			}
			w[c] = ch
		}
		windows[i] = w
		for tt := 0; tt < config.OutputLength; tt++ {
			labels[i*config.OutputLength+tt] = 0.01 * float64(i+config.InputLength+tt)
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       20,
		LR:           1e-3,
		BatchSize:    16,
		GradClip:     1.0,
		WarmupEpochs: 5,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	t.Logf("small inputLen regression: final_loss=%.6f", result.FinalLoss)
}

func TestNHiTS_PoolKernelEqualsInputLen(t *testing.T) {
	// Edge case: poolKernel == inputLen should produce pooledLen=1 and work correctly.
	engine, ops := newTestEngine()

	config := NHiTSConfig{
		InputLength:  8,
		OutputLength: 4,
		Channels:     1,
		PoolKernels:  []int{8},
		HiddenSize:   16,
		NumMLPLayers: 2,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}
	m.initWeightsSmall()

	numSamples := 16
	windows := make([][][]float64, numSamples)
	labels := make([]float64, numSamples*config.OutputLength)

	for i := 0; i < numSamples; i++ {
		ch := make([]float64, config.InputLength)
		for tt := 0; tt < config.InputLength; tt++ {
			ch[tt] = 0.1 * float64(tt+i)
		}
		windows[i] = [][]float64{ch}
		for tt := 0; tt < config.OutputLength; tt++ {
			labels[i*config.OutputLength+tt] = 0.1 * float64(config.InputLength+tt+i)
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       20,
		LR:           1e-3,
		BatchSize:    8,
		GradClip:     1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	t.Logf("poolKernel==inputLen: final_loss=%.6f", result.FinalLoss)
}

func TestNHiTS_PredictWindowed_NormalizationApplied(t *testing.T) {
	engine, ops := newTestEngine()

	nChannels := 3
	inputLen := 8
	outputLen := 4
	config := NHiTSConfig{
		InputLength:  inputLen,
		OutputLength: outputLen,
		Channels:     nChannels,
		PoolKernels:  []int{2, 4},
		HiddenSize:   16,
		NumMLPLayers: 2,
	}

	m, err := NewNHiTS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNHiTS: %v", err)
	}
	m.initWeightsSmall()

	windows, labels := makeMultiScaleWindows(100, nChannels, inputLen, outputLen)

	_, err = m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       10,
		LR:           1e-3,
		BatchSize:    32,
		GradClip:     1.0,
		WarmupEpochs: 3,
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

func TestNHiTS_ForwardBatchEngine(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name     string
		config   NHiTSConfig
		batch    int
	}{
		{
			name: "single channel single stack",
			config: NHiTSConfig{
				InputLength:  12,
				OutputLength: 6,
				Channels:     1,
				PoolKernels:  []int{2},
				HiddenSize:   16,
			},
			batch: 4,
		},
		{
			name: "multi channel 3 stacks",
			config: NHiTSConfig{
				InputLength:  24,
				OutputLength: 12,
				Channels:     3,
				PoolKernels:  []int{2, 4, 8},
				HiddenSize:   32,
			},
			batch: 5,
		},
		{
			name: "single sample",
			config: NHiTSConfig{
				InputLength:  8,
				OutputLength: 4,
				Channels:     2,
				PoolKernels:  []int{2, 4},
				HiddenSize:   16,
				NumMLPLayers: 3,
			},
			batch: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewNHiTS(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewNHiTS: %v", err)
			}
			m.initWeightsSmall()

			channels := tt.config.Channels
			inputLen := tt.config.InputLength
			outputLen := tt.config.OutputLength

			// Build 3D input [batch, channels, inputLen].
			data3D := make([]float32, tt.batch*channels*inputLen)
			for i := range data3D {
				data3D[i] = float32(i)*0.01 + 0.1
			}
			input3D, err := tensor.New[float32]([]int{tt.batch, channels, inputLen}, data3D)
			if err != nil {
				t.Fatalf("tensor.New 3D: %v", err)
			}

			// Batched forward.
			batchOut, err := m.forwardBatchEngine(ctx, input3D)
			if err != nil {
				t.Fatalf("forwardBatchEngine: %v", err)
			}
			batchShape := batchOut.Shape()
			if len(batchShape) != 2 || batchShape[0] != tt.batch || batchShape[1] != outputLen {
				t.Fatalf("batched output shape = %v, want [%d, %d]", batchShape, tt.batch, outputLen)
			}

			// Sample-by-sample forward using existing Forward method.
			// Forward expects [batch, channels * inputLen] 2D input.
			for b := 0; b < tt.batch; b++ {
				sampleData := make([]float32, channels*inputLen)
				for c := 0; c < channels; c++ {
					for ti := 0; ti < inputLen; ti++ {
						sampleData[c*inputLen+ti] = data3D[b*channels*inputLen+c*inputLen+ti]
					}
				}
				sampleIn, err := tensor.New[float32]([]int{1, channels * inputLen}, sampleData)
				if err != nil {
					t.Fatalf("tensor.New sample: %v", err)
				}
				sampleOut, err := m.Forward(ctx, sampleIn)
				if err != nil {
					t.Fatalf("Forward sample %d: %v", b, err)
				}

				sampleOutData := sampleOut.Data()
				batchOutData := batchOut.Data()
				for o := 0; o < outputLen; o++ {
					got := batchOutData[b*outputLen+o]
					want := sampleOutData[o]
					if diff := math.Abs(float64(got - want)); diff > 1e-5 {
						t.Errorf("batch=%d, output[%d]: batched=%.8f, single=%.8f, diff=%.2e",
							b, o, got, want, diff)
					}
				}
			}
		})
	}
}
