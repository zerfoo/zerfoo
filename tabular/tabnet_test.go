package tabular

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func defaultTabNetConfig() TabNetConfig {
	return TabNetConfig{
		InputDim:              8,
		OutputDim:             3,
		NSteps:                3,
		RelaxationFactor:      1.5,
		SparsityCoefficient:   0.001,
		FeatureTransformerDim: 16,
	}
}

func TestNewTabNet(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  TabNetConfig
		wantErr bool
	}{
		{
			name:   "valid config",
			config: defaultTabNetConfig(),
		},
		{
			name: "single step",
			config: TabNetConfig{
				InputDim:              4,
				OutputDim:             3,
				NSteps:                1,
				RelaxationFactor:      1.0,
				SparsityCoefficient:   0.0,
				FeatureTransformerDim: 8,
			},
		},
		{
			name: "zero input dim",
			config: TabNetConfig{
				InputDim:              0,
				OutputDim:             3,
				NSteps:                3,
				RelaxationFactor:      1.5,
				SparsityCoefficient:   0.001,
				FeatureTransformerDim: 16,
			},
			wantErr: true,
		},
		{
			name: "zero output dim",
			config: TabNetConfig{
				InputDim:              8,
				OutputDim:             0,
				NSteps:                3,
				RelaxationFactor:      1.5,
				SparsityCoefficient:   0.001,
				FeatureTransformerDim: 16,
			},
			wantErr: true,
		},
		{
			name: "zero steps",
			config: TabNetConfig{
				InputDim:              8,
				OutputDim:             3,
				NSteps:                0,
				RelaxationFactor:      1.5,
				SparsityCoefficient:   0.001,
				FeatureTransformerDim: 16,
			},
			wantErr: true,
		},
		{
			name: "negative relaxation factor",
			config: TabNetConfig{
				InputDim:              8,
				OutputDim:             3,
				NSteps:                3,
				RelaxationFactor:      -1.0,
				SparsityCoefficient:   0.001,
				FeatureTransformerDim: 16,
			},
			wantErr: true,
		},
		{
			name: "negative sparsity coefficient",
			config: TabNetConfig{
				InputDim:              8,
				OutputDim:             3,
				NSteps:                3,
				RelaxationFactor:      1.5,
				SparsityCoefficient:   -0.1,
				FeatureTransformerDim: 16,
			},
			wantErr: true,
		},
		{
			name: "zero feature transformer dim",
			config: TabNetConfig{
				InputDim:              8,
				OutputDim:             3,
				NSteps:                3,
				RelaxationFactor:      1.5,
				SparsityCoefficient:   0.001,
				FeatureTransformerDim: 0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewTabNet(tt.config, engine, ops)
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
			if len(m.attentiveSteps) != tt.config.NSteps {
				t.Errorf("expected %d attentive steps, got %d", tt.config.NSteps, len(m.attentiveSteps))
			}
			if len(m.featureTransformers) != tt.config.NSteps {
				t.Errorf("expected %d feature transformers, got %d", tt.config.NSteps, len(m.featureTransformers))
			}
		})
	}
}

func TestTabNet_Forward(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name      string
		config    TabNetConfig
		batchSize int
		wantErr   bool
	}{
		{
			name:      "single sample",
			config:    defaultTabNetConfig(),
			batchSize: 1,
		},
		{
			name:      "batch of 4",
			config:    defaultTabNetConfig(),
			batchSize: 4,
		},
		{
			name: "small config",
			config: TabNetConfig{
				InputDim:              4,
				OutputDim:             2,
				NSteps:                2,
				RelaxationFactor:      1.0,
				SparsityCoefficient:   0.0,
				FeatureTransformerDim: 8,
			},
			batchSize: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewTabNet(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewTabNet: %v", err)
			}

			inputData := make([]float32, tt.batchSize*tt.config.InputDim)
			for i := range inputData {
				inputData[i] = float32(i) * 0.1
			}
			input, err := tensor.New[float32]([]int{tt.batchSize, tt.config.InputDim}, inputData)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			logits, err := m.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			gotShape := logits.Shape()
			if len(gotShape) != 2 || gotShape[0] != tt.batchSize || gotShape[1] != tt.config.OutputDim {
				t.Errorf("expected shape [%d, %d], got %v", tt.batchSize, tt.config.OutputDim, gotShape)
			}

			// Verify logits are finite.
			for i, v := range logits.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("logit[%d] = %v, expected finite", i, v)
				}
			}

			// Verify attention masks were stored.
			masks := m.AttentionMasks()
			if len(masks) != tt.config.NSteps {
				t.Errorf("expected %d attention masks, got %d", tt.config.NSteps, len(masks))
			}
			for step, mask := range masks {
				maskShape := mask.Shape()
				if len(maskShape) != 2 || maskShape[0] != tt.batchSize || maskShape[1] != tt.config.InputDim {
					t.Errorf("step %d: expected mask shape [%d, %d], got %v", step, tt.batchSize, tt.config.InputDim, maskShape)
				}
			}
		})
	}
}

func TestTabNet_Forward_WrongShape(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	m, err := NewTabNet(defaultTabNetConfig(), engine, ops)
	if err != nil {
		t.Fatalf("NewTabNet: %v", err)
	}

	// Wrong input dim.
	badInput, err := tensor.New[float32]([]int{1, 5}, make([]float32, 5))
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	_, err = m.Forward(ctx, badInput)
	if err == nil {
		t.Fatal("expected error for wrong input dim, got nil")
	}
}

func TestTabNet_Predict(t *testing.T) {
	engine, ops := newTestEngine()

	config := defaultTabNetConfig()
	m, err := NewTabNet(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTabNet: %v", err)
	}

	features := make([]float64, config.InputDim)
	for i := range features {
		features[i] = float64(i) * 0.5
	}

	dir, conf, err := m.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if dir < Long || dir > Flat {
		t.Errorf("direction %d is not in [Long, Short, Flat]", dir)
	}
	if conf <= 0 || conf > 1 {
		t.Errorf("confidence %f is not in (0, 1]", conf)
	}
}

func TestTabNet_Predict_WrongFeatureCount(t *testing.T) {
	engine, ops := newTestEngine()

	m, err := NewTabNet(defaultTabNetConfig(), engine, ops)
	if err != nil {
		t.Fatalf("NewTabNet: %v", err)
	}

	_, _, err = m.Predict([]float64{1.0, 2.0})
	if err == nil {
		t.Fatal("expected error for wrong feature count, got nil")
	}
}

func TestTabNet_Sparsemax(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
		dim   int
	}{
		{
			name:  "uniform input",
			input: []float32{1.0, 1.0, 1.0, 1.0},
			dim:   4,
		},
		{
			name:  "one dominant",
			input: []float32{5.0, 0.1, 0.1, 0.1},
			dim:   4,
		},
		{
			name:  "two features",
			input: []float32{2.0, 1.0},
			dim:   2,
		},
		{
			name:  "negative values",
			input: []float32{-1.0, -2.0, 3.0, -0.5},
			dim:   4,
		},
		{
			name:  "large spread",
			input: []float32{10.0, 0.0, 0.0, 0.0, 0.0},
			dim:   5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, err := tensor.New[float32]([]int{1, tt.dim}, tt.input)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			result, err := SparsemaxDirect(input)
			if err != nil {
				t.Fatalf("sparsemax: %v", err)
			}

			data := result.Data()

			// All values must be non-negative.
			sum := float32(0)
			for i, v := range data {
				if v < 0 {
					t.Errorf("sparsemax[%d] = %f, expected >= 0", i, v)
				}
				sum += v
			}

			// Values must sum to 1 (within tolerance).
			if math.Abs(float64(sum)-1.0) > 1e-5 {
				t.Errorf("sparsemax sum = %f, expected 1.0", sum)
			}
		})
	}
}

func TestTabNet_Sparsemax_Sparsity(t *testing.T) {
	// With one dominant value and several small ones, sparsemax should
	// produce zeros for the small values (unlike softmax which always > 0).
	input, err := tensor.New[float32]([]int{1, 5}, []float32{10.0, 0.0, 0.0, 0.0, 0.0})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	result, err := SparsemaxDirect(input)
	if err != nil {
		t.Fatalf("sparsemax: %v", err)
	}

	data := result.Data()

	// The dominant feature should get all the weight.
	if data[0] < 0.99 {
		t.Errorf("dominant feature got %f, expected ~1.0", data[0])
	}

	// At least some values should be exactly zero.
	zeros := 0
	for _, v := range data {
		if v == 0 {
			zeros++
		}
	}
	if zeros == 0 {
		t.Error("sparsemax produced no exact zeros for highly skewed input")
	}
}

func TestTabNet_Sparsemax_BatchConsistency(t *testing.T) {
	// Two rows with same values should produce same sparsemax output.
	row := []float32{3.0, 1.0, 0.5, 2.0}
	batchInput := make([]float32, 8)
	copy(batchInput[:4], row)
	copy(batchInput[4:], row)

	input, err := tensor.New[float32]([]int{2, 4}, batchInput)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	result, err := SparsemaxDirect(input)
	if err != nil {
		t.Fatalf("sparsemax: %v", err)
	}

	data := result.Data()
	for i := 0; i < 4; i++ {
		if data[i] != data[4+i] {
			t.Errorf("batch inconsistency at feature %d: row0=%f, row1=%f", i, data[i], data[4+i])
		}
	}
}

func TestTabNet_FeatureImportance(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := defaultTabNetConfig()
	m, err := NewTabNet(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTabNet: %v", err)
	}

	// Before forward pass, feature importance should fail.
	_, err = m.FeatureImportance(ctx)
	if err == nil {
		t.Fatal("expected error before forward pass, got nil")
	}

	// Run forward pass.
	inputData := make([]float32, config.InputDim)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.3
	}
	input, err := tensor.New[float32]([]int{1, config.InputDim}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	_, err = m.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Feature importance should now work.
	importance, err := m.FeatureImportance(ctx)
	if err != nil {
		t.Fatalf("FeatureImportance: %v", err)
	}

	impShape := importance.Shape()
	if len(impShape) != 2 || impShape[0] != 1 || impShape[1] != config.InputDim {
		t.Errorf("expected importance shape [1, %d], got %v", config.InputDim, impShape)
	}

	// All importance values should be non-negative (sum of sparsemax outputs).
	impData := importance.Data()
	totalImportance := float32(0)
	for i, v := range impData {
		if v < 0 {
			t.Errorf("importance[%d] = %f, expected >= 0", i, v)
		}
		totalImportance += v
	}

	// Total importance should equal NSteps (each step's mask sums to 1).
	expectedTotal := float32(config.NSteps)
	if math.Abs(float64(totalImportance-expectedTotal)) > 1e-4 {
		t.Errorf("total importance = %f, expected %f (NSteps)", totalImportance, expectedTotal)
	}
}

func TestTabNet_Deterministic(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := TabNetConfig{
		InputDim:              4,
		OutputDim:             3,
		NSteps:                2,
		RelaxationFactor:      1.5,
		SparsityCoefficient:   0.001,
		FeatureTransformerDim: 8,
	}

	m, err := NewTabNet(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTabNet: %v", err)
	}

	inputData := []float32{1.0, -2.0, 3.0, -4.0}
	input, err := tensor.New[float32]([]int{1, 4}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	logits1, err := m.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward 1: %v", err)
	}
	data1 := make([]float32, len(logits1.Data()))
	copy(data1, logits1.Data())

	for i := 0; i < 5; i++ {
		logits, err := m.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward %d: %v", i+2, err)
		}
		for j, v := range logits.Data() {
			if v != data1[j] {
				t.Errorf("run %d: logit[%d] = %f, expected %f", i+2, j, v, data1[j])
			}
		}
	}
}
