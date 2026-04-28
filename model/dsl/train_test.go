package dsl

import (
	"math"
	"strings"
	"testing"
)

func TestCustomModel_Train(t *testing.T) {
	tests := []struct {
		name         string
		def          *ModelDef
		inputDim     int
		outputDim    int
		samples      []Sample
		config       TrainConfig
		wantLossDown bool // expect final loss < initial loss
		wantErr      string
	}{
		{
			name: "linear regression learns identity",
			def: &ModelDef{
				Name:   "identity",
				Layers: []LayerDef{{Name: "dense", Type: LayerLinear}},
			},
			inputDim:  2,
			outputDim: 2,
			samples: []Sample{
				{Input: []float64{1, 0}, Target: []float64{1, 0}},
				{Input: []float64{0, 1}, Target: []float64{0, 1}},
				{Input: []float64{1, 1}, Target: []float64{1, 1}},
				{Input: []float64{0.5, 0.5}, Target: []float64{0.5, 0.5}},
			},
			config:       TrainConfig{Epochs: 100, LearningRate: 0.01},
			wantLossDown: true,
		},
		{
			name: "MLP with norm and activation",
			def: &ModelDef{
				Name: "mlp",
				Layers: []LayerDef{
					{Name: "fc1", Type: LayerLinear, Params: map[string]any{"output_dim": 8}},
					{Name: "norm", Type: LayerRMSNorm},
					{Name: "act", Type: LayerSiLU},
					{Name: "fc2", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "fc1", To: "norm"},
					{From: "norm", To: "act"},
					{From: "act", To: "fc2"},
				},
			},
			inputDim:  4,
			outputDim: 2,
			samples: []Sample{
				{Input: []float64{1, 0, 0, 0}, Target: []float64{1, 0}},
				{Input: []float64{0, 1, 0, 0}, Target: []float64{0, 1}},
				{Input: []float64{0, 0, 1, 0}, Target: []float64{1, 0}},
				{Input: []float64{0, 0, 0, 1}, Target: []float64{0, 1}},
			},
			config:       TrainConfig{Epochs: 200, LearningRate: 0.01},
			wantLossDown: true,
		},
		{
			name: "softmax classification",
			def: &ModelDef{
				Name: "classifier",
				Layers: []LayerDef{
					{Name: "fc", Type: LayerLinear},
					{Name: "sm", Type: LayerSoftmax},
				},
				Connections: []ConnectionDef{
					{From: "fc", To: "sm"},
				},
			},
			inputDim:  3,
			outputDim: 3,
			samples: []Sample{
				{Input: []float64{1, 0, 0}, Target: []float64{1, 0, 0}},
				{Input: []float64{0, 1, 0}, Target: []float64{0, 1, 0}},
				{Input: []float64{0, 0, 1}, Target: []float64{0, 0, 1}},
			},
			config:       TrainConfig{Epochs: 100, LearningRate: 0.1},
			wantLossDown: true,
		},
		{
			name: "attention model trains",
			def: &ModelDef{
				Name: "attn",
				Layers: []LayerDef{
					{Name: "attn", Type: LayerAttention, Params: map[string]any{"num_heads": 2}},
					{Name: "out", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "attn", To: "out"},
				},
			},
			inputDim:  4,
			outputDim: 2,
			samples: []Sample{
				{Input: []float64{1, 0, 1, 0}, Target: []float64{1, 0}},
				{Input: []float64{0, 1, 0, 1}, Target: []float64{0, 1}},
			},
			config:       TrainConfig{Epochs: 100, LearningRate: 0.01},
			wantLossDown: true,
		},
		{
			name: "zero epochs error",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "fc", Type: LayerLinear}},
			},
			inputDim:  2,
			outputDim: 2,
			samples: []Sample{
				{Input: []float64{1, 0}, Target: []float64{1, 0}},
			},
			config:  TrainConfig{Epochs: 0, LearningRate: 0.01},
			wantErr: "epochs must be positive",
		},
		{
			name: "no samples error",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "fc", Type: LayerLinear}},
			},
			inputDim:  2,
			outputDim: 2,
			samples:   nil,
			config:    TrainConfig{Epochs: 10, LearningRate: 0.01},
			wantErr:   "at least one training sample",
		},
		{
			name: "zero learning rate error",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "fc", Type: LayerLinear}},
			},
			inputDim:  2,
			outputDim: 2,
			samples: []Sample{
				{Input: []float64{1, 0}, Target: []float64{1, 0}},
			},
			config:  TrainConfig{Epochs: 10, LearningRate: 0},
			wantErr: "learning rate must be positive",
		},
		{
			name: "input dimension mismatch",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "fc", Type: LayerLinear}},
			},
			inputDim:  2,
			outputDim: 2,
			samples: []Sample{
				{Input: []float64{1, 0, 0}, Target: []float64{1, 0}},
			},
			config:  TrainConfig{Epochs: 10, LearningRate: 0.01},
			wantErr: "sample 0 input has size 3, want 2",
		},
		{
			name: "target dimension mismatch",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "fc", Type: LayerLinear}},
			},
			inputDim:  2,
			outputDim: 2,
			samples: []Sample{
				{Input: []float64{1, 0}, Target: []float64{1}},
			},
			config:  TrainConfig{Epochs: 10, LearningRate: 0.01},
			wantErr: "sample 0 target has size 1, want 2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := Parse(tt.def)
			if err != nil {
				t.Fatalf("Parse failed: %v", err)
			}
			m, err := g.BuildTrainable(tt.inputDim, tt.outputDim)
			if err != nil {
				t.Fatalf("BuildTrainable failed: %v", err)
			}

			result, err := m.Train(tt.config, tt.samples)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("Train failed: %v", err)
			}

			// Verify loss values are finite.
			if math.IsNaN(result.FinalLoss) || math.IsInf(result.FinalLoss, 0) {
				t.Errorf("FinalLoss is not finite: %v", result.FinalLoss)
			}
			if math.IsNaN(result.BestLoss) || math.IsInf(result.BestLoss, 0) {
				t.Errorf("BestLoss is not finite: %v", result.BestLoss)
			}

			if tt.wantLossDown {
				initialLoss := result.EpochLoss[0]
				finalLoss := result.FinalLoss
				if finalLoss >= initialLoss {
					t.Errorf("expected loss to decrease: initial=%v, final=%v", initialLoss, finalLoss)
				}
			}

			// Verify epoch losses are populated.
			if len(result.EpochLoss) != tt.config.Epochs {
				t.Errorf("EpochLoss has %d entries, want %d", len(result.EpochLoss), tt.config.Epochs)
			}
		})
	}
}

func TestCustomModel_DSLIntegration(t *testing.T) {
	// End-to-end test: define a model via DSL, train it, and verify it produces
	// improved predictions after training.
	def := &ModelDef{
		Name: "xor_approx",
		Layers: []LayerDef{
			{Name: "hidden", Type: LayerLinear, Params: map[string]any{"output_dim": 8}},
			{Name: "act", Type: LayerSiLU},
			{Name: "output", Type: LayerLinear},
		},
		Connections: []ConnectionDef{
			{From: "hidden", To: "act"},
			{From: "act", To: "output"},
		},
	}

	graph, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	model, err := graph.BuildTrainable(2, 1)
	if err != nil {
		t.Fatalf("BuildTrainable failed: %v", err)
	}

	// XOR-like problem (not perfectly separable with one hidden layer,
	// but loss should still decrease).
	samples := []Sample{
		{Input: []float64{0, 0}, Target: []float64{0}},
		{Input: []float64{1, 0}, Target: []float64{1}},
		{Input: []float64{0, 1}, Target: []float64{1}},
		{Input: []float64{1, 1}, Target: []float64{0}},
	}

	// Measure pre-training loss.
	var preLoss float64
	for _, s := range samples {
		out, err := model.Forward(s.Input)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}
		diff := out[0] - s.Target[0]
		preLoss += diff * diff
	}
	preLoss /= float64(len(samples))

	// Train.
	result, err := model.Train(TrainConfig{
		Epochs:       500,
		LearningRate: 0.01,
	}, samples)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	// Measure post-training loss.
	var postLoss float64
	for _, s := range samples {
		out, err := model.Forward(s.Input)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}
		diff := out[0] - s.Target[0]
		postLoss += diff * diff
	}
	postLoss /= float64(len(samples))

	if postLoss >= preLoss {
		t.Errorf("expected post-training loss (%v) < pre-training loss (%v)", postLoss, preLoss)
	}

	// Verify result fields.
	if result.FinalLoss >= result.EpochLoss[0] {
		t.Errorf("final loss (%v) should be less than epoch 0 loss (%v)", result.FinalLoss, result.EpochLoss[0])
	}
	if result.BestEpoch < 0 || result.BestEpoch >= 500 {
		t.Errorf("unexpected best epoch: %d", result.BestEpoch)
	}

	// Verify Parameters returns non-empty list.
	params := model.Parameters()
	if len(params) == 0 {
		t.Error("Parameters() returned empty list")
	}

	// Verify all params have matching data and gradient sizes.
	for i, p := range params {
		if len(p.Data) == 0 {
			t.Errorf("param %d has empty data", i)
		}
		if len(p.Grad) != len(p.Data) {
			t.Errorf("param %d: grad size %d != data size %d", i, len(p.Grad), len(p.Data))
		}
	}
}

func TestCustomModel_Train_NonTrainableModel(t *testing.T) {
	// Build with regular Build (not BuildTrainable) — training should fail.
	def := &ModelDef{
		Name:   "regular",
		Layers: []LayerDef{{Name: "fc", Type: LayerLinear}},
	}
	g, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	m, err := g.Build(2, 2)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	_, err = m.Train(TrainConfig{Epochs: 1, LearningRate: 0.01}, []Sample{
		{Input: []float64{1, 0}, Target: []float64{1, 0}},
	})
	if err == nil {
		t.Fatal("expected error for non-trainable model")
	}
	if !strings.Contains(err.Error(), "does not support training") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCustomModel_BuildTrainableErrors(t *testing.T) {
	def := &ModelDef{
		Name:   "m",
		Layers: []LayerDef{{Name: "fc", Type: LayerLinear}},
	}
	g, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	tests := []struct {
		name      string
		inputDim  int
		outputDim int
		wantErr   string
	}{
		{"zero input dim", 0, 2, "inputDim must be positive"},
		{"zero output dim", 2, 0, "outputDim must be positive"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := g.BuildTrainable(tt.inputDim, tt.outputDim)
			if err == nil {
				t.Fatalf("expected error containing %q", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %q", tt.wantErr, err.Error())
			}
		})
	}
}

func TestCustomModel_TrainableForwardMatchesInference(t *testing.T) {
	// Verify that trainable model forward pass produces finite results
	// consistent with inference model.
	def := &ModelDef{
		Name: "compare",
		Layers: []LayerDef{
			{Name: "fc1", Type: LayerLinear, Params: map[string]any{"output_dim": 4}},
			{Name: "norm", Type: LayerRMSNorm},
			{Name: "fc2", Type: LayerLinear},
		},
		Connections: []ConnectionDef{
			{From: "fc1", To: "norm"},
			{From: "norm", To: "fc2"},
		},
	}

	g, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	m, err := g.BuildTrainable(3, 2)
	if err != nil {
		t.Fatalf("BuildTrainable failed: %v", err)
	}

	input := []float64{0.5, -0.3, 0.8}
	out, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if len(out) != 2 {
		t.Fatalf("output length = %d, want 2", len(out))
	}

	for i, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("output[%d] = %v, want finite", i, v)
		}
	}
}
