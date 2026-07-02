package dsl

import (
	"math"
	"strings"
	"testing"
)

func TestDSL_Parse(t *testing.T) {
	tests := []struct {
		name    string
		def     *ModelDef
		wantErr string
	}{
		{
			name:    "nil definition",
			def:     nil,
			wantErr: "model definition is nil",
		},
		{
			name:    "empty name",
			def:     &ModelDef{Name: "", Layers: []LayerDef{{Name: "a", Type: LayerLinear}}},
			wantErr: "model name is required",
		},
		{
			name:    "no layers",
			def:     &ModelDef{Name: "m"},
			wantErr: "at least one layer is required",
		},
		{
			name: "empty layer name",
			def: &ModelDef{
				Name:   "m",
				Layers: []LayerDef{{Name: "", Type: LayerLinear}},
			},
			wantErr: "empty name",
		},
		{
			name: "duplicate layer name",
			def: &ModelDef{
				Name: "m",
				Layers: []LayerDef{
					{Name: "a", Type: LayerLinear},
					{Name: "a", Type: LayerSiLU},
				},
			},
			wantErr: "duplicate layer name",
		},
		{
			name: "unsupported layer type",
			def: &ModelDef{
				Name:   "m",
				Layers: []LayerDef{{Name: "a", Type: "unknown"}},
			},
			wantErr: "unsupported layer type",
		},
		{
			name: "connection references unknown from",
			def: &ModelDef{
				Name:        "m",
				Layers:      []LayerDef{{Name: "a", Type: LayerLinear}},
				Connections: []ConnectionDef{{From: "x", To: "a"}},
			},
			wantErr: "unknown layer",
		},
		{
			name: "connection references unknown to",
			def: &ModelDef{
				Name:        "m",
				Layers:      []LayerDef{{Name: "a", Type: LayerLinear}},
				Connections: []ConnectionDef{{From: "a", To: "y"}},
			},
			wantErr: "unknown layer",
		},
		{
			name: "cycle detected",
			def: &ModelDef{
				Name: "m",
				Layers: []LayerDef{
					{Name: "a", Type: LayerLinear},
					{Name: "b", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "a", To: "b"},
					{From: "b", To: "a"},
				},
			},
			wantErr: "cycle detected",
		},
		{
			name: "valid single layer",
			def: &ModelDef{
				Name:   "simple",
				Layers: []LayerDef{{Name: "dense", Type: LayerLinear}},
			},
		},
		{
			name: "valid chain",
			def: &ModelDef{
				Name: "chain",
				Layers: []LayerDef{
					{Name: "linear1", Type: LayerLinear},
					{Name: "norm", Type: LayerRMSNorm},
					{Name: "act", Type: LayerSiLU},
					{Name: "linear2", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "linear1", To: "norm"},
					{From: "norm", To: "act"},
					{From: "act", To: "linear2"},
				},
			},
		},
		{
			name: "all layer types",
			def: &ModelDef{
				Name: "all_types",
				Layers: []LayerDef{
					{Name: "linear", Type: LayerLinear},
					{Name: "rmsnorm", Type: LayerRMSNorm},
					{Name: "silu", Type: LayerSiLU},
					{Name: "softmax", Type: LayerSoftmax},
					{Name: "attn", Type: LayerAttention},
				},
				Connections: []ConnectionDef{
					{From: "linear", To: "rmsnorm"},
					{From: "rmsnorm", To: "silu"},
					{From: "silu", To: "softmax"},
					{From: "softmax", To: "attn"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := Parse(tt.def)
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
				t.Fatalf("unexpected error: %v", err)
			}
			if g.Name() != tt.def.Name {
				t.Errorf("Name() = %q, want %q", g.Name(), tt.def.Name)
			}
			if len(g.Order()) != len(tt.def.Layers) {
				t.Errorf("Order() has %d layers, want %d", len(g.Order()), len(tt.def.Layers))
			}
		})
	}
}

func TestDSL_ParseGraphProperties(t *testing.T) {
	def := &ModelDef{
		Name: "graph_props",
		Layers: []LayerDef{
			{Name: "input_linear", Type: LayerLinear},
			{Name: "norm", Type: LayerRMSNorm},
			{Name: "output_linear", Type: LayerLinear},
		},
		Connections: []ConnectionDef{
			{From: "input_linear", To: "norm"},
			{From: "norm", To: "output_linear"},
		},
	}

	g, err := Parse(def)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	inputs := g.Inputs()
	if len(inputs) != 1 || inputs[0] != "input_linear" {
		t.Errorf("Inputs() = %v, want [input_linear]", inputs)
	}

	outputs := g.Outputs()
	if len(outputs) != 1 || outputs[0] != "output_linear" {
		t.Errorf("Outputs() = %v, want [output_linear]", outputs)
	}
}

func TestDSL_BuildGraph(t *testing.T) {
	tests := []struct {
		name      string
		def       *ModelDef
		inputDim  int
		outputDim int
		wantErr   string
	}{
		{
			name: "valid build",
			def: &ModelDef{
				Name: "buildable",
				Layers: []LayerDef{
					{Name: "dense", Type: LayerLinear, Params: map[string]any{"output_dim": 8}},
					{Name: "norm", Type: LayerRMSNorm},
					{Name: "act", Type: LayerSiLU},
					{Name: "out", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "dense", To: "norm"},
					{From: "norm", To: "act"},
					{From: "act", To: "out"},
				},
			},
			inputDim:  4,
			outputDim: 2,
		},
		{
			name: "zero input dim",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "a", Type: LayerLinear}},
			},
			inputDim:  0,
			outputDim: 2,
			wantErr:   "inputDim must be positive",
		},
		{
			name: "zero output dim",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "a", Type: LayerLinear}},
			},
			inputDim:  4,
			outputDim: 0,
			wantErr:   "outputDim must be positive",
		},
		{
			name: "invalid output_dim param",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "a", Type: LayerLinear, Params: map[string]any{"output_dim": "abc"}}},
			},
			inputDim:  4,
			outputDim: 2,
			wantErr:   "invalid output_dim",
		},
		{
			name: "negative output_dim param",
			def: &ModelDef{
				Name:   "bad",
				Layers: []LayerDef{{Name: "a", Type: LayerLinear, Params: map[string]any{"output_dim": -1}}},
			},
			inputDim:  4,
			outputDim: 2,
			wantErr:   "output_dim must be positive",
		},
		{
			name: "attention with num_heads",
			def: &ModelDef{
				Name: "attn_model",
				Layers: []LayerDef{
					{Name: "attn", Type: LayerAttention, Params: map[string]any{"num_heads": 2}},
				},
			},
			inputDim:  4,
			outputDim: 4,
		},
		{
			name: "attention dim not divisible by heads",
			def: &ModelDef{
				Name: "bad_attn",
				Layers: []LayerDef{
					{Name: "attn", Type: LayerAttention, Params: map[string]any{"num_heads": 3}},
				},
			},
			inputDim:  4,
			outputDim: 4,
			wantErr:   "not divisible",
		},
		{
			name: "softmax single layer",
			def: &ModelDef{
				Name:   "softmax_only",
				Layers: []LayerDef{{Name: "sm", Type: LayerSoftmax}},
			},
			inputDim:  3,
			outputDim: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := Parse(tt.def)
			if err != nil {
				t.Fatalf("Parse failed: %v", err)
			}
			m, err := g.Build(tt.inputDim, tt.outputDim)
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
				t.Fatalf("Build failed: %v", err)
			}
			if m == nil {
				t.Fatal("Build returned nil model")
			}
		})
	}
}

func TestDSL_CustomModel(t *testing.T) {
	tests := []struct {
		name      string
		def       *ModelDef
		inputDim  int
		outputDim int
		inputLen  int
		wantLen   int
	}{
		{
			name: "linear chain",
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
			inputLen:  4,
			wantLen:   2,
		},
		{
			name: "single softmax",
			def: &ModelDef{
				Name:   "softmax_model",
				Layers: []LayerDef{{Name: "sm", Type: LayerSoftmax}},
			},
			inputDim:  3,
			outputDim: 3,
			inputLen:  3,
			wantLen:   3,
		},
		{
			name: "attention model",
			def: &ModelDef{
				Name: "attn_model",
				Layers: []LayerDef{
					{Name: "attn", Type: LayerAttention, Params: map[string]any{"num_heads": 2}},
					{Name: "norm", Type: LayerRMSNorm},
				},
				Connections: []ConnectionDef{
					{From: "attn", To: "norm"},
				},
			},
			inputDim:  4,
			outputDim: 4,
			inputLen:  4,
			wantLen:   4,
		},
		{
			name: "rmsnorm epsilon param",
			def: &ModelDef{
				Name: "norm_model",
				Layers: []LayerDef{
					{Name: "norm", Type: LayerRMSNorm, Params: map[string]any{"epsilon": 1e-5}},
				},
			},
			inputDim:  4,
			outputDim: 4,
			inputLen:  4,
			wantLen:   4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := Parse(tt.def)
			if err != nil {
				t.Fatalf("Parse failed: %v", err)
			}
			m, err := g.Build(tt.inputDim, tt.outputDim)
			if err != nil {
				t.Fatalf("Build failed: %v", err)
			}

			input := make([]float64, tt.inputLen)
			for i := range input {
				input[i] = float64(i+1) * 0.1
			}

			out, err := m.Forward(input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			if len(out) != tt.wantLen {
				t.Fatalf("output length = %d, want %d", len(out), tt.wantLen)
			}

			// Verify no NaN or Inf.
			for i, v := range out {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Errorf("output[%d] = %v, want finite", i, v)
				}
			}
		})
	}
}

func TestDSL_ForwardInputMismatch(t *testing.T) {
	def := &ModelDef{
		Name:   "simple",
		Layers: []LayerDef{{Name: "dense", Type: LayerLinear}},
	}
	g, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	m, err := g.Build(4, 2)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	_, err = m.Forward([]float64{1, 2, 3}) // wrong size
	if err == nil {
		t.Fatal("expected error for wrong input size")
	}
	if !strings.Contains(err.Error(), "expected input of size 4") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestDSL_SoftmaxSumsToOne(t *testing.T) {
	def := &ModelDef{
		Name:   "sm",
		Layers: []LayerDef{{Name: "softmax", Type: LayerSoftmax}},
	}
	g, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	m, err := g.Build(4, 4)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	out, err := m.Forward([]float64{1.0, 2.0, 3.0, 4.0})
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	var sum float64
	for _, v := range out {
		sum += v
		if v < 0 || v > 1 {
			t.Errorf("softmax output %v not in [0, 1]", v)
		}
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("softmax sum = %v, want 1.0", sum)
	}
}

func TestDSL_RMSNormNormalizes(t *testing.T) {
	def := &ModelDef{
		Name:   "norm",
		Layers: []LayerDef{{Name: "norm", Type: LayerRMSNorm}},
	}
	g, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}
	m, err := g.Build(4, 4)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	out, err := m.Forward([]float64{2.0, 2.0, 2.0, 2.0})
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// RMS of [2,2,2,2] = 2, so output should be ~[1,1,1,1].
	for i, v := range out {
		if math.Abs(v-1.0) > 1e-5 {
			t.Errorf("output[%d] = %v, want ~1.0", i, v)
		}
	}
}
