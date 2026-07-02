package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestTiRexBuildValidGraph(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TiRexConfig{
		NumLayers:  4,
		InputDim:   8,
		HiddenDim:  16,
		Horizon:    12,
		NumVars:    3,
		BlockTypes: []string{"slstm", "mlstm", "slstm", "mlstm"},
	}

	g, err := BuildTiRex[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTiRex: %v", err)
	}
	if g == nil {
		t.Fatal("BuildTiRex returned nil graph")
	}
}

func TestTiRexBuildDefaultBlockTypes(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TiRexConfig{
		NumLayers: 3,
		InputDim:  8,
		HiddenDim: 16,
		Horizon:   12,
		NumVars:   3,
	}

	g, err := BuildTiRex[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTiRex with default block types: %v", err)
	}
	if g == nil {
		t.Fatal("BuildTiRex returned nil graph")
	}
}

func TestTiRexForwardOutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TiRexConfig{
		NumLayers:  2,
		InputDim:   4,
		HiddenDim:  8,
		Horizon:    6,
		NumVars:    2,
		BlockTypes: []string{"slstm", "mlstm"},
	}

	g, err := BuildTiRex[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTiRex: %v", err)
	}

	batch := 2
	seqLen := 10
	data := make([]float32, batch*seqLen*cfg.InputDim)
	for i := range data {
		data[i] = float32(i%100) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, cfg.InputDim}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()
	output, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.Horizon, cfg.NumVars}
	if len(got) != len(want) {
		t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTiRexValidationErrors(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	tests := []struct {
		name string
		cfg  TiRexConfig
	}{
		{"zero layers", TiRexConfig{NumLayers: 0, InputDim: 4, HiddenDim: 8, Horizon: 6, NumVars: 2}},
		{"zero input dim", TiRexConfig{NumLayers: 2, InputDim: 0, HiddenDim: 8, Horizon: 6, NumVars: 2}},
		{"zero hidden dim", TiRexConfig{NumLayers: 2, InputDim: 4, HiddenDim: 0, Horizon: 6, NumVars: 2}},
		{"zero horizon", TiRexConfig{NumLayers: 2, InputDim: 4, HiddenDim: 8, Horizon: 0, NumVars: 2}},
		{"zero num vars", TiRexConfig{NumLayers: 2, InputDim: 4, HiddenDim: 8, Horizon: 6, NumVars: 0}},
		{"wrong block types length", TiRexConfig{NumLayers: 2, InputDim: 4, HiddenDim: 8, Horizon: 6, NumVars: 2, BlockTypes: []string{"slstm"}}},
		{"invalid block type", TiRexConfig{NumLayers: 2, InputDim: 4, HiddenDim: 8, Horizon: 6, NumVars: 2, BlockTypes: []string{"slstm", "invalid"}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildTiRex[float32](tensors, &tt.cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestTiRexSingleBlock(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TiRexConfig{
		NumLayers:  1,
		InputDim:   4,
		HiddenDim:  8,
		Horizon:    3,
		NumVars:    1,
		BlockTypes: []string{"mlstm"},
	}

	g, err := BuildTiRex[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTiRex: %v", err)
	}

	batch := 1
	seqLen := 5
	data := make([]float32, batch*seqLen*cfg.InputDim)
	for i := range data {
		data[i] = 0.1
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, cfg.InputDim}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.Horizon, cfg.NumVars}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}
