package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestMoiraiBuildValidGraph(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &MoiraiConfig{
		NumLayers:         2,
		HiddenDim:         32,
		NumHeads:          4,
		InputDim:          16,
		NumFreqEmbeddings: 10,
		Horizon:           12,
		NumVars:           3,
	}

	g, err := BuildMoirai[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildMoirai: %v", err)
	}
	if g == nil {
		t.Fatal("BuildMoirai returned nil graph")
	}
}

func TestMoiraiForwardOutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &MoiraiConfig{
		NumLayers:         2,
		HiddenDim:         32,
		NumHeads:          4,
		InputDim:          16,
		NumFreqEmbeddings: 10,
		Horizon:           12,
		NumVars:           3,
	}

	g, err := BuildMoirai[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildMoirai: %v", err)
	}

	batch := 2
	data := make([]float32, batch*cfg.NumVars*cfg.InputDim)
	for i := range data {
		data[i] = float32(i%100) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, cfg.NumVars, cfg.InputDim}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
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

func TestMoiraiSingleLayer(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &MoiraiConfig{
		NumLayers:         1,
		HiddenDim:         16,
		NumHeads:          2,
		InputDim:          8,
		NumFreqEmbeddings: 5,
		Horizon:           4,
		NumVars:           2,
	}

	g, err := BuildMoirai[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildMoirai: %v", err)
	}

	batch := 1
	data := make([]float32, batch*cfg.NumVars*cfg.InputDim)
	for i := range data {
		data[i] = 0.1
	}
	input, err := tensor.New[float32]([]int{batch, cfg.NumVars, cfg.InputDim}, data)
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

func TestMoiraiValidationErrors(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	base := MoiraiConfig{
		NumLayers:         2,
		HiddenDim:         32,
		NumHeads:          4,
		InputDim:          16,
		NumFreqEmbeddings: 10,
		Horizon:           12,
		NumVars:           3,
	}

	tests := []struct {
		name string
		cfg  MoiraiConfig
	}{
		{"zero NumLayers", func() MoiraiConfig { c := base; c.NumLayers = 0; return c }()},
		{"zero HiddenDim", func() MoiraiConfig { c := base; c.HiddenDim = 0; return c }()},
		{"zero NumHeads", func() MoiraiConfig { c := base; c.NumHeads = 0; return c }()},
		{"HiddenDim not divisible by NumHeads", func() MoiraiConfig { c := base; c.HiddenDim = 33; return c }()},
		{"zero InputDim", func() MoiraiConfig { c := base; c.InputDim = 0; return c }()},
		{"zero NumFreqEmbeddings", func() MoiraiConfig { c := base; c.NumFreqEmbeddings = 0; return c }()},
		{"zero Horizon", func() MoiraiConfig { c := base; c.Horizon = 0; return c }()},
		{"zero NumVars", func() MoiraiConfig { c := base; c.NumVars = 0; return c }()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildMoirai[float32](tensors, &tt.cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestMoiraiOutputNonZero(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &MoiraiConfig{
		NumLayers:         1,
		HiddenDim:         16,
		NumHeads:          2,
		InputDim:          8,
		NumFreqEmbeddings: 5,
		Horizon:           4,
		NumVars:           2,
	}

	g, err := BuildMoirai[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildMoirai: %v", err)
	}

	batch := 1
	data := make([]float32, batch*cfg.NumVars*cfg.InputDim)
	for i := range data {
		data[i] = float32(i+1) * 0.05
	}
	input, err := tensor.New[float32]([]int{batch, cfg.NumVars, cfg.InputDim}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outData := output.Data()
	allZero := true
	for _, v := range outData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("output is all zeros; expected non-zero values from forward pass")
	}
}
