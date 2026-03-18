package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestPatchTSTForward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := PatchTSTConfig{
		PatchLen:  16,
		Stride:    8,
		NumLayers: 2,
		NumHeads:  4,
		DModel:    64,
		Horizon:   24,
		NumVars:   7,
	}

	g, err := BuildPatchTST[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildPatchTST: %v", err)
	}

	batch, seqLen := 2, 96
	data := make([]float32, batch*seqLen*cfg.NumVars)
	for i := range data {
		data[i] = float32(i%100) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, cfg.NumVars}, data)
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

func TestPatchTSTConfigValidation(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	tests := []struct {
		name string
		cfg  PatchTSTConfig
	}{
		{"zero PatchLen", PatchTSTConfig{PatchLen: 0, Stride: 8, NumLayers: 2, NumHeads: 4, DModel: 64, Horizon: 24, NumVars: 7}},
		{"zero Stride", PatchTSTConfig{PatchLen: 16, Stride: 0, NumLayers: 2, NumHeads: 4, DModel: 64, Horizon: 24, NumVars: 7}},
		{"zero NumLayers", PatchTSTConfig{PatchLen: 16, Stride: 8, NumLayers: 0, NumHeads: 4, DModel: 64, Horizon: 24, NumVars: 7}},
		{"zero NumHeads", PatchTSTConfig{PatchLen: 16, Stride: 8, NumLayers: 2, NumHeads: 0, DModel: 64, Horizon: 24, NumVars: 7}},
		{"zero DModel", PatchTSTConfig{PatchLen: 16, Stride: 8, NumLayers: 2, NumHeads: 4, DModel: 0, Horizon: 24, NumVars: 7}},
		{"DModel not divisible by NumHeads", PatchTSTConfig{PatchLen: 16, Stride: 8, NumLayers: 2, NumHeads: 4, DModel: 65, Horizon: 24, NumVars: 7}},
		{"zero Horizon", PatchTSTConfig{PatchLen: 16, Stride: 8, NumLayers: 2, NumHeads: 4, DModel: 64, Horizon: 0, NumVars: 7}},
		{"zero NumVars", PatchTSTConfig{PatchLen: 16, Stride: 8, NumLayers: 2, NumHeads: 4, DModel: 64, Horizon: 24, NumVars: 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildPatchTST[float32](tt.cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestPatchTSTSingleBatch(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := PatchTSTConfig{
		PatchLen:  8,
		Stride:    8,
		NumLayers: 1,
		NumHeads:  2,
		DModel:    16,
		Horizon:   12,
		NumVars:   3,
	}

	g, err := BuildPatchTST[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildPatchTST: %v", err)
	}

	batch, seqLen := 1, 32
	data := make([]float32, batch*seqLen*cfg.NumVars)
	for i := range data {
		data[i] = float32(i) * 0.001
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, cfg.NumVars}, data)
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
