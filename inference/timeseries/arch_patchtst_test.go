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

// TestPatchTSTChannelIndependence verifies that each variable's output depends
// only on its own input, not on other variables. This confirms the projection
// head uses channel-independent projection with no cross-variable mixing.
func TestPatchTSTChannelIndependence(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := PatchTSTConfig{
		PatchLen:  8,
		Stride:    8,
		NumLayers: 1,
		NumHeads:  2,
		DModel:    16,
		Horizon:   4,
		NumVars:   3,
	}

	batch, seqLen := 1, 32

	// Create two inputs that differ only in variable 0.
	dataA := make([]float32, batch*seqLen*cfg.NumVars)
	dataB := make([]float32, batch*seqLen*cfg.NumVars)
	for i := range dataA {
		dataA[i] = float32(i%50) * 0.01
		dataB[i] = dataA[i]
	}
	// Perturb only variable 0 in input B.
	for s := 0; s < seqLen; s++ {
		dataB[s*cfg.NumVars+0] += 1.0
	}

	inputA, err := tensor.New[float32]([]int{batch, seqLen, cfg.NumVars}, dataA)
	if err != nil {
		t.Fatalf("create inputA: %v", err)
	}
	inputB, err := tensor.New[float32]([]int{batch, seqLen, cfg.NumVars}, dataB)
	if err != nil {
		t.Fatalf("create inputB: %v", err)
	}

	nodeA, err := newPatchTSTNode[float32](cfg, engine, numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("newPatchTSTNode: %v", err)
	}

	ctx := context.Background()
	outA, err := nodeA.Forward(ctx, inputA)
	if err != nil {
		t.Fatalf("Forward A: %v", err)
	}
	outB, err := nodeA.Forward(ctx, inputB)
	if err != nil {
		t.Fatalf("Forward B: %v", err)
	}

	// Verify output shape is [batch, horizon, numVars].
	wantShape := []int{batch, cfg.Horizon, cfg.NumVars}
	for i, s := range outA.Shape() {
		if s != wantShape[i] {
			t.Fatalf("output shape[%d]: got %d, want %d", i, s, wantShape[i])
		}
	}

	dA := outA.Data()
	dB := outB.Data()

	// Variable 0 should differ (we perturbed its input).
	var var0Diff float32
	for h := 0; h < cfg.Horizon; h++ {
		idx := h*cfg.NumVars + 0
		diff := dA[idx] - dB[idx]
		if diff < 0 {
			diff = -diff
		}
		var0Diff += diff
	}
	if var0Diff < 1e-6 {
		t.Error("variable 0 output should differ between A and B, but it did not")
	}

	// Variables 1 and 2 should be identical (their inputs are the same).
	// Channel-independent projection means no cross-variable mixing.
	for v := 1; v < cfg.NumVars; v++ {
		for h := 0; h < cfg.Horizon; h++ {
			idx := h*cfg.NumVars + v
			if dA[idx] != dB[idx] {
				t.Errorf("variable %d horizon %d: got A=%.8f B=%.8f, want identical (no cross-variable mixing)", v, h, dA[idx], dB[idx])
			}
		}
	}
}
