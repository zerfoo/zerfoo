package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestTFTForward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := TFTConfig{
		NumStaticFeatures:   4,
		NumTemporalFeatures: 8,
		HiddenDim:           32,
		NumHeads:            4,
		NumLSTMLayers:       2,
		HorizonLen:          10,
		Quantiles:           []float32{0.1, 0.5, 0.9},
	}

	g, err := BuildTFT[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildTFT: %v", err)
	}

	batch, seqLen := 2, 20
	temporalData := make([]float32, batch*seqLen*cfg.NumTemporalFeatures)
	for i := range temporalData {
		temporalData[i] = float32(i%100) * 0.01
	}
	temporal, err := tensor.New[float32]([]int{batch, seqLen, cfg.NumTemporalFeatures}, temporalData)
	if err != nil {
		t.Fatalf("create temporal input: %v", err)
	}

	staticData := make([]float32, batch*cfg.NumStaticFeatures)
	for i := range staticData {
		staticData[i] = float32(i%10) * 0.1
	}
	static, err := tensor.New[float32]([]int{batch, cfg.NumStaticFeatures}, staticData)
	if err != nil {
		t.Fatalf("create static input: %v", err)
	}

	ctx := context.Background()
	output, err := g.Forward(ctx, temporal, static)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.HorizonLen, len(cfg.Quantiles)}
	if len(got) != len(want) {
		t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}

	// Verify output contains finite values.
	outData := output.Data()
	for i, v := range outData {
		if v != v { // NaN check
			t.Errorf("output[%d] is NaN", i)
		}
	}
}

func TestTFTConfig(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	validQuantiles := []float32{0.1, 0.5, 0.9}

	tests := []struct {
		name string
		cfg  TFTConfig
	}{
		{
			"zero NumStaticFeatures",
			TFTConfig{NumStaticFeatures: 0, NumTemporalFeatures: 8, HiddenDim: 32, NumHeads: 4, NumLSTMLayers: 2, HorizonLen: 10, Quantiles: validQuantiles},
		},
		{
			"zero NumTemporalFeatures",
			TFTConfig{NumStaticFeatures: 4, NumTemporalFeatures: 0, HiddenDim: 32, NumHeads: 4, NumLSTMLayers: 2, HorizonLen: 10, Quantiles: validQuantiles},
		},
		{
			"zero HiddenDim",
			TFTConfig{NumStaticFeatures: 4, NumTemporalFeatures: 8, HiddenDim: 0, NumHeads: 4, NumLSTMLayers: 2, HorizonLen: 10, Quantiles: validQuantiles},
		},
		{
			"zero NumHeads",
			TFTConfig{NumStaticFeatures: 4, NumTemporalFeatures: 8, HiddenDim: 32, NumHeads: 0, NumLSTMLayers: 2, HorizonLen: 10, Quantiles: validQuantiles},
		},
		{
			"HiddenDim not divisible by NumHeads",
			TFTConfig{NumStaticFeatures: 4, NumTemporalFeatures: 8, HiddenDim: 33, NumHeads: 4, NumLSTMLayers: 2, HorizonLen: 10, Quantiles: validQuantiles},
		},
		{
			"zero NumLSTMLayers",
			TFTConfig{NumStaticFeatures: 4, NumTemporalFeatures: 8, HiddenDim: 32, NumHeads: 4, NumLSTMLayers: 0, HorizonLen: 10, Quantiles: validQuantiles},
		},
		{
			"zero HorizonLen",
			TFTConfig{NumStaticFeatures: 4, NumTemporalFeatures: 8, HiddenDim: 32, NumHeads: 4, NumLSTMLayers: 2, HorizonLen: 0, Quantiles: validQuantiles},
		},
		{
			"empty Quantiles",
			TFTConfig{NumStaticFeatures: 4, NumTemporalFeatures: 8, HiddenDim: 32, NumHeads: 4, NumLSTMLayers: 2, HorizonLen: 10, Quantiles: nil},
		},
		{
			"negative NumStaticFeatures",
			TFTConfig{NumStaticFeatures: -1, NumTemporalFeatures: 8, HiddenDim: 32, NumHeads: 4, NumLSTMLayers: 2, HorizonLen: 10, Quantiles: validQuantiles},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildTFT[float32](tt.cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}
