package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestTimeMixerBuildValidGraph(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TimeMixerConfig{
		InputLen:   24,
		OutputLen:  12,
		NumVars:    3,
		NumScales:  4,
		HiddenSize: 32,
		NumLayers:  2,
	}

	g, err := BuildTimeMixer[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTimeMixer: %v", err)
	}
	if g == nil {
		t.Fatal("BuildTimeMixer returned nil graph")
	}
}

func TestTimeMixerBuildDefaults(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TimeMixerConfig{
		InputLen:  24,
		OutputLen: 12,
		NumVars:   3,
	}

	g, err := BuildTimeMixer[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTimeMixer with defaults: %v", err)
	}
	if g == nil {
		t.Fatal("BuildTimeMixer returned nil graph")
	}
}

func TestTimeMixerForwardOutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TimeMixerConfig{
		InputLen:   16,
		OutputLen:  8,
		NumVars:    2,
		NumScales:  3,
		HiddenSize: 16,
		NumLayers:  2,
	}

	g, err := BuildTimeMixer[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTimeMixer: %v", err)
	}

	batch := 2
	data := make([]float32, batch*cfg.InputLen*cfg.NumVars)
	for i := range data {
		data[i] = float32(i%100) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, cfg.InputLen, cfg.NumVars}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()
	output, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.OutputLen, cfg.NumVars}
	if len(got) != len(want) {
		t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTimeMixerValidationErrors(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	tests := []struct {
		name string
		cfg  TimeMixerConfig
	}{
		{"zero input len", TimeMixerConfig{InputLen: 0, OutputLen: 12, NumVars: 3}},
		{"zero output len", TimeMixerConfig{InputLen: 24, OutputLen: 0, NumVars: 3}},
		{"zero num vars", TimeMixerConfig{InputLen: 24, OutputLen: 12, NumVars: 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildTimeMixer[float32](tensors, &tt.cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestTimeMixerSingleScale(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TimeMixerConfig{
		InputLen:   8,
		OutputLen:  4,
		NumVars:    1,
		NumScales:  1,
		HiddenSize: 8,
		NumLayers:  1,
	}

	g, err := BuildTimeMixer[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTimeMixer: %v", err)
	}

	batch := 1
	data := make([]float32, batch*cfg.InputLen*cfg.NumVars)
	for i := range data {
		data[i] = 0.5
	}
	input, err := tensor.New[float32]([]int{batch, cfg.InputLen, cfg.NumVars}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.OutputLen, cfg.NumVars}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTimeMixerBatchIndependence(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TimeMixerConfig{
		InputLen:   12,
		OutputLen:  6,
		NumVars:    2,
		NumScales:  2,
		HiddenSize: 8,
		NumLayers:  1,
	}

	g, err := BuildTimeMixer[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTimeMixer: %v", err)
	}

	// Run with batch=1.
	singleData := make([]float32, cfg.InputLen*cfg.NumVars)
	for i := range singleData {
		singleData[i] = float32(i) * 0.1
	}
	singleInput, err := tensor.New[float32]([]int{1, cfg.InputLen, cfg.NumVars}, singleData)
	if err != nil {
		t.Fatalf("create single input: %v", err)
	}
	singleOut, err := g.Forward(context.Background(), singleInput)
	if err != nil {
		t.Fatalf("Forward single: %v", err)
	}

	// Run with batch=2 (duplicate the same sample).
	doubleData := make([]float32, 2*cfg.InputLen*cfg.NumVars)
	copy(doubleData, singleData)
	copy(doubleData[cfg.InputLen*cfg.NumVars:], singleData)
	doubleInput, err := tensor.New[float32]([]int{2, cfg.InputLen, cfg.NumVars}, doubleData)
	if err != nil {
		t.Fatalf("create double input: %v", err)
	}
	doubleOut, err := g.Forward(context.Background(), doubleInput)
	if err != nil {
		t.Fatalf("Forward double: %v", err)
	}

	// Both batch elements should produce the same output as the single-batch run.
	sData := singleOut.Data()
	dData := doubleOut.Data()
	outputSize := cfg.OutputLen * cfg.NumVars
	for i := range outputSize {
		if abs32(sData[i]-dData[i]) > 1e-5 {
			t.Errorf("batch 0 output[%d]: single=%f, double=%f", i, sData[i], dData[i])
		}
		if abs32(sData[i]-dData[outputSize+i]) > 1e-5 {
			t.Errorf("batch 1 output[%d]: single=%f, double=%f", i, sData[i], dData[outputSize+i])
		}
	}
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
