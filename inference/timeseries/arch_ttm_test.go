package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestTTMBuildValidGraph(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TTMConfig{
		ContextLen:     64,
		ForecastLen:    24,
		NumChannels:    3,
		PatchLen:       8,
		DModel:         32,
		NumMixerLayers: 2,
		ChannelMixing:  false,
		Expansion:      2,
	}

	g, err := BuildTTM[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTTM: %v", err)
	}
	if g == nil {
		t.Fatal("BuildTTM returned nil graph")
	}
}

func TestTTMForwardOutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TTMConfig{
		ContextLen:     64,
		ForecastLen:    24,
		NumChannels:    3,
		PatchLen:       8,
		DModel:         32,
		NumMixerLayers: 2,
		ChannelMixing:  false,
		Expansion:      2,
	}

	g, err := BuildTTM[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTTM: %v", err)
	}

	batch := 2
	data := make([]float32, batch*cfg.ContextLen*cfg.NumChannels)
	for i := range data {
		data[i] = float32(i%100) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, cfg.ContextLen, cfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()
	output, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.ForecastLen, cfg.NumChannels}
	if len(got) != len(want) {
		t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTTMForwardSingleBatch(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TTMConfig{
		ContextLen:     32,
		ForecastLen:    8,
		NumChannels:    1,
		PatchLen:       8,
		DModel:         16,
		NumMixerLayers: 1,
		ChannelMixing:  false,
		Expansion:      2,
	}

	g, err := BuildTTM[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTTM: %v", err)
	}

	batch := 1
	data := make([]float32, batch*cfg.ContextLen*cfg.NumChannels)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	input, err := tensor.New[float32]([]int{batch, cfg.ContextLen, cfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.ForecastLen, cfg.NumChannels}
	if len(got) != len(want) {
		t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTTMForwardWithChannelMixing(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TTMConfig{
		ContextLen:     64,
		ForecastLen:    16,
		NumChannels:    4,
		PatchLen:       16,
		DModel:         32,
		NumMixerLayers: 2,
		ChannelMixing:  true,
		Expansion:      2,
	}

	g, err := BuildTTM[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTTM: %v", err)
	}

	batch := 2
	data := make([]float32, batch*cfg.ContextLen*cfg.NumChannels)
	for i := range data {
		data[i] = float32(i%50) * 0.02
	}
	input, err := tensor.New[float32]([]int{batch, cfg.ContextLen, cfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.ForecastLen, cfg.NumChannels}
	if len(got) != len(want) {
		t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTTMConfigValidation(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	tests := []struct {
		name string
		cfg  *TTMConfig
	}{
		{
			"zero ContextLen",
			&TTMConfig{ContextLen: 0, ForecastLen: 24, NumChannels: 3, PatchLen: 8, DModel: 32, NumMixerLayers: 2, Expansion: 2},
		},
		{
			"zero ForecastLen",
			&TTMConfig{ContextLen: 64, ForecastLen: 0, NumChannels: 3, PatchLen: 8, DModel: 32, NumMixerLayers: 2, Expansion: 2},
		},
		{
			"zero NumChannels",
			&TTMConfig{ContextLen: 64, ForecastLen: 24, NumChannels: 0, PatchLen: 8, DModel: 32, NumMixerLayers: 2, Expansion: 2},
		},
		{
			"zero PatchLen",
			&TTMConfig{ContextLen: 64, ForecastLen: 24, NumChannels: 3, PatchLen: 0, DModel: 32, NumMixerLayers: 2, Expansion: 2},
		},
		{
			"zero DModel",
			&TTMConfig{ContextLen: 64, ForecastLen: 24, NumChannels: 3, PatchLen: 8, DModel: 0, NumMixerLayers: 2, Expansion: 2},
		},
		{
			"zero NumMixerLayers",
			&TTMConfig{ContextLen: 64, ForecastLen: 24, NumChannels: 3, PatchLen: 8, DModel: 32, NumMixerLayers: 0, Expansion: 2},
		},
		{
			"zero Expansion",
			&TTMConfig{ContextLen: 64, ForecastLen: 24, NumChannels: 3, PatchLen: 8, DModel: 32, NumMixerLayers: 2, Expansion: 0},
		},
		{
			"ContextLen not divisible by PatchLen",
			&TTMConfig{ContextLen: 65, ForecastLen: 24, NumChannels: 3, PatchLen: 8, DModel: 32, NumMixerLayers: 2, Expansion: 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildTTM[float32](tensors, tt.cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestTTMWithGGUFWeights(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := &TTMConfig{
		ContextLen:     32,
		ForecastLen:    8,
		NumChannels:    2,
		PatchLen:       8,
		DModel:         16,
		NumMixerLayers: 1,
		ChannelMixing:  false,
		Expansion:      2,
	}

	// Create synthetic GGUF-like weights.
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	// Embedding weight: [patch_len, d_model] = [8, 16]
	embData := make([]float32, 8*16)
	for i := range embData {
		embData[i] = float32(i%7-3) * 0.1
	}
	embTensor, err := tensor.New[float32]([]int{8, 16}, embData)
	if err != nil {
		t.Fatalf("create embedding tensor: %v", err)
	}
	tensors["embedding.weight"] = embTensor

	// Head weight: [d_model, forecast_len] = [16, 8]
	headData := make([]float32, 16*8)
	for i := range headData {
		headData[i] = float32(i%5-2) * 0.05
	}
	headTensor, err := tensor.New[float32]([]int{16, 8}, headData)
	if err != nil {
		t.Fatalf("create head tensor: %v", err)
	}
	tensors["head.linear.weight"] = headTensor

	g, err := BuildTTM[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTTM: %v", err)
	}

	batch := 1
	data := make([]float32, batch*cfg.ContextLen*cfg.NumChannels)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, cfg.ContextLen, cfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.ForecastLen, cfg.NumChannels}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTTMOutputFinite(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TTMConfig{
		ContextLen:     32,
		ForecastLen:    8,
		NumChannels:    2,
		PatchLen:       8,
		DModel:         16,
		NumMixerLayers: 1,
		ChannelMixing:  false,
		Expansion:      2,
	}

	g, err := BuildTTM[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildTTM: %v", err)
	}

	batch := 1
	data := make([]float32, batch*cfg.ContextLen*cfg.NumChannels)
	for i := range data {
		data[i] = float32(i%10) + 1.0 // non-zero varying data
	}
	input, err := tensor.New[float32]([]int{batch, cfg.ContextLen, cfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outData := output.Data()
	for i, v := range outData {
		if v != v { // NaN check
			t.Errorf("output[%d] is NaN", i)
		}
		if v > 1e10 || v < -1e10 {
			t.Errorf("output[%d] = %f, looks like overflow", i, v)
		}
	}
}

// BenchmarkTTMForward benchmarks TTM inference matching the GTS-T3 Python config.
func BenchmarkTTMForward(b *testing.B) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &TTMConfig{
		ContextLen:     32,
		ForecastLen:    8,
		NumChannels:    1,
		PatchLen:       8,
		DModel:         64,
		NumMixerLayers: 2,
		ChannelMixing:  false,
		Expansion:      2,
	}
	cfg.NumPatches = cfg.ContextLen / cfg.PatchLen

	g, err := BuildTTM[float32](tensors, cfg, engine)
	if err != nil {
		b.Fatalf("BuildTTM: %v", err)
	}

	data := make([]float32, cfg.ContextLen)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	input, _ := tensor.New[float32]([]int{1, cfg.ContextLen, 1}, data)

	ctx := context.Background()
	// Warmup.
	for range 5 {
		g.Forward(ctx, input)
	}

	b.ResetTimer()
	for b.Loop() {
		g.Forward(ctx, input)
	}
}
