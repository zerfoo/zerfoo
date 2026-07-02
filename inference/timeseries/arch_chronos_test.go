package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestChronosBuildValidGraph(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &ChronosConfig{
		NumEncoderLayers: 2,
		NumDecoderLayers: 2,
		DModel:           32,
		NumHeads:         4,
		DFF:              64,
		VocabSize:        16,
		Horizon:          4,
	}

	g, err := BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}
	if g == nil {
		t.Fatal("BuildChronos returned nil graph")
	}
}

func TestChronosForwardOutputShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &ChronosConfig{
		NumEncoderLayers: 1,
		NumDecoderLayers: 1,
		DModel:           16,
		NumHeads:         2,
		DFF:              32,
		VocabSize:        8,
		Horizon:          3,
	}

	g, err := BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	batch := 2
	seqLen := 5
	// Create token ID input: values in [0, vocab_size).
	data := make([]float32, batch*seqLen)
	for i := range data {
		data[i] = float32(i % cfg.VocabSize)
	}
	input, err := tensor.New[float32]([]int{batch, seqLen}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()
	output, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.Horizon, cfg.VocabSize}
	if len(got) != len(want) {
		t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}

func TestChronosValidationErrors(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	tests := []struct {
		name string
		cfg  ChronosConfig
	}{
		{"zero encoder layers", ChronosConfig{NumEncoderLayers: 0, NumDecoderLayers: 2, DModel: 16, NumHeads: 2, DFF: 32, VocabSize: 8, Horizon: 3}},
		{"zero decoder layers", ChronosConfig{NumEncoderLayers: 2, NumDecoderLayers: 0, DModel: 16, NumHeads: 2, DFF: 32, VocabSize: 8, Horizon: 3}},
		{"zero d_model", ChronosConfig{NumEncoderLayers: 2, NumDecoderLayers: 2, DModel: 0, NumHeads: 2, DFF: 32, VocabSize: 8, Horizon: 3}},
		{"zero num_heads", ChronosConfig{NumEncoderLayers: 2, NumDecoderLayers: 2, DModel: 16, NumHeads: 0, DFF: 32, VocabSize: 8, Horizon: 3}},
		{"d_model not divisible by num_heads", ChronosConfig{NumEncoderLayers: 2, NumDecoderLayers: 2, DModel: 15, NumHeads: 4, DFF: 32, VocabSize: 8, Horizon: 3}},
		{"zero d_ff", ChronosConfig{NumEncoderLayers: 2, NumDecoderLayers: 2, DModel: 16, NumHeads: 2, DFF: 0, VocabSize: 8, Horizon: 3}},
		{"zero vocab_size", ChronosConfig{NumEncoderLayers: 2, NumDecoderLayers: 2, DModel: 16, NumHeads: 2, DFF: 32, VocabSize: 0, Horizon: 3}},
		{"zero horizon", ChronosConfig{NumEncoderLayers: 2, NumDecoderLayers: 2, DModel: 16, NumHeads: 2, DFF: 32, VocabSize: 8, Horizon: 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildChronos[float32](tensors, &tt.cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestChronosSingleLayerBatch1(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	cfg := &ChronosConfig{
		NumEncoderLayers: 1,
		NumDecoderLayers: 1,
		DModel:           8,
		NumHeads:         2,
		DFF:              16,
		VocabSize:        4,
		Horizon:          2,
	}

	g, err := BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	// Single batch, sequence length 3, token IDs in [0, 4).
	data := []float32{0, 1, 2}
	input, err := tensor.New[float32]([]int{1, 3}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{1, cfg.Horizon, cfg.VocabSize}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}
