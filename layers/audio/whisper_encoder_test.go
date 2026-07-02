package audio

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestWhisperEncoder_New(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	cfg := WhisperEncoderConfig{
		NumMels:    80,
		HiddenDim:  256,
		NumHeads:   4,
		NumLayers:  2,
		KernelSize: 3,
	}

	enc, err := NewWhisperEncoder[float32]("encoder", engine, ops, cfg)
	if err != nil {
		t.Fatalf("NewWhisperEncoder: %v", err)
	}
	if enc == nil {
		t.Fatal("encoder should not be nil")
	}
	if enc.OpType() != "WhisperEncoder" {
		t.Errorf("OpType = %q, want %q", enc.OpType(), "WhisperEncoder")
	}
}

func TestWhisperEncoder_NewValidation(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name string
		cfg  WhisperEncoderConfig
	}{
		{"zero mels", WhisperEncoderConfig{NumMels: 0, HiddenDim: 256, NumHeads: 4, NumLayers: 2, KernelSize: 3}},
		{"zero hidden", WhisperEncoderConfig{NumMels: 80, HiddenDim: 0, NumHeads: 4, NumLayers: 2, KernelSize: 3}},
		{"zero heads", WhisperEncoderConfig{NumMels: 80, HiddenDim: 256, NumHeads: 0, NumLayers: 2, KernelSize: 3}},
		{"zero layers", WhisperEncoderConfig{NumMels: 80, HiddenDim: 256, NumHeads: 4, NumLayers: 0, KernelSize: 3}},
		{"zero kernel", WhisperEncoderConfig{NumMels: 80, HiddenDim: 256, NumHeads: 4, NumLayers: 2, KernelSize: 0}},
		{"hidden not divisible by heads", WhisperEncoderConfig{NumMels: 80, HiddenDim: 255, NumHeads: 4, NumLayers: 2, KernelSize: 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewWhisperEncoder[float32]("enc", engine, ops, tt.cfg)
			if err == nil {
				t.Error("expected error for invalid config")
			}
		})
	}
}

func TestWhisperEncoder_ForwardShape(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	cfg := WhisperEncoderConfig{
		NumMels:    80,
		HiddenDim:  64,
		NumHeads:   4,
		NumLayers:  2,
		KernelSize: 3,
	}

	enc, err := NewWhisperEncoder[float32]("encoder", engine, ops, cfg)
	if err != nil {
		t.Fatalf("NewWhisperEncoder: %v", err)
	}

	// Synthetic mel input: [batch=1, num_mels=80, T_frames=100]
	batchSize := 1
	numMels := 80
	tFrames := 100
	inputData := make([]float32, batchSize*numMels*tFrames)
	for i := range inputData {
		inputData[i] = float32(i%17) * 0.01
	}
	input, err := tensor.New[float32]([]int{batchSize, numMels, tFrames}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	output, err := enc.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 2 {
		t.Fatalf("output should be 2D, got %dD: %v", len(shape), shape)
	}

	// After two Conv1D layers with stride=2, T_frames is downsampled:
	// After conv1: (100 + 2*1 - 3)/2 + 1 = 50
	// After conv2: (50 + 2*1 - 3)/2 + 1 = 25
	expectedFrames := 25
	if shape[0] != expectedFrames {
		t.Errorf("output frames = %d, want %d", shape[0], expectedFrames)
	}
	if shape[1] != cfg.HiddenDim {
		t.Errorf("output hidden_dim = %d, want %d", shape[1], cfg.HiddenDim)
	}
}

func TestWhisperEncoder_ForwardFiniteValues(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	cfg := WhisperEncoderConfig{
		NumMels:    16,
		HiddenDim:  32,
		NumHeads:   4,
		NumLayers:  1,
		KernelSize: 3,
	}

	enc, err := NewWhisperEncoder[float32]("encoder", engine, ops, cfg)
	if err != nil {
		t.Fatalf("NewWhisperEncoder: %v", err)
	}

	// Small input: [1, 16, 20]
	inputData := make([]float32, 1*16*20)
	for i := range inputData {
		inputData[i] = float32(i%7) * 0.1
	}
	input, err := tensor.New[float32]([]int{1, 16, 20}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	output, err := enc.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Check all values are finite
	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] is not finite: %v", i, v)
		}
	}
}

func TestWhisperEncoder_Parameters(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	cfg := WhisperEncoderConfig{
		NumMels:    80,
		HiddenDim:  64,
		NumHeads:   4,
		NumLayers:  2,
		KernelSize: 3,
	}

	enc, err := NewWhisperEncoder[float32]("encoder", engine, ops, cfg)
	if err != nil {
		t.Fatalf("NewWhisperEncoder: %v", err)
	}

	params := enc.Parameters()
	if len(params) == 0 {
		t.Error("encoder should have trainable parameters")
	}

	// Should have: conv1 (weight+bias) + conv2 (weight+bias) +
	// per layer: layernorm1 (gamma+beta) + QKV projections (3 weights) + out projection (weight) +
	//            layernorm2 (gamma+beta) + FFN (2 weights + 2 biases)
	// Plus positional encoding parameter
	// Exact count depends on implementation; just verify non-empty and all named
	for _, p := range params {
		if p.Name == "" {
			t.Error("parameter with empty name")
		}
	}
}

func TestWhisperEncoder_ForwardInputValidation(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	cfg := WhisperEncoderConfig{
		NumMels:    80,
		HiddenDim:  64,
		NumHeads:   4,
		NumLayers:  1,
		KernelSize: 3,
	}

	enc, err := NewWhisperEncoder[float32]("encoder", engine, ops, cfg)
	if err != nil {
		t.Fatalf("NewWhisperEncoder: %v", err)
	}

	t.Run("no inputs", func(t *testing.T) {
		_, err := enc.Forward(context.Background())
		if err == nil {
			t.Error("expected error for no inputs")
		}
	})

	t.Run("wrong dimensions", func(t *testing.T) {
		input, _ := tensor.New[float32]([]int{80, 100}, make([]float32, 8000))
		_, err := enc.Forward(context.Background(), input)
		if err == nil {
			t.Error("expected error for 2D input")
		}
	})

	t.Run("wrong channel count", func(t *testing.T) {
		input, _ := tensor.New[float32]([]int{1, 40, 100}, make([]float32, 4000))
		_, err := enc.Forward(context.Background(), input)
		if err == nil {
			t.Error("expected error for wrong channel count")
		}
	})
}

func TestWhisperEncoder_Attributes(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	cfg := WhisperEncoderConfig{
		NumMels:    80,
		HiddenDim:  256,
		NumHeads:   4,
		NumLayers:  2,
		KernelSize: 3,
	}

	enc, err := NewWhisperEncoder[float32]("encoder", engine, ops, cfg)
	if err != nil {
		t.Fatalf("NewWhisperEncoder: %v", err)
	}

	attrs := enc.Attributes()
	if attrs["num_mels"] != 80 {
		t.Errorf("num_mels = %v, want 80", attrs["num_mels"])
	}
	if attrs["hidden_dim"] != 256 {
		t.Errorf("hidden_dim = %v, want 256", attrs["hidden_dim"])
	}
	if attrs["num_heads"] != 4 {
		t.Errorf("num_heads = %v, want 4", attrs["num_heads"])
	}
	if attrs["num_layers"] != 2 {
		t.Errorf("num_layers = %v, want 2", attrs["num_layers"])
	}
}
