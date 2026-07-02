package inference

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// makeWhisperTestTensors creates a minimal set of Whisper-architecture tensors
// with the naming convention used by Whisper GGUF models.
func makeWhisperTestTensors(wc WhisperConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	ffnDim := wc.HiddenDim * 4

	fill := func(shape []int, scale float32) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = scale * float32(math.Sin(float64(i)*0.01))
		}
		t, _ := tensor.New(shape, data)
		return t
	}
	ones := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 1.0
		}
		t, _ := tensor.New(shape, data)
		return t
	}
	zeros := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		t, _ := tensor.New(shape, data)
		return t
	}

	// Conv frontend.
	tensors["encoder.conv1.weight"] = fill([]int{wc.HiddenDim, wc.NumMels, wc.KernelSize}, 0.02)
	tensors["encoder.conv1.bias"] = zeros([]int{wc.HiddenDim})
	tensors["encoder.conv2.weight"] = fill([]int{wc.HiddenDim, wc.HiddenDim, wc.KernelSize}, 0.02)
	tensors["encoder.conv2.bias"] = zeros([]int{wc.HiddenDim})

	// Per-block tensors.
	for i := 0; i < wc.NumLayers; i++ {
		prefix := "encoder.blocks." + itoa(i) + "."

		// Attention LN.
		tensors[prefix+"attn_ln.weight"] = ones([]int{wc.HiddenDim})
		tensors[prefix+"attn_ln.bias"] = zeros([]int{wc.HiddenDim})

		// Attention projections.
		tensors[prefix+"attn.query.weight"] = fill([]int{wc.HiddenDim, wc.HiddenDim}, 0.02)
		tensors[prefix+"attn.key.weight"] = fill([]int{wc.HiddenDim, wc.HiddenDim}, 0.02)
		tensors[prefix+"attn.value.weight"] = fill([]int{wc.HiddenDim, wc.HiddenDim}, 0.02)
		tensors[prefix+"attn.out.weight"] = fill([]int{wc.HiddenDim, wc.HiddenDim}, 0.02)

		// MLP LN.
		tensors[prefix+"mlp_ln.weight"] = ones([]int{wc.HiddenDim})
		tensors[prefix+"mlp_ln.bias"] = zeros([]int{wc.HiddenDim})

		// FFN.
		tensors[prefix+"mlp.0.weight"] = fill([]int{ffnDim, wc.HiddenDim}, 0.02)
		tensors[prefix+"mlp.0.bias"] = zeros([]int{ffnDim})
		tensors[prefix+"mlp.2.weight"] = fill([]int{wc.HiddenDim, ffnDim}, 0.02)
		tensors[prefix+"mlp.2.bias"] = zeros([]int{wc.HiddenDim})
	}

	// Post layer norm.
	tensors["encoder.ln_post.weight"] = ones([]int{wc.HiddenDim})
	tensors["encoder.ln_post.bias"] = zeros([]int{wc.HiddenDim})

	return tensors
}

func TestWhisperLoad(t *testing.T) {
	wc := WhisperConfig{
		NumMels:    16,
		HiddenDim:  32,
		NumHeads:   4,
		NumLayers:  2,
		KernelSize: 3,
	}

	tensors := makeWhisperTestTensors(wc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildWhisperEncoder(wc, tensors, engine)
	if err != nil {
		t.Fatalf("BuildWhisperEncoder: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	// Whisper encoder has no text embedding.
	if emb != nil {
		t.Fatal("expected nil embedding for Whisper encoder")
	}
}

func TestWhisperLoad_ForwardNonNaN(t *testing.T) {
	wc := WhisperConfig{
		NumMels:    16,
		HiddenDim:  32,
		NumHeads:   4,
		NumLayers:  2,
		KernelSize: 3,
	}

	tensors := makeWhisperTestTensors(wc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := BuildWhisperEncoder(wc, tensors, engine)
	if err != nil {
		t.Fatalf("BuildWhisperEncoder: %v", err)
	}

	// Create a synthetic mel spectrogram input: [batch=1, num_mels=16, T_frames=32]
	tFrames := 32
	inputSize := 1 * wc.NumMels * tFrames
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = float32(math.Sin(float64(i) * 0.1))
	}
	input, err := tensor.New([]int{1, wc.NumMels, tFrames}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 2 {
		t.Fatalf("expected 2D output [T_downsampled, hidden_dim], got shape %v", shape)
	}
	if shape[1] != wc.HiddenDim {
		t.Fatalf("output hidden_dim = %d, want %d", shape[1], wc.HiddenDim)
	}

	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at index %d", i)
		}
	}
}

func TestWhisperConfigFromGGUF(t *testing.T) {
	tests := []struct {
		name string
		cfg  *gguf.ModelConfig
		want WhisperConfig
	}{
		{
			name: "standard whisper-base",
			cfg: &gguf.ModelConfig{
				Architecture: "whisper",
				HiddenSize:   512,
				NumHeads:     8,
				NumLayers:    6,
			},
			want: WhisperConfig{
				NumMels:    80,
				HiddenDim:  512,
				NumHeads:   8,
				NumLayers:  6,
				KernelSize: 3,
			},
		},
		{
			name: "default num_heads",
			cfg: &gguf.ModelConfig{
				Architecture: "whisper",
				HiddenSize:   384,
				NumLayers:    4,
			},
			want: WhisperConfig{
				NumMels:    80,
				HiddenDim:  384,
				NumHeads:   6,
				NumLayers:  4,
				KernelSize: 3,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := WhisperConfigFromGGUF(tt.cfg)
			if got.NumMels != tt.want.NumMels {
				t.Errorf("NumMels = %d, want %d", got.NumMels, tt.want.NumMels)
			}
			if got.HiddenDim != tt.want.HiddenDim {
				t.Errorf("HiddenDim = %d, want %d", got.HiddenDim, tt.want.HiddenDim)
			}
			if got.NumHeads != tt.want.NumHeads {
				t.Errorf("NumHeads = %d, want %d", got.NumHeads, tt.want.NumHeads)
			}
			if got.NumLayers != tt.want.NumLayers {
				t.Errorf("NumLayers = %d, want %d", got.NumLayers, tt.want.NumLayers)
			}
			if got.KernelSize != tt.want.KernelSize {
				t.Errorf("KernelSize = %d, want %d", got.KernelSize, tt.want.KernelSize)
			}
		})
	}
}

func TestWhisperLoad_MissingTensor(t *testing.T) {
	wc := WhisperConfig{
		NumMels:    16,
		HiddenDim:  32,
		NumHeads:   4,
		NumLayers:  1,
		KernelSize: 3,
	}

	// Empty tensors -- should fail with a clear error.
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := BuildWhisperEncoder(wc, tensors, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestWhisperBuildArchGraph(t *testing.T) {
	wc := WhisperConfig{
		NumMels:    16,
		HiddenDim:  32,
		NumHeads:   4,
		NumLayers:  1,
		KernelSize: 3,
	}

	tensors := makeWhisperTestTensors(wc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := &gguf.ModelConfig{
		Architecture: "whisper",
		HiddenSize:   32,
		NumHeads:     4,
		NumLayers:    1,
	}

	g, emb, err := buildArchGraph("whisper", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(whisper): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb != nil {
		t.Fatal("expected nil embedding for Whisper encoder")
	}
}
