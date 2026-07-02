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

// makeRWKVTestTensors creates a minimal set of RWKV-architecture tensors
// with the naming convention used by GGUF RWKV models.
func makeRWKVTestTensors(rc RWKVConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	fill := func(shape []int, scale float32) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = scale * float32(math.Sin(float64(i)*0.01+0.5))
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
		t, _ := tensor.New(shape, make([]float32, size))
		return t
	}
	decayInit := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			// log(-log(decay_rate)) — typical RWKV init: small positive values
			data[i] = float32(-math.Log(float64(i%rc.HeadSize+1) * 0.1))
		}
		t, _ := tensor.New(shape, data)
		return t
	}

	H := rc.HiddenSize
	// FFN hidden size: typically 4*H in RWKV, use 4*H for the test.
	ffnSize := 4 * H

	// Global tensors.
	tensors["token_embd.weight"] = fill([]int{rc.VocabSize, H}, 0.02)
	tensors["output.weight"] = fill([]int{rc.VocabSize, H}, 0.02)
	tensors["output_norm.weight"] = ones([]int{H})
	tensors["output_norm.bias"] = zeros([]int{H})

	// Layer 0 pre-LN (ln0).
	tensors["blocks.0.ln0.weight"] = ones([]int{H})
	tensors["blocks.0.ln0.bias"] = zeros([]int{H})

	for i := 0; i < rc.NumLayers; i++ {
		prefix := "blocks." + itoa(i) + "."
		p := prefix

		// LayerNorm weights.
		tensors[p+"ln1.weight"] = ones([]int{H})
		tensors[p+"ln1.bias"] = zeros([]int{H})
		tensors[p+"ln2.weight"] = ones([]int{H})
		tensors[p+"ln2.bias"] = zeros([]int{H})

		// Time mixing weights.
		att := p + "att."
		tensors[att+"time_mix_r"] = fill([]int{1, 1, H}, 0.5)
		tensors[att+"time_mix_k"] = fill([]int{1, 1, H}, 0.5)
		tensors[att+"time_mix_v"] = fill([]int{1, 1, H}, 0.5)
		tensors[att+"time_mix_g"] = fill([]int{1, 1, H}, 0.5)
		tensors[att+"time_decay"] = decayInit([]int{rc.NumHeads, rc.HeadSize})
		tensors[att+"time_faaaa"] = zeros([]int{rc.NumHeads, rc.HeadSize})
		tensors[att+"receptance.weight"] = fill([]int{H, H}, 0.02)
		tensors[att+"key.weight"] = fill([]int{H, H}, 0.02)
		tensors[att+"value.weight"] = fill([]int{H, H}, 0.02)
		tensors[att+"gate.weight"] = fill([]int{H, H}, 0.02)
		tensors[att+"output.weight"] = fill([]int{H, H}, 0.02)
		tensors[att+"ln_x.weight"] = ones([]int{H})
		tensors[att+"ln_x.bias"] = zeros([]int{H})

		// Channel mixing weights.
		ffn := p + "ffn."
		tensors[ffn+"time_mix_k"] = fill([]int{1, 1, H}, 0.5)
		tensors[ffn+"time_mix_r"] = fill([]int{1, 1, H}, 0.5)
		tensors[ffn+"key.weight"] = fill([]int{ffnSize, H}, 0.02)
		tensors[ffn+"value.weight"] = fill([]int{H, ffnSize}, 0.02)
		tensors[ffn+"receptance.weight"] = fill([]int{H, H}, 0.02)
	}

	return tensors
}

func TestRWKVBuild(t *testing.T) {
	rc := RWKVConfig{
		NumLayers:    2,
		HiddenSize:   16,
		VocabSize:    32,
		HeadSize:     8,
		NumHeads:     2,
		LayerNormEps: 1e-5,
	}

	tensors := makeRWKVTestTensors(rc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildRWKV(rc, tensors, engine)
	if err != nil {
		t.Fatalf("BuildRWKV: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestRWKVForward(t *testing.T) {
	tests := []struct {
		name    string
		numLayers int
		hidden  int
		vocab   int
		headSize int
		tokenIDs []float32
	}{
		{
			name:      "single token",
			numLayers: 2,
			hidden:    16,
			vocab:     32,
			headSize:  8,
			tokenIDs:  []float32{1},
		},
		{
			name:      "short sequence",
			numLayers: 2,
			hidden:    16,
			vocab:     32,
			headSize:  8,
			tokenIDs:  []float32{1, 5, 10, 3},
		},
		{
			name:      "deeper model",
			numLayers: 4,
			hidden:    16,
			vocab:     64,
			headSize:  8,
			tokenIDs:  []float32{0, 7, 15, 2, 31},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			numHeads := tt.hidden / tt.headSize
			rc := RWKVConfig{
				NumLayers:    tt.numLayers,
				HiddenSize:   tt.hidden,
				VocabSize:    tt.vocab,
				HeadSize:     tt.headSize,
				NumHeads:     numHeads,
				LayerNormEps: 1e-5,
			}

			tensors := makeRWKVTestTensors(rc)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := BuildRWKV(rc, tensors, engine)
			if err != nil {
				t.Fatalf("BuildRWKV: %v", err)
			}

			seqLen := len(tt.tokenIDs)
			input, err := tensor.New([]int{1, seqLen}, tt.tokenIDs)
			if err != nil {
				t.Fatalf("create input tensor: %v", err)
			}

			output, err := g.Forward(context.Background(), input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			outShape := output.Shape()
			if len(outShape) != 3 || outShape[0] != 1 || outShape[1] != seqLen || outShape[2] != tt.vocab {
				t.Fatalf("unexpected output shape: %v, want [1, %d, %d]", outShape, seqLen, tt.vocab)
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
		})
	}
}

func TestRWKVForward_MissingTensor(t *testing.T) {
	rc := RWKVConfig{
		NumLayers:    1,
		HiddenSize:   16,
		VocabSize:    32,
		HeadSize:     8,
		NumHeads:     2,
		LayerNormEps: 1e-5,
	}

	// Empty tensor map — should fail with clear error.
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := BuildRWKV(rc, tensors, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestRWKVConfigFromGGUF(t *testing.T) {
	tests := []struct {
		name   string
		hidden int
		wantH  int
		wantN  int
		wantHS int
	}{
		{
			name:   "standard 512 model",
			hidden: 512,
			wantH:  512,
			wantN:  8, // 512/64
			wantHS: 64,
		},
		{
			name:   "small 64 model",
			hidden: 64,
			wantH:  64,
			wantN:  1,
			wantHS: 64,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &gguf.ModelConfig{
				Architecture: "rwkv",
				HiddenSize:   tt.hidden,
				NumLayers:    4,
				VocabSize:    100,
			}
			rc := RWKVConfigFromGGUF(cfg)
			if rc.HiddenSize != tt.wantH {
				t.Errorf("HiddenSize = %d, want %d", rc.HiddenSize, tt.wantH)
			}
			if rc.NumHeads != tt.wantN {
				t.Errorf("NumHeads = %d, want %d", rc.NumHeads, tt.wantN)
			}
			if rc.HeadSize != tt.wantHS {
				t.Errorf("HeadSize = %d, want %d", rc.HeadSize, tt.wantHS)
			}
		})
	}
}

func TestRWKVRegistered(t *testing.T) {
	_, ok := GetArchitecture("rwkv")
	if !ok {
		t.Fatal("rwkv architecture not registered")
	}
}
