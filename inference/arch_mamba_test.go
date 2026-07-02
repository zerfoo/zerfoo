package inference

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeMambaTestTensors creates a minimal set of Mamba-architecture tensors
// with the naming convention used by GGUF Mamba models.
func makeMambaTestTensors(mc MambaConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	dtRank := int(math.Ceil(float64(mc.DModel) / 16))

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
	logInit := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		cols := shape[len(shape)-1]
		for i := range data {
			data[i] = float32(math.Log(float64(i%cols + 1)))
		}
		t, _ := tensor.New(shape, data)
		return t
	}

	// Global tensors.
	tensors["token_embd.weight"] = fill([]int{mc.VocabSize, mc.DModel}, 0.02)
	tensors["output.weight"] = fill([]int{mc.VocabSize, mc.DModel}, 0.02)
	tensors["output_norm.weight"] = ones([]int{mc.DModel})

	// Per-layer tensors.
	for i := 0; i < mc.NumLayers; i++ {
		prefix := "mamba." + itoa(i) + "."
		tensors[prefix+"norm.weight"] = ones([]int{mc.DModel})
		tensors[prefix+"in_proj.weight"] = fill([]int{2 * mc.DInner, mc.DModel}, 0.02)
		tensors[prefix+"conv1d.weight"] = fill([]int{mc.DInner, 1, mc.DConv}, 0.02)
		tensors[prefix+"x_proj.weight"] = fill([]int{dtRank + 2*mc.DState, mc.DInner}, 0.02)
		tensors[prefix+"dt_proj.weight"] = fill([]int{mc.DInner, dtRank}, 0.02)
		tensors[prefix+"A_log"] = logInit([]int{mc.DInner, mc.DState})
		tensors[prefix+"D"] = ones([]int{mc.DInner})
		tensors[prefix+"out_proj.weight"] = fill([]int{mc.DModel, mc.DInner}, 0.02)
	}

	return tensors
}

func TestMamba3Build(t *testing.T) {
	mc := MambaConfig{
		NumLayers:  2,
		DModel:     16,
		DState:     4,
		DConv:      4,
		DInner:     32,
		VocabSize:  32,
		EOSTokenID: 0,
		RMSNormEps: 1e-5,
	}

	tensors := makeMambaTestTensors(mc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildMamba3(mc, tensors, engine)
	if err != nil {
		t.Fatalf("BuildMamba3: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestMamba3Build_ForwardNonNaN(t *testing.T) {
	mc := MambaConfig{
		NumLayers:  2,
		DModel:     16,
		DState:     4,
		DConv:      4,
		DInner:     32,
		VocabSize:  32,
		EOSTokenID: 0,
		RMSNormEps: 1e-5,
	}

	tensors := makeMambaTestTensors(mc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := BuildMamba3(mc, tensors, engine)
	if err != nil {
		t.Fatalf("BuildMamba3: %v", err)
	}

	// Forward pass with token IDs [1, 5, 10, 3].
	tokenIDs := []float32{1, 5, 10, 3}
	seqLen := len(tokenIDs)

	input, err := tensor.New([]int{1, seqLen}, tokenIDs)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != mc.VocabSize {
		t.Fatalf("unexpected output shape: %v, want [1, %d, %d]", shape, seqLen, mc.VocabSize)
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

func TestMamba3Config(t *testing.T) {
	tests := []struct {
		name string
		meta map[string]interface{}
		want MambaConfig
	}{
		{
			name: "standard fields",
			meta: map[string]interface{}{
				"d_model":           float64(768),
				"d_state":           float64(16),
				"d_inner":           float64(1536),
				"d_conv":            float64(4),
				"num_hidden_layers": float64(24),
				"vocab_size":        float64(50280),
				"eos_token_id":      float64(0),
			},
			want: MambaConfig{
				NumLayers:  24,
				DModel:     768,
				DState:     16,
				DConv:      4,
				DInner:     1536,
				VocabSize:  50280,
				EOSTokenID: 0,
				RMSNormEps: 1e-5,
			},
		},
		{
			name: "fallback field names",
			meta: map[string]interface{}{
				"hidden_size":       float64(256),
				"num_layers":        float64(4),
				"vocab_size":        float64(1000),
			},
			want: MambaConfig{
				NumLayers:  4,
				DModel:     256,
				DState:     16,  // default
				DConv:      4,   // default
				DInner:     512, // 2 * d_model default
				VocabSize:  1000,
				EOSTokenID: 0,
				RMSNormEps: 1e-5,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MambaConfigFromMetadata(tt.meta)
			if got.NumLayers != tt.want.NumLayers {
				t.Errorf("NumLayers = %d, want %d", got.NumLayers, tt.want.NumLayers)
			}
			if got.DModel != tt.want.DModel {
				t.Errorf("DModel = %d, want %d", got.DModel, tt.want.DModel)
			}
			if got.DState != tt.want.DState {
				t.Errorf("DState = %d, want %d", got.DState, tt.want.DState)
			}
			if got.DConv != tt.want.DConv {
				t.Errorf("DConv = %d, want %d", got.DConv, tt.want.DConv)
			}
			if got.DInner != tt.want.DInner {
				t.Errorf("DInner = %d, want %d", got.DInner, tt.want.DInner)
			}
			if got.VocabSize != tt.want.VocabSize {
				t.Errorf("VocabSize = %d, want %d", got.VocabSize, tt.want.VocabSize)
			}
			if got.EOSTokenID != tt.want.EOSTokenID {
				t.Errorf("EOSTokenID = %d, want %d", got.EOSTokenID, tt.want.EOSTokenID)
			}
		})
	}
}

func TestMamba3Build_MissingTensor(t *testing.T) {
	mc := MambaConfig{
		NumLayers:  1,
		DModel:     16,
		DState:     4,
		DConv:      4,
		DInner:     32,
		VocabSize:  32,
		EOSTokenID: 0,
		RMSNormEps: 1e-5,
	}

	// Empty tensors -- should fail with a clear error.
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := BuildMamba3(mc, tensors, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}
