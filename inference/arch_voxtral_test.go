package inference

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// testVoxtralConfig returns a minimal Voxtral configuration for testing.
func testVoxtralConfig() (*gguf.ModelConfig, VoxtralConfig) {
	cfg := &gguf.ModelConfig{
		Architecture:     "voxtral",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	vc := VoxtralConfig{
		AudioHiddenDim:        32,
		AudioNumLayers:        1,
		AudioNumHeads:         4,
		AudioNumMels:          16,
		AudioIntermediateSize: 64,
		AudioKernelSize:       3,
		StackFactor:           4,
	}
	return cfg, vc
}

// makeVoxtralTestTensors creates a minimal set of Voxtral-architecture tensors
// for testing, including audio encoder, adapter, and text decoder weights.
func makeVoxtralTestTensors(cfg *gguf.ModelConfig, vc VoxtralConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

	aHidden := vc.AudioHiddenDim
	aInter := vc.AudioIntermediateSize
	aMels := vc.AudioNumMels
	kernel := vc.AudioKernelSize
	stackDim := vc.StackFactor * aHidden

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

	// --- Audio encoder tensors (Voxtral mmproj naming) ---
	tensors["a.conv1d.0.weight"] = fill([]int{aHidden, aMels, kernel}, 0.02)
	tensors["a.conv1d.0.bias"] = zeros([]int{aHidden})
	tensors["a.conv1d.1.weight"] = fill([]int{aHidden, aHidden, kernel}, 0.02)
	tensors["a.conv1d.1.bias"] = zeros([]int{aHidden})

	for i := 0; i < vc.AudioNumLayers; i++ {
		prefix := fmt.Sprintf("a.blk.%d.", i)
		tensors[prefix+"ln1.weight"] = ones([]int{aHidden})
		tensors[prefix+"ln1.bias"] = zeros([]int{aHidden})
		tensors[prefix+"attn_q.weight"] = fill([]int{aHidden, aHidden}, 0.02)
		tensors[prefix+"attn_k.weight"] = fill([]int{aHidden, aHidden}, 0.02)
		tensors[prefix+"attn_v.weight"] = fill([]int{aHidden, aHidden}, 0.02)
		tensors[prefix+"attn_o.weight"] = fill([]int{aHidden, aHidden}, 0.02)
		tensors[prefix+"attn_q.bias"] = zeros([]int{aHidden})
		tensors[prefix+"attn_k.bias"] = zeros([]int{aHidden})
		tensors[prefix+"attn_v.bias"] = zeros([]int{aHidden})
		tensors[prefix+"ln2.weight"] = ones([]int{aHidden})
		tensors[prefix+"ln2.bias"] = zeros([]int{aHidden})
		tensors[prefix+"ffn_up.weight"] = fill([]int{aInter, aHidden}, 0.02)
		tensors[prefix+"ffn_up.bias"] = zeros([]int{aInter})
		tensors[prefix+"ffn_down.weight"] = fill([]int{aHidden, aInter}, 0.02)
		tensors[prefix+"ffn_down.bias"] = zeros([]int{aHidden})
	}

	tensors["a.post_ln.weight"] = ones([]int{aHidden})
	tensors["a.post_ln.bias"] = zeros([]int{aHidden})

	// --- Adapter MLP tensors ---
	tensors["mm.a.mlp.0.weight"] = fill([]int{hidden, stackDim}, 0.02)
	tensors["mm.a.mlp.0.bias"] = zeros([]int{hidden})
	tensors["mm.a.mlp.2.weight"] = fill([]int{hidden, hidden}, 0.02)
	tensors["mm.a.mlp.2.bias"] = zeros([]int{hidden})

	// --- Text decoder tensors (standard Llama naming) ---
	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
	}

	return tensors
}

func TestBuildVoxtralGraph_Builds(t *testing.T) {
	cfg, vc := testVoxtralConfig()
	tensors := makeVoxtralTestTensors(cfg, vc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildVoxtralModel(vc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildVoxtralModel: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestVoxtralForward(t *testing.T) {
	cfg, vc := testVoxtralConfig()
	tensors := makeVoxtralTestTensors(cfg, vc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := BuildVoxtralModel(vc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildVoxtralModel: %v", err)
	}

	// Create a synthetic mel spectrogram input: [batch=1, num_mels, T_frames]
	tFrames := 32
	inputSize := 1 * vc.AudioNumMels * tFrames
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	input, err := tensor.New([]int{1, vc.AudioNumMels, tFrames}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	// Output should be [batch=1, numTokens, vocabSize].
	if len(shape) != 3 {
		t.Fatalf("expected 3D output, got shape %v", shape)
	}
	if shape[0] != 1 {
		t.Fatalf("batch = %d, want 1", shape[0])
	}
	if shape[2] != cfg.VocabSize {
		t.Fatalf("vocab size = %d, want %d", shape[2], cfg.VocabSize)
	}

	// Verify no NaN/Inf in output.
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

func TestVoxtralConfigFromGGUF(t *testing.T) {
	tests := []struct {
		name string
		cfg  *gguf.ModelConfig
		want VoxtralConfig
	}{
		{
			name: "defaults",
			cfg:  &gguf.ModelConfig{Architecture: "voxtral"},
			want: VoxtralConfig{
				AudioHiddenDim:        1280,
				AudioNumLayers:        32,
				AudioNumHeads:         20,
				AudioNumMels:          128,
				AudioIntermediateSize: 5120,
				AudioKernelSize:       3,
				StackFactor:           4,
			},
		},
		{
			name: "custom audio config",
			cfg: &gguf.ModelConfig{
				Architecture:              "voxtral",
				AudioHiddenSize:           1024,
				AudioNumLayers:            24,
				AudioNumHeads:             16,
				AudioNumMels:              80,
				AudioIntermediateSize:     4096,
				AudioProjectorStackFactor: 2,
			},
			want: VoxtralConfig{
				AudioHiddenDim:        1024,
				AudioNumLayers:        24,
				AudioNumHeads:         16,
				AudioNumMels:          80,
				AudioIntermediateSize: 4096,
				AudioKernelSize:       3,
				StackFactor:           2,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := VoxtralConfigFromGGUF(tt.cfg)
			if got.AudioHiddenDim != tt.want.AudioHiddenDim {
				t.Errorf("AudioHiddenDim = %d, want %d", got.AudioHiddenDim, tt.want.AudioHiddenDim)
			}
			if got.AudioNumLayers != tt.want.AudioNumLayers {
				t.Errorf("AudioNumLayers = %d, want %d", got.AudioNumLayers, tt.want.AudioNumLayers)
			}
			if got.AudioNumHeads != tt.want.AudioNumHeads {
				t.Errorf("AudioNumHeads = %d, want %d", got.AudioNumHeads, tt.want.AudioNumHeads)
			}
			if got.AudioNumMels != tt.want.AudioNumMels {
				t.Errorf("AudioNumMels = %d, want %d", got.AudioNumMels, tt.want.AudioNumMels)
			}
			if got.AudioIntermediateSize != tt.want.AudioIntermediateSize {
				t.Errorf("AudioIntermediateSize = %d, want %d", got.AudioIntermediateSize, tt.want.AudioIntermediateSize)
			}
			if got.StackFactor != tt.want.StackFactor {
				t.Errorf("StackFactor = %d, want %d", got.StackFactor, tt.want.StackFactor)
			}
		})
	}
}

func TestBuildVoxtralGraph_MissingTensor(t *testing.T) {
	cfg, vc := testVoxtralConfig()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := BuildVoxtralModel(vc, tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildVoxtralGraph_ArchRegistry(t *testing.T) {
	cfg, vc := testVoxtralConfig()
	tensors := makeVoxtralTestTensors(cfg, vc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Set audio config on the gguf.ModelConfig so buildVoxtralGraph can extract it.
	cfg.AudioHiddenSize = vc.AudioHiddenDim
	cfg.AudioNumLayers = vc.AudioNumLayers
	cfg.AudioNumHeads = vc.AudioNumHeads
	cfg.AudioNumMels = vc.AudioNumMels
	cfg.AudioIntermediateSize = vc.AudioIntermediateSize
	cfg.AudioProjectorStackFactor = vc.StackFactor

	g, emb, err := buildArchGraph("voxtral", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(voxtral): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestVoxtralRegistration(t *testing.T) {
	_, ok := GetArchitecture("voxtral")
	if !ok {
		t.Fatal("voxtral architecture not registered")
	}
}

func TestParseVoxtralConfig(t *testing.T) {
	registry := DefaultArchConfigRegistry()
	raw := map[string]interface{}{
		"model_type":          "voxtral",
		"vocab_size":          float64(32000),
		"hidden_size":         float64(4096),
		"num_hidden_layers":   float64(32),
		"num_attention_heads": float64(32),
		"num_key_value_heads": float64(8),
		"intermediate_size":   float64(14336),
		"rope_theta":          float64(500000),
	}

	meta, err := registry.Parse(raw)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if meta.Architecture != "voxtral" {
		t.Errorf("Architecture = %q, want %q", meta.Architecture, "voxtral")
	}
	if meta.HiddenSize != 4096 {
		t.Errorf("HiddenSize = %d, want 4096", meta.HiddenSize)
	}
	if meta.NumQueryHeads != 32 {
		t.Errorf("NumQueryHeads = %d, want 32", meta.NumQueryHeads)
	}
}
