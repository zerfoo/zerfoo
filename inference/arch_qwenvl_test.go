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

func makeQwenVLTestTensors(cfg *gguf.ModelConfig, qc QwenVLConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

	vHidden := qc.VisionHiddenDim
	vFFN := vHidden * 4
	numPatches := (qc.ImageSize / qc.PatchSize) * (qc.ImageSize / qc.PatchSize)
	patchDim := qc.NumChannels * qc.PatchSize * qc.PatchSize

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

	// --- Vision encoder tensors ---
	tensors["vision.patch_embed.weight"] = fill([]int{vHidden, patchDim}, 0.02)
	tensors["vision.patch_embed.bias"] = zeros([]int{vHidden})
	tensors["vision.class_embedding"] = fill([]int{vHidden}, 0.02)
	tensors["vision.position_embedding"] = fill([]int{numPatches + 1, vHidden}, 0.01)

	for i := 0; i < qc.VisionNumLayers; i++ {
		prefix := fmt.Sprintf("vision.blocks.%d.", i)
		tensors[prefix+"ln1.weight"] = ones([]int{vHidden})
		tensors[prefix+"ln1.bias"] = zeros([]int{vHidden})
		tensors[prefix+"attn.q_proj.weight"] = fill([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.k_proj.weight"] = fill([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.v_proj.weight"] = fill([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.o_proj.weight"] = fill([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"ln2.weight"] = ones([]int{vHidden})
		tensors[prefix+"ln2.bias"] = zeros([]int{vHidden})
		tensors[prefix+"mlp.fc1.weight"] = fill([]int{vFFN, vHidden}, 0.02)
		tensors[prefix+"mlp.fc1.bias"] = zeros([]int{vFFN})
		tensors[prefix+"mlp.fc2.weight"] = fill([]int{vHidden, vFFN}, 0.02)
		tensors[prefix+"mlp.fc2.bias"] = zeros([]int{vHidden})
	}

	tensors["vision.ln_post.weight"] = ones([]int{vHidden})
	tensors["vision.ln_post.bias"] = zeros([]int{vHidden})

	// --- Multi-modal projector tensors ---
	tensors["mm_projector.0.weight"] = fill([]int{hidden, vHidden}, 0.02)
	tensors["mm_projector.0.bias"] = zeros([]int{hidden})
	if qc.ProjectorType == "mlp" {
		tensors["mm_projector.2.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors["mm_projector.2.bias"] = zeros([]int{hidden})
	}

	// --- Text decoder tensors (Qwen2 naming with bias) ---
	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + fmt.Sprintf("%d", i) + "."
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		// Qwen2 attention biases.
		tensors[prefix+"self_attn.q_proj.bias"] = zeros([]int{hidden})
		tensors[prefix+"self_attn.k_proj.bias"] = zeros([]int{kvDim})
		tensors[prefix+"self_attn.v_proj.bias"] = zeros([]int{kvDim})
		tensors[prefix+"post_attention_layernorm.weight"] = ones([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fill([]int{hidden, inter}, 0.02)
	}

	return tensors
}

func testQwenVLConfig() (*gguf.ModelConfig, QwenVLConfig) {
	cfg := &gguf.ModelConfig{
		Architecture:     "qwen_vl",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        1000000.0,
	}
	qc := QwenVLConfig{
		ImageSize:       28,
		PatchSize:       14,
		VisionHiddenDim: 32,
		VisionNumHeads:  4,
		VisionNumLayers: 1,
		NumChannels:     3,
		ProjectorType:   "mlp",
	}
	return cfg, qc
}

func TestBuildQwenVLGraph_Builds(t *testing.T) {
	cfg, qc := testQwenVLConfig()
	tensors := makeQwenVLTestTensors(cfg, qc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildQwenVLModel(qc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildQwenVLModel: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestQwenVLForward(t *testing.T) {
	cfg, qc := testQwenVLConfig()
	tensors := makeQwenVLTestTensors(cfg, qc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := BuildQwenVLModel(qc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildQwenVLModel: %v", err)
	}

	// Create a synthetic image input: [1, 3, 28, 28].
	imageSize := qc.ImageSize
	channels := qc.NumChannels
	inputSize := 1 * channels * imageSize * imageSize
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	input, err := tensor.New([]int{1, channels, imageSize, imageSize}, inputData)
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

func TestQwenVLForward_MultiImage(t *testing.T) {
	cfg, qc := testQwenVLConfig()
	tensors := makeQwenVLTestTensors(cfg, qc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Run forward twice with different images to confirm multi-image support.
	// Each image is processed independently through the same graph.
	for imgIdx := 0; imgIdx < 2; imgIdx++ {
		g, _, err := BuildQwenVLModel(qc, tensors, cfg, engine)
		if err != nil {
			t.Fatalf("image %d: BuildQwenVLModel: %v", imgIdx, err)
		}

		imageSize := qc.ImageSize
		channels := qc.NumChannels
		inputSize := 1 * channels * imageSize * imageSize
		inputData := make([]float32, inputSize)
		for i := range inputData {
			inputData[i] = float32(math.Sin(float64(i+imgIdx*100)*0.1)) * 0.5
		}
		input, err := tensor.New([]int{1, channels, imageSize, imageSize}, inputData)
		if err != nil {
			t.Fatalf("image %d: create input tensor: %v", imgIdx, err)
		}

		output, err := g.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("image %d: forward: %v", imgIdx, err)
		}

		data := output.Data()
		for i, v := range data {
			if math.IsNaN(float64(v)) {
				t.Fatalf("image %d: NaN at index %d", imgIdx, i)
			}
			if math.IsInf(float64(v), 0) {
				t.Fatalf("image %d: Inf at index %d", imgIdx, i)
			}
		}
	}
}

func TestQwenVLForward_LinearProjector(t *testing.T) {
	cfg, qc := testQwenVLConfig()
	qc.ProjectorType = "linear"
	tensors := makeQwenVLTestTensors(cfg, qc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := BuildQwenVLModel(qc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildQwenVLModel: %v", err)
	}

	imageSize := qc.ImageSize
	channels := qc.NumChannels
	inputSize := 1 * channels * imageSize * imageSize
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	input, err := tensor.New([]int{1, channels, imageSize, imageSize}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
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

func TestQwenVLConfigFromGGUF(t *testing.T) {
	tests := []struct {
		name string
		cfg  *gguf.ModelConfig
		want QwenVLConfig
	}{
		{
			name: "defaults",
			cfg:  &gguf.ModelConfig{Architecture: "qwen_vl"},
			want: QwenVLConfig{
				ImageSize:       448,
				PatchSize:       14,
				VisionHiddenDim: 1024,
				VisionNumHeads:  16,
				VisionNumLayers: 24,
				NumChannels:     3,
				ProjectorType:   "mlp",
			},
		},
		{
			name: "custom vision config",
			cfg: &gguf.ModelConfig{
				Architecture:     "qwen_vl",
				VisionImageSize:  224,
				VisionPatchSize:  16,
				VisionHiddenSize: 768,
				VisionNumHeads:   12,
				VisionNumLayers:  12,
				ProjectorType:    "linear",
			},
			want: QwenVLConfig{
				ImageSize:       224,
				PatchSize:       16,
				VisionHiddenDim: 768,
				VisionNumHeads:  12,
				VisionNumLayers: 12,
				NumChannels:     3,
				ProjectorType:   "linear",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := QwenVLConfigFromGGUF(tt.cfg)
			if got.ImageSize != tt.want.ImageSize {
				t.Errorf("ImageSize = %d, want %d", got.ImageSize, tt.want.ImageSize)
			}
			if got.PatchSize != tt.want.PatchSize {
				t.Errorf("PatchSize = %d, want %d", got.PatchSize, tt.want.PatchSize)
			}
			if got.VisionHiddenDim != tt.want.VisionHiddenDim {
				t.Errorf("VisionHiddenDim = %d, want %d", got.VisionHiddenDim, tt.want.VisionHiddenDim)
			}
			if got.VisionNumHeads != tt.want.VisionNumHeads {
				t.Errorf("VisionNumHeads = %d, want %d", got.VisionNumHeads, tt.want.VisionNumHeads)
			}
			if got.VisionNumLayers != tt.want.VisionNumLayers {
				t.Errorf("VisionNumLayers = %d, want %d", got.VisionNumLayers, tt.want.VisionNumLayers)
			}
			if got.ProjectorType != tt.want.ProjectorType {
				t.Errorf("ProjectorType = %q, want %q", got.ProjectorType, tt.want.ProjectorType)
			}
		})
	}
}

func TestBuildQwenVLGraph_MissingTensor(t *testing.T) {
	cfg, qc := testQwenVLConfig()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := BuildQwenVLModel(qc, tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildQwenVLGraph_ArchRegistry(t *testing.T) {
	cfg, qc := testQwenVLConfig()
	tensors := makeQwenVLTestTensors(cfg, qc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Set vision config on the gguf.ModelConfig so buildQwenVLGraph can extract it.
	cfg.VisionImageSize = qc.ImageSize
	cfg.VisionPatchSize = qc.PatchSize
	cfg.VisionHiddenSize = qc.VisionHiddenDim
	cfg.VisionNumHeads = qc.VisionNumHeads
	cfg.VisionNumLayers = qc.VisionNumLayers
	cfg.ProjectorType = qc.ProjectorType

	g, emb, err := buildArchGraph("qwen_vl", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(qwen_vl): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestQwenVLForward_NoBias(t *testing.T) {
	cfg, qc := testQwenVLConfig()
	tensors := makeQwenVLTestTensors(cfg, qc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Remove bias tensors to test graceful fallback.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + fmt.Sprintf("%d", i) + "."
		delete(tensors, prefix+"self_attn.q_proj.bias")
		delete(tensors, prefix+"self_attn.k_proj.bias")
		delete(tensors, prefix+"self_attn.v_proj.bias")
	}

	g, _, err := BuildQwenVLModel(qc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildQwenVLModel: %v", err)
	}

	imageSize := qc.ImageSize
	channels := qc.NumChannels
	inputSize := 1 * channels * imageSize * imageSize
	inputData := make([]float32, inputSize)
	for i := range inputData {
		inputData[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	input, err := tensor.New([]int{1, channels, imageSize, imageSize}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
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
