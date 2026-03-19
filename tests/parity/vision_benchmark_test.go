package parity_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// visionModelConfig describes a vision model for parity/benchmark tests.
type visionModelConfig struct {
	Name    string
	EnvVar  string // env var pointing to the GGUF file
	Arch    string // architecture name (llava, qwen_vl)
	ImageH  int    // image height
	ImageW  int    // image width
	Channels int
}

var llavaConfig = visionModelConfig{
	Name:     "LLaVA",
	EnvVar:   "LLAVA_GGUF_PATH",
	Arch:     "llava",
	ImageH:   336,
	ImageW:   336,
	Channels: 3,
}

var qwenVLConfig = visionModelConfig{
	Name:     "QwenVL",
	EnvVar:   "QWENVL_GGUF_PATH",
	Arch:     "qwen_vl",
	ImageH:   448,
	ImageW:   448,
	Channels: 3,
}

// --- Full-model tests (require GGUF model files) ---

// TestLLaVA_VisionPipeline loads a LLaVA GGUF model and runs a full
// vision pipeline: synthetic image -> vision encoder -> projector -> decoder -> logits.
// Skipped when LLAVA_GGUF_PATH is not set.
func TestLLaVA_VisionPipeline(t *testing.T) {
	runVisionPipelineTest(t, llavaConfig)
}

// TestQwenVL_VisionPipeline loads a Qwen-VL GGUF model and runs a full
// vision pipeline: synthetic image -> vision encoder -> projector -> decoder -> logits.
// Skipped when QWENVL_GGUF_PATH is not set.
func TestQwenVL_VisionPipeline(t *testing.T) {
	runVisionPipelineTest(t, qwenVLConfig)
}

func runVisionPipelineTest(t *testing.T, vcfg visionModelConfig) {
	t.Helper()

	ggufPath := os.Getenv(vcfg.EnvVar)
	if ggufPath == "" {
		t.Skipf("%s not set; skipping %s vision pipeline test", vcfg.EnvVar, vcfg.Name)
	}
	if _, err := os.Stat(ggufPath); err != nil {
		t.Skipf("%s file not found: %v", vcfg.EnvVar, err)
	}

	mdl, err := inference.LoadFile(ggufPath)
	if err != nil {
		t.Fatalf("inference.LoadFile(%s): %v", vcfg.Name, err)
	}

	// Use a simple text prompt to test the text pathway loads correctly.
	ctx := context.Background()
	result, err := mdl.Generate(ctx, "Describe this image.",
		inference.WithTemperature(0),
		inference.WithMaxTokens(16),
	)
	if err != nil {
		t.Fatalf("%s Generate failed: %v", vcfg.Name, err)
	}
	if result == "" {
		t.Fatalf("%s generated empty output", vcfg.Name)
	}
	t.Logf("%s output: %q", vcfg.Name, result)
}

// --- Synthetic-weight graph-level tests (no model files required) ---

// TestVision_LLaVA_SyntheticForward builds a LLaVA graph with synthetic weights
// and runs a forward pass with a synthetic image tensor.
func TestVision_LLaVA_SyntheticForward(t *testing.T) {
	cfg, lc := syntheticLLaVAConfig()
	tensors := makeSyntheticLLaVATensors(cfg, lc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := inference.BuildLLaVAModel(lc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildLLaVAModel: %v", err)
	}

	input := syntheticImageTensor(t, lc.NumChannels, lc.ImageSize, lc.ImageSize)
	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	validateVisionOutput(t, "LLaVA", output, cfg.VocabSize)
}

// TestVision_QwenVL_SyntheticForward builds a Qwen-VL graph with synthetic weights
// and runs a forward pass with a synthetic image tensor.
func TestVision_QwenVL_SyntheticForward(t *testing.T) {
	cfg, qc := syntheticQwenVLConfig()
	tensors := makeSyntheticQwenVLTensors(cfg, qc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := inference.BuildQwenVLModel(qc, tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildQwenVLModel: %v", err)
	}

	input := syntheticImageTensor(t, qc.NumChannels, qc.ImageSize, qc.ImageSize)
	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	validateVisionOutput(t, "QwenVL", output, cfg.VocabSize)
}

// --- Benchmarks (synthetic weights, no model files required) ---

// BenchmarkLLaVA_Throughput measures vision pipeline throughput for LLaVA
// using a synthetic model with small dimensions.
func BenchmarkLLaVA_Throughput(b *testing.B) {
	cfg, lc := syntheticLLaVAConfig()
	tensors := makeSyntheticLLaVATensors(cfg, lc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := inference.BuildLLaVAModel(lc, tensors, cfg, engine)
	if err != nil {
		b.Fatalf("BuildLLaVAModel: %v", err)
	}

	input := syntheticImageTensorB(b, lc.NumChannels, lc.ImageSize, lc.ImageSize)
	ctx := context.Background()

	// Warmup.
	if _, err := g.Forward(ctx, input); err != nil {
		b.Fatalf("warmup forward: %v", err)
	}

	b.ResetTimer()
	for range b.N {
		output, err := g.Forward(ctx, input)
		if err != nil {
			b.Fatalf("forward: %v", err)
		}
		// Prevent compiler from optimizing away the forward call.
		if output == nil {
			b.Fatal("nil output")
		}
	}
}

// BenchmarkQwenVL_Throughput measures vision pipeline throughput for Qwen-VL
// using a synthetic model with small dimensions.
func BenchmarkQwenVL_Throughput(b *testing.B) {
	cfg, qc := syntheticQwenVLConfig()
	tensors := makeSyntheticQwenVLTensors(cfg, qc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := inference.BuildQwenVLModel(qc, tensors, cfg, engine)
	if err != nil {
		b.Fatalf("BuildQwenVLModel: %v", err)
	}

	input := syntheticImageTensorB(b, qc.NumChannels, qc.ImageSize, qc.ImageSize)
	ctx := context.Background()

	// Warmup.
	if _, err := g.Forward(ctx, input); err != nil {
		b.Fatalf("warmup forward: %v", err)
	}

	b.ResetTimer()
	for range b.N {
		output, err := g.Forward(ctx, input)
		if err != nil {
			b.Fatalf("forward: %v", err)
		}
		if output == nil {
			b.Fatal("nil output")
		}
	}
}

// --- Helpers ---

func syntheticLLaVAConfig() (*gguf.ModelConfig, inference.LLaVAConfig) {
	cfg := &gguf.ModelConfig{
		Architecture:     "llava",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       2,
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
	lc := inference.LLaVAConfig{
		ImageSize:       28,
		PatchSize:       14,
		VisionHiddenDim: 32,
		VisionNumHeads:  4,
		VisionNumLayers: 1,
		NumChannels:     3,
		ProjectorType:   "mlp",
	}
	return cfg, lc
}

func syntheticQwenVLConfig() (*gguf.ModelConfig, inference.QwenVLConfig) {
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
	qc := inference.QwenVLConfig{
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

func makeSyntheticLLaVATensors(cfg *gguf.ModelConfig, lc inference.LLaVAConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

	vHidden := lc.VisionHiddenDim
	vFFN := vHidden * 4
	numPatches := (lc.ImageSize / lc.PatchSize) * (lc.ImageSize / lc.PatchSize)
	patchDim := lc.NumChannels * lc.PatchSize * lc.PatchSize

	// Vision encoder tensors.
	tensors["vision.patch_embed.weight"] = fillTensor([]int{vHidden, patchDim}, 0.02)
	tensors["vision.patch_embed.bias"] = zeroTensor([]int{vHidden})
	tensors["vision.class_embedding"] = fillTensor([]int{vHidden}, 0.02)
	tensors["vision.position_embedding"] = fillTensor([]int{numPatches + 1, vHidden}, 0.01)

	for i := 0; i < lc.VisionNumLayers; i++ {
		prefix := fmt.Sprintf("vision.blocks.%d.", i)
		tensors[prefix+"ln1.weight"] = onesTensor([]int{vHidden})
		tensors[prefix+"ln1.bias"] = zeroTensor([]int{vHidden})
		tensors[prefix+"attn.q_proj.weight"] = fillTensor([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.k_proj.weight"] = fillTensor([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.v_proj.weight"] = fillTensor([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.o_proj.weight"] = fillTensor([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"ln2.weight"] = onesTensor([]int{vHidden})
		tensors[prefix+"ln2.bias"] = zeroTensor([]int{vHidden})
		tensors[prefix+"mlp.fc1.weight"] = fillTensor([]int{vFFN, vHidden}, 0.02)
		tensors[prefix+"mlp.fc1.bias"] = zeroTensor([]int{vFFN})
		tensors[prefix+"mlp.fc2.weight"] = fillTensor([]int{vHidden, vFFN}, 0.02)
		tensors[prefix+"mlp.fc2.bias"] = zeroTensor([]int{vHidden})
	}

	tensors["vision.ln_post.weight"] = onesTensor([]int{vHidden})
	tensors["vision.ln_post.bias"] = zeroTensor([]int{vHidden})

	// Multi-modal projector tensors.
	tensors["mm_projector.0.weight"] = fillTensor([]int{hidden, vHidden}, 0.02)
	tensors["mm_projector.0.bias"] = zeroTensor([]int{hidden})
	if lc.ProjectorType == "mlp" {
		tensors["mm_projector.2.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors["mm_projector.2.bias"] = zeroTensor([]int{hidden})
	}

	// Text decoder tensors (standard Llama naming).
	tensors["model.embed_tokens.weight"] = fillTensor([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = onesTensor([]int{hidden})
	tensors["lm_head.weight"] = fillTensor([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		tensors[prefix+"input_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fillTensor([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fillTensor([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors[prefix+"post_attention_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fillTensor([]int{hidden, inter}, 0.02)
	}

	return tensors
}

func makeSyntheticQwenVLTensors(cfg *gguf.ModelConfig, qc inference.QwenVLConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

	vHidden := qc.VisionHiddenDim
	vFFN := vHidden * 4
	numPatches := (qc.ImageSize / qc.PatchSize) * (qc.ImageSize / qc.PatchSize)
	patchDim := qc.NumChannels * qc.PatchSize * qc.PatchSize

	// Vision encoder tensors.
	tensors["vision.patch_embed.weight"] = fillTensor([]int{vHidden, patchDim}, 0.02)
	tensors["vision.patch_embed.bias"] = zeroTensor([]int{vHidden})
	tensors["vision.class_embedding"] = fillTensor([]int{vHidden}, 0.02)
	tensors["vision.position_embedding"] = fillTensor([]int{numPatches + 1, vHidden}, 0.01)

	for i := 0; i < qc.VisionNumLayers; i++ {
		prefix := fmt.Sprintf("vision.blocks.%d.", i)
		tensors[prefix+"ln1.weight"] = onesTensor([]int{vHidden})
		tensors[prefix+"ln1.bias"] = zeroTensor([]int{vHidden})
		tensors[prefix+"attn.q_proj.weight"] = fillTensor([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.k_proj.weight"] = fillTensor([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.v_proj.weight"] = fillTensor([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"attn.o_proj.weight"] = fillTensor([]int{vHidden, vHidden}, 0.02)
		tensors[prefix+"ln2.weight"] = onesTensor([]int{vHidden})
		tensors[prefix+"ln2.bias"] = zeroTensor([]int{vHidden})
		tensors[prefix+"mlp.fc1.weight"] = fillTensor([]int{vFFN, vHidden}, 0.02)
		tensors[prefix+"mlp.fc1.bias"] = zeroTensor([]int{vFFN})
		tensors[prefix+"mlp.fc2.weight"] = fillTensor([]int{vHidden, vFFN}, 0.02)
		tensors[prefix+"mlp.fc2.bias"] = zeroTensor([]int{vHidden})
	}

	tensors["vision.ln_post.weight"] = onesTensor([]int{vHidden})
	tensors["vision.ln_post.bias"] = zeroTensor([]int{vHidden})

	// Multi-modal projector tensors.
	tensors["mm_projector.0.weight"] = fillTensor([]int{hidden, vHidden}, 0.02)
	tensors["mm_projector.0.bias"] = zeroTensor([]int{hidden})
	if qc.ProjectorType == "mlp" {
		tensors["mm_projector.2.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors["mm_projector.2.bias"] = zeroTensor([]int{hidden})
	}

	// Text decoder tensors (Qwen2 naming with attention bias).
	tensors["model.embed_tokens.weight"] = fillTensor([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = onesTensor([]int{hidden})
	tensors["lm_head.weight"] = fillTensor([]int{vocab, hidden}, 0.02)

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)
		tensors[prefix+"input_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"self_attn.q_proj.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fillTensor([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fillTensor([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fillTensor([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.q_proj.bias"] = zeroTensor([]int{hidden})
		tensors[prefix+"self_attn.k_proj.bias"] = zeroTensor([]int{kvDim})
		tensors[prefix+"self_attn.v_proj.bias"] = zeroTensor([]int{kvDim})
		tensors[prefix+"post_attention_layernorm.weight"] = onesTensor([]int{hidden})
		tensors[prefix+"mlp.gate_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.up_proj.weight"] = fillTensor([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.down_proj.weight"] = fillTensor([]int{hidden, inter}, 0.02)
	}

	return tensors
}

func fillTensor(shape []int, scale float32) *tensor.TensorNumeric[float32] {
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

func onesTensor(shape []int) *tensor.TensorNumeric[float32] {
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

func zeroTensor(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	t, _ := tensor.New(shape, data)
	return t
}

func syntheticImageTensor(t *testing.T, channels, height, width int) *tensor.TensorNumeric[float32] {
	t.Helper()
	size := 1 * channels * height * width
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	img, err := tensor.New([]int{1, channels, height, width}, data)
	if err != nil {
		t.Fatalf("create image tensor: %v", err)
	}
	return img
}

func syntheticImageTensorB(b *testing.B, channels, height, width int) *tensor.TensorNumeric[float32] {
	b.Helper()
	size := 1 * channels * height * width
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	img, err := tensor.New([]int{1, channels, height, width}, data)
	if err != nil {
		b.Fatalf("create image tensor: %v", err)
	}
	return img
}

func validateVisionOutput(t *testing.T, name string, output *tensor.TensorNumeric[float32], vocabSize int) {
	t.Helper()

	shape := output.Shape()
	if len(shape) != 3 {
		t.Fatalf("%s: expected 3D output, got shape %v", name, shape)
	}
	if shape[0] != 1 {
		t.Fatalf("%s: batch = %d, want 1", name, shape[0])
	}
	if shape[2] != vocabSize {
		t.Fatalf("%s: vocab size = %d, want %d", name, shape[2], vocabSize)
	}

	t.Logf("%s output shape: %v", name, shape)

	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("%s: NaN at index %d", name, i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("%s: Inf at index %d", name, i)
		}
	}
}
