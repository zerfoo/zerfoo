package inference

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// makeFalconTestTensors creates a minimal set of Falcon architecture tensors
// for testing. Tensor names follow the canonical model.layers.N. convention
// used by the loader after GGUF name mapping.
func makeFalconTestTensors(cfg *gguf.ModelConfig, withBias bool) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	kvDim := (hidden / cfg.NumHeads) * cfg.NumKVHeads

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

	// Global tensors.
	tensors["model.embed_tokens.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["model.norm.weight"] = ones([]int{hidden})
	if withBias {
		tensors["model.norm.bias"] = zeros([]int{hidden})
	}
	tensors["lm_head.weight"] = fill([]int{vocab, hidden}, 0.02)

	// Per-layer tensors.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + itoa(i) + "."

		// LayerNorm (gamma + optional beta).
		tensors[prefix+"input_layernorm.weight"] = ones([]int{hidden})
		if withBias {
			tensors[prefix+"input_layernorm.bias"] = zeros([]int{hidden})
		}

		// Attention projections.
		tensors[prefix+"self_attn.q_proj.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"self_attn.k_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.v_proj.weight"] = fill([]int{kvDim, hidden}, 0.02)
		tensors[prefix+"self_attn.o_proj.weight"] = fill([]int{hidden, hidden}, 0.02)

		// FFN projections.
		tensors[prefix+"mlp.dense_h_to_4h.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"mlp.dense_4h_to_h.weight"] = fill([]int{hidden, inter}, 0.02)
		if withBias {
			tensors[prefix+"mlp.dense_h_to_4h.bias"] = zeros([]int{inter})
			tensors[prefix+"mlp.dense_4h_to_h.bias"] = zeros([]int{hidden})
		}
	}

	return tensors
}

// falconMQAConfig returns a small MQA Falcon config (1 KV head).
func falconMQAConfig() *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "falcon",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       1, // MQA: single KV head
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
}

// falconGQAConfig returns a small GQA Falcon config (2 KV heads).
func falconGQAConfig() *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "falcon",
		VocabSize:        32,
		HiddenSize:       16,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2, // GQA: 2 KV heads (Falcon 40B / 180B pattern)
		IntermediateSize: 32,
		MaxSeqLen:        64,
		RopeTheta:        10000.0,
	}
}

// TestBuildFalconGraph_Builds verifies that the graph builder succeeds for
// MQA and GQA variants, with and without LayerNorm bias.
func TestBuildFalconGraph_Builds(t *testing.T) {
	tests := []struct {
		name     string
		cfg      *gguf.ModelConfig
		withBias bool
	}{
		{"MQA_with_bias", falconMQAConfig(), true},
		{"MQA_no_bias", falconMQAConfig(), false},
		{"GQA_with_bias", falconGQAConfig(), true},
		{"GQA_no_bias", falconGQAConfig(), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensors := makeFalconTestTensors(tt.cfg, tt.withBias)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildFalconGraph(tensors, tt.cfg, engine)
			if err != nil {
				t.Fatalf("buildFalconGraph: %v", err)
			}
			if g == nil {
				t.Fatal("graph is nil")
			}
			if emb == nil {
				t.Fatal("embedding is nil")
			}
		})
	}
}

// TestBuildFalconGraph_MissingTensor verifies that missing required tensors
// return an error.
func TestBuildFalconGraph_MissingTensor(t *testing.T) {
	cfg := falconMQAConfig()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildFalconGraph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors, got nil")
	}
}

// TestFalconForward verifies that a forward pass produces non-NaN output
// for both MQA and GQA configurations.
func TestFalconForward(t *testing.T) {
	tests := []struct {
		name     string
		cfg      *gguf.ModelConfig
		withBias bool
	}{
		{"MQA_with_bias", falconMQAConfig(), true},
		{"MQA_no_bias", falconMQAConfig(), false},
		{"GQA_with_bias", falconGQAConfig(), true},
		{"GQA_no_bias", falconGQAConfig(), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensors := makeFalconTestTensors(tt.cfg, tt.withBias)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildFalconGraph(tensors, tt.cfg, engine)
			if err != nil {
				t.Fatalf("buildFalconGraph: %v", err)
			}

			assertGraphForwardNonNaN(t, g, tt.cfg.VocabSize)
		})
	}
}

// TestFalconForward_Deterministic verifies that two identical forward passes
// produce the same result.
func TestFalconForward_Deterministic(t *testing.T) {
	cfg := falconMQAConfig()
	tensors := makeFalconTestTensors(cfg, true)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildFalconGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildFalconGraph: %v", err)
	}

	tokenIDs := []float32{1, 5, 10}
	input, err := tensor.New([]int{1, 3}, tokenIDs)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()

	out1, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("first forward: %v", err)
	}
	data1 := make([]float32, len(out1.Data()))
	copy(data1, out1.Data())

	out2, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("second forward: %v", err)
	}
	data2 := out2.Data()

	if len(data1) != len(data2) {
		t.Fatalf("output length mismatch: %d vs %d", len(data1), len(data2))
	}
	const tol = 1e-5
	for i := range data1 {
		diff := float64(data1[i]) - float64(data2[i])
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Fatalf("output differs at index %d: %v vs %v (diff=%v)", i, data1[i], data2[i], diff)
		}
	}
}

// TestFalconForward_DifferentInputsDifferentOutputs verifies that different
// inputs produce different outputs (the model is not collapsing).
func TestFalconForward_DifferentInputsDifferentOutputs(t *testing.T) {
	cfg := falconMQAConfig()
	tensors := makeFalconTestTensors(cfg, true)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildFalconGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildFalconGraph: %v", err)
	}

	ctx := context.Background()
	input1, _ := tensor.New([]int{1, 3}, []float32{1, 2, 3})
	input2, _ := tensor.New([]int{1, 3}, []float32{4, 5, 6})

	out1, err := g.Forward(ctx, input1)
	if err != nil {
		t.Fatalf("forward input1: %v", err)
	}
	out2, err := g.Forward(ctx, input2)
	if err != nil {
		t.Fatalf("forward input2: %v", err)
	}

	d1 := out1.Data()
	d2 := out2.Data()

	identical := true
	for i := range d1 {
		if d1[i] != d2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("different inputs produced identical output; model may be collapsing")
	}
}

// TestFalconForward_OutputShape verifies that the output shape matches
// [1, seqLen, vocabSize] for various sequence lengths.
func TestFalconForward_OutputShape(t *testing.T) {
	tests := []struct {
		name   string
		seqLen int
		tokens []float32
	}{
		{"single_token", 1, []float32{5}},
		{"three_tokens", 3, []float32{1, 5, 10}},
		{"four_tokens", 4, []float32{1, 5, 10, 3}},
	}

	cfg := falconMQAConfig()
	tensors := makeFalconTestTensors(cfg, true)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildFalconGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildFalconGraph: %v", err)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, err := tensor.New([]int{1, tt.seqLen}, tt.tokens)
			if err != nil {
				t.Fatalf("create input: %v", err)
			}

			output, err := g.Forward(context.Background(), input)
			if err != nil {
				t.Fatalf("forward: %v", err)
			}

			shape := output.Shape()
			if len(shape) != 3 || shape[0] != 1 || shape[1] != tt.seqLen || shape[2] != cfg.VocabSize {
				t.Fatalf("output shape = %v, want [1, %d, %d]", shape, tt.seqLen, cfg.VocabSize)
			}
		})
	}
}

// TestFalconForward_MQAVsGQA verifies that MQA and GQA variants produce
// different outputs (different number of KV heads changes computation).
func TestFalconForward_MQAVsGQA(t *testing.T) {
	mqaCfg := falconMQAConfig()
	gqaCfg := falconGQAConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	mqaTensors := makeFalconTestTensors(mqaCfg, false)
	gqaTensors := makeFalconTestTensors(gqaCfg, false)

	mqaGraph, _, err := buildFalconGraph(mqaTensors, mqaCfg, engine)
	if err != nil {
		t.Fatalf("buildFalconGraph MQA: %v", err)
	}
	gqaGraph, _, err := buildFalconGraph(gqaTensors, gqaCfg, engine)
	if err != nil {
		t.Fatalf("buildFalconGraph GQA: %v", err)
	}

	input, _ := tensor.New([]int{1, 3}, []float32{1, 5, 10})
	ctx := context.Background()

	mqaOut, err := mqaGraph.Forward(ctx, input)
	if err != nil {
		t.Fatalf("MQA forward: %v", err)
	}
	gqaOut, err := gqaGraph.Forward(ctx, input)
	if err != nil {
		t.Fatalf("GQA forward: %v", err)
	}

	mqaData := mqaOut.Data()
	gqaData := gqaOut.Data()

	if len(mqaData) != len(gqaData) {
		t.Fatalf("output length mismatch: MQA=%d, GQA=%d", len(mqaData), len(gqaData))
	}

	identical := true
	for i := range mqaData {
		if mqaData[i] != gqaData[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("MQA and GQA produced identical output; expected different outputs for different KV heads")
	}
}

// TestFalconForward_TwoLayerResidualFlow verifies that a 2-layer model
// produces different output than a 1-layer model (second layer has effect).
func TestFalconForward_TwoLayerResidualFlow(t *testing.T) {
	cfg1 := falconMQAConfig()
	cfg1.NumLayers = 1

	cfg2 := falconMQAConfig()
	cfg2.NumLayers = 2

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	t1 := makeFalconTestTensors(cfg1, true)
	t2 := makeFalconTestTensors(cfg2, true)

	g1, _, err := buildFalconGraph(t1, cfg1, engine)
	if err != nil {
		t.Fatalf("buildFalconGraph 1-layer: %v", err)
	}
	g2, _, err := buildFalconGraph(t2, cfg2, engine)
	if err != nil {
		t.Fatalf("buildFalconGraph 2-layer: %v", err)
	}

	input, _ := tensor.New([]int{1, 3}, []float32{1, 5, 10})
	ctx := context.Background()

	out1, err := g1.Forward(ctx, input)
	if err != nil {
		t.Fatalf("1-layer forward: %v", err)
	}
	out2, err := g2.Forward(ctx, input)
	if err != nil {
		t.Fatalf("2-layer forward: %v", err)
	}

	d1 := out1.Data()
	d2 := out2.Data()

	if len(d1) != len(d2) {
		t.Fatalf("output length mismatch: 1-layer=%d, 2-layer=%d", len(d1), len(d2))
	}

	identical := true
	for i := range d1 {
		if d1[i] != d2[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("1-layer and 2-layer models produced identical output; second layer has no effect")
	}
}

// TestBuildArchGraph_FalconDispatches verifies that the architecture registry
// dispatches correctly to the Falcon builder.
func TestBuildArchGraph_FalconDispatches(t *testing.T) {
	cfg := falconMQAConfig()
	tensors := makeFalconTestTensors(cfg, true)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("falcon", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(falcon): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}
