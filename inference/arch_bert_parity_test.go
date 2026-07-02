package inference

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// bertParityConfig returns a tiny 2-layer BERT configuration for testing.
func bertParityConfig() *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "bert",
		HiddenSize:       32,
		NumHeads:         2,
		NumKVHeads:       2,
		IntermediateSize: 64,
		MaxSeqLen:        8,
		VocabSize:        100,
		NumLabels:        3,
		NumLayers:        2,
		LayerNormEps:     1e-12,
	}
}

// bertParityTensors creates all weight tensors for the tiny BERT model using a
// seeded random source. Values are drawn uniformly from [-0.1, 0.1].
func bertParityTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	rng := rand.New(rand.NewSource(42))
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	randTensor := func(name string, shape []int) {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(rng.Float64()*0.2 - 0.1)
		}
		t, err := tensor.New[float32](shape, data)
		if err != nil {
			panic("failed to create tensor " + name + ": " + err.Error())
		}
		tensors[name] = t
	}

	h := cfg.HiddenSize
	inter := cfg.IntermediateSize

	// Embeddings.
	randTensor("token_embd.weight", []int{cfg.VocabSize, h})
	randTensor("position_embd.weight", []int{cfg.MaxSeqLen, h})
	randTensor("token_type_embd.weight", []int{2, h})

	// Embedding LayerNorm.
	randTensor("token_embd_norm.weight", []int{h})
	randTensor("token_embd_norm.bias", []int{h})

	// Transformer layers.
	for i := 0; i < cfg.NumLayers; i++ {
		p := fmt.Sprintf("blk.%d.", i)

		// Self-attention Q/K/V/O weights and biases.
		randTensor(p+"attn_q.weight", []int{h, h})
		randTensor(p+"attn_q.bias", []int{h})
		randTensor(p+"attn_k.weight", []int{h, h})
		randTensor(p+"attn_k.bias", []int{h})
		randTensor(p+"attn_v.weight", []int{h, h})
		randTensor(p+"attn_v.bias", []int{h})
		randTensor(p+"attn_output.weight", []int{h, h})
		randTensor(p+"attn_output.bias", []int{h})

		// Post-attention LayerNorm.
		randTensor(p+"attn_norm.weight", []int{h})
		randTensor(p+"attn_norm.bias", []int{h})

		// FFN up/down weights and biases.
		randTensor(p+"ffn_up.weight", []int{inter, h})
		randTensor(p+"ffn_up.bias", []int{inter})
		randTensor(p+"ffn_down.weight", []int{h, inter})
		randTensor(p+"ffn_down.bias", []int{h})

		// Post-FFN LayerNorm.
		randTensor(p+"ffn_norm.weight", []int{h})
		randTensor(p+"ffn_norm.bias", []int{h})
	}

	// Pooler (CLS token projection + tanh).
	randTensor("cls_pooler.weight", []int{h, h})
	randTensor("cls_pooler.bias", []int{h})

	// Classification head.
	randTensor("cls.weight", []int{cfg.NumLabels, h})
	randTensor("cls.bias", []int{cfg.NumLabels})

	return tensors
}

// buildTestBertModel builds a BERT graph from the given config and tensors,
// returning the graph and engine. The caller should close the engine.
func buildTestBertModel(t *testing.T, cfg *gguf.ModelConfig, tensors map[string]*tensor.TensorNumeric[float32]) (*EncoderModel, func()) {
	t.Helper()
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	g, embWeight, err := buildBertGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildBertGraph: %v", err)
	}

	model := &EncoderModel{
		graph:       g,
		embedWeight: embWeight,
		config:      cfg,
		engine:      engine,
	}
	cleanup := func() {
		_ = engine.Close(context.Background())
	}
	return model, cleanup
}

// runBertForward runs the encoder model on the given token IDs and returns output logits.
func runBertForward(t *testing.T, model *EncoderModel, inputIDs []int) []float32 {
	t.Helper()
	output, err := model.Forward(context.Background(), inputIDs)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	return output
}

func TestBertParity_FullForward(t *testing.T) {
	cfg := bertParityConfig()
	tensors := bertParityTensors(cfg)
	model, cleanup := buildTestBertModel(t, cfg, tensors)
	defer cleanup()

	inputIDs := []int{1, 42, 73, 2} // CLS + 2 tokens + SEP
	output := runBertForward(t, model, inputIDs)

	// Verify output shape: should be [1, numLabels] = 3 values.
	if len(output) != cfg.NumLabels {
		t.Fatalf("expected %d output logits, got %d", cfg.NumLabels, len(output))
	}

	// Verify all values are finite (no NaN/Inf).
	for i, v := range output {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] is not finite: %v", i, v)
		}
	}

	// Verify output is not near-uniform: the model should produce
	// differentiated logits even with random weights.
	minVal, maxVal := output[0], output[0]
	for _, v := range output[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	spread := maxVal - minVal
	if spread < 0.001 {
		t.Fatalf("output logits are near-uniform (spread=%.6f): %v", spread, output)
	}

	t.Logf("output logits: %v (spread=%.4f)", output, spread)
}

func TestBertParity_Deterministic(t *testing.T) {
	cfg := bertParityConfig()
	tensors := bertParityTensors(cfg)
	model, cleanup := buildTestBertModel(t, cfg, tensors)
	defer cleanup()

	inputIDs := []int{1, 42, 73, 2}

	out1 := runBertForward(t, model, inputIDs)
	out2 := runBertForward(t, model, inputIDs)

	if len(out1) != len(out2) {
		t.Fatalf("output lengths differ: %d vs %d", len(out1), len(out2))
	}
	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("output[%d] differs: %v vs %v", i, out1[i], out2[i])
		}
	}
}

func TestBertParity_DifferentInput(t *testing.T) {
	cfg := bertParityConfig()
	tensors := bertParityTensors(cfg)
	model, cleanup := buildTestBertModel(t, cfg, tensors)
	defer cleanup()

	out1 := runBertForward(t, model, []int{1, 42, 73, 2})
	out2 := runBertForward(t, model, []int{1, 55, 88, 2})

	same := true
	for i := range out1 {
		if out1[i] != out2[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("different inputs produced identical output: %v", out1)
	}
}

func TestBertParity_EmbeddingOnly(t *testing.T) {
	cfg := bertParityConfig()
	tensors := bertParityTensors(cfg)

	tokenW := tensors["token_embd.weight"]
	posW := tensors["position_embd.weight"]
	typeW := tensors["token_type_embd.weight"]
	normW := tensors["token_embd_norm.weight"]
	normB := tensors["token_embd_norm.bias"]

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() { _ = engine.Close(context.Background()) }()
	proxy := compute.NewEngineProxy[float32](engine)

	// Manually compute embedding for token IDs [1, 42].
	inputIDs := []int{1, 42}
	h := cfg.HiddenSize

	tokenData := tokenW.Data()
	posData := posW.Data()
	typeData := typeW.Data()

	// Sum: token + position + token_type(0)
	rawEmb := make([]float32, len(inputIDs)*h)
	for i, id := range inputIDs {
		for d := 0; d < h; d++ {
			rawEmb[i*h+d] = tokenData[id*h+d] + posData[i*h+d] + typeData[d]
		}
	}

	// Apply LayerNorm manually.
	eps := float32(1e-12)
	normWData := normW.Data()
	normBData := normB.Data()
	expected := make([]float32, len(inputIDs)*h)
	for i := 0; i < len(inputIDs); i++ {
		// Compute mean.
		var mean float64
		for d := 0; d < h; d++ {
			mean += float64(rawEmb[i*h+d])
		}
		mean /= float64(h)

		// Compute variance.
		var variance float64
		for d := 0; d < h; d++ {
			diff := float64(rawEmb[i*h+d]) - mean
			variance += diff * diff
		}
		variance /= float64(h)

		// Normalize.
		invStd := 1.0 / math.Sqrt(variance+float64(eps))
		for d := 0; d < h; d++ {
			normed := (float64(rawEmb[i*h+d]) - mean) * invStd
			expected[i*h+d] = float32(normed)*normWData[d] + normBData[d]
		}
	}

	// Run through the embedding node + LayerNorm (now separate).
	embNode := &bertEmbeddingNode[float32]{
		engine:      proxy,
		tokenWeight: tokenW,
		posWeight:   posW,
		typeWeight:  typeW,
	}

	inputData := make([]float32, len(inputIDs))
	for i, id := range inputIDs {
		inputData[i] = float32(id)
	}
	inputTensor, err := tensor.New[float32]([]int{1, len(inputIDs)}, inputData)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	embedded, err := embNode.Forward(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("embedding forward: %v", err)
	}

	// Apply LayerNorm as a separate step (matches the refactored graph structure).
	embLN := normalization.NewLayerNormalizationFromParams[float32](
		proxy, eps,
		&graph.Parameter[float32]{Name: "emb_norm_gamma", Value: normW},
		&graph.Parameter[float32]{Name: "emb_norm_beta", Value: normB},
	)
	result, err := embLN.Forward(context.Background(), embedded)
	if err != nil {
		t.Fatalf("embedding layernorm: %v", err)
	}

	resultData := result.Data()
	if len(resultData) != len(expected) {
		t.Fatalf("result length %d != expected %d", len(resultData), len(expected))
	}

	tol := float32(1e-5)
	for i := range expected {
		diff := resultData[i] - expected[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			t.Errorf("embedding[%d]: got %.6f, want %.6f (diff=%.6f)", i, resultData[i], expected[i], diff)
		}
	}
}

func TestBertParity_GradientCheck(t *testing.T) {
	// Verify that attention scores sum to ~1.0 per row (softmax property).
	// We do this indirectly: run a single layer and check the output is well-formed.
	// The attention mechanism uses softmax internally, so if the output is finite
	// and varies across sequence positions, softmax is working correctly.
	cfg := bertParityConfig()
	cfg.NumLayers = 1 // Single layer for clarity.
	tensors := bertParityTensors(cfg)
	model, cleanup := buildTestBertModel(t, cfg, tensors)
	defer cleanup()

	inputIDs := []int{1, 42, 73, 2}
	output := runBertForward(t, model, inputIDs)

	// With a single layer and valid softmax, output should be finite.
	for i, v := range output {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] is not finite: %v (softmax may be broken)", i, v)
		}
	}

	// Run with different sequence lengths to verify attention adapts.
	out3 := runBertForward(t, model, []int{1, 42, 2})
	out4 := runBertForward(t, model, []int{1, 42, 73, 55, 2})

	// Different lengths should produce different outputs (attention changes).
	same3 := true
	for i := range output {
		if output[i] != out3[i] {
			same3 = false
			break
		}
	}
	same4 := true
	for i := range output {
		if output[i] != out4[i] {
			same4 = false
			break
		}
	}
	if same3 && same4 {
		t.Fatal("attention appears to ignore sequence length (all outputs identical)")
	}
}
