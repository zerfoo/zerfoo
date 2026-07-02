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

func gpt2BaseConfig() *gguf.ModelConfig {
	return &gguf.ModelConfig{
		Architecture:     "gpt2",
		VocabSize:        128,
		HiddenSize:       64,
		NumLayers:        2,
		NumHeads:         2,
		NumKVHeads:       2, // MHA: num_kv_heads == num_heads
		IntermediateSize: 256,
		MaxSeqLen:        64,
		LayerNormEps:     1e-5,
	}
}

// makeGPT2TestTensors creates a minimal set of GPT-2-architecture tensors for
// testing. All weights use small deterministic values to produce non-NaN output.
func makeGPT2TestTensors(cfg *gguf.ModelConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	hidden := cfg.HiddenSize
	inter := cfg.IntermediateSize
	vocab := cfg.VocabSize
	maxPos := cfg.MaxSeqLen

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

	// Global embedding tensors.
	tensors["token_embd.weight"] = fill([]int{vocab, hidden}, 0.02)
	tensors["position_embd.weight"] = fill([]int{maxPos, hidden}, 0.02)

	// Final LayerNorm.
	tensors["output_norm.weight"] = ones([]int{hidden})
	tensors["output_norm.bias"] = zeros([]int{hidden})

	// Per-layer tensors.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "blk." + itoa(i) + "."

		// Attention LayerNorm.
		tensors[prefix+"attn_norm.weight"] = ones([]int{hidden})
		tensors[prefix+"attn_norm.bias"] = zeros([]int{hidden})

		// Merged QKV weight: [3*hidden, hidden] for MHA.
		tensors[prefix+"attn_qkv.weight"] = fill([]int{3 * hidden, hidden}, 0.02)
		tensors[prefix+"attn_qkv.bias"] = zeros([]int{3 * hidden})

		// Output projection.
		tensors[prefix+"attn_output.weight"] = fill([]int{hidden, hidden}, 0.02)
		tensors[prefix+"attn_output.bias"] = zeros([]int{hidden})

		// FFN LayerNorm.
		tensors[prefix+"ffn_norm.weight"] = ones([]int{hidden})
		tensors[prefix+"ffn_norm.bias"] = zeros([]int{hidden})

		// FFN weights and biases (2-matrix: up + down).
		tensors[prefix+"ffn_up.weight"] = fill([]int{inter, hidden}, 0.02)
		tensors[prefix+"ffn_up.bias"] = zeros([]int{inter})
		tensors[prefix+"ffn_down.weight"] = fill([]int{hidden, inter}, 0.02)
		tensors[prefix+"ffn_down.bias"] = zeros([]int{hidden})
	}

	return tensors
}

func TestBuildGPT2Graph_Builds(t *testing.T) {
	cfg := gpt2BaseConfig()
	tensors := makeGPT2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildGPT2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGPT2Graph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildGPT2Graph_ForwardProducesVocabLogits(t *testing.T) {
	cfg := gpt2BaseConfig()
	tensors := makeGPT2TestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGPT2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGPT2Graph: %v", err)
	}

	// Input: token IDs [1, 4].
	tokenIDs := []float32{1, 5, 10, 3}
	input, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	// GPT-2 LM head: output is [batch, seqLen, vocabSize].
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 4 || shape[2] != cfg.VocabSize {
		t.Fatalf("unexpected output shape: %v, want [1, 4, %d]", shape, cfg.VocabSize)
	}

	// Check no NaN/Inf.
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

func TestBuildGPT2Graph_TiedEmbeddings(t *testing.T) {
	cfg := gpt2BaseConfig()
	tensors := makeGPT2TestTensors(cfg)

	// Verify output.weight is absent (tied to token_embd.weight).
	if _, ok := tensors["output.weight"]; ok {
		t.Fatal("test tensors should not have output.weight for tied embedding test")
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGPT2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGPT2Graph with tied embeddings: %v", err)
	}

	input, err := tensor.New([]int{1, 2}, []float32{1, 2})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[2] != cfg.VocabSize {
		t.Fatalf("unexpected output shape with tied embeddings: %v, want [1, 2, %d]", shape, cfg.VocabSize)
	}

	// Verify output is non-NaN.
	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d with tied embeddings", i)
		}
	}
}

func TestBuildGPT2Graph_UntiedEmbeddings(t *testing.T) {
	cfg := gpt2BaseConfig()
	tensors := makeGPT2TestTensors(cfg)

	// Add explicit output.weight (untied).
	size := cfg.VocabSize * cfg.HiddenSize
	outData := make([]float32, size)
	for i := range outData {
		outData[i] = 0.01 * float32(math.Cos(float64(i)*0.03))
	}
	outW, _ := tensor.New([]int{cfg.VocabSize, cfg.HiddenSize}, outData)
	tensors["output.weight"] = outW

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildGPT2Graph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildGPT2Graph with untied embeddings: %v", err)
	}

	input, err := tensor.New([]int{1, 2}, []float32{1, 2})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[2] != cfg.VocabSize {
		t.Fatalf("unexpected output shape: %v", shape)
	}
}

func TestBuildGPT2Graph_MissingTensor(t *testing.T) {
	cfg := gpt2BaseConfig()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := buildGPT2Graph(tensors, cfg, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestBuildGPT2Graph_Registered(t *testing.T) {
	builder, ok := GetArchitecture("gpt2")
	if !ok {
		t.Fatal("gpt2 architecture not registered")
	}
	if builder == nil {
		t.Fatal("gpt2 builder is nil")
	}
}

func TestGPT2EmbeddingNode_PositionReset(t *testing.T) {
	hidden := 4
	vocabSize := 8
	maxPos := 16

	tokenData := make([]float32, vocabSize*hidden)
	for i := range tokenData {
		tokenData[i] = float32(i) * 0.1
	}
	tokenW, _ := tensor.New([]int{vocabSize, hidden}, tokenData)

	posData := make([]float32, maxPos*hidden)
	for i := range posData {
		posData[i] = float32(i) * 0.01
	}
	posW, _ := tensor.New([]int{maxPos, hidden}, posData)

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := &gpt2EmbeddingNode[float32]{
		engine:      engine,
		tokenWeight: tokenW,
		posWeight:   posW,
	}

	ctx := context.Background()

	// First forward: positions 0,1.
	input1, _ := tensor.New([]int{1, 2}, []float32{0, 1})
	out1, err := node.Forward(ctx, input1)
	if err != nil {
		t.Fatalf("first forward: %v", err)
	}
	if node.posOffset != 2 {
		t.Fatalf("posOffset after first forward: got %d, want 2", node.posOffset)
	}

	// Second forward (decode): position 2.
	input2, _ := tensor.New([]int{1, 1}, []float32{2})
	out2, err := node.Forward(ctx, input2)
	if err != nil {
		t.Fatalf("second forward: %v", err)
	}
	if node.posOffset != 3 {
		t.Fatalf("posOffset after second forward: got %d, want 3", node.posOffset)
	}
	_ = out2

	// Reset.
	node.Reset()
	if node.posOffset != 0 {
		t.Fatalf("posOffset after reset: got %d, want 0", node.posOffset)
	}

	// Third forward after reset: positions 0,1 again.
	out3, err := node.Forward(ctx, input1)
	if err != nil {
		t.Fatalf("forward after reset: %v", err)
	}

	// Output after reset should match first forward.
	data1 := out1.Data()
	data3 := out3.Data()
	for i := range data1 {
		if data1[i] != data3[i] {
			t.Fatalf("output mismatch after reset at %d: %f != %f", i, data1[i], data3[i])
		}
	}
}

func TestSplitQKV(t *testing.T) {
	numHeads := 2
	headDim := 4
	hidden := numHeads * headDim

	// Merged QKV: [3*hidden, hidden] = [24, 8]
	totalRows := 3 * hidden
	data := make([]float32, totalRows*hidden)
	for i := range data {
		data[i] = float32(i)
	}
	qkv, _ := tensor.New([]int{totalRows, hidden}, data)

	q, k, v, err := splitQKV(qkv, numHeads, numHeads, headDim)
	if err != nil {
		t.Fatalf("splitQKV: %v", err)
	}

	// Check shapes.
	if qs := q.Shape(); qs[0] != hidden || qs[1] != hidden {
		t.Fatalf("Q shape: %v, want [%d, %d]", qs, hidden, hidden)
	}
	if ks := k.Shape(); ks[0] != hidden || ks[1] != hidden {
		t.Fatalf("K shape: %v, want [%d, %d]", ks, hidden, hidden)
	}
	if vs := v.Shape(); vs[0] != hidden || vs[1] != hidden {
		t.Fatalf("V shape: %v, want [%d, %d]", vs, hidden, hidden)
	}

	// Check Q contains first rows.
	qData := q.Data()
	for i := 0; i < hidden*hidden; i++ {
		if qData[i] != float32(i) {
			t.Fatalf("Q data mismatch at %d: got %f, want %f", i, qData[i], float32(i))
		}
	}
}

func TestSplitQKVBias(t *testing.T) {
	numHeads := 2
	headDim := 4
	hidden := numHeads * headDim

	data := make([]float32, 3*hidden)
	for i := range data {
		data[i] = float32(i)
	}
	qkvBias, _ := tensor.New([]int{3 * hidden}, data)

	qB, kB, vB, err := splitQKVBias(qkvBias, numHeads, numHeads, headDim)
	if err != nil {
		t.Fatalf("splitQKVBias: %v", err)
	}

	if len(qB.Data()) != hidden {
		t.Fatalf("Q bias len: %d, want %d", len(qB.Data()), hidden)
	}
	if len(kB.Data()) != hidden {
		t.Fatalf("K bias len: %d, want %d", len(kB.Data()), hidden)
	}
	if len(vB.Data()) != hidden {
		t.Fatalf("V bias len: %d, want %d", len(vB.Data()), hidden)
	}

	// Check values.
	if qB.Data()[0] != 0 {
		t.Fatalf("Q bias[0]: %f, want 0", qB.Data()[0])
	}
	if kB.Data()[0] != float32(hidden) {
		t.Fatalf("K bias[0]: %f, want %f", kB.Data()[0], float32(hidden))
	}
	if vB.Data()[0] != float32(2*hidden) {
		t.Fatalf("V bias[0]: %f, want %f", vB.Data()[0], float32(2*hidden))
	}
}
