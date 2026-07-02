package inference

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/zerfoo/model/gguf"
)

func TestBuildPhiGraph_Builds(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           2,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		PartialRotaryFactor: 0.5,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildPhiGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildPhiGraph: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildPhiGraph_ForwardNonNaN(t *testing.T) {
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           2,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		PartialRotaryFactor: 0.5,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildPhiGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildPhiGraph: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildPhiGraph_FullRotaryFactor(t *testing.T) {
	// PartialRotaryFactor = 1.0 should behave like full RoPE (same as Llama).
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           1,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		PartialRotaryFactor: 1.0,
	}
	tensors := makeLlamaTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildPhiGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildPhiGraph with factor=1.0: %v", err)
	}

	assertGraphForwardNonNaN(t, g, cfg.VocabSize)
}

func TestBuildPhiGraph_TiedEmbedding(t *testing.T) {
	// Phi should work without lm_head.weight (tied to embedding).
	cfg := &gguf.ModelConfig{
		Architecture:        "phi3",
		VocabSize:           32,
		HiddenSize:          16,
		NumLayers:           1,
		NumHeads:            4,
		NumKVHeads:          2,
		IntermediateSize:    32,
		MaxSeqLen:           64,
		RopeTheta:           10000.0,
		PartialRotaryFactor: 0.5,
	}
	tensors := makeLlamaTestTensors(cfg)
	delete(tensors, "lm_head.weight")
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := buildPhiGraph(tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildPhiGraph with tied embedding: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
}

func TestBuildPhiGraph_PartialRotaryFactors(t *testing.T) {
	tests := []struct {
		name       string
		factor     float32
		hiddenSize int
		numHeads   int
	}{
		{"half", 0.5, 16, 4},
		{"three_quarters", 0.75, 32, 4},
		{"full", 1.0, 16, 4},
		{"zero_defaults_to_full", 0.0, 16, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &gguf.ModelConfig{
				Architecture:        "phi3",
				VocabSize:           32,
				HiddenSize:          tt.hiddenSize,
				NumLayers:           1,
				NumHeads:            tt.numHeads,
				NumKVHeads:          2,
				IntermediateSize:    32,
				MaxSeqLen:           64,
				RopeTheta:           10000.0,
				PartialRotaryFactor: tt.factor,
			}
			tensors := makeLlamaTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, _, err := buildPhiGraph(tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildPhiGraph with factor=%v: %v", tt.factor, err)
			}

			assertGraphForwardNonNaN(t, g, cfg.VocabSize)
		})
	}
}

// writeTestGGUF_Phi creates a minimal GGUF file with phi3 architecture and
// merged QKV tensors (blk.N.attn_qkv.weight) to verify the full LoadFile
// path including QKV split.
func writeTestGGUF_Phi(t *testing.T, dir string) string {
	t.Helper()

	hidden := 16
	inter := 32
	vocab := 32
	numLayers := 1
	numHeads := 4
	numKVHeads := 2
	headDim := hidden / numHeads // 4
	ropeDimCount := headDim / 2  // partial rotary factor = 0.5

	qRows := numHeads * headDim      // 16
	kRows := numKVHeads * headDim     // 8
	vRows := numKVHeads * headDim     // 8
	qkvRows := qRows + kRows + vRows // 32

	// Row-major shapes (outermost first).
	tensors := []tensorDef{
		{"token_embd.weight", []int{vocab, hidden}, ztensorgguf.TypeF32},
		{"output_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"output.weight", []int{vocab, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_qkv.weight", []int{qkvRows, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_output.weight", []int{hidden, hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_gate.weight", []int{inter, hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_up.weight", []int{inter, hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_down.weight", []int{hidden, inter}, ztensorgguf.TypeF32},
	}

	tokStrings := make([]string, vocab)
	tokStrings[0] = "<unk>"
	tokStrings[1] = "<s>"
	tokStrings[2] = "</s>"
	for i := 3; i < vocab; i++ {
		tokStrings[i] = string(rune('a' + i - 3))
	}

	w := ztensorgguf.NewWriter()

	w.AddMetadataString("general.architecture", "phi3")
	w.AddMetadataString("general.name", "test-phi3-merged-qkv")
	w.AddMetadataUint32("phi3.vocab_size", uint32(vocab))
	w.AddMetadataUint32("phi3.embedding_length", uint32(hidden))
	w.AddMetadataUint32("phi3.block_count", uint32(numLayers))
	w.AddMetadataUint32("phi3.attention.head_count", uint32(numHeads))
	w.AddMetadataUint32("phi3.attention.head_count_kv", uint32(numKVHeads))
	w.AddMetadataUint32("phi3.feed_forward_length", uint32(inter))
	w.AddMetadataUint32("phi3.context_length", uint32(64))
	w.AddMetadataFloat32("phi3.rope.freq_base", 10000.0)
	w.AddMetadataUint32("phi3.rope.dimension_count", uint32(ropeDimCount))
	w.AddMetadataString("tokenizer.ggml.model", "gpt2")
	w.AddMetadataStringArray("tokenizer.ggml.tokens", tokStrings)
	w.AddMetadataStringArray("tokenizer.ggml.merges", nil)
	w.AddMetadataUint32("tokenizer.ggml.bos_token_id", 1)
	w.AddMetadataUint32("tokenizer.ggml.eos_token_id", 2)
	w.AddMetadataUint32("tokenizer.ggml.unknown_token_id", 0)

	for _, td := range tensors {
		n := numElements(td.shape)
		w.AddTensorF32(td.name, td.shape, generateF32Data(n))
	}

	path := filepath.Join(dir, "test_phi.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	if err := w.Write(f); err != nil {
		t.Fatalf("write GGUF: %v", err)
	}

	return path
}

func TestLoadFile_Phi_MergedQKV(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF_Phi(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile Phi (merged QKV): %v", err)
	}
	defer func() { _ = m.Close() }()

	cfg := m.Config()
	if cfg.Architecture != "phi3" {
		t.Errorf("Architecture = %q, want %q", cfg.Architecture, "phi3")
	}
	if cfg.NumQueryHeads != 4 {
		t.Errorf("NumQueryHeads = %d, want 4", cfg.NumQueryHeads)
	}
	if cfg.NumKeyValueHeads != 2 {
		t.Errorf("NumKeyValueHeads = %d, want 2", cfg.NumKeyValueHeads)
	}
	if cfg.HiddenSize != 16 {
		t.Errorf("HiddenSize = %d, want 16", cfg.HiddenSize)
	}
}

func TestLoadFile_Phi_MergedQKV_Generate(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF_Phi(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile Phi: %v", err)
	}
	defer func() { _ = m.Close() }()

	result, err := m.Generate(t.Context(), "hello", WithMaxTokens(3), WithTemperature(0))
	if err != nil {
		t.Fatalf("Generate Phi: %v", err)
	}
	if result == "" {
		t.Error("Generate returned empty string")
	}
}

func TestBuildArchGraph_Phi(t *testing.T) {
	tests := []struct {
		arch string
	}{
		{"phi3"},
		{"phi"},
	}

	for _, tt := range tests {
		t.Run(tt.arch, func(t *testing.T) {
			cfg := &gguf.ModelConfig{
				Architecture:        tt.arch,
				VocabSize:           32,
				HiddenSize:          16,
				NumLayers:           1,
				NumHeads:            4,
				NumKVHeads:          2,
				IntermediateSize:    32,
				MaxSeqLen:           64,
				RopeTheta:           10000.0,
				PartialRotaryFactor: 0.5,
			}
			tensors := makeLlamaTestTensors(cfg)
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			g, emb, err := buildArchGraph(tt.arch, tensors, cfg, engine)
			if err != nil {
				t.Fatalf("buildArchGraph(%q): %v", tt.arch, err)
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
