package inference

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/numeric"
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
	path := filepath.Join(dir, "test_phi.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	w := &ggufWriter{f: f, t: t}

	hidden := 16
	inter := 32
	vocab := 32
	numLayers := 1
	numHeads := 4
	numKVHeads := 2
	headDim := hidden / numHeads // 4
	ropeDimCount := headDim / 2  // partial rotary factor = 0.5

	qRows := numHeads * headDim                   // 16
	kRows := numKVHeads * headDim                  // 8
	vRows := numKVHeads * headDim                  // 8
	qkvRows := qRows + kRows + vRows              // 32

	type tensorDef struct {
		name  string
		shape []uint64
	}
	// GGUF dimensions in GGML order (innermost-first).
	tensors := []tensorDef{
		{"token_embd.weight", []uint64{uint64(hidden), uint64(vocab)}},
		{"output_norm.weight", []uint64{uint64(hidden)}},
		{"output.weight", []uint64{uint64(hidden), uint64(vocab)}},
		{"blk.0.attn_norm.weight", []uint64{uint64(hidden)}},
		{"blk.0.attn_qkv.weight", []uint64{uint64(hidden), uint64(qkvRows)}},
		{"blk.0.attn_output.weight", []uint64{uint64(hidden), uint64(hidden)}},
		{"blk.0.ffn_norm.weight", []uint64{uint64(hidden)}},
		{"blk.0.ffn_gate.weight", []uint64{uint64(hidden), uint64(inter)}},
		{"blk.0.ffn_up.weight", []uint64{uint64(hidden), uint64(inter)}},
		{"blk.0.ffn_down.weight", []uint64{uint64(inter), uint64(hidden)}},
	}

	tokStrings := make([]string, vocab)
	tokStrings[0] = "<unk>"
	tokStrings[1] = "<s>"
	tokStrings[2] = "</s>"
	for i := 3; i < vocab; i++ {
		tokStrings[i] = string(rune('a' + i - 3))
	}

	metadataCount := 12 + 4 + 1 // 11 base + rope.dimension_count + tokenizer(4+1)

	w.writeUint32(0x46554747) // Magic
	w.writeUint32(3)          // Version
	w.writeUint64(uint64(len(tensors)))
	w.writeUint64(uint64(metadataCount))

	w.writeStringKV("general.architecture", "phi3")
	w.writeStringKV("general.name", "test-phi3-merged-qkv")
	w.writeUint32KV("phi3.vocab_size", uint32(vocab))
	w.writeUint32KV("phi3.embedding_length", uint32(hidden))
	w.writeUint32KV("phi3.block_count", uint32(numLayers))
	w.writeUint32KV("phi3.attention.head_count", uint32(numHeads))
	w.writeUint32KV("phi3.attention.head_count_kv", uint32(numKVHeads))
	w.writeUint32KV("phi3.feed_forward_length", uint32(inter))
	w.writeUint32KV("phi3.context_length", uint32(64))
	w.writeFloat32KV("phi3.rope.freq_base", 10000.0)
	w.writeUint32KV("phi3.rope.dimension_count", uint32(ropeDimCount))
	w.writeStringKV("tokenizer.ggml.model", "gpt2")
	w.writeStringArrayKV("tokenizer.ggml.tokens", tokStrings)
	w.writeStringArrayKV("tokenizer.ggml.merges", nil)
	w.writeUint32KV("tokenizer.ggml.bos_token_id", 1)
	w.writeUint32KV("tokenizer.ggml.eos_token_id", 2)
	w.writeUint32KV("tokenizer.ggml.unknown_token_id", 0)

	offsets := make([]uint64, len(tensors))
	var currentOffset uint64
	for i, td := range tensors {
		offsets[i] = currentOffset
		numElements := uint64(1)
		for _, d := range td.shape {
			numElements *= d
		}
		currentOffset += numElements * 4
	}

	for i, td := range tensors {
		w.writeGGUFString(td.name)
		w.writeUint32(uint32(len(td.shape)))
		for _, d := range td.shape {
			w.writeUint64(d)
		}
		w.writeUint32(0) // GGMLTypeF32
		w.writeUint64(offsets[i])
	}

	pos, _ := f.Seek(0, 1)
	padding := (32 - pos%32) % 32
	if padding > 0 {
		pad := make([]byte, padding)
		_, _ = f.Write(pad)
	}

	for _, td := range tensors {
		numElements := uint64(1)
		for _, d := range td.shape {
			numElements *= d
		}
		for j := range numElements {
			val := float32(math.Sin(float64(j)*0.01)) * 0.02
			w.writeFloat32Raw(val)
		}
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
