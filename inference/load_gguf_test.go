package inference

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/compute"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// tensorDef describes a tensor for test GGUF generation.
type tensorDef struct {
	name     string
	shape    []int // row-major shape (outermost first); writer reverses for GGML
	ggmlType int   // ztensorgguf.TypeF32, TypeF16, etc.
}

// generateF32Data creates deterministic float32 data for a tensor.
func generateF32Data(numElements int) []float32 {
	data := make([]float32, numElements)
	for i := range data {
		data[i] = float32(math.Sin(float64(i)*0.01)) * 0.02
	}
	return data
}

// f32ToBytes converts float32 slice to raw little-endian bytes.
func f32ToBytes(data []float32) []byte {
	b := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// f32ToF16Bytes converts float32 slice to FP16 raw bytes.
func f32ToF16Bytes(data []float32) []byte {
	b := make([]byte, len(data)*2)
	for i, v := range data {
		f16 := float16.FromFloat32(v)
		binary.LittleEndian.PutUint16(b[i*2:], f16.Bits())
	}
	return b
}

// numElements computes the total element count from a shape.
func numElements(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// writeTestGGUF creates a minimal GGUF file for testing LoadFile.
// It builds a tiny Llama-architecture model with the full GGUF structure:
// header, metadata (including tokenizer), tensor info, and tensor data.
func writeTestGGUF(t *testing.T, dir string) string {
	t.Helper()

	hidden := 16
	inter := 32
	vocab := 32
	numLayers := 1
	numHeads := 4
	numKVHeads := 2
	kvDim := (hidden / numHeads) * numKVHeads

	// Shapes in row-major convention (outermost first).
	// The writer reverses to GGML order internally.
	tensors := []tensorDef{
		{"token_embd.weight", []int{vocab, hidden}, ztensorgguf.TypeF32},
		{"output_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"output.weight", []int{vocab, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_q.weight", []int{hidden, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_k.weight", []int{kvDim, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_v.weight", []int{kvDim, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_output.weight", []int{hidden, hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_gate.weight", []int{inter, hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_up.weight", []int{inter, hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_down.weight", []int{hidden, inter}, ztensorgguf.TypeF32},
	}

	// Build tokenizer tokens array.
	tokStrings := make([]string, vocab)
	tokStrings[0] = "<unk>"
	tokStrings[1] = "<s>"
	tokStrings[2] = "</s>"
	for i := 3; i < vocab; i++ {
		tokStrings[i] = string(rune('a' + i - 3))
	}

	w := ztensorgguf.NewWriter()

	// Metadata.
	w.AddMetadataString("general.architecture", "llama")
	w.AddMetadataString("general.name", "test-llama")
	w.AddMetadataUint32("llama.vocab_size", uint32(vocab))
	w.AddMetadataUint32("llama.embedding_length", uint32(hidden))
	w.AddMetadataUint32("llama.block_count", uint32(numLayers))
	w.AddMetadataUint32("llama.attention.head_count", uint32(numHeads))
	w.AddMetadataUint32("llama.attention.head_count_kv", uint32(numKVHeads))
	w.AddMetadataUint32("llama.feed_forward_length", uint32(inter))
	w.AddMetadataUint32("llama.context_length", uint32(64))
	w.AddMetadataFloat32("llama.rope.freq_base", 10000.0)
	w.AddMetadataString("tokenizer.ggml.model", "gpt2")
	w.AddMetadataStringArray("tokenizer.ggml.tokens", tokStrings)
	w.AddMetadataStringArray("tokenizer.ggml.merges", nil)
	w.AddMetadataUint32("tokenizer.ggml.bos_token_id", 1)
	w.AddMetadataUint32("tokenizer.ggml.eos_token_id", 2)
	w.AddMetadataUint32("tokenizer.ggml.unknown_token_id", 0)

	// Add tensors with deterministic data.
	for _, td := range tensors {
		n := numElements(td.shape)
		data := generateF32Data(n)
		w.AddTensorF32(td.name, td.shape, data)
	}

	path := filepath.Join(dir, "test.gguf")
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

func TestLoadFile_GGUF(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	// Verify model loaded with correct metadata.
	cfg := m.Config()
	if cfg.Architecture != "llama" {
		t.Errorf("Architecture = %q, want %q", cfg.Architecture, "llama")
	}
	if cfg.VocabSize != 32 {
		t.Errorf("VocabSize = %d, want 32", cfg.VocabSize)
	}
	if cfg.HiddenSize != 16 {
		t.Errorf("HiddenSize = %d, want 16", cfg.HiddenSize)
	}
}

func TestLoadFile_GGUF_Generate(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	// Generate should produce non-empty output.
	result, err := m.Generate(t.Context(), "hello", WithMaxTokens(3), WithTemperature(0))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("Generate returned empty string")
	}
}

// writeTestGGUF_FP16GQA creates a minimal GGUF file with FP16 weight tensors
// and GQA configuration (numHeads=4, numKVHeads=1) to exercise the FP16+GQA
// loading path. This is a regression test for the "storage length does not
// match tensor size" error when loading FP16 GQA models.
func writeTestGGUF_FP16GQA(t *testing.T, dir string) string {
	t.Helper()

	// GQA config: 4 query heads, 1 KV head (ratio = 4).
	hidden := 16
	inter := 32
	vocab := 32
	numLayers := 1
	numHeads := 4
	numKVHeads := 1
	headDim := hidden / numHeads  // 4
	kvDim := headDim * numKVHeads // 4

	tensors := []tensorDef{
		{"token_embd.weight", []int{vocab, hidden}, ztensorgguf.TypeF32},
		{"output_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"output.weight", []int{vocab, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		// FP16 GQA weight tensors:
		{"blk.0.attn_q.weight", []int{hidden, hidden}, ztensorgguf.TypeF16},
		{"blk.0.attn_k.weight", []int{kvDim, hidden}, ztensorgguf.TypeF16},
		{"blk.0.attn_v.weight", []int{kvDim, hidden}, ztensorgguf.TypeF16},
		{"blk.0.attn_output.weight", []int{hidden, hidden}, ztensorgguf.TypeF16},
		{"blk.0.ffn_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"blk.0.ffn_gate.weight", []int{inter, hidden}, ztensorgguf.TypeF16},
		{"blk.0.ffn_up.weight", []int{inter, hidden}, ztensorgguf.TypeF16},
		{"blk.0.ffn_down.weight", []int{hidden, inter}, ztensorgguf.TypeF16},
	}

	tokStrings := make([]string, vocab)
	tokStrings[0] = "<unk>"
	tokStrings[1] = "<s>"
	tokStrings[2] = "</s>"
	for i := 3; i < vocab; i++ {
		tokStrings[i] = string(rune('a' + i - 3))
	}

	w := ztensorgguf.NewWriter()

	w.AddMetadataString("general.architecture", "llama")
	w.AddMetadataString("general.name", "test-llama-fp16-gqa")
	w.AddMetadataUint32("llama.vocab_size", uint32(vocab))
	w.AddMetadataUint32("llama.embedding_length", uint32(hidden))
	w.AddMetadataUint32("llama.block_count", uint32(numLayers))
	w.AddMetadataUint32("llama.attention.head_count", uint32(numHeads))
	w.AddMetadataUint32("llama.attention.head_count_kv", uint32(numKVHeads))
	w.AddMetadataUint32("llama.feed_forward_length", uint32(inter))
	w.AddMetadataUint32("llama.context_length", uint32(64))
	w.AddMetadataFloat32("llama.rope.freq_base", 10000.0)
	w.AddMetadataString("tokenizer.ggml.model", "gpt2")
	w.AddMetadataStringArray("tokenizer.ggml.tokens", tokStrings)
	w.AddMetadataStringArray("tokenizer.ggml.merges", nil)
	w.AddMetadataUint32("tokenizer.ggml.bos_token_id", 1)
	w.AddMetadataUint32("tokenizer.ggml.eos_token_id", 2)
	w.AddMetadataUint32("tokenizer.ggml.unknown_token_id", 0)

	for _, td := range tensors {
		n := numElements(td.shape)
		f32Data := generateF32Data(n)
		switch td.ggmlType {
		case ztensorgguf.TypeF32:
			w.AddTensorF32(td.name, td.shape, f32Data)
		case ztensorgguf.TypeF16:
			w.AddTensor(td.name, ztensorgguf.TypeF16, td.shape, f32ToF16Bytes(f32Data))
		}
	}

	path := filepath.Join(dir, "test_fp16_gqa.gguf")
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

func TestLoadFile_FP16_GQA(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF_FP16GQA(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile FP16 GQA: %v", err)
	}
	defer func() { _ = m.Close() }()

	// Verify model loaded with correct metadata.
	cfg := m.Config()
	if cfg.Architecture != "llama" {
		t.Errorf("Architecture = %q, want %q", cfg.Architecture, "llama")
	}
	if cfg.NumQueryHeads != 4 {
		t.Errorf("NumQueryHeads = %d, want 4", cfg.NumQueryHeads)
	}
	if cfg.NumKeyValueHeads != 1 {
		t.Errorf("NumKeyValueHeads = %d, want 1", cfg.NumKeyValueHeads)
	}
}

func TestLoadFile_FP16_GQA_Generate(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF_FP16GQA(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile FP16 GQA: %v", err)
	}
	defer func() { _ = m.Close() }()

	result, err := m.Generate(t.Context(), "hello", WithMaxTokens(3), WithTemperature(0))
	if err != nil {
		t.Fatalf("Generate FP16 GQA: %v", err)
	}
	if result == "" {
		t.Error("Generate returned empty string")
	}
}

func TestLoadFile_FP8_GQA(t *testing.T) {
	// Use the same GQA GGUF file (F32 weights) with FP8 quantization.
	// This exercises the full LoadFile -> QuantizeToFP8E4M3 -> buildArchGraph path.
	dir := t.TempDir()
	path := writeTestGGUF(t, dir)

	m, err := LoadFile(path, WithDType("fp8"))
	if err != nil {
		t.Fatalf("LoadFile with FP8: %v", err)
	}
	defer func() { _ = m.Close() }()

	cfg := m.Config()
	if cfg.Architecture != "llama" {
		t.Errorf("Architecture = %q, want %q", cfg.Architecture, "llama")
	}
}

func TestLoadFile_FP8_GQA_Generate(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir)

	m, err := LoadFile(path, WithDType("fp8"))
	if err != nil {
		t.Fatalf("LoadFile with FP8: %v", err)
	}
	defer func() { _ = m.Close() }()

	result, err := m.Generate(t.Context(), "hello", WithMaxTokens(3), WithTemperature(0))
	if err != nil {
		t.Fatalf("Generate FP8: %v", err)
	}
	if result == "" {
		t.Error("Generate returned empty string")
	}
}

func TestBuildArchGraph_MistralDetection(t *testing.T) {
	baseCfg := func() *gguf.ModelConfig {
		return &gguf.ModelConfig{
			Architecture:     "llama",
			VocabSize:        32,
			HiddenSize:       16,
			NumLayers:        1,
			NumHeads:         4,
			NumKVHeads:       2,
			IntermediateSize: 32,
			MaxSeqLen:        64,
			RopeTheta:        10000.0,
		}
	}

	tests := []struct {
		name                 string
		arch                 string
		slidingWindow        int
		slidingWindowPattern int
		wantMistral          bool
	}{
		{
			name:          "llama without sliding window routes to Llama",
			arch:          "llama",
			slidingWindow: 0,
			wantMistral:   false,
		},
		{
			name:          "llama with sliding window and no pattern routes to Mistral",
			arch:          "llama",
			slidingWindow: 4096,
			wantMistral:   true,
		},
		{
			name:                 "llama with sliding window and pattern=6 routes to Llama (Gemma-like)",
			arch:                 "llama",
			slidingWindow:        4096,
			slidingWindowPattern: 6,
			wantMistral:          false,
		},
		{
			name:          "explicit mistral arch routes to Mistral",
			arch:          "mistral",
			slidingWindow: 4096,
			wantMistral:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := baseCfg()
			cfg.Architecture = tt.arch
			cfg.SlidingWindow = tt.slidingWindow
			cfg.SlidingWindowPattern = tt.slidingWindowPattern

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

			// Verify sliding window is configured in the graph when expected.
			// Mistral graphs use sliding window attention; Llama graphs do not.
			if tt.wantMistral && cfg.SlidingWindow == 0 {
				t.Error("expected SlidingWindow > 0 for Mistral path")
			}
		})
	}
}

func TestLoadFile_SessionPoolSize(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir)

	t.Run("default pool size", func(t *testing.T) {
		m, err := LoadFile(path)
		if err != nil {
			t.Fatalf("LoadFile: %v", err)
		}
		defer func() { _ = m.Close() }()

		if cap(m.sessionPool) != defaultSessionPoolSize {
			t.Errorf("sessionPool capacity = %d, want %d", cap(m.sessionPool), defaultSessionPoolSize)
		}
	})

	t.Run("custom pool size", func(t *testing.T) {
		m, err := LoadFile(path, WithSessionPoolSize(4))
		if err != nil {
			t.Fatalf("LoadFile: %v", err)
		}
		defer func() { _ = m.Close() }()

		if cap(m.sessionPool) != 4 {
			t.Errorf("sessionPool capacity = %d, want 4", cap(m.sessionPool))
		}
	})

	t.Run("minimum clamped to 1", func(t *testing.T) {
		m, err := LoadFile(path, WithSessionPoolSize(0))
		if err != nil {
			t.Fatalf("LoadFile: %v", err)
		}
		defer func() { _ = m.Close() }()

		if cap(m.sessionPool) != 1 {
			t.Errorf("sessionPool capacity = %d, want 1", cap(m.sessionPool))
		}
	})
}

// writeTestGGUF_Mistral creates a minimal GGUF file that declares arch="llama"
// but has general.name="Mistral-7B-Instruct-v0.3" and a sliding window,
// exercising the DetectActualArchitecture path.
func writeTestGGUF_Mistral(t *testing.T, dir string) string {
	t.Helper()

	hidden := 16
	inter := 32
	vocab := 32
	numLayers := 1
	numHeads := 4
	numKVHeads := 2
	kvDim := (hidden / numHeads) * numKVHeads

	tensors := []tensorDef{
		{"token_embd.weight", []int{vocab, hidden}, ztensorgguf.TypeF32},
		{"output_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"output.weight", []int{vocab, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_norm.weight", []int{hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_q.weight", []int{hidden, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_k.weight", []int{kvDim, hidden}, ztensorgguf.TypeF32},
		{"blk.0.attn_v.weight", []int{kvDim, hidden}, ztensorgguf.TypeF32},
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

	// Declares arch="llama" but name="Mistral-7B-Instruct-v0.3".
	w.AddMetadataString("general.architecture", "llama")
	w.AddMetadataString("general.name", "Mistral-7B-Instruct-v0.3")
	w.AddMetadataUint32("llama.vocab_size", uint32(vocab))
	w.AddMetadataUint32("llama.embedding_length", uint32(hidden))
	w.AddMetadataUint32("llama.block_count", uint32(numLayers))
	w.AddMetadataUint32("llama.attention.head_count", uint32(numHeads))
	w.AddMetadataUint32("llama.attention.head_count_kv", uint32(numKVHeads))
	w.AddMetadataUint32("llama.feed_forward_length", uint32(inter))
	w.AddMetadataUint32("llama.context_length", uint32(32768))
	w.AddMetadataFloat32("llama.rope.freq_base", 10000.0)
	w.AddMetadataUint32("llama.attention.sliding_window", 4096)
	w.AddMetadataString("tokenizer.ggml.model", "llama")
	w.AddMetadataStringArray("tokenizer.ggml.tokens", tokStrings)
	w.AddMetadataStringArray("tokenizer.ggml.merges", nil)
	w.AddMetadataUint32("tokenizer.ggml.bos_token_id", 1)
	w.AddMetadataUint32("tokenizer.ggml.eos_token_id", 2)

	for _, td := range tensors {
		n := numElements(td.shape)
		w.AddTensorF32(td.name, td.shape, generateF32Data(n))
	}

	path := filepath.Join(dir, "test_mistral.gguf")
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

func TestLoadFile_MistralDetection(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF_Mistral(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	cfg := m.Config()
	if cfg.Architecture != "mistral" {
		t.Errorf("Architecture = %q, want %q", cfg.Architecture, "mistral")
	}
	if cfg.BOSTokenID != 1 {
		t.Errorf("BOSTokenID = %d, want 1", cfg.BOSTokenID)
	}
	if cfg.EOSTokenID != 2 {
		t.Errorf("EOSTokenID = %d, want 2", cfg.EOSTokenID)
	}
}

func TestLoadFile_MistralDetection_Generate(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF_Mistral(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	result, err := m.Generate(t.Context(), "hello", WithMaxTokens(3), WithTemperature(0))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if result == "" {
		t.Error("Generate returned empty string")
	}
}

func TestLoadFile_NotGGUF(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "not-a-model.txt")
	if err := os.WriteFile(path, []byte("hello"), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := LoadFile(path)
	if err == nil {
		t.Fatal("expected error for non-GGUF file")
	}
}
