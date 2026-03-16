package inference

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/float16"
)

// writeTestGGUF creates a minimal GGUF file for testing LoadFile.
// It builds a tiny Llama-architecture model with the full GGUF structure:
// header, metadata (including tokenizer), tensor info, and tensor data.
func writeTestGGUF(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "test.gguf")
	f, err := os.Create(path) //nolint:gosec // test file in temp dir
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	w := &ggufWriter{f: f, t: t}

	// Header: magic, version, tensor count, metadata KV count.
	hidden := 16
	inter := 32
	vocab := 32
	numLayers := 1
	numHeads := 4
	numKVHeads := 2
	kvDim := (hidden / numHeads) * numKVHeads

	// Pre-compute tensor info for building the header.
	type tensorDef struct {
		name  string
		shape []uint64
	}
	// GGUF stores dimensions in GGML order (innermost-first: ne[0]=columns,
	// ne[1]=rows). The loader reverses these to PyTorch convention.
	tensors := []tensorDef{
		{"token_embd.weight", []uint64{uint64(hidden), uint64(vocab)}},
		{"output_norm.weight", []uint64{uint64(hidden)}},
		{"output.weight", []uint64{uint64(hidden), uint64(vocab)}},
		{"blk.0.attn_norm.weight", []uint64{uint64(hidden)}},
		{"blk.0.attn_q.weight", []uint64{uint64(hidden), uint64(hidden)}},
		{"blk.0.attn_k.weight", []uint64{uint64(hidden), uint64(kvDim)}},
		{"blk.0.attn_v.weight", []uint64{uint64(hidden), uint64(kvDim)}},
		{"blk.0.attn_output.weight", []uint64{uint64(hidden), uint64(hidden)}},
		{"blk.0.ffn_norm.weight", []uint64{uint64(hidden)}},
		{"blk.0.ffn_gate.weight", []uint64{uint64(hidden), uint64(inter)}},
		{"blk.0.ffn_up.weight", []uint64{uint64(hidden), uint64(inter)}},
		{"blk.0.ffn_down.weight", []uint64{uint64(inter), uint64(hidden)}},
	}

	// Build tokenizer tokens array.
	tokStrings := make([]string, vocab)
	tokStrings[0] = "<unk>"
	tokStrings[1] = "<s>"
	tokStrings[2] = "</s>"
	for i := 3; i < vocab; i++ {
		tokStrings[i] = string(rune('a' + i - 3))
	}

	// Count metadata KVs.
	metadataCount := 11 // arch, name, vocab_size, embedding_length, block_count,
	// head_count, head_count_kv, feed_forward_length, context_length,
	// rope.freq_base, tokenizer model
	metadataCount += 4 // tokenizer tokens, bos, eos, unk
	metadataCount++    // tokenizer merges (empty)

	// Write header.
	w.writeUint32(0x46554747) // Magic "GGUF" in little-endian
	w.writeUint32(3)          // Version
	w.writeUint64(uint64(len(tensors)))
	w.writeUint64(uint64(metadataCount))

	// Write metadata.
	w.writeStringKV("general.architecture", "llama")
	w.writeStringKV("general.name", "test-llama")
	w.writeUint32KV("llama.vocab_size", uint32(vocab))
	w.writeUint32KV("llama.embedding_length", uint32(hidden))
	w.writeUint32KV("llama.block_count", uint32(numLayers))
	w.writeUint32KV("llama.attention.head_count", uint32(numHeads))
	w.writeUint32KV("llama.attention.head_count_kv", uint32(numKVHeads))
	w.writeUint32KV("llama.feed_forward_length", uint32(inter))
	w.writeUint32KV("llama.context_length", uint32(64))
	w.writeFloat32KV("llama.rope.freq_base", 10000.0)
	w.writeStringKV("tokenizer.ggml.model", "gpt2")
	w.writeStringArrayKV("tokenizer.ggml.tokens", tokStrings)
	w.writeStringArrayKV("tokenizer.ggml.merges", nil)
	w.writeUint32KV("tokenizer.ggml.bos_token_id", 1)
	w.writeUint32KV("tokenizer.ggml.eos_token_id", 2)
	w.writeUint32KV("tokenizer.ggml.unknown_token_id", 0)

	// Compute tensor data offsets.
	offsets := make([]uint64, len(tensors))
	var currentOffset uint64
	for i, td := range tensors {
		offsets[i] = currentOffset
		numElements := uint64(1)
		for _, d := range td.shape {
			numElements *= d
		}
		currentOffset += numElements * 4 // float32
	}

	// Write tensor info.
	for i, td := range tensors {
		w.writeGGUFString(td.name)
		w.writeUint32(uint32(len(td.shape)))
		for _, d := range td.shape {
			w.writeUint64(d)
		}
		w.writeUint32(0) // GGMLTypeF32
		w.writeUint64(offsets[i])
	}

	// Align to 32 bytes.
	pos, _ := f.Seek(0, 1)
	padding := (32 - pos%32) % 32
	if padding > 0 {
		pad := make([]byte, padding)
		_, _ = f.Write(pad)
	}

	// Write tensor data: small deterministic values.
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

// ggufWriter helps write GGUF binary format.
type ggufWriter struct {
	f *os.File
	t *testing.T
}

func (w *ggufWriter) writeUint32(v uint32) {
	if err := binary.Write(w.f, binary.LittleEndian, v); err != nil {
		w.t.Fatalf("write uint32: %v", err)
	}
}

func (w *ggufWriter) writeUint64(v uint64) {
	if err := binary.Write(w.f, binary.LittleEndian, v); err != nil {
		w.t.Fatalf("write uint64: %v", err)
	}
}

func (w *ggufWriter) writeFloat32Raw(v float32) {
	if err := binary.Write(w.f, binary.LittleEndian, v); err != nil {
		w.t.Fatalf("write float32: %v", err)
	}
}

func (w *ggufWriter) writeGGUFString(s string) {
	w.writeUint64(uint64(len(s)))
	if _, err := w.f.Write([]byte(s)); err != nil {
		w.t.Fatalf("write string bytes: %v", err)
	}
}

func (w *ggufWriter) writeStringKV(key, value string) {
	w.writeGGUFString(key)
	w.writeUint32(8) // TypeString
	w.writeGGUFString(value)
}

func (w *ggufWriter) writeUint32KV(key string, value uint32) {
	w.writeGGUFString(key)
	w.writeUint32(4) // TypeUint32
	w.writeUint32(value)
}

func (w *ggufWriter) writeFloat32KV(key string, value float32) {
	w.writeGGUFString(key)
	w.writeUint32(6) // TypeFloat32
	w.writeFloat32Raw(value)
}

func (w *ggufWriter) writeStringArrayKV(key string, values []string) {
	w.writeGGUFString(key)
	w.writeUint32(9) // TypeArray
	w.writeUint32(8) // TypeString elements
	w.writeUint64(uint64(len(values)))
	for _, s := range values {
		w.writeGGUFString(s)
	}
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
	path := filepath.Join(dir, "test_fp16_gqa.gguf")
	f, err := os.Create(path) //nolint:gosec // test file in temp dir
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	w := &ggufWriter{f: f, t: t}

	// GQA config: 4 query heads, 1 KV head (ratio = 4).
	hidden := 16
	inter := 32
	vocab := 32
	numLayers := 1
	numHeads := 4
	numKVHeads := 1
	headDim := hidden / numHeads // 4
	kvDim := headDim * numKVHeads // 4

	type tensorDef struct {
		name     string
		shape    []uint64
		ggmlType uint32 // 0=F32, 1=F16
	}
	// GGUF stores dimensions in GGML order (innermost-first).
	tensors := []tensorDef{
		{"token_embd.weight", []uint64{uint64(hidden), uint64(vocab)}, 0},
		{"output_norm.weight", []uint64{uint64(hidden)}, 0},
		{"output.weight", []uint64{uint64(hidden), uint64(vocab)}, 0},
		{"blk.0.attn_norm.weight", []uint64{uint64(hidden)}, 0},
		// These are the FP16 GQA weight tensors:
		{"blk.0.attn_q.weight", []uint64{uint64(hidden), uint64(hidden)}, 1},         // [hidden, hidden] -> [hidden, hidden]
		{"blk.0.attn_k.weight", []uint64{uint64(hidden), uint64(kvDim)}, 1},           // [hidden, kvDim] -> [kvDim, hidden]
		{"blk.0.attn_v.weight", []uint64{uint64(hidden), uint64(kvDim)}, 1},           // [hidden, kvDim] -> [kvDim, hidden]
		{"blk.0.attn_output.weight", []uint64{uint64(hidden), uint64(hidden)}, 1},     // [hidden, hidden] -> [hidden, hidden]
		{"blk.0.ffn_norm.weight", []uint64{uint64(hidden)}, 0},
		{"blk.0.ffn_gate.weight", []uint64{uint64(hidden), uint64(inter)}, 1},
		{"blk.0.ffn_up.weight", []uint64{uint64(hidden), uint64(inter)}, 1},
		{"blk.0.ffn_down.weight", []uint64{uint64(inter), uint64(hidden)}, 1},
	}

	tokStrings := make([]string, vocab)
	tokStrings[0] = "<unk>"
	tokStrings[1] = "<s>"
	tokStrings[2] = "</s>"
	for i := 3; i < vocab; i++ {
		tokStrings[i] = string(rune('a' + i - 3))
	}

	metadataCount := 11 + 4 + 1

	w.writeUint32(0x46554747) // Magic
	w.writeUint32(3)          // Version
	w.writeUint64(uint64(len(tensors)))
	w.writeUint64(uint64(metadataCount))

	w.writeStringKV("general.architecture", "llama")
	w.writeStringKV("general.name", "test-llama-fp16-gqa")
	w.writeUint32KV("llama.vocab_size", uint32(vocab))
	w.writeUint32KV("llama.embedding_length", uint32(hidden))
	w.writeUint32KV("llama.block_count", uint32(numLayers))
	w.writeUint32KV("llama.attention.head_count", uint32(numHeads))
	w.writeUint32KV("llama.attention.head_count_kv", uint32(numKVHeads))
	w.writeUint32KV("llama.feed_forward_length", uint32(inter))
	w.writeUint32KV("llama.context_length", uint32(64))
	w.writeFloat32KV("llama.rope.freq_base", 10000.0)
	w.writeStringKV("tokenizer.ggml.model", "gpt2")
	w.writeStringArrayKV("tokenizer.ggml.tokens", tokStrings)
	w.writeStringArrayKV("tokenizer.ggml.merges", nil)
	w.writeUint32KV("tokenizer.ggml.bos_token_id", 1)
	w.writeUint32KV("tokenizer.ggml.eos_token_id", 2)
	w.writeUint32KV("tokenizer.ggml.unknown_token_id", 0)

	// Compute tensor data offsets.
	offsets := make([]uint64, len(tensors))
	var currentOffset uint64
	for i, td := range tensors {
		offsets[i] = currentOffset
		numElements := uint64(1)
		for _, d := range td.shape {
			numElements *= d
		}
		bytesPerElem := uint64(4) // F32
		if td.ggmlType == 1 {
			bytesPerElem = 2 // F16
		}
		currentOffset += numElements * bytesPerElem
	}

	// Write tensor info.
	for i, td := range tensors {
		w.writeGGUFString(td.name)
		w.writeUint32(uint32(len(td.shape)))
		for _, d := range td.shape {
			w.writeUint64(d)
		}
		w.writeUint32(td.ggmlType)
		w.writeUint64(offsets[i])
	}

	// Align to 32 bytes.
	pos, _ := f.Seek(0, 1)
	padding := (32 - pos%32) % 32
	if padding > 0 {
		pad := make([]byte, padding)
		_, _ = f.Write(pad)
	}

	// Write tensor data.
	for _, td := range tensors {
		numElements := uint64(1)
		for _, d := range td.shape {
			numElements *= d
		}
		for j := range numElements {
			val := float32(math.Sin(float64(j)*0.01)) * 0.02
			if td.ggmlType == 1 {
				w.writeFloat16Raw(val)
			} else {
				w.writeFloat32Raw(val)
			}
		}
	}

	return path
}

func (w *ggufWriter) writeFloat16Raw(v float32) {
	f16 := float16.FromFloat32(v)
	if err := binary.Write(w.f, binary.LittleEndian, f16.Bits()); err != nil {
		w.t.Fatalf("write float16: %v", err)
	}
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
