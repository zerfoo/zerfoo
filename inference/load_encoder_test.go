package inference

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// writeTestBertGGUF creates a minimal BERT-architecture GGUF file for testing.
func writeTestBertGGUF(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "bert_test.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	w := &ggufWriter{f: f, t: t}

	hidden := 48
	inter := 96
	vocab := 32
	maxPos := 64
	numLayers := 2
	numHeads := 4
	numLabels := 3

	type tensorDef struct {
		name  string
		shape []uint64
	}

	// GGUF stores dimensions in GGML order (innermost-first).
	var tensors []tensorDef

	// Global embeddings.
	tensors = append(tensors,
		tensorDef{"token_embd.weight", []uint64{uint64(hidden), uint64(vocab)}},
		tensorDef{"position_embd.weight", []uint64{uint64(hidden), uint64(maxPos)}},
		tensorDef{"token_type_embd.weight", []uint64{uint64(hidden), 2}},
		tensorDef{"token_embd_norm.weight", []uint64{uint64(hidden)}},
		tensorDef{"token_embd_norm.bias", []uint64{uint64(hidden)}},
	)

	// Per-layer tensors.
	for i := 0; i < numLayers; i++ {
		prefix := "blk." + itoa(i) + "."
		tensors = append(tensors,
			tensorDef{prefix + "attn_q.weight", []uint64{uint64(hidden), uint64(hidden)}},
			tensorDef{prefix + "attn_q.bias", []uint64{uint64(hidden)}},
			tensorDef{prefix + "attn_k.weight", []uint64{uint64(hidden), uint64(hidden)}},
			tensorDef{prefix + "attn_k.bias", []uint64{uint64(hidden)}},
			tensorDef{prefix + "attn_v.weight", []uint64{uint64(hidden), uint64(hidden)}},
			tensorDef{prefix + "attn_v.bias", []uint64{uint64(hidden)}},
			tensorDef{prefix + "attn_output.weight", []uint64{uint64(hidden), uint64(hidden)}},
			tensorDef{prefix + "attn_output.bias", []uint64{uint64(hidden)}},
			tensorDef{prefix + "attn_norm.weight", []uint64{uint64(hidden)}},
			tensorDef{prefix + "attn_norm.bias", []uint64{uint64(hidden)}},
			tensorDef{prefix + "ffn_up.weight", []uint64{uint64(hidden), uint64(inter)}},
			tensorDef{prefix + "ffn_up.bias", []uint64{uint64(inter)}},
			tensorDef{prefix + "ffn_down.weight", []uint64{uint64(inter), uint64(hidden)}},
			tensorDef{prefix + "ffn_down.bias", []uint64{uint64(hidden)}},
			tensorDef{prefix + "ffn_norm.weight", []uint64{uint64(hidden)}},
			tensorDef{prefix + "ffn_norm.bias", []uint64{uint64(hidden)}},
		)
	}

	// Pooler (CLS token projection + tanh).
	tensors = append(tensors,
		tensorDef{"cls_pooler.weight", []uint64{uint64(hidden), uint64(hidden)}},
		tensorDef{"cls_pooler.bias", []uint64{uint64(hidden)}},
	)

	// Classification head.
	tensors = append(tensors,
		tensorDef{"cls.weight", []uint64{uint64(hidden), uint64(numLabels)}},
		tensorDef{"cls.bias", []uint64{uint64(numLabels)}},
	)

	// Build tokenizer tokens array.
	tokStrings := make([]string, vocab)
	tokStrings[0] = "<unk>"
	tokStrings[1] = "<s>"
	tokStrings[2] = "</s>"
	for i := 3; i < vocab; i++ {
		tokStrings[i] = string(rune('a' + i - 3))
	}

	// Count metadata KVs:
	// general.architecture, general.name,
	// bert.vocab_size, bert.embedding_length, bert.block_count,
	// bert.attention.head_count, bert.attention.head_count_kv,
	// bert.feed_forward_length, bert.context_length,
	// bert.attention.layer_norm_epsilon, bert.num_labels,
	// tokenizer.ggml.model, tokenizer.ggml.tokens, tokenizer.ggml.merges,
	// tokenizer.ggml.bos_token_id, tokenizer.ggml.eos_token_id,
	// tokenizer.ggml.unknown_token_id
	metadataCount := 17

	// Write header.
	w.writeUint32(0x46554747) // Magic "GGUF"
	w.writeUint32(3)          // Version
	w.writeUint64(uint64(len(tensors)))
	w.writeUint64(uint64(metadataCount))

	// Write metadata.
	w.writeStringKV("general.architecture", "bert")
	w.writeStringKV("general.name", "test-bert")
	w.writeUint32KV("bert.vocab_size", uint32(vocab))
	w.writeUint32KV("bert.embedding_length", uint32(hidden))
	w.writeUint32KV("bert.block_count", uint32(numLayers))
	w.writeUint32KV("bert.attention.head_count", uint32(numHeads))
	w.writeUint32KV("bert.attention.head_count_kv", uint32(numHeads))
	w.writeUint32KV("bert.feed_forward_length", uint32(inter))
	w.writeUint32KV("bert.context_length", uint32(maxPos))
	w.writeFloat32KV("bert.attention.layer_norm_epsilon", 1e-12)
	w.writeUint32KV("bert.num_labels", uint32(numLabels))
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
	// Use ones for norm weights, zeros for biases, sin pattern for others.
	for _, td := range tensors {
		numElements := uint64(1)
		for _, d := range td.shape {
			numElements *= d
		}
		isNormWeight := len(td.name) > 7 && (td.name[len(td.name)-7:] == ".weight") &&
			(contains(td.name, "norm") || contains(td.name, "embd_norm"))
		isBias := len(td.name) > 5 && td.name[len(td.name)-5:] == ".bias"
		for j := range numElements {
			var val float32
			switch {
			case isNormWeight:
				val = 1.0
			case isBias:
				val = 0.0
			default:
				val = float32(math.Sin(float64(j)*0.01)) * 0.02
			}
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatalf("write tensor data: %v", err)
			}
		}
	}

	return path
}

// contains checks if substr is in s (simple, for test use only).
func contains(s, substr string) bool {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestBuildArchGraph_BertRouting(t *testing.T) {
	cfg := bertBaseConfig()
	tensors := makeBertTestTensors(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := buildArchGraph("bert", tensors, cfg, engine)
	if err != nil {
		t.Fatalf("buildArchGraph(bert): %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding weight is nil")
	}
}

func TestLoadEncoderFile_GGUF(t *testing.T) {
	dir := t.TempDir()
	path := writeTestBertGGUF(t, dir)

	m, err := LoadEncoderFile(path)
	if err != nil {
		t.Fatalf("LoadEncoderFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	if m.Config().Architecture != "bert" {
		t.Errorf("Architecture = %q, want %q", m.Config().Architecture, "bert")
	}
	if m.Graph() == nil {
		t.Fatal("graph is nil")
	}
	if m.Engine() == nil {
		t.Fatal("engine is nil")
	}
}

func TestLoadEncoderFile_RejectsDecoder(t *testing.T) {
	// Write a llama GGUF and try to load it as encoder.
	dir := t.TempDir()
	path := writeTestGGUF(t, dir)

	_, err := LoadEncoderFile(path)
	if err == nil {
		t.Fatal("expected error for non-encoder architecture")
	}
}

func TestEncoderModel_Forward(t *testing.T) {
	dir := t.TempDir()
	path := writeTestBertGGUF(t, dir)

	m, err := LoadEncoderFile(path)
	if err != nil {
		t.Fatalf("LoadEncoderFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	inputIDs := []int{1, 5, 10, 3}
	logits, err := m.Forward(context.Background(), inputIDs)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expectedShape := m.OutputShape()
	expectedLen := expectedShape[0] * expectedShape[1]
	if len(logits) != expectedLen {
		t.Fatalf("logits length = %d, want %d (shape %v)", len(logits), expectedLen, expectedShape)
	}

	// Check no NaN/Inf.
	for i, v := range logits {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at index %d", i)
		}
	}
}

func TestEncoderModel_Forward_EmptyInput(t *testing.T) {
	dir := t.TempDir()
	path := writeTestBertGGUF(t, dir)

	m, err := LoadEncoderFile(path)
	if err != nil {
		t.Fatalf("LoadEncoderFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	_, err = m.Forward(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestEncoderModel_OutputShape(t *testing.T) {
	dir := t.TempDir()
	path := writeTestBertGGUF(t, dir)

	m, err := LoadEncoderFile(path)
	if err != nil {
		t.Fatalf("LoadEncoderFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	shape := m.OutputShape()
	if len(shape) != 2 {
		t.Fatalf("OutputShape len = %d, want 2", len(shape))
	}
	if shape[0] != 1 {
		t.Errorf("OutputShape[0] = %d, want 1", shape[0])
	}
	if shape[1] != 3 { // numLabels=3 from writeTestBertGGUF
		t.Errorf("OutputShape[1] = %d, want 3", shape[1])
	}
}

func TestIsEncoderArchitecture(t *testing.T) {
	tests := []struct {
		arch string
		want bool
	}{
		{"bert", true},
		{"roberta", true},
		{"llama", false},
		{"gemma", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := IsEncoderArchitecture(tt.arch); got != tt.want {
			t.Errorf("IsEncoderArchitecture(%q) = %v, want %v", tt.arch, got, tt.want)
		}
	}
}
