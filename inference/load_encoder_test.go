package inference

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/ztensor/numeric"
)

// writeTestBertGGUF creates a minimal BERT-architecture GGUF file for testing.
func writeTestBertGGUF(t *testing.T, dir string) string {
	t.Helper()

	hidden := 48
	inter := 96
	vocab := 32
	maxPos := 64
	numLayers := 2
	numHeads := 4
	numLabels := 3

	// Row-major shapes (outermost first).
	type bertTensorDef struct {
		name  string
		shape []int
	}

	var tensors []bertTensorDef

	// Global embeddings.
	tensors = append(tensors,
		bertTensorDef{"token_embd.weight", []int{vocab, hidden}},
		bertTensorDef{"position_embd.weight", []int{maxPos, hidden}},
		bertTensorDef{"token_type_embd.weight", []int{2, hidden}},
		bertTensorDef{"token_embd_norm.weight", []int{hidden}},
		bertTensorDef{"token_embd_norm.bias", []int{hidden}},
	)

	// Per-layer tensors.
	for i := 0; i < numLayers; i++ {
		prefix := "blk." + itoa(i) + "."
		tensors = append(tensors,
			bertTensorDef{prefix + "attn_q.weight", []int{hidden, hidden}},
			bertTensorDef{prefix + "attn_q.bias", []int{hidden}},
			bertTensorDef{prefix + "attn_k.weight", []int{hidden, hidden}},
			bertTensorDef{prefix + "attn_k.bias", []int{hidden}},
			bertTensorDef{prefix + "attn_v.weight", []int{hidden, hidden}},
			bertTensorDef{prefix + "attn_v.bias", []int{hidden}},
			bertTensorDef{prefix + "attn_output.weight", []int{hidden, hidden}},
			bertTensorDef{prefix + "attn_output.bias", []int{hidden}},
			bertTensorDef{prefix + "attn_norm.weight", []int{hidden}},
			bertTensorDef{prefix + "attn_norm.bias", []int{hidden}},
			bertTensorDef{prefix + "ffn_up.weight", []int{inter, hidden}},
			bertTensorDef{prefix + "ffn_up.bias", []int{inter}},
			bertTensorDef{prefix + "ffn_down.weight", []int{hidden, inter}},
			bertTensorDef{prefix + "ffn_down.bias", []int{hidden}},
			bertTensorDef{prefix + "ffn_norm.weight", []int{hidden}},
			bertTensorDef{prefix + "ffn_norm.bias", []int{hidden}},
		)
	}

	// Pooler (CLS token projection + tanh).
	tensors = append(tensors,
		bertTensorDef{"cls_pooler.weight", []int{hidden, hidden}},
		bertTensorDef{"cls_pooler.bias", []int{hidden}},
	)

	// Classification head.
	tensors = append(tensors,
		bertTensorDef{"cls.weight", []int{numLabels, hidden}},
		bertTensorDef{"cls.bias", []int{numLabels}},
	)

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
	w.AddMetadataString("general.architecture", "bert")
	w.AddMetadataString("general.name", "test-bert")
	w.AddMetadataUint32("bert.vocab_size", uint32(vocab))
	w.AddMetadataUint32("bert.embedding_length", uint32(hidden))
	w.AddMetadataUint32("bert.block_count", uint32(numLayers))
	w.AddMetadataUint32("bert.attention.head_count", uint32(numHeads))
	w.AddMetadataUint32("bert.attention.head_count_kv", uint32(numHeads))
	w.AddMetadataUint32("bert.feed_forward_length", uint32(inter))
	w.AddMetadataUint32("bert.context_length", uint32(maxPos))
	w.AddMetadataFloat32("bert.attention.layer_norm_epsilon", 1e-12)
	w.AddMetadataUint32("bert.num_labels", uint32(numLabels))
	w.AddMetadataString("tokenizer.ggml.model", "gpt2")
	w.AddMetadataStringArray("tokenizer.ggml.tokens", tokStrings)
	w.AddMetadataStringArray("tokenizer.ggml.merges", nil)
	w.AddMetadataUint32("tokenizer.ggml.bos_token_id", 1)
	w.AddMetadataUint32("tokenizer.ggml.eos_token_id", 2)
	w.AddMetadataUint32("tokenizer.ggml.unknown_token_id", 0)

	// Add tensors with appropriate data: ones for norm weights, zeros for biases,
	// sin pattern for others.
	for _, td := range tensors {
		n := numElements(td.shape)
		isNormWeight := strings.HasSuffix(td.name, ".weight") &&
			(strings.Contains(td.name, "norm") || strings.Contains(td.name, "embd_norm"))
		isBias := strings.HasSuffix(td.name, ".bias")

		data := make([]float32, n)
		for j := range data {
			switch {
			case isNormWeight:
				data[j] = 1.0
			case isBias:
				data[j] = 0.0
			default:
				data[j] = float32(math.Sin(float64(j)*0.01)) * 0.02
			}
		}
		w.AddTensorF32(td.name, td.shape, data)
	}

	path := filepath.Join(dir, "bert_test.gguf")
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
