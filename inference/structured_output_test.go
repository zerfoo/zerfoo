package inference

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	ztensorgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/zerfoo/generate/grammar"
)

// writeTestGGUF_JSON creates a minimal GGUF file whose vocabulary includes
// the single-byte tokens needed for JSON generation: {, }, ", :, comma,
// digits 0-9, minus, and lowercase letters a-z.
func writeTestGGUF_JSON(t *testing.T, dir string) string {
	t.Helper()

	hidden := 16
	inter := 32
	numLayers := 1
	numHeads := 4
	numKVHeads := 2
	kvDim := (hidden / numHeads) * numKVHeads

	// Build a vocabulary with JSON-relevant single-byte tokens.
	// Indices 0-2: special tokens, then JSON structural chars and content.
	var tokStrings []string
	tokStrings = append(tokStrings, "<unk>", "<s>", "</s>")
	// JSON structural characters.
	for _, ch := range []byte{'{', '}', '"', ':', ','} {
		tokStrings = append(tokStrings, string(ch))
	}
	// Digits 0-9 and minus.
	for _, ch := range []byte{'-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'} {
		tokStrings = append(tokStrings, string(ch))
	}
	// Lowercase letters a-z.
	for ch := byte('a'); ch <= 'z'; ch++ {
		tokStrings = append(tokStrings, string(ch))
	}
	// Space for JSON formatting.
	tokStrings = append(tokStrings, " ")
	vocab := len(tokStrings) // 3 + 5 + 11 + 26 + 1 = 46

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

	w := ztensorgguf.NewWriter()

	w.AddMetadataString("general.architecture", "llama")
	w.AddMetadataString("general.name", "test-llama-json")
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
		w.AddTensorF32(td.name, td.shape, generateF32Data(n))
	}

	path := filepath.Join(dir, "test_json.gguf")
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

func TestStructuredOutput(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF_JSON(t, dir)

	m, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	defer func() { _ = m.Close() }()

	// Schema: object with "name" (string, max 5 chars) and "age" (integer).
	schema := &grammar.JSONSchema{
		Type: "object",
		Properties: map[string]*grammar.JSONSchema{
			"name": {Type: "string", MaxLength: 5},
			"age":  {Type: "integer"},
		},
		Required: []string{"name", "age"},
	}

	g, err := grammar.Convert(schema)
	if err != nil {
		t.Fatalf("grammar.Convert: %v", err)
	}

	result, err := m.Generate(t.Context(), "hello",
		WithGrammar(g),
		WithMaxTokens(50),
		WithTemperature(0),
	)
	if err != nil {
		t.Fatalf("Generate with grammar: %v", err)
	}
	if result == "" {
		t.Fatal("Generate returned empty string")
	}

	// Validate output is parseable JSON.
	var parsed map[string]any
	if err := json.Unmarshal([]byte(result), &parsed); err != nil {
		t.Fatalf("output is not valid JSON: %v\nraw output: %q", err, result)
	}

	// Validate "name" key exists and is a string.
	nameVal, ok := parsed["name"]
	if !ok {
		t.Fatalf("output missing required key %q: %s", "name", result)
	}
	if _, ok := nameVal.(string); !ok {
		t.Errorf("key %q is %T, want string", "name", nameVal)
	}

	// Validate "age" key exists and is a number (JSON numbers decode as float64).
	ageVal, ok := parsed["age"]
	if !ok {
		t.Fatalf("output missing required key %q: %s", "age", result)
	}
	ageFloat, ok := ageVal.(float64)
	if !ok {
		t.Errorf("key %q is %T, want number", "age", ageVal)
	}
	if ageFloat != float64(int64(ageFloat)) {
		t.Errorf("key %q = %v, want integer value", "age", ageFloat)
	}
}
