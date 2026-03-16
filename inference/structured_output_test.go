package inference

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/generate/grammar"
)

// writeTestGGUF_JSON creates a minimal GGUF file whose vocabulary includes
// the single-byte tokens needed for JSON generation: {, }, ", :, comma,
// digits 0-9, minus, and lowercase letters a-z.
func writeTestGGUF_JSON(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "test_json.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	w := &ggufWriter{f: f, t: t}

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

	type tensorDef struct {
		name  string
		shape []uint64
	}
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

	metadataCount := 11 + 4 + 1

	w.writeUint32(0x46554747) // Magic "GGUF"
	w.writeUint32(3)          // Version
	w.writeUint64(uint64(len(tensors)))
	w.writeUint64(uint64(metadataCount))

	w.writeStringKV("general.architecture", "llama")
	w.writeStringKV("general.name", "test-llama-json")
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
			if err := binary.Write(f, binary.LittleEndian, val); err != nil {
				t.Fatalf("write tensor data: %v", err)
			}
		}
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
