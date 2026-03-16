package inference

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildSyntheticGGUFFile creates a minimal GGUF file on disk with metadata and tensor data.
func buildSyntheticGGUFFile(t *testing.T, dir string) string {
	t.Helper()
	var buf bytes.Buffer

	// Header
	bw(&buf, gguf.Magic)
	bw(&buf, uint32(3)) // version
	bw(&buf, uint64(2)) // 2 tensors
	bw(&buf, uint64(5)) // 5 metadata KV pairs

	// Metadata
	writeKV(&buf, "general.architecture", "llama")
	writeKV(&buf, "general.name", "test-gguf-model")
	writeKV(&buf, "llama.embedding_length", uint32(64))
	writeKV(&buf, "llama.block_count", uint32(2))
	writeKV(&buf, "llama.attention.head_count", uint32(4))

	// Tensor info: token_embd.weight (4 x 64 F32)
	// GGUF stores dims in GGML order (innermost-first): [cols=64, rows=4].
	writeStr(&buf, "token_embd.weight")
	bw(&buf, uint32(2)) // 2 dimensions
	bw(&buf, uint64(64))
	bw(&buf, uint64(4))
	bw(&buf, uint32(gguf.GGMLTypeF32))
	bw(&buf, uint64(0)) // offset

	// Tensor info: blk.0.attn_q.weight (64 Q4_0)
	writeStr(&buf, "blk.0.attn_q.weight")
	bw(&buf, uint32(1))
	bw(&buf, uint64(64))
	bw(&buf, uint32(gguf.GGMLTypeQ4_0))
	bw(&buf, uint64(4*4*64)) // after F32 tensor

	// Pad to 32-byte alignment.
	pos := buf.Len()
	const alignment = 32
	padded := (pos + alignment - 1) / alignment * alignment
	for range padded - pos {
		buf.WriteByte(0)
	}

	// Tensor data: token_embd.weight (4*64 = 256 float32 values)
	for i := range 256 {
		bw(&buf, float32(i)/256.0)
	}

	// Tensor data: blk.0.attn_q.weight (Q4_0: 64 elements = 2 blocks = 36 bytes)
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i-32) / 32.0
	}
	q4 := tensor.QuantizeQ4(src)
	buf.Write(q4.RawBytes())

	path := filepath.Join(dir, "test.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func bw(buf *bytes.Buffer, data any) {
	if err := binary.Write(buf, binary.LittleEndian, data); err != nil {
		panic(err)
	}
}

func writeStr(buf *bytes.Buffer, s string) {
	bw(buf, uint64(len(s)))
	buf.WriteString(s)
}

func writeKV(buf *bytes.Buffer, key string, val any) {
	writeStr(buf, key)
	switch v := val.(type) {
	case string:
		bw(buf, gguf.TypeString)
		writeStr(buf, v)
	case uint32:
		bw(buf, gguf.TypeUint32)
		bw(buf, v)
	}
}

func TestLoadGGUF(t *testing.T) {
	dir := t.TempDir()
	path := buildSyntheticGGUFFile(t, dir)

	m, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}

	// Verify config.
	if m.Config.Architecture != "llama" {
		t.Errorf("Architecture = %q, want llama", m.Config.Architecture)
	}
	if m.Config.Name != "test-gguf-model" {
		t.Errorf("Name = %q, want test-gguf-model", m.Config.Name)
	}
	if m.Config.HiddenSize != 64 {
		t.Errorf("HiddenSize = %d, want 64", m.Config.HiddenSize)
	}
	if m.Config.NumLayers != 2 {
		t.Errorf("NumLayers = %d, want 2", m.Config.NumLayers)
	}
	if m.Config.NumHeads != 4 {
		t.Errorf("NumHeads = %d, want 4", m.Config.NumHeads)
	}

	// Verify tensors were name-mapped.
	if len(m.Tensors) != 2 {
		t.Fatalf("Tensor count = %d, want 2", len(m.Tensors))
	}

	// token_embd.weight → model.embed_tokens.weight
	embed, ok := m.Tensors["model.embed_tokens.weight"]
	if !ok {
		t.Fatal("model.embed_tokens.weight not found")
	}
	if embed.Shape()[0] != 4 || embed.Shape()[1] != 64 {
		t.Errorf("embed shape = %v, want [4 64]", embed.Shape())
	}

	// blk.0.attn_q.weight → model.layers.0.self_attn.q_proj.weight
	attnQ, ok := m.Tensors["model.layers.0.self_attn.q_proj.weight"]
	if !ok {
		t.Fatal("model.layers.0.self_attn.q_proj.weight not found")
	}
	if attnQ.Shape()[0] != 64 {
		t.Errorf("attn_q shape = %v, want [64]", attnQ.Shape())
	}

	// Verify Q4 tensor is stored as Q4Storage.
	if _, ok := attnQ.GetStorage().(*tensor.Q4Storage); !ok {
		t.Errorf("expected Q4Storage, got %T", attnQ.GetStorage())
	}

	// Verify F32 tensor values.
	data := embed.Data()
	if math.Abs(float64(data[0])) > 0.001 {
		t.Errorf("embed[0] = %v, want ~0", data[0])
	}
}

func TestLoadGGUF_ToModelMetadata(t *testing.T) {
	dir := t.TempDir()
	path := buildSyntheticGGUFFile(t, dir)

	m, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}

	meta := m.ToModelMetadata()
	if meta.Architecture != "llama" {
		t.Errorf("Architecture = %q, want llama", meta.Architecture)
	}
	if meta.HiddenSize != 64 {
		t.Errorf("HiddenSize = %d, want 64", meta.HiddenSize)
	}
	if meta.NumLayers != 2 {
		t.Errorf("NumLayers = %d, want 2", meta.NumLayers)
	}
	if meta.NumQueryHeads != 4 {
		t.Errorf("NumQueryHeads = %d, want 4", meta.NumQueryHeads)
	}
}

func TestLoadGGUF_FileNotFound(t *testing.T) {
	_, err := LoadGGUF("/nonexistent/model.gguf")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoadGGUF_InvalidFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.gguf")
	if err := os.WriteFile(path, []byte("not a gguf file"), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := LoadGGUF(path)
	if err == nil {
		t.Error("expected error for invalid GGUF file")
	}
}
