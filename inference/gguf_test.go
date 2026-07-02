package inference

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	ztensorgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// buildSyntheticGGUFFile creates a minimal GGUF file on disk with metadata and tensor data.
func buildSyntheticGGUFFile(t *testing.T, dir string) string {
	t.Helper()

	w := ztensorgguf.NewWriter()

	// Metadata.
	w.AddMetadataString("general.architecture", "llama")
	w.AddMetadataString("general.name", "test-gguf-model")
	w.AddMetadataUint32("llama.embedding_length", 64)
	w.AddMetadataUint32("llama.block_count", 2)
	w.AddMetadataUint32("llama.attention.head_count", 4)

	// Tensor: token_embd.weight (4 x 64 F32, row-major shape [4, 64]).
	f32Data := make([]float32, 4*64)
	for i := range f32Data {
		f32Data[i] = float32(i) / 256.0
	}
	w.AddTensorF32("token_embd.weight", []int{4, 64}, f32Data)

	// Tensor: blk.0.attn_q.weight (64 Q4_0, raw bytes).
	src := make([]float32, 64)
	for i := range src {
		src[i] = float32(i-32) / 32.0
	}
	q4 := tensor.QuantizeQ4(src)
	w.AddTensor("blk.0.attn_q.weight", ztensorgguf.TypeQ4_0, []int{64}, q4.RawBytes())

	path := filepath.Join(dir, "test.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := w.Write(f); err != nil {
		t.Fatalf("write GGUF: %v", err)
	}

	return path
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

// newQ4Tensor builds a Q4-backed tensor of numElements values filled with a
// smooth ramp. Size must be a multiple of 32 (Q4_0 block size).
func newQ4Tensor(t *testing.T, numElements int) *tensor.TensorNumeric[float32] {
	t.Helper()
	if numElements%32 != 0 {
		t.Fatalf("numElements=%d must be a multiple of 32", numElements)
	}
	src := make([]float32, numElements)
	for i := range src {
		src[i] = float32(i-numElements/2) / float32(numElements/2)
	}
	q4 := tensor.QuantizeQ4(src)
	tn, err := tensor.NewWithStorage[float32]([]int{numElements}, q4)
	if err != nil {
		t.Fatalf("NewWithStorage Q4: %v", err)
	}
	return tn
}

// TestUpgradeEmbeddingPrecision_PLEEmbedQ8Flag validates the T99.2.2.7 gate:
// ZERFOO_GEMMA4_PLE_EMBED_Q8=1 routes `model.ple_embed_tokens.weight` through
// the Q4->Q8 upgrade path; when the flag is unset, the PLE table is untouched
// (baseline bit-identical). The default `model.embed_tokens.weight` and
// `lm_head.weight` upgrades happen in both cases.
func TestUpgradeEmbeddingPrecision_PLEEmbedQ8Flag(t *testing.T) {
	makeTensors := func() map[string]*tensor.TensorNumeric[float32] {
		return map[string]*tensor.TensorNumeric[float32]{
			"model.embed_tokens.weight":     newQ4Tensor(t, 128),
			"lm_head.weight":                newQ4Tensor(t, 128),
			"model.ple_embed_tokens.weight": newQ4Tensor(t, 256),
		}
	}

	t.Run("flag_unset_keeps_ple_embed_q4", func(t *testing.T) {
		t.Setenv("ZERFOO_GEMMA4_PLE_EMBED_Q8", "")
		tensors := makeTensors()
		upgradeEmbeddingPrecision(tensors)

		if _, ok := tensors["model.embed_tokens.weight"].GetStorage().(*tensor.Q8Storage); !ok {
			t.Errorf("embed_tokens: expected Q8Storage, got %T",
				tensors["model.embed_tokens.weight"].GetStorage())
		}
		if _, ok := tensors["lm_head.weight"].GetStorage().(*tensor.Q8Storage); !ok {
			t.Errorf("lm_head: expected Q8Storage, got %T",
				tensors["lm_head.weight"].GetStorage())
		}
		if _, ok := tensors["model.ple_embed_tokens.weight"].GetStorage().(*tensor.Q4Storage); !ok {
			t.Errorf("ple_embed_tokens: expected Q4Storage (unchanged), got %T",
				tensors["model.ple_embed_tokens.weight"].GetStorage())
		}
	})

	t.Run("flag_set_upgrades_ple_embed_to_q8", func(t *testing.T) {
		t.Setenv("ZERFOO_GEMMA4_PLE_EMBED_Q8", "1")
		tensors := makeTensors()
		upgradeEmbeddingPrecision(tensors)

		if _, ok := tensors["model.embed_tokens.weight"].GetStorage().(*tensor.Q8Storage); !ok {
			t.Errorf("embed_tokens: expected Q8Storage, got %T",
				tensors["model.embed_tokens.weight"].GetStorage())
		}
		if _, ok := tensors["lm_head.weight"].GetStorage().(*tensor.Q8Storage); !ok {
			t.Errorf("lm_head: expected Q8Storage, got %T",
				tensors["lm_head.weight"].GetStorage())
		}
		if _, ok := tensors["model.ple_embed_tokens.weight"].GetStorage().(*tensor.Q8Storage); !ok {
			t.Errorf("ple_embed_tokens: expected Q8Storage, got %T",
				tensors["model.ple_embed_tokens.weight"].GetStorage())
		}
	})

	t.Run("flag_set_no_ple_tensor_is_noop", func(t *testing.T) {
		t.Setenv("ZERFOO_GEMMA4_PLE_EMBED_Q8", "1")
		tensors := map[string]*tensor.TensorNumeric[float32]{
			"model.embed_tokens.weight": newQ4Tensor(t, 128),
		}
		// Must not panic when ple_embed_tokens is absent.
		upgradeEmbeddingPrecision(tensors)
		if _, ok := tensors["model.embed_tokens.weight"].GetStorage().(*tensor.Q8Storage); !ok {
			t.Errorf("embed_tokens: expected Q8Storage, got %T",
				tensors["model.embed_tokens.weight"].GetStorage())
		}
	})
}
