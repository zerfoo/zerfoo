package serve

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/inference/lora"
	"github.com/zerfoo/zerfoo/model/gguf"
)

func TestAdapterCacheHandle_ResolveAdapter_MissingFile(t *testing.T) {
	h := &AdapterCacheHandle{
		cache: lora.NewAdapterCache(2),
		dir:   t.TempDir(),
	}

	_, err := h.resolveAdapter("nonexistent")
	if err == nil {
		t.Fatal("expected error for missing adapter file, got nil")
	}
}

func TestAdapterCacheHandle_ResolveAdapter_InvalidGGUF(t *testing.T) {
	dir := t.TempDir()

	// Write a file that is not valid GGUF.
	if err := os.WriteFile(filepath.Join(dir, "bad.gguf"), []byte("not a gguf file"), 0644); err != nil {
		t.Fatal(err)
	}

	h := &AdapterCacheHandle{
		cache: lora.NewAdapterCache(2),
		dir:   dir,
	}

	_, err := h.resolveAdapter("bad")
	if err == nil {
		t.Fatal("expected error for invalid GGUF, got nil")
	}
}

// --- SERVE-2: LoRA adapter-name path traversal fixtures and tests ---
//
// resolveAdapter must reject any adapter name that is not a plain
// alphanumeric/underscore/hyphen token *before* touching the filesystem, and
// must never resolve to a path outside h.dir even if the regex gate were
// somehow bypassed. These tests plant a "secret" GGUF file outside the
// configured adapter directory and confirm that no crafted name can reach it.

// bwTest wraps binary.Write for test fixture construction.
func bwTest(t *testing.T, buf *bytes.Buffer, data any) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, data); err != nil {
		t.Fatal(err)
	}
}

func writeGGUFString(t *testing.T, buf *bytes.Buffer, s string) {
	t.Helper()
	bwTest(t, buf, uint64(len(s)))
	buf.WriteString(s)
}

// writeLoRAGGUFFile writes a minimal-but-real GGUF v3 file with LoRA
// metadata and a single F32 tensor pair to path, mirroring the fixture
// builder in inference/lora/adapter_test.go so that resolveAdapter can be
// exercised against a real, loadable adapter rather than a mock.
func writeLoRAGGUFFile(t *testing.T, path string, rank uint32, alpha float32) {
	t.Helper()

	const layerName = "model.layers.0.self_attn.q_proj"
	const inDim, outDim = 8, 16
	aData := make([]float32, int(rank)*inDim)
	bData := make([]float32, outDim*int(rank))
	for i := range aData {
		aData[i] = 0.1
	}
	for i := range bData {
		bData[i] = 0.2
	}

	type tensorEntry struct {
		name string
		dims []uint64
		data []float32
	}
	tensors := []tensorEntry{
		{name: "lora_a." + layerName, dims: []uint64{uint64(inDim), uint64(rank)}, data: aData},
		{name: "lora_b." + layerName, dims: []uint64{uint64(rank), uint64(outDim)}, data: bData},
	}

	var buf bytes.Buffer
	bwTest(t, &buf, gguf.Magic)
	bwTest(t, &buf, uint32(3)) // version
	bwTest(t, &buf, uint64(len(tensors)))
	bwTest(t, &buf, uint64(2)) // metadata count

	writeGGUFString(t, &buf, "lora.rank")
	bwTest(t, &buf, gguf.TypeUint32)
	bwTest(t, &buf, rank)
	writeGGUFString(t, &buf, "lora.alpha")
	bwTest(t, &buf, gguf.TypeFloat32)
	bwTest(t, &buf, alpha)

	var offset uint64
	for _, te := range tensors {
		writeGGUFString(t, &buf, te.name)
		bwTest(t, &buf, uint32(len(te.dims)))
		for _, d := range te.dims {
			bwTest(t, &buf, d)
		}
		bwTest(t, &buf, uint32(gguf.GGMLTypeF32))
		bwTest(t, &buf, offset)
		offset += uint64(len(te.data) * 4)
	}

	pos := buf.Len()
	const alignment = 32
	padded := (pos + alignment - 1) / alignment * alignment
	for range padded - pos {
		buf.WriteByte(0)
	}

	for _, te := range tensors {
		for _, v := range te.data {
			bwTest(t, &buf, math.Float32bits(v))
		}
	}

	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestAdapterCacheHandle_ResolveAdapter_ValidNameLoadsRealFixture(t *testing.T) {
	dir := t.TempDir()
	writeLoRAGGUFFile(t, filepath.Join(dir, "my-adapter.gguf"), 4, 8.0)

	h := &AdapterCacheHandle{
		cache: lora.NewAdapterCache(2),
		dir:   dir,
	}

	adapter, err := h.resolveAdapter("my-adapter")
	if err != nil {
		t.Fatalf("resolveAdapter(%q) = %v, want success", "my-adapter", err)
	}
	if adapter.Rank != 4 {
		t.Errorf("Rank = %d, want 4", adapter.Rank)
	}
}

func TestAdapterCacheHandle_ResolveAdapter_PathTraversalRejected(t *testing.T) {
	base := t.TempDir()
	dir := filepath.Join(base, "adapters")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Plant a real, loadable adapter GGUF *outside* the configured adapter
	// directory. If any crafted name below successfully resolves, this is
	// the file it would have opened.
	secretPath := filepath.Join(base, "secret.gguf")
	writeLoRAGGUFFile(t, secretPath, 4, 8.0)

	maliciousNames := []string{
		"../secret",
		"../../secret",
		"../../../../../../etc/passwd",
		"a/../../secret",
		"/etc/passwd",
		"..",
		"./../secret",
		"secret\x00.gguf",
	}

	for _, name := range maliciousNames {
		t.Run(name, func(t *testing.T) {
			h := &AdapterCacheHandle{
				cache: lora.NewAdapterCache(2),
				dir:   dir,
			}

			_, err := h.resolveAdapter(name)
			if err == nil {
				t.Fatalf("resolveAdapter(%q) succeeded, want error", name)
			}
			if h.cache.Size() != 0 {
				t.Errorf("resolveAdapter(%q): cache poisoned by traversal, size = %d, want 0", name, h.cache.Size())
			}
		})
	}
}

func TestWithAdapterCache(t *testing.T) {
	dir := t.TempDir()
	opt := WithAdapterCache(dir, 5)

	s := &Server{}
	opt(s)

	if s.adapterCache == nil {
		t.Fatal("expected adapterCache to be set")
	}
	if s.adapterCache.dir != dir {
		t.Errorf("dir = %q, want %q", s.adapterCache.dir, dir)
	}
}
