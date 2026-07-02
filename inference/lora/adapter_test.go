package lora

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
)

// bw wraps binary.Write for test helpers.
func bw(buf *bytes.Buffer, data any) {
	if err := binary.Write(buf, binary.LittleEndian, data); err != nil {
		panic(err)
	}
}

func writeString(buf *bytes.Buffer, s string) {
	bw(buf, uint64(len(s)))
	buf.WriteString(s)
}

func writeValue(buf *bytes.Buffer, val any) {
	switch v := val.(type) {
	case uint32:
		bw(buf, gguf.TypeUint32)
		bw(buf, v)
	case float32:
		bw(buf, gguf.TypeFloat32)
		bw(buf, v)
	case string:
		bw(buf, gguf.TypeString)
		writeString(buf, v)
	}
}

// buildLoRAGGUF creates a synthetic GGUF v3 file with LoRA metadata and F32 tensor pairs.
// Each tensor pair has A=[rank, inDim] and B=[outDim, rank] in GGML dimension order
// (innermost first: ne[0]=cols, ne[1]=rows).
func buildLoRAGGUF(t *testing.T, rank uint32, alpha float32, layers []testLayer) *bytes.Reader {
	t.Helper()

	metadata := map[string]any{
		"lora.rank":  rank,
		"lora.alpha": alpha,
	}

	type tensorEntry struct {
		name string
		dims []uint64
		data []float32
	}

	var tensors []tensorEntry
	for _, l := range layers {
		// A shape [rank, inDim] -> GGML dims: [inDim, rank]
		tensors = append(tensors, tensorEntry{
			name: "lora_a." + l.name,
			dims: []uint64{uint64(l.inDim), uint64(rank)},
			data: l.aData,
		})
		// B shape [outDim, rank] -> GGML dims: [rank, outDim]
		tensors = append(tensors, tensorEntry{
			name: "lora_b." + l.name,
			dims: []uint64{uint64(rank), uint64(l.outDim)},
			data: l.bData,
		})
	}

	var buf bytes.Buffer

	// Header.
	bw(&buf, gguf.Magic)
	bw(&buf, uint32(3)) // version
	bw(&buf, uint64(len(tensors)))
	bw(&buf, uint64(len(metadata)))

	// Metadata.
	for key, val := range metadata {
		writeString(&buf, key)
		writeValue(&buf, val)
	}

	// Tensor info entries. We compute offsets sequentially.
	var offset uint64
	for _, te := range tensors {
		writeString(&buf, te.name)
		bw(&buf, uint32(len(te.dims)))
		for _, d := range te.dims {
			bw(&buf, d)
		}
		bw(&buf, uint32(gguf.GGMLTypeF32))
		bw(&buf, offset)
		offset += uint64(len(te.data) * 4)
	}

	// Pad to 32-byte alignment for tensor data.
	pos := buf.Len()
	const alignment = 32
	padded := (pos + alignment - 1) / alignment * alignment
	for range padded - pos {
		buf.WriteByte(0)
	}

	// Tensor data.
	for _, te := range tensors {
		for _, v := range te.data {
			bw(&buf, math.Float32bits(v))
		}
	}

	return bytes.NewReader(buf.Bytes())
}

type testLayer struct {
	name   string
	inDim  int
	outDim int
	aData  []float32
	bData  []float32
}

func makeFloats(n int, val float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = val
	}
	return s
}

func TestLoadAdapter(t *testing.T) {
	rank := uint32(4)
	alpha := float32(8.0)
	inDim := 8
	outDim := 16

	r := buildLoRAGGUF(t, rank, alpha, []testLayer{
		{
			name:   "model.layers.0.self_attn.q_proj",
			inDim:  inDim,
			outDim: outDim,
			aData:  makeFloats(int(rank)*inDim, 0.1),
			bData:  makeFloats(outDim*int(rank), 0.2),
		},
		{
			name:   "model.layers.0.self_attn.v_proj",
			inDim:  inDim,
			outDim: outDim,
			aData:  makeFloats(int(rank)*inDim, 0.3),
			bData:  makeFloats(outDim*int(rank), 0.4),
		},
	})

	adapter, err := LoadAdapter("test.gguf", r)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	if adapter.Rank != int(rank) {
		t.Errorf("Rank = %d, want %d", adapter.Rank, rank)
	}
	if adapter.Alpha != float64(alpha) {
		t.Errorf("Alpha = %f, want %f", adapter.Alpha, float64(alpha))
	}
	wantScale := float64(alpha) / float64(rank)
	if adapter.ScaleFactor != wantScale {
		t.Errorf("ScaleFactor = %f, want %f", adapter.ScaleFactor, wantScale)
	}
	if len(adapter.Layers) != 2 {
		t.Fatalf("len(Layers) = %d, want 2", len(adapter.Layers))
	}

	// Verify layer names.
	names := adapter.LayerNames()
	if len(names) != 2 {
		t.Fatalf("LayerNames() = %v, want 2 entries", names)
	}
	wantNames := []string{
		"model.layers.0.self_attn.q_proj",
		"model.layers.0.self_attn.v_proj",
	}
	for i, name := range names {
		if name != wantNames[i] {
			t.Errorf("LayerNames()[%d] = %q, want %q", i, name, wantNames[i])
		}
	}

	// Verify HasLayer.
	if !adapter.HasLayer("model.layers.0.self_attn.q_proj") {
		t.Error("HasLayer(q_proj) = false, want true")
	}
	if adapter.HasLayer("nonexistent") {
		t.Error("HasLayer(nonexistent) = true, want false")
	}

	// Verify A and B shapes.
	layer := adapter.Layers["model.layers.0.self_attn.q_proj"]
	if len(layer.A) != int(rank) {
		t.Errorf("A rows = %d, want %d", len(layer.A), rank)
	}
	if len(layer.A[0]) != inDim {
		t.Errorf("A cols = %d, want %d", len(layer.A[0]), inDim)
	}
	if len(layer.B) != outDim {
		t.Errorf("B rows = %d, want %d", len(layer.B), outDim)
	}
	if len(layer.B[0]) != int(rank) {
		t.Errorf("B cols = %d, want %d", len(layer.B[0]), rank)
	}
}

func TestLoadAdapterScaleFactor(t *testing.T) {
	tests := []struct {
		name      string
		rank      uint32
		alpha     float32
		wantScale float64
	}{
		{"rank4_alpha8", 4, 8.0, 2.0},
		{"rank16_alpha16", 16, 16.0, 1.0},
		{"rank8_alpha4", 8, 4.0, 0.5},
		{"rank1_alpha1", 1, 1.0, 1.0},
		{"rank32_alpha64", 32, 64.0, 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := buildLoRAGGUF(t, tt.rank, tt.alpha, []testLayer{
				{
					name:   "layer.0.weight",
					inDim:  4,
					outDim: 4,
					aData:  makeFloats(int(tt.rank)*4, 1.0),
					bData:  makeFloats(4*int(tt.rank), 1.0),
				},
			})

			adapter, err := LoadAdapter("test.gguf", r)
			if err != nil {
				t.Fatalf("LoadAdapter: %v", err)
			}
			if adapter.ScaleFactor != tt.wantScale {
				t.Errorf("ScaleFactor = %f, want %f", adapter.ScaleFactor, tt.wantScale)
			}
		})
	}
}

func TestLoadAdapterMissingMetadata(t *testing.T) {
	tests := []struct {
		name     string
		metadata map[string]any
		wantErr  string
	}{
		{
			name:     "missing_rank",
			metadata: map[string]any{"lora.alpha": float32(8.0)},
			wantErr:  "missing metadata key \"lora.rank\"",
		},
		{
			name:     "missing_alpha",
			metadata: map[string]any{"lora.rank": uint32(4)},
			wantErr:  "missing metadata key \"lora.alpha\"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build a minimal GGUF with no tensors but with given metadata.
			var buf bytes.Buffer
			bw(&buf, gguf.Magic)
			bw(&buf, uint32(3))
			bw(&buf, uint64(0)) // no tensors
			bw(&buf, uint64(len(tt.metadata)))
			for key, val := range tt.metadata {
				writeString(&buf, key)
				writeValue(&buf, val)
			}

			r := bytes.NewReader(buf.Bytes())
			_, err := LoadAdapter("test.gguf", r)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if got := err.Error(); !contains(got, tt.wantErr) {
				t.Errorf("error = %q, want to contain %q", got, tt.wantErr)
			}
		})
	}
}

func TestLoadAdapterZeroRank(t *testing.T) {
	var buf bytes.Buffer
	bw(&buf, gguf.Magic)
	bw(&buf, uint32(3))
	bw(&buf, uint64(0))
	bw(&buf, uint64(2))
	writeString(&buf, "lora.rank")
	writeValue(&buf, uint32(0))
	writeString(&buf, "lora.alpha")
	writeValue(&buf, float32(8.0))

	r := bytes.NewReader(buf.Bytes())
	_, err := LoadAdapter("test.gguf", r)
	if err == nil {
		t.Fatal("expected error for zero rank, got nil")
	}
	if got := err.Error(); !contains(got, "must be > 0") {
		t.Errorf("error = %q, want to contain %q", got, "must be > 0")
	}
}

func TestLoadAdapterUnpairedTensors(t *testing.T) {
	// Build GGUF with only lora_a (no matching lora_b).
	var buf bytes.Buffer
	bw(&buf, gguf.Magic)
	bw(&buf, uint32(3))
	bw(&buf, uint64(1)) // 1 tensor
	bw(&buf, uint64(2)) // 2 metadata

	writeString(&buf, "lora.rank")
	writeValue(&buf, uint32(4))
	writeString(&buf, "lora.alpha")
	writeValue(&buf, float32(8.0))

	// Single tensor: lora_a.layer0
	writeString(&buf, "lora_a.layer0")
	bw(&buf, uint32(2))       // 2 dims
	bw(&buf, uint64(8))       // inDim
	bw(&buf, uint64(4))       // rank
	bw(&buf, uint32(0))       // F32
	bw(&buf, uint64(0))       // offset

	// Pad to alignment.
	pos := buf.Len()
	padded := (pos + 31) / 32 * 32
	for range padded - pos {
		buf.WriteByte(0)
	}

	// Tensor data: 4*8 = 32 floats.
	for range 32 {
		bw(&buf, math.Float32bits(1.0))
	}

	r := bytes.NewReader(buf.Bytes())
	_, err := LoadAdapter("test.gguf", r)
	if err == nil {
		t.Fatal("expected error for unpaired tensor, got nil")
	}
	if got := err.Error(); !contains(got, "no matching lora_b") {
		t.Errorf("error = %q, want to contain %q", got, "no matching lora_b")
	}
}

func TestLoadAdapterShapeMismatch(t *testing.T) {
	rank := uint32(4)
	// Build GGUF where A's first dim != rank.
	r := buildLoRAGGUF(t, rank, 8.0, []testLayer{
		{
			name:   "layer0",
			inDim:  8,
			outDim: 16,
			aData:  makeFloats(int(rank)*8, 1.0),
			bData:  makeFloats(16*int(rank), 1.0),
		},
	})

	// This should succeed since shapes match.
	adapter, err := LoadAdapter("test.gguf", r)
	if err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}
	if adapter.Rank != int(rank) {
		t.Errorf("Rank = %d, want %d", adapter.Rank, rank)
	}
}

func TestLoadAdapterNoTensorPairs(t *testing.T) {
	// GGUF with metadata but no LoRA tensors (just a random tensor).
	var buf bytes.Buffer
	bw(&buf, gguf.Magic)
	bw(&buf, uint32(3))
	bw(&buf, uint64(1)) // 1 tensor
	bw(&buf, uint64(2)) // 2 metadata

	writeString(&buf, "lora.rank")
	writeValue(&buf, uint32(4))
	writeString(&buf, "lora.alpha")
	writeValue(&buf, float32(8.0))

	// Tensor without lora_ prefix.
	writeString(&buf, "some_other_tensor")
	bw(&buf, uint32(2))
	bw(&buf, uint64(4))
	bw(&buf, uint64(4))
	bw(&buf, uint32(0)) // F32
	bw(&buf, uint64(0))

	pos := buf.Len()
	padded := (pos + 31) / 32 * 32
	for range padded - pos {
		buf.WriteByte(0)
	}
	for range 16 {
		bw(&buf, math.Float32bits(1.0))
	}

	r := bytes.NewReader(buf.Bytes())
	_, err := LoadAdapter("test.gguf", r)
	if err == nil {
		t.Fatal("expected error for no LoRA tensor pairs, got nil")
	}
	if got := err.Error(); !contains(got, "no LoRA tensor pairs found") {
		t.Errorf("error = %q, want to contain %q", got, "no LoRA tensor pairs found")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && containsStr(s, sub)
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
