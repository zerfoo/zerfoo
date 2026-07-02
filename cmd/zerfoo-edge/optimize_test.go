package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
)

// writeTestGGUF creates a minimal valid GGUF v3 file at path with the given
// metadata and tensor definitions. Tensor data is filled with float32 zeros.
func writeTestGGUF(t *testing.T, path string, metadata map[string]any, tensors []gguf.TensorInfo) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()

	// Header: magic, version, tensor_count, metadata_kv_count.
	write := func(v any) {
		t.Helper()
		if err := binary.Write(f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	write(gguf.Magic)
	write(uint32(3)) // version 3
	write(uint64(len(tensors)))
	write(uint64(len(metadata)))

	// Write metadata in a deterministic order.
	keys := sortedKeys(metadata)
	for _, key := range keys {
		writeGGUFString(t, f, key)
		writeGGUFValue(t, f, metadata[key])
	}

	// Write tensor info entries. Compute offsets sequentially.
	var offset uint64
	type tensorMeta struct {
		dataSize int
		offset   uint64
	}
	metas := make([]tensorMeta, len(tensors))
	for i, ti := range tensors {
		writeGGUFString(t, f, ti.Name)
		write(uint32(len(ti.Dimensions)))
		for _, d := range ti.Dimensions {
			write(d)
		}
		write(uint32(ti.Type))
		write(offset)
		metas[i].offset = offset

		numElems := 1
		for _, d := range ti.Dimensions {
			numElems *= int(d)
		}
		var dataSize int
		switch ti.Type {
		case gguf.GGMLTypeF32:
			dataSize = numElems * 4
		case gguf.GGMLTypeF16:
			dataSize = numElems * 2
		case gguf.GGMLTypeQ4_0:
			nBlocks := (numElems + 31) / 32
			dataSize = nBlocks * 18
		case gguf.GGMLTypeQ8_0:
			nBlocks := (numElems + 31) / 32
			dataSize = nBlocks * 34
		default:
			dataSize = numElems * 4 // fallback
		}
		metas[i].dataSize = dataSize
		offset += uint64(dataSize)
	}

	// Pad to 32-byte alignment.
	pos, err := f.Seek(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	const alignment = 32
	aligned := (pos + alignment - 1) / alignment * alignment
	if pad := aligned - pos; pad > 0 {
		if _, err := f.Write(make([]byte, pad)); err != nil {
			t.Fatal(err)
		}
	}

	// Write tensor data (zeros).
	for _, m := range metas {
		if _, err := f.Write(make([]byte, m.dataSize)); err != nil {
			t.Fatal(err)
		}
	}
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// Use simple insertion sort for small maps.
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

func writeGGUFString(t *testing.T, f *os.File, s string) {
	t.Helper()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(s))); err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(s); err != nil {
		t.Fatal(err)
	}
}

func writeGGUFValue(t *testing.T, f *os.File, v any) {
	t.Helper()
	write := func(val any) {
		t.Helper()
		if err := binary.Write(f, binary.LittleEndian, val); err != nil {
			t.Fatal(err)
		}
	}
	switch val := v.(type) {
	case string:
		write(uint32(gguf.TypeString))
		writeGGUFString(t, f, val)
	case uint32:
		write(uint32(gguf.TypeUint32))
		write(val)
	case float32:
		write(uint32(gguf.TypeFloat32))
		write(val)
	case bool:
		write(uint32(gguf.TypeBool))
		var b uint8
		if val {
			b = 1
		}
		write(b)
	case int32:
		write(uint32(gguf.TypeInt32))
		write(val)
	case uint64:
		write(uint32(gguf.TypeUint64))
		write(val)
	default:
		t.Fatalf("unsupported test metadata type %T", v)
	}
}

// buildTestMetadata returns a minimal set of GGUF metadata for a llama model.
func buildTestMetadata() map[string]any {
	return map[string]any{
		"general.architecture":               "llama",
		"general.name":                        "test-model",
		"general.file_type":                   uint32(2), // Q4_0
		"llama.vocab_size":                    uint32(32000),
		"llama.embedding_length":              uint32(128),
		"llama.block_count":                   uint32(2),
		"llama.attention.head_count":          uint32(4),
		"llama.attention.head_count_kv":       uint32(4),
		"llama.feed_forward_length":           uint32(256),
		"llama.context_length":                uint32(2048),
		"llama.rope.freq_base":               float32(10000),
		"llama.attention.layer_norm_rms_epsilon": float32(1e-5),
		// Non-essential keys that should be stripped.
		"general.description":   "A test model for unit testing",
		"general.author":        "test",
		"general.license":       "MIT",
		"tokenizer.ggml.model":  "llama",
		"tokenizer.ggml.tokens": "fake",
	}
}

// buildTestTensors returns minimal tensor info for a 2-layer llama model.
func buildTestTensors() []gguf.TensorInfo {
	return []gguf.TensorInfo{
		{Name: "token_embd.weight", Dimensions: []uint64{128, 32000}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.attn_q.weight", Dimensions: []uint64{128, 128}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.attn_k.weight", Dimensions: []uint64{128, 128}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.attn_v.weight", Dimensions: []uint64{128, 128}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.attn_output.weight", Dimensions: []uint64{128, 128}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.attn_norm.weight", Dimensions: []uint64{128}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.ffn_gate.weight", Dimensions: []uint64{128, 256}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.ffn_up.weight", Dimensions: []uint64{128, 256}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.ffn_down.weight", Dimensions: []uint64{256, 128}, Type: gguf.GGMLTypeF32},
		{Name: "blk.0.ffn_norm.weight", Dimensions: []uint64{128}, Type: gguf.GGMLTypeF32},
		{Name: "output_norm.weight", Dimensions: []uint64{128}, Type: gguf.GGMLTypeF32},
		{Name: "output.weight", Dimensions: []uint64{128, 32000}, Type: gguf.GGMLTypeF32},
	}
}

func TestOptimizeForEdge_BasicConversion(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "input.gguf")
	outputPath := filepath.Join(dir, "output.gguf")

	writeTestGGUF(t, inputPath, buildTestMetadata(), buildTestTensors())

	err := OptimizeForEdge(inputPath, outputPath, EdgeOptConfig{
		TargetArch: "arm64",
	})
	if err != nil {
		t.Fatalf("OptimizeForEdge: %v", err)
	}

	// Verify output file exists and is a valid GGUF.
	outFile, err := os.Open(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = outFile.Close() }()

	gf, err := gguf.Parse(outFile)
	if err != nil {
		t.Fatalf("parse output GGUF: %v", err)
	}

	// Verify edge-specific metadata was added.
	if v, ok := gf.Metadata["edge.optimized"]; !ok {
		t.Error("missing edge.optimized metadata")
	} else if b, ok := v.(bool); !ok || !b {
		t.Errorf("edge.optimized = %v, want true", v)
	}

	if v, ok := gf.GetString("edge.target_arch"); !ok || v != "arm64" {
		t.Errorf("edge.target_arch = %q, want %q", v, "arm64")
	}

	// Verify architecture metadata was preserved.
	if v, ok := gf.GetString("general.architecture"); !ok || v != "llama" {
		t.Errorf("general.architecture = %q, want %q", v, "llama")
	}

	// Verify tensor count is preserved.
	if got := len(gf.Tensors); got != len(buildTestTensors()) {
		t.Errorf("tensor count = %d, want %d", got, len(buildTestTensors()))
	}
}

func TestOptimizeForEdge_MetadataStrip(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "input.gguf")
	outputPath := filepath.Join(dir, "output.gguf")

	writeTestGGUF(t, inputPath, buildTestMetadata(), buildTestTensors())

	err := OptimizeForEdge(inputPath, outputPath, EdgeOptConfig{
		TargetArch: "arm64",
	})
	if err != nil {
		t.Fatalf("OptimizeForEdge: %v", err)
	}

	outFile, err := os.Open(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = outFile.Close() }()

	gf, err := gguf.Parse(outFile)
	if err != nil {
		t.Fatalf("parse output: %v", err)
	}

	// These non-essential keys should have been stripped.
	strippedKeys := []string{
		"general.description",
		"general.author",
		"general.license",
		"tokenizer.ggml.model",
		"tokenizer.ggml.tokens",
	}
	for _, key := range strippedKeys {
		if _, ok := gf.Metadata[key]; ok {
			t.Errorf("metadata key %q should have been stripped", key)
		}
	}

	// These essential keys should be preserved.
	preservedKeys := []string{
		"general.architecture",
		"general.name",
		"general.file_type",
		"llama.vocab_size",
		"llama.embedding_length",
		"llama.block_count",
		"llama.attention.head_count",
		"llama.attention.head_count_kv",
		"llama.feed_forward_length",
		"llama.context_length",
		"llama.rope.freq_base",
		"llama.attention.layer_norm_rms_epsilon",
	}
	for _, key := range preservedKeys {
		if _, ok := gf.Metadata[key]; !ok {
			t.Errorf("essential metadata key %q was stripped", key)
		}
	}
}

func TestOptimizeForEdge_QuantizationSelect(t *testing.T) {
	tests := []struct {
		name     string
		quant    string
		wantType gguf.GGMLType
	}{
		{"q4_0", "q4_0", gguf.GGMLTypeQ4_0},
		{"q4_1", "q4_1", gguf.GGMLTypeQ4_1},
		{"q8_0", "q8_0", gguf.GGMLTypeQ8_0},
		{"keep_original", "", gguf.GGMLTypeF32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			inputPath := filepath.Join(dir, "input.gguf")
			outputPath := filepath.Join(dir, "output.gguf")

			writeTestGGUF(t, inputPath, buildTestMetadata(), buildTestTensors())

			err := OptimizeForEdge(inputPath, outputPath, EdgeOptConfig{
				TargetArch:   "arm64",
				Quantization: tt.quant,
			})
			if err != nil {
				t.Fatalf("OptimizeForEdge: %v", err)
			}

			outFile, err := os.Open(outputPath)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = outFile.Close() }()

			gf, err := gguf.Parse(outFile)
			if err != nil {
				t.Fatalf("parse output: %v", err)
			}

			// Check that weight tensors (not embedding/norm/output) got the target type.
			for _, ti := range gf.Tensors {
				isWeight := !strings.Contains(ti.Name, "embed") &&
					!strings.Contains(ti.Name, "output") &&
					!strings.Contains(ti.Name, "norm")
				if isWeight && tt.quant != "" {
					if ti.Type != tt.wantType {
						t.Errorf("tensor %q type = %d, want %d", ti.Name, ti.Type, tt.wantType)
					}
				}
				// Embedding/norm/output tensors should keep original type.
				if !isWeight {
					if ti.Type != gguf.GGMLTypeF32 {
						t.Errorf("tensor %q (protected) type = %d, want F32", ti.Name, ti.Type)
					}
				}
			}

			// Verify quantization metadata is set when quantization is requested.
			if tt.quant != "" {
				if v, ok := gf.GetString("edge.quantization"); !ok || v != tt.quant {
					t.Errorf("edge.quantization = %q, want %q", v, tt.quant)
				}
			}
		})
	}
}

func TestOptimizeForEdge_InvalidInput(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		config    EdgeOptConfig
		wantErr   string
		writeFile bool
		content   []byte
	}{
		{
			name:    "nonexistent_file",
			input:   "/nonexistent/path/model.gguf",
			config:  EdgeOptConfig{TargetArch: "arm64"},
			wantErr: "open input",
		},
		{
			name:      "invalid_magic",
			config:    EdgeOptConfig{TargetArch: "arm64"},
			wantErr:   "invalid GGUF magic",
			writeFile: true,
			content:   []byte{0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00},
		},
		{
			name:    "invalid_target_arch",
			config:  EdgeOptConfig{TargetArch: "mips"},
			wantErr: "unsupported target_arch",
		},
		{
			name:    "invalid_quantization",
			config:  EdgeOptConfig{TargetArch: "arm64", Quantization: "q2_k"},
			wantErr: "unsupported quantization",
		},
		{
			name:    "negative_memory",
			config:  EdgeOptConfig{TargetArch: "arm64", MaxMemoryMB: -1},
			wantErr: "max_memory_mb must be non-negative",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			inputPath := tt.input
			if tt.writeFile {
				inputPath = filepath.Join(dir, "bad.gguf")
				if err := os.WriteFile(inputPath, tt.content, 0o644); err != nil {
					t.Fatal(err)
				}
			} else if inputPath == "" {
				// Need a valid GGUF for config validation tests that pass file open.
				inputPath = filepath.Join(dir, "input.gguf")
				writeTestGGUF(t, inputPath, buildTestMetadata(), buildTestTensors())
			}

			outputPath := filepath.Join(dir, "output.gguf")
			err := OptimizeForEdge(inputPath, outputPath, tt.config)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want substring %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestOptimizeForEdge_MemoryBudget(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "input.gguf")
	outputPath := filepath.Join(dir, "output.gguf")

	writeTestGGUF(t, inputPath, buildTestMetadata(), buildTestTensors())

	// Compute total tensor data size to set a budget that's too small.
	tensors := buildTestTensors()
	var totalBytes int
	for _, ti := range tensors {
		n := 1
		for _, d := range ti.Dimensions {
			n *= int(d)
		}
		totalBytes += n * 4 // all F32
	}

	// Set a budget that's too small (1 MB, but tensors are ~32MB for the embeddings).
	err := OptimizeForEdge(inputPath, outputPath, EdgeOptConfig{
		TargetArch:  "arm64",
		MaxMemoryMB: 1,
	})
	if err == nil {
		t.Fatal("expected memory budget error")
	}
	if !strings.Contains(err.Error(), "exceeds memory budget") {
		t.Errorf("error = %q, want 'exceeds memory budget'", err.Error())
	}

	// Verify that a generous budget succeeds.
	totalMB := int(math.Ceil(float64(totalBytes) / (1024 * 1024)))
	err = OptimizeForEdge(inputPath, outputPath, EdgeOptConfig{
		TargetArch:  "x86",
		MaxMemoryMB: totalMB + 10,
	})
	if err != nil {
		t.Fatalf("generous budget should succeed: %v", err)
	}

	// Verify the memory budget metadata was written.
	outFile, err := os.Open(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = outFile.Close() }()

	gf, err := gguf.Parse(outFile)
	if err != nil {
		t.Fatalf("parse output: %v", err)
	}

	if v, ok := gf.GetUint32("edge.max_memory_mb"); !ok {
		t.Error("missing edge.max_memory_mb metadata")
	} else if int(v) != totalMB+10 {
		t.Errorf("edge.max_memory_mb = %d, want %d", v, totalMB+10)
	}
}
