package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeMinimalGGUF writes a minimal valid GGUF file with the given tensors.
// Returns the raw bytes. Each tensor is F32 with data starting after a 32-byte
// aligned header.
func writeMinimalGGUF(t *testing.T, tensors []TensorInfo, tensorData [][]byte, metadata map[string]any) []byte {
	t.Helper()
	var buf []byte

	// Magic + version.
	buf = binary.LittleEndian.AppendUint32(buf, Magic)
	buf = binary.LittleEndian.AppendUint32(buf, 3) // GGUF v3

	// Tensor count + metadata KV count.
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(tensors)))
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(metadata)))

	// Write metadata.
	for key, val := range metadata {
		buf = appendGGUFString(buf, key)
		buf = appendGGUFValue(buf, val)
	}

	// Write tensor info entries.
	// Calculate data offsets: each tensor's data follows the previous one.
	var dataOffset uint64
	for i, ti := range tensors {
		buf = appendGGUFString(buf, ti.Name)
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(ti.Dimensions)))
		for _, d := range ti.Dimensions {
			buf = binary.LittleEndian.AppendUint64(buf, d)
		}
		buf = binary.LittleEndian.AppendUint32(buf, uint32(ti.Type))
		buf = binary.LittleEndian.AppendUint64(buf, dataOffset)
		if i < len(tensorData) {
			dataOffset += uint64(len(tensorData[i]))
		}
	}

	// Align to 32 bytes.
	for len(buf)%32 != 0 {
		buf = append(buf, 0)
	}

	// Append tensor data.
	for _, data := range tensorData {
		buf = append(buf, data...)
	}

	return buf
}

func appendGGUFString(buf []byte, s string) []byte {
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(s)))
	return append(buf, []byte(s)...)
}

func appendGGUFValue(buf []byte, val any) []byte {
	switch v := val.(type) {
	case string:
		buf = binary.LittleEndian.AppendUint32(buf, TypeString)
		return appendGGUFString(buf, v)
	case uint32:
		buf = binary.LittleEndian.AppendUint32(buf, TypeUint32)
		return binary.LittleEndian.AppendUint32(buf, v)
	case uint64:
		buf = binary.LittleEndian.AppendUint32(buf, TypeUint64)
		return binary.LittleEndian.AppendUint64(buf, v)
	default:
		panic("unsupported test metadata type")
	}
}

func f32Bytes(values []float32) []byte {
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:i*4+4], math.Float32bits(v))
	}
	return raw
}

func TestDiscoverShards(t *testing.T) {
	dir := t.TempDir()

	// Create 3 shard files.
	for i := 1; i <= 3; i++ {
		name := filepath.Join(dir, "Model-Q4_K_M-"+shardName(i, 3))
		if err := os.WriteFile(name, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	paths, err := discoverShards(filepath.Join(dir, "Model-Q4_K_M-00001-of-00003.gguf"))
	if err != nil {
		t.Fatalf("discoverShards: %v", err)
	}
	if len(paths) != 3 {
		t.Fatalf("got %d paths, want 3", len(paths))
	}
	for i, p := range paths {
		if filepath.Base(p) != "Model-Q4_K_M-"+shardName(i+1, 3) {
			t.Errorf("path[%d] = %q, unexpected", i, filepath.Base(p))
		}
	}
}

func TestDiscoverShards_NotSplit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	paths, err := discoverShards(path)
	if err != nil {
		t.Fatalf("discoverShards: %v", err)
	}
	if paths != nil {
		t.Errorf("expected nil for non-split file, got %v", paths)
	}
}

func TestDiscoverShards_MissingShard(t *testing.T) {
	dir := t.TempDir()

	// Create shards 1 and 3 but not 2.
	for _, i := range []int{1, 3} {
		name := filepath.Join(dir, "Model-"+shardName(i, 3))
		if err := os.WriteFile(name, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	_, err := discoverShards(filepath.Join(dir, "Model-00001-of-00003.gguf"))
	if err == nil {
		t.Fatal("expected error for missing shard")
	}
}

func shardName(i, total int) string {
	return shardNameFmt(i, total) + ".gguf"
}

func shardNameFmt(i, total int) string {
	return padInt(i) + "-of-" + padInt(total)
}

func padInt(n int) string {
	s := ""
	for d := 10000; d >= 1; d /= 10 {
		s += string(rune('0' + (n/d)%10))
	}
	return s
}

func TestParseSplit(t *testing.T) {
	dir := t.TempDir()

	// Shard 1: has metadata + tensor "weight_a" (4 floats).
	valuesA := []float32{1.0, 2.0, 3.0, 4.0}
	rawA := f32Bytes(valuesA)
	shard1 := writeMinimalGGUF(t,
		[]TensorInfo{{Name: "weight_a", Dimensions: []uint64{4}, Type: GGMLTypeF32}},
		[][]byte{rawA},
		map[string]any{"general.architecture": "llama"},
	)

	// Shard 2: has tensor "weight_b" (3 floats).
	valuesB := []float32{10.0, 20.0, 30.0}
	rawB := f32Bytes(valuesB)
	shard2 := writeMinimalGGUF(t,
		[]TensorInfo{{Name: "weight_b", Dimensions: []uint64{3}, Type: GGMLTypeF32}},
		[][]byte{rawB},
		map[string]any{},
	)

	// Write shard files.
	if err := os.WriteFile(filepath.Join(dir, "Model-00001-of-00002.gguf"), shard1, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "Model-00002-of-00002.gguf"), shard2, 0o600); err != nil {
		t.Fatal(err)
	}

	sf, err := ParseSplit(filepath.Join(dir, "Model-00001-of-00002.gguf"))
	if err != nil {
		t.Fatalf("ParseSplit: %v", err)
	}
	if sf == nil {
		t.Fatal("expected SplitFile, got nil")
	}

	if len(sf.Shards) != 2 {
		t.Fatalf("got %d shards, want 2", len(sf.Shards))
	}
	if len(sf.File.Tensors) != 2 {
		t.Fatalf("got %d tensors, want 2", len(sf.File.Tensors))
	}

	// Check shard index.
	if sf.ShardIndex["weight_a"] != 0 {
		t.Errorf("weight_a in shard %d, want 0", sf.ShardIndex["weight_a"])
	}
	if sf.ShardIndex["weight_b"] != 1 {
		t.Errorf("weight_b in shard %d, want 1", sf.ShardIndex["weight_b"])
	}

	// Check metadata comes from shard 0.
	arch, ok := sf.File.GetString("general.architecture")
	if !ok || arch != "llama" {
		t.Errorf("architecture = %q, want 'llama'", arch)
	}
}

func TestParseSplit_NotSplit(t *testing.T) {
	dir := t.TempDir()
	data := writeMinimalGGUF(t,
		[]TensorInfo{{Name: "w", Dimensions: []uint64{4}, Type: GGMLTypeF32}},
		[][]byte{f32Bytes([]float32{1, 2, 3, 4})},
		map[string]any{},
	)
	path := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	sf, err := ParseSplit(path)
	if err != nil {
		t.Fatalf("ParseSplit: %v", err)
	}
	if sf != nil {
		t.Error("expected nil for non-split file")
	}
}

func TestLoadTensorsMmapSplit(t *testing.T) {
	dir := t.TempDir()

	valuesA := []float32{1.0, 2.0, 3.0, 4.0}
	valuesB := []float32{10.0, 20.0, 30.0}
	rawA := f32Bytes(valuesA)
	rawB := f32Bytes(valuesB)

	shard1 := writeMinimalGGUF(t,
		[]TensorInfo{{Name: "weight_a", Dimensions: []uint64{4}, Type: GGMLTypeF32}},
		[][]byte{rawA},
		map[string]any{"general.architecture": "llama"},
	)
	shard2 := writeMinimalGGUF(t,
		[]TensorInfo{{Name: "weight_b", Dimensions: []uint64{3}, Type: GGMLTypeF32}},
		[][]byte{rawB},
		map[string]any{},
	)

	path1 := filepath.Join(dir, "Model-00001-of-00002.gguf")
	path2 := filepath.Join(dir, "Model-00002-of-00002.gguf")
	if err := os.WriteFile(path1, shard1, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path2, shard2, 0o600); err != nil {
		t.Fatal(err)
	}

	sf, err := ParseSplit(path1)
	if err != nil {
		t.Fatalf("ParseSplit: %v", err)
	}

	// Use the raw shard bytes as "mmap'd" regions.
	mappedShards := [][]byte{shard1, shard2}

	result, err := LoadTensorsMmapSplit(sf, mappedShards)
	if err != nil {
		t.Fatalf("LoadTensorsMmapSplit: %v", err)
	}

	// Verify tensor A from shard 0.
	tA := result["weight_a"]
	if tA == nil {
		t.Fatal("weight_a not found")
	}
	gotA := tA.Data()
	for i, want := range valuesA {
		if gotA[i] != want {
			t.Errorf("weight_a[%d] = %v, want %v", i, gotA[i], want)
		}
	}

	// Verify tensor B from shard 1.
	tB := result["weight_b"]
	if tB == nil {
		t.Fatal("weight_b not found")
	}
	gotB := tB.Data()
	for i, want := range valuesB {
		if gotB[i] != want {
			t.Errorf("weight_b[%d] = %v, want %v", i, gotB[i], want)
		}
	}
}

func TestLoadTensorsSplit_Heap(t *testing.T) {
	dir := t.TempDir()

	valuesA := []float32{1.0, 2.0, 3.0, 4.0}
	valuesB := []float32{10.0, 20.0, 30.0}
	rawA := f32Bytes(valuesA)
	rawB := f32Bytes(valuesB)

	shard1 := writeMinimalGGUF(t,
		[]TensorInfo{{Name: "weight_a", Dimensions: []uint64{4}, Type: GGMLTypeF32}},
		[][]byte{rawA},
		map[string]any{"general.architecture": "llama"},
	)
	shard2 := writeMinimalGGUF(t,
		[]TensorInfo{{Name: "weight_b", Dimensions: []uint64{3}, Type: GGMLTypeF32}},
		[][]byte{rawB},
		map[string]any{},
	)

	path1 := filepath.Join(dir, "Model-00001-of-00002.gguf")
	path2 := filepath.Join(dir, "Model-00002-of-00002.gguf")
	if err := os.WriteFile(path1, shard1, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path2, shard2, 0o600); err != nil {
		t.Fatal(err)
	}

	sf, err := ParseSplit(path1)
	if err != nil {
		t.Fatalf("ParseSplit: %v", err)
	}

	// Open shard files for heap-based loading.
	readers := make([]*os.File, 2)
	readers[0], err = os.Open(path1)
	if err != nil {
		t.Fatal(err)
	}
	defer readers[0].Close()
	readers[1], err = os.Open(path2)
	if err != nil {
		t.Fatal(err)
	}
	defer readers[1].Close()

	result, err := LoadTensorsSplit(sf, readers)
	if err != nil {
		t.Fatalf("LoadTensorsSplit: %v", err)
	}

	gotA := result["weight_a"].Data()
	for i, want := range valuesA {
		if gotA[i] != want {
			t.Errorf("weight_a[%d] = %v, want %v", i, gotA[i], want)
		}
	}

	gotB := result["weight_b"].Data()
	for i, want := range valuesB {
		if gotB[i] != want {
			t.Errorf("weight_b[%d] = %v, want %v", i, gotB[i], want)
		}
	}
}
