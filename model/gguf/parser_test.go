package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// bw wraps binary.Write and panics on error (test helper only).
func bw(buf *bytes.Buffer, data any) {
	if err := binary.Write(buf, binary.LittleEndian, data); err != nil {
		panic(err)
	}
}

// buildSyntheticGGUF creates a minimal GGUF v3 file in memory with the
// given metadata and tensor info.
func buildSyntheticGGUF(t *testing.T, metadata map[string]any, tensors []TensorInfo) *bytes.Reader {
	t.Helper()
	var buf bytes.Buffer

	bw(&buf, Magic)
	bw(&buf, uint32(3))
	bw(&buf, uint64(len(tensors)))
	bw(&buf, uint64(len(metadata)))

	for key, val := range metadata {
		writeTestString(&buf, key)
		writeTestValue(&buf, val)
	}

	for _, ti := range tensors {
		writeTestString(&buf, ti.Name)
		bw(&buf, uint32(len(ti.Dimensions)))
		for _, d := range ti.Dimensions {
			bw(&buf, d)
		}
		bw(&buf, uint32(ti.Type))
		bw(&buf, ti.Offset)
	}

	return bytes.NewReader(buf.Bytes())
}

func writeTestString(buf *bytes.Buffer, s string) {
	bw(buf, uint64(len(s)))
	buf.WriteString(s)
}

func writeTestValue(buf *bytes.Buffer, val any) {
	switch v := val.(type) {
	case string:
		bw(buf, TypeString)
		writeTestString(buf, v)
	case uint32:
		bw(buf, TypeUint32)
		bw(buf, v)
	case float32:
		bw(buf, TypeFloat32)
		bw(buf, v)
	case bool:
		bw(buf, TypeBool)
		if v {
			bw(buf, uint8(1))
		} else {
			bw(buf, uint8(0))
		}
	case uint64:
		bw(buf, TypeUint64)
		bw(buf, v)
	case int32:
		bw(buf, TypeInt32)
		bw(buf, v)
	case float64:
		bw(buf, TypeFloat64)
		bw(buf, v)
	case []any:
		bw(buf, TypeArray)
		var elemType uint32
		if len(v) > 0 {
			switch v[0].(type) {
			case uint32:
				elemType = TypeUint32
			case string:
				elemType = TypeString
			}
		}
		bw(buf, elemType)
		bw(buf, uint64(len(v)))
		for _, elem := range v {
			switch e := elem.(type) {
			case uint32:
				bw(buf, e)
			case string:
				writeTestString(buf, e)
			}
		}
	}
}

func TestParse_EmptyFile(t *testing.T) {
	r := buildSyntheticGGUF(t, nil, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if f.Version != 3 {
		t.Errorf("Version = %d, want 3", f.Version)
	}
	if len(f.Metadata) != 0 {
		t.Errorf("Metadata count = %d, want 0", len(f.Metadata))
	}
	if len(f.Tensors) != 0 {
		t.Errorf("Tensor count = %d, want 0", len(f.Tensors))
	}
}

func TestParse_Metadata(t *testing.T) {
	meta := map[string]any{
		"general.architecture": "llama",
		"llama.block_count":    uint32(32),
		"general.name":         "test-model",
	}
	r := buildSyntheticGGUF(t, meta, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if s, ok := f.GetString("general.architecture"); !ok || s != "llama" {
		t.Errorf("architecture = %q, %v, want llama", s, ok)
	}
	if u, ok := f.GetUint32("llama.block_count"); !ok || u != 32 {
		t.Errorf("block_count = %d, %v, want 32", u, ok)
	}
	if s, ok := f.GetString("general.name"); !ok || s != "test-model" {
		t.Errorf("name = %q, %v, want test-model", s, ok)
	}
}

func TestParse_TensorInfo(t *testing.T) {
	tensors := []TensorInfo{
		{
			Name:       "blk.0.attn_q.weight",
			Dimensions: []uint64{4096, 4096},
			Type:       GGMLTypeQ4_0,
			Offset:     0,
		},
		{
			Name:       "blk.0.attn_k.weight",
			Dimensions: []uint64{4096, 1024},
			Type:       GGMLTypeF16,
			Offset:     8388608,
		},
	}
	r := buildSyntheticGGUF(t, nil, tensors)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if len(f.Tensors) != 2 {
		t.Fatalf("Tensor count = %d, want 2", len(f.Tensors))
	}

	ti := f.Tensors[0]
	if ti.Name != "blk.0.attn_q.weight" {
		t.Errorf("tensor[0].Name = %q, want blk.0.attn_q.weight", ti.Name)
	}
	if len(ti.Dimensions) != 2 || ti.Dimensions[0] != 4096 || ti.Dimensions[1] != 4096 {
		t.Errorf("tensor[0].Dimensions = %v, want [4096 4096]", ti.Dimensions)
	}
	if ti.Type != GGMLTypeQ4_0 {
		t.Errorf("tensor[0].Type = %d, want %d (Q4_0)", ti.Type, GGMLTypeQ4_0)
	}

	ti2 := f.Tensors[1]
	if ti2.Name != "blk.0.attn_k.weight" {
		t.Errorf("tensor[1].Name = %q", ti2.Name)
	}
	if ti2.Offset != 8388608 {
		t.Errorf("tensor[1].Offset = %d, want 8388608", ti2.Offset)
	}
}

func TestParse_DataOffset(t *testing.T) {
	r := buildSyntheticGGUF(t, nil, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	// Data offset should be aligned to 32 bytes.
	if f.DataOffset%32 != 0 {
		t.Errorf("DataOffset %d is not 32-byte aligned", f.DataOffset)
	}
}

func TestParse_InvalidMagic(t *testing.T) {
	var buf bytes.Buffer
	bw(&buf, uint32(0xDEADBEEF))
	_, err := Parse(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Error("expected error for invalid magic")
	}
}

func TestParse_UnsupportedVersion(t *testing.T) {
	var buf bytes.Buffer
	bw(&buf, Magic)
	bw(&buf, uint32(1)) // v1 unsupported
	_, err := Parse(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Error("expected error for unsupported version")
	}
}

func TestParse_Truncated(t *testing.T) {
	var buf bytes.Buffer
	bw(&buf, Magic)
	_, err := Parse(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Error("expected error for truncated file")
	}
}

func TestParse_BoolMetadata(t *testing.T) {
	meta := map[string]any{
		"test.flag": true,
	}
	r := buildSyntheticGGUF(t, meta, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	v, ok := f.Metadata["test.flag"]
	if !ok {
		t.Fatal("test.flag not found")
	}
	if b, ok := v.(bool); !ok || !b {
		t.Errorf("test.flag = %v, want true", v)
	}
}

func TestParse_Float32Metadata(t *testing.T) {
	meta := map[string]any{
		"test.value": float32(3.14),
	}
	r := buildSyntheticGGUF(t, meta, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	v, ok := f.GetFloat32("test.value")
	if !ok || v < 3.13 || v > 3.15 {
		t.Errorf("test.value = %f, want ~3.14", v)
	}
}

func TestParse_ArrayMetadata(t *testing.T) {
	meta := map[string]any{
		"test.ids": []any{uint32(1), uint32(2), uint32(3)},
	}
	r := buildSyntheticGGUF(t, meta, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	v, ok := f.Metadata["test.ids"]
	if !ok {
		t.Fatal("test.ids not found")
	}
	arr, ok := v.([]any)
	if !ok || len(arr) != 3 {
		t.Fatalf("test.ids = %v, want []any of length 3", v)
	}
	for i, expected := range []uint32{1, 2, 3} {
		if arr[i] != expected {
			t.Errorf("test.ids[%d] = %v, want %d", i, arr[i], expected)
		}
	}
}

func TestGetString_Missing(t *testing.T) {
	f := &File{Metadata: map[string]any{}}
	_, ok := f.GetString("missing")
	if ok {
		t.Error("GetString(missing) should return false")
	}
}

func TestGetUint32_Missing(t *testing.T) {
	f := &File{Metadata: map[string]any{}}
	_, ok := f.GetUint32("missing")
	if ok {
		t.Error("GetUint32(missing) should return false")
	}
}

func TestGetUint32_WrongType(t *testing.T) {
	f := &File{Metadata: map[string]any{"key": "string"}}
	_, ok := f.GetUint32("key")
	if ok {
		t.Error("GetUint32 with string value should return false")
	}
}

func TestParse_TensorCountOverflow(t *testing.T) {
	var buf bytes.Buffer
	bw(&buf, Magic)
	bw(&buf, uint32(3))       // version
	bw(&buf, uint64(100_001)) // tensor count exceeds limit
	bw(&buf, uint64(0))       // metadata kv count
	_, err := Parse(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for tensor count > 100000")
	}
}

func TestParse_MetadataKVCountOverflow(t *testing.T) {
	var buf bytes.Buffer
	bw(&buf, Magic)
	bw(&buf, uint32(3))         // version
	bw(&buf, uint64(0))         // tensor count
	bw(&buf, uint64(1_000_001)) // metadata kv count exceeds limit
	_, err := Parse(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for metadata kv count > 1000000")
	}
}

// TestParse_TensorDimsExceedsMax verifies a crafted tensor declaring more
// than maxTensorDims dimensions is rejected with an error (deep-review 002,
// finding F3) rather than allocating an unbounded []uint64 or crashing
// later in shape-dependent code.
func TestParse_TensorDimsExceedsMax(t *testing.T) {
	tensors := []TensorInfo{{
		Name:       "attack.too_many_dims",
		Dimensions: make([]uint64, maxTensorDims+1), // 9 dims, all zero-valued
		Type:       GGMLTypeF32,
		Offset:     0,
	}}
	r := buildSyntheticGGUF(t, nil, tensors)
	_, err := Parse(r)
	if err == nil {
		t.Fatal("expected error for tensor with more than maxTensorDims dimensions")
	}
}

// TestParse_TensorDimsHugeCount verifies the numDims cap rejects a
// file-controlled dimension count before attempting to allocate or read any
// per-dimension values -- the actual F3 attack shape, where a corrupt file
// declares an enormous numDims (e.g. near uint32 max) that would otherwise
// force `make([]uint64, numDims)` to attempt a multi-gigabyte allocation.
func TestParse_TensorDimsHugeCount(t *testing.T) {
	var buf bytes.Buffer
	bw(&buf, Magic)
	bw(&buf, uint32(3)) // version
	bw(&buf, uint64(1)) // tensor count
	bw(&buf, uint64(0)) // metadata kv count
	writeTestString(&buf, "attack.huge_ndims")
	bw(&buf, uint32(0xFFFFFFFF)) // numDims: ~4 billion, no dimension data follows

	_, err := Parse(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for tensor with huge (file-controlled) dimension count")
	}
}

// TestParse_TensorDimsAtMax verifies exactly maxTensorDims dimensions is
// still accepted -- the cap must not be stricter than documented.
func TestParse_TensorDimsAtMax(t *testing.T) {
	dims := make([]uint64, maxTensorDims)
	for i := range dims {
		dims[i] = 2
	}
	tensors := []TensorInfo{{
		Name:       "test.at_max_dims",
		Dimensions: dims,
		Type:       GGMLTypeF32,
		Offset:     0,
	}}
	r := buildSyntheticGGUF(t, nil, tensors)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: unexpected error at exactly maxTensorDims dimensions: %v", err)
	}
	if len(f.Tensors) != 1 || len(f.Tensors[0].Dimensions) != maxTensorDims {
		t.Fatalf("Tensors[0].Dimensions length = %d, want %d", len(f.Tensors[0].Dimensions), maxTensorDims)
	}
}

// TestParse_TensorDimsLegitimateShapes verifies real-world tensor ranks
// (1D bias through 4D conv-like kernels) all parse cleanly and are
// unaffected by the maxTensorDims cap.
func TestParse_TensorDimsLegitimateShapes(t *testing.T) {
	tests := []struct {
		name string
		dims []uint64
	}{
		{"1D bias", []uint64{4096}},
		{"2D dense weight", []uint64{4096, 4096}},
		{"3D conv-like kernel", []uint64{64, 3, 3}},
		{"4D conv kernel", []uint64{64, 32, 3, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensors := []TensorInfo{{
				Name:       "test." + tt.name,
				Dimensions: tt.dims,
				Type:       GGMLTypeF32,
				Offset:     0,
			}}
			r := buildSyntheticGGUF(t, nil, tensors)
			f, err := Parse(r)
			if err != nil {
				t.Fatalf("Parse: unexpected error: %v", err)
			}
			if len(f.Tensors) != 1 || len(f.Tensors[0].Dimensions) != len(tt.dims) {
				t.Fatalf("Dimensions = %v, want length %d", f.Tensors[0].Dimensions, len(tt.dims))
			}
		})
	}
}
