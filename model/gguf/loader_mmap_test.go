package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/tensor"
)

// TestLoadTensorsMmap_F32 creates a minimal GGUF-like setup with F32 tensors
// and verifies that LoadTensorsMmap creates MmapStorage-backed tensors that
// dequantize correctly.
func TestLoadTensorsMmap_F32(t *testing.T) {
	// Create raw F32 data for a [2, 3] tensor (6 elements).
	values := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:i*4+4], math.Float32bits(v))
	}

	// Simulate a mapped file: some header bytes + tensor data.
	headerSize := 64
	mapped := make([]byte, headerSize+len(raw))
	copy(mapped[headerSize:], raw)

	// Build a minimal File struct.
	gf := &File{
		DataOffset: int64(headerSize),
		Tensors: []TensorInfo{
			{
				Name:       "test.weight",
				Dimensions: []uint64{3, 2}, // GGML order: columns=3, rows=2
				Type:       GGMLTypeF32,
				Offset:     0,
			},
		},
	}

	result, err := LoadTensorsMmap(gf, mapped)
	if err != nil {
		t.Fatalf("LoadTensorsMmap: %v", err)
	}

	tt, ok := result["test.weight"]
	if !ok {
		t.Fatal("tensor 'test.weight' not found in result")
	}

	// Shape should be reversed from GGML order: [2, 3].
	shape := tt.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape = %v, want [2, 3]", shape)
	}

	// Verify data matches.
	got := tt.Data()
	for i, want := range values {
		if got[i] != want {
			t.Errorf("Data()[%d] = %v, want %v", i, got[i], want)
		}
	}

	// Verify the storage is MmapStorage.
	if _, ok := tt.GetStorage().(*tensor.MmapStorage); !ok {
		t.Errorf("storage type = %T, want *tensor.MmapStorage", tt.GetStorage())
	}
}

// TestLoadTensorsMmap_Q4_0 tests mmap loading of Q4_0 quantized tensors.
func TestLoadTensorsMmap_Q4_0(t *testing.T) {
	// Create Q4_0 data for 64 elements (2 blocks).
	values := make([]float32, 64)
	for i := range values {
		values[i] = float32(i-32) * 0.1
	}

	// Quantize using the existing Q4 quantizer to get reference raw bytes.
	q4 := tensor.QuantizeQ4(values)
	raw := q4.RawBytes()

	// Create mapped data with header.
	headerSize := 128
	mapped := make([]byte, headerSize+len(raw))
	copy(mapped[headerSize:], raw)

	gf := &File{
		DataOffset: int64(headerSize),
		Tensors: []TensorInfo{
			{
				Name:       "test.q4weight",
				Dimensions: []uint64{64}, // 1D tensor
				Type:       GGMLTypeQ4_0,
				Offset:     0,
			},
		},
	}

	result, err := LoadTensorsMmap(gf, mapped)
	if err != nil {
		t.Fatalf("LoadTensorsMmap: %v", err)
	}

	tt := result["test.q4weight"]
	if tt == nil {
		t.Fatal("tensor not found")
	}

	// Compare mmap dequantization against existing Q4Storage dequantization.
	expected := q4.Slice()
	got := tt.Data()
	for i := range expected {
		if diff := math.Abs(float64(got[i] - expected[i])); diff > 1e-5 {
			t.Errorf("Data()[%d] = %v, want %v (diff=%v)", i, got[i], expected[i], diff)
		}
	}
}

// TestLoadTensorsMmap_Q8_0 tests mmap loading of Q8_0 quantized tensors.
func TestLoadTensorsMmap_Q8_0(t *testing.T) {
	values := make([]float32, 64)
	for i := range values {
		values[i] = float32(i-32) * 0.05
	}

	// Build Q8_0 raw bytes in GGUF format (34 bytes/block with fp16 scale).
	const blockSize = 32
	const blockBytes = 34
	n := len(values)
	nBlocks := (n + blockSize - 1) / blockSize
	raw := make([]byte, nBlocks*blockBytes)

	for bi := range nBlocks {
		offset := bi * blockSize
		var absMax float32
		for j := range blockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = values[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}
		var scale float32
		if absMax > 0 {
			scale = absMax / 127.0
		}
		off := bi * blockBytes
		fp16Scale := float16.FromFloat32(scale)
		binary.LittleEndian.PutUint16(raw[off:off+2], fp16Scale.Bits())
		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for j := range blockSize {
			var v float32
			if offset+j < n {
				v = values[offset+j]
			}
			q := int(math.Round(float64(v * invScale)))
			if q < -128 {
				q = -128
			}
			if q > 127 {
				q = 127
			}
			raw[off+2+j] = byte(int8(q))
		}
	}

	headerSize := 64
	mapped := make([]byte, headerSize+len(raw))
	copy(mapped[headerSize:], raw)

	gf := &File{
		DataOffset: int64(headerSize),
		Tensors: []TensorInfo{
			{
				Name:       "test.q8weight",
				Dimensions: []uint64{64},
				Type:       GGMLTypeQ8_0,
				Offset:     0,
			},
		},
	}

	result, err := LoadTensorsMmap(gf, mapped)
	if err != nil {
		t.Fatalf("LoadTensorsMmap: %v", err)
	}

	tt := result["test.q8weight"]
	got := tt.Data()
	for i, v := range values {
		if diff := math.Abs(float64(got[i] - v)); diff > 0.02 {
			t.Errorf("Data()[%d] = %v, want ~%v (diff=%v)", i, got[i], v, diff)
		}
	}
}

// TestLoadTensorsMmap_MultipleTensors tests loading multiple tensors from the same mapped region.
func TestLoadTensorsMmap_MultipleTensors(t *testing.T) {
	// Two F32 tensors back-to-back.
	values1 := []float32{1.0, 2.0, 3.0, 4.0}
	values2 := []float32{10.0, 20.0, 30.0}
	raw1 := make([]byte, len(values1)*4)
	raw2 := make([]byte, len(values2)*4)
	for i, v := range values1 {
		binary.LittleEndian.PutUint32(raw1[i*4:i*4+4], math.Float32bits(v))
	}
	for i, v := range values2 {
		binary.LittleEndian.PutUint32(raw2[i*4:i*4+4], math.Float32bits(v))
	}

	headerSize := 32
	mapped := make([]byte, headerSize+len(raw1)+len(raw2))
	copy(mapped[headerSize:], raw1)
	copy(mapped[headerSize+len(raw1):], raw2)

	gf := &File{
		DataOffset: int64(headerSize),
		Tensors: []TensorInfo{
			{
				Name:       "t1",
				Dimensions: []uint64{4},
				Type:       GGMLTypeF32,
				Offset:     0,
			},
			{
				Name:       "t2",
				Dimensions: []uint64{3},
				Type:       GGMLTypeF32,
				Offset:     uint64(len(raw1)),
			},
		},
	}

	result, err := LoadTensorsMmap(gf, mapped)
	if err != nil {
		t.Fatalf("LoadTensorsMmap: %v", err)
	}

	for i, want := range values1 {
		if got := result["t1"].Data()[i]; got != want {
			t.Errorf("t1.Data()[%d] = %v, want %v", i, got, want)
		}
	}
	for i, want := range values2 {
		if got := result["t2"].Data()[i]; got != want {
			t.Errorf("t2.Data()[%d] = %v, want %v", i, got, want)
		}
	}
}

// TestLoadTensorsMmap_OutOfBounds tests that LoadTensorsMmap returns an error
// when a tensor extends beyond the mapped region.
func TestLoadTensorsMmap_OutOfBounds(t *testing.T) {
	mapped := make([]byte, 100)
	gf := &File{
		DataOffset: 50,
		Tensors: []TensorInfo{
			{
				Name:       "too_big",
				Dimensions: []uint64{100},
				Type:       GGMLTypeF32,
				Offset:     0,
			},
		},
	}

	_, err := LoadTensorsMmap(gf, mapped)
	if err == nil {
		t.Fatal("expected error for out-of-bounds tensor, got nil")
	}
}

func TestLoadTensorsMmap_TQ2_0(t *testing.T) {
	// Create ternary data: 8 values {-1, 0, 1, 1, 0, -1, 1, 0}.
	values := []int8{-1, 0, 1, 1, 0, -1, 1, 0}
	ts := tensor.NewTernaryStorageFrom(values)
	raw := ts.RawBytes()

	headerSize := 64
	mapped := make([]byte, headerSize+len(raw))
	copy(mapped[headerSize:], raw)

	gf := &File{
		DataOffset: int64(headerSize),
		Tensors: []TensorInfo{
			{
				Name:       "test.ternary",
				Dimensions: []uint64{8},
				Type:       GGMLTypeTQ2_0,
				Offset:     0,
			},
		},
	}

	loaded, err := LoadTensorsMmap(gf, mapped)
	if err != nil {
		t.Fatalf("LoadTensorsMmap: %v", err)
	}

	tns := loaded["test.ternary"]
	if tns == nil {
		t.Fatal("tensor test.ternary not found")
	}

	// Verify TernaryStorage is preserved so MatMul can dispatch to ternary GEMV.
	if _, ok := tns.GetStorage().(*tensor.TernaryStorage); !ok {
		t.Errorf("expected TernaryStorage, got %T", tns.GetStorage())
	}

	got := tns.Data()
	for i, want := range values {
		if got[i] != float32(want) {
			t.Errorf("index %d: got %v, want %v", i, got[i], float32(want))
		}
	}
}
