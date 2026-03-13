package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/tensor"
)

// buildGGUFWithTensors creates a synthetic GGUF file with tensor data.
func buildGGUFWithTensors(t *testing.T, tensors []TensorInfo, tensorData []byte) *bytes.Reader {
	t.Helper()
	var buf bytes.Buffer

	bw(&buf, Magic)
	bw(&buf, uint32(3)) // version
	bw(&buf, uint64(len(tensors)))
	bw(&buf, uint64(0)) // no metadata

	for _, ti := range tensors {
		writeTestString(&buf, ti.Name)
		bw(&buf, uint32(len(ti.Dimensions)))
		for _, d := range ti.Dimensions {
			bw(&buf, d)
		}
		bw(&buf, uint32(ti.Type))
		bw(&buf, ti.Offset)
	}

	// Pad to 32-byte alignment.
	pos := buf.Len()
	const alignment = 32
	padded := (pos + alignment - 1) / alignment * alignment
	for range padded - pos {
		buf.WriteByte(0)
	}

	// Append tensor data.
	buf.Write(tensorData)

	return bytes.NewReader(buf.Bytes())
}

func TestLoadTensors_F32(t *testing.T) {
	// Create a 2x2 F32 tensor.
	data := make([]byte, 4*4)
	vals := []float32{1.0, 2.0, 3.0, 4.0}
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}

	tensors := []TensorInfo{{
		Name:       "test.weight",
		Dimensions: []uint64{2, 2},
		Type:       GGMLTypeF32,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, data)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns, ok := loaded["test.weight"]
	if !ok {
		t.Fatal("tensor test.weight not found")
	}
	if tns.Shape()[0] != 2 || tns.Shape()[1] != 2 {
		t.Errorf("shape = %v, want [2 2]", tns.Shape())
	}

	got := tns.Data()
	for i, want := range vals {
		if got[i] != want {
			t.Errorf("index %d: got %v, want %v", i, got[i], want)
		}
	}
}

func TestLoadTensors_F16(t *testing.T) {
	// Create a 1x4 F16 tensor.
	vals := []float32{1.0, -0.5, 0.25, 3.14}
	data := make([]byte, 2*len(vals))
	for i, v := range vals {
		f16 := float16.FromFloat32(v)
		binary.LittleEndian.PutUint16(data[i*2:], f16.Bits())
	}

	tensors := []TensorInfo{{
		Name:       "test.f16",
		Dimensions: []uint64{4},
		Type:       GGMLTypeF16,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, data)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.f16"]
	got := tns.Data()
	for i, want := range vals {
		// F16 has limited precision.
		diff := float32(math.Abs(float64(got[i] - want)))
		if diff > 0.01 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want, diff)
		}
	}
}

func TestLoadTensors_Q4_0(t *testing.T) {
	// Create Q4_0 tensor: quantize known data, serialize as GGUF blocks.
	src := make([]float32, 64) // 2 blocks
	for i := range src {
		src[i] = float32(i-32) / 32.0
	}

	q4 := tensor.QuantizeQ4(src)
	rawBytes := q4.RawBytes()

	tensors := []TensorInfo{{
		Name:       "test.q4",
		Dimensions: []uint64{64},
		Type:       GGMLTypeQ4_0,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, rawBytes)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.q4"]
	if tns.Shape()[0] != 64 {
		t.Errorf("shape = %v, want [64]", tns.Shape())
	}

	// Verify storage is Q4Storage (not dequantized to float32).
	if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
		t.Errorf("expected Q4Storage, got %T", tns.GetStorage())
	}

	// Verify dequantized values match original quantization.
	got := tns.Data()
	ref := q4.Slice()
	for i := range ref {
		if got[i] != ref[i] {
			t.Errorf("index %d: got %v, want %v", i, got[i], ref[i])
		}
	}
}

func TestLoadTensors_Q8_0(t *testing.T) {
	// GGUF Q8_0 format: 2-byte float16 scale + 32 int8 quants = 34 bytes per block.
	// Create a synthetic Q8_0 block manually.
	src := make([]float32, 32)
	for i := range src {
		src[i] = float32(i-16) / 16.0 // [-1, 1]
	}

	q8 := tensor.QuantizeQ8(src)
	scale := q8.BlockScale(0)
	quants := q8.BlockQuants(0)

	// Serialize in GGUF Q8_0 format (fp16 scale).
	data := make([]byte, 34)
	f16Scale := float16.FromFloat32(scale)
	binary.LittleEndian.PutUint16(data[0:2], f16Scale.Bits())
	for i, q := range quants {
		data[2+i] = byte(q)
	}

	tensors := []TensorInfo{{
		Name:       "test.q8",
		Dimensions: []uint64{32},
		Type:       GGMLTypeQ8_0,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, data)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.q8"]

	// Verify storage is Q8Storage.
	if _, ok := tns.GetStorage().(*tensor.Q8Storage); !ok {
		t.Errorf("expected Q8Storage, got %T", tns.GetStorage())
	}

	// Verify dequantized values are close to original.
	got := tns.Data()
	for i, want := range src {
		diff := float32(math.Abs(float64(got[i] - want)))
		// fp16 scale loses some precision, so allow wider tolerance.
		if diff > 0.05 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want, diff)
		}
	}
}

func TestLoadTensors_MultipleTensors(t *testing.T) {
	// Two F32 tensors at different offsets.
	data1 := make([]byte, 8)
	binary.LittleEndian.PutUint32(data1[0:4], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(data1[4:8], math.Float32bits(2.0))

	data2 := make([]byte, 8)
	binary.LittleEndian.PutUint32(data2[0:4], math.Float32bits(3.0))
	binary.LittleEndian.PutUint32(data2[4:8], math.Float32bits(4.0))

	var combined []byte
	combined = append(combined, data1...)
	combined = append(combined, data2...)

	tensors := []TensorInfo{
		{Name: "a", Dimensions: []uint64{2}, Type: GGMLTypeF32, Offset: 0},
		{Name: "b", Dimensions: []uint64{2}, Type: GGMLTypeF32, Offset: 8},
	}

	r := buildGGUFWithTensors(t, tensors, combined)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	if len(loaded) != 2 {
		t.Fatalf("expected 2 tensors, got %d", len(loaded))
	}

	aSlice := loaded["a"].Data()
	if aSlice[0] != 1.0 || aSlice[1] != 2.0 {
		t.Errorf("tensor a = %v, want [1 2]", aSlice)
	}

	bSlice := loaded["b"].Data()
	if bSlice[0] != 3.0 || bSlice[1] != 4.0 {
		t.Errorf("tensor b = %v, want [3 4]", bSlice)
	}
}

func TestLoadTensors_Empty(t *testing.T) {
	r := buildGGUFWithTensors(t, nil, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}
	if len(loaded) != 0 {
		t.Errorf("expected 0 tensors, got %d", len(loaded))
	}
}

func TestLoadTensors_UnsupportedType(t *testing.T) {
	data := make([]byte, 16)
	tensors := []TensorInfo{{
		Name:       "test.unsupported",
		Dimensions: []uint64{4},
		Type:       GGMLType(99),
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, data)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	_, err = LoadTensors(f, r)
	if err == nil {
		t.Error("expected error for unsupported tensor type")
	}
}

func TestLoadTensors_BF16(t *testing.T) {
	// Create a 1x4 BF16 tensor using float16.BFloat16FromFloat32.
	vals := []float32{1.0, -0.5, 0.25, 3.14}
	data := make([]byte, 2*len(vals))
	for i, v := range vals {
		bf := float16.BFloat16FromFloat32(v)
		binary.LittleEndian.PutUint16(data[i*2:], bf.Bits())
	}

	tensors := []TensorInfo{{
		Name:       "test.bf16",
		Dimensions: []uint64{4},
		Type:       GGMLTypeBF16,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, data)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.bf16"]
	if tns == nil {
		t.Fatal("tensor test.bf16 not found")
	}

	// Verify storage is BFloat16Storage.
	if _, ok := tns.GetStorage().(*tensor.BFloat16Storage); !ok {
		t.Errorf("expected BFloat16Storage, got %T", tns.GetStorage())
	}

	// Verify dequantized values are close to original.
	got := tns.Data()
	for i, want := range vals {
		diff := float32(math.Abs(float64(got[i] - want)))
		// BF16 has ~7-bit mantissa, allow some tolerance.
		if diff > 0.02 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want, diff)
		}
	}

	// Verify memory is halved vs F32.
	bf16Storage := tns.GetStorage().(*tensor.BFloat16Storage)
	if bf16Storage.ByteSize() != len(vals)*2 {
		t.Errorf("ByteSize() = %d, want %d", bf16Storage.ByteSize(), len(vals)*2)
	}
}
