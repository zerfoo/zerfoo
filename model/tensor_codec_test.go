package model

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
)

// --- EncodeTensor tests for all types ---

func TestEncodeTensor_Float64(t *testing.T) {
	data := []float64{1.5, -2.75, 3.125}
	src, err := tensor.New[float64]([]int{3}, data)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	if encoded.Dtype != zmf.Tensor_FLOAT64 {
		t.Errorf("expected FLOAT64, got %v", encoded.Dtype)
	}
	if len(encoded.Data) != 3*8 {
		t.Errorf("expected 24 bytes, got %d", len(encoded.Data))
	}
	// Verify round-trip
	for i, want := range data {
		bits := binary.LittleEndian.Uint64(encoded.Data[i*8 : i*8+8])
		got := math.Float64frombits(bits)
		if got != want {
			t.Errorf("element %d: got %v, want %v", i, got, want)
		}
	}
}

func TestEncodeTensor_Float16(t *testing.T) {
	f16vals := []float16.Float16{
		float16.FromFloat32(1.0),
		float16.FromFloat32(2.0),
	}
	src, err := tensor.New[float16.Float16]([]int{2}, f16vals)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	if encoded.Dtype != zmf.Tensor_FLOAT16 {
		t.Errorf("expected FLOAT16, got %v", encoded.Dtype)
	}
	if len(encoded.Data) != 2*2 {
		t.Errorf("expected 4 bytes, got %d", len(encoded.Data))
	}
}

func TestEncodeTensor_Int8(t *testing.T) {
	data := []int8{-128, 0, 127}
	src, err := tensor.New[int8]([]int{3}, data)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	if encoded.Dtype != zmf.Tensor_INT8 {
		t.Errorf("expected INT8, got %v", encoded.Dtype)
	}
	if len(encoded.Data) != 3 {
		t.Errorf("expected 3 bytes, got %d", len(encoded.Data))
	}
}

func TestEncodeTensor_Int16(t *testing.T) {
	data := []int16{-32768, 0, 32767}
	src, err := tensor.New[int16]([]int{3}, data)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	if encoded.Dtype != zmf.Tensor_INT16 {
		t.Errorf("expected INT16, got %v", encoded.Dtype)
	}
	if len(encoded.Data) != 3*2 {
		t.Errorf("expected 6 bytes, got %d", len(encoded.Data))
	}
}

func TestEncodeTensor_Int32(t *testing.T) {
	data := []int32{-100, 0, 100}
	src, err := tensor.New[int32]([]int{3}, data)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	if encoded.Dtype != zmf.Tensor_INT32 {
		t.Errorf("expected INT32, got %v", encoded.Dtype)
	}
	if len(encoded.Data) != 3*4 {
		t.Errorf("expected 12 bytes, got %d", len(encoded.Data))
	}
}

func TestEncodeTensor_Int64(t *testing.T) {
	data := []int64{-1000, 0, 1000}
	src, err := tensor.New[int64]([]int{3}, data)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	if encoded.Dtype != zmf.Tensor_INT64 {
		t.Errorf("expected INT64, got %v", encoded.Dtype)
	}
	if len(encoded.Data) != 3*8 {
		t.Errorf("expected 24 bytes, got %d", len(encoded.Data))
	}
}

// --- DecodeTensor tests ---

func TestDecodeTensor_Float16(t *testing.T) {
	// Encode two float16 values
	f16vals := []float16.Float16{
		float16.FromFloat32(1.5),
		float16.FromFloat32(-2.0),
	}
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint16(rawData[0:2], f16vals[0].Bits())
	binary.LittleEndian.PutUint16(rawData[2:4], f16vals[1].Bits())

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT16,
		Shape: []int64{2},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float16.Float16](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	if decoded.Size() != 2 {
		t.Errorf("expected size 2, got %d", decoded.Size())
	}
}

func TestDecodeTensor_Float16_ToFloat32(t *testing.T) {
	f16val := float16.FromFloat32(3.0)
	rawData := make([]byte, 2)
	binary.LittleEndian.PutUint16(rawData[0:2], f16val.Bits())

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT16,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	data := decoded.Data()
	if len(data) != 1 {
		t.Fatalf("expected 1 element, got %d", len(data))
	}
	if data[0] != 3.0 {
		t.Errorf("expected 3.0, got %v", data[0])
	}
}

func TestDecodeTensor_Int8(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT8,
		Shape: []int64{3},
		Data:  []byte{0x80, 0x00, 0x7F}, // -128, 0, 127
	}

	decoded, err := DecodeTensor[int8](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	data := decoded.Data()
	expected := []int8{-128, 0, 127}
	for i, want := range expected {
		if data[i] != want {
			t.Errorf("element %d: got %d, want %d", i, data[i], want)
		}
	}
}

func TestDecodeTensor_Float32_InvalidDataLength(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT32,
		Shape: []int64{1},
		Data:  []byte{0x01, 0x02, 0x03}, // 3 bytes, not multiple of 4
	}

	_, err := DecodeTensor[float32](proto)
	if err == nil {
		t.Error("expected error for invalid float32 data length")
	}
}

func TestDecodeTensor_Float16_InvalidDataLength(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT16,
		Shape: []int64{1},
		Data:  []byte{0x01}, // 1 byte, not multiple of 2
	}

	_, err := DecodeTensor[float16.Float16](proto)
	if err == nil {
		t.Error("expected error for invalid float16 data length")
	}
}

func TestDecodeTensor_Int8_InvalidDataLength(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT8,
		Shape: []int64{3},
		Data:  []byte{0x01, 0x02}, // 2 bytes, expected 3
	}

	_, err := DecodeTensor[int8](proto)
	if err == nil {
		t.Error("expected error for invalid int8 data length")
	}
}

func TestDecodeTensor_UnsupportedDtype(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_DataType(999),
		Shape: []int64{1},
		Data:  []byte{0x01},
	}

	_, err := DecodeTensor[float32](proto)
	if err == nil {
		t.Error("expected error for unsupported dtype")
	}
}

func TestDecodeTensor_Float32_UnsupportedDestType(t *testing.T) {
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint32(rawData, math.Float32bits(1.0))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT32,
		Shape: []int64{1},
		Data:  rawData,
	}

	// int8 is not a valid destination for FLOAT32 source
	_, err := DecodeTensor[int8](proto)
	if err == nil {
		t.Error("expected error for unsupported destination type for FLOAT32")
	}
}

func TestDecodeTensor_Float16_UnsupportedDestType(t *testing.T) {
	rawData := make([]byte, 2)
	binary.LittleEndian.PutUint16(rawData, float16.FromFloat32(1.0).Bits())

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT16,
		Shape: []int64{1},
		Data:  rawData,
	}

	// int8 is not a valid destination for FLOAT16 source
	_, err := DecodeTensor[int8](proto)
	if err == nil {
		t.Error("expected error for unsupported destination type for FLOAT16")
	}
}

func TestDecodeTensor_Int8_UnsupportedDestType(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT8,
		Shape: []int64{1},
		Data:  []byte{0x01},
	}

	// float32 is not a valid destination for INT8 source
	_, err := DecodeTensor[float32](proto)
	if err == nil {
		t.Error("expected error for unsupported destination type for INT8")
	}
}

// --- Round-trip tests for additional types ---

func TestEncodeDecode_Int8_RoundTrip(t *testing.T) {
	data := []int8{-128, -1, 0, 1, 127}
	src, _ := tensor.New[int8]([]int{5}, data)
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	decoded, err := DecodeTensor[int8](encoded)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	decodedData := decoded.Data()
	for i, want := range data {
		if decodedData[i] != want {
			t.Errorf("element %d: got %d, want %d", i, decodedData[i], want)
		}
	}
}

func TestEncodeDecode_Float16_RoundTrip(t *testing.T) {
	f16vals := []float16.Float16{
		float16.FromFloat32(1.0),
		float16.FromFloat32(-2.5),
		float16.FromFloat32(0.0),
	}
	src, _ := tensor.New[float16.Float16]([]int{3}, f16vals)
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	decoded, err := DecodeTensor[float16.Float16](encoded)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	decodedData := decoded.Data()
	for i, want := range f16vals {
		if decodedData[i].Bits() != want.Bits() {
			t.Errorf("element %d: bits differ", i)
		}
	}
}

func TestDecodeTensor_FLOAT64_RoundTrip(t *testing.T) {
	data := []float64{1.5, -2.75}
	src, _ := tensor.New[float64]([]int{2}, data)
	encoded, err := EncodeTensor(src)
	if err != nil {
		t.Fatalf("EncodeTensor failed: %v", err)
	}
	decoded, err := DecodeTensor[float64](encoded)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	got := decoded.Data()
	for i, want := range data {
		if got[i] != want {
			t.Errorf("element %d: got %v, want %v", i, got[i], want)
		}
	}
}

// Float32 source -> BFloat16 destination
func TestDecodeTensor_Float32_ToBFloat16(t *testing.T) {
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint32(rawData, math.Float32bits(2.0))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT32,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float16.BFloat16](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	if decoded.Size() != 1 {
		t.Errorf("expected size 1, got %d", decoded.Size())
	}
}

// Float16 source -> BFloat16 destination
func TestDecodeTensor_Float16_ToBFloat16(t *testing.T) {
	f16val := float16.FromFloat32(4.0)
	rawData := make([]byte, 2)
	binary.LittleEndian.PutUint16(rawData, f16val.Bits())

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT16,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float16.BFloat16](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	if decoded.Size() != 1 {
		t.Errorf("expected size 1, got %d", decoded.Size())
	}
}

// Float32 source -> Float16 destination
func TestDecodeTensor_Float32_ToFloat16(t *testing.T) {
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint32(rawData, math.Float32bits(5.0))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT32,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float16.Float16](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	if decoded.Size() != 1 {
		t.Errorf("expected size 1, got %d", decoded.Size())
	}
}

// --- New dtype decode tests (T37.3) ---

func TestDecodeTensor_BFloat16_ToBFloat16(t *testing.T) {
	bf1 := float16.BFloat16FromFloat32(1.0)
	bf2 := float16.BFloat16FromFloat32(2.0)
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint16(rawData[0:2], bf1.Bits())
	binary.LittleEndian.PutUint16(rawData[2:4], bf2.Bits())

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_BFLOAT16,
		Shape: []int64{2},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float16.BFloat16](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	data := decoded.Data()
	if data[0].Bits() != bf1.Bits() {
		t.Errorf("element 0: bits differ")
	}
	if data[1].Bits() != bf2.Bits() {
		t.Errorf("element 1: bits differ")
	}
}

func TestDecodeTensor_BFloat16_ToFloat32(t *testing.T) {
	bf := float16.BFloat16FromFloat32(3.0)
	rawData := make([]byte, 2)
	binary.LittleEndian.PutUint16(rawData, bf.Bits())

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_BFLOAT16,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	data := decoded.Data()
	if data[0] != 3.0 {
		t.Errorf("expected 3.0, got %v", data[0])
	}
}

func TestDecodeTensor_BFloat16_InvalidDataLength(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_BFLOAT16,
		Shape: []int64{1},
		Data:  []byte{0x01}, // 1 byte, not multiple of 2
	}
	_, err := DecodeTensor[float16.BFloat16](proto)
	if err == nil {
		t.Error("expected error for invalid bfloat16 data length")
	}
}

func TestDecodeTensor_INT32_ToInt32(t *testing.T) {
	rawData := make([]byte, 8)
	var n0, n1 int32 = -100, 200
	binary.LittleEndian.PutUint32(rawData[0:4], uint32(n0))
	binary.LittleEndian.PutUint32(rawData[4:8], uint32(n1))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT32,
		Shape: []int64{2},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[int32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	data := decoded.Data()
	if data[0] != -100 || data[1] != 200 {
		t.Errorf("expected [-100, 200], got %v", data)
	}
}

func TestDecodeTensor_INT32_ToFloat32(t *testing.T) {
	rawData := make([]byte, 4)
	binary.LittleEndian.PutUint32(rawData, uint32(int32(42)))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT32,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	if decoded.Data()[0] != 42.0 {
		t.Errorf("expected 42.0, got %v", decoded.Data()[0])
	}
}

func TestDecodeTensor_INT32_InvalidDataLength(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT32,
		Shape: []int64{1},
		Data:  []byte{0x01, 0x02, 0x03}, // 3 bytes, not multiple of 4
	}
	_, err := DecodeTensor[int32](proto)
	if err == nil {
		t.Error("expected error for invalid int32 data length")
	}
}

func TestDecodeTensor_INT64_ToInt64(t *testing.T) {
	rawData := make([]byte, 16)
	var v0, v1 int64 = -1000, 2000
	binary.LittleEndian.PutUint64(rawData[0:8], uint64(v0))
	binary.LittleEndian.PutUint64(rawData[8:16], uint64(v1))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT64,
		Shape: []int64{2},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[int64](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	data := decoded.Data()
	if data[0] != -1000 || data[1] != 2000 {
		t.Errorf("expected [-1000, 2000], got %v", data)
	}
}

func TestDecodeTensor_INT64_ToFloat32(t *testing.T) {
	rawData := make([]byte, 8)
	binary.LittleEndian.PutUint64(rawData, uint64(int64(512)))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT64,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	if decoded.Data()[0] != 512.0 {
		t.Errorf("expected 512.0, got %v", decoded.Data()[0])
	}
}

func TestDecodeTensor_INT64_InvalidDataLength(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_INT64,
		Shape: []int64{1},
		Data:  []byte{0x01, 0x02, 0x03, 0x04}, // 4 bytes, not multiple of 8
	}
	_, err := DecodeTensor[int64](proto)
	if err == nil {
		t.Error("expected error for invalid int64 data length")
	}
}

func TestDecodeTensor_FLOAT64_ToFloat64(t *testing.T) {
	val := 1.5
	rawData := make([]byte, 8)
	binary.LittleEndian.PutUint64(rawData, math.Float64bits(val))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT64,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float64](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	if decoded.Data()[0] != val {
		t.Errorf("expected %v, got %v", val, decoded.Data()[0])
	}
}

func TestDecodeTensor_FLOAT64_ToFloat32(t *testing.T) {
	val := 2.5
	rawData := make([]byte, 8)
	binary.LittleEndian.PutUint64(rawData, math.Float64bits(val))

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT64,
		Shape: []int64{1},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	if decoded.Data()[0] != float32(val) {
		t.Errorf("expected %v, got %v", float32(val), decoded.Data()[0])
	}
}

func TestDecodeTensor_FLOAT64_InvalidDataLength(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_FLOAT64,
		Shape: []int64{1},
		Data:  []byte{0x01, 0x02, 0x03, 0x04}, // 4 bytes, not multiple of 8
	}
	_, err := DecodeTensor[float64](proto)
	if err == nil {
		t.Error("expected error for invalid float64 data length")
	}
}

func TestDecodeTensor_UINT8_ToUint8(t *testing.T) {
	rawData := []byte{0, 127, 255}

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_UINT8,
		Shape: []int64{3},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[uint8](proto)
	if err != nil {
		t.Fatalf("DecodeTensor failed: %v", err)
	}
	data := decoded.Data()
	expected := []uint8{0, 127, 255}
	for i, want := range expected {
		if data[i] != want {
			t.Errorf("element %d: got %d, want %d", i, data[i], want)
		}
	}
}

func TestDecodeTensor_UINT8_InvalidDataLength(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_UINT8,
		Shape: []int64{3},
		Data:  []byte{0x01, 0x02}, // 2 bytes, expected 3
	}
	_, err := DecodeTensor[uint8](proto)
	if err == nil {
		t.Error("expected error for invalid uint8 data length")
	}
}

// --- Q4_0 / Q8_0 quantized decode tests ---

// encodeQ4Block creates raw Q4_0 block bytes (18 bytes) for testing.
func encodeQ4Block(scale float16.Float16, nibbles [16]byte) []byte {
	b := make([]byte, 18)
	binary.LittleEndian.PutUint16(b[0:2], scale.Bits())
	copy(b[2:], nibbles[:])
	return b
}

// encodeQ8Block creates raw Q8_0 block bytes (36 bytes) for testing.
func encodeQ8Block(scale float32, data [32]int8) []byte {
	b := make([]byte, 36)
	binary.LittleEndian.PutUint32(b[0:4], math.Float32bits(scale))
	for i, v := range data {
		b[4+i] = byte(v)
	}
	return b
}

func TestDecodeTensor_Q4_0_ToFloat32(t *testing.T) {
	// Build a single Q4_0 block: scale=0.5, all nibbles encode value 0 (offset 8).
	scale := float16.FromFloat32(0.5)
	var nibbles [16]byte
	for i := range nibbles {
		nibbles[i] = 0x88 // two zeros: (0+8) | ((0+8)<<4)
	}
	rawData := encodeQ4Block(scale, nibbles)

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_Q4_0,
		Shape: []int64{32},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor Q4_0 failed: %v", err)
	}
	if decoded.Size() != 32 {
		t.Errorf("expected size 32, got %d", decoded.Size())
	}
	// All values should be 0 (q=0, scale=0.5, result=0*0.5=0).
	for i, v := range decoded.Data() {
		if v != 0 {
			t.Errorf("element %d: got %v, want 0", i, v)
		}
	}
}

func TestDecodeTensor_Q4_0_NonZeroValues(t *testing.T) {
	scale := float16.FromFloat32(1.0)
	var nibbles [16]byte
	// In GGML split format: low nibble -> first half (pos 0-15),
	// high nibble -> second half (pos 16-31).
	// nibbles[0] = 0x79: low nibble 9 (q=1) -> pos 0, high nibble 7 (q=-1) -> pos 16.
	nibbles[0] = 0x79
	// Rest: zeros (0x88 = low 8/high 8, both q=0).
	for i := 1; i < 16; i++ {
		nibbles[i] = 0x88
	}
	rawData := encodeQ4Block(scale, nibbles)

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_Q4_0,
		Shape: []int64{32},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor Q4_0 failed: %v", err)
	}
	data := decoded.Data()
	if data[0] != 1.0 {
		t.Errorf("element 0: got %v, want 1.0", data[0])
	}
	if data[16] != -1.0 {
		t.Errorf("element 16: got %v, want -1.0", data[16])
	}
}

func TestDecodeTensor_Q4_0_TooShort(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_Q4_0,
		Shape: []int64{32},
		Data:  []byte{0x01, 0x02}, // only 2 bytes, need 18
	}
	_, err := DecodeTensor[float32](proto)
	if err == nil {
		t.Error("expected error for too-short Q4_0 data")
	}
}

func TestDecodeTensor_Q4_0_UsesQ4Storage(t *testing.T) {
	scale := float16.FromFloat32(1.0)
	var nibbles [16]byte
	nibbles[0] = 0x79 // q0=1, q1=-1
	for i := 1; i < 16; i++ {
		nibbles[i] = 0x88
	}
	rawData := encodeQ4Block(scale, nibbles)

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_Q4_0,
		Shape: []int64{32},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor Q4_0 failed: %v", err)
	}

	// The tensor should be backed by Q4Storage, not dense float32.
	q4, ok := decoded.GetStorage().(*tensor.Q4Storage)
	if !ok {
		t.Fatalf("expected Q4Storage backing, got %T", decoded.GetStorage())
	}
	if q4.NumBlocks() != 1 {
		t.Errorf("expected 1 Q4 block, got %d", q4.NumBlocks())
	}

	// Data() should still return correct dequantized values via Q4Storage.Slice().
	// In GGML split format: low nibble -> first half, high nibble -> second half.
	data := decoded.Data()
	if data[0] != 1.0 {
		t.Errorf("element 0: got %v, want 1.0", data[0])
	}
	if data[16] != -1.0 {
		t.Errorf("element 16: got %v, want -1.0", data[16])
	}
}

func TestDecodeTensor_Q4_0_UnsupportedDestType(t *testing.T) {
	scale := float16.FromFloat32(1.0)
	var nibbles [16]byte
	for i := range nibbles {
		nibbles[i] = 0x88
	}
	rawData := encodeQ4Block(scale, nibbles)

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_Q4_0,
		Shape: []int64{32},
		Data:  rawData,
	}

	_, err := DecodeTensor[int8](proto)
	if err == nil {
		t.Error("expected error for unsupported dest type for Q4_0")
	}
}

func TestDecodeTensor_Q8_0_ToFloat32(t *testing.T) {
	var data [32]int8
	data[0] = 127  // max positive
	data[1] = -128 // max negative
	// rest are 0
	rawData := encodeQ8Block(0.5, data)

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_Q8_0,
		Shape: []int64{32},
		Data:  rawData,
	}

	decoded, err := DecodeTensor[float32](proto)
	if err != nil {
		t.Fatalf("DecodeTensor Q8_0 failed: %v", err)
	}
	d := decoded.Data()
	if d[0] != 127*0.5 {
		t.Errorf("element 0: got %v, want %v", d[0], 127*0.5)
	}
	if d[1] != -128*0.5 {
		t.Errorf("element 1: got %v, want %v", d[1], -128*0.5)
	}
	if d[2] != 0 {
		t.Errorf("element 2: got %v, want 0", d[2])
	}
}

func TestDecodeTensor_Q8_0_TooShort(t *testing.T) {
	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_Q8_0,
		Shape: []int64{32},
		Data:  []byte{0x01, 0x02, 0x03, 0x04}, // only 4 bytes, need 36
	}
	_, err := DecodeTensor[float32](proto)
	if err == nil {
		t.Error("expected error for too-short Q8_0 data")
	}
}

func TestDecodeTensor_Q8_0_UnsupportedDestType(t *testing.T) {
	var data [32]int8
	rawData := encodeQ8Block(1.0, data)

	proto := &zmf.Tensor{
		Dtype: zmf.Tensor_Q8_0,
		Shape: []int64{32},
		Data:  rawData,
	}

	_, err := DecodeTensor[int8](proto)
	if err == nil {
		t.Error("expected error for unsupported dest type for Q8_0")
	}
}
