package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/ztensor/tensor"
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

func TestDecodeQ5KTensor_ReQuantizesToQ4(t *testing.T) {
	// Q5_K: 256 elements per super-block, 176 bytes per block.
	const numElements = 256
	const blockBytes = 176
	raw := make([]byte, blockBytes) // all zeros — valid Q5_K block

	tns, err := decodeQ5KTensor([]int{numElements}, numElements, raw)
	if err != nil {
		t.Fatalf("decodeQ5KTensor: %v", err)
	}

	// Verify shape.
	if tns.Shape()[0] != numElements {
		t.Errorf("shape = %v, want [%d]", tns.Shape(), numElements)
	}

	// Q5_K is re-quantized to Q4_0 for uniform fast GEMV decode path.
	if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
		t.Fatalf("expected Q4Storage (re-quantized from Q5_K), got %T", tns.GetStorage())
	}

	// All-zero Q5_K block dequantizes to all zeros.
	got := tns.Data()
	for i, v := range got {
		if v != 0 {
			t.Errorf("index %d: got %v, want 0", i, v)
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

func TestQuantizeToFP8E4M3(t *testing.T) {
	tests := []struct {
		name   string
		vals   []float32
		shape  []uint64
		maxRel float64
	}{
		{
			name:   "2x2 matrix",
			vals:   []float32{1.0, 2.0, 3.0, 4.0},
			shape:  []uint64{2, 2},
			maxRel: 0.1,
		},
		{
			name:   "1D vector",
			vals:   []float32{-10.0, 0.0, 5.0, 100.0},
			shape:  []uint64{4},
			maxRel: 0.1,
		},
		{
			name:   "powers of two",
			vals:   []float32{1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0},
			shape:  []uint64{8},
			maxRel: 0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build a GGUF file with an F32 tensor.
			data := make([]byte, len(tt.vals)*4)
			for i, v := range tt.vals {
				binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
			}

			tensors := []TensorInfo{{
				Name:       "w",
				Dimensions: tt.shape,
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

			// Quantize to FP8.
			quantized, err := QuantizeToFP8E4M3(loaded)
			if err != nil {
				t.Fatalf("QuantizeToFP8E4M3: %v", err)
			}

			tns := quantized["w"]

			// Verify storage type is FP8E4M3Storage.
			if _, ok := tns.GetStorage().(*tensor.FP8E4M3Storage); !ok {
				t.Fatalf("expected FP8E4M3Storage, got %T", tns.GetStorage())
			}

			// Verify dequantized values are close to original.
			got := tns.Data()
			for i, want := range tt.vals {
				if want == 0 {
					if got[i] != 0 {
						t.Errorf("[%d] got %g, want 0", i, got[i])
					}
					continue
				}
				rel := math.Abs(float64(got[i]-want)) / math.Abs(float64(want))
				if rel > tt.maxRel {
					t.Errorf("[%d] got %g, want %g, rel error %g > %g", i, got[i], want, rel, tt.maxRel)
				}
			}
		})
	}
}

func TestQuantizeToFP8E4M3_MemoryReduction(t *testing.T) {
	// Create an F32 tensor with 1024 elements (4096 bytes as F32).
	vals := make([]float32, 1024)
	for i := range vals {
		vals[i] = float32(i-512) / 512.0
	}

	data := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}

	tensors := []TensorInfo{{
		Name:       "w",
		Dimensions: []uint64{1024},
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

	quantized, err := QuantizeToFP8E4M3(loaded)
	if err != nil {
		t.Fatalf("QuantizeToFP8E4M3: %v", err)
	}

	fp8 := quantized["w"].GetStorage().(*tensor.FP8E4M3Storage)
	// FP8 stores 1 byte per element. Len() should equal element count.
	if fp8.Len() != 1024 {
		t.Errorf("FP8 storage Len() = %d, want 1024", fp8.Len())
	}
}

func TestQuantizeToFP8E4M3_MultipleTensors(t *testing.T) {
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

	quantized, err := QuantizeToFP8E4M3(loaded)
	if err != nil {
		t.Fatalf("QuantizeToFP8E4M3: %v", err)
	}

	for _, name := range []string{"a", "b"} {
		if _, ok := quantized[name].GetStorage().(*tensor.FP8E4M3Storage); !ok {
			t.Errorf("tensor %q: expected FP8E4M3Storage, got %T", name, quantized[name].GetStorage())
		}
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
	if len(bf16Storage.RawBytes()) != len(vals)*2 {
		t.Errorf("RawBytes length = %d, want %d", len(bf16Storage.RawBytes()), len(vals)*2)
	}
}

func TestDecodeQ6KTensor_ReQuantizesToQ4(t *testing.T) {
	// Create 256 elements (1 Q6_K super-block = 210 bytes).
	const numElements = 256
	const blockBytes = 210
	raw := make([]byte, blockBytes)

	// Q6_K block layout (210 bytes):
	//   ql: bytes [0, 128)   — low 4 bits of quants
	//   qh: bytes [128, 192) — high 2 bits of quants
	//   sc: bytes [192, 208) — int8 sub-block scales (16 values)
	//   d:  bytes [208, 210) — float16 super-block scale

	// Set super-block scale d = 1.0.
	scaleBits := float16.FromFloat32(1.0).Bits()
	binary.LittleEndian.PutUint16(raw[208:210], scaleBits)

	// Set non-zero sub-block scales (int8).
	for i := range 16 {
		raw[192+i] = byte(i + 1)
	}

	// Set non-zero quantized values in ql region (first 128 bytes).
	for i := range 128 {
		raw[i] = byte((i % 15) + 1)
	}

	shape := []int{numElements}
	tns, err := decodeQ6KTensor(shape, numElements, raw)
	if err != nil {
		t.Fatalf("decodeQ6KTensor: %v", err)
	}

	// Verify shape.
	if len(tns.Shape()) != 1 || tns.Shape()[0] != numElements {
		t.Errorf("shape = %v, want [%d]", tns.Shape(), numElements)
	}

	// Q6_K is re-quantized to Q4_0 for uniform fast GEMV decode path.
	if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
		t.Fatalf("expected Q4Storage (re-quantized from Q6_K), got %T", tns.GetStorage())
	}

	// Verify the tensor contains dequantized float32 data (not all zeros,
	// since we set non-zero scale and quants).
	data := tns.Data()
	if len(data) != numElements {
		t.Fatalf("data length = %d, want %d", len(data), numElements)
	}
	hasNonZero := false
	for _, v := range data {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("expected non-zero dequantized values")
	}

	// Verify approximate round-trip: Q6_K → F32 → Q4_0 → F32 is lossy,
	// so values should be within re-quantization tolerance of the Q6_K originals.
	q6k, err := tensor.NewQ6KStorageFromRaw(raw, numElements)
	if err != nil {
		t.Fatalf("NewQ6KStorageFromRaw: %v", err)
	}
	ref := make([]float32, numElements)
	q6k.Dequantize(ref)
	maxErr := float32(0)
	for i := range ref {
		diff := data[i] - ref[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxErr {
			maxErr = diff
		}
	}
	// Q4_0 has 4-bit precision per block of 32 values; typical error is
	// proportional to the range of values in each block.
	if maxErr > 50 {
		t.Errorf("max re-quantization error = %v, want < 50", maxErr)
	}
}

func TestLoadTensors_DimensionExceedsMaxInt32(t *testing.T) {
	// A single dimension > math.MaxInt32 should be rejected.
	tensors := []TensorInfo{{
		Name:       "test.overflow",
		Dimensions: []uint64{uint64(math.MaxInt32) + 1},
		Type:       GGMLTypeF32,
		Offset:     0,
	}}
	r := buildGGUFWithTensors(t, tensors, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	_, err = LoadTensors(f, r)
	if err == nil {
		t.Fatal("expected error for dimension > MaxInt32")
	}
}

func TestLoadTensors_TotalElementsOverflow(t *testing.T) {
	// Two dimensions that individually fit in int32 but whose product exceeds 1<<34.
	// 131072 * 131073 = 17,179,999,296 > 17,179,869,184 (1<<34)
	tensors := []TensorInfo{{
		Name:       "test.overflow",
		Dimensions: []uint64{131072, 131073},
		Type:       GGMLTypeF32,
		Offset:     0,
	}}
	r := buildGGUFWithTensors(t, tensors, nil)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	_, err = LoadTensors(f, r)
	if err == nil {
		t.Fatal("expected error for total elements > 1<<34")
	}
}

func TestLoadTensors_ValidLargeDimensions(t *testing.T) {
	// Dimensions that are large but within bounds should parse without error
	// (though LoadTensors will fail at the read stage since we provide no data).
	// This verifies the overflow check itself does not reject valid sizes.
	// 4096 * 4096 = 16M elements, well within 1<<34.
	tensors := []TensorInfo{{
		Name:       "test.valid",
		Dimensions: []uint64{4096, 4096},
		Type:       GGMLTypeF32,
		Offset:     0,
	}}
	r := buildGGUFWithTensors(t, tensors, make([]byte, 4096*4096*4))
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	_, err = LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors should succeed for valid dimensions: %v", err)
	}
}

func TestQuantizeToFP8E4M3_Skips1D(t *testing.T) {
	// 1D tensor (norm weight) should NOT be quantized.
	norm, _ := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	// 2D tensor (weight matrix) should be quantized.
	weight, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})

	tensors := map[string]*tensor.TensorNumeric[float32]{
		"norm":   norm,
		"weight": weight,
	}

	result, err := QuantizeToFP8E4M3(tensors)
	if err != nil {
		t.Fatalf("QuantizeToFP8E4M3: %v", err)
	}

	// norm should still be plain storage (not FP8).
	if _, ok := result["norm"].GetStorage().(*tensor.FP8E4M3Storage); ok {
		t.Error("1D tensor should not be quantized to FP8")
	}

	// weight should be FP8.
	if _, ok := result["weight"].GetStorage().(*tensor.FP8E4M3Storage); !ok {
		t.Error("2D tensor should be quantized to FP8")
	}
}

func TestLoadTensors_TQ2_0(t *testing.T) {
	// Create a ternary tensor with 8 values: {-1, 0, 1, 1, 0, -1, 1, 0}.
	// TernaryStorage encoding: 00=-1, 01=0, 10=1 (2 bits each, 4 per byte).
	values := []int8{-1, 0, 1, 1, 0, -1, 1, 0}
	ts := tensor.NewTernaryStorageFrom(values)
	raw := ts.RawBytes()

	tensors := []TensorInfo{{
		Name:       "test.ternary",
		Dimensions: []uint64{8},
		Type:       GGMLTypeTQ2_0,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, raw)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.ternary"]
	if tns == nil {
		t.Fatal("tensor test.ternary not found")
	}
	if tns.Shape()[0] != 8 {
		t.Errorf("shape = %v, want [8]", tns.Shape())
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

func TestLoadTensors_TQ2_0_2D(t *testing.T) {
	// 2x4 ternary matrix.
	values := []int8{-1, 0, 1, 1, 0, -1, 1, 0}
	ts := tensor.NewTernaryStorageFrom(values)
	raw := ts.RawBytes()

	tensors := []TensorInfo{{
		Name:       "test.ternary2d",
		Dimensions: []uint64{4, 2}, // GGUF order: cols=4, rows=2
		Type:       GGMLTypeTQ2_0,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, raw)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.ternary2d"]
	if tns == nil {
		t.Fatal("tensor test.ternary2d not found")
	}
	// Reversed from GGUF order: shape should be [2, 4].
	if tns.Shape()[0] != 2 || tns.Shape()[1] != 4 {
		t.Errorf("shape = %v, want [2 4]", tns.Shape())
	}

	// Verify TernaryStorage is preserved for 2D tensors.
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

func TestGGMLTypeIQConstants(t *testing.T) {
	tests := []struct {
		name string
		typ  GGMLType
		want uint32
	}{
		{"IQ2_XXS", GGMLTypeIQ2_XXS, 16},
		{"IQ3_S", GGMLTypeIQ3_S, 21},
		{"IQ4_NL", GGMLTypeIQ4_NL, 25},
	}
	for _, tt := range tests {
		if uint32(tt.typ) != tt.want {
			t.Errorf("GGMLType%s = %d, want %d", tt.name, uint32(tt.typ), tt.want)
		}
	}
}

func TestTensorByteSize_IQTypes(t *testing.T) {
	tests := []struct {
		name        string
		typ         GGMLType
		numElements int
		wantBytes   int
	}{
		{"IQ4_NL/32", GGMLTypeIQ4_NL, 32, 18},
		{"IQ4_NL/64", GGMLTypeIQ4_NL, 64, 36},
		{"IQ4_NL/33", GGMLTypeIQ4_NL, 33, 36},  // 2 blocks
		{"IQ3_S/256", GGMLTypeIQ3_S, 256, 110},
		{"IQ3_S/512", GGMLTypeIQ3_S, 512, 220},
		{"IQ3_S/257", GGMLTypeIQ3_S, 257, 220},  // 2 blocks
		{"IQ2_XXS/256", GGMLTypeIQ2_XXS, 256, 68},
		{"IQ2_XXS/512", GGMLTypeIQ2_XXS, 512, 136},
		{"IQ2_XXS/257", GGMLTypeIQ2_XXS, 257, 136}, // 2 blocks
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := TensorByteSize(tt.typ, tt.numElements)
			if err != nil {
				t.Fatalf("TensorByteSize: %v", err)
			}
			if got != tt.wantBytes {
				t.Errorf("got %d, want %d", got, tt.wantBytes)
			}
		})
	}
}

func TestLoadTensors_IQ4_NL(t *testing.T) {
	// Create IQ4_NL tensor: 32 elements = 1 block = 18 bytes.
	const numElements = 32
	raw := make([]byte, 18)
	// Set fp16 scale = 1.0.
	scaleBits := float16.FromFloat32(1.0).Bits()
	binary.LittleEndian.PutUint16(raw[0:2], scaleBits)
	// Set nibbles to non-zero pattern.
	for i := range 16 {
		raw[2+i] = byte(i%16) | byte((i+1)%16)<<4
	}

	tensors := []TensorInfo{{
		Name:       "test.iq4nl",
		Dimensions: []uint64{uint64(numElements)},
		Type:       GGMLTypeIQ4_NL,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, raw)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.iq4nl"]
	if tns == nil {
		t.Fatal("tensor test.iq4nl not found")
	}
	if tns.Shape()[0] != numElements {
		t.Errorf("shape = %v, want [%d]", tns.Shape(), numElements)
	}

	// Should be re-quantized to Q4Storage.
	if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
		t.Errorf("expected Q4Storage, got %T", tns.GetStorage())
	}

	// Verify data is not all zeros (non-trivial scale + nibbles).
	got := tns.Data()
	hasNonZero := false
	for _, v := range got {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("expected non-zero dequantized values")
	}
}

func TestLoadTensors_IQ3_S(t *testing.T) {
	// Create IQ3_S tensor: 256 elements = 1 super-block = 110 bytes.
	const numElements = 256
	raw := make([]byte, 110)
	// Set fp16 scale = 1.0.
	scaleBits := float16.FromFloat32(1.0).Bits()
	binary.LittleEndian.PutUint16(raw[0:2], scaleBits)
	// Set non-zero sub-block scales (bytes 106-109).
	for i := range 4 {
		raw[106+i] = 0x11 // scale=1 for both nibbles
	}

	tensors := []TensorInfo{{
		Name:       "test.iq3s",
		Dimensions: []uint64{uint64(numElements)},
		Type:       GGMLTypeIQ3_S,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, raw)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.iq3s"]
	if tns == nil {
		t.Fatal("tensor test.iq3s not found")
	}
	if tns.Shape()[0] != numElements {
		t.Errorf("shape = %v, want [%d]", tns.Shape(), numElements)
	}

	// Should be re-quantized to Q4Storage.
	if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
		t.Errorf("expected Q4Storage, got %T", tns.GetStorage())
	}
}

func TestLoadTensors_IQ2_XXS(t *testing.T) {
	// Create IQ2_XXS tensor: 256 elements = 1 block = 68 bytes.
	const numElements = 256
	raw := make([]byte, 68)
	// Set float32 scale = 1.0 (first 4 bytes).
	binary.LittleEndian.PutUint32(raw[0:4], math.Float32bits(1.0))
	// Set non-zero packed data (bytes 4-67).
	for i := range 64 {
		raw[4+i] = byte(i % 256) // diverse 2-bit patterns
	}

	tensors := []TensorInfo{{
		Name:       "test.iq2xxs",
		Dimensions: []uint64{uint64(numElements)},
		Type:       GGMLTypeIQ2_XXS,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, raw)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.iq2xxs"]
	if tns == nil {
		t.Fatal("tensor test.iq2xxs not found")
	}
	if tns.Shape()[0] != numElements {
		t.Errorf("shape = %v, want [%d]", tns.Shape(), numElements)
	}

	// Should be re-quantized to Q4Storage.
	if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
		t.Errorf("expected Q4Storage, got %T", tns.GetStorage())
	}

	// Verify data has non-zero values.
	got := tns.Data()
	hasNonZero := false
	for _, v := range got {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("expected non-zero dequantized values")
	}
}

// TestDecodeIQ4NL_Accuracy verifies IQ4_NL dequantization matches the lookup table.
func TestDecodeIQ4NL_Accuracy(t *testing.T) {
	const numElements = 32
	raw := make([]byte, 18)
	scaleBits := float16.FromFloat32(2.0).Bits()
	binary.LittleEndian.PutUint16(raw[0:2], scaleBits)

	// Pack nibbles: low=15 (table[1.3312578]), high=0 (table[-1.0]) for each byte.
	for i := range 16 {
		raw[2+i] = 0x0F
	}

	tns, err := decodeIQ4NLTensor([]int{numElements}, numElements, raw)
	if err != nil {
		t.Fatalf("decodeIQ4NLTensor: %v", err)
	}

	// DequantizeIQ4NL: dst[2*i] = scale * table[low], dst[2*i+1] = scale * table[high].
	// low=15 -> table[1.3312578] * 2.0 = +2.6625 (even indices)
	// high=0 -> table[-1.0]     * 2.0 = -2.0    (odd indices)
	// After Q4 re-quant, signs should be preserved.
	got := tns.Data()
	for i := range numElements {
		if i%2 == 0 {
			if got[i] < 0 {
				t.Errorf("index %d: expected positive, got %v", i, got[i])
			}
		} else {
			if got[i] > 0 {
				t.Errorf("index %d: expected negative, got %v", i, got[i])
			}
		}
	}
}

// TestDecodeIQ3S_Accuracy verifies IQ3_S dequantization roundtrip preserves range.
func TestDecodeIQ3S_Accuracy(t *testing.T) {
	const numElements = 256
	raw := make([]byte, 110)
	scaleBits := float16.FromFloat32(0.5).Bits()
	binary.LittleEndian.PutUint16(raw[0:2], scaleBits)

	// Set sub-block scales to non-zero.
	for i := range 4 {
		raw[106+i] = 0x33 // scale=3 for both nibbles -> subScale = 0.5 * (1+2*3) = 3.5
	}

	// Set some grid indices (qs region) and signs to produce non-trivial values.
	for i := range 64 {
		raw[2+i] = byte((i * 7) % 256) // diverse grid indices
	}
	for i := range 32 {
		raw[74+i] = byte(i * 3) // diverse sign patterns
	}

	tns, err := decodeIQ3STensor([]int{numElements}, numElements, raw)
	if err != nil {
		t.Fatalf("decodeIQ3STensor: %v", err)
	}

	got := tns.Data()
	if len(got) != numElements {
		t.Fatalf("data length = %d, want %d", len(got), numElements)
	}

	// Verify some values are positive and some negative (sign bits applied).
	var pos, neg int
	for _, v := range got {
		if v > 0 {
			pos++
		} else if v < 0 {
			neg++
		}
	}
	if pos == 0 {
		t.Error("expected some positive values")
	}
	if neg == 0 {
		t.Error("expected some negative values")
	}
}

// TestDecodeIQ2XXS_Accuracy verifies IQ2_XXS dequantization roundtrip.
func TestDecodeIQ2XXS_Accuracy(t *testing.T) {
	const numElements = 256
	raw := make([]byte, 68)
	binary.LittleEndian.PutUint32(raw[0:4], math.Float32bits(0.5))

	// The grid maps 2-bit pairs: 00->-1, 01->-1/3, 10->1/3, 11->1
	// Byte 0b10_01_00_11 = 0x93 encodes [1.0, -1.0, -1/3, 1/3] * 0.5
	for i := range 64 {
		raw[4+i] = 0x93
	}

	tns, err := decodeIQ2XXSTensor([]int{numElements}, numElements, raw)
	if err != nil {
		t.Fatalf("decodeIQ2XXSTensor: %v", err)
	}

	// IQ2_XXS -> F32 -> Q4_0. Check the original dequantized values are as expected.
	s := tensor.NewIQ2XXSStorage(numElements)
	s.SetBlock(0, 0.5, raw[4:68])
	ref := s.Dequantize()

	// The Q4_0 re-quantized values should be within reasonable error of the IQ2_XXS values.
	got := tns.Data()
	maxErr := float32(0)
	for i := range ref {
		diff := got[i] - ref[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxErr {
			maxErr = diff
		}
	}
	// IQ2_XXS values are in [-0.5, 0.5], Q4_0 re-quantization error should be small.
	if maxErr > 0.3 {
		t.Errorf("max re-quantization error = %v, want < 0.3", maxErr)
	}
}

// TestLoadTensors_IQ4_NL_2D verifies 2D IQ4_NL tensor shape reversal.
func TestLoadTensors_IQ4_NL_2D(t *testing.T) {
	const rows, cols = 2, 32
	const numElements = rows * cols
	nBlocks := (numElements + 31) / 32
	raw := make([]byte, nBlocks*18)
	for bi := range nBlocks {
		scaleBits := float16.FromFloat32(1.0).Bits()
		binary.LittleEndian.PutUint16(raw[bi*18:bi*18+2], scaleBits)
		for j := range 16 {
			raw[bi*18+2+j] = byte(j)
		}
	}

	tensors := []TensorInfo{{
		Name:       "test.iq4nl2d",
		Dimensions: []uint64{cols, rows}, // GGUF order
		Type:       GGMLTypeIQ4_NL,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, raw)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["test.iq4nl2d"]
	// Reversed from GGUF order: shape should be [rows, cols].
	if tns.Shape()[0] != rows || tns.Shape()[1] != cols {
		t.Errorf("shape = %v, want [%d %d]", tns.Shape(), rows, cols)
	}
}

// TestIQuant_GEMVPath verifies that IQ-type tensors loaded through GGUF produce
// reasonable GEMV output via the Q4 re-quantization path.
func TestIQuant_GEMVPath(t *testing.T) {
	// Create a 64x32 IQ4_NL weight matrix (64 rows, 32 cols = 2 blocks of 32).
	const rows, cols = 2, 32
	const numElements = rows * cols
	nBlocks := numElements / 32
	raw := make([]byte, nBlocks*18)
	for bi := range nBlocks {
		scaleBits := float16.FromFloat32(1.0).Bits()
		binary.LittleEndian.PutUint16(raw[bi*18:bi*18+2], scaleBits)
		for j := range 16 {
			// Nibbles: low=7 (table[0.0]), high=14 (table[1.0])
			raw[bi*18+2+j] = 0xE7
		}
	}

	tensors := []TensorInfo{{
		Name:       "w",
		Dimensions: []uint64{cols, rows}, // GGUF: cols first
		Type:       GGMLTypeIQ4_NL,
		Offset:     0,
	}}

	r := buildGGUFWithTensors(t, tensors, raw)
	f, err := Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	loaded, err := LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["w"]
	q4, ok := tns.GetStorage().(*tensor.Q4Storage)
	if !ok {
		t.Fatalf("expected Q4Storage, got %T", tns.GetStorage())
	}

	// GEMV: C[1,2] = X[1,32] * W[2,32]^T -> need dequant then manual dot product.
	// This tests the full pipeline: IQ4_NL -> Q4_0 -> dequant -> matmul.
	x := make([]float32, cols)
	for i := range x {
		x[i] = 1.0 // all-ones input vector
	}

	// Dequantize and compute dot product manually.
	wF32 := make([]float32, numElements)
	q4.Dequantize(wF32)

	for row := range rows {
		dot := float32(0)
		for col := range cols {
			dot += x[col] * wF32[row*cols+col]
		}
		// With scale=1.0, nibble 7 maps to table[0.0]=0, nibble 14 maps to table[1.0]=1.0.
		// After Q4 re-quant, values won't be exact but the dot product should be non-zero
		// and finite.
		if math.IsNaN(float64(dot)) || math.IsInf(float64(dot), 0) {
			t.Errorf("row %d: dot product is %v", row, dot)
		}
	}
}

// TestLoadTensors_IQ_SingleBlock tests IQ types with exactly one block each.
func TestLoadTensors_IQ_SingleBlock(t *testing.T) {
	tests := []struct {
		name        string
		typ         GGMLType
		numElements int
		blockBytes  int
	}{
		{"IQ4_NL/1block", GGMLTypeIQ4_NL, 32, 18},
		{"IQ3_S/1block", GGMLTypeIQ3_S, 256, 110},
		{"IQ2_XXS/1block", GGMLTypeIQ2_XXS, 256, 68},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := make([]byte, tt.blockBytes)
			if tt.typ == GGMLTypeIQ2_XXS {
				// IQ2_XXS uses float32 scale.
				binary.LittleEndian.PutUint32(raw[0:4], math.Float32bits(1.0))
			} else {
				// IQ4_NL and IQ3_S use fp16 scale.
				scaleBits := float16.FromFloat32(1.0).Bits()
				binary.LittleEndian.PutUint16(raw[0:2], scaleBits)
			}

			tensors := []TensorInfo{{
				Name:       "test." + tt.name,
				Dimensions: []uint64{uint64(tt.numElements)},
				Type:       tt.typ,
				Offset:     0,
			}}

			r := buildGGUFWithTensors(t, tensors, raw)
			f, err := Parse(r)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}

			loaded, err := LoadTensors(f, r)
			if err != nil {
				t.Fatalf("LoadTensors: %v", err)
			}

			tns := loaded["test."+tt.name]
			if tns == nil {
				t.Fatalf("tensor not found")
			}
			if tns.Shape()[0] != tt.numElements {
				t.Errorf("shape = %v, want [%d]", tns.Shape(), tt.numElements)
			}
			if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
				t.Errorf("expected Q4Storage, got %T", tns.GetStorage())
			}
		})
	}
}

// TestIQuant_MultipleBlocks tests IQ types spanning multiple blocks.
func TestIQuant_MultipleBlocks(t *testing.T) {
	t.Run("IQ4_NL/3blocks", func(t *testing.T) {
		const numElements = 96 // 3 blocks of 32
		raw := make([]byte, 3*18)
		for bi := range 3 {
			scaleBits := float16.FromFloat32(float32(bi+1) * 0.5).Bits()
			binary.LittleEndian.PutUint16(raw[bi*18:bi*18+2], scaleBits)
			for j := range 16 {
				raw[bi*18+2+j] = byte((bi*16 + j) % 256)
			}
		}

		tns, err := decodeIQ4NLTensor([]int{numElements}, numElements, raw)
		if err != nil {
			t.Fatalf("decodeIQ4NLTensor: %v", err)
		}
		if tns.Shape()[0] != numElements {
			t.Errorf("shape = %v, want [%d]", tns.Shape(), numElements)
		}
		// Different scales per block means different value ranges.
		got := tns.Data()
		hasNonZero := false
		for _, v := range got {
			if v != 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Error("expected non-zero values across multiple blocks")
		}
	})

	t.Run("IQ3_S/2blocks", func(t *testing.T) {
		const numElements = 512 // 2 super-blocks of 256
		raw := make([]byte, 2*110)
		for bi := range 2 {
			off := bi * 110
			scaleBits := float16.FromFloat32(float32(bi+1)).Bits()
			binary.LittleEndian.PutUint16(raw[off:off+2], scaleBits)
			// Set sub-block scales.
			for i := range 4 {
				raw[off+106+i] = 0x22
			}
		}

		tns, err := decodeIQ3STensor([]int{numElements}, numElements, raw)
		if err != nil {
			t.Fatalf("decodeIQ3STensor: %v", err)
		}
		if tns.Shape()[0] != numElements {
			t.Errorf("shape = %v, want [%d]", tns.Shape(), numElements)
		}
	})

	t.Run("IQ2_XXS/2blocks", func(t *testing.T) {
		const numElements = 512 // 2 blocks of 256
		raw := make([]byte, 2*68)
		for bi := range 2 {
			off := bi * 68
			binary.LittleEndian.PutUint32(raw[off:off+4], math.Float32bits(float32(bi+1)*0.25))
			for j := range 64 {
				raw[off+4+j] = byte((bi*64 + j) % 256)
			}
		}

		tns, err := decodeIQ2XXSTensor([]int{numElements}, numElements, raw)
		if err != nil {
			t.Fatalf("decodeIQ2XXSTensor: %v", err)
		}
		if tns.Shape()[0] != numElements {
			t.Errorf("shape = %v, want [%d]", tns.Shape(), numElements)
		}
		// Block 0 has scale 0.25, block 1 has scale 0.5 — different magnitudes.
		got := tns.Data()
		hasNonZero := false
		for _, v := range got {
			if v != 0 {
				hasNonZero = true
				break
			}
		}
		if !hasNonZero {
			t.Error("expected non-zero values across multiple blocks")
		}
	})
}

// TestIQuant_ZeroScale tests that IQ types with zero scale produce all-zero output.
func TestIQuant_ZeroScale(t *testing.T) {
	t.Run("IQ4_NL", func(t *testing.T) {
		raw := make([]byte, 18) // scale = 0 (default)
		for i := range 16 {
			raw[2+i] = 0xFF // all nibbles set
		}
		tns, err := decodeIQ4NLTensor([]int{32}, 32, raw)
		if err != nil {
			t.Fatalf("decodeIQ4NLTensor: %v", err)
		}
		for i, v := range tns.Data() {
			if v != 0 {
				t.Errorf("index %d: got %v, want 0 (zero scale)", i, v)
				break
			}
		}
	})

	t.Run("IQ3_S", func(t *testing.T) {
		raw := make([]byte, 110) // scale = 0 (default)
		tns, err := decodeIQ3STensor([]int{256}, 256, raw)
		if err != nil {
			t.Fatalf("decodeIQ3STensor: %v", err)
		}
		for i, v := range tns.Data() {
			if v != 0 {
				t.Errorf("index %d: got %v, want 0 (zero scale)", i, v)
				break
			}
		}
	})

	t.Run("IQ2_XXS", func(t *testing.T) {
		raw := make([]byte, 68) // scale = 0 (default float32 zero)
		for i := range 64 {
			raw[4+i] = 0xFF
		}
		tns, err := decodeIQ2XXSTensor([]int{256}, 256, raw)
		if err != nil {
			t.Fatalf("decodeIQ2XXSTensor: %v", err)
		}
		for i, v := range tns.Data() {
			if v != 0 {
				t.Errorf("index %d: got %v, want 0 (zero scale)", i, v)
				break
			}
		}
	})
}

func TestTensorByteSize_TQ2_0(t *testing.T) {
	tests := []struct {
		numElements int
		wantBytes   int
	}{
		{1, 1},   // 1 element = 1 byte (ceil(1/4))
		{4, 1},   // 4 elements = 1 byte
		{5, 2},   // 5 elements = 2 bytes
		{8, 2},   // 8 elements = 2 bytes
		{32, 8},  // 32 elements = 8 bytes
		{33, 9},  // 33 elements = 9 bytes
	}
	for _, tt := range tests {
		got, err := TensorByteSize(GGMLTypeTQ2_0, tt.numElements)
		if err != nil {
			t.Errorf("TensorByteSize(TQ2_0, %d): unexpected error: %v", tt.numElements, err)
			continue
		}
		if got != tt.wantBytes {
			t.Errorf("TensorByteSize(TQ2_0, %d) = %d, want %d", tt.numElements, got, tt.wantBytes)
		}
	}
}

func TestTensorByteSize_Q2K(t *testing.T) {
	tests := []struct {
		numElements int
		wantBytes   int
	}{
		{256, 84},
		{512, 168},
		{257, 168}, // 2 blocks
	}
	for _, tt := range tests {
		got, err := TensorByteSize(GGMLTypeQ2_K, tt.numElements)
		if err != nil {
			t.Fatalf("TensorByteSize(Q2_K, %d): %v", tt.numElements, err)
		}
		if got != tt.wantBytes {
			t.Errorf("TensorByteSize(Q2_K, %d) = %d, want %d", tt.numElements, got, tt.wantBytes)
		}
	}
}

func TestTensorByteSize_Q3K(t *testing.T) {
	tests := []struct {
		numElements int
		wantBytes   int
	}{
		{256, 110},
		{512, 220},
		{257, 220}, // 2 blocks
	}
	for _, tt := range tests {
		got, err := TensorByteSize(GGMLTypeQ3_K, tt.numElements)
		if err != nil {
			t.Fatalf("TensorByteSize(Q3_K, %d): %v", tt.numElements, err)
		}
		if got != tt.wantBytes {
			t.Errorf("TensorByteSize(Q3_K, %d) = %d, want %d", tt.numElements, got, tt.wantBytes)
		}
	}
}

func TestDecodeQ2KTensor(t *testing.T) {
	const blockSize = 256
	const blockBytes = 84
	raw := make([]byte, blockBytes)

	// Set d = 1.0 (fp16), dmin = 0.0 (fp16).
	d16 := float16.FromFloat32(1.0)
	binary.LittleEndian.PutUint16(raw[0:2], d16.Bits())
	binary.LittleEndian.PutUint16(raw[2:4], 0) // dmin = 0

	// Set all scales to 1 (low nibble = 1, high nibble = 0 for min).
	for i := range 16 {
		raw[4+i] = 0x01 // scale=1, min=0
	}

	// Set all qs to 0x55 = 01 01 01 01 in binary -> all quant values = 1.
	for i := range 64 {
		raw[20+i] = 0x55
	}

	tns, err := decodeQ2KTensor([]int{blockSize}, blockSize, raw)
	if err != nil {
		t.Fatalf("decodeQ2KTensor: %v", err)
	}

	data := tns.Data()
	if len(data) != blockSize {
		t.Fatalf("got %d elements, want %d", len(data), blockSize)
	}

	// With d=1, scale=1, min=0, quant=1: value = 1*1*1 - 0*0 = 1.0
	// After Q4 re-quantization, values should be approximately 1.0.
	for i, v := range data {
		if v < 0.5 || v > 1.5 {
			t.Errorf("index %d: got %v, want ~1.0", i, v)
			break
		}
	}
}

func TestDecodeQ3KTensor(t *testing.T) {
	const blockSize = 256
	const blockBytes = 110
	raw := make([]byte, blockBytes)

	// Set d = 1.0 (fp16) at offset 108.
	d16 := float16.FromFloat32(1.0)
	binary.LittleEndian.PutUint16(raw[108:110], d16.Bits())

	// hmask: all zeros -> high bit = 0 for all, so q |= 4 -> base quant += 4.
	// qs: all zeros -> low 2 bits = 0 for all.
	// So each 3-bit value = 4, then q - 4 = 0 -> all output values = 0.

	tns, err := decodeQ3KTensor([]int{blockSize}, blockSize, raw)
	if err != nil {
		t.Fatalf("decodeQ3KTensor: %v", err)
	}

	data := tns.Data()
	if len(data) != blockSize {
		t.Fatalf("got %d elements, want %d", len(data), blockSize)
	}

	// All values should be ~0 (q-4 = 0, regardless of scale).
	for i, v := range data {
		if v < -0.5 || v > 0.5 {
			t.Errorf("index %d: got %v, want ~0.0", i, v)
			break
		}
	}
}

// TestQ5_0ReQuantizationQuality measures quality loss when re-quantizing Q5_0
// weights to Q4_K and Q4_0. Generates normally distributed float32 data
// (stddev ~0.02, typical for transformer weight matrices), encodes to Q5_0,
// then measures the error introduced by each re-quantization target.
func TestQ5_0ReQuantizationQuality(t *testing.T) {
	// Use 8192 elements: enough for stable statistics (256 Q5_0 blocks,
	// 32 Q4_K super-blocks).
	const numElements = 8192

	// Generate normally distributed weights: mean=0, stddev=0.02.
	rng := rand.New(rand.NewPCG(42, 0))
	original := make([]float32, numElements)
	for i := range original {
		original[i] = float32(rng.NormFloat64()) * 0.02
	}

	// Step 1: Quantize float32 -> Q5_0.
	q5Raw := testQuantizeQ5_0(original)
	q5s, err := tensor.NewQ5_0StorageFromRaw(q5Raw, numElements)
	if err != nil {
		t.Fatalf("NewQ5_0StorageFromRaw: %v", err)
	}
	q5f32 := make([]float32, numElements)
	q5s.Dequantize(q5f32)

	// Step 2a: Re-quantize Q5_0 -> Q4_K.
	q4kRaw := testQuantizeQ4K(q5f32)
	q4ks, err := tensor.NewQ4KStorageFromRaw(q4kRaw, numElements)
	if err != nil {
		t.Fatalf("NewQ4KStorageFromRaw: %v", err)
	}
	q4kf32 := make([]float32, numElements)
	q4ks.Dequantize(q4kf32)

	// Step 2b: Re-quantize Q5_0 -> Q4_0.
	q4s := tensor.QuantizeQ4(q5f32)
	q4f32 := make([]float32, numElements)
	q4s.Dequantize(q4f32)

	// Compute error metrics: Q5_0 dequantized values are the reference.
	q4kStats := computeErrorStats(q5f32, q4kf32)
	q4Stats := computeErrorStats(q5f32, q4f32)

	// Print comparison table.
	t.Logf("\n%-15s %12s %12s %12s %12s",
		"Path", "maxErr", "meanErr", "RMSE", "zeroPercent")
	t.Logf("%-15s %12s %12s %12s %12s",
		"----", "------", "-------", "----", "-----------")
	t.Logf("%-15s %12.6f %12.6f %12.6f %11.2f%%",
		"Q5_0 -> Q4_K", q4kStats.maxErr, q4kStats.meanErr, q4kStats.rmse, q4kStats.zeroPct)
	t.Logf("%-15s %12.6f %12.6f %12.6f %11.2f%%",
		"Q5_0 -> Q4_0", q4Stats.maxErr, q4Stats.meanErr, q4Stats.rmse, q4Stats.zeroPct)

	// Verify Q4_K is at least as good as Q4_0 on RMSE.
	if q4kStats.rmse > q4Stats.rmse*1.1 {
		t.Errorf("Q4_K RMSE (%f) should not be significantly worse than Q4_0 RMSE (%f)",
			q4kStats.rmse, q4Stats.rmse)
	}

	// Sanity: errors should be non-zero (we have non-trivial data).
	if q4kStats.maxErr == 0 {
		t.Error("Q4_K maxErr is 0; expected non-zero error from re-quantization")
	}
	if q4Stats.maxErr == 0 {
		t.Error("Q4_0 maxErr is 0; expected non-zero error from re-quantization")
	}

	// Sanity: zero percentage should be low for normally distributed data.
	if q4kStats.zeroPct > 50 {
		t.Errorf("Q4_K zero percent %.2f%% is unexpectedly high", q4kStats.zeroPct)
	}
	if q4Stats.zeroPct > 50 {
		t.Errorf("Q4_0 zero percent %.2f%% is unexpectedly high", q4Stats.zeroPct)
	}
}

type errorStats struct {
	maxErr  float64
	meanErr float64
	rmse    float64
	zeroPct float64
}

func (s errorStats) String() string {
	return fmt.Sprintf("maxErr=%.6f meanErr=%.6f RMSE=%.6f zeroPercent=%.2f%%",
		s.maxErr, s.meanErr, s.rmse, s.zeroPct)
}

// computeErrorStats computes element-wise error statistics between reference
// and actual float32 slices.
func computeErrorStats(ref, actual []float32) errorStats {
	n := len(ref)
	var maxErr, sumErr, sumSqErr float64
	var zeroCount int
	for i := range n {
		diff := math.Abs(float64(actual[i] - ref[i]))
		if diff > maxErr {
			maxErr = diff
		}
		sumErr += diff
		sumSqErr += diff * diff
		if actual[i] == 0 {
			zeroCount++
		}
	}
	return errorStats{
		maxErr:  maxErr,
		meanErr: sumErr / float64(n),
		rmse:    math.Sqrt(sumSqErr / float64(n)),
		zeroPct: float64(zeroCount) / float64(n) * 100,
	}
}

// testQuantizeQ5_0 quantizes float32 values into Q5_0 raw block bytes.
// Q5_0: 32 values per block, 22 bytes per block.
// Layout: 2 bytes fp16 scale + 4 bytes qh (high bits) + 16 bytes qs (low nibbles).
// Symmetric quantization: maps [-absmax, absmax] to [-16, 15].
func testQuantizeQ5_0(src []float32) []byte {
	const blockSize = 32
	const blockBytes = 22
	n := len(src)
	nBlocks := (n + blockSize - 1) / blockSize
	raw := make([]byte, nBlocks*blockBytes)

	for bi := range nBlocks {
		offset := bi * blockSize

		// Find absmax for this block.
		var absMax float32
		for j := range blockSize {
			idx := offset + j
			var v float32
			if idx < n {
				v = src[idx]
			}
			if av := float32(math.Abs(float64(v))); av > absMax {
				absMax = av
			}
		}

		// Scale maps [-absMax, absMax] to [-16, 15].
		var d float32
		if absMax > 0 {
			d = absMax / 15.0
		}
		dFP16 := float16.FromFloat32(d)

		blk := raw[bi*blockBytes : (bi+1)*blockBytes]
		binary.LittleEndian.PutUint16(blk[0:2], dFP16.Bits())

		var invD float32
		if d > 0 {
			invD = 1.0 / dFP16.ToFloat32()
		}

		var qh uint32
		for j := range 16 {
			var v0, v1 float32
			if offset+j < n {
				v0 = src[offset+j]
			}
			if offset+j+16 < n {
				v1 = src[offset+j+16]
			}

			// Quantize to 5-bit signed: range [-16, 15], stored as unsigned [0, 31].
			q0 := clampTestInt(int(math.Round(float64(v0*invD))), -16, 15)
			q1 := clampTestInt(int(math.Round(float64(v1*invD))), -16, 15)

			u0 := q0 + 16 // unsigned [0, 31]
			u1 := q1 + 16

			// Low 4 bits go into qs (packed nibbles).
			blk[6+j] = byte(u0&0x0F) | (byte(u1&0x0F) << 4)

			// High bit (bit 4) goes into qh.
			if u0&0x10 != 0 {
				qh |= 1 << j
			}
			if u1&0x10 != 0 {
				qh |= 1 << (j + 16)
			}
		}
		binary.LittleEndian.PutUint32(blk[2:6], qh)
	}
	return raw
}

// testQuantizeQ4K quantizes float32 values into Q4_K raw super-block bytes.
// Q4_K: 256 values per super-block, 144 bytes per block, 8 sub-blocks of 32.
// Asymmetric quantization: each sub-block has its own scale and min, packed into
// 6-bit fields with a shared fp16 super-block scale and super-block dmin.
func testQuantizeQ4K(src []float32) []byte {
	const superBlockSize = 256
	const blockBytes = 144
	const numSubBlocks = 8
	const subBlockSize = 32

	n := len(src)
	nBlocks := (n + superBlockSize - 1) / superBlockSize
	raw := make([]byte, nBlocks*blockBytes)

	for bi := range nBlocks {
		off := bi * superBlockSize
		var values [superBlockSize]float32
		end := off + superBlockSize
		if end > n {
			end = n
		}
		copy(values[:], src[off:end])

		// Compute per-sub-block scale and min.
		var subScales, subMins [numSubBlocks]float32
		for sb := range numSubBlocks {
			sOff := sb * subBlockSize
			minVal := values[sOff]
			maxVal := values[sOff]
			for j := 1; j < subBlockSize; j++ {
				v := values[sOff+j]
				if v < minVal {
					minVal = v
				}
				if v > maxVal {
					maxVal = v
				}
			}
			if minVal > 0 {
				minVal = 0
			}
			subScales[sb] = (maxVal - minVal) / 15.0
			subMins[sb] = -minVal
		}

		// Find max scale and min across sub-blocks for the super-block header.
		var maxScale, maxMin float32
		for sb := range numSubBlocks {
			if subScales[sb] > maxScale {
				maxScale = subScales[sb]
			}
			if subMins[sb] > maxMin {
				maxMin = subMins[sb]
			}
		}

		d := maxScale / 63.0
		dmin := maxMin / 63.0

		// Quantize sub-block scales and mins to 6-bit.
		var scalesQ, minsQ [numSubBlocks]uint8
		for sb := range numSubBlocks {
			if d > 0 {
				scalesQ[sb] = uint8(math.Round(float64(subScales[sb] / d)))
				if scalesQ[sb] > 63 {
					scalesQ[sb] = 63
				}
			}
			if dmin > 0 {
				minsQ[sb] = uint8(math.Round(float64(subMins[sb] / dmin)))
				if minsQ[sb] > 63 {
					minsQ[sb] = 63
				}
			}
		}

		blk := raw[bi*blockBytes : (bi+1)*blockBytes]

		// fp16 d and dmin.
		dFP16 := float16.FromFloat32(d)
		dminFP16 := float16.FromFloat32(dmin)
		binary.LittleEndian.PutUint16(blk[0:2], dFP16.Bits())
		binary.LittleEndian.PutUint16(blk[2:4], dminFP16.Bits())

		// Pack 6-bit scales and mins into 12 bytes at blk[4:16].
		for i := range 4 {
			blk[4+i] = (scalesQ[i] & 63) | ((scalesQ[4+i] >> 4) << 6)
			blk[8+i] = (minsQ[i] & 63) | ((minsQ[4+i] >> 4) << 6)
		}
		for i := range 4 {
			blk[12+i] = (scalesQ[4+i] & 0xF) | ((minsQ[4+i] & 0xF) << 4)
		}

		// Quantize values to 4-bit per sub-block pair.
		dRT := dFP16.ToFloat32()
		dminRT := dminFP16.ToFloat32()
		for group := range 4 {
			sb0 := group * 2
			sb1 := group*2 + 1

			sc0 := dRT * float32(scalesQ[sb0])
			mn0 := dminRT * float32(minsQ[sb0])
			sc1 := dRT * float32(scalesQ[sb1])
			mn1 := dminRT * float32(minsQ[sb1])

			var invScale0, invScale1 float32
			if sc0 > 0 {
				invScale0 = 1.0 / sc0
			}
			if sc1 > 0 {
				invScale1 = 1.0 / sc1
			}

			baseOut := group * 64
			baseQ := group * 32
			for l := range 32 {
				v0 := values[baseOut+l]
				v1 := values[baseOut+l+32]
				q0 := clampTestInt(int(math.Round(float64((v0+mn0)*invScale0))), 0, 15)
				q1 := clampTestInt(int(math.Round(float64((v1+mn1)*invScale1))), 0, 15)
				blk[16+baseQ+l] = byte(q0) | (byte(q1) << 4)
			}
		}
	}
	return raw
}

func clampTestInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// TestDecodeKQuant_NativeEmbeddingGuard exercises the T99.2.2.9 / H21 guard:
// when ZERFOO_GEMMA4_PLE_NATIVE_Q4K=1 AND the tensor is 2D with shape[0] above
// the embedding-vocab threshold (>50000), decodeQ4KTensor / decodeQ5KTensor /
// decodeQ6KTensor must return native *Q4KStorage / *Q5KStorage / *Q6KStorage
// instead of re-quantizing to *Q4Storage. With the flag unset, all three must
// fall back to the existing Q4_0 path; non-embedding shapes must re-quantize
// regardless of the flag.
//
// Motivation: the default decoders take a lossy round-trip
// (K-quant -> f32 -> Q4_0) that degrades embedding gather tables on the
// Gemma 4 Edge PLE path. See docs/plan.md T99.2.2.9 and docs/devlog.md H21.
func TestDecodeKQuant_NativeEmbeddingGuard(t *testing.T) {
	// Embedding-shape tensor: 2D with shape[0] = 50176 (> 50000) and 256 total
	// elements per row chosen so numElements is a multiple of 256 for K-quant
	// block alignment. Small shape[1] keeps raw block allocation modest.
	const embedVocab = 50176
	const embedDim = 2
	const embedElements = embedVocab * embedDim // 100352 = 392 * 256

	// Non-embedding shape: 2D with shape[0] below threshold. Same numElements
	// so we can reuse the same raw buffer per type.
	const nonEmbedRows = 1024
	const nonEmbedCols = embedElements / nonEmbedRows // 98

	type decoder func(shape []int, n int, raw []byte) (*tensor.TensorNumeric[float32], error)

	type kind struct {
		name       string
		decode     decoder
		blockBytes int
		// isNativeStorage returns true if the tensor's storage is the native
		// K-quant type (as opposed to the re-quantized *Q4Storage).
		isNativeStorage func(t *tensor.TensorNumeric[float32]) bool
	}

	kinds := []kind{
		{
			name:       "Q4_K",
			decode:     decodeQ4KTensor,
			blockBytes: 144,
			isNativeStorage: func(tn *tensor.TensorNumeric[float32]) bool {
				_, ok := tn.GetStorage().(*tensor.Q4KStorage)
				return ok
			},
		},
		{
			name:       "Q5_K",
			decode:     decodeQ5KTensor,
			blockBytes: 176,
			isNativeStorage: func(tn *tensor.TensorNumeric[float32]) bool {
				_, ok := tn.GetStorage().(*tensor.Q5KStorage)
				return ok
			},
		},
		{
			name:       "Q6_K",
			decode:     decodeQ6KTensor,
			blockBytes: 210,
			isNativeStorage: func(tn *tensor.TensorNumeric[float32]) bool {
				_, ok := tn.GetStorage().(*tensor.Q6KStorage)
				return ok
			},
		},
	}

	type scenario struct {
		name       string
		shape      []int
		flag       string // value for ZERFOO_GEMMA4_PLE_NATIVE_Q4K ("" = unset)
		wantNative bool
	}

	scenarios := []scenario{
		{
			name:       "flag-off-embedding-shape-rewrites-to-Q4",
			shape:      []int{embedVocab, embedDim},
			flag:       "",
			wantNative: false,
		},
		{
			name:       "flag-on-embedding-shape-keeps-native",
			shape:      []int{embedVocab, embedDim},
			flag:       "1",
			wantNative: true,
		},
		{
			name:       "flag-on-non-embedding-shape-still-rewrites-to-Q4",
			shape:      []int{nonEmbedRows, nonEmbedCols},
			flag:       "1",
			wantNative: false,
		},
		{
			name:       "flag-on-1D-shape-still-rewrites-to-Q4",
			shape:      []int{embedElements},
			flag:       "1",
			wantNative: false,
		},
		// Any other truthy-looking value is treated as unset (strict "1" check).
		{
			name:       "flag-true-string-not-recognized-rewrites-to-Q4",
			shape:      []int{embedVocab, embedDim},
			flag:       "true",
			wantNative: false,
		},
	}

	for _, k := range kinds {
		nBlocks := (embedElements + 255) / 256
		raw := make([]byte, nBlocks*k.blockBytes) // zero-filled: a valid block.

		for _, sc := range scenarios {
			t.Run(k.name+"/"+sc.name, func(t *testing.T) {
				t.Setenv("ZERFOO_GEMMA4_PLE_NATIVE_Q4K", sc.flag)

				tn, err := k.decode(sc.shape, embedElements, raw)
				if err != nil {
					t.Fatalf("%s decode: %v", k.name, err)
				}
				if tn == nil {
					t.Fatalf("%s decode returned nil tensor", k.name)
				}

				// Shape must be preserved regardless of branch taken.
				got := tn.Shape()
				if len(got) != len(sc.shape) {
					t.Fatalf("shape rank = %d, want %d", len(got), len(sc.shape))
				}
				for i := range sc.shape {
					if got[i] != sc.shape[i] {
						t.Fatalf("shape[%d] = %d, want %d", i, got[i], sc.shape[i])
					}
				}

				native := k.isNativeStorage(tn)
				_, isQ4 := tn.GetStorage().(*tensor.Q4Storage)

				switch {
				case sc.wantNative && !native:
					t.Fatalf("storage = %T, want native (e.g., *Q%sStorage)",
						tn.GetStorage(), k.name)
				case !sc.wantNative && !isQ4:
					t.Fatalf("storage = %T, want *tensor.Q4Storage "+
						"(re-quantized %s->Q4_0 default path)",
						tn.GetStorage(), k.name)
				}
			})
		}
	}
}

// TestPLENativeKQuantEnabled checks the env-var helper is strictly "1".
func TestPLENativeKQuantEnabled(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"", false},
		{"0", false},
		{"1", true},
		{"true", false},
		{"yes", false},
		{" 1 ", false}, // whitespace is not trimmed
	}
	for _, c := range cases {
		t.Run("val="+c.val, func(t *testing.T) {
			t.Setenv("ZERFOO_GEMMA4_PLE_NATIVE_Q4K", c.val)
			if got := pleNativeKQuantEnabled(); got != c.want {
				t.Fatalf("pleNativeKQuantEnabled()=%v, want %v (val=%q)",
					got, c.want, c.val)
			}
		})
	}
}

// TestIsEmbeddingShape validates the shape classifier used by the guard.
func TestIsEmbeddingShape(t *testing.T) {
	cases := []struct {
		name  string
		shape []int
		want  bool
	}{
		{"2D-vocab-above-threshold", []int{262144, 8960}, true},
		{"2D-vocab-at-threshold+1", []int{50001, 1}, true},
		{"2D-vocab-at-threshold", []int{50000, 256}, false},
		{"2D-vocab-below-threshold", []int{49999, 4096}, false},
		{"1D-even-if-large", []int{262144}, false},
		{"3D-even-if-large-first", []int{262144, 4, 4}, false},
		{"empty-shape", []int{}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isEmbeddingShape(c.shape); got != c.want {
				t.Fatalf("isEmbeddingShape(%v)=%v, want %v", c.shape, got, c.want)
			}
		})
	}
}
