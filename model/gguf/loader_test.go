package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
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

	// Q5_K should use native Q5KStorage (no re-quantization).
	if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
		t.Fatalf("expected Q4Storage after re-quant, got %T", tns.GetStorage())
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

	// Q6_K should use native Q6KStorage (no re-quantization).
	if _, ok := tns.GetStorage().(*tensor.Q4Storage); !ok {
		t.Fatalf("expected Q4Storage after re-quant, got %T", tns.GetStorage())
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
