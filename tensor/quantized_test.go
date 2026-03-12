package tensor

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/device"
)

func TestQuantizeQ4_RoundTrip(t *testing.T) { //nolint:dupl // Q4 and Q8 round-trip tests are structurally similar but differ in tolerance and quantizer.
	tests := []struct {
		name    string
		input   []float32
		maxErr  float32
		wantLen int
	}{
		{
			name:    "32 values in [-1,1]",
			input:   linspace(-1, 1, 32),
			maxErr:  0.1,
			wantLen: 32,
		},
		{
			name:    "64 values (2 blocks)",
			input:   linspace(-0.5, 0.5, 64),
			maxErr:  0.1,
			wantLen: 64,
		},
		{
			name:    "zeros",
			input:   make([]float32, 32),
			maxErr:  0.001,
			wantLen: 32,
		},
		{
			name:    "not multiple of 32 (48 values, padded to 64)",
			input:   linspace(-1, 1, 48),
			maxErr:  0.1,
			wantLen: 48,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := QuantizeQ4(tt.input)
			if q.Len() != tt.wantLen {
				t.Fatalf("Len() = %d, want %d", q.Len(), tt.wantLen)
			}

			dst := make([]float32, tt.wantLen)
			q.Dequantize(dst)

			for i, v := range dst {
				diff := float32(math.Abs(float64(v - tt.input[i])))
				if diff > tt.maxErr {
					t.Errorf("index %d: got %v, want %v (err=%v > %v)", i, v, tt.input[i], diff, tt.maxErr)
				}
			}
		})
	}
}

func TestQ4Storage_DeviceType(t *testing.T) {
	q := QuantizeQ4(make([]float32, 32))
	if q.DeviceType() != device.CPU {
		t.Errorf("DeviceType() = %v, want CPU", q.DeviceType())
	}
}

func TestQ4Storage_Slice(t *testing.T) {
	input := linspace(-1, 1, 32)
	q := QuantizeQ4(input)
	s := q.Slice()
	if len(s) != 32 {
		t.Fatalf("Slice() len = %d, want 32", len(s))
	}
	// Slice should return dequantized values within tolerance.
	for i, v := range s {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > 0.1 {
			t.Errorf("Slice[%d] = %v, want ~%v", i, v, input[i])
		}
	}
}

func TestQ4Storage_CompressionRatio(t *testing.T) {
	n := 1024
	input := linspace(-1, 1, n)
	q := QuantizeQ4(input)

	fp32Bytes := n * 4 // 4 bytes per float32
	q4Bytes := q.ByteSize()

	ratio := float64(fp32Bytes) / float64(q4Bytes)
	// Q4_0: 18 bytes per 32 values, FP32: 128 bytes per 32 values. Ratio ~7.1x
	// With padding to block boundary, should be close to 7x.
	if ratio < 6.0 {
		t.Errorf("compression ratio = %.1fx, want >= 6x", ratio)
	}
}

func TestQuantizeQ4_ExtremeValues(t *testing.T) {
	input := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100}
	q := QuantizeQ4(input)
	dst := make([]float32, 32)
	q.Dequantize(dst)

	// The last value (100) should be quantized to the max 4-bit value.
	// After dequantize it should be close to 100.
	if dst[31] < 50 {
		t.Errorf("extreme value: got %v, expected something close to 100", dst[31])
	}
}

func TestQuantizeQ4_NegativeValues(t *testing.T) {
	input := make([]float32, 32)
	for i := range input {
		input[i] = -float32(i) / 31.0
	}
	q := QuantizeQ4(input)
	dst := make([]float32, 32)
	q.Dequantize(dst)

	for i, v := range dst {
		diff := float32(math.Abs(float64(v - input[i])))
		if diff > 0.1 {
			t.Errorf("index %d: got %v, want %v (err=%v)", i, v, input[i], diff)
		}
	}
}

func TestQ4Storage_BlockCount(t *testing.T) {
	tests := []struct {
		n          int
		wantBlocks int
	}{
		{32, 1},
		{64, 2},
		{48, 2}, // padded to 64
		{1, 1},  // padded to 32
	}
	for _, tt := range tests {
		q := QuantizeQ4(make([]float32, tt.n))
		if q.NumBlocks() != tt.wantBlocks {
			t.Errorf("n=%d: NumBlocks() = %d, want %d", tt.n, q.NumBlocks(), tt.wantBlocks)
		}
	}
}

func TestNewQ4StorageFromRaw(t *testing.T) {
	// Quantize known data, serialize to raw bytes, reconstruct, and verify.
	input := linspace(-1, 1, 64) // 2 blocks
	orig := QuantizeQ4(input)
	raw := orig.RawBytes()

	got, err := NewQ4StorageFromRaw(raw, 64)
	if err != nil {
		t.Fatalf("NewQ4StorageFromRaw: %v", err)
	}
	if got.Len() != 64 {
		t.Fatalf("Len() = %d, want 64", got.Len())
	}
	if got.NumBlocks() != 2 {
		t.Fatalf("NumBlocks() = %d, want 2", got.NumBlocks())
	}

	// Dequantize and compare with original.
	origSlice := orig.Slice()
	gotSlice := got.Slice()
	for i := range origSlice {
		if origSlice[i] != gotSlice[i] {
			t.Errorf("index %d: got %v, want %v", i, gotSlice[i], origSlice[i])
		}
	}
}

func TestNewQ4StorageFromRaw_Errors(t *testing.T) {
	tests := []struct {
		name string
		raw  []byte
		n    int
	}{
		{"too short", make([]byte, 17), 32},
		{"zero elements", nil, 0},
		{"negative elements", nil, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewQ4StorageFromRaw(tt.raw, tt.n)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestMergeQ4Storage(t *testing.T) {
	// Create two Q4 storages from known data.
	a := QuantizeQ4(linspace(-1, 1, 64)) // 2 blocks
	b := QuantizeQ4(linspace(0, 2, 32))  // 1 block

	merged := MergeQ4Storage(a, b)
	if merged.NumBlocks() != 3 {
		t.Fatalf("NumBlocks() = %d, want 3", merged.NumBlocks())
	}
	if merged.Len() != 96 {
		t.Fatalf("Len() = %d, want 96", merged.Len())
	}

	// Verify merged data matches concatenation of originals.
	aData := make([]float32, a.Len())
	a.Dequantize(aData)
	bData := make([]float32, b.Len())
	b.Dequantize(bData)
	mergedData := make([]float32, merged.Len())
	merged.Dequantize(mergedData)

	for i := range aData {
		if mergedData[i] != aData[i] {
			t.Errorf("index %d: merged=%v, want=%v", i, mergedData[i], aData[i])
		}
	}
	for i := range bData {
		if mergedData[len(aData)+i] != bData[i] {
			t.Errorf("index %d: merged=%v, want=%v", len(aData)+i, mergedData[len(aData)+i], bData[i])
		}
	}
}

func BenchmarkDequantizeQ4(b *testing.B) {
	input := linspace(-1, 1, 4096)
	q := QuantizeQ4(input)
	dst := make([]float32, 4096)

	b.ResetTimer()
	b.SetBytes(int64(4096 * 4))
	for range b.N {
		q.Dequantize(dst)
	}
}

// linspace generates n evenly spaced values in [lo, hi].
func linspace(lo, hi float32, n int) []float32 {
	out := make([]float32, n)
	if n == 1 {
		out[0] = lo
		return out
	}
	step := (hi - lo) / float32(n-1)
	for i := range out {
		out[i] = lo + step*float32(i)
	}
	return out
}
