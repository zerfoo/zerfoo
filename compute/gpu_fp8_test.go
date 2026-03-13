package compute

import (
	"context"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestGPUEngine_MatMulFP8BWeight(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()

	tests := []struct {
		name    string
		m, k, n int
	}{
		{"2x2", 2, 2, 2},
		{"2x3x2", 2, 3, 2},
		{"4x4", 4, 4, 4},
		{"1x4x1", 1, 4, 1},
		{"8x16x8", 8, 16, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			aData := make([]float32, tt.m*tt.k)
			bData := make([]float32, tt.k*tt.n)
			for i := range aData {
				aData[i] = float32(i%7+1) * 0.1
			}
			for i := range bData {
				bData[i] = float32(i%5+1) * 0.1
			}

			// Compute expected result using FP32 CPU engine.
			cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
			cpuA, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			cpuB, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)
			expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			// Create A as FP32, B with FP8E4M3Storage.
			a, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			fp8B := tensor.NewFP8E4M3Storage(bData)
			b, _ := tensor.NewWithStorage[float32]([]int{tt.k, tt.n}, fp8B)

			// Upload weights to GPU.
			if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{b}); err != nil {
				t.Fatalf("UploadWeights: %v", err)
			}

			got, err := eng.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul FP8 B: %v", err)
			}

			gotData := got.Data()
			expData := expected.Data()
			if len(gotData) != len(expData) {
				t.Fatalf("output size mismatch: got %d, want %d", len(gotData), len(expData))
			}

			var maxRelErr float64
			for i := range gotData {
				if expData[i] == 0 {
					continue
				}
				rel := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
				if rel > maxRelErr {
					maxRelErr = rel
				}
			}

			// FP8 quantization is lossy; allow up to 5% relative error.
			if maxRelErr > 0.05 {
				t.Errorf("max relative error %.4f exceeds 0.05 threshold", maxRelErr)
				t.Logf("expected: %v", expData)
				t.Logf("got:      %v", gotData)
			}
		})
	}
}

func TestGPUEngine_MatMulFP8AWeight(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()

	tests := []struct {
		name    string
		m, k, n int
	}{
		{"2x2", 2, 2, 2},
		{"4x4", 4, 4, 4},
		{"8x16x8", 8, 16, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			aData := make([]float32, tt.m*tt.k)
			bData := make([]float32, tt.k*tt.n)
			for i := range aData {
				aData[i] = float32(i%7+1) * 0.1
			}
			for i := range bData {
				bData[i] = float32(i%5+1) * 0.1
			}

			// Compute expected result using FP32 CPU engine.
			cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
			cpuA, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			cpuB, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)
			expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			// Create A with FP8E4M3Storage, B as FP32.
			fp8A := tensor.NewFP8E4M3Storage(aData)
			a, _ := tensor.NewWithStorage[float32]([]int{tt.m, tt.k}, fp8A)
			b, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)

			// Upload weights to GPU.
			if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{a}); err != nil {
				t.Fatalf("UploadWeights: %v", err)
			}

			got, err := eng.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul FP8 A: %v", err)
			}

			gotData := got.Data()
			expData := expected.Data()
			if len(gotData) != len(expData) {
				t.Fatalf("output size mismatch: got %d, want %d", len(gotData), len(expData))
			}

			var maxRelErr float64
			for i := range gotData {
				if expData[i] == 0 {
					continue
				}
				rel := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
				if rel > maxRelErr {
					maxRelErr = rel
				}
			}

			// FP8 quantization is lossy; allow up to 5% relative error.
			if maxRelErr > 0.05 {
				t.Errorf("max relative error %.4f exceeds 0.05 threshold", maxRelErr)
				t.Logf("expected: %v", expData)
				t.Logf("got:      %v", gotData)
			}
		})
	}
}

func TestFP8E4M3Storage_GPUPtrRoundTrip(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0}
	fs := tensor.NewFP8E4M3Storage(data)

	// Initially nil.
	ptr, byteSize, deviceID := fs.GPUPtr()
	if ptr != nil || byteSize != 0 || deviceID != 0 {
		t.Fatalf("expected nil GPU ptr, got %v %d %d", ptr, byteSize, deviceID)
	}

	// Set and get using a real address.
	var dummy [4]byte
	dummyPtr := unsafe.Pointer(&dummy[0])
	fs.SetGPUPtr(dummyPtr, 4, 1)
	ptr, byteSize, deviceID = fs.GPUPtr()
	if ptr != dummyPtr || byteSize != 4 || deviceID != 1 {
		t.Fatalf("GPU ptr mismatch: got %v %d %d", ptr, byteSize, deviceID)
	}

	// Scale ptr.
	if fs.ScaleGPUPtr() != nil {
		t.Fatal("expected nil scale GPU ptr")
	}
	var scaleVal float32 = 1.0
	scalePtr := unsafe.Pointer(&scaleVal)
	fs.SetScaleGPUPtr(scalePtr)
	if fs.ScaleGPUPtr() != scalePtr {
		t.Fatal("scale GPU ptr mismatch")
	}
}

func TestFP8E4M3Storage_RawBytes(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0}
	fs := tensor.NewFP8E4M3Storage(data)
	raw := fs.RawBytes()
	if len(raw) != 4 {
		t.Fatalf("expected 4 raw bytes, got %d", len(raw))
	}
}

func TestDTypeFP8_Constant(t *testing.T) {
	// Verify DTypeFP8 is distinct from DTypeF32 and DTypeFP16.
	if DTypeFP8 == DTypeF32 {
		t.Fatal("DTypeFP8 should not equal DTypeF32")
	}
	if DTypeFP8 == DTypeFP16 {
		t.Fatal("DTypeFP8 should not equal DTypeFP16")
	}
}

func TestGPUEngine_SetDTypeFP8(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	eng.SetDType(DTypeFP8)
	if eng.DTypeValue() != DTypeFP8 {
		t.Fatalf("expected DTypeFP8, got %d", eng.DTypeValue())
	}
}

func TestGPUEngine_FP8StorageDispatchDetected(t *testing.T) {
	// Verify that FP8E4M3Storage triggers the FP8 dispatch path (not the generic path).
	// We check this by creating FP8 storage and verifying the type assertion works.
	data := []float32{1.0, 2.0, 3.0, 4.0}
	fs := tensor.NewFP8E4M3Storage(data)
	tn, _ := tensor.NewWithStorage[float32]([]int{2, 2}, fs)

	_, ok := any(tn.GetStorage()).(*tensor.FP8E4M3Storage)
	if !ok {
		t.Fatal("FP8E4M3Storage dispatch not detected via type assertion")
	}
}
