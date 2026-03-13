package compute

import (
	"context"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// fakeMemPool is a test-only MemPool that allocates host memory via byte slices.
// It tracks allocation counts to verify reuse behavior.
type fakeMemPool struct {
	allocCount int
	freeCount  int
	live       map[unsafe.Pointer][]byte
}

func newFakeMemPool() *fakeMemPool {
	return &fakeMemPool{live: make(map[unsafe.Pointer][]byte)}
}

func (p *fakeMemPool) Alloc(_ int, byteSize int) (unsafe.Pointer, error) {
	p.allocCount++
	buf := make([]byte, byteSize)
	ptr := unsafe.Pointer(&buf[0])
	p.live[ptr] = buf
	return ptr, nil
}

func (p *fakeMemPool) Free(_ int, ptr unsafe.Pointer, _ int) {
	p.freeCount++
	delete(p.live, ptr)
}

func (p *fakeMemPool) AllocManaged(_, byteSize int) (unsafe.Pointer, error) {
	return p.Alloc(0, byteSize)
}

func (p *fakeMemPool) FreeManaged(_ int, ptr unsafe.Pointer, byteSize int) {
	p.Free(0, ptr, byteSize)
}

func (p *fakeMemPool) Drain() error           { return nil }
func (p *fakeMemPool) Stats() (int, int)       { return len(p.live), 0 }

var _ gpuapi.MemPool = (*fakeMemPool)(nil)

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

func TestFP8Scratchpad_EnsureGrows(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad
	deviceID := 0

	tests := []struct {
		name     string
		size     int
		wantGrow bool // expect a new allocation
	}{
		{"initial_1024", 1024, true},
		{"same_size_reuse", 1024, false},
		{"smaller_reuse", 512, false},
		{"grow_2048", 2048, true},
		{"grow_4096", 4096, true},
		{"equal_after_grow", 4096, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allocsBefore := pool.allocCount
			ptr, err := s.ensureA(pool, deviceID, tt.size)
			if err != nil {
				t.Fatalf("ensure(%d): %v", tt.size, err)
			}
			if ptr == nil {
				t.Fatal("ensure returned nil pointer")
			}
			grew := pool.allocCount > allocsBefore
			if grew != tt.wantGrow {
				t.Errorf("grew=%v, want %v (allocs before=%d after=%d)",
					grew, tt.wantGrow, allocsBefore, pool.allocCount)
			}
		})
	}

	// After all ensures, the internal size should be the max requested.
	if s.fp16BufASize != 4096 {
		t.Errorf("fp16BufSize=%d, want 4096", s.fp16BufASize)
	}
}

func TestFP8Scratchpad_EnsureReusesPointer(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad

	ptr1, err := s.ensureA(pool, 0, 1024)
	if err != nil {
		t.Fatal(err)
	}
	ptr2, err := s.ensureA(pool, 0, 512)
	if err != nil {
		t.Fatal(err)
	}
	if ptr1 != ptr2 {
		t.Error("ensure returned different pointer for smaller request; expected reuse")
	}
	ptr3, err := s.ensureA(pool, 0, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if ptr1 != ptr3 {
		t.Error("ensure returned different pointer for equal request; expected reuse")
	}
}

func TestFP8Scratchpad_Free(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad

	// Allocate a buffer and a fake scaleOne.
	if _, err := s.ensureA(pool, 0, 256); err != nil {
		t.Fatal(err)
	}
	scalePtr, err := pool.Alloc(0, f32Size)
	if err != nil {
		t.Fatal(err)
	}
	s.scaleOne = scalePtr

	if len(pool.live) != 2 {
		t.Fatalf("expected 2 live allocations, got %d", len(pool.live))
	}

	s.free(pool, 0)

	if s.fp16BufA != nil {
		t.Error("fp16Buf not nil after free")
	}
	if s.fp16BufASize != 0 {
		t.Error("fp16BufSize not 0 after free")
	}
	if s.scaleOne != nil {
		t.Error("scaleOne not nil after free")
	}
}

func TestFP8Scratchpad_FreeIdempotent(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad

	// Calling free on a zero-value scratchpad should not panic.
	s.free(pool, 0)
	if pool.freeCount != 0 {
		t.Errorf("free on empty scratchpad caused %d frees", pool.freeCount)
	}
}

// TestGPUEngine_FP8DequantFallbackBWeight directly tests the fp8DequantMatMulB
// fallback path (DequantFP8E4M3ToFP16 + MixedFP16Gemm) to verify numerical
// correctness independent of cublasLt availability.
func TestGPUEngine_FP8DequantFallbackBWeight(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	if eng.blas == nil {
		t.Skip("BLAS not available; fallback path requires cuBLAS")
	}

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
		{"16x32x16", 16, 32, 16},
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

			// CPU FP32 reference.
			cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
			cpuA, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			cpuB, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)
			expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			// Build FP8 B weight and FP32 A.
			a, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			fp8B := tensor.NewFP8E4M3Storage(bData)
			b, _ := tensor.NewWithStorage[float32]([]int{tt.k, tt.n}, fp8B)

			// Upload FP8 weights to GPU.
			if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{b}); err != nil {
				t.Fatalf("UploadWeights: %v", err)
			}

			// Get the GPU-resident FP8 device pointer for B.
			devB, _, _ := fp8B.GPUPtr()
			if devB == nil {
				t.Fatal("FP8 B weight not uploaded to GPU")
			}

			outShape := []int{tt.m, tt.n}

			// Call the dequant fallback directly, bypassing cublasLt.
			got, err := eng.fp8DequantMatMulB(a, fp8B, devB, tt.m, tt.n, tt.k, outShape)
			if err != nil {
				t.Fatalf("fp8DequantMatMulB: %v", err)
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

			if maxRelErr > 1e-2 {
				t.Errorf("max relative error %.6f exceeds 1e-2 threshold", maxRelErr)
				t.Logf("expected: %v", expData)
				t.Logf("got:      %v", gotData)
			}
		})
	}
}

// TestGPUEngine_FP8DequantFallbackAWeight directly tests the fp8DequantMatMulA
// fallback path (DequantFP8E4M3ToFP16 + MixedFP16Gemm) to verify numerical
// correctness independent of cublasLt availability.
func TestGPUEngine_FP8DequantFallbackAWeight(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	if eng.blas == nil {
		t.Skip("BLAS not available; fallback path requires cuBLAS")
	}

	ctx := context.Background()

	tests := []struct {
		name    string
		m, k, n int
	}{
		{"2x2", 2, 2, 2},
		{"4x4", 4, 4, 4},
		{"8x16x8", 8, 16, 8},
		{"16x32x16", 16, 32, 16},
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

			// CPU FP32 reference.
			cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
			cpuA, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			cpuB, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)
			expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			// Build FP8 A weight and FP32 B.
			fp8A := tensor.NewFP8E4M3Storage(aData)
			a, _ := tensor.NewWithStorage[float32]([]int{tt.m, tt.k}, fp8A)
			b, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)

			// Upload FP8 weights to GPU.
			if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{a}); err != nil {
				t.Fatalf("UploadWeights: %v", err)
			}

			// Get the GPU-resident FP8 device pointer for A.
			devA, _, _ := fp8A.GPUPtr()
			if devA == nil {
				t.Fatal("FP8 A weight not uploaded to GPU")
			}

			// Call the dequant fallback directly, bypassing cublasLt.
			got, err := eng.fp8DequantMatMulA(fp8A, devA, b, tt.m, tt.n, tt.k)
			if err != nil {
				t.Fatalf("fp8DequantMatMulA: %v", err)
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

			if maxRelErr > 1e-2 {
				t.Errorf("max relative error %.6f exceeds 1e-2 threshold", maxRelErr)
				t.Logf("expected: %v", expData)
				t.Logf("got:      %v", gotData)
			}
		})
	}
}

func TestFP8Scratchpad_EnsureCGrows(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad
	deviceID := 0

	tests := []struct {
		name     string
		size     int
		wantGrow bool
	}{
		{"initial_1024", 1024, true},
		{"same_size_reuse", 1024, false},
		{"smaller_reuse", 512, false},
		{"grow_2048", 2048, true},
		{"grow_4096", 4096, true},
		{"equal_after_grow", 4096, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allocsBefore := pool.allocCount
			ptr, err := s.ensureC(pool, deviceID, tt.size)
			if err != nil {
				t.Fatalf("ensureC(%d): %v", tt.size, err)
			}
			if ptr == nil {
				t.Fatal("ensureC returned nil pointer")
			}
			grew := pool.allocCount > allocsBefore
			if grew != tt.wantGrow {
				t.Errorf("grew=%v, want %v (allocs before=%d after=%d)",
					grew, tt.wantGrow, allocsBefore, pool.allocCount)
			}
		})
	}

	if s.f32BufCSize != 4096 {
		t.Errorf("f32BufCSize=%d, want 4096", s.f32BufCSize)
	}
}

func TestFP8Scratchpad_EnsureCReusesPointer(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad

	ptr1, err := s.ensureC(pool, 0, 1024)
	if err != nil {
		t.Fatal(err)
	}
	ptr2, err := s.ensureC(pool, 0, 512)
	if err != nil {
		t.Fatal(err)
	}
	if ptr1 != ptr2 {
		t.Error("ensureC returned different pointer for smaller request; expected reuse")
	}
	ptr3, err := s.ensureC(pool, 0, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if ptr1 != ptr3 {
		t.Error("ensureC returned different pointer for equal request; expected reuse")
	}
}

func TestFP8Scratchpad_FreeCleansOutputBuffer(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad

	if _, err := s.ensureC(pool, 0, 256); err != nil {
		t.Fatal(err)
	}
	if len(pool.live) != 1 {
		t.Fatalf("expected 1 live allocation, got %d", len(pool.live))
	}

	s.free(pool, 0)

	if s.f32BufC != nil {
		t.Error("f32BufC not nil after free")
	}
	if s.f32BufCSize != 0 {
		t.Error("f32BufCSize not 0 after free")
	}
	if len(pool.live) != 0 {
		t.Errorf("expected 0 live allocations after free, got %d", len(pool.live))
	}
}

func TestFP8Scratchpad_OutputBufferReusedAcrossCalls(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad

	// Simulate two MatMul calls with the same output size.
	ptr1, err := s.ensureC(pool, 0, 4096)
	if err != nil {
		t.Fatal(err)
	}
	ptr2, err := s.ensureC(pool, 0, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if ptr1 != ptr2 {
		t.Error("output buffer not reused across calls with same size")
	}
	if pool.allocCount != 1 {
		t.Errorf("allocCount=%d, want 1 (second call should reuse)", pool.allocCount)
	}

	// Different size but smaller should also reuse.
	ptr3, err := s.ensureC(pool, 0, 2048)
	if err != nil {
		t.Fatal(err)
	}
	if ptr3 != ptr1 {
		t.Error("output buffer not reused for smaller request")
	}
	if pool.allocCount != 1 {
		t.Errorf("allocCount=%d, want 1 (smaller request should reuse)", pool.allocCount)
	}
}

func TestFP8Scratchpad_GrowFreesOldBuffer(t *testing.T) {
	pool := newFakeMemPool()
	var s fp8Scratchpad

	if _, err := s.ensureA(pool, 0, 128); err != nil {
		t.Fatal(err)
	}
	if pool.allocCount != 1 || pool.freeCount != 0 {
		t.Fatalf("after first ensure: allocs=%d frees=%d", pool.allocCount, pool.freeCount)
	}

	// Growing should free the old buffer and allocate a new one.
	if _, err := s.ensureA(pool, 0, 256); err != nil {
		t.Fatal(err)
	}
	if pool.allocCount != 2 {
		t.Errorf("allocs=%d, want 2", pool.allocCount)
	}
	if pool.freeCount != 1 {
		t.Errorf("frees=%d, want 1 (old buffer should be freed)", pool.freeCount)
	}
}
