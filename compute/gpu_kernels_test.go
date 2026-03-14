package compute

import (
	"context"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/log"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// fakeStream implements gpuapi.Stream for tests that don't need real CUDA.
type fakeStream struct{}

func (fakeStream) Synchronize() error        { return nil }
func (fakeStream) Destroy() error            { return nil }
func (fakeStream) Ptr() unsafe.Pointer       { return nil }

var _ gpuapi.Stream = fakeStream{}

// fakeRuntime implements gpuapi.Runtime for tests that don't need real CUDA.
// Memcpy performs a host-to-host copy so getDevicePtr's H2D path works.
type fakeRuntime struct{}

func (fakeRuntime) DeviceType() device.Type { return device.CPU }
func (fakeRuntime) SetDevice(int) error           { return nil }
func (fakeRuntime) GetDeviceCount() (int, error)  { return 1, nil }
func (fakeRuntime) Malloc(int) (unsafe.Pointer, error) {
	return nil, nil
}
func (fakeRuntime) Free(unsafe.Pointer) error { return nil }
func (fakeRuntime) Memcpy(dst, src unsafe.Pointer, count int, _ gpuapi.MemcpyKind) error {
	// Simulate H2D copy as host-to-host memcpy for testing.
	dstSlice := unsafe.Slice((*byte)(dst), count)
	srcSlice := unsafe.Slice((*byte)(src), count)
	copy(dstSlice, srcSlice)
	return nil
}
func (fakeRuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, _ gpuapi.MemcpyKind, _ gpuapi.Stream) error {
	dstSlice := unsafe.Slice((*byte)(dst), count)
	srcSlice := unsafe.Slice((*byte)(src), count)
	copy(dstSlice, srcSlice)
	return nil
}
func (fakeRuntime) MemcpyPeer(_ unsafe.Pointer, _ int, _ unsafe.Pointer, _ int, _ int) error {
	return nil
}
func (fakeRuntime) CreateStream() (gpuapi.Stream, error) {
	return fakeStream{}, nil
}

var _ gpuapi.Runtime = fakeRuntime{}

// newFakeGPUEngine builds a GPUEngine with fake pool, runtime, and stream
// so that getDevicePtr can be tested without real CUDA hardware.
func newFakeGPUEngine(t *testing.T) (*GPUEngine[float32], *fakeMemPool) {
	t.Helper()
	pool := newFakeMemPool()
	eng := &GPUEngine[float32]{
		cpu:      NewCPUEngine[float32](numeric.Float32Ops{}),
		runtime:  fakeRuntime{},
		pool:     pool,
		stream:   fakeStream{},
		logger:   log.Nop(),
		deviceID: 0,
		dtype:    DTypeF32,
	}
	return eng, pool
}

func TestGetDevicePtr_CPUStorage_ReturnsValidPointer(t *testing.T) {
	eng, pool := newFakeGPUEngine(t)

	data := make([]float32, 1024)
	for i := range data {
		data[i] = float32(i)
	}
	tsr, err := tensor.New[float32]([]int{32, 32}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ptr, cleanup, err := getDevicePtr(eng, tsr)
	if err != nil {
		t.Fatalf("getDevicePtr: %v", err)
	}
	if ptr == nil {
		t.Fatal("getDevicePtr returned nil pointer for CPUStorage tensor")
	}
	if pool.allocCount != 1 {
		t.Errorf("expected 1 alloc, got %d", pool.allocCount)
	}

	cleanup()
	if pool.freeCount != 1 {
		t.Errorf("expected 1 free after cleanup, got %d", pool.freeCount)
	}
}

func TestGetDevicePtr_GPUStorage_ReturnsExistingPointer(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng := newTestGPUEngine(t)

	// Create a GPU-resident tensor via a MatMul so it has GPUStorage.
	a, _ := tensor.New[float32]([]int{4, 4}, []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	})
	ctx := context.Background()
	gpuTensor, err := eng.MatMul(ctx, a, a)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	// gpuTensor should have GPUStorage; getDevicePtr should return the existing
	// pointer without allocating.
	ptr, cleanup, err := getDevicePtr(eng, gpuTensor)
	if err != nil {
		t.Fatalf("getDevicePtr: %v", err)
	}
	if ptr == nil {
		t.Fatal("getDevicePtr returned nil for GPUStorage tensor")
	}

	// Cleanup should be a no-op (not free the underlying GPU memory).
	cleanup()

	// Tensor should still be usable after cleanup (proves no free occurred).
	data := gpuTensor.Data()
	if len(data) != 16 {
		t.Fatalf("expected 16 elements, got %d", len(data))
	}
	// Identity * Identity = Identity; check diagonal.
	for i := 0; i < 4; i++ {
		if data[i*4+i] != 1.0 {
			t.Errorf("diagonal[%d] = %f, want 1.0", i, data[i*4+i])
		}
	}
}

func TestGetDevicePtr_SequentialCalls_NoOverlap(t *testing.T) {
	eng, pool := newFakeGPUEngine(t)

	sizes := []int{512, 1024, 2048, 4096}
	pointers := make([]unsafe.Pointer, 0, len(sizes))
	cleanups := make([]func(), 0, len(sizes))

	// Allocate multiple tensors sequentially (simulating graph forward pass).
	for _, n := range sizes {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i)
		}
		tsr, err := tensor.New[float32]([]int{n}, data)
		if err != nil {
			t.Fatalf("tensor.New(%d): %v", n, err)
		}

		ptr, cleanup, err := getDevicePtr(eng, tsr)
		if err != nil {
			t.Fatalf("getDevicePtr(%d): %v", n, err)
		}
		if ptr == nil {
			t.Fatalf("getDevicePtr(%d) returned nil", n)
		}

		// Verify no overlap with previously allocated pointers.
		for j, prev := range pointers {
			if ptr == prev {
				t.Errorf("getDevicePtr(%d) returned same pointer as call %d", n, j)
			}
		}

		pointers = append(pointers, ptr)
		cleanups = append(cleanups, cleanup)
	}

	if pool.allocCount != len(sizes) {
		t.Errorf("expected %d allocs, got %d", len(sizes), pool.allocCount)
	}

	// All pointers should still be live before cleanup.
	liveCount, _ := pool.Stats()
	if liveCount != len(sizes) {
		t.Errorf("expected %d live allocations, got %d", len(sizes), liveCount)
	}

	// Run all cleanups.
	for _, cleanup := range cleanups {
		cleanup()
	}
	if pool.freeCount != len(sizes) {
		t.Errorf("expected %d frees, got %d", len(sizes), pool.freeCount)
	}

	// No live allocations remaining.
	liveCount, _ = pool.Stats()
	if liveCount != 0 {
		t.Errorf("expected 0 live allocations after cleanup, got %d", liveCount)
	}
}

func TestGetDevicePtr_CleanupFreesMemory(t *testing.T) {
	eng, pool := newFakeGPUEngine(t)

	tests := []struct {
		name string
		n    int
	}{
		{"small_64", 64},
		{"medium_4096", 4096},
		{"large_65536", 65536},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := make([]float32, tt.n)
			tsr, err := tensor.New[float32]([]int{tt.n}, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			allocBefore := pool.allocCount
			freeBefore := pool.freeCount

			ptr, cleanup, err := getDevicePtr(eng, tsr)
			if err != nil {
				t.Fatalf("getDevicePtr: %v", err)
			}
			if ptr == nil {
				t.Fatal("getDevicePtr returned nil")
			}
			if pool.allocCount != allocBefore+1 {
				t.Error("expected one allocation")
			}

			cleanup()
			if pool.freeCount != freeBefore+1 {
				t.Error("cleanup did not free memory")
			}
		})
	}
}

func TestGetDevicePtr_LargeCPUStorage_CUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng := newTestGPUEngine(t)
	ctx := context.Background()

	tests := []struct {
		name string
		rows int
		cols int
	}{
		{"1K_elements", 32, 32},
		{"64K_elements", 256, 256},
		{"1M_elements", 1024, 1024},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := tt.rows * tt.cols
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i) * 0.001
			}
			tsr, err := tensor.New[float32]([]int{tt.rows, tt.cols}, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			ptr, cleanup, err := getDevicePtr(eng, tsr)
			if err != nil {
				t.Fatalf("getDevicePtr: %v", err)
			}
			if ptr == nil {
				t.Fatal("getDevicePtr returned nil pointer")
			}
			defer cleanup()

			// Verify data round-trip by using the pointer in a GPU op.
			identity, err := tensor.New[float32]([]int{tt.cols, tt.cols}, nil)
			if err != nil {
				t.Fatalf("tensor.New identity: %v", err)
			}
			identityData := identity.Data()
			for i := 0; i < tt.cols; i++ {
				identityData[i*tt.cols+i] = 1.0
			}

			result, err := eng.MatMul(ctx, tsr, identity)
			if err != nil {
				t.Fatalf("MatMul: %v", err)
			}

			got := result.Data()
			for i := 0; i < min(10, len(got)); i++ {
				if got[i] != data[i] {
					t.Errorf("round-trip mismatch at [%d]: got %f, want %f", i, got[i], data[i])
				}
			}
		})
	}
}

func TestGetDevicePtr_SequentialCalls_CUDA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng := newTestGPUEngine(t)
	ctx := context.Background()

	// Simulate a forward pass: create multiple tensors, get device ptrs, use them.
	const numSteps = 5
	const dim = 128

	for step := 0; step < numSteps; step++ {
		data := make([]float32, dim*dim)
		for i := range data {
			data[i] = float32(step*dim*dim+i) * 0.0001
		}
		tsr, err := tensor.New[float32]([]int{dim, dim}, data)
		if err != nil {
			t.Fatalf("step %d: tensor.New: %v", step, err)
		}

		ptr, cleanup, err := getDevicePtr(eng, tsr)
		if err != nil {
			t.Fatalf("step %d: getDevicePtr: %v", step, err)
		}
		if ptr == nil {
			t.Fatalf("step %d: nil pointer", step)
		}

		// Use the pointer in a real GPU operation to verify it's valid.
		ones, _ := tensor.New[float32]([]int{dim, 1}, nil)
		onesData := ones.Data()
		for i := range onesData {
			onesData[i] = 1.0
		}
		_, err = eng.MatMul(ctx, tsr, ones)
		if err != nil {
			t.Fatalf("step %d: MatMul failed (pointer may be invalid): %v", step, err)
		}

		cleanup()
	}
}
