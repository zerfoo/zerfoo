package cuda

import (
	"testing"
	"unsafe"
)

func TestGetDeviceCount(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount failed: %v", err)
	}

	if count < 1 {
		t.Fatalf("expected at least 1 CUDA device, got %d", count)
	}
}

func TestSetDevice(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	err := SetDevice(0)
	if err != nil {
		t.Fatalf("SetDevice(0) failed: %v", err)
	}
}

func TestMallocAndFree(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	size := 1024 // 1KB
	ptr, err := Malloc(size)

	if err != nil {
		t.Fatalf("Malloc failed: %v", err)
	}

	if ptr == nil {
		t.Fatal("Malloc returned nil pointer")
	}

	err = Free(ptr)
	if err != nil {
		t.Fatalf("Free failed: %v", err)
	}
}

func TestMemcpyRoundTrip(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	// Allocate host data
	src := []float32{1.0, 2.0, 3.0, 4.0}
	byteSize := len(src) * int(unsafe.Sizeof(src[0]))

	// Allocate device memory
	devPtr, err := Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc failed: %v", err)
	}

	defer func() {
		if freeErr := Free(devPtr); freeErr != nil {
			t.Errorf("Free failed: %v", freeErr)
		}
	}()

	// Copy host to device
	err = Memcpy(devPtr, unsafe.Pointer(&src[0]), byteSize, MemcpyHostToDevice)
	if err != nil {
		t.Fatalf("Memcpy H2D failed: %v", err)
	}

	// Copy device to host
	dst := make([]float32, len(src))

	err = Memcpy(unsafe.Pointer(&dst[0]), devPtr, byteSize, MemcpyDeviceToHost)
	if err != nil {
		t.Fatalf("Memcpy D2H failed: %v", err)
	}

	// Verify round trip
	for i := range src {
		if src[i] != dst[i] {
			t.Errorf("round trip mismatch at index %d: expected %f, got %f", i, src[i], dst[i])
		}
	}
}

func TestStreamCreateDestroySync(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	stream, err := CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}

	if err := stream.Destroy(); err != nil {
		t.Fatalf("Destroy failed: %v", err)
	}
}

func TestStreamPtr(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	stream, err := CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}

	defer func() { _ = stream.Destroy() }()

	// Ptr should return non-nil for a valid stream
	// Note: The default stream (0) is a valid pointer value, so we just
	// verify no panic.
	_ = stream.Ptr()
}

func TestMemcpyAsyncRoundTrip(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	stream, err := CreateStream()
	if err != nil {
		t.Fatalf("CreateStream failed: %v", err)
	}

	defer func() { _ = stream.Destroy() }()

	src := []float32{10.0, 20.0, 30.0}
	byteSize := len(src) * int(unsafe.Sizeof(src[0]))

	devPtr, err := Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc failed: %v", err)
	}

	defer func() { _ = Free(devPtr) }()

	// Async H2D
	if err := MemcpyAsync(devPtr, unsafe.Pointer(&src[0]), byteSize, MemcpyHostToDevice, stream); err != nil {
		t.Fatalf("MemcpyAsync H2D failed: %v", err)
	}

	// Sync to ensure H2D is complete
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize after H2D failed: %v", err)
	}

	// Async D2H
	dst := make([]float32, len(src))
	if err := MemcpyAsync(unsafe.Pointer(&dst[0]), devPtr, byteSize, MemcpyDeviceToHost, stream); err != nil {
		t.Fatalf("MemcpyAsync D2H failed: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize after D2H failed: %v", err)
	}

	for i := range src {
		if src[i] != dst[i] {
			t.Errorf("async round trip mismatch at %d: got %f, want %f", i, dst[i], src[i])
		}
	}
}

func TestMemcpyAsyncNilStream(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	src := []float32{1.0, 2.0}
	byteSize := len(src) * int(unsafe.Sizeof(src[0]))

	devPtr, err := Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc failed: %v", err)
	}

	defer func() { _ = Free(devPtr) }()

	// nil stream should use the default stream
	if err := MemcpyAsync(devPtr, unsafe.Pointer(&src[0]), byteSize, MemcpyHostToDevice, nil); err != nil {
		t.Fatalf("MemcpyAsync with nil stream failed: %v", err)
	}

	dst := make([]float32, len(src))
	if err := MemcpyAsync(unsafe.Pointer(&dst[0]), devPtr, byteSize, MemcpyDeviceToHost, nil); err != nil {
		t.Fatalf("MemcpyAsync D2H with nil stream failed: %v", err)
	}

	for i := range src {
		if src[i] != dst[i] {
			t.Errorf("nil stream round trip mismatch at %d: got %f, want %f", i, dst[i], src[i])
		}
	}
}

func TestDeviceGetAttribute(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	// Query a well-known attribute (managed memory support).
	val, err := DeviceGetAttribute(cudaDevAttrManagedMemory, 0)
	if err != nil {
		t.Fatalf("DeviceGetAttribute(cudaDevAttrManagedMemory, 0) failed: %v", err)
	}
	// Value should be 0 or 1.
	if val != 0 && val != 1 {
		t.Errorf("expected 0 or 1 for managedMemory attribute, got %d", val)
	}
	t.Logf("device 0: managedMemory=%d", val)
}

func TestManagedMemorySupported(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	supported := ManagedMemorySupported(0)
	t.Logf("device 0: ManagedMemorySupported=%v", supported)

	// If supported, verify both attributes are non-zero.
	if supported {
		managed, _ := DeviceGetAttribute(cudaDevAttrManagedMemory, 0)
		concurrent, _ := DeviceGetAttribute(cudaDevAttrConcurrentManagedAccess, 0)
		if managed == 0 || concurrent == 0 {
			t.Errorf("ManagedMemorySupported=true but managed=%d, concurrent=%d", managed, concurrent)
		}
	}
}

func TestManagedMemorySupported_InvalidDevice(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	// Invalid device should return false, not panic.
	supported := ManagedMemorySupported(9999)
	if supported {
		t.Error("ManagedMemorySupported(9999) should return false for invalid device")
	}
}

func TestMallocManagedRoundTrip(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}
	if !ManagedMemorySupported(0) {
		t.Skip("managed memory not supported on device 0")
	}

	src := []float32{1.0, 2.0, 3.0, 4.0}
	byteSize := len(src) * int(unsafe.Sizeof(src[0]))

	ptr, err := MallocManaged(byteSize)
	if err != nil {
		t.Fatalf("MallocManaged failed: %v", err)
	}
	defer func() { _ = Free(ptr) }()

	// Write directly to managed memory from CPU.
	dst := unsafe.Slice((*float32)(ptr), len(src))
	copy(dst, src)

	// Read back from managed memory on CPU.
	for i, v := range dst {
		if v != src[i] {
			t.Errorf("managed memory mismatch at %d: got %f, want %f", i, v, src[i])
		}
	}
}

func TestMemcpyDeviceToDevice(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	src := []float32{5.0, 6.0, 7.0, 8.0}
	byteSize := len(src) * int(unsafe.Sizeof(src[0]))

	// Allocate two device buffers
	devA, err := Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc A failed: %v", err)
	}

	defer func() {
		if freeErr := Free(devA); freeErr != nil {
			t.Errorf("Free A failed: %v", freeErr)
		}
	}()

	devB, err := Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc B failed: %v", err)
	}

	defer func() {
		if freeErr := Free(devB); freeErr != nil {
			t.Errorf("Free B failed: %v", freeErr)
		}
	}()

	// H2D into A
	err = Memcpy(devA, unsafe.Pointer(&src[0]), byteSize, MemcpyHostToDevice)
	if err != nil {
		t.Fatalf("Memcpy H2D failed: %v", err)
	}

	// D2D from A to B
	err = Memcpy(devB, devA, byteSize, MemcpyDeviceToDevice)
	if err != nil {
		t.Fatalf("Memcpy D2D failed: %v", err)
	}

	// D2H from B
	dst := make([]float32, len(src))

	err = Memcpy(unsafe.Pointer(&dst[0]), devB, byteSize, MemcpyDeviceToHost)
	if err != nil {
		t.Fatalf("Memcpy D2H failed: %v", err)
	}

	for i := range src {
		if src[i] != dst[i] {
			t.Errorf("D2D round trip mismatch at index %d: expected %f, got %f", i, src[i], dst[i])
		}
	}
}
