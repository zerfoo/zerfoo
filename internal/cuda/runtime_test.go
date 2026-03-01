//go:build cuda

package cuda

import (
	"testing"
	"unsafe"
)

func TestGetDeviceCount(t *testing.T) {
	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount failed: %v", err)
	}

	if count < 1 {
		t.Fatalf("expected at least 1 CUDA device, got %d", count)
	}
}

func TestSetDevice(t *testing.T) {
	err := SetDevice(0)
	if err != nil {
		t.Fatalf("SetDevice(0) failed: %v", err)
	}
}

func TestMallocAndFree(t *testing.T) {
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

func TestMemcpyDeviceToDevice(t *testing.T) {
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
