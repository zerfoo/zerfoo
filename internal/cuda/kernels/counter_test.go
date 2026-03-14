package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func TestIncrementCounter(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	// Allocate a single int32 on GPU.
	devPtr, err := cuda.Malloc(4)
	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}
	defer func() { _ = cuda.Free(devPtr) }()

	// Reset counter to 0.
	if err := ResetCounter(devPtr, 0, stream.Ptr()); err != nil {
		t.Fatalf("ResetCounter: %v", err)
	}

	// Increment 100 times with delta=1.
	for i := 0; i < 100; i++ {
		if err := IncrementCounter(devPtr, 1, stream.Ptr()); err != nil {
			t.Fatalf("IncrementCounter iteration %d: %v", i, err)
		}
	}

	// Synchronize and copy back.
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	var result int32
	if err := cuda.Memcpy(unsafe.Pointer(&result), devPtr, 4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	if result != 100 {
		t.Errorf("counter = %d, want 100", result)
	}
}

func TestResetCounter(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devPtr, err := cuda.Malloc(4)
	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}
	defer func() { _ = cuda.Free(devPtr) }()

	// Set counter to 42.
	if err := ResetCounter(devPtr, 42, stream.Ptr()); err != nil {
		t.Fatalf("ResetCounter: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	var result int32
	if err := cuda.Memcpy(unsafe.Pointer(&result), devPtr, 4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	if result != 42 {
		t.Errorf("counter = %d, want 42", result)
	}
}

func TestIncrementCounterWithDelta(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devPtr, err := cuda.Malloc(4)
	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}
	defer func() { _ = cuda.Free(devPtr) }()

	// Reset to 0, then increment by 5 twenty times.
	if err := ResetCounter(devPtr, 0, stream.Ptr()); err != nil {
		t.Fatalf("ResetCounter: %v", err)
	}

	for i := 0; i < 20; i++ {
		if err := IncrementCounter(devPtr, 5, stream.Ptr()); err != nil {
			t.Fatalf("IncrementCounter iteration %d: %v", i, err)
		}
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	var result int32
	if err := cuda.Memcpy(unsafe.Pointer(&result), devPtr, 4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	if result != 100 {
		t.Errorf("counter = %d, want 100", result)
	}
}

func TestCounterGracefulWithoutCUDA(t *testing.T) {
	if cuda.Available() {
		t.Skip("CUDA available, skipping graceful-failure test")
	}

	if err := IncrementCounter(nil, 1, nil); err == nil {
		t.Error("IncrementCounter should return error without CUDA")
	}
	if err := ResetCounter(nil, 0, nil); err == nil {
		t.Error("ResetCounter should return error without CUDA")
	}
}
