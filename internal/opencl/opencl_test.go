package opencl

import (
	"testing"
)

func TestAvailable(t *testing.T) {
	// Available must not panic regardless of whether OpenCL is present.
	avail := Available()
	t.Logf("opencl.Available() = %v", avail)
}

func TestGetDeviceCount(t *testing.T) {
	if !Available() {
		t.Skip("OpenCL not available")
	}
	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount: %v", err)
	}
	t.Logf("OpenCL GPU device count: %d", count)
}

func TestNewContextAndDestroy(t *testing.T) {
	if !Available() {
		t.Skip("OpenCL not available")
	}
	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount: %v", err)
	}
	if count == 0 {
		t.Skip("no OpenCL GPU devices found")
	}

	ctx, err := NewContext(-1) // first available GPU
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	if err := ctx.Destroy(); err != nil {
		t.Fatalf("Destroy: %v", err)
	}
}

func TestMallocFree(t *testing.T) {
	if !Available() {
		t.Skip("OpenCL not available")
	}
	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount: %v", err)
	}
	if count == 0 {
		t.Skip("no OpenCL GPU devices found")
	}

	ctx, err := NewContext(-1)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	tests := []struct {
		name string
		size int
	}{
		{"small", 256},
		{"page", 4096},
		{"large", 1 << 20}, // 1 MiB
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ptr, err := ctx.Malloc(tt.size)
			if err != nil {
				t.Fatalf("Malloc(%d): %v", tt.size, err)
			}
			if ptr == nil {
				t.Fatalf("Malloc(%d) returned nil pointer", tt.size)
			}
			if err := ctx.Free(ptr); err != nil {
				t.Fatalf("Free: %v", err)
			}
		})
	}
}

func TestStreamCreateDestroy(t *testing.T) {
	if !Available() {
		t.Skip("OpenCL not available")
	}
	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount: %v", err)
	}
	if count == 0 {
		t.Skip("no OpenCL GPU devices found")
	}

	ctx, err := NewContext(-1)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer ctx.Destroy()

	stream, err := ctx.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
	if err := stream.Destroy(); err != nil {
		t.Fatalf("Destroy: %v", err)
	}
}
