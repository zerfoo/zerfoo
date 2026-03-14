package hip

import (
	"testing"
	"unsafe"
)

func TestAvailable(t *testing.T) {
	// Available must not panic regardless of whether HIP is present.
	avail := Available()
	t.Logf("hip.Available() = %v", avail)
}

func TestGetDeviceCount(t *testing.T) {
	if !Available() {
		t.Skip("HIP runtime not available")
	}
	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount: %v", err)
	}
	if count < 1 {
		t.Fatalf("expected at least 1 device, got %d", count)
	}
	t.Logf("HIP device count: %d", count)
}

func TestMallocFree(t *testing.T) {
	if !Available() {
		t.Skip("HIP runtime not available")
	}

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
			ptr, err := Malloc(tt.size)
			if err != nil {
				t.Fatalf("Malloc(%d): %v", tt.size, err)
			}
			if ptr == nil {
				t.Fatalf("Malloc(%d) returned nil pointer", tt.size)
			}
			if err := Free(ptr); err != nil {
				t.Fatalf("Free: %v", err)
			}
		})
	}
}

func TestMemcpyRoundTrip(t *testing.T) {
	if !Available() {
		t.Skip("HIP runtime not available")
	}

	const n = 256
	src := make([]byte, n)
	for i := range src {
		src[i] = byte(i)
	}

	devPtr, err := Malloc(n)
	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}
	defer func() { _ = Free(devPtr) }()

	// Host -> Device
	if err := Memcpy(devPtr, unsafe.Pointer(&src[0]), n, MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	// Device -> Host
	dst := make([]byte, n)
	if err := Memcpy(unsafe.Pointer(&dst[0]), devPtr, n, MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	for i := range src {
		if src[i] != dst[i] {
			t.Fatalf("mismatch at byte %d: want %d, got %d", i, src[i], dst[i])
		}
	}
}

func TestStreamCreateDestroy(t *testing.T) {
	if !Available() {
		t.Skip("HIP runtime not available")
	}

	stream, err := CreateStream()
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
