package rocblas

import (
	"testing"
)

func TestAvailable(t *testing.T) {
	// Available must not panic regardless of whether rocBLAS is present.
	avail := Available()
	t.Logf("rocblas.Available() = %v", avail)
}

func TestCreateDestroyHandle(t *testing.T) {
	if !Available() {
		t.Skip("rocBLAS not available")
	}

	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	if err := h.Destroy(); err != nil {
		t.Fatalf("Destroy: %v", err)
	}
}
