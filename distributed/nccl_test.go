package distributed

import (
	"testing"
	"unsafe"
)

func TestNCCLAvailability(t *testing.T) {
	avail := IsNCCLAvailable()
	t.Logf("NCCL available: %v", avail)

	if !avail {
		t.Skip("NCCL not available, skipping further checks")
	}

	// If NCCL is available, verify we can generate a unique ID.
	id, err := NCCLGetUniqueID()
	if err != nil {
		t.Fatalf("NCCLGetUniqueID: %v", err)
	}

	// The ID should not be all zeros.
	allZero := true
	for _, b := range id {
		if b != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("NCCLGetUniqueID returned all-zero ID")
	}
}

func TestNCCLAllGather(t *testing.T) {
	if !IsNCCLAvailable() {
		t.Skip("NCCL not available")
	}

	id, err := NCCLGetUniqueID()
	if err != nil {
		t.Fatalf("NCCLGetUniqueID: %v", err)
	}

	// Single-rank test: AllGather with 1 rank is a copy.
	comm, err := NCCLCommInitRank(1, 0, id)
	if err != nil {
		t.Fatalf("NCCLCommInitRank: %v", err)
	}
	defer comm.Destroy()

	if comm.Rank() != 0 {
		t.Errorf("Rank() = %d, want 0", comm.Rank())
	}
	if comm.NRanks() != 1 {
		t.Errorf("NRanks() = %d, want 1", comm.NRanks())
	}

	// For a real test with GPU memory, we need CUDA device memory.
	// This test verifies the API compiles and the comm initializes.
	// Full GPU tests run on DGX with actual device memory.
	t.Log("NCCLAllGather: comm initialized successfully for single rank")
}

func TestNCCLReduceScatter(t *testing.T) {
	if !IsNCCLAvailable() {
		t.Skip("NCCL not available")
	}

	id, err := NCCLGetUniqueID()
	if err != nil {
		t.Fatalf("NCCLGetUniqueID: %v", err)
	}

	comm, err := NCCLCommInitRank(1, 0, id)
	if err != nil {
		t.Fatalf("NCCLCommInitRank: %v", err)
	}
	defer comm.Destroy()

	t.Log("NCCLReduceScatter: comm initialized successfully for single rank")
}

func TestNCCLGracefulSkip(t *testing.T) {
	// This test always passes. It confirms that IsNCCLAvailable()
	// returns a boolean without panicking, and that nil-comm errors
	// are handled gracefully.
	_ = IsNCCLAvailable()

	err := NCCLAllGather(nil, nil, nil, 0, 0)
	if err == nil {
		t.Error("expected error for nil comm in NCCLAllGather")
	}

	err = NCCLReduceScatter(nil, nil, nil, 0, 0)
	if err == nil {
		t.Error("expected error for nil comm in NCCLReduceScatter")
	}
}

func TestNCCLCommDestroyNil(t *testing.T) {
	// Destroy on nil comm should not panic.
	var comm *NCCLComm
	if err := comm.Destroy(); err != nil {
		t.Errorf("Destroy(nil) returned error: %v", err)
	}
}

func TestNCCLCommDestroyZeroHandle(t *testing.T) {
	comm := &NCCLComm{handle: 0}
	if err := comm.Destroy(); err != nil {
		t.Errorf("Destroy(zero handle) returned error: %v", err)
	}
}

// Ensure unused import of unsafe is consumed by the test file.
var _ = unsafe.Pointer(nil)
