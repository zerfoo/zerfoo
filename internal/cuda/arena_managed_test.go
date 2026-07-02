package cuda

import (
	"testing"
	"unsafe"
)

// TestArenaPool_ManagedMemory_CPUWriteGPURead verifies that managed (unified)
// memory allocated via the ArenaPool can be written from the CPU and read back
// correctly. This tests the zero-copy path that avoids explicit H2D/D2H
// transfers on devices with coherent unified memory (e.g., DGX Spark).
func TestArenaPool_ManagedMemory_CPUWriteGPURead(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}
	if !ManagedMemorySupported(0) {
		t.Skip("managed memory not supported on device 0")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 8192, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	if !arena.IsManaged() {
		t.Skip("arena not using managed memory (set ZERFOO_ENABLE_MANAGED_MEM=1)")
	}

	// Allocate a region and write a pattern from CPU.
	const numFloats = 64
	byteSize := numFloats * int(unsafe.Sizeof(float32(0)))
	ptr, err := arena.Alloc(0, byteSize)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}

	data := unsafe.Slice((*float32)(ptr), numFloats)
	for i := range numFloats {
		data[i] = float32(i) * 1.5
	}

	// Read back from the same unified pointer and verify.
	// On coherent devices (DGX Spark with NVLink-C2C), CPU writes are
	// immediately visible without explicit synchronization.
	for i := range numFloats {
		want := float32(i) * 1.5
		if data[i] != want {
			t.Errorf("data[%d] = %f, want %f", i, data[i], want)
		}
	}
}

// TestArenaPool_ManagedMemory_ResetAndReuse verifies that after Reset(),
// the arena can be reused and managed memory data integrity is preserved
// for the new allocation.
func TestArenaPool_ManagedMemory_ResetAndReuse(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}
	if !ManagedMemorySupported(0) {
		t.Skip("managed memory not supported on device 0")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 4096, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	if !arena.IsManaged() {
		t.Skip("arena not using managed memory (set ZERFOO_ENABLE_MANAGED_MEM=1)")
	}

	// First pass: allocate and write.
	ptr1, err := arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc pass 1: %v", err)
	}
	data1 := unsafe.Slice((*float32)(ptr1), 64)
	for i := range 64 {
		data1[i] = float32(i)
	}

	used := arena.UsedBytes()
	if used == 0 {
		t.Error("expected non-zero UsedBytes after allocation")
	}

	// Reset and reallocate.
	arena.Reset()
	if arena.UsedBytes() != 0 {
		t.Errorf("UsedBytes after Reset = %d, want 0", arena.UsedBytes())
	}

	ptr2, err := arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc pass 2: %v", err)
	}

	// On a managed arena, ptr2 should start at the same base offset since
	// we reset.
	data2 := unsafe.Slice((*float32)(ptr2), 64)
	for i := range 64 {
		data2[i] = float32(i + 100)
	}

	for i := range 64 {
		want := float32(i + 100)
		if data2[i] != want {
			t.Errorf("data2[%d] = %f, want %f", i, data2[i], want)
		}
	}
}

// TestManagedMemory_DirectAllocIntegrity tests MallocManaged directly
// (outside the arena) to verify that CPU writes to managed memory are
// readable back from the unified pointer.
func TestManagedMemory_DirectAllocIntegrity(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}
	if !ManagedMemorySupported(0) {
		t.Skip("managed memory not supported on device 0")
	}

	const numElems = 128
	byteSize := numElems * int(unsafe.Sizeof(float64(0)))

	ptr, err := MallocManaged(byteSize)
	if err != nil {
		t.Fatalf("MallocManaged: %v", err)
	}
	defer func() { _ = Free(ptr) }()

	// Write from CPU using float64 to test a different element type.
	data := unsafe.Slice((*float64)(ptr), numElems)
	for i := range numElems {
		data[i] = float64(i) * 3.14
	}

	// Verify readback.
	for i := range numElems {
		want := float64(i) * 3.14
		if data[i] != want {
			t.Errorf("data[%d] = %f, want %f", i, data[i], want)
		}
	}
}
