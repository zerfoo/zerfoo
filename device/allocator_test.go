package device

import "testing"

func TestCPUAllocator(t *testing.T) {
	allocator := NewCPUAllocator()

	t.Run("Allocate Valid Size", func(t *testing.T) {
		mem, err := allocator.Allocate(1024)
		if err != nil {
			t.Fatalf("Allocate failed with error: %v", err)
		}
		slice, ok := mem.([]byte)
		if !ok {
			t.Fatalf("allocated memory is not a []byte slice")
		}
		if len(slice) != 1024 {
			t.Errorf("expected allocated size to be 1024, got %d", len(slice))
		}
	})

	t.Run("Allocate Zero Size", func(t *testing.T) {
		mem, err := allocator.Allocate(0)
		if err != nil {
			t.Fatalf("Allocate(0) failed with error: %v", err)
		}
		slice, ok := mem.([]byte)
		if !ok {
			t.Fatalf("allocated memory is not a []byte slice")
		}
		if len(slice) != 0 {
			t.Errorf("expected allocated size to be 0, got %d", len(slice))
		}
	})

	t.Run("Allocate Negative Size", func(t *testing.T) {
		_, err := allocator.Allocate(-1)
		if err == nil {
			t.Fatal("expected an error for negative allocation size, but got nil")
		}
	})

	t.Run("Free", func(t *testing.T) {
		mem, _ := allocator.Allocate(16)
		err := allocator.Free(mem)
		if err != nil {
			t.Errorf("Free() should not return an error for cpuAllocator, but got: %v", err)
		}
	})
}
