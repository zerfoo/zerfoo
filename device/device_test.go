package device

import "testing"

func TestGetDevice(t *testing.T) {
	t.Run("Get CPU Device", func(t *testing.T) {
		dev, err := Get("cpu")
		if err != nil {
			t.Fatalf(`expected to get "cpu" device, but got error: %v`, err)
		}
		if dev.ID() != "cpu" {
			t.Errorf(`expected device ID "cpu", got "%s"`, dev.ID())
		}
		if dev.Type() != CPU {
			t.Errorf("expected device type CPU, got %v", dev.Type())
		}
	})

	t.Run("Get Non-existent Device", func(t *testing.T) {
		_, err := Get("gpu:0")
		if err == nil {
			t.Fatal(`expected an error for non-existent device, but got nil`)
		}
	})
}

func TestCPUDevice(t *testing.T) {
	dev, err := Get("cpu")
	if err != nil {
		t.Fatalf("failed to get cpu device: %v", err)
	}

	allocator := dev.GetAllocator()
	if allocator == nil {
		t.Fatal("cpu device allocator is nil")
	}

	// Check if it's the correct type of allocator
	_, ok := allocator.(*cpuAllocator)
	if !ok {
		t.Error("expected a *cpuAllocator, but got a different type")
	}
}
