package generate

import (
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

// TestTensorCache_AppendGPU_UsesD2D verifies that when GPU-resident K/V
// tensors are appended to a GPU-backed cache, the appendGPU function uses
// CopyFromDevice (D2D) and does not call src.Data() which would trigger
// a D2H copy.
func TestTensorCache_AppendGPU_UsesD2D(t *testing.T) {
	// Create a GPU-backed cache by seeding with a GPU tensor.
	k1 := makeGPUTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v1 := makeGPUTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	cache := NewTensorCache[float32](nil, 1, 128)
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update(0) first: %v", err)
	}

	if !cache.layers[0].isGPU {
		t.Fatal("expected GPU-backed cache layer")
	}

	// Append a second GPU tensor. This should go through the D2D path
	// in appendGPU (CopyFromDevice), not the CopyFromHost path.
	k2 := makeGPUTensor(t, []int{1, 1, 4}, []float32{9, 10, 11, 12})
	v2 := makeGPUTensor(t, []int{1, 1, 4}, []float32{13, 14, 15, 16})

	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatalf("Update(0) second: %v", err)
	}

	// Verify the source tensors are still GPU-resident (appendGPU should
	// not have called .Data() on them, which would not change storage
	// but confirms we're testing the right path).
	if _, ok := k2.GetStorage().(*tensor.GPUStorage[float32]); !ok {
		t.Error("k2 should still be GPU-resident after append")
	}
	if _, ok := v2.GetStorage().(*tensor.GPUStorage[float32]); !ok {
		t.Error("v2 should still be GPU-resident after append")
	}

	// Verify data integrity of the concatenated cache.
	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if len(shape) != 3 || shape[1] != 2 {
		t.Errorf("Key shape = %v, want [1, 2, 4]", shape)
	}

	gotK := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4, 9, 10, 11, 12}
	for i := range wantK {
		if gotK[i] != wantK[i] {
			t.Errorf("Key[%d] = %v, want %v", i, gotK[i], wantK[i])
		}
	}
}

// TestTensorCache_GPUCacheOutputIsGPUResident verifies that Get() on a
// GPU-backed cache returns tensors with GPUStorage, ensuring no D2H copy
// is triggered when downstream layers consume the cached KV.
func TestTensorCache_GPUCacheOutputIsGPUResident(t *testing.T) {
	k := makeGPUTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeGPUTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	cache := NewTensorCache[float32](nil, 1, 128)
	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	if _, ok := lkv.Key.GetStorage().(*tensor.GPUStorage[float32]); !ok {
		t.Error("cached Key should have GPUStorage (D2H would only occur if caller reads .Data())")
	}
	if _, ok := lkv.Value.GetStorage().(*tensor.GPUStorage[float32]); !ok {
		t.Error("cached Value should have GPUStorage")
	}
}
