package generate

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

func TestSSMStateNew(t *testing.T) {
	numLayers := 4
	dInner := 64
	dState := 16

	state := NewSSMState[float32](numLayers, dInner, dState)

	if state.NumLayers != numLayers {
		t.Fatalf("NumLayers = %d, want %d", state.NumLayers, numLayers)
	}
	if state.DInner != dInner {
		t.Fatalf("DInner = %d, want %d", state.DInner, dInner)
	}
	if state.DState != dState {
		t.Fatalf("DState = %d, want %d", state.DState, dState)
	}
	if len(state.States) != numLayers {
		t.Fatalf("len(States) = %d, want %d", len(state.States), numLayers)
	}

	// All states should be initialized to zeros.
	for layer := range numLayers {
		h, err := state.GetLayer(layer)
		if err != nil {
			t.Fatalf("GetLayer(%d): %v", layer, err)
		}
		shape := h.Shape()
		if len(shape) != 3 || shape[0] != 1 || shape[1] != dInner || shape[2] != dState {
			t.Fatalf("layer %d shape = %v, want [1, %d, %d]", layer, shape, dInner, dState)
		}
		data := h.Data()
		for i, v := range data {
			if v != 0 {
				t.Fatalf("layer %d data[%d] = %v, want 0", layer, i, v)
			}
		}
	}
}

func TestSSMStateReset(t *testing.T) {
	state := NewSSMState[float32](2, 32, 8)

	// Write non-zero values into layer 0.
	h, _ := state.GetLayer(0)
	data := h.Data()
	for i := range data {
		data[i] = float32(i + 1)
	}

	// Verify non-zero.
	h, _ = state.GetLayer(0)
	nonZero := false
	for _, v := range h.Data() {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("expected non-zero data before reset")
	}

	// Reset should clear all states to zero.
	state.Reset()

	for layer := range 2 {
		h, _ := state.GetLayer(layer)
		for i, v := range h.Data() {
			if v != 0 {
				t.Fatalf("after Reset, layer %d data[%d] = %v, want 0", layer, i, v)
			}
		}
	}
}

func TestSSMStateSetLayer(t *testing.T) {
	state := NewSSMState[float32](2, 16, 4)

	// Create a new state tensor with known values.
	newData := make([]float32, 16*4)
	for i := range newData {
		newData[i] = 42.0
	}
	newTensor, err := tensor.New[float32]([]int{1, 16, 4}, newData)
	if err != nil {
		t.Fatalf("creating tensor: %v", err)
	}

	if err := state.SetLayer(0, newTensor); err != nil {
		t.Fatalf("SetLayer(0): %v", err)
	}

	h, _ := state.GetLayer(0)
	for i, v := range h.Data() {
		if v != 42.0 {
			t.Fatalf("after SetLayer, data[%d] = %v, want 42", i, v)
		}
	}
}

func TestSSMStateGetLayerOutOfRange(t *testing.T) {
	state := NewSSMState[float32](2, 16, 4)

	if _, err := state.GetLayer(-1); err == nil {
		t.Fatal("GetLayer(-1) should return error")
	}
	if _, err := state.GetLayer(2); err == nil {
		t.Fatal("GetLayer(2) should return error for 2-layer state")
	}
}

func TestSSMStateSetLayerOutOfRange(t *testing.T) {
	state := NewSSMState[float32](2, 16, 4)

	newData := make([]float32, 16*4)
	newTensor, _ := tensor.New[float32]([]int{1, 16, 4}, newData)

	if err := state.SetLayer(-1, newTensor); err == nil {
		t.Fatal("SetLayer(-1) should return error")
	}
	if err := state.SetLayer(2, newTensor); err == nil {
		t.Fatal("SetLayer(2) should return error for 2-layer state")
	}
}

func TestSSMStateMemoryBytes(t *testing.T) {
	dInner := 64
	dState := 16
	numLayers := 4
	elemSize := int64(unsafe.Sizeof(float32(0)))

	state := NewSSMState[float32](numLayers, dInner, dState)

	want := int64(numLayers) * int64(dInner) * int64(dState) * elemSize
	got := state.MemoryBytes()
	if got != want {
		t.Fatalf("MemoryBytes() = %d, want %d", got, want)
	}
}

func TestSSMStateMemoryIndependentOfSeqLen(t *testing.T) {
	// SSM state memory should be the same regardless of what "sequence length"
	// the model has processed — it depends only on d_inner and d_state.
	dInner := 128
	dState := 16
	numLayers := 8

	state := NewSSMState[float32](numLayers, dInner, dState)
	mem := state.MemoryBytes()

	// Memory should be numLayers * dInner * dState * 4 bytes (float32).
	expected := int64(numLayers) * int64(dInner) * int64(dState) * 4
	if mem != expected {
		t.Fatalf("MemoryBytes() = %d, want %d", mem, expected)
	}

	// Doubling dState should double memory.
	state2 := NewSSMState[float32](numLayers, dInner, dState*2)
	mem2 := state2.MemoryBytes()
	if mem2 != mem*2 {
		t.Fatalf("doubling dState: MemoryBytes() = %d, want %d", mem2, mem*2)
	}
}
