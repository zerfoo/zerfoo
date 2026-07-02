package generate

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

// SSMState holds the recurrent hidden state h_t for each MambaBlock layer.
// Unlike KV cache which grows with sequence length O(seq_len * d_model),
// SSM state is O(d_state * d_inner) per layer — constant regardless of
// sequence length.
type SSMState[T tensor.Numeric] struct {
	States    []*tensor.TensorNumeric[T] // one per layer: [1, d_inner, d_state]
	NumLayers int
	DInner    int
	DState    int
}

// NewSSMState creates an SSMState for the specified number of layers,
// with each layer's hidden state initialized to zeros.
func NewSSMState[T tensor.Numeric](numLayers, dInner, dState int) *SSMState[T] {
	states := make([]*tensor.TensorNumeric[T], numLayers)
	size := dInner * dState
	for i := range numLayers {
		data := make([]T, size)
		t, _ := tensor.New[T]([]int{1, dInner, dState}, data)
		states[i] = t
	}
	return &SSMState[T]{
		States:    states,
		NumLayers: numLayers,
		DInner:    dInner,
		DState:    dState,
	}
}

// Reset clears all layer states to zero, retaining the allocated tensors.
func (s *SSMState[T]) Reset() {
	size := s.DInner * s.DState
	for i := range s.States {
		data := make([]T, size)
		t, _ := tensor.New[T]([]int{1, s.DInner, s.DState}, data)
		s.States[i] = t
	}
}

// GetLayer returns the hidden state tensor for the given layer.
func (s *SSMState[T]) GetLayer(i int) (*tensor.TensorNumeric[T], error) {
	if i < 0 || i >= len(s.States) {
		return nil, fmt.Errorf("layer index %d out of range [0, %d)", i, len(s.States))
	}
	return s.States[i], nil
}

// SetLayer sets the hidden state tensor for the given layer.
func (s *SSMState[T]) SetLayer(i int, h *tensor.TensorNumeric[T]) error {
	if i < 0 || i >= len(s.States) {
		return fmt.Errorf("layer index %d out of range [0, %d)", i, len(s.States))
	}
	s.States[i] = h
	return nil
}

// MemoryBytes returns the total memory used by all layer states in bytes.
// This is O(numLayers * dInner * dState) — independent of sequence length.
func (s *SSMState[T]) MemoryBytes() int64 {
	var zero T
	elemSize := int64(unsafe.Sizeof(zero))
	return int64(s.NumLayers) * int64(s.DInner) * int64(s.DState) * elemSize
}
