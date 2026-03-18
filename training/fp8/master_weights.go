package fp8

import (
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// masterEntry tracks one FP8Linear layer and its FP32 master weight copy.
type masterEntry[T tensor.Numeric] struct {
	layer   *FP8Linear[T]
	fp32    *tensor.TensorNumeric[float32]
	nParams int
}

// MasterWeightStore maintains FP32 master copies of FP8 model parameters.
// Optimizer updates the FP32 copies; FP8 copies are updated by casting after each step.
type MasterWeightStore[T tensor.Numeric] struct {
	entries []masterEntry[T]
}

// NewMasterWeightStore creates a store for the given FP8Linear layers.
// It copies each layer's current master weights into a float32 tensor.
func NewMasterWeightStore[T tensor.Numeric](layers []*FP8Linear[T]) (*MasterWeightStore[T], error) {
	entries := make([]masterEntry[T], len(layers))
	for i, layer := range layers {
		param := layer.masterWeight
		shape := param.Value.Shape()
		srcData := param.Value.Data()

		f32Data := make([]float32, len(srcData))
		for j, v := range srcData {
			f32Data[j] = float32(v)
		}

		fp32Tensor, err := tensor.New[float32](shape, f32Data)
		if err != nil {
			return nil, fmt.Errorf("create FP32 copy for layer %q: %w", layer.name, err)
		}

		entries[i] = masterEntry[T]{
			layer:   layer,
			fp32:    fp32Tensor,
			nParams: len(srcData),
		}
	}
	return &MasterWeightStore[T]{entries: entries}, nil
}

// FP32Params returns the FP32 master copy of all parameters.
// These are the parameters that the optimizer should update.
func (s *MasterWeightStore[T]) FP32Params() []*tensor.TensorNumeric[float32] {
	out := make([]*tensor.TensorNumeric[float32], len(s.entries))
	for i := range s.entries {
		out[i] = s.entries[i].fp32
	}
	return out
}

// SyncToFP8 casts updated FP32 master weights back to FP8 in each FP8Linear.
// Call after each optimizer step.
func (s *MasterWeightStore[T]) SyncToFP8() error {
	for _, e := range s.entries {
		f32Data := e.fp32.Data()

		// Update the layer's master weight (full precision T) from FP32 copy.
		masterData := e.layer.masterWeight.Value.Data()
		if len(masterData) != len(f32Data) {
			return fmt.Errorf("master weight size mismatch for layer %q: got %d, want %d",
				e.layer.name, len(masterData), len(f32Data))
		}
		for i, v := range f32Data {
			masterData[i] = T(v)
		}

		// Re-quantize to FP8 snapshot.
		if err := e.layer.SyncFP8Weights(); err != nil {
			return fmt.Errorf("sync FP8 weights for layer %q: %w", e.layer.name, err)
		}
	}
	return nil
}

// MemoryBytes returns total bytes used by FP32 master weight copies.
// Each float32 parameter uses 4 bytes.
func (s *MasterWeightStore[T]) MemoryBytes() int64 {
	var total int64
	for _, e := range s.entries {
		total += int64(e.nParams) * 4
	}
	return total
}
