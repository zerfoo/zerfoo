package gguf

import (
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// LoadTensorsMmap creates tensors backed by MmapStorage that reference slices
// of the memory-mapped GGUF file data. No tensor data is copied -- each
// MmapStorage points directly into the mapped region. Dequantization happens
// lazily on first access via MmapStorage.Slice().
//
// mapped must be the entire GGUF file memory-mapped into a byte slice.
// The caller must keep the mapping alive for the lifetime of the returned tensors.
func LoadTensorsMmap(f *File, mapped []byte) (map[string]*tensor.TensorNumeric[float32], error) {
	result := make(map[string]*tensor.TensorNumeric[float32], len(f.Tensors))

	for i := range f.Tensors {
		ti := &f.Tensors[i]

		// Compute number of elements with overflow checks.
		numElements, err := computeNumElements(ti.Name, ti.Dimensions)
		if err != nil {
			return nil, err
		}

		// Convert dimensions to int for tensor API.
		// GGUF stores dimensions in GGML order (innermost-first: ne[0]=columns,
		// ne[1]=rows). Reverse to match PyTorch convention (outermost-first:
		// shape[0]=rows, shape[1]=columns).
		shape := make([]int, len(ti.Dimensions))
		for j, d := range ti.Dimensions {
			shape[len(ti.Dimensions)-1-j] = int(d)
		}

		// Compute byte size of this tensor's data.
		dataSize, err := TensorByteSize(ti.Type, int(numElements))
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}

		// Slice into the mmap'd region at the tensor's offset.
		if ti.Offset > math.MaxInt64 {
			return nil, fmt.Errorf("tensor %q: offset out of range", ti.Name)
		}
		offset := f.DataOffset + int64(ti.Offset)
		sz := int64(dataSize)
		if offset < 0 || sz < 0 || offset > int64(len(mapped)) || sz > int64(len(mapped))-offset {
			return nil, fmt.Errorf("tensor %q: offset+size out of mmap range (need offset %d + %d bytes, have %d)",
				ti.Name, offset, dataSize, len(mapped))
		}
		raw := mapped[offset : offset+sz]

		// Ternary tensors are decoded eagerly because MmapStorage does not
		// support the TQ2_0 quantization type. The copy is cheap since
		// ternary packing is only 2 bits per element.
		if ti.Type == GGMLTypeTQ2_0 {
			t, err := decodeTernaryTensor(shape, int(numElements), raw)
			if err != nil {
				return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
			}
			result[ti.Name] = t
			continue
		}

		// Map GGUF type to tensor.GGMLType.
		qtype, err := mapGGMLType(ti.Type)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", ti.Name, err)
		}

		// Create MmapStorage -- no copy, just a slice reference.
		s, err := tensor.NewMmapStorage(raw, int(numElements), qtype)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: mmap storage: %w", ti.Name, err)
		}

		t, err := tensor.NewWithStorage[float32](shape, s)
		if err != nil {
			return nil, fmt.Errorf("tensor %q: create tensor: %w", ti.Name, err)
		}
		result[ti.Name] = t
	}

	return result, nil
}

// mapGGMLType converts a GGUF GGMLType to the tensor package's GGMLType.
func mapGGMLType(t GGMLType) (tensor.GGMLType, error) {
	switch t {
	case GGMLTypeF32:
		return tensor.GGMLTypeF32, nil
	case GGMLTypeF16:
		return tensor.GGMLTypeF16, nil
	case GGMLTypeQ4_0:
		return tensor.GGMLTypeQ4_0, nil
	case GGMLTypeQ4_1:
		return tensor.GGMLTypeQ4_1, nil
	case GGMLTypeQ5_0:
		return tensor.GGMLTypeQ5_0, nil
	case GGMLTypeQ5_1:
		return tensor.GGMLTypeQ5_1, nil
	case GGMLTypeQ8_0:
		return tensor.GGMLTypeQ8_0, nil
	case GGMLTypeQ4_K:
		return tensor.GGMLTypeQ4_K, nil
	case GGMLTypeQ5_K:
		return tensor.GGMLTypeQ5_K, nil
	case GGMLTypeQ6_K:
		return tensor.GGMLTypeQ6_K, nil
	case GGMLTypeBF16:
		return tensor.GGMLTypeBF16, nil
	default:
		return 0, fmt.Errorf("unsupported GGML type %d for mmap loading", t)
	}
}
