package tensor

import (
	"errors"
	"fmt"
)

// Reshape returns a new TensorNumeric with a different shape that shares the same underlying data.
// The new shape must have the same total number of elements as the original tensor.
// This operation is a "view" and does not copy the data.
func (t *TensorNumeric[T]) Reshape(newShape []int) (*TensorNumeric[T], error) {
	newSize := 1
	inferredDim := -1
	for i, dim := range newShape {
		switch {
		case dim > 0:
			newSize *= dim
		case dim == -1:
			if inferredDim != -1 {
				return nil, errors.New("only one dimension can be inferred")
			}
			inferredDim = i
		default:
			return nil, fmt.Errorf("invalid shape dimension: %d; must be positive or -1", dim)
		}
	}

	if inferredDim != -1 {
		if t.Size()%newSize != 0 {
			return nil, fmt.Errorf("cannot infer dimension for size %d and new size %d", t.Size(), newSize)
		}
		newShape[inferredDim] = t.Size() / newSize
		newSize = t.Size()
	}

	if newSize != t.Size() {
		return nil, fmt.Errorf("cannot reshape tensor of size %d into shape %v with size %d", t.Size(), newShape, newSize)
	}

	// For a reshaped tensor, strides need to be recalculated.
	newStrides := make([]int, len(newShape))
	stride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}

	return &TensorNumeric[T]{
		shape:   newShape,
		strides: newStrides,
		data:    t.data, // Share the underlying data
		isView:  true,
	}, nil
}
