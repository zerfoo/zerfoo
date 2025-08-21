package tensor

import (
	"errors"
	"fmt"
)

// At retrieves the value at the specified indices.
// It returns an error if the number of indices does not match the tensor's dimensions
// or if any index is out of bounds.
func (t *TensorNumeric[T]) At(indices ...int) (T, error) {
	if t.Dims() == 0 {
		// For a 0-dimensional tensor (scalar), it should only be accessed with no indices.
		if len(indices) != 0 {
			var zero T
			return zero, errors.New("0-dimensional tensor cannot be accessed with indices")
		}
		// A 0-dimensional tensor always has one element in its data slice.
		return t.data[0], nil
	}

	if len(indices) != t.Dims() {
		var zero T
		return zero, fmt.Errorf("number of indices (%d) does not match tensor dimensions (%d)", len(indices), t.Dims())
	}

	offset := 0
	for i, index := range indices {
		if index < 0 || index >= t.shape[i] {
			var zero T
			return zero, fmt.Errorf("index %d is out of bounds for dimension %d with size %d", index, i, t.shape[i])
		}
		offset += index * t.strides[i]
	}

	return t.data[offset], nil
}

// Set updates the value at the specified indices.
// It returns an error if the number of indices does not match the tensor's dimensions,
// if any index is out of bounds, or if the tensor is a read-only view.
func (t *TensorNumeric[T]) Set(value T, indices ...int) error {
	if t.isView {
		// This is a simplification. A production-ready framework might allow setting on views.
		return errors.New("cannot set values on a tensor view")
	}

	if len(indices) != t.Dims() {
		return fmt.Errorf("number of indices (%d) does not match tensor dimensions (%d)", len(indices), t.Dims())
	}

	offset := 0
	for i, index := range indices {
		if index < 0 || index >= t.shape[i] {
			return fmt.Errorf("index %d is out of bounds for dimension %d with size %d", index, i, t.shape[i])
		}
		offset += index * t.strides[i]
	}

	t.data[offset] = value

	return nil
}

// Slice creates a new TensorNumeric view for the specified range.
// A slice is defined by a start and end index for each dimension.
// The returned tensor shares the same underlying data.
func (t *TensorNumeric[T]) Slice(ranges ...[2]int) (*TensorNumeric[T], error) {
	if t.Dims() == 0 {
		return nil, errors.New("cannot slice a 0-dimensional tensor")
	}

	if len(ranges) > len(t.shape) {
		return nil, errors.New("too many slice ranges for tensor dimensions")
	}

	newShape := make([]int, len(t.shape))
	copy(newShape, t.shape)
	offset := 0

	for i, r := range ranges {
		start, end := r[0], r[1]
		if start < 0 || end > t.shape[i] || start > end {
			return nil, fmt.Errorf("invalid slice range [%d:%d] for dimension %d with size %d", start, end, i, t.shape[i])
		}
		newShape[i] = end - start
		offset += start * t.strides[i]
	}

	return &TensorNumeric[T]{
		shape:   newShape,
		strides: t.strides,
		data:    t.data[offset:],
		isView:  true,
	}, nil
}
