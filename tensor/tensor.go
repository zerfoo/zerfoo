package tensor

import (
	"errors"
	"fmt"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
)

// Numeric defines the constraint for numeric types that can be used in Tensors.
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint32 | ~uint64 |
		~float32 | ~float64 |
		float8.Float8 |
		float16.Float16
}

// Tensor represents an n-dimensional array of a generic numeric type T.
type Tensor[T Numeric] struct {
	shape   []int
	strides []int
	data    []T
	// isView indicates if the tensor is a view of another tensor's data.
	isView bool
}

// New creates a new Tensor with the given shape and initializes it with the provided data.
// If data is nil, it allocates a new slice of the appropriate size.
// The length of the data slice must match the total number of elements calculated from the shape.
func New[T Numeric](shape []int, data []T) (*Tensor[T], error) {
	if len(shape) == 0 {
		if len(data) > 1 {
			return nil, errors.New("cannot create 0-dimensional tensor with more than one data element")
		}
		if len(data) == 0 {
			data = make([]T, 1) // For a scalar, data should have one element
		}

		return &Tensor[T]{
			shape:   shape,
			strides: []int{}, // Strides for 0-dim tensor is empty
			data:    data,
			isView:  false,
		}, nil
	}

	size := 1
	for _, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid shape dimension: %d; must be non-negative", dim)
		}
		size *= dim
	}

	// If size is 0, and data is provided, data must also be empty.
	if size == 0 && len(data) > 0 {
		return nil, errors.New("cannot create tensor with size 0 but non-empty data")
	}

	// If data is nil and size is 0, allocate an empty slice.
	if data == nil && size == 0 {
		data = make([]T, 0)
	}

	// If data is nil and size > 0, allocate a new slice of the appropriate size.
	if data == nil && size > 0 {
		data = make([]T, size)
	}

	if len(data) != size {
		return nil, fmt.Errorf("data length (%d) does not match tensor size (%d)", len(data), size)
	}

	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return &Tensor[T]{
		shape:   shape,
		strides: strides,
		data:    data,
		isView:  false,
	}, nil
}

// Shape returns a copy of the tensor's shape.
func (t *Tensor[T]) Shape() []int {
	shapeCopy := make([]int, len(t.shape))
	copy(shapeCopy, t.shape)

	return shapeCopy
}

// SetShape sets the tensor's shape.
func (t *Tensor[T]) SetShape(shape []int) {
	t.shape = shape
}

// Strides returns a copy of the tensor's strides.
func (t *Tensor[T]) Strides() []int {
	stridesCopy := make([]int, len(t.strides))
	copy(stridesCopy, t.strides)

	return stridesCopy
}

// SetStrides sets the tensor's strides.
func (t *Tensor[T]) SetStrides(strides []int) {
	t.strides = strides
}

// Data returns a slice representing the underlying data of the tensor.
func (t *Tensor[T]) Data() []T {
	if t.isView {
		if t.Dims() == 0 {
			// For a 0-dimensional tensor (scalar) view, return its single element.
			// The At method for a 0-dimensional tensor with no indices will return the scalar value.
			val, _ := t.At()

			return []T{val}
		}

		if t.Size() == 0 {
			return []T{} // Return empty slice for views with size 0
		}

		// For views, we need to construct the data slice based on the view's shape and strides
		// This is a simplified implementation and might not be efficient for all cases.
		data := make([]T, t.Size())
		indices := make([]int, t.Dims())
		i := 0
		var iter func(dim int)
		iter = func(dim int) {
			if dim == t.Dims() {
				val, _ := t.At(indices...)
				data[i] = val
				i++

				return
			}
			for j := range t.shape[dim] {
				indices[dim] = j
				iter(dim + 1)
			}
		}
		iter(0)

		return data
	}

	return t.data
}

// SetData sets the underlying data of the tensor.
func (t *Tensor[T]) SetData(data []T) {
	t.data = data
}

// Copy creates a deep copy of the tensor.
func (t *Tensor[T]) Copy() *Tensor[T] {
	newData := make([]T, t.Size())
	copy(newData, t.Data())
	newTensor, _ := New(t.shape, newData)

	return newTensor
}

// Size returns the total number of elements in the tensor.
func (t *Tensor[T]) Size() int {
	if len(t.shape) == 0 {
		return 1 // A 0-dimensional tensor (scalar) has a size of 1
	}
	size := 1
	for _, dim := range t.shape {
		size *= dim
	}

	return size
}

// Dims returns the number of dimensions of the tensor.
func (t *Tensor[T]) Dims() int {
	return len(t.shape)
}

// String returns a human-readable representation of the tensor.
func (t *Tensor[T]) String() string {
	return fmt.Sprintf("Tensor(shape=%v, data=%v)", t.shape, t.Data())
}

// Each iterates over each element of the tensor and applies the given function.
// This is useful for operations that need to read every value, respecting strides.
func (t *Tensor[T]) Each(f func(val T)) {
	if t.Dims() == 0 {
		// For a 0-dimensional tensor (scalar), apply the function to its single value.
		// A 0-dimensional tensor always has a size of 1 if created correctly.
		if t.Size() == 1 {
			f(t.data[0])
		}

		return
	}

	// This is a naive implementation for iteration.
	// A more optimized version would use a single loop over the contiguous data if not a view.
	indices := make([]int, t.Dims())
	t.eachRecursive(indices, 0, f)
}

func (t *Tensor[T]) eachRecursive(indices []int, dim int, f func(val T)) {
	if t.Dims() == 0 {
		return
	}
	if dim == t.Dims() {
		val, _ := t.At(indices...)
		f(val)

		return
	}
	for i := range t.shape[dim] {
		indices[dim] = i
		t.eachRecursive(indices, dim+1, f)
	}
}

// ShapeEquals returns true if the shapes of two tensors are identical.
func (t *Tensor[T]) ShapeEquals(other *Tensor[T]) bool {
	if len(t.shape) != len(other.shape) {
		return false
	}
	for i, dim := range t.shape {
		if dim != other.shape[i] {
			return false
		}
	}

	return true
}
