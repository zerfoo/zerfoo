package tensor

import (
	"errors"
	"fmt"
	"reflect"
	"unsafe"

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

// Float defines the constraint for floating-point types.
type Float interface {
	~float32 | ~float64
}

// Tensor is an interface that all concrete tensor types must implement.
// This allows the graph to be type-agnostic at a high level.
type Tensor interface {
	Shape() []int
	DType() reflect.Type
	// We add a private method to ensure only our tensor types can implement this.
	isTensor()
}

// TensorNumeric[T] represents an n-dimensional array of a generic numeric type T.
type TensorNumeric[T Numeric] struct {
	shape   []int
	strides []int
	data    []T
	isView  bool
}

// isTensor is a private method to satisfy the Tensor interface.
func (t *TensorNumeric[T]) isTensor() {}

// DType returns the reflect.Type of the tensor's elements.
func (t *TensorNumeric[T]) DType() reflect.Type {
	var zero T
	return reflect.TypeOf(zero)
}

// New creates a new TensorNumeric with the given shape and initializes it with the provided data.
func New[T Numeric](shape []int, data []T) (*TensorNumeric[T], error) {
	if len(shape) == 0 {
		if len(data) > 1 {
			return nil, errors.New("cannot create 0-dimensional tensor with more than one data element")
		}
		if len(data) == 0 {
			data = make([]T, 1)
		}
		return &TensorNumeric[T]{
			shape:   shape,
			strides: []int{},
			data:    data,
		}, nil
	}

	size := 1
	for _, dim := range shape {
		if dim < 0 {
			return nil, fmt.Errorf("invalid shape dimension: %d; must be non-negative", dim)
		}
		size *= dim
	}

	if data == nil {
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

	return &TensorNumeric[T]{
		shape:   shape,
		strides: strides,
		data:    data,
	}, nil
}

// NewFromType creates a new tensor of a specific reflect.Type.
// This is used when the concrete generic type is not known at compile time.
func NewFromType(t reflect.Type, shape []int, data any) (Tensor, error) {
	// t is expected to be a pointer type, like *tensor.TensorNumeric[float32]
	if t.Kind() != reflect.Ptr {
		return nil, fmt.Errorf("type must be a pointer to a tensor type, got %s", t.Kind())
	}
	elemType := t.Elem() // This gives us tensor.TensorNumeric[float32]

	// Get the generic type argument (e.g., float32)
	if elemType.NumField() == 0 { // Simplified check
		return nil, fmt.Errorf("cannot determine generic type from %s", elemType.Name())
	}
	// This is a bit of a hack, relying on the internal structure.
	// A more robust solution might involve a registry of tensor types.
	dataType := elemType.Field(2).Type.Elem() // data []T -> T

	// Create a new instance of the concrete tensor type
	// e.g., reflect.New(tensor.TensorNumeric[float32])
	_ = reflect.New(elemType)

	// Call the generic New function using reflection
	newFn := reflect.ValueOf(New[float32]) // Placeholder type, will be replaced
	switch dataType.Kind() {
	case reflect.Float32:
		newFn = reflect.ValueOf(New[float32])
	case reflect.Float64:
		newFn = reflect.ValueOf(New[float64])
	case reflect.Int:
		newFn = reflect.ValueOf(New[int])
	case reflect.Int32:
		newFn = reflect.ValueOf(New[int32])
	case reflect.Int64:
		newFn = reflect.ValueOf(New[int64])
	// Add other supported types here
	default:
		return nil, fmt.Errorf("unsupported data type for NewFromType: %s", dataType.Kind())
	}

	// Prepare arguments for the call
	args := []reflect.Value{
		reflect.ValueOf(shape),
		reflect.ValueOf(data), // data is `any`, needs to be correct type or nil
	}
	if data == nil {
		// Create a nil slice of the correct type
		args[1] = reflect.Zero(reflect.SliceOf(dataType))
	}

	results := newFn.Call(args)
	if !results[1].IsNil() {
		return nil, results[1].Interface().(error)
	}

	return results[0].Interface().(Tensor), nil
}


// Shape returns a copy of the tensor's shape.
func (t *TensorNumeric[T]) Shape() []int {
	shapeCopy := make([]int, len(t.shape))
	copy(shapeCopy, t.shape)
	return shapeCopy
}

// Data returns a slice representing the underlying data of the tensor.
func (t *TensorNumeric[T]) Data() []T {
	return t.data
}

// SetData sets the underlying data of the tensor.
func (t *TensorNumeric[T]) SetData(data []T) {
	t.data = data
}

// Strides returns a copy of the tensor's strides.
func (t *TensorNumeric[T]) Strides() []int {
	stridesCopy := make([]int, len(t.strides))
	copy(stridesCopy, t.strides)
	return stridesCopy
}

// SetStrides sets the tensor's strides.
func (t *TensorNumeric[T]) SetStrides(strides []int) {
	t.strides = strides
}

// SetShape sets the tensor's shape.
func (t *TensorNumeric[T]) SetShape(shape []int) {
	t.shape = shape
}

// Size returns the total number of elements in the tensor.
func (t *TensorNumeric[T]) Size() int {
	if len(t.shape) == 0 {
		return 1
	}
	size := 1
	for _, dim := range t.shape {
		size *= dim
	}
	return size
}

// Dims returns the number of dimensions of the tensor.
func (t *TensorNumeric[T]) Dims() int {
	return len(t.shape)
}

// ShapeEquals returns true if the shapes of two tensors are identical.
func (t *TensorNumeric[T]) ShapeEquals(other *TensorNumeric[T]) bool {
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

// Bytes returns the underlying data of the tensor as a byte slice.
func (t *TensorNumeric[T]) Bytes() ([]byte, error) {
	var zero T
	switch any(zero).(type) {
	case float32:
		float32Data := *(*[]float32)(unsafe.Pointer(&t.data))
		return Float32ToBytes(float32Data)
	default:
		return nil, fmt.Errorf("Bytes is currently only implemented for float32, not %T", zero)
	}
}

// Float32ToBytes converts a float32 slice to a byte slice.
func Float32ToBytes(f []float32) ([]byte, error) {
	header := (*reflect.SliceHeader)(unsafe.Pointer(&f))
	header.Len *= 4
	header.Cap *= 4
	return *(*[]byte)(unsafe.Pointer(header)), nil
}