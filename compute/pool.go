package compute

import (
	"fmt"
	"strconv"
	"sync"

	"github.com/zerfoo/zerfoo/tensor"
)

// TensorPool provides reusable tensor buffers keyed by shape.
// Acquire returns a tensor from the pool or allocates a new one.
// Release returns a tensor to the pool for future reuse.
// The pool is safe for concurrent use.
type TensorPool[T tensor.Numeric] struct {
	mu    sync.Mutex
	pools map[string][]*tensor.TensorNumeric[T]
}

// NewTensorPool creates a new empty tensor pool.
func NewTensorPool[T tensor.Numeric]() *TensorPool[T] {
	return &TensorPool[T]{
		pools: make(map[string][]*tensor.TensorNumeric[T]),
	}
}

// Acquire returns a tensor with the given shape. If the pool has a matching
// buffer, it is returned (zeroed). Otherwise a new tensor is allocated.
func (p *TensorPool[T]) Acquire(shape []int) (*tensor.TensorNumeric[T], error) {
	key := shapeKey(shape)

	p.mu.Lock()
	if list := p.pools[key]; len(list) > 0 {
		t := list[len(list)-1]
		p.pools[key] = list[:len(list)-1]
		p.mu.Unlock()
		zeroData(t.Data())
		return t, nil
	}
	p.mu.Unlock()

	return tensor.New[T](shape, nil)
}

// Release returns a tensor to the pool for future reuse.
// For GPU-backed tensors, the device memory is freed immediately (returned
// to the GPU MemPool for reuse) rather than holding the tensor reference,
// since GPU memory is a scarce resource managed by a separate pool.
// The tensor must not be used after calling Release.
func (p *TensorPool[T]) Release(t *tensor.TensorNumeric[T]) {
	if t == nil {
		return
	}

	// GPU tensors: free device memory immediately so the GPU MemPool can
	// reuse it. Don't hold the tensor in the CPU pool.
	if _, ok := t.GetStorage().(*tensor.GPUStorage[T]); ok {
		t.Release()
		return
	}

	key := shapeKey(t.Shape())

	p.mu.Lock()
	p.pools[key] = append(p.pools[key], t)
	p.mu.Unlock()
}

// Len returns the total number of tensors currently in the pool.
func (p *TensorPool[T]) Len() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := 0
	for _, list := range p.pools {
		n += len(list)
	}
	return n
}

func shapeKey(shape []int) string {
	if len(shape) == 0 {
		return "scalar"
	}
	// Fast path for common rank-2 and rank-3 tensors to avoid fmt.Sprint overhead.
	switch len(shape) {
	case 1:
		return strconv.Itoa(shape[0])
	case 2:
		return strconv.Itoa(shape[0]) + "x" + strconv.Itoa(shape[1])
	case 3:
		return strconv.Itoa(shape[0]) + "x" + strconv.Itoa(shape[1]) + "x" + strconv.Itoa(shape[2])
	default:
		return fmt.Sprint(shape)
	}
}

func zeroData[T tensor.Numeric](data []T) {
	var zero T
	for i := range data {
		data[i] = zero
	}
}
