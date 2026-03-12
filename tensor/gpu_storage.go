package tensor

import (
	"fmt"
	"log"
	"runtime"
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
)

// GPUStorage is a GPU device-backed Storage implementation.
// Slice() copies data from the GPU to a new CPU slice (not zero-copy).
// Set() copies data from a CPU slice to the GPU.
// Each GPUStorage tracks which device it resides on via deviceID.
// When managed is true, the storage uses unified memory (cudaMallocManaged)
// and TrySlice/TrySet access the pointer directly without Memcpy.
type GPUStorage[T Numeric] struct {
	devicePtr unsafe.Pointer // GPU device pointer
	length    int            // number of elements
	byteSize  int            // total bytes = length * sizeof(T)
	deviceID  int            // GPU device ordinal
	runtime   gpuapi.Runtime // GPU runtime for memory operations
	managed   bool           // true if allocated via cudaMallocManaged
	pool      gpuapi.MemPool // pool used for managed alloc/free (nil for discrete)
}

// NewGPUStorage allocates GPU device memory for the given number of elements
// on the specified device. An optional deviceID selects the GPU (default 0).
func NewGPUStorage[T Numeric](length int, deviceID ...int) (*GPUStorage[T], error) {
	dev := 0
	if len(deviceID) > 0 {
		dev = deviceID[0]
	}

	rt := getDefaultRuntime()
	if rt == nil {
		return nil, fmt.Errorf("NewGPUStorage: no GPU runtime available")
	}
	if err := rt.SetDevice(dev); err != nil {
		return nil, err
	}

	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	byteSize := length * elemSize

	devPtr, err := rt.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	gs := &GPUStorage[T]{
		devicePtr: devPtr,
		length:    length,
		byteSize:  byteSize,
		deviceID:  dev,
		runtime:   rt,
	}
	runtime.SetFinalizer(gs, func(s *GPUStorage[T]) { _ = s.Free() })

	return gs, nil
}

// NewGPUStorageFromSlice allocates GPU device memory, copies data from a CPU
// slice, and returns a GPUStorage on the specified device. An optional
// deviceID selects the GPU (default 0).
func NewGPUStorageFromSlice[T Numeric](data []T, deviceID ...int) (*GPUStorage[T], error) {
	s, err := NewGPUStorage[T](len(data), deviceID...)
	if err != nil {
		return nil, err
	}

	if len(data) > 0 {
		src := unsafe.Pointer(unsafe.SliceData(data))
		if err := s.runtime.Memcpy(s.devicePtr, src, s.byteSize, gpuapi.MemcpyHostToDevice); err != nil {
			// Clean up on failure
			_ = s.runtime.Free(s.devicePtr)

			return nil, err
		}
	}

	return s, nil
}

// NewGPUStorageFromPtr wraps an existing GPU device pointer as a GPUStorage.
// A GC finalizer ensures the device memory is freed if Release() is not called.
// An optional deviceID records which device the pointer belongs to (default 0).
func NewGPUStorageFromPtr[T Numeric](devPtr unsafe.Pointer, length int, deviceID ...int) (*GPUStorage[T], error) {
	dev := 0
	if len(deviceID) > 0 {
		dev = deviceID[0]
	}

	var zero T
	elemSize := int(unsafe.Sizeof(zero))

	rt := getDefaultRuntime()
	if rt == nil {
		return nil, fmt.Errorf("NewGPUStorageFromPtr: no GPU runtime available")
	}

	gs := &GPUStorage[T]{
		devicePtr: devPtr,
		length:    length,
		byteSize:  length * elemSize,
		deviceID:  dev,
		runtime:   rt,
	}
	runtime.SetFinalizer(gs, func(s *GPUStorage[T]) { _ = s.Free() })

	return gs, nil
}

// NewManagedGPUStorage allocates unified (managed) GPU memory via pool.AllocManaged.
// The returned storage is host-accessible: TrySlice and TrySet skip Memcpy.
// This is beneficial on hardware with coherent unified memory (e.g. DGX Spark
// NVLink-C2C). On backends that do not support managed memory, AllocManaged
// returns an error.
func NewManagedGPUStorage[T Numeric](pool gpuapi.MemPool, length int, deviceID ...int) (*GPUStorage[T], error) {
	dev := 0
	if len(deviceID) > 0 {
		dev = deviceID[0]
	}

	rt := getDefaultRuntime()
	if rt == nil {
		return nil, fmt.Errorf("NewManagedGPUStorage: no GPU runtime available")
	}
	if err := rt.SetDevice(dev); err != nil {
		return nil, err
	}

	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	byteSize := length * elemSize

	devPtr, err := pool.AllocManaged(dev, byteSize)
	if err != nil {
		return nil, err
	}

	gs := &GPUStorage[T]{
		devicePtr: devPtr,
		length:    length,
		byteSize:  byteSize,
		deviceID:  dev,
		runtime:   rt,
		managed:   true,
		pool:      pool,
	}
	runtime.SetFinalizer(gs, func(s *GPUStorage[T]) { _ = s.Free() })

	return gs, nil
}

// Managed returns true if this storage uses unified (managed) memory.
func (s *GPUStorage[T]) Managed() bool { return s.managed }

// Len returns the number of elements.
func (s *GPUStorage[T]) Len() int { return s.length }

// DeviceID returns the GPU device ordinal this storage resides on.
func (s *GPUStorage[T]) DeviceID() int { return s.deviceID }

// TrySlice copies device memory to a new CPU slice.
// For managed storage, the data is read directly from the unified pointer
// without a D2H Memcpy. Returns an error if the copy fails.
func (s *GPUStorage[T]) TrySlice() ([]T, error) {
	if s.length == 0 {
		return []T{}, nil
	}

	_ = s.runtime.SetDevice(s.deviceID)

	if s.managed {
		src := unsafe.Slice((*T)(s.devicePtr), s.length)
		host := make([]T, s.length)
		copy(host, src)
		return host, nil
	}

	host := make([]T, s.length)
	dst := unsafe.Pointer(unsafe.SliceData(host))

	if err := s.runtime.Memcpy(dst, s.devicePtr, s.byteSize, gpuapi.MemcpyDeviceToHost); err != nil {
		return nil, fmt.Errorf("GPUStorage.TrySlice: %w", err)
	}

	return host, nil
}

// Slice copies device memory to a new CPU slice and returns it.
// On error, logs a warning and returns a zero-valued slice.
func (s *GPUStorage[T]) Slice() []T {
	data, err := s.TrySlice()
	if err != nil {
		log.Printf("WARNING: %v; returning zero slice of length %d", err, s.length)

		return make([]T, s.length)
	}

	return data
}

// CopyTo copies GPU device memory into an existing CPU slice without allocating.
// The destination must have at least Len() elements. Returns an error on failure.
func (s *GPUStorage[T]) CopyTo(dst []T) error {
	if s.length == 0 {
		return nil
	}
	if len(dst) < s.length {
		return fmt.Errorf("GPUStorage.CopyTo: dst too small (%d < %d)", len(dst), s.length)
	}

	_ = s.runtime.SetDevice(s.deviceID)

	if s.managed {
		src := unsafe.Slice((*T)(s.devicePtr), s.length)
		copy(dst, src)
		return nil
	}

	dstPtr := unsafe.Pointer(unsafe.SliceData(dst))
	return s.runtime.Memcpy(dstPtr, s.devicePtr, s.byteSize, gpuapi.MemcpyDeviceToHost)
}

// TrySet copies data from a CPU slice to the GPU, replacing the current contents.
// If the new slice has a different length, the old device memory is freed and
// new memory is allocated. For managed storage, data is written directly to the
// unified pointer without Memcpy. Returns an error on failure.
func (s *GPUStorage[T]) TrySet(data []T) error {
	_ = s.runtime.SetDevice(s.deviceID)

	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	newByteSize := len(data) * elemSize

	if len(data) != s.length {
		if s.managed && s.pool != nil {
			s.pool.FreeManaged(s.deviceID, s.devicePtr, s.byteSize)
		} else {
			_ = s.runtime.Free(s.devicePtr)
		}

		var ptr unsafe.Pointer
		var err error
		if s.managed && s.pool != nil {
			ptr, err = s.pool.AllocManaged(s.deviceID, newByteSize)
		} else {
			ptr, err = s.runtime.Malloc(newByteSize)
		}
		if err != nil {
			s.devicePtr = nil
			s.length = 0
			s.byteSize = 0

			return fmt.Errorf("GPUStorage.TrySet: malloc: %w", err)
		}

		s.devicePtr = ptr
		s.length = len(data)
		s.byteSize = newByteSize
	}

	if len(data) > 0 {
		if s.managed {
			dst := unsafe.Slice((*T)(s.devicePtr), len(data))
			copy(dst, data)
		} else {
			src := unsafe.Pointer(unsafe.SliceData(data))
			if err := s.runtime.Memcpy(s.devicePtr, src, s.byteSize, gpuapi.MemcpyHostToDevice); err != nil {
				return fmt.Errorf("GPUStorage.TrySet: memcpy: %w", err)
			}
		}
	}

	return nil
}

// Set copies data from a CPU slice to the GPU, replacing the current contents.
// On error, logs a warning instead of panicking.
func (s *GPUStorage[T]) Set(data []T) {
	if err := s.TrySet(data); err != nil {
		log.Printf("WARNING: %v", err)
	}
}

// DeviceType returns the device type for this storage.
func (s *GPUStorage[T]) DeviceType() device.Type { return s.runtime.DeviceType() }

// Ptr returns the raw GPU device pointer.
func (s *GPUStorage[T]) Ptr() unsafe.Pointer { return s.devicePtr }

// Free releases the GPU device memory. After calling Free, the storage must
// not be used. For managed storage, the pointer is returned to the pool.
func (s *GPUStorage[T]) Free() error {
	if s.devicePtr == nil {
		return nil
	}

	if s.managed && s.pool != nil {
		s.pool.FreeManaged(s.deviceID, s.devicePtr, s.byteSize)
		s.devicePtr = nil
		s.length = 0
		s.byteSize = 0
		return nil
	}

	err := s.runtime.Free(s.devicePtr)
	s.devicePtr = nil
	s.length = 0
	s.byteSize = 0

	return err
}

// NewGPUStorageView creates a non-owning view into an existing GPUStorage
// starting at offsetElems elements from the beginning. The returned storage
// shares the parent's device memory -- no finalizer is set, so the parent
// must outlive the view.
func NewGPUStorageView[T Numeric](parent *GPUStorage[T], offsetElems, length int) *GPUStorage[T] {
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	return &GPUStorage[T]{
		devicePtr: unsafe.Add(parent.devicePtr, offsetElems*elemSize),
		length:    length,
		byteSize:  length * elemSize,
		deviceID:  parent.deviceID,
		runtime:   parent.runtime,
	}
}

// Statically assert that GPUStorage satisfies the Storage interface.
var _ Storage[float32] = (*GPUStorage[float32])(nil)
