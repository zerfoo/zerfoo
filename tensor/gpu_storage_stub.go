//go:build !cuda && !rocm && !opencl

package tensor

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
)

// GPUStorage is a stub for non-GPU builds. The type exists so that code can
// compile type assertions against *GPUStorage[T]; those assertions will always
// fail at runtime because no GPU backend is linked.
type GPUStorage[T Numeric] struct {
	length int
}

func NewGPUStorage[T Numeric](length int, deviceID ...int) (*GPUStorage[T], error) {
	return nil, fmt.Errorf("GPUStorage: no GPU backend compiled (need -tags cuda, rocm, or opencl)")
}

func NewGPUStorageFromSlice[T Numeric](data []T, deviceID ...int) (*GPUStorage[T], error) {
	return nil, fmt.Errorf("GPUStorage: no GPU backend compiled")
}

func NewGPUStorageFromPtr[T Numeric](devPtr unsafe.Pointer, length int, deviceID ...int) (*GPUStorage[T], error) {
	return nil, fmt.Errorf("GPUStorage: no GPU backend compiled")
}

func NewManagedGPUStorage[T Numeric](pool gpuapi.MemPool, length int, deviceID ...int) (*GPUStorage[T], error) {
	return nil, fmt.Errorf("GPUStorage: no GPU backend compiled")
}

func (s *GPUStorage[T]) Managed() bool         { return false }
func (s *GPUStorage[T]) Len() int               { return s.length }
func (s *GPUStorage[T]) DeviceID() int          { return 0 }
func (s *GPUStorage[T]) TrySlice() ([]T, error) { return nil, fmt.Errorf("no GPU backend") }
func (s *GPUStorage[T]) Slice() []T             { return nil }
func (s *GPUStorage[T]) TrySet(data []T) error  { return fmt.Errorf("no GPU backend") }
func (s *GPUStorage[T]) Set(data []T)           {}
func (s *GPUStorage[T]) DeviceType() device.Type { return device.CPU }
func (s *GPUStorage[T]) Ptr() unsafe.Pointer    { return nil }
func (s *GPUStorage[T]) Free() error            { return nil }
