package device

import (
	"fmt"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cudaDevice represents an NVIDIA GPU accessible via the CUDA runtime.
type cudaDevice struct {
	id        string
	deviceID  int
	allocator Allocator
}

// newCUDADevice creates a CUDA device instance for the given device ordinal.
func newCUDADevice(deviceID int) *cudaDevice {
	return &cudaDevice{
		id:        fmt.Sprintf("cuda:%d", deviceID),
		deviceID:  deviceID,
		allocator: NewCUDAAllocator(deviceID),
	}
}

// ID returns the device's identifier (e.g., "cuda:0").
func (d *cudaDevice) ID() string { return d.id }

// GetAllocator returns the CUDA device memory allocator.
func (d *cudaDevice) GetAllocator() Allocator { return d.allocator }

// Type returns the CUDA device type.
func (d *cudaDevice) Type() Type { return CUDA }

// Statically assert that the type implements the interface.
var _ Device = (*cudaDevice)(nil)

func init() {
	count, err := cuda.GetDeviceCount()
	if err != nil {
		return // No CUDA runtime or driver; silently skip registration
	}

	for i := range count {
		registerDevice(newCUDADevice(i))
	}
}
