package device

import (
	"fmt"
	"sync"
)

// Device represents a physical or logical compute device (e.g., CPU, GPU).
// It provides access to the device's properties and its memory allocator.
type Device interface {
	// ID returns the unique identifier for the device (e.g., "cpu", "cuda:0").
	ID() string
	// GetAllocator returns the memory allocator associated with this device.
	GetAllocator() Allocator
	// Type returns the type of the device
	Type() Type
}

// Type is an enum for the kind of device.
type Type int

const (
	// CPU represents the Central Processing Unit device type.
	CPU Type = iota
	CUDA // CUDA represents the NVIDIA GPU device type.
)

// --- Device Registry ---

var (
	devices      = make(map[string]Device)
	devicesMutex = &sync.RWMutex{}
)

// registerDevice adds a device to the global registry.
func registerDevice(dev Device) {
	devicesMutex.Lock()
	defer devicesMutex.Unlock()
	devices[dev.ID()] = dev
}

// Get returns a registered device by its ID.
// It returns an error if no device with that ID is found.
func Get(id string) (Device, error) {
	devicesMutex.RLock()
	defer devicesMutex.RUnlock()
	dev, ok := devices[id]
	if !ok {
		return nil, fmt.Errorf("device not found: %s", id)
	}
	return dev, nil
}

// --- CPU Device ---

// cpuDevice represents the system's main CPU.
type cpuDevice struct {
	id        string
	allocator Allocator
}

// newCPUDevice creates the singleton CPU device instance.
func newCPUDevice() *cpuDevice {
	return &cpuDevice{
		id:        "cpu",
		allocator: NewCPUAllocator(),
	}
}

// ID returns the device's identifier.
func (d *cpuDevice) ID() string {
	return d.id
}

// GetAllocator returns the CPU's memory allocator.
func (d *cpuDevice) GetAllocator() Allocator {
	return d.allocator
}

// Type returns the device type.
func (d *cpuDevice) Type() Type {
	return CPU
}

// init registers the default CPU device when the package is imported.
func init() {
	registerDevice(newCPUDevice())
}
