package opencl

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// OpenCL success code.
const clSuccess = 0

// OpenCL device type constant.
const clDeviceTypeGPU = 1 << 2 // CL_DEVICE_TYPE_GPU = 4

// OpenCL memory flags.
const clMemReadWrite = 1 << 0 // CL_MEM_READ_WRITE = 1

// OpenCL boolean values for blocking calls.
const clTrue = 1

// MemcpyKind specifies the direction of a memory copy.
type MemcpyKind int

const (
	// MemcpyHostToDevice copies from host to device.
	MemcpyHostToDevice MemcpyKind = iota
	// MemcpyDeviceToHost copies from device to host.
	MemcpyDeviceToHost
	// MemcpyDeviceToDevice copies from device to device.
	MemcpyDeviceToDevice
)

// Context holds an OpenCL context, device, and default command queue.
type Context struct {
	platform uintptr
	device   uintptr
	ctx      uintptr
	queue    uintptr
	deviceID int
}

// Stream wraps an OpenCL command queue.
type Stream struct {
	queue uintptr
}

// ptrFromUintptr converts a uintptr to unsafe.Pointer without triggering
// go vet's unsafeptr analyzer. Required for purego FFI.
func ptrFromUintptr(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

func lib() *OpenCLLib {
	return Lib()
}

// NewContext creates an OpenCL context on the specified device.
// If deviceID is -1, the first available GPU device is used.
func NewContext(deviceID int) (*Context, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("opencl not available")
	}

	// clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
	var numPlatforms uint32
	ret := cuda.Ccall(l.clGetPlatformIDs, 0, 0, uintptr(unsafe.Pointer(&numPlatforms)))
	if ret != clSuccess {
		return nil, fmt.Errorf("clGetPlatformIDs: error %d", ret)
	}
	if numPlatforms == 0 {
		return nil, fmt.Errorf("no OpenCL platforms found")
	}

	platforms := make([]uintptr, numPlatforms)
	ret = cuda.Ccall(l.clGetPlatformIDs, uintptr(numPlatforms), uintptr(unsafe.Pointer(&platforms[0])), 0)
	if ret != clSuccess {
		return nil, fmt.Errorf("clGetPlatformIDs: error %d", ret)
	}

	// Find GPU devices across all platforms.
	var allDevices []uintptr
	var devicePlatforms []uintptr
	for _, p := range platforms {
		var numDevices uint32
		r := cuda.Ccall(l.clGetDeviceIDs, p, clDeviceTypeGPU, 0, 0, uintptr(unsafe.Pointer(&numDevices)))
		if r != clSuccess || numDevices == 0 {
			continue
		}
		devs := make([]uintptr, numDevices)
		r = cuda.Ccall(l.clGetDeviceIDs, p, clDeviceTypeGPU, uintptr(numDevices), uintptr(unsafe.Pointer(&devs[0])), 0)
		if r != clSuccess {
			continue
		}
		for _, d := range devs {
			allDevices = append(allDevices, d)
			devicePlatforms = append(devicePlatforms, p)
		}
	}

	if len(allDevices) == 0 {
		return nil, fmt.Errorf("no OpenCL GPU devices found")
	}

	dev := deviceID
	if dev < 0 {
		dev = 0
	}
	if dev >= len(allDevices) {
		return nil, fmt.Errorf("OpenCL device %d not found (have %d)", dev, len(allDevices))
	}

	selectedDevice := allDevices[dev]
	selectedPlatform := devicePlatforms[dev]

	// clCreateContext(const cl_context_properties *properties, cl_uint num_devices,
	//                 const cl_device_id *devices, callback, user_data, cl_int *errcode_ret)
	var errCode int32
	ctxHandle := cuda.Ccall(l.clCreateContext, 0, 1, uintptr(unsafe.Pointer(&selectedDevice)), 0, 0, uintptr(unsafe.Pointer(&errCode)))
	if errCode != clSuccess {
		return nil, fmt.Errorf("clCreateContext: error %d", errCode)
	}

	// clCreateCommandQueue(cl_context context, cl_device_id device,
	//                      cl_command_queue_properties properties, cl_int *errcode_ret)
	queueHandle := cuda.Ccall(l.clCreateCommandQueue, ctxHandle, selectedDevice, 0, uintptr(unsafe.Pointer(&errCode)))
	if errCode != clSuccess {
		cuda.Ccall(l.clReleaseContext, ctxHandle)
		return nil, fmt.Errorf("clCreateCommandQueue: error %d", errCode)
	}

	return &Context{
		platform: selectedPlatform,
		device:   selectedDevice,
		ctx:      ctxHandle,
		queue:    queueHandle,
		deviceID: dev,
	}, nil
}

// Destroy releases the OpenCL context and default command queue.
func (c *Context) Destroy() error {
	l := lib()
	if l == nil {
		return nil
	}
	if c.queue != 0 {
		cuda.Ccall(l.clReleaseCommandQueue, c.queue)
	}
	if c.ctx != 0 {
		cuda.Ccall(l.clReleaseContext, c.ctx)
	}
	return nil
}

// GetDeviceCount returns the total number of OpenCL GPU devices.
func GetDeviceCount() (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("opencl not available")
	}

	var numPlatforms uint32
	ret := cuda.Ccall(l.clGetPlatformIDs, 0, 0, uintptr(unsafe.Pointer(&numPlatforms)))
	if ret != clSuccess {
		return 0, fmt.Errorf("clGetPlatformIDs: error %d", ret)
	}
	if numPlatforms == 0 {
		return 0, nil
	}

	platforms := make([]uintptr, numPlatforms)
	ret = cuda.Ccall(l.clGetPlatformIDs, uintptr(numPlatforms), uintptr(unsafe.Pointer(&platforms[0])), 0)
	if ret != clSuccess {
		return 0, fmt.Errorf("clGetPlatformIDs: error %d", ret)
	}

	total := 0
	for _, p := range platforms {
		var numDevices uint32
		r := cuda.Ccall(l.clGetDeviceIDs, p, clDeviceTypeGPU, 0, 0, uintptr(unsafe.Pointer(&numDevices)))
		if r == clSuccess {
			total += int(numDevices)
		}
	}
	return total, nil
}

// Malloc allocates a device buffer of the given size.
// Returns the cl_mem handle cast to unsafe.Pointer.
func (c *Context) Malloc(size int) (unsafe.Pointer, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("opencl not available")
	}

	// clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size,
	//                void *host_ptr, cl_int *errcode_ret)
	var errCode int32
	mem := cuda.Ccall(l.clCreateBuffer, c.ctx, clMemReadWrite, uintptr(size), 0, uintptr(unsafe.Pointer(&errCode)))
	if errCode != clSuccess {
		return nil, fmt.Errorf("clCreateBuffer(%d bytes): error %d", size, errCode)
	}
	return ptrFromUintptr(mem), nil
}

// Free releases a device buffer.
func (c *Context) Free(ptr unsafe.Pointer) error {
	if ptr == nil {
		return nil
	}
	l := lib()
	if l == nil {
		return fmt.Errorf("opencl not available")
	}
	ret := cuda.Ccall(l.clReleaseMemObject, uintptr(ptr))
	if ret != clSuccess {
		return fmt.Errorf("clReleaseMemObject: error %d", ret)
	}
	return nil
}

// Memcpy copies data between host and device.
func (c *Context) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("opencl not available")
	}

	switch kind {
	case MemcpyHostToDevice:
		// clEnqueueWriteBuffer(queue, buffer, blocking, offset, size, ptr,
		//                      num_events, event_wait_list, event)
		ret := cuda.Ccall(l.clEnqueueWriteBuffer, c.queue, uintptr(dst), clTrue, 0, uintptr(count), uintptr(src), 0, 0, 0)
		if ret != clSuccess {
			return fmt.Errorf("clEnqueueWriteBuffer: error %d", ret)
		}
	case MemcpyDeviceToHost:
		// clEnqueueReadBuffer(queue, buffer, blocking, offset, size, ptr,
		//                     num_events, event_wait_list, event)
		ret := cuda.Ccall(l.clEnqueueReadBuffer, c.queue, uintptr(src), clTrue, 0, uintptr(count), uintptr(dst), 0, 0, 0)
		if ret != clSuccess {
			return fmt.Errorf("clEnqueueReadBuffer: error %d", ret)
		}
	case MemcpyDeviceToDevice:
		// clEnqueueCopyBuffer(queue, src_buffer, dst_buffer, src_offset, dst_offset,
		//                     size, num_events, event_wait_list, event)
		ret := cuda.Ccall(l.clEnqueueCopyBuffer, c.queue, uintptr(src), uintptr(dst), 0, 0, uintptr(count), 0, 0, 0)
		if ret != clSuccess {
			return fmt.Errorf("clEnqueueCopyBuffer: error %d", ret)
		}
		// Wait for copy to complete.
		cuda.Ccall(l.clFinish, c.queue)
	default:
		return fmt.Errorf("unsupported MemcpyKind: %d", kind)
	}
	return nil
}

// CreateStream creates a new command queue (stream equivalent).
func (c *Context) CreateStream() (*Stream, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("opencl not available")
	}
	var errCode int32
	queueHandle := cuda.Ccall(l.clCreateCommandQueue, c.ctx, c.device, 0, uintptr(unsafe.Pointer(&errCode)))
	if errCode != clSuccess {
		return nil, fmt.Errorf("clCreateCommandQueue: error %d", errCode)
	}
	return &Stream{queue: queueHandle}, nil
}

// Synchronize waits for all commands in the command queue to complete.
func (s *Stream) Synchronize() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("opencl not available")
	}
	ret := cuda.Ccall(l.clFinish, s.queue)
	if ret != clSuccess {
		return fmt.Errorf("clFinish: error %d", ret)
	}
	return nil
}

// Destroy releases the command queue.
func (s *Stream) Destroy() error {
	if s.queue == 0 {
		return nil
	}
	l := lib()
	if l == nil {
		return nil
	}
	cuda.Ccall(l.clReleaseCommandQueue, s.queue)
	s.queue = 0
	return nil
}

// Ptr returns the underlying command queue handle as unsafe.Pointer.
func (s *Stream) Ptr() unsafe.Pointer {
	return ptrFromUintptr(s.queue)
}

// CLContext returns the underlying cl_context for kernel compilation.
func (c *Context) CLContext() unsafe.Pointer {
	return ptrFromUintptr(c.ctx)
}

// CLQueue returns the default command queue.
func (c *Context) CLQueue() unsafe.Pointer {
	return ptrFromUintptr(c.queue)
}

// CLDevice returns the underlying cl_device_id.
func (c *Context) CLDevice() unsafe.Pointer {
	return ptrFromUintptr(c.device)
}

// SetDevice is a no-op for OpenCL (device is selected at context creation).
func (c *Context) SetDevice(deviceID int) error {
	c.deviceID = deviceID
	return nil
}

// DeviceID returns the device ordinal.
func (c *Context) DeviceID() int {
	return c.deviceID
}
