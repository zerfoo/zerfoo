//go:build opencl && !cuda && !rocm

package inference

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// createEngine returns a compute engine for the given device string.
// With OpenCL support, "opencl" and "opencl:N" create an OpenCLEngine on the
// specified device; "cpu" creates a CPUEngine.
func createEngine(device string) (compute.Engine[float32], error) {
	devType, deviceID, err := parseDevice(device)
	if err != nil {
		return nil, err
	}
	if devType == "cpu" {
		return compute.NewCPUEngine[float32](numeric.Float32Ops{}), nil
	}
	return compute.NewOpenCLEngine[float32](numeric.Float32Ops{}, deviceID)
}
