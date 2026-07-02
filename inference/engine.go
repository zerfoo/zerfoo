//go:build !rocm && !opencl

package inference

import (
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
)

// createEngine returns a compute engine for the given device string.
// "cuda" or "cuda:N" creates a GPUEngine if CUDA is available at runtime.
// "cpu" always creates a CPUEngine.
func createEngine(device string) (compute.Engine[float32], error) {
	devType, deviceID, err := parseDevice(device)
	if err != nil {
		return nil, err
	}
	switch devType {
	case "cpu":
		return compute.NewCPUEngine[float32](numeric.Float32Ops{}), nil
	case "cuda":
		if !cuda.Available() {
			return nil, fmt.Errorf("CUDA device requested but CUDA runtime not available")
		}
		return compute.NewGPUEngine[float32](numeric.Float32Ops{}, deviceID)
	case "rocm":
		return nil, fmt.Errorf("ROCm device requested but binary built without rocm build tag")
	case "opencl":
		return nil, fmt.Errorf("OpenCL device requested but binary built without opencl build tag")
	default:
		return nil, fmt.Errorf("unknown device type: %s", devType)
	}
}
