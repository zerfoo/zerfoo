//go:build !rocm && !opencl

package tensor

import (
	"sync"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
)

// defaultRuntime is lazily initialized on first use. It provides backward
// compatibility for callers that do not supply a runtime explicitly.
var (
	defaultRuntime     gpuapi.Runtime
	defaultRuntimeOnce sync.Once
)

func getDefaultRuntime() gpuapi.Runtime {
	defaultRuntimeOnce.Do(func() {
		if cuda.Available() {
			defaultRuntime = gpuapi.NewCUDARuntime()
		}
	})
	return defaultRuntime
}
