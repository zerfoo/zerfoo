package miopen

import (
	"fmt"
	"os"
	"sync"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// MIOpenLib holds dlopen handles and resolved function pointers for
// MIOpen functions. All function pointers are resolved at Open()
// time via dlsym.
type MIOpenLib struct {
	handle uintptr

	miopenCreate  uintptr
	miopenDestroy uintptr

	miopenSetStream uintptr

	miopenCreateTensorDescriptor  uintptr
	miopenSet4dTensorDescriptor   uintptr
	miopenDestroyTensorDescriptor uintptr

	miopenCreateConvolutionDescriptor  uintptr
	miopenInitConvolutionDescriptor    uintptr
	miopenSetConvolutionGroupCount     uintptr
	miopenDestroyConvolutionDescriptor uintptr

	miopenCreateActivationDescriptor  uintptr
	miopenSetActivationDescriptor     uintptr
	miopenDestroyActivationDescriptor uintptr

	miopenCreatePoolingDescriptor  uintptr
	miopenSet2dPoolingDescriptor   uintptr
	miopenDestroyPoolingDescriptor uintptr

	miopenConvolutionForwardGetWorkSpaceSize uintptr
	miopenFindConvolutionForwardAlgorithm    uintptr
	miopenConvolutionForward                 uintptr

	miopenBatchNormalizationForwardInference uintptr
	miopenActivationForward                  uintptr
	miopenPoolingForward                     uintptr
	miopenSoftmaxForwardV2                   uintptr
	miopenOpTensor                           uintptr
	miopenGetPoolingForwardOutputDim         uintptr
	miopenPoolingGetWorkSpaceSize            uintptr
}

var (
	globalLib  *MIOpenLib
	globalOnce sync.Once
	errGlobal  error
)

// miopenLibPathOverrideEnv names an environment variable that, if set to a
// vetted absolute path, is tried before trustedMiopenLibPaths. See
// cuda.VetAbsoluteLibPath for the safety checks applied.
const miopenLibPathOverrideEnv = "ZERFOO_MIOPEN_LIB_PATH"

// trustedMiopenLibPaths use /opt/rocm/lib, AMD's standard ROCm install
// prefix (see the identical rationale on trustedHipLibPaths in
// internal/hip/purego.go).
var trustedMiopenLibPaths = []string{
	"/opt/rocm/lib/libMIOpen.so.1",
	"/opt/rocm/lib/libMIOpen.so",
}

// miopenPaths lists the shared library candidates to try, in order.
//
// SECURITY (CUDA-2, docs/deep-reviews/002-full-codebase.md): the trailing
// bare-soname entries are a DOCUMENTED residual trust assumption -- see the
// equivalent comment on hipPaths in internal/hip/purego.go for the full
// rationale.
var miopenPaths = buildMiopenLibPaths()

func buildMiopenLibPaths() []string {
	paths := make([]string, 0, len(trustedMiopenLibPaths)+3)
	if override := os.Getenv(miopenLibPathOverrideEnv); override != "" {
		if vetted, ok := cuda.VetAbsoluteLibPath(override); ok {
			paths = append(paths, vetted)
		}
	}
	paths = append(paths, trustedMiopenLibPaths...)
	paths = append(paths, "libMIOpen.so.1", "libMIOpen.so")
	return paths
}

// Open loads libMIOpen via dlopen and resolves all function pointers.
func Open() (*MIOpenLib, error) {
	lib := &MIOpenLib{}

	var lastErr error
	for _, path := range miopenPaths {
		h, err := cuda.DlopenPath(path)
		if err == nil {
			lib.handle = h
			break
		}
		lastErr = err
	}
	if lib.handle == 0 {
		return nil, fmt.Errorf("miopen: dlopen libMIOpen failed: %w", lastErr)
	}

	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"miopenCreate", &lib.miopenCreate},
		{"miopenDestroy", &lib.miopenDestroy},
		{"miopenSetStream", &lib.miopenSetStream},
		{"miopenCreateTensorDescriptor", &lib.miopenCreateTensorDescriptor},
		{"miopenSet4dTensorDescriptor", &lib.miopenSet4dTensorDescriptor},
		{"miopenDestroyTensorDescriptor", &lib.miopenDestroyTensorDescriptor},
		{"miopenCreateConvolutionDescriptor", &lib.miopenCreateConvolutionDescriptor},
		{"miopenInitConvolutionDescriptor", &lib.miopenInitConvolutionDescriptor},
		{"miopenSetConvolutionGroupCount", &lib.miopenSetConvolutionGroupCount},
		{"miopenDestroyConvolutionDescriptor", &lib.miopenDestroyConvolutionDescriptor},
		{"miopenCreateActivationDescriptor", &lib.miopenCreateActivationDescriptor},
		{"miopenSetActivationDescriptor", &lib.miopenSetActivationDescriptor},
		{"miopenDestroyActivationDescriptor", &lib.miopenDestroyActivationDescriptor},
		{"miopenCreatePoolingDescriptor", &lib.miopenCreatePoolingDescriptor},
		{"miopenSet2dPoolingDescriptor", &lib.miopenSet2dPoolingDescriptor},
		{"miopenDestroyPoolingDescriptor", &lib.miopenDestroyPoolingDescriptor},
		{"miopenConvolutionForwardGetWorkSpaceSize", &lib.miopenConvolutionForwardGetWorkSpaceSize},
		{"miopenFindConvolutionForwardAlgorithm", &lib.miopenFindConvolutionForwardAlgorithm},
		{"miopenConvolutionForward", &lib.miopenConvolutionForward},
		{"miopenBatchNormalizationForwardInference", &lib.miopenBatchNormalizationForwardInference},
		{"miopenActivationForward", &lib.miopenActivationForward},
		{"miopenPoolingForward", &lib.miopenPoolingForward},
		{"miopenSoftmaxForward_V2", &lib.miopenSoftmaxForwardV2},
		{"miopenOpTensor", &lib.miopenOpTensor},
		{"miopenGetPoolingForwardOutputDim", &lib.miopenGetPoolingForwardOutputDim},
		{"miopenPoolingGetWorkSpaceSize", &lib.miopenPoolingGetWorkSpaceSize},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(lib.handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("miopen: %w", err)
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Available returns true if MIOpen is loadable on this machine.
func Available() bool {
	globalOnce.Do(func() {
		globalLib, errGlobal = Open()
	})
	return errGlobal == nil
}

// Lib returns the global MIOpenLib instance, or nil if not available.
func Lib() *MIOpenLib {
	if !Available() {
		return nil
	}
	return globalLib
}

func lib() *MIOpenLib {
	return Lib()
}
