package miopen

import (
	"fmt"
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

var miopenPaths = []string{
	"libMIOpen.so.1",
	"libMIOpen.so",
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
