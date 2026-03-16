package cudnn

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cuDNN status codes.
const cudnnStatusSuccess = 0

// cuDNN data type constants (cudnnDataType_t).
const (
	cudnnDataFloat  = 0
	cudnnDataDouble = 1
	cudnnDataHalf   = 2
	cudnnDataInt8   = 3
	cudnnDataInt32  = 4
)

// cuDNN tensor format constants (cudnnTensorFormat_t).
const (
	cudnnTensorNCHW = 0
	cudnnTensorNHWC = 1
)

// cuDNN activation mode constants (cudnnActivationMode_t).
const (
	cudnnActivationSigmoid     = 0
	cudnnActivationRelu        = 1
	cudnnActivationTanh        = 2
	cudnnActivationClippedRelu = 3
	cudnnActivationElu         = 4
)

// cuDNN pooling mode constants (cudnnPoolingMode_t).
const (
	cudnnPoolingMax                    = 0
	cudnnPoolingAverageCountIncludePad = 1
	cudnnPoolingAverageCountExcludePad = 2
)

// cuDNN NaN propagation constants (cudnnNanPropagation_t).
const (
	cudnnNotPropagateNan = 0
	cudnnPropagateNan    = 1
)

// cuDNN convolution mode constants (cudnnConvolutionMode_t).
const (
	cudnnConvolution      = 0
	cudnnCrossCorrelation = 1
)

// cuDNN batch normalization mode constants (cudnnBatchNormMode_t).
const (
	cudnnBatchNormPerActivation = 0
	cudnnBatchNormSpatial       = 1
)

// cuDNN softmax algorithm constants (cudnnSoftmaxAlgorithm_t).
const (
	cudnnSoftmaxFast     = 0
	cudnnSoftmaxAccurate = 1
	cudnnSoftmaxLog      = 2
)

// cuDNN softmax mode constants (cudnnSoftmaxMode_t).
const (
	cudnnSoftmaxModeInstance = 0
	cudnnSoftmaxModeChannel  = 1
)

// cuDNN convolution forward algorithm constants (cudnnConvolutionFwdAlgo_t).
const (
	cudnnConvFwdAlgoImplicitGemm        = 0
	cudnnConvFwdAlgoImplicitPrecompGemm = 1
	cudnnConvFwdAlgoGemm                = 2
	cudnnConvFwdAlgoFFT                 = 4
	cudnnConvFwdAlgoWinograd            = 6
)

// cuDNN convolution backward data algorithm constants (cudnnConvolutionBwdDataAlgo_t).
const (
	cudnnConvBwdDataAlgo0        = 0
	cudnnConvBwdDataAlgo1        = 1
	cudnnConvBwdDataAlgoFFT      = 2
	cudnnConvBwdDataAlgoWinograd = 4
)

// cuDNN convolution backward filter algorithm constants (cudnnConvolutionBwdFilterAlgo_t).
const (
	cudnnConvBwdFilterAlgo0        = 0
	cudnnConvBwdFilterAlgo1        = 1
	cudnnConvBwdFilterAlgoFFT      = 2
	cudnnConvBwdFilterAlgo3        = 3
	cudnnConvBwdFilterAlgoWinograd = 4
)

// --- Data types ---

// DataType maps to cudnnDataType_t.
type DataType int

const (
	Float32 DataType = cudnnDataFloat
	Float64 DataType = cudnnDataDouble
	Float16 DataType = cudnnDataHalf
	Int32   DataType = cudnnDataInt32
	Int8    DataType = cudnnDataInt8
)

// TensorFormat maps to cudnnTensorFormat_t.
type TensorFormat int

const (
	NCHW TensorFormat = cudnnTensorNCHW
	NHWC TensorFormat = cudnnTensorNHWC
)

// ActivationMode maps to cudnnActivationMode_t.
type ActivationMode int

const (
	ActivationSigmoid     ActivationMode = cudnnActivationSigmoid
	ActivationReLU        ActivationMode = cudnnActivationRelu
	ActivationTanh        ActivationMode = cudnnActivationTanh
	ActivationClippedReLU ActivationMode = cudnnActivationClippedRelu
	ActivationELU         ActivationMode = cudnnActivationElu
)

// PoolingMode maps to cudnnPoolingMode_t.
type PoolingMode int

const (
	PoolingMax                    PoolingMode = cudnnPoolingMax
	PoolingAverageCountIncludePad PoolingMode = cudnnPoolingAverageCountIncludePad
	PoolingAverageCountExcludePad PoolingMode = cudnnPoolingAverageCountExcludePad
)

// NanPropagation maps to cudnnNanPropagation_t.
type NanPropagation int

const (
	NotPropagateNan NanPropagation = cudnnNotPropagateNan
	PropagateNan    NanPropagation = cudnnPropagateNan
)

// ConvolutionMode maps to cudnnConvolutionMode_t.
type ConvolutionMode int

const (
	Convolution      ConvolutionMode = cudnnConvolution
	CrossCorrelation ConvolutionMode = cudnnCrossCorrelation
)

// BatchNormMode maps to cudnnBatchNormMode_t.
type BatchNormMode int

const (
	BatchNormPerActivation BatchNormMode = cudnnBatchNormPerActivation
	BatchNormSpatial       BatchNormMode = cudnnBatchNormSpatial
)

// SoftmaxAlgorithm maps to cudnnSoftmaxAlgorithm_t.
type SoftmaxAlgorithm int

const (
	SoftmaxFast     SoftmaxAlgorithm = cudnnSoftmaxFast
	SoftmaxAccurate SoftmaxAlgorithm = cudnnSoftmaxAccurate
	SoftmaxLog      SoftmaxAlgorithm = cudnnSoftmaxLog
)

// SoftmaxMode maps to cudnnSoftmaxMode_t.
type SoftmaxMode int

const (
	SoftmaxModeInstance SoftmaxMode = cudnnSoftmaxModeInstance
	SoftmaxModeChannel  SoftmaxMode = cudnnSoftmaxModeChannel
)

// ConvFwdAlgo maps to cudnnConvolutionFwdAlgo_t.
type ConvFwdAlgo int

const (
	ConvFwdAlgoImplicitGemm        ConvFwdAlgo = cudnnConvFwdAlgoImplicitGemm
	ConvFwdAlgoImplicitPrecompGemm ConvFwdAlgo = cudnnConvFwdAlgoImplicitPrecompGemm
	ConvFwdAlgoGemm                ConvFwdAlgo = cudnnConvFwdAlgoGemm
	ConvFwdAlgoFFT                 ConvFwdAlgo = cudnnConvFwdAlgoFFT
	ConvFwdAlgoWinograd            ConvFwdAlgo = cudnnConvFwdAlgoWinograd
)

// ConvBwdDataAlgo maps to cudnnConvolutionBwdDataAlgo_t.
type ConvBwdDataAlgo int

const (
	ConvBwdDataAlgo0        ConvBwdDataAlgo = cudnnConvBwdDataAlgo0
	ConvBwdDataAlgo1        ConvBwdDataAlgo = cudnnConvBwdDataAlgo1
	ConvBwdDataAlgoFFT      ConvBwdDataAlgo = cudnnConvBwdDataAlgoFFT
	ConvBwdDataAlgoWinograd ConvBwdDataAlgo = cudnnConvBwdDataAlgoWinograd
)

// ConvBwdFilterAlgo maps to cudnnConvolutionBwdFilterAlgo_t.
type ConvBwdFilterAlgo int

const (
	ConvBwdFilterAlgo0        ConvBwdFilterAlgo = cudnnConvBwdFilterAlgo0
	ConvBwdFilterAlgo1        ConvBwdFilterAlgo = cudnnConvBwdFilterAlgo1
	ConvBwdFilterAlgoFFT      ConvBwdFilterAlgo = cudnnConvBwdFilterAlgoFFT
	ConvBwdFilterAlgo3        ConvBwdFilterAlgo = cudnnConvBwdFilterAlgo3
	ConvBwdFilterAlgoWinograd ConvBwdFilterAlgo = cudnnConvBwdFilterAlgoWinograd
)

// --- Library loading ---

// cuDNNLib holds dlopen handle and resolved function pointers for cuDNN.
type cuDNNLib struct {
	handle uintptr

	// Handle lifecycle
	cudnnCreate         uintptr
	cudnnDestroy        uintptr
	cudnnSetStream      uintptr
	cudnnGetErrorString uintptr

	// Tensor descriptor
	cudnnCreateTensorDescriptor  uintptr
	cudnnDestroyTensorDescriptor uintptr
	cudnnSetTensor4dDescriptor   uintptr
	cudnnSetTensorNdDescriptor   uintptr

	// Filter descriptor
	cudnnCreateFilterDescriptor  uintptr
	cudnnDestroyFilterDescriptor uintptr
	cudnnSetFilter4dDescriptor   uintptr

	// Convolution descriptor
	cudnnCreateConvolutionDescriptor  uintptr
	cudnnDestroyConvolutionDescriptor uintptr
	cudnnSetConvolution2dDescriptor   uintptr
	cudnnSetConvolutionGroupCount     uintptr

	// Activation descriptor
	cudnnCreateActivationDescriptor  uintptr
	cudnnDestroyActivationDescriptor uintptr
	cudnnSetActivationDescriptor     uintptr

	// Pooling descriptor
	cudnnCreatePoolingDescriptor  uintptr
	cudnnDestroyPoolingDescriptor uintptr
	cudnnSetPooling2dDescriptor   uintptr

	// Forward operations
	cudnnConvolutionForward                 uintptr
	cudnnGetConvolutionForwardWorkspaceSize uintptr
	cudnnBatchNormalizationForwardInference uintptr
	cudnnBatchNormalizationForwardTraining  uintptr
	cudnnActivationForward                  uintptr
	cudnnPoolingForward                     uintptr
	cudnnAddTensor                          uintptr
	cudnnSoftmaxForward                     uintptr

	// Backward operations
	cudnnConvolutionBackwardData                   uintptr
	cudnnGetConvolutionBackwardDataWorkspaceSize   uintptr
	cudnnConvolutionBackwardFilter                 uintptr
	cudnnGetConvolutionBackwardFilterWorkspaceSize uintptr
	cudnnBatchNormalizationBackward                uintptr
	cudnnActivationBackward                        uintptr
	cudnnPoolingBackward                           uintptr
}

var (
	globalCuDNN     *cuDNNLib
	globalCuDNNOnce sync.Once
	errCuDNN        error
)

// cudnnPaths lists the shared library names to try, in order.
var cudnnPaths = []string{
	"libcudnn.so.9",
	"libcudnn.so.8",
	"libcudnn.so",
}

// openCuDNN loads libcudnn via dlopen and resolves all function pointers.
func openCuDNN() (*cuDNNLib, error) {
	lib := &cuDNNLib{}

	var lastErr error
	for _, path := range cudnnPaths {
		h, err := cuda.DlopenPath(path)
		if err == nil {
			lib.handle = h
			break
		}
		lastErr = err
	}
	if lib.handle == 0 {
		return nil, fmt.Errorf("cudnn: dlopen failed: %w", lastErr)
	}

	type sym struct {
		name string
		ptr  *uintptr
	}
	syms := []sym{
		{"cudnnCreate", &lib.cudnnCreate},
		{"cudnnDestroy", &lib.cudnnDestroy},
		{"cudnnSetStream", &lib.cudnnSetStream},
		{"cudnnGetErrorString", &lib.cudnnGetErrorString},

		{"cudnnCreateTensorDescriptor", &lib.cudnnCreateTensorDescriptor},
		{"cudnnDestroyTensorDescriptor", &lib.cudnnDestroyTensorDescriptor},
		{"cudnnSetTensor4dDescriptor", &lib.cudnnSetTensor4dDescriptor},
		{"cudnnSetTensorNdDescriptor", &lib.cudnnSetTensorNdDescriptor},

		{"cudnnCreateFilterDescriptor", &lib.cudnnCreateFilterDescriptor},
		{"cudnnDestroyFilterDescriptor", &lib.cudnnDestroyFilterDescriptor},
		{"cudnnSetFilter4dDescriptor", &lib.cudnnSetFilter4dDescriptor},

		{"cudnnCreateConvolutionDescriptor", &lib.cudnnCreateConvolutionDescriptor},
		{"cudnnDestroyConvolutionDescriptor", &lib.cudnnDestroyConvolutionDescriptor},
		{"cudnnSetConvolution2dDescriptor", &lib.cudnnSetConvolution2dDescriptor},
		{"cudnnSetConvolutionGroupCount", &lib.cudnnSetConvolutionGroupCount},

		{"cudnnCreateActivationDescriptor", &lib.cudnnCreateActivationDescriptor},
		{"cudnnDestroyActivationDescriptor", &lib.cudnnDestroyActivationDescriptor},
		{"cudnnSetActivationDescriptor", &lib.cudnnSetActivationDescriptor},

		{"cudnnCreatePoolingDescriptor", &lib.cudnnCreatePoolingDescriptor},
		{"cudnnDestroyPoolingDescriptor", &lib.cudnnDestroyPoolingDescriptor},
		{"cudnnSetPooling2dDescriptor", &lib.cudnnSetPooling2dDescriptor},

		{"cudnnConvolutionForward", &lib.cudnnConvolutionForward},
		{"cudnnGetConvolutionForwardWorkspaceSize", &lib.cudnnGetConvolutionForwardWorkspaceSize},
		{"cudnnBatchNormalizationForwardInference", &lib.cudnnBatchNormalizationForwardInference},
		{"cudnnBatchNormalizationForwardTraining", &lib.cudnnBatchNormalizationForwardTraining},
		{"cudnnActivationForward", &lib.cudnnActivationForward},
		{"cudnnPoolingForward", &lib.cudnnPoolingForward},
		{"cudnnAddTensor", &lib.cudnnAddTensor},
		{"cudnnSoftmaxForward", &lib.cudnnSoftmaxForward},

		{"cudnnConvolutionBackwardData", &lib.cudnnConvolutionBackwardData},
		{"cudnnGetConvolutionBackwardDataWorkspaceSize", &lib.cudnnGetConvolutionBackwardDataWorkspaceSize},
		{"cudnnConvolutionBackwardFilter", &lib.cudnnConvolutionBackwardFilter},
		{"cudnnGetConvolutionBackwardFilterWorkspaceSize", &lib.cudnnGetConvolutionBackwardFilterWorkspaceSize},
		{"cudnnBatchNormalizationBackward", &lib.cudnnBatchNormalizationBackward},
		{"cudnnActivationBackward", &lib.cudnnActivationBackward},
		{"cudnnPoolingBackward", &lib.cudnnPoolingBackward},
	}
	for _, s := range syms {
		addr, err := cuda.Dlsym(lib.handle, s.name)
		if err != nil {
			return nil, fmt.Errorf("cudnn: %w", err)
		}
		*s.ptr = addr
	}

	return lib, nil
}

// Available returns true if libcudnn can be loaded on this machine.
func Available() bool {
	globalCuDNNOnce.Do(func() {
		globalCuDNN, errCuDNN = openCuDNN()
	})
	return errCuDNN == nil
}

func lib() *cuDNNLib {
	if !Available() {
		return nil
	}
	return globalCuDNN
}

// statusError checks a cuDNN status return code. Returns nil on success.
func statusError(status uintptr, context string) error {
	if status == cudnnStatusSuccess {
		return nil
	}
	l := lib()
	if l == nil {
		return fmt.Errorf("%s: cudnn error %d (library not loaded)", context, status)
	}
	ptr := cuda.Ccall(l.cudnnGetErrorString, status)
	msg := "unknown error"
	if ptr != 0 {
		msg = goStringFromPtr(ptr)
	}
	return fmt.Errorf("%s: %s", context, msg)
}

// ptrFromUintptr converts a uintptr to unsafe.Pointer without triggering
// go vet's unsafeptr analyzer. Required for purego FFI.
func ptrFromUintptr(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// goStringFromPtr converts a C string pointer to a Go string.
func goStringFromPtr(p uintptr) string {
	if p == 0 {
		return ""
	}
	ptr := (*byte)(ptrFromUintptr(p))
	var n int
	for *(*byte)(unsafe.Add(unsafe.Pointer(ptr), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(ptr, n))
}

// --- Handle ---

// Handle wraps a cudnnHandle_t. Create one per GPUEngine.
type Handle struct {
	h uintptr
}

// CreateHandle creates a new cuDNN handle.
func CreateHandle() (*Handle, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudnnCreate: cudnn not available")
	}
	var h uintptr
	ret := cuda.Ccall(l.cudnnCreate, uintptr(unsafe.Pointer(&h)))
	if err := statusError(ret, "cudnnCreate"); err != nil {
		return nil, err
	}
	return &Handle{h: h}, nil
}

// SetStream associates the handle with a CUDA stream.
func (h *Handle) SetStream(stream unsafe.Pointer) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSetStream: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnSetStream, h.h, uintptr(stream))
	return statusError(ret, "cudnnSetStream")
}

// Destroy releases the cuDNN handle resources.
func (h *Handle) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnDestroy: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnDestroy, h.h)
	return statusError(ret, "cudnnDestroy")
}

// --- TensorDescriptor ---

// TensorDescriptor wraps a cudnnTensorDescriptor_t.
type TensorDescriptor struct {
	d uintptr
}

// CreateTensorDescriptor allocates a new tensor descriptor.
func CreateTensorDescriptor() (*TensorDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudnnCreateTensorDescriptor: cudnn not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.cudnnCreateTensorDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "cudnnCreateTensorDescriptor"); err != nil {
		return nil, err
	}
	return &TensorDescriptor{d: d}, nil
}

// Set4d sets the tensor descriptor to a 4D layout (N, C, H, W).
func (t *TensorDescriptor) Set4d(format TensorFormat, dtype DataType, n, c, h, w int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSetTensor4dDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnSetTensor4dDescriptor,
		t.d,
		uintptr(format),
		uintptr(dtype),
		uintptr(n), uintptr(c), uintptr(h), uintptr(w),
	)
	return statusError(ret, "cudnnSetTensor4dDescriptor")
}

// SetNd sets the tensor descriptor to an N-dimensional layout.
func (t *TensorDescriptor) SetNd(dtype DataType, dims, strides []int) error {
	if len(dims) != len(strides) {
		return fmt.Errorf("cudnnSetTensorNdDescriptor: dims length %d != strides length %d", len(dims), len(strides))
	}
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSetTensorNdDescriptor: cudnn not available")
	}
	nd := len(dims)
	cDims := make([]int32, nd)
	cStrides := make([]int32, nd)
	for i := 0; i < nd; i++ {
		cDims[i] = int32(dims[i])
		cStrides[i] = int32(strides[i])
	}
	ret := cuda.Ccall(l.cudnnSetTensorNdDescriptor,
		t.d,
		uintptr(dtype),
		uintptr(nd),
		uintptr(unsafe.Pointer(&cDims[0])),
		uintptr(unsafe.Pointer(&cStrides[0])),
	)
	return statusError(ret, "cudnnSetTensorNdDescriptor")
}

// Destroy releases the tensor descriptor.
func (t *TensorDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnDestroyTensorDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnDestroyTensorDescriptor, t.d)
	return statusError(ret, "cudnnDestroyTensorDescriptor")
}

// --- FilterDescriptor ---

// FilterDescriptor wraps a cudnnFilterDescriptor_t.
type FilterDescriptor struct {
	d uintptr
}

// CreateFilterDescriptor allocates a new filter descriptor.
func CreateFilterDescriptor() (*FilterDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudnnCreateFilterDescriptor: cudnn not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.cudnnCreateFilterDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "cudnnCreateFilterDescriptor"); err != nil {
		return nil, err
	}
	return &FilterDescriptor{d: d}, nil
}

// Set4d sets the filter descriptor to a 4D layout (K, C, H, W).
func (f *FilterDescriptor) Set4d(dtype DataType, format TensorFormat, k, c, h, w int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSetFilter4dDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnSetFilter4dDescriptor,
		f.d,
		uintptr(dtype),
		uintptr(format),
		uintptr(k), uintptr(c), uintptr(h), uintptr(w),
	)
	return statusError(ret, "cudnnSetFilter4dDescriptor")
}

// Destroy releases the filter descriptor.
func (f *FilterDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnDestroyFilterDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnDestroyFilterDescriptor, f.d)
	return statusError(ret, "cudnnDestroyFilterDescriptor")
}

// --- ConvolutionDescriptor ---

// ConvolutionDescriptor wraps a cudnnConvolutionDescriptor_t.
type ConvolutionDescriptor struct {
	d uintptr
}

// CreateConvolutionDescriptor allocates a new convolution descriptor.
func CreateConvolutionDescriptor() (*ConvolutionDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudnnCreateConvolutionDescriptor: cudnn not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.cudnnCreateConvolutionDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "cudnnCreateConvolutionDescriptor"); err != nil {
		return nil, err
	}
	return &ConvolutionDescriptor{d: d}, nil
}

// Set2d configures the convolution descriptor for 2D convolution.
func (c *ConvolutionDescriptor) Set2d(padH, padW, strideH, strideW, dilationH, dilationW int, mode ConvolutionMode, dtype DataType) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSetConvolution2dDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnSetConvolution2dDescriptor,
		c.d,
		uintptr(padH), uintptr(padW),
		uintptr(strideH), uintptr(strideW),
		uintptr(dilationH), uintptr(dilationW),
		uintptr(mode),
		uintptr(dtype),
	)
	return statusError(ret, "cudnnSetConvolution2dDescriptor")
}

// SetGroupCount sets the number of groups for grouped convolution.
func (c *ConvolutionDescriptor) SetGroupCount(groups int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSetConvolutionGroupCount: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnSetConvolutionGroupCount, c.d, uintptr(groups))
	return statusError(ret, "cudnnSetConvolutionGroupCount")
}

// Destroy releases the convolution descriptor.
func (c *ConvolutionDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnDestroyConvolutionDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnDestroyConvolutionDescriptor, c.d)
	return statusError(ret, "cudnnDestroyConvolutionDescriptor")
}

// --- ActivationDescriptor ---

// ActivationDescriptor wraps a cudnnActivationDescriptor_t.
type ActivationDescriptor struct {
	d uintptr
}

// CreateActivationDescriptor allocates a new activation descriptor.
func CreateActivationDescriptor() (*ActivationDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudnnCreateActivationDescriptor: cudnn not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.cudnnCreateActivationDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "cudnnCreateActivationDescriptor"); err != nil {
		return nil, err
	}
	return &ActivationDescriptor{d: d}, nil
}

// Set configures the activation descriptor.
func (a *ActivationDescriptor) Set(mode ActivationMode, nanProp NanPropagation, coef float64) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSetActivationDescriptor: cudnn not available")
	}
	// Pass float64 as two uintptr halves (double is 8 bytes = 1 uintptr on 64-bit).
	coefBits := *(*uintptr)(unsafe.Pointer(&coef))
	ret := cuda.Ccall(l.cudnnSetActivationDescriptor,
		a.d,
		uintptr(mode),
		uintptr(nanProp),
		coefBits,
	)
	return statusError(ret, "cudnnSetActivationDescriptor")
}

// Destroy releases the activation descriptor.
func (a *ActivationDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnDestroyActivationDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnDestroyActivationDescriptor, a.d)
	return statusError(ret, "cudnnDestroyActivationDescriptor")
}

// --- PoolingDescriptor ---

// PoolingDescriptor wraps a cudnnPoolingDescriptor_t.
type PoolingDescriptor struct {
	d uintptr
}

// CreatePoolingDescriptor allocates a new pooling descriptor.
func CreatePoolingDescriptor() (*PoolingDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudnnCreatePoolingDescriptor: cudnn not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.cudnnCreatePoolingDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "cudnnCreatePoolingDescriptor"); err != nil {
		return nil, err
	}
	return &PoolingDescriptor{d: d}, nil
}

// Set2d configures the pooling descriptor for 2D pooling.
func (p *PoolingDescriptor) Set2d(mode PoolingMode, nanProp NanPropagation, windowH, windowW, padH, padW, strideH, strideW int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSetPooling2dDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnSetPooling2dDescriptor,
		p.d,
		uintptr(mode),
		uintptr(nanProp),
		uintptr(windowH), uintptr(windowW),
		uintptr(padH), uintptr(padW),
		uintptr(strideH), uintptr(strideW),
	)
	return statusError(ret, "cudnnSetPooling2dDescriptor")
}

// Destroy releases the pooling descriptor.
func (p *PoolingDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnDestroyPoolingDescriptor: cudnn not available")
	}
	ret := cuda.Ccall(l.cudnnDestroyPoolingDescriptor, p.d)
	return statusError(ret, "cudnnDestroyPoolingDescriptor")
}

// --- Forward Operations ---

// ConvolutionForward performs a forward convolution.
func (h *Handle) ConvolutionForward(
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	wDesc *FilterDescriptor, w unsafe.Pointer,
	convDesc *ConvolutionDescriptor,
	algo ConvFwdAlgo,
	workspace unsafe.Pointer, workspaceSize int,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnConvolutionForward: cudnn not available")
	}
	a := alpha
	b := beta
	ret := cuda.Ccall(l.cudnnConvolutionForward,
		h.h,
		uintptr(unsafe.Pointer(&a)),
		xDesc.d, uintptr(x),
		wDesc.d, uintptr(w),
		convDesc.d,
		uintptr(algo),
		uintptr(workspace), uintptr(workspaceSize),
		uintptr(unsafe.Pointer(&b)),
		yDesc.d, uintptr(y),
	)
	return statusError(ret, "cudnnConvolutionForward")
}

// GetConvolutionForwardWorkspaceSize returns the workspace size in bytes.
func (h *Handle) GetConvolutionForwardWorkspaceSize(
	xDesc *TensorDescriptor,
	wDesc *FilterDescriptor,
	convDesc *ConvolutionDescriptor,
	yDesc *TensorDescriptor,
	algo ConvFwdAlgo,
) (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("cudnnGetConvolutionForwardWorkspaceSize: cudnn not available")
	}
	var size uintptr
	ret := cuda.Ccall(l.cudnnGetConvolutionForwardWorkspaceSize,
		h.h,
		xDesc.d,
		wDesc.d,
		convDesc.d,
		yDesc.d,
		uintptr(algo),
		uintptr(unsafe.Pointer(&size)),
	)
	if err := statusError(ret, "cudnnGetConvolutionForwardWorkspaceSize"); err != nil {
		return 0, err
	}
	return int(size), nil
}

// BatchNormalizationForwardInference performs batch normalization in inference mode.
func (h *Handle) BatchNormalizationForwardInference(
	mode BatchNormMode,
	alpha, beta float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	bnScaleBiasMeanVarDesc *TensorDescriptor,
	bnScale, bnBias unsafe.Pointer,
	estimatedMean, estimatedVariance unsafe.Pointer,
	epsilon float64,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnBatchNormalizationForwardInference: cudnn not available")
	}
	a := alpha
	b := beta
	epsBits := *(*uintptr)(unsafe.Pointer(&epsilon))
	ret := cuda.Ccall(l.cudnnBatchNormalizationForwardInference,
		h.h,
		uintptr(mode),
		uintptr(unsafe.Pointer(&a)),
		uintptr(unsafe.Pointer(&b)),
		xDesc.d, uintptr(x),
		yDesc.d, uintptr(y),
		bnScaleBiasMeanVarDesc.d,
		uintptr(bnScale), uintptr(bnBias),
		uintptr(estimatedMean), uintptr(estimatedVariance),
		epsBits,
	)
	return statusError(ret, "cudnnBatchNormalizationForwardInference")
}

// ActivationForward applies an activation function element-wise.
func (h *Handle) ActivationForward(
	actDesc *ActivationDescriptor,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnActivationForward: cudnn not available")
	}
	a := alpha
	b := beta
	ret := cuda.Ccall(l.cudnnActivationForward,
		h.h,
		actDesc.d,
		uintptr(unsafe.Pointer(&a)),
		xDesc.d, uintptr(x),
		uintptr(unsafe.Pointer(&b)),
		yDesc.d, uintptr(y),
	)
	return statusError(ret, "cudnnActivationForward")
}

// PoolingForward performs a forward pooling operation.
func (h *Handle) PoolingForward(
	poolDesc *PoolingDescriptor,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnPoolingForward: cudnn not available")
	}
	a := alpha
	b := beta
	ret := cuda.Ccall(l.cudnnPoolingForward,
		h.h,
		poolDesc.d,
		uintptr(unsafe.Pointer(&a)),
		xDesc.d, uintptr(x),
		uintptr(unsafe.Pointer(&b)),
		yDesc.d, uintptr(y),
	)
	return statusError(ret, "cudnnPoolingForward")
}

// AddTensor adds a bias tensor to the output: y = alpha*b + beta*y.
func (h *Handle) AddTensor(
	alpha float32,
	bDesc *TensorDescriptor, b unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnAddTensor: cudnn not available")
	}
	a := alpha
	bt := beta
	ret := cuda.Ccall(l.cudnnAddTensor,
		h.h,
		uintptr(unsafe.Pointer(&a)),
		bDesc.d, uintptr(b),
		uintptr(unsafe.Pointer(&bt)),
		yDesc.d, uintptr(y),
	)
	return statusError(ret, "cudnnAddTensor")
}

// --- Backward Operations ---

// ConvolutionBackwardData computes the gradient of the input for 2D convolution.
func (h *Handle) ConvolutionBackwardData(
	alpha float32,
	wDesc *FilterDescriptor, w unsafe.Pointer,
	dyDesc *TensorDescriptor, dy unsafe.Pointer,
	convDesc *ConvolutionDescriptor,
	algo ConvBwdDataAlgo,
	workspace unsafe.Pointer, workspaceSize int,
	beta float32,
	dxDesc *TensorDescriptor, dx unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnConvolutionBackwardData: cudnn not available")
	}
	a := alpha
	b := beta
	ret := cuda.Ccall(l.cudnnConvolutionBackwardData,
		h.h,
		uintptr(unsafe.Pointer(&a)),
		wDesc.d, uintptr(w),
		dyDesc.d, uintptr(dy),
		convDesc.d,
		uintptr(algo),
		uintptr(workspace), uintptr(workspaceSize),
		uintptr(unsafe.Pointer(&b)),
		dxDesc.d, uintptr(dx),
	)
	return statusError(ret, "cudnnConvolutionBackwardData")
}

// GetConvolutionBackwardDataWorkspaceSize returns the workspace size in bytes.
func (h *Handle) GetConvolutionBackwardDataWorkspaceSize(
	wDesc *FilterDescriptor,
	dyDesc *TensorDescriptor,
	convDesc *ConvolutionDescriptor,
	dxDesc *TensorDescriptor,
	algo ConvBwdDataAlgo,
) (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("cudnnGetConvolutionBackwardDataWorkspaceSize: cudnn not available")
	}
	var size uintptr
	ret := cuda.Ccall(l.cudnnGetConvolutionBackwardDataWorkspaceSize,
		h.h,
		wDesc.d,
		dyDesc.d,
		convDesc.d,
		dxDesc.d,
		uintptr(algo),
		uintptr(unsafe.Pointer(&size)),
	)
	if err := statusError(ret, "cudnnGetConvolutionBackwardDataWorkspaceSize"); err != nil {
		return 0, err
	}
	return int(size), nil
}

// ConvolutionBackwardFilter computes the gradient of the filter for 2D convolution.
func (h *Handle) ConvolutionBackwardFilter(
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	dyDesc *TensorDescriptor, dy unsafe.Pointer,
	convDesc *ConvolutionDescriptor,
	algo ConvBwdFilterAlgo,
	workspace unsafe.Pointer, workspaceSize int,
	beta float32,
	dwDesc *FilterDescriptor, dw unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnConvolutionBackwardFilter: cudnn not available")
	}
	a := alpha
	b := beta
	ret := cuda.Ccall(l.cudnnConvolutionBackwardFilter,
		h.h,
		uintptr(unsafe.Pointer(&a)),
		xDesc.d, uintptr(x),
		dyDesc.d, uintptr(dy),
		convDesc.d,
		uintptr(algo),
		uintptr(workspace), uintptr(workspaceSize),
		uintptr(unsafe.Pointer(&b)),
		dwDesc.d, uintptr(dw),
	)
	return statusError(ret, "cudnnConvolutionBackwardFilter")
}

// GetConvolutionBackwardFilterWorkspaceSize returns the workspace size in bytes.
func (h *Handle) GetConvolutionBackwardFilterWorkspaceSize(
	xDesc *TensorDescriptor,
	dyDesc *TensorDescriptor,
	convDesc *ConvolutionDescriptor,
	dwDesc *FilterDescriptor,
	algo ConvBwdFilterAlgo,
) (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("cudnnGetConvolutionBackwardFilterWorkspaceSize: cudnn not available")
	}
	var size uintptr
	ret := cuda.Ccall(l.cudnnGetConvolutionBackwardFilterWorkspaceSize,
		h.h,
		xDesc.d,
		dyDesc.d,
		convDesc.d,
		dwDesc.d,
		uintptr(algo),
		uintptr(unsafe.Pointer(&size)),
	)
	if err := statusError(ret, "cudnnGetConvolutionBackwardFilterWorkspaceSize"); err != nil {
		return 0, err
	}
	return int(size), nil
}

// BatchNormalizationForwardTraining performs batch normalization in training mode.
func (h *Handle) BatchNormalizationForwardTraining(
	mode BatchNormMode,
	alpha, beta float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	bnScaleBiasMeanVarDesc *TensorDescriptor,
	bnScale, bnBias unsafe.Pointer,
	expAvgFactor float64,
	runningMean, runningVariance unsafe.Pointer,
	epsilon float64,
	saveMean, saveInvVariance unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnBatchNormalizationForwardTraining: cudnn not available")
	}
	a := alpha
	b := beta
	expBits := *(*uintptr)(unsafe.Pointer(&expAvgFactor))
	epsBits := *(*uintptr)(unsafe.Pointer(&epsilon))
	ret := cuda.Ccall(l.cudnnBatchNormalizationForwardTraining,
		h.h,
		uintptr(mode),
		uintptr(unsafe.Pointer(&a)),
		uintptr(unsafe.Pointer(&b)),
		xDesc.d, uintptr(x),
		yDesc.d, uintptr(y),
		bnScaleBiasMeanVarDesc.d,
		uintptr(bnScale), uintptr(bnBias),
		expBits,
		uintptr(runningMean), uintptr(runningVariance),
		epsBits,
		uintptr(saveMean), uintptr(saveInvVariance),
	)
	return statusError(ret, "cudnnBatchNormalizationForwardTraining")
}

// BatchNormalizationBackward computes gradients for batch normalization.
func (h *Handle) BatchNormalizationBackward(
	mode BatchNormMode,
	alphaDataDiff, betaDataDiff float32,
	alphaParamDiff, betaParamDiff float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	dyDesc *TensorDescriptor, dy unsafe.Pointer,
	dxDesc *TensorDescriptor, dx unsafe.Pointer,
	bnScaleBiasDiffDesc *TensorDescriptor,
	bnScale unsafe.Pointer,
	dBnScale, dBnBias unsafe.Pointer,
	epsilon float64,
	saveMean, saveInvVariance unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnBatchNormalizationBackward: cudnn not available")
	}
	add := alphaDataDiff
	bdd := betaDataDiff
	apd := alphaParamDiff
	bpd := betaParamDiff
	epsBits := *(*uintptr)(unsafe.Pointer(&epsilon))
	ret := cuda.Ccall(l.cudnnBatchNormalizationBackward,
		h.h,
		uintptr(mode),
		uintptr(unsafe.Pointer(&add)),
		uintptr(unsafe.Pointer(&bdd)),
		uintptr(unsafe.Pointer(&apd)),
		uintptr(unsafe.Pointer(&bpd)),
		xDesc.d, uintptr(x),
		dyDesc.d, uintptr(dy),
		dxDesc.d, uintptr(dx),
		bnScaleBiasDiffDesc.d,
		uintptr(bnScale),
		uintptr(dBnScale), uintptr(dBnBias),
		epsBits,
		uintptr(saveMean), uintptr(saveInvVariance),
	)
	return statusError(ret, "cudnnBatchNormalizationBackward")
}

// ActivationBackward computes the gradient of an activation function.
func (h *Handle) ActivationBackward(
	actDesc *ActivationDescriptor,
	alpha float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	dyDesc *TensorDescriptor, dy unsafe.Pointer,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	dxDesc *TensorDescriptor, dx unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnActivationBackward: cudnn not available")
	}
	a := alpha
	b := beta
	ret := cuda.Ccall(l.cudnnActivationBackward,
		h.h,
		actDesc.d,
		uintptr(unsafe.Pointer(&a)),
		yDesc.d, uintptr(y),
		dyDesc.d, uintptr(dy),
		xDesc.d, uintptr(x),
		uintptr(unsafe.Pointer(&b)),
		dxDesc.d, uintptr(dx),
	)
	return statusError(ret, "cudnnActivationBackward")
}

// PoolingBackward computes the gradient of a pooling operation.
func (h *Handle) PoolingBackward(
	poolDesc *PoolingDescriptor,
	alpha float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	dyDesc *TensorDescriptor, dy unsafe.Pointer,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	dxDesc *TensorDescriptor, dx unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnPoolingBackward: cudnn not available")
	}
	a := alpha
	b := beta
	ret := cuda.Ccall(l.cudnnPoolingBackward,
		h.h,
		poolDesc.d,
		uintptr(unsafe.Pointer(&a)),
		yDesc.d, uintptr(y),
		dyDesc.d, uintptr(dy),
		xDesc.d, uintptr(x),
		uintptr(unsafe.Pointer(&b)),
		dxDesc.d, uintptr(dx),
	)
	return statusError(ret, "cudnnPoolingBackward")
}

// SoftmaxForward computes softmax over the channel dimension.
func (h *Handle) SoftmaxForward(
	algo SoftmaxAlgorithm,
	mode SoftmaxMode,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudnnSoftmaxForward: cudnn not available")
	}
	a := alpha
	b := beta
	ret := cuda.Ccall(l.cudnnSoftmaxForward,
		h.h,
		uintptr(algo),
		uintptr(mode),
		uintptr(unsafe.Pointer(&a)),
		xDesc.d, uintptr(x),
		uintptr(unsafe.Pointer(&b)),
		yDesc.d, uintptr(y),
	)
	return statusError(ret, "cudnnSoftmaxForward")
}
