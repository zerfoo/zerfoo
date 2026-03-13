package miopen

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// miopenStatusSuccess is the MIOpen status code for success.
const miopenStatusSuccess = 0

// statusError converts a MIOpen status to a Go error, or nil on success.
func statusError(status uintptr, context string) error {
	if status == miopenStatusSuccess {
		return nil
	}
	return fmt.Errorf("%s: MIOpen status %d", context, int(status))
}

// --- Data types and enums ---

// DataType mirrors miopenDataType_t.
type DataType int

const (
	Float32 DataType = 1 // miopenFloat
	Float16 DataType = 2 // miopenHalf
)

// TensorLayout selects NCHW or NHWC.
type TensorLayout int

const (
	NCHW TensorLayout = 0
	NHWC TensorLayout = 1
)

// ActivationMode mirrors miopenActivationMode_t.
type ActivationMode int

const (
	ActivationReLU    ActivationMode = 3 // miopenActivationRELU
	ActivationSigmoid ActivationMode = 0 // miopenActivationLOGISTIC
	ActivationTanh    ActivationMode = 2 // miopenActivationTANH
	ActivationELU     ActivationMode = 6 // miopenActivationELU
)

// PoolingMode mirrors miopenPoolingMode_t.
type PoolingMode int

const (
	PoolingMax                    PoolingMode = 0 // miopenPoolingMax
	PoolingAverageCountIncludePad PoolingMode = 1 // miopenPoolingAverageInclusive
	PoolingAverageCountExcludePad PoolingMode = 2 // miopenPoolingAverage
)

// BatchNormMode mirrors miopenBatchNormMode_t.
type BatchNormMode int

const (
	BatchNormPerActivation BatchNormMode = 0 // miopenBNPerActivation
	BatchNormSpatial       BatchNormMode = 1 // miopenBNSpatial
)

// ConvMode mirrors miopenConvolutionMode_t.
type ConvMode int

const (
	ConvolutionMode ConvMode = 0 // miopenConvolution
	TransposeMode   ConvMode = 1 // miopenTranspose
)

// SoftmaxAlgorithm selects the softmax computation variant.
type SoftmaxAlgorithm int

const (
	SoftmaxAccurate SoftmaxAlgorithm = 0 // MIOPEN_SOFTMAX_ACCURATE
	SoftmaxLog      SoftmaxAlgorithm = 1 // MIOPEN_SOFTMAX_LOG
	SoftmaxFast     SoftmaxAlgorithm = 2 // MIOPEN_SOFTMAX_FAST
)

// SoftmaxMode selects the dimension for softmax.
type SoftmaxMode int

const (
	SoftmaxModeChannel  SoftmaxMode = 1 // MIOPEN_SOFTMAX_MODE_CHANNEL
	SoftmaxModeInstance SoftmaxMode = 0 // MIOPEN_SOFTMAX_MODE_INSTANCE
)

// miopenTensorOpAdd is the add operation for miopenOpTensor.
const miopenTensorOpAdd = 0

// float64Bits reinterprets a float64 as a uintptr for passing to ccall.
func float64Bits(f float64) uintptr {
	return uintptr(math.Float64bits(f))
}

// --- Handle ---

// Handle wraps a miopenHandle_t (opaque pointer).
type Handle struct {
	handle uintptr
}

// CreateHandle creates a new MIOpen handle.
func CreateHandle() (*Handle, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("miopenCreate: miopen not available")
	}
	var h uintptr
	ret := cuda.Ccall(l.miopenCreate, uintptr(unsafe.Pointer(&h)))
	if err := statusError(ret, "miopenCreate"); err != nil {
		return nil, err
	}
	return &Handle{handle: h}, nil
}

// SetStream associates a HIP stream with this MIOpen handle.
func (h *Handle) SetStream(streamPtr unsafe.Pointer) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenSetStream: miopen not available")
	}
	ret := cuda.Ccall(l.miopenSetStream, h.handle, uintptr(streamPtr))
	return statusError(ret, "miopenSetStream")
}

// Destroy releases the MIOpen handle.
func (h *Handle) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenDestroy: miopen not available")
	}
	ret := cuda.Ccall(l.miopenDestroy, h.handle)
	return statusError(ret, "miopenDestroy")
}

// --- Tensor Descriptor ---

// TensorDescriptor wraps miopenTensorDescriptor_t (opaque pointer).
type TensorDescriptor struct {
	desc uintptr
}

// CreateTensorDescriptor creates a new tensor descriptor.
func CreateTensorDescriptor() (*TensorDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("miopenCreateTensorDescriptor: miopen not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.miopenCreateTensorDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "miopenCreateTensorDescriptor"); err != nil {
		return nil, err
	}
	return &TensorDescriptor{desc: d}, nil
}

// Set4d configures a 4D NCHW tensor descriptor.
func (t *TensorDescriptor) Set4d(dt DataType, n, c, h, w int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenSet4dTensorDescriptor: miopen not available")
	}
	ret := cuda.Ccall(l.miopenSet4dTensorDescriptor, t.desc,
		uintptr(dt), uintptr(n), uintptr(c), uintptr(h), uintptr(w))
	return statusError(ret, "miopenSet4dTensorDescriptor")
}

// Destroy releases the tensor descriptor.
func (t *TensorDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenDestroyTensorDescriptor: miopen not available")
	}
	ret := cuda.Ccall(l.miopenDestroyTensorDescriptor, t.desc)
	return statusError(ret, "miopenDestroyTensorDescriptor")
}

// --- Convolution Descriptor ---

// ConvolutionDescriptor wraps miopenConvolutionDescriptor_t.
type ConvolutionDescriptor struct {
	desc uintptr
}

// CreateConvolutionDescriptor creates a new convolution descriptor.
func CreateConvolutionDescriptor() (*ConvolutionDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("miopenCreateConvolutionDescriptor: miopen not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.miopenCreateConvolutionDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "miopenCreateConvolutionDescriptor"); err != nil {
		return nil, err
	}
	return &ConvolutionDescriptor{desc: d}, nil
}

// Set2d configures a 2D convolution descriptor.
func (c *ConvolutionDescriptor) Set2d(padH, padW, strH, strW, dilH, dilW int, mode ConvMode) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenInitConvolutionDescriptor: miopen not available")
	}
	ret := cuda.Ccall(l.miopenInitConvolutionDescriptor, c.desc, uintptr(mode),
		uintptr(padH), uintptr(padW), uintptr(strH), uintptr(strW), uintptr(dilH), uintptr(dilW))
	return statusError(ret, "miopenInitConvolutionDescriptor")
}

// SetGroupCount sets the group count for grouped convolutions.
func (c *ConvolutionDescriptor) SetGroupCount(groups int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenSetConvolutionGroupCount: miopen not available")
	}
	ret := cuda.Ccall(l.miopenSetConvolutionGroupCount, c.desc, uintptr(groups))
	return statusError(ret, "miopenSetConvolutionGroupCount")
}

// Destroy releases the convolution descriptor.
func (c *ConvolutionDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenDestroyConvolutionDescriptor: miopen not available")
	}
	ret := cuda.Ccall(l.miopenDestroyConvolutionDescriptor, c.desc)
	return statusError(ret, "miopenDestroyConvolutionDescriptor")
}

// --- Activation Descriptor ---

// ActivationDescriptor wraps miopenActivationDescriptor_t.
type ActivationDescriptor struct {
	desc uintptr
}

// CreateActivationDescriptor creates a new activation descriptor.
func CreateActivationDescriptor() (*ActivationDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("miopenCreateActivationDescriptor: miopen not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.miopenCreateActivationDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "miopenCreateActivationDescriptor"); err != nil {
		return nil, err
	}
	return &ActivationDescriptor{desc: d}, nil
}

// Set configures the activation descriptor.
func (a *ActivationDescriptor) Set(mode ActivationMode, alpha, beta, gamma float64) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenSetActivationDescriptor: miopen not available")
	}
	ret := cuda.Ccall(l.miopenSetActivationDescriptor, a.desc, uintptr(mode),
		float64Bits(alpha), float64Bits(beta), float64Bits(gamma))
	return statusError(ret, "miopenSetActivationDescriptor")
}

// Destroy releases the activation descriptor.
func (a *ActivationDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenDestroyActivationDescriptor: miopen not available")
	}
	ret := cuda.Ccall(l.miopenDestroyActivationDescriptor, a.desc)
	return statusError(ret, "miopenDestroyActivationDescriptor")
}

// --- Pooling Descriptor ---

// PoolingDescriptor wraps miopenPoolingDescriptor_t.
type PoolingDescriptor struct {
	desc uintptr
}

// CreatePoolingDescriptor creates a new pooling descriptor.
func CreatePoolingDescriptor() (*PoolingDescriptor, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("miopenCreatePoolingDescriptor: miopen not available")
	}
	var d uintptr
	ret := cuda.Ccall(l.miopenCreatePoolingDescriptor, uintptr(unsafe.Pointer(&d)))
	if err := statusError(ret, "miopenCreatePoolingDescriptor"); err != nil {
		return nil, err
	}
	return &PoolingDescriptor{desc: d}, nil
}

// Set2d configures a 2D pooling descriptor.
func (p *PoolingDescriptor) Set2d(mode PoolingMode, windowH, windowW, padH, padW, strideH, strideW int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenSet2dPoolingDescriptor: miopen not available")
	}
	ret := cuda.Ccall(l.miopenSet2dPoolingDescriptor, p.desc, uintptr(mode),
		uintptr(windowH), uintptr(windowW), uintptr(padH), uintptr(padW), uintptr(strideH), uintptr(strideW))
	return statusError(ret, "miopenSet2dPoolingDescriptor")
}

// Destroy releases the pooling descriptor.
func (p *PoolingDescriptor) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenDestroyPoolingDescriptor: miopen not available")
	}
	ret := cuda.Ccall(l.miopenDestroyPoolingDescriptor, p.desc)
	return statusError(ret, "miopenDestroyPoolingDescriptor")
}

// --- Forward Operations ---

// ConvolutionForwardGetWorkspaceSize returns the workspace size in bytes.
func (h *Handle) ConvolutionForwardGetWorkspaceSize(
	xDesc *TensorDescriptor,
	wDesc *TensorDescriptor,
	convDesc *ConvolutionDescriptor,
	yDesc *TensorDescriptor,
) (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("miopenConvolutionForwardGetWorkSpaceSize: miopen not available")
	}
	var size uintptr
	ret := cuda.Ccall(l.miopenConvolutionForwardGetWorkSpaceSize, h.handle,
		wDesc.desc, xDesc.desc, convDesc.desc, yDesc.desc, uintptr(unsafe.Pointer(&size)))
	return int(size), statusError(ret, "miopenConvolutionForwardGetWorkSpaceSize")
}

// FindConvolutionForwardAlgorithm finds the best convolution algorithm.
// Returns the algorithm enum value suitable for ConvolutionForward.
func (h *Handle) FindConvolutionForwardAlgorithm(
	xDesc *TensorDescriptor, x unsafe.Pointer,
	wDesc *TensorDescriptor, w unsafe.Pointer,
	convDesc *ConvolutionDescriptor,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	workspace unsafe.Pointer, wsSize int,
) (uintptr, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("miopenFindConvolutionForwardAlgorithm: miopen not available")
	}
	// miopenConvAlgoPerf_t is a struct; we allocate enough space for one result.
	// The struct layout: fwd_algo (int32), time (float), memory (size_t).
	// We allocate a generous buffer.
	var resultBuf [64]byte
	var returnedCount int32
	ret := cuda.Ccall(l.miopenFindConvolutionForwardAlgorithm, h.handle,
		xDesc.desc, uintptr(x), wDesc.desc, uintptr(w),
		convDesc.desc, yDesc.desc, uintptr(y),
		uintptr(1), uintptr(unsafe.Pointer(&returnedCount)),
		uintptr(unsafe.Pointer(&resultBuf[0])),
		uintptr(workspace), uintptr(wsSize), uintptr(0))
	if err := statusError(ret, "miopenFindConvolutionForwardAlgorithm"); err != nil {
		return 0, err
	}
	// Extract fwd_algo from the first field of the result struct.
	algo := *(*int32)(unsafe.Pointer(&resultBuf[0]))
	return uintptr(algo), nil
}

// ConvolutionForward performs the forward convolution.
func (h *Handle) ConvolutionForward(
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	wDesc *TensorDescriptor, w unsafe.Pointer,
	convDesc *ConvolutionDescriptor,
	algo uintptr,
	workspace unsafe.Pointer, wsSize int,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenConvolutionForward: miopen not available")
	}
	ret := cuda.Ccall(l.miopenConvolutionForward, h.handle,
		uintptr(unsafe.Pointer(&alpha)),
		xDesc.desc, uintptr(x), wDesc.desc, uintptr(w), convDesc.desc,
		algo,
		uintptr(unsafe.Pointer(&beta)),
		yDesc.desc, uintptr(y),
		uintptr(workspace), uintptr(wsSize))
	return statusError(ret, "miopenConvolutionForward")
}

// BatchNormalizationForwardInference performs batch normalization in inference mode.
func (h *Handle) BatchNormalizationForwardInference(
	mode BatchNormMode,
	alpha, beta float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	bnScaleBiasMeanVarDesc *TensorDescriptor,
	scale, bias, mean, variance unsafe.Pointer,
	epsilon float64,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenBatchNormalizationForwardInference: miopen not available")
	}
	ret := cuda.Ccall(l.miopenBatchNormalizationForwardInference, h.handle,
		uintptr(mode),
		uintptr(unsafe.Pointer(&alpha)), uintptr(unsafe.Pointer(&beta)),
		xDesc.desc, uintptr(x), yDesc.desc, uintptr(y),
		bnScaleBiasMeanVarDesc.desc,
		uintptr(scale), uintptr(bias), uintptr(mean), uintptr(variance),
		float64Bits(epsilon))
	return statusError(ret, "miopenBatchNormalizationForwardInference")
}

// ActivationForward applies an activation function.
func (h *Handle) ActivationForward(
	actDesc *ActivationDescriptor,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenActivationForward: miopen not available")
	}
	ret := cuda.Ccall(l.miopenActivationForward, h.handle, actDesc.desc,
		uintptr(unsafe.Pointer(&alpha)), xDesc.desc, uintptr(x),
		uintptr(unsafe.Pointer(&beta)), yDesc.desc, uintptr(y))
	return statusError(ret, "miopenActivationForward")
}

// PoolingForward performs 2D pooling.
func (h *Handle) PoolingForward(
	poolDesc *PoolingDescriptor,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
	workspaceIndex bool,
	workspace unsafe.Pointer, wsSize int,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenPoolingForward: miopen not available")
	}
	var wsIdx uintptr
	if workspaceIndex {
		wsIdx = 1
	}
	ret := cuda.Ccall(l.miopenPoolingForward, h.handle, poolDesc.desc,
		uintptr(unsafe.Pointer(&alpha)), xDesc.desc, uintptr(x),
		uintptr(unsafe.Pointer(&beta)), yDesc.desc, uintptr(y),
		wsIdx, uintptr(workspace), uintptr(wsSize))
	return statusError(ret, "miopenPoolingForward")
}

// SoftmaxForward computes softmax.
func (h *Handle) SoftmaxForward(
	algo SoftmaxAlgorithm, mode SoftmaxMode,
	alpha float32,
	xDesc *TensorDescriptor, x unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenSoftmaxForward_V2: miopen not available")
	}
	ret := cuda.Ccall(l.miopenSoftmaxForwardV2, h.handle,
		uintptr(unsafe.Pointer(&alpha)), xDesc.desc, uintptr(x),
		uintptr(unsafe.Pointer(&beta)), yDesc.desc, uintptr(y),
		uintptr(algo), uintptr(mode))
	return statusError(ret, "miopenSoftmaxForward_V2")
}

// OpTensorAdd adds tensors: y = alpha * b + beta * y.
func (h *Handle) OpTensorAdd(
	alpha float32,
	bDesc *TensorDescriptor, b unsafe.Pointer,
	beta float32,
	yDesc *TensorDescriptor, y unsafe.Pointer,
) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("miopenOpTensor: miopen not available")
	}
	zero := float32(0)
	ret := cuda.Ccall(l.miopenOpTensor, h.handle, uintptr(miopenTensorOpAdd),
		uintptr(unsafe.Pointer(&alpha)), bDesc.desc, uintptr(b),
		uintptr(unsafe.Pointer(&zero)), bDesc.desc, uintptr(b),
		uintptr(unsafe.Pointer(&beta)), yDesc.desc, uintptr(y))
	return statusError(ret, "miopenOpTensor(add)")
}

// GetPoolingForwardOutputDim returns the output dimensions for a pooling operation.
func (h *Handle) GetPoolingForwardOutputDim(
	poolDesc *PoolingDescriptor,
	xDesc *TensorDescriptor,
) (n, c, outH, outW int, err error) {
	l := lib()
	if l == nil {
		return 0, 0, 0, 0, fmt.Errorf("miopenGetPoolingForwardOutputDim: miopen not available")
	}
	var cn, cc, ch, cw int32
	ret := cuda.Ccall(l.miopenGetPoolingForwardOutputDim, poolDesc.desc, xDesc.desc,
		uintptr(unsafe.Pointer(&cn)), uintptr(unsafe.Pointer(&cc)),
		uintptr(unsafe.Pointer(&ch)), uintptr(unsafe.Pointer(&cw)))
	err = statusError(ret, "miopenGetPoolingForwardOutputDim")
	return int(cn), int(cc), int(ch), int(cw), err
}

// PoolingGetWorkSpaceSize returns the workspace size for pooling.
func (h *Handle) PoolingGetWorkSpaceSize(yDesc *TensorDescriptor) (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("miopenPoolingGetWorkSpaceSize: miopen not available")
	}
	var size uintptr
	ret := cuda.Ccall(l.miopenPoolingGetWorkSpaceSize, yDesc.desc, uintptr(unsafe.Pointer(&size)))
	return int(size), statusError(ret, "miopenPoolingGetWorkSpaceSize")
}
