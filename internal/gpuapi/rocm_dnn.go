package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/hip"
	"github.com/zerfoo/zerfoo/internal/miopen"
)

// ROCmDNN implements the DNN interface using MIOpen.
type ROCmDNN struct {
	handle *miopen.Handle
}

// NewROCmDNN creates a new MIOpen adapter.
func NewROCmDNN() (*ROCmDNN, error) {
	h, err := miopen.CreateHandle()
	if err != nil {
		return nil, err
	}
	return &ROCmDNN{handle: h}, nil
}

// NewROCmDNNFromHandle wraps an existing MIOpen handle.
func NewROCmDNNFromHandle(h *miopen.Handle) *ROCmDNN {
	return &ROCmDNN{handle: h}
}

func (d *ROCmDNN) SetStream(stream Stream) error {
	var ptr unsafe.Pointer
	if stream != nil {
		ptr = stream.Ptr()
	}
	return d.handle.SetStream(ptr)
}

func (d *ROCmDNN) Destroy() error {
	return d.handle.Destroy()
}

// Handle returns the underlying MIOpen handle.
func (d *ROCmDNN) Handle() *miopen.Handle {
	return d.handle
}

func (d *ROCmDNN) ConvForward(
	x unsafe.Pointer, xShape [4]int,
	w unsafe.Pointer, wShape [4]int,
	bias unsafe.Pointer,
	y unsafe.Pointer, yShape [4]int,
	pads [2]int, strides [2]int, dilations [2]int,
	groups int,
	stream Stream,
) error {
	xDesc, err := miopenTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("ConvForward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	wDesc, err := miopenTensor4d(wShape)
	if err != nil {
		return fmt.Errorf("ConvForward: wDesc: %w", err)
	}
	defer func() { _ = wDesc.Destroy() }()

	convDesc, err := miopen.CreateConvolutionDescriptor()
	if err != nil {
		return fmt.Errorf("ConvForward: convDesc: %w", err)
	}
	defer func() { _ = convDesc.Destroy() }()
	if err := convDesc.Set2d(pads[0], pads[1], strides[0], strides[1], dilations[0], dilations[1], miopen.ConvolutionMode); err != nil {
		return fmt.Errorf("ConvForward: set convDesc: %w", err)
	}
	if groups > 1 {
		if err := convDesc.SetGroupCount(groups); err != nil {
			return fmt.Errorf("ConvForward: set groups: %w", err)
		}
	}

	yDesc, err := miopenTensor4d(yShape)
	if err != nil {
		return fmt.Errorf("ConvForward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	// MIOpen requires workspace for algorithm search and execution.
	wsSize, err := d.handle.ConvolutionForwardGetWorkspaceSize(xDesc, wDesc, convDesc, yDesc)
	if err != nil {
		return fmt.Errorf("ConvForward: workspace size: %w", err)
	}

	var wsPtr unsafe.Pointer
	if wsSize > 0 {
		wsPtr, err = hipMallocTemp(wsSize)
		if err != nil {
			return fmt.Errorf("ConvForward: workspace alloc: %w", err)
		}
		defer hipFreeTemp(wsPtr)
	}

	// Find the best algorithm.
	algo, err := d.handle.FindConvolutionForwardAlgorithm(
		xDesc, x, wDesc, w, convDesc, yDesc, y, wsPtr, wsSize,
	)
	if err != nil {
		return fmt.Errorf("ConvForward: find algo: %w", err)
	}

	if err := d.handle.ConvolutionForward(
		1.0, xDesc, x, wDesc, w, convDesc, algo, wsPtr, wsSize, 0.0, yDesc, y,
	); err != nil {
		return fmt.Errorf("ConvForward: %w", err)
	}

	// Add bias if provided.
	if bias != nil {
		bDesc, err := miopenTensor4d([4]int{1, yShape[1], 1, 1})
		if err != nil {
			return fmt.Errorf("ConvForward: bDesc: %w", err)
		}
		defer func() { _ = bDesc.Destroy() }()
		if err := d.handle.OpTensorAdd(1.0, bDesc, bias, 1.0, yDesc, y); err != nil {
			return fmt.Errorf("ConvForward: add bias: %w", err)
		}
	}

	return nil
}

func (d *ROCmDNN) ConvBackwardData(
	w unsafe.Pointer, wShape [4]int,
	dy unsafe.Pointer, dyShape [4]int,
	dx unsafe.Pointer, dxShape [4]int,
	pads [2]int, strides [2]int, dilations [2]int,
	groups int,
	stream Stream,
) error {
	return fmt.Errorf("ConvBackwardData: not yet implemented")
}

func (d *ROCmDNN) ConvBackwardFilter(
	x unsafe.Pointer, xShape [4]int,
	dy unsafe.Pointer, dyShape [4]int,
	dw unsafe.Pointer, dwShape [4]int,
	pads [2]int, strides [2]int, dilations [2]int,
	groups int,
	stream Stream,
) error {
	return fmt.Errorf("ConvBackwardFilter: not yet implemented")
}

//nolint:dupl // ROCm mirrors CUDA API shape intentionally
func (d *ROCmDNN) BatchNormForwardInference(
	x unsafe.Pointer, xShape [4]int,
	scale, bias, mean, variance unsafe.Pointer,
	channels int,
	epsilon float64,
	y unsafe.Pointer,
	stream Stream,
) error {
	xDesc, err := miopenTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormForwardInference: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := miopenTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormForwardInference: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	bnDesc, err := miopenTensor4d([4]int{1, channels, 1, 1})
	if err != nil {
		return fmt.Errorf("BatchNormForwardInference: bnDesc: %w", err)
	}
	defer func() { _ = bnDesc.Destroy() }()

	return d.handle.BatchNormalizationForwardInference(
		miopen.BatchNormSpatial,
		1.0, 0.0,
		xDesc, x,
		yDesc, y,
		bnDesc,
		scale, bias,
		mean, variance,
		epsilon,
	)
}

func (d *ROCmDNN) BatchNormForwardTraining(
	x unsafe.Pointer, xShape [4]int,
	scale, bias unsafe.Pointer,
	channels int,
	epsilon, expAvgFactor float64,
	runningMean, runningVariance unsafe.Pointer,
	saveMean, saveInvVariance unsafe.Pointer,
	y unsafe.Pointer,
	stream Stream,
) error {
	return fmt.Errorf("BatchNormForwardTraining: not yet implemented")
}

func (d *ROCmDNN) BatchNormBackward(
	x unsafe.Pointer, xShape [4]int,
	dy unsafe.Pointer,
	scale unsafe.Pointer,
	channels int,
	saveMean, saveInvVariance unsafe.Pointer,
	dx, dScale, dBias unsafe.Pointer,
	stream Stream,
) error {
	return fmt.Errorf("BatchNormBackward: not yet implemented")
}

func (d *ROCmDNN) ActivationForward(
	mode ActivationMode,
	x unsafe.Pointer, shape [4]int,
	y unsafe.Pointer,
	stream Stream,
) error {
	xDesc, err := miopenTensor4d(shape)
	if err != nil {
		return fmt.Errorf("ActivationForward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := miopenTensor4d(shape)
	if err != nil {
		return fmt.Errorf("ActivationForward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	actDesc, err := miopen.CreateActivationDescriptor()
	if err != nil {
		return fmt.Errorf("ActivationForward: actDesc: %w", err)
	}
	defer func() { _ = actDesc.Destroy() }()
	if err := actDesc.Set(miopenActivationMode(mode), 0.0, 0.0, 0.0); err != nil {
		return fmt.Errorf("ActivationForward: set actDesc: %w", err)
	}

	return d.handle.ActivationForward(actDesc, 1.0, xDesc, x, 0.0, yDesc, y)
}

func (d *ROCmDNN) ActivationBackward(
	mode ActivationMode,
	y unsafe.Pointer, dy unsafe.Pointer,
	x unsafe.Pointer, dx unsafe.Pointer,
	shape [4]int,
	stream Stream,
) error {
	return fmt.Errorf("ActivationBackward: not yet implemented")
}

func (d *ROCmDNN) PoolingForward(
	mode PoolingMode,
	x unsafe.Pointer, xShape [4]int,
	y unsafe.Pointer, yShape [4]int,
	windowH, windowW, padH, padW, strideH, strideW int,
	stream Stream,
) error {
	xDesc, err := miopenTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("PoolingForward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := miopenTensor4d(yShape)
	if err != nil {
		return fmt.Errorf("PoolingForward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	poolDesc, err := miopen.CreatePoolingDescriptor()
	if err != nil {
		return fmt.Errorf("PoolingForward: poolDesc: %w", err)
	}
	defer func() { _ = poolDesc.Destroy() }()
	if err := poolDesc.Set2d(miopenPoolingMode(mode), windowH, windowW, padH, padW, strideH, strideW); err != nil {
		return fmt.Errorf("PoolingForward: set poolDesc: %w", err)
	}

	// MIOpen pooling requires workspace for index tracking.
	wsSize, err := d.handle.PoolingGetWorkSpaceSize(yDesc)
	if err != nil {
		return fmt.Errorf("PoolingForward: workspace size: %w", err)
	}

	var wsPtr unsafe.Pointer
	if wsSize > 0 {
		wsPtr, err = hipMallocTemp(wsSize)
		if err != nil {
			return fmt.Errorf("PoolingForward: workspace alloc: %w", err)
		}
		defer hipFreeTemp(wsPtr)
	}

	return d.handle.PoolingForward(poolDesc, 1.0, xDesc, x, 0.0, yDesc, y, false, wsPtr, wsSize)
}

func (d *ROCmDNN) PoolingBackward(
	mode PoolingMode,
	y unsafe.Pointer, dy unsafe.Pointer, yShape [4]int,
	x unsafe.Pointer, dx unsafe.Pointer, xShape [4]int,
	windowH, windowW, padH, padW, strideH, strideW int,
	stream Stream,
) error {
	return fmt.Errorf("PoolingBackward: not yet implemented")
}

func (d *ROCmDNN) SoftmaxForward(
	x unsafe.Pointer, shape [4]int,
	y unsafe.Pointer,
	stream Stream,
) error {
	xDesc, err := miopenTensor4d(shape)
	if err != nil {
		return fmt.Errorf("SoftmaxForward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := miopenTensor4d(shape)
	if err != nil {
		return fmt.Errorf("SoftmaxForward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	return d.handle.SoftmaxForward(miopen.SoftmaxAccurate, miopen.SoftmaxModeChannel, 1.0, xDesc, x, 0.0, yDesc, y)
}

func (d *ROCmDNN) AddTensor(
	alpha float32,
	b unsafe.Pointer, bShape [4]int,
	beta float32,
	y unsafe.Pointer, yShape [4]int,
	stream Stream,
) error {
	bDesc, err := miopenTensor4d(bShape)
	if err != nil {
		return fmt.Errorf("AddTensor: bDesc: %w", err)
	}
	defer func() { _ = bDesc.Destroy() }()

	yDesc, err := miopenTensor4d(yShape)
	if err != nil {
		return fmt.Errorf("AddTensor: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	return d.handle.OpTensorAdd(alpha, bDesc, b, beta, yDesc, y)
}

// --- helpers ---

// miopenTensor4d creates and configures an NCHW float32 tensor descriptor.
func miopenTensor4d(shape [4]int) (*miopen.TensorDescriptor, error) {
	desc, err := miopen.CreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	if err := desc.Set4d(miopen.Float32, shape[0], shape[1], shape[2], shape[3]); err != nil {
		_ = desc.Destroy()
		return nil, err
	}
	return desc, nil
}

// miopenActivationMode converts gpuapi.ActivationMode to miopen.ActivationMode.
func miopenActivationMode(mode ActivationMode) miopen.ActivationMode {
	switch mode {
	case ActivationSigmoid:
		return miopen.ActivationSigmoid
	case ActivationReLU:
		return miopen.ActivationReLU
	case ActivationTanh:
		return miopen.ActivationTanh
	case ActivationELU:
		return miopen.ActivationELU
	default:
		return miopen.ActivationReLU
	}
}

// miopenPoolingMode converts gpuapi.PoolingMode to miopen.PoolingMode.
func miopenPoolingMode(mode PoolingMode) miopen.PoolingMode {
	switch mode {
	case PoolingMax:
		return miopen.PoolingMax
	case PoolingAverageCountIncludePad:
		return miopen.PoolingAverageCountIncludePad
	case PoolingAverageCountExcludePad:
		return miopen.PoolingAverageCountExcludePad
	default:
		return miopen.PoolingMax
	}
}

// hipMallocTemp allocates temporary device memory for workspace buffers.
func hipMallocTemp(size int) (unsafe.Pointer, error) {
	return hip.Malloc(size)
}

// hipFreeTemp frees temporary device memory.
func hipFreeTemp(ptr unsafe.Pointer) {
	hip.Free(ptr) //nolint:errcheck // workspace cleanup is best-effort
}

// Compile-time interface assertion.
var _ DNN = (*ROCmDNN)(nil)
