package gpuapi

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/cudnn"
)

// CUDADNN implements the DNN interface using cuDNN.
type CUDADNN struct {
	handle *cudnn.Handle
}

// NewCUDADNN creates a new cuDNN adapter.
func NewCUDADNN() (*CUDADNN, error) {
	h, err := cudnn.CreateHandle()
	if err != nil {
		return nil, err
	}
	return &CUDADNN{handle: h}, nil
}

// NewCUDADNNFromHandle wraps an existing cuDNN handle.
func NewCUDADNNFromHandle(h *cudnn.Handle) *CUDADNN {
	return &CUDADNN{handle: h}
}

func (d *CUDADNN) SetStream(stream Stream) error {
	var ptr unsafe.Pointer
	if stream != nil {
		ptr = stream.Ptr()
	}
	return d.handle.SetStream(ptr)
}

func (d *CUDADNN) Destroy() error {
	return d.handle.Destroy()
}

// Handle returns the underlying cuDNN handle for backward compatibility.
func (d *CUDADNN) Handle() *cudnn.Handle {
	return d.handle
}

func (d *CUDADNN) ConvForward(
	x unsafe.Pointer, xShape [4]int,
	w unsafe.Pointer, wShape [4]int,
	bias unsafe.Pointer,
	y unsafe.Pointer, yShape [4]int,
	pads [2]int, strides [2]int, dilations [2]int,
	groups int,
	stream Stream,
) error {
	xDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("ConvForward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	wDesc, err := cudnn.CreateFilterDescriptor()
	if err != nil {
		return fmt.Errorf("ConvForward: wDesc: %w", err)
	}
	defer func() { _ = wDesc.Destroy() }()
	if err := wDesc.Set4d(cudnn.Float32, cudnn.NCHW, wShape[0], wShape[1], wShape[2], wShape[3]); err != nil {
		return fmt.Errorf("ConvForward: set wDesc: %w", err)
	}

	convDesc, err := cudnn.CreateConvolutionDescriptor()
	if err != nil {
		return fmt.Errorf("ConvForward: convDesc: %w", err)
	}
	defer func() { _ = convDesc.Destroy() }()
	if err := convDesc.Set2d(pads[0], pads[1], strides[0], strides[1], dilations[0], dilations[1], cudnn.CrossCorrelation, cudnn.Float32); err != nil {
		return fmt.Errorf("ConvForward: set convDesc: %w", err)
	}
	if groups > 1 {
		if err := convDesc.SetGroupCount(groups); err != nil {
			return fmt.Errorf("ConvForward: set groups: %w", err)
		}
	}

	yDesc, err := makeTensor4d(yShape)
	if err != nil {
		return fmt.Errorf("ConvForward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	algo := cudnn.ConvFwdAlgoImplicitGemm

	wsSize, err := d.handle.GetConvolutionForwardWorkspaceSize(xDesc, wDesc, convDesc, yDesc, algo)
	if err != nil {
		return fmt.Errorf("ConvForward: workspace size: %w", err)
	}

	var wsPtr unsafe.Pointer
	if wsSize > 0 {
		var allocErr error
		wsPtr, allocErr = cudaMallocTemp(wsSize)
		if allocErr != nil {
			return fmt.Errorf("ConvForward: workspace alloc: %w", allocErr)
		}
		defer cudaFreeTemp(wsPtr)
	}

	if err := d.handle.ConvolutionForward(1.0, xDesc, x, wDesc, w, convDesc, algo, wsPtr, wsSize, 0.0, yDesc, y); err != nil {
		return fmt.Errorf("ConvForward: %w", err)
	}

	if bias != nil {
		bDesc, err := makeTensor4d([4]int{1, yShape[1], 1, 1})
		if err != nil {
			return fmt.Errorf("ConvForward: bDesc: %w", err)
		}
		defer func() { _ = bDesc.Destroy() }()
		if err := d.handle.AddTensor(1.0, bDesc, bias, 1.0, yDesc, y); err != nil {
			return fmt.Errorf("ConvForward: add bias: %w", err)
		}
	}

	return nil
}

func (d *CUDADNN) ConvBackwardData(
	w unsafe.Pointer, wShape [4]int,
	dy unsafe.Pointer, dyShape [4]int,
	dx unsafe.Pointer, dxShape [4]int,
	pads [2]int, strides [2]int, dilations [2]int,
	groups int,
	stream Stream,
) error {
	wDesc, err := cudnn.CreateFilterDescriptor()
	if err != nil {
		return fmt.Errorf("ConvBackwardData: wDesc: %w", err)
	}
	defer func() { _ = wDesc.Destroy() }()
	if err := wDesc.Set4d(cudnn.Float32, cudnn.NCHW, wShape[0], wShape[1], wShape[2], wShape[3]); err != nil {
		return fmt.Errorf("ConvBackwardData: set wDesc: %w", err)
	}

	dyDesc, err := makeTensor4d(dyShape)
	if err != nil {
		return fmt.Errorf("ConvBackwardData: dyDesc: %w", err)
	}
	defer func() { _ = dyDesc.Destroy() }()

	dxDesc, err := makeTensor4d(dxShape)
	if err != nil {
		return fmt.Errorf("ConvBackwardData: dxDesc: %w", err)
	}
	defer func() { _ = dxDesc.Destroy() }()

	convDesc, err := cudnn.CreateConvolutionDescriptor()
	if err != nil {
		return fmt.Errorf("ConvBackwardData: convDesc: %w", err)
	}
	defer func() { _ = convDesc.Destroy() }()
	if err := convDesc.Set2d(pads[0], pads[1], strides[0], strides[1], dilations[0], dilations[1], cudnn.CrossCorrelation, cudnn.Float32); err != nil {
		return fmt.Errorf("ConvBackwardData: set convDesc: %w", err)
	}
	if groups > 1 {
		if err := convDesc.SetGroupCount(groups); err != nil {
			return fmt.Errorf("ConvBackwardData: set groups: %w", err)
		}
	}

	algo := cudnn.ConvBwdDataAlgo1

	wsSize, err := d.handle.GetConvolutionBackwardDataWorkspaceSize(wDesc, dyDesc, convDesc, dxDesc, algo)
	if err != nil {
		return fmt.Errorf("ConvBackwardData: workspace size: %w", err)
	}

	var wsPtr unsafe.Pointer
	if wsSize > 0 {
		var allocErr error
		wsPtr, allocErr = cudaMallocTemp(wsSize)
		if allocErr != nil {
			return fmt.Errorf("ConvBackwardData: workspace alloc: %w", allocErr)
		}
		defer cudaFreeTemp(wsPtr)
	}

	if err := d.handle.ConvolutionBackwardData(1.0, wDesc, w, dyDesc, dy, convDesc, algo, wsPtr, wsSize, 0.0, dxDesc, dx); err != nil {
		return fmt.Errorf("ConvBackwardData: %w", err)
	}
	return nil
}

func (d *CUDADNN) ConvBackwardFilter(
	x unsafe.Pointer, xShape [4]int,
	dy unsafe.Pointer, dyShape [4]int,
	dw unsafe.Pointer, dwShape [4]int,
	pads [2]int, strides [2]int, dilations [2]int,
	groups int,
	stream Stream,
) error {
	xDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("ConvBackwardFilter: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	dyDesc, err := makeTensor4d(dyShape)
	if err != nil {
		return fmt.Errorf("ConvBackwardFilter: dyDesc: %w", err)
	}
	defer func() { _ = dyDesc.Destroy() }()

	dwDesc, err := cudnn.CreateFilterDescriptor()
	if err != nil {
		return fmt.Errorf("ConvBackwardFilter: dwDesc: %w", err)
	}
	defer func() { _ = dwDesc.Destroy() }()
	if err := dwDesc.Set4d(cudnn.Float32, cudnn.NCHW, dwShape[0], dwShape[1], dwShape[2], dwShape[3]); err != nil {
		return fmt.Errorf("ConvBackwardFilter: set dwDesc: %w", err)
	}

	convDesc, err := cudnn.CreateConvolutionDescriptor()
	if err != nil {
		return fmt.Errorf("ConvBackwardFilter: convDesc: %w", err)
	}
	defer func() { _ = convDesc.Destroy() }()
	if err := convDesc.Set2d(pads[0], pads[1], strides[0], strides[1], dilations[0], dilations[1], cudnn.CrossCorrelation, cudnn.Float32); err != nil {
		return fmt.Errorf("ConvBackwardFilter: set convDesc: %w", err)
	}
	if groups > 1 {
		if err := convDesc.SetGroupCount(groups); err != nil {
			return fmt.Errorf("ConvBackwardFilter: set groups: %w", err)
		}
	}

	algo := cudnn.ConvBwdFilterAlgo1

	wsSize, err := d.handle.GetConvolutionBackwardFilterWorkspaceSize(xDesc, dyDesc, convDesc, dwDesc, algo)
	if err != nil {
		return fmt.Errorf("ConvBackwardFilter: workspace size: %w", err)
	}

	var wsPtr unsafe.Pointer
	if wsSize > 0 {
		var allocErr error
		wsPtr, allocErr = cudaMallocTemp(wsSize)
		if allocErr != nil {
			return fmt.Errorf("ConvBackwardFilter: workspace alloc: %w", allocErr)
		}
		defer cudaFreeTemp(wsPtr)
	}

	if err := d.handle.ConvolutionBackwardFilter(1.0, xDesc, x, dyDesc, dy, convDesc, algo, wsPtr, wsSize, 0.0, dwDesc, dw); err != nil {
		return fmt.Errorf("ConvBackwardFilter: %w", err)
	}
	return nil
}

//nolint:dupl // CUDA and ROCm DNN are structurally identical by design
func (d *CUDADNN) BatchNormForwardInference(
	x unsafe.Pointer, xShape [4]int,
	scale, bias, mean, variance unsafe.Pointer,
	channels int,
	epsilon float64,
	y unsafe.Pointer,
	stream Stream,
) error {
	xDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormForwardInference: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormForwardInference: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	bnDesc, err := makeTensor4d([4]int{1, channels, 1, 1})
	if err != nil {
		return fmt.Errorf("BatchNormForwardInference: bnDesc: %w", err)
	}
	defer func() { _ = bnDesc.Destroy() }()

	return d.handle.BatchNormalizationForwardInference(
		cudnn.BatchNormSpatial,
		1.0, 0.0,
		xDesc, x,
		yDesc, y,
		bnDesc,
		scale, bias,
		mean, variance,
		epsilon,
	)
}

func (d *CUDADNN) BatchNormForwardTraining(
	x unsafe.Pointer, xShape [4]int,
	scale, bias unsafe.Pointer,
	channels int,
	epsilon, expAvgFactor float64,
	runningMean, runningVariance unsafe.Pointer,
	saveMean, saveInvVariance unsafe.Pointer,
	y unsafe.Pointer,
	stream Stream,
) error {
	xDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormForwardTraining: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormForwardTraining: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	bnDesc, err := makeTensor4d([4]int{1, channels, 1, 1})
	if err != nil {
		return fmt.Errorf("BatchNormForwardTraining: bnDesc: %w", err)
	}
	defer func() { _ = bnDesc.Destroy() }()

	return d.handle.BatchNormalizationForwardTraining(
		cudnn.BatchNormSpatial,
		1.0, 0.0,
		xDesc, x,
		yDesc, y,
		bnDesc,
		scale, bias,
		expAvgFactor,
		runningMean, runningVariance,
		epsilon,
		saveMean, saveInvVariance,
	)
}

func (d *CUDADNN) BatchNormBackward(
	x unsafe.Pointer, xShape [4]int,
	dy unsafe.Pointer,
	scale unsafe.Pointer,
	channels int,
	saveMean, saveInvVariance unsafe.Pointer,
	dx, dScale, dBias unsafe.Pointer,
	stream Stream,
) error {
	xDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormBackward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	dyDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormBackward: dyDesc: %w", err)
	}
	defer func() { _ = dyDesc.Destroy() }()

	dxDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("BatchNormBackward: dxDesc: %w", err)
	}
	defer func() { _ = dxDesc.Destroy() }()

	bnDesc, err := makeTensor4d([4]int{1, channels, 1, 1})
	if err != nil {
		return fmt.Errorf("BatchNormBackward: bnDesc: %w", err)
	}
	defer func() { _ = bnDesc.Destroy() }()

	return d.handle.BatchNormalizationBackward(
		cudnn.BatchNormSpatial,
		1.0, 0.0,
		1.0, 0.0,
		xDesc, x,
		dyDesc, dy,
		dxDesc, dx,
		bnDesc,
		scale,
		dScale, dBias,
		1e-5,
		saveMean, saveInvVariance,
	)
}

func (d *CUDADNN) ActivationForward(
	mode ActivationMode,
	x unsafe.Pointer, shape [4]int,
	y unsafe.Pointer,
	stream Stream,
) error {
	xDesc, err := makeTensor4d(shape)
	if err != nil {
		return fmt.Errorf("ActivationForward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := makeTensor4d(shape)
	if err != nil {
		return fmt.Errorf("ActivationForward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	actDesc, err := cudnn.CreateActivationDescriptor()
	if err != nil {
		return fmt.Errorf("ActivationForward: actDesc: %w", err)
	}
	defer func() { _ = actDesc.Destroy() }()
	if err := actDesc.Set(cudnnActivationMode(mode), cudnn.NotPropagateNan, 0.0); err != nil {
		return fmt.Errorf("ActivationForward: set actDesc: %w", err)
	}

	return d.handle.ActivationForward(actDesc, 1.0, xDesc, x, 0.0, yDesc, y)
}

func (d *CUDADNN) ActivationBackward(
	mode ActivationMode,
	y unsafe.Pointer, dy unsafe.Pointer,
	x unsafe.Pointer, dx unsafe.Pointer,
	shape [4]int,
	stream Stream,
) error {
	yDesc, err := makeTensor4d(shape)
	if err != nil {
		return fmt.Errorf("ActivationBackward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	dyDesc, err := makeTensor4d(shape)
	if err != nil {
		return fmt.Errorf("ActivationBackward: dyDesc: %w", err)
	}
	defer func() { _ = dyDesc.Destroy() }()

	xDesc, err := makeTensor4d(shape)
	if err != nil {
		return fmt.Errorf("ActivationBackward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	dxDesc, err := makeTensor4d(shape)
	if err != nil {
		return fmt.Errorf("ActivationBackward: dxDesc: %w", err)
	}
	defer func() { _ = dxDesc.Destroy() }()

	actDesc, err := cudnn.CreateActivationDescriptor()
	if err != nil {
		return fmt.Errorf("ActivationBackward: actDesc: %w", err)
	}
	defer func() { _ = actDesc.Destroy() }()
	if err := actDesc.Set(cudnnActivationMode(mode), cudnn.NotPropagateNan, 0.0); err != nil {
		return fmt.Errorf("ActivationBackward: set actDesc: %w", err)
	}

	return d.handle.ActivationBackward(actDesc, 1.0, yDesc, y, dyDesc, dy, xDesc, x, 0.0, dxDesc, dx)
}

func (d *CUDADNN) PoolingForward(
	mode PoolingMode,
	x unsafe.Pointer, xShape [4]int,
	y unsafe.Pointer, yShape [4]int,
	windowH, windowW, padH, padW, strideH, strideW int,
	stream Stream,
) error {
	xDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("PoolingForward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := makeTensor4d(yShape)
	if err != nil {
		return fmt.Errorf("PoolingForward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	poolDesc, err := cudnn.CreatePoolingDescriptor()
	if err != nil {
		return fmt.Errorf("PoolingForward: poolDesc: %w", err)
	}
	defer func() { _ = poolDesc.Destroy() }()
	if err := poolDesc.Set2d(cudnnPoolingMode(mode), cudnn.NotPropagateNan, windowH, windowW, padH, padW, strideH, strideW); err != nil {
		return fmt.Errorf("PoolingForward: set poolDesc: %w", err)
	}

	return d.handle.PoolingForward(poolDesc, 1.0, xDesc, x, 0.0, yDesc, y)
}

func (d *CUDADNN) PoolingBackward(
	mode PoolingMode,
	y unsafe.Pointer, dy unsafe.Pointer, yShape [4]int,
	x unsafe.Pointer, dx unsafe.Pointer, xShape [4]int,
	windowH, windowW, padH, padW, strideH, strideW int,
	stream Stream,
) error {
	yDesc, err := makeTensor4d(yShape)
	if err != nil {
		return fmt.Errorf("PoolingBackward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	dyDesc, err := makeTensor4d(yShape)
	if err != nil {
		return fmt.Errorf("PoolingBackward: dyDesc: %w", err)
	}
	defer func() { _ = dyDesc.Destroy() }()

	xDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("PoolingBackward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	dxDesc, err := makeTensor4d(xShape)
	if err != nil {
		return fmt.Errorf("PoolingBackward: dxDesc: %w", err)
	}
	defer func() { _ = dxDesc.Destroy() }()

	poolDesc, err := cudnn.CreatePoolingDescriptor()
	if err != nil {
		return fmt.Errorf("PoolingBackward: poolDesc: %w", err)
	}
	defer func() { _ = poolDesc.Destroy() }()
	if err := poolDesc.Set2d(cudnnPoolingMode(mode), cudnn.NotPropagateNan, windowH, windowW, padH, padW, strideH, strideW); err != nil {
		return fmt.Errorf("PoolingBackward: set poolDesc: %w", err)
	}

	return d.handle.PoolingBackward(poolDesc, 1.0, yDesc, y, dyDesc, dy, xDesc, x, 0.0, dxDesc, dx)
}

func (d *CUDADNN) SoftmaxForward(
	x unsafe.Pointer, shape [4]int,
	y unsafe.Pointer,
	stream Stream,
) error {
	xDesc, err := makeTensor4d(shape)
	if err != nil {
		return fmt.Errorf("SoftmaxForward: xDesc: %w", err)
	}
	defer func() { _ = xDesc.Destroy() }()

	yDesc, err := makeTensor4d(shape)
	if err != nil {
		return fmt.Errorf("SoftmaxForward: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	return d.handle.SoftmaxForward(cudnn.SoftmaxAccurate, cudnn.SoftmaxModeChannel, 1.0, xDesc, x, 0.0, yDesc, y)
}

func (d *CUDADNN) AddTensor(
	alpha float32,
	b unsafe.Pointer, bShape [4]int,
	beta float32,
	y unsafe.Pointer, yShape [4]int,
	stream Stream,
) error {
	bDesc, err := makeTensor4d(bShape)
	if err != nil {
		return fmt.Errorf("AddTensor: bDesc: %w", err)
	}
	defer func() { _ = bDesc.Destroy() }()

	yDesc, err := makeTensor4d(yShape)
	if err != nil {
		return fmt.Errorf("AddTensor: yDesc: %w", err)
	}
	defer func() { _ = yDesc.Destroy() }()

	return d.handle.AddTensor(alpha, bDesc, b, beta, yDesc, y)
}

// --- helpers ---

// makeTensor4d creates and configures a NCHW float32 tensor descriptor.
func makeTensor4d(shape [4]int) (*cudnn.TensorDescriptor, error) {
	desc, err := cudnn.CreateTensorDescriptor()
	if err != nil {
		return nil, err
	}
	if err := desc.Set4d(cudnn.NCHW, cudnn.Float32, shape[0], shape[1], shape[2], shape[3]); err != nil {
		_ = desc.Destroy()
		return nil, err
	}
	return desc, nil
}

// cudnnActivationMode converts gpuapi.ActivationMode to cudnn.ActivationMode.
func cudnnActivationMode(mode ActivationMode) cudnn.ActivationMode {
	switch mode {
	case ActivationSigmoid:
		return cudnn.ActivationSigmoid
	case ActivationReLU:
		return cudnn.ActivationReLU
	case ActivationTanh:
		return cudnn.ActivationTanh
	case ActivationClippedReLU:
		return cudnn.ActivationClippedReLU
	case ActivationELU:
		return cudnn.ActivationELU
	default:
		return cudnn.ActivationReLU
	}
}

// cudnnPoolingMode converts gpuapi.PoolingMode to cudnn.PoolingMode.
func cudnnPoolingMode(mode PoolingMode) cudnn.PoolingMode {
	switch mode {
	case PoolingMax:
		return cudnn.PoolingMax
	case PoolingAverageCountIncludePad:
		return cudnn.PoolingAverageCountIncludePad
	case PoolingAverageCountExcludePad:
		return cudnn.PoolingAverageCountExcludePad
	default:
		return cudnn.PoolingMax
	}
}

// cudaMallocTemp allocates temporary device memory for workspace buffers.
func cudaMallocTemp(size int) (unsafe.Pointer, error) {
	return cuda.Malloc(size)
}

// cudaFreeTemp frees temporary device memory.
func cudaFreeTemp(ptr unsafe.Pointer) {
	cuda.Free(ptr) //nolint:errcheck // workspace cleanup is best-effort
}

func init() {
	if cudnn.Available() {
		DNNFactory = func() (DNN, error) { return NewCUDADNN() }
	}
}

// Compile-time interface assertion.
var _ DNN = (*CUDADNN)(nil)
