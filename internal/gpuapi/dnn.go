package gpuapi

import "unsafe"

// ActivationMode selects the activation function for DNN operations.
type ActivationMode int

const (
	ActivationSigmoid ActivationMode = iota
	ActivationReLU
	ActivationTanh
	ActivationClippedReLU
	ActivationELU
)

// PoolingMode selects the pooling strategy for DNN operations.
type PoolingMode int

const (
	PoolingMax PoolingMode = iota
	PoolingAverageCountIncludePad
	PoolingAverageCountExcludePad
)

// BatchNormMode selects the batch normalization mode.
type BatchNormMode int

const (
	BatchNormPerActivation BatchNormMode = iota
	BatchNormSpatial
)

// DNN abstracts GPU-accelerated deep neural network primitives.
// Each vendor (cuDNN, MIOpen) provides an implementation.
//
// All tensor pointers are device memory. Shapes follow NCHW layout.
// The implementation handles descriptor creation internally.
type DNN interface {
	// ConvForward performs 2D convolution.
	// x: [N,C_in,H,W], w: [C_out,C_in/groups,kH,kW], y: [N,C_out,outH,outW].
	// bias is optional (nil to skip).
	// pads: [padH, padW] (symmetric), strides: [sH, sW], dilations: [dH, dW].
	ConvForward(
		x unsafe.Pointer, xShape [4]int,
		w unsafe.Pointer, wShape [4]int,
		bias unsafe.Pointer,
		y unsafe.Pointer, yShape [4]int,
		pads [2]int, strides [2]int, dilations [2]int,
		groups int,
		stream Stream,
	) error

	// ConvBackwardData computes the gradient of the input for 2D convolution.
	// w: [C_out,C_in/groups,kH,kW], dy: [N,C_out,outH,outW], dx: [N,C_in,H,W].
	ConvBackwardData(
		w unsafe.Pointer, wShape [4]int,
		dy unsafe.Pointer, dyShape [4]int,
		dx unsafe.Pointer, dxShape [4]int,
		pads [2]int, strides [2]int, dilations [2]int,
		groups int,
		stream Stream,
	) error

	// ConvBackwardFilter computes the gradient of the filter for 2D convolution.
	// x: [N,C_in,H,W], dy: [N,C_out,outH,outW], dw: [C_out,C_in/groups,kH,kW].
	ConvBackwardFilter(
		x unsafe.Pointer, xShape [4]int,
		dy unsafe.Pointer, dyShape [4]int,
		dw unsafe.Pointer, dwShape [4]int,
		pads [2]int, strides [2]int, dilations [2]int,
		groups int,
		stream Stream,
	) error

	// BatchNormForwardInference performs batch normalization using running statistics.
	// x: [N,C,H,W], scale/bias/mean/variance: [C], y: [N,C,H,W].
	BatchNormForwardInference(
		x unsafe.Pointer, xShape [4]int,
		scale, bias, mean, variance unsafe.Pointer,
		channels int,
		epsilon float64,
		y unsafe.Pointer,
		stream Stream,
	) error

	// BatchNormForwardTraining performs batch normalization computing batch statistics.
	// x: [N,C,H,W], scale/bias: [C], y: [N,C,H,W].
	// saveMean and saveInvVariance are outputs for the backward pass, each [C].
	// runningMean and runningVariance are updated in-place with exponential averaging.
	BatchNormForwardTraining(
		x unsafe.Pointer, xShape [4]int,
		scale, bias unsafe.Pointer,
		channels int,
		epsilon, expAvgFactor float64,
		runningMean, runningVariance unsafe.Pointer,
		saveMean, saveInvVariance unsafe.Pointer,
		y unsafe.Pointer,
		stream Stream,
	) error

	// BatchNormBackward computes gradients for batch normalization.
	// x: [N,C,H,W], dy: [N,C,H,W], scale: [C].
	// saveMean, saveInvVariance: [C] (from BatchNormForwardTraining).
	// dx: [N,C,H,W], dScale, dBias: [C].
	BatchNormBackward(
		x unsafe.Pointer, xShape [4]int,
		dy unsafe.Pointer,
		scale unsafe.Pointer,
		channels int,
		saveMean, saveInvVariance unsafe.Pointer,
		dx, dScale, dBias unsafe.Pointer,
		stream Stream,
	) error

	// ActivationForward applies an activation function element-wise.
	// x and y have the same shape [N,C,H,W].
	ActivationForward(
		mode ActivationMode,
		x unsafe.Pointer, shape [4]int,
		y unsafe.Pointer,
		stream Stream,
	) error

	// ActivationBackward computes the gradient of an activation function.
	// x: original input, y: forward output, dy: upstream gradient, dx: output gradient.
	// All have shape [N,C,H,W].
	ActivationBackward(
		mode ActivationMode,
		y unsafe.Pointer, dy unsafe.Pointer,
		x unsafe.Pointer, dx unsafe.Pointer,
		shape [4]int,
		stream Stream,
	) error

	// PoolingForward performs 2D pooling.
	// x: [N,C,H,W], y: [N,C,outH,outW].
	PoolingForward(
		mode PoolingMode,
		x unsafe.Pointer, xShape [4]int,
		y unsafe.Pointer, yShape [4]int,
		windowH, windowW, padH, padW, strideH, strideW int,
		stream Stream,
	) error

	// PoolingBackward computes the gradient of 2D pooling.
	// y: forward output, dy: upstream gradient, x: forward input, dx: output gradient.
	PoolingBackward(
		mode PoolingMode,
		y unsafe.Pointer, dy unsafe.Pointer, yShape [4]int,
		x unsafe.Pointer, dx unsafe.Pointer, xShape [4]int,
		windowH, windowW, padH, padW, strideH, strideW int,
		stream Stream,
	) error

	// SoftmaxForward computes softmax over the channel dimension.
	// x and y have the same shape [N,C,H,W].
	SoftmaxForward(
		x unsafe.Pointer, shape [4]int,
		y unsafe.Pointer,
		stream Stream,
	) error

	// AddTensor performs y = alpha*b + beta*y for bias addition.
	// b: [1,C,1,1], y: [N,C,H,W].
	AddTensor(
		alpha float32,
		b unsafe.Pointer, bShape [4]int,
		beta float32,
		y unsafe.Pointer, yShape [4]int,
		stream Stream,
	) error

	// SetStream associates the DNN handle with an asynchronous stream.
	SetStream(stream Stream) error

	// Destroy releases the DNN handle resources.
	Destroy() error
}
