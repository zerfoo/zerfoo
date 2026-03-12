package compute

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/tensor"
)

// Conv2dForward performs 2D convolution using the GPU DNN backend.
// x must be [N, C_in, H, W], w must be [C_out, C_in/groups, kH, kW].
// bias is optional (nil to skip). pads is [top, left, bottom, right].
// Returns error if padding is asymmetric (cuDNN requires symmetric padding).
func (e *GPUEngine[T]) Conv2dForward(
	_ context.Context,
	x, w *tensor.TensorNumeric[T],
	bias *tensor.TensorNumeric[T],
	strides [2]int,
	pads [4]int,
	dilations [2]int,
	groups int,
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("Conv2dForward: DNN not available (build without -tags cuda)")
	}
	// Only float32 has a DNN path.
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("Conv2dForward: only float32 supported, got %T", zero)
	}

	e.setDevice()

	xShape := x.Shape()
	wShape := w.Shape()
	if len(xShape) != 4 {
		return nil, fmt.Errorf("Conv2dForward: x must be 4D [N,C,H,W], got %v", xShape)
	}
	if len(wShape) != 4 {
		return nil, fmt.Errorf("Conv2dForward: w must be 4D [C_out,C_in/g,kH,kW], got %v", wShape)
	}

	// DNN requires symmetric padding.
	padH, padW := pads[0], pads[1]
	if pads[0] != pads[2] || pads[1] != pads[3] {
		return nil, fmt.Errorf("Conv2dForward: DNN requires symmetric padding, got [%d,%d,%d,%d]", pads[0], pads[1], pads[2], pads[3])
	}

	n, cIn, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3]
	cOut, _, kH, kW := wShape[0], wShape[1], wShape[2], wShape[3]
	sH, sW := strides[0], strides[1]
	dH, dW := dilations[0], dilations[1]

	// Compute output dimensions.
	outH := (inH+2*padH-dH*(kH-1)-1)/sH + 1
	outW := (inW+2*padW-dW*(kW-1)-1)/sW + 1

	// --- Device pointers ---

	devX, cleanupX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: getDevicePtr(x): %w", err)
	}
	defer cleanupX()

	devW, cleanupW, err := getDevicePtr(e, w)
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: getDevicePtr(w): %w", err)
	}
	defer cleanupW()

	outElems := n * cOut * outH * outW
	devY, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("Conv2dForward: output alloc: %w", err)
	}

	// --- Bias pointer ---
	var devB unsafe.Pointer
	var cleanupB func()
	if bias != nil {
		bShape := bias.Shape()
		if len(bShape) != 1 || bShape[0] != cOut {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: bias must be [%d], got %v", cOut, bShape)
		}
		devB, cleanupB, err = getDevicePtr(e, bias)
		if err != nil {
			e.pool.Free(e.deviceID, devY, outElems*f32Size)
			return nil, fmt.Errorf("Conv2dForward: getDevicePtr(bias): %w", err)
		}
		defer cleanupB()
	}

	// --- DNN forward ---

	if err := e.dnn.ConvForward(
		devX, [4]int{n, cIn, inH, inW},
		devW, [4]int{wShape[0], wShape[1], wShape[2], wShape[3]},
		devB,
		devY, [4]int{n, cOut, outH, outW},
		[2]int{padH, padW}, strides, dilations,
		groups,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		return nil, fmt.Errorf("Conv2dForward: %w", err)
	}

	return makeGPUResult[T](e, []int{n, cOut, outH, outW}, devY, outElems)
}

// BatchNormForwardInference performs batch normalization in inference mode
// using pre-computed running mean and variance via the GPU DNN backend.
// x must be [N, C, H, W]. scale, bias, mean, variance must each be [C].
func (e *GPUEngine[T]) BatchNormForwardInference(
	_ context.Context,
	x, scale, bias, mean, variance *tensor.TensorNumeric[T],
	epsilon float64,
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("BatchNormForwardInference: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("BatchNormForwardInference: only float32 supported")
	}
	e.setDevice()

	xShape := x.Shape()
	if len(xShape) != 4 {
		return nil, fmt.Errorf("BatchNormForwardInference: x must be 4D, got %v", xShape)
	}
	n, c, h, w := xShape[0], xShape[1], xShape[2], xShape[3]

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devScale, cleanScale, err := getDevicePtr(e, scale)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(scale): %w", err)
	}
	defer cleanScale()

	devBias, cleanBias, err := getDevicePtr(e, bias)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(bias): %w", err)
	}
	defer cleanBias()

	devMean, cleanMean, err := getDevicePtr(e, mean)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(mean): %w", err)
	}
	defer cleanMean()

	devVar, cleanVar, err := getDevicePtr(e, variance)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: getDevicePtr(var): %w", err)
	}
	defer cleanVar()

	outElems := n * c * h * w
	devY, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("BatchNormForwardInference: output alloc: %w", err)
	}

	if err := e.dnn.BatchNormForwardInference(
		devX, [4]int{n, c, h, w},
		devScale, devBias, devMean, devVar,
		c,
		epsilon,
		devY,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		return nil, fmt.Errorf("BatchNormForwardInference: %w", err)
	}

	return makeGPUResult[T](e, xShape, devY, outElems)
}

// CudnnActivationForward applies an activation function via the GPU DNN backend.
// mode selects the activation: ActivationReLU, ActivationSigmoid, ActivationTanh.
// The input tensor shape is preserved in the output.
func (e *GPUEngine[T]) CudnnActivationForward(
	_ context.Context,
	x *tensor.TensorNumeric[T],
	mode gpuapi.ActivationMode,
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("CudnnActivationForward: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("CudnnActivationForward: only float32 supported")
	}
	e.setDevice()

	shape := x.Shape()
	numElems := 1
	for _, d := range shape {
		numElems *= d
	}

	// Pack shape into 4D for DNN (N=1, C=numElems, H=1, W=1 for 1D/2D/3D).
	n4, c4, h4, w4 := 1, 1, 1, 1
	switch len(shape) {
	case 4:
		n4, c4, h4, w4 = shape[0], shape[1], shape[2], shape[3]
	case 3:
		n4, c4, h4 = shape[0], shape[1], shape[2]
	case 2:
		n4, c4 = shape[0], shape[1]
	case 1:
		c4 = shape[0]
	default:
		// Flatten to 1D.
		c4 = numElems
	}

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devY, err := e.pool.Alloc(e.deviceID, numElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationForward: output alloc: %w", err)
	}

	if err := e.dnn.ActivationForward(mode, devX, [4]int{n4, c4, h4, w4}, devY, e.stream); err != nil {
		e.pool.Free(e.deviceID, devY, numElems*f32Size)
		return nil, fmt.Errorf("CudnnActivationForward: %w", err)
	}

	return makeGPUResult[T](e, shape, devY, numElems)
}

// CudnnPoolingForward performs 2D pooling via the GPU DNN backend.
// x must be [N, C, H, W]. Returns [N, C, outH, outW].
func (e *GPUEngine[T]) CudnnPoolingForward(
	_ context.Context,
	x *tensor.TensorNumeric[T],
	mode gpuapi.PoolingMode,
	windowH, windowW, padH, padW, strideH, strideW int,
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("CudnnPoolingForward: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("CudnnPoolingForward: only float32 supported")
	}
	e.setDevice()

	xShape := x.Shape()
	if len(xShape) != 4 {
		return nil, fmt.Errorf("CudnnPoolingForward: x must be 4D, got %v", xShape)
	}
	n, c, inH, inW := xShape[0], xShape[1], xShape[2], xShape[3]
	outH := (inH+2*padH-windowH)/strideH + 1
	outW := (inW+2*padW-windowW)/strideW + 1

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	outElems := n * c * outH * outW
	devY, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingForward: output alloc: %w", err)
	}

	if err := e.dnn.PoolingForward(
		mode,
		devX, [4]int{n, c, inH, inW},
		devY, [4]int{n, c, outH, outW},
		windowH, windowW, padH, padW, strideH, strideW,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		return nil, fmt.Errorf("CudnnPoolingForward: %w", err)
	}

	return makeGPUResult[T](e, []int{n, c, outH, outW}, devY, outElems)
}

// CudnnSoftmaxForward computes softmax via the GPU DNN backend over the channel dimension.
// x must be [N, C, H, W] (or reshaped to fit).
func (e *GPUEngine[T]) CudnnSoftmaxForward(
	_ context.Context,
	x *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("CudnnSoftmaxForward: only float32 supported")
	}
	e.setDevice()

	shape := x.Shape()
	numElems := 1
	for _, d := range shape {
		numElems *= d
	}

	// DNN softmax operates over the C dimension in NCHW.
	n4, c4, h4, w4 := 1, 1, 1, 1
	switch len(shape) {
	case 4:
		n4, c4, h4, w4 = shape[0], shape[1], shape[2], shape[3]
	case 3:
		n4 = shape[0] * shape[1]
		c4 = shape[2]
	case 2:
		n4, c4 = shape[0], shape[1]
	case 1:
		c4 = shape[0]
	}

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devY, err := e.pool.Alloc(e.deviceID, numElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("CudnnSoftmaxForward: output alloc: %w", err)
	}

	if err := e.dnn.SoftmaxForward(devX, [4]int{n4, c4, h4, w4}, devY, e.stream); err != nil {
		e.pool.Free(e.deviceID, devY, numElems*f32Size)
		return nil, fmt.Errorf("CudnnSoftmaxForward: %w", err)
	}

	return makeGPUResult[T](e, shape, devY, numElems)
}

// Conv2dBackwardData computes the gradient of the convolution input via cuDNN.
// w: [C_out, C_in/groups, kH, kW], dy: [N, C_out, outH, outW].
// Returns dx: [N, C_in, H, W].
func (e *GPUEngine[T]) Conv2dBackwardData(
	_ context.Context,
	w *tensor.TensorNumeric[T],
	dy *tensor.TensorNumeric[T],
	dxShape [4]int,
	strides [2]int,
	pads [4]int,
	dilations [2]int,
	groups int,
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("Conv2dBackwardData: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("Conv2dBackwardData: only float32 supported, got %T", zero)
	}
	e.setDevice()

	wShape := w.Shape()
	dyShape := dy.Shape()
	if len(wShape) != 4 {
		return nil, fmt.Errorf("Conv2dBackwardData: w must be 4D, got %v", wShape)
	}
	if len(dyShape) != 4 {
		return nil, fmt.Errorf("Conv2dBackwardData: dy must be 4D, got %v", dyShape)
	}

	padH, padW := pads[0], pads[1]
	if pads[0] != pads[2] || pads[1] != pads[3] {
		return nil, fmt.Errorf("Conv2dBackwardData: DNN requires symmetric padding, got [%d,%d,%d,%d]", pads[0], pads[1], pads[2], pads[3])
	}

	devW, cleanW, err := getDevicePtr(e, w)
	if err != nil {
		return nil, fmt.Errorf("Conv2dBackwardData: getDevicePtr(w): %w", err)
	}
	defer cleanW()

	devDY, cleanDY, err := getDevicePtr(e, dy)
	if err != nil {
		return nil, fmt.Errorf("Conv2dBackwardData: getDevicePtr(dy): %w", err)
	}
	defer cleanDY()

	dxElems := dxShape[0] * dxShape[1] * dxShape[2] * dxShape[3]
	devDX, err := e.pool.Alloc(e.deviceID, dxElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("Conv2dBackwardData: output alloc: %w", err)
	}

	if err := e.dnn.ConvBackwardData(
		devW, [4]int{wShape[0], wShape[1], wShape[2], wShape[3]},
		devDY, [4]int{dyShape[0], dyShape[1], dyShape[2], dyShape[3]},
		devDX, dxShape,
		[2]int{padH, padW}, strides, dilations,
		groups,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devDX, dxElems*f32Size)
		return nil, fmt.Errorf("Conv2dBackwardData: %w", err)
	}

	return makeGPUResult[T](e, dxShape[:], devDX, dxElems)
}

// Conv2dBackwardFilter computes the gradient of the convolution filter via cuDNN.
// x: [N, C_in, H, W], dy: [N, C_out, outH, outW].
// Returns dw: [C_out, C_in/groups, kH, kW].
func (e *GPUEngine[T]) Conv2dBackwardFilter(
	_ context.Context,
	x *tensor.TensorNumeric[T],
	dy *tensor.TensorNumeric[T],
	dwShape [4]int,
	strides [2]int,
	pads [4]int,
	dilations [2]int,
	groups int,
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("Conv2dBackwardFilter: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("Conv2dBackwardFilter: only float32 supported, got %T", zero)
	}
	e.setDevice()

	xShape := x.Shape()
	dyShape := dy.Shape()
	if len(xShape) != 4 {
		return nil, fmt.Errorf("Conv2dBackwardFilter: x must be 4D, got %v", xShape)
	}
	if len(dyShape) != 4 {
		return nil, fmt.Errorf("Conv2dBackwardFilter: dy must be 4D, got %v", dyShape)
	}

	padH, padW := pads[0], pads[1]
	if pads[0] != pads[2] || pads[1] != pads[3] {
		return nil, fmt.Errorf("Conv2dBackwardFilter: DNN requires symmetric padding, got [%d,%d,%d,%d]", pads[0], pads[1], pads[2], pads[3])
	}

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("Conv2dBackwardFilter: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devDY, cleanDY, err := getDevicePtr(e, dy)
	if err != nil {
		return nil, fmt.Errorf("Conv2dBackwardFilter: getDevicePtr(dy): %w", err)
	}
	defer cleanDY()

	dwElems := dwShape[0] * dwShape[1] * dwShape[2] * dwShape[3]
	devDW, err := e.pool.Alloc(e.deviceID, dwElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("Conv2dBackwardFilter: output alloc: %w", err)
	}

	if err := e.dnn.ConvBackwardFilter(
		devX, [4]int{xShape[0], xShape[1], xShape[2], xShape[3]},
		devDY, [4]int{dyShape[0], dyShape[1], dyShape[2], dyShape[3]},
		devDW, dwShape,
		[2]int{padH, padW}, strides, dilations,
		groups,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devDW, dwElems*f32Size)
		return nil, fmt.Errorf("Conv2dBackwardFilter: %w", err)
	}

	return makeGPUResult[T](e, dwShape[:], devDW, dwElems)
}

// BatchNormForwardTraining performs batch normalization computing batch statistics.
// x: [N, C, H, W], scale/bias: [C].
// Returns y: [N, C, H, W], saveMean: [C], saveInvVariance: [C].
// Updates runningMean and runningVariance in-place.
func (e *GPUEngine[T]) BatchNormForwardTraining(
	_ context.Context,
	x, scale, bias *tensor.TensorNumeric[T],
	runningMean, runningVariance *tensor.TensorNumeric[T],
	epsilon, expAvgFactor float64,
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], *tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: only float32 supported")
	}
	e.setDevice()

	xShape := x.Shape()
	if len(xShape) != 4 {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: x must be 4D, got %v", xShape)
	}
	n, c, h, w := xShape[0], xShape[1], xShape[2], xShape[3]

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devScale, cleanScale, err := getDevicePtr(e, scale)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: getDevicePtr(scale): %w", err)
	}
	defer cleanScale()

	devBias, cleanBias, err := getDevicePtr(e, bias)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: getDevicePtr(bias): %w", err)
	}
	defer cleanBias()

	devRunMean, cleanRM, err := getDevicePtr(e, runningMean)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: getDevicePtr(runningMean): %w", err)
	}
	defer cleanRM()

	devRunVar, cleanRV, err := getDevicePtr(e, runningVariance)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: getDevicePtr(runningVar): %w", err)
	}
	defer cleanRV()

	outElems := n * c * h * w
	devY, err := e.pool.Alloc(e.deviceID, outElems*f32Size)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: output alloc: %w", err)
	}

	devSaveMean, err := e.pool.Alloc(e.deviceID, c*f32Size)
	if err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: saveMean alloc: %w", err)
	}

	devSaveInvVar, err := e.pool.Alloc(e.deviceID, c*f32Size)
	if err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		e.pool.Free(e.deviceID, devSaveMean, c*f32Size)
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: saveInvVar alloc: %w", err)
	}

	if err := e.dnn.BatchNormForwardTraining(
		devX, [4]int{n, c, h, w},
		devScale, devBias,
		c,
		epsilon, expAvgFactor,
		devRunMean, devRunVar,
		devSaveMean, devSaveInvVar,
		devY,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devY, outElems*f32Size)
		e.pool.Free(e.deviceID, devSaveMean, c*f32Size)
		e.pool.Free(e.deviceID, devSaveInvVar, c*f32Size)
		return nil, nil, nil, fmt.Errorf("BatchNormForwardTraining: %w", err)
	}

	y, err := makeGPUResult[T](e, xShape, devY, outElems)
	if err != nil {
		e.pool.Free(e.deviceID, devSaveMean, c*f32Size)
		e.pool.Free(e.deviceID, devSaveInvVar, c*f32Size)
		return nil, nil, nil, err
	}

	saveMeanT, err := makeGPUResult[T](e, []int{c}, devSaveMean, c)
	if err != nil {
		e.pool.Free(e.deviceID, devSaveInvVar, c*f32Size)
		return nil, nil, nil, err
	}

	saveInvVarT, err := makeGPUResult[T](e, []int{c}, devSaveInvVar, c)
	if err != nil {
		return nil, nil, nil, err
	}

	return y, saveMeanT, saveInvVarT, nil
}

// CudnnBatchNormBackward computes gradients for batch normalization via cuDNN.
// x: [N, C, H, W], dy: [N, C, H, W], scale: [C].
// saveMean, saveInvVariance: [C] (from BatchNormForwardTraining).
// Returns dx: [N, C, H, W], dScale: [C], dBias: [C].
func (e *GPUEngine[T]) CudnnBatchNormBackward(
	_ context.Context,
	x, dy, scale *tensor.TensorNumeric[T],
	saveMean, saveInvVariance *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], *tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: only float32 supported")
	}
	e.setDevice()

	xShape := x.Shape()
	if len(xShape) != 4 {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: x must be 4D, got %v", xShape)
	}
	n, c, h, w := xShape[0], xShape[1], xShape[2], xShape[3]

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devDY, cleanDY, err := getDevicePtr(e, dy)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: getDevicePtr(dy): %w", err)
	}
	defer cleanDY()

	devScale, cleanScale, err := getDevicePtr(e, scale)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: getDevicePtr(scale): %w", err)
	}
	defer cleanScale()

	devSaveMean, cleanSM, err := getDevicePtr(e, saveMean)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: getDevicePtr(saveMean): %w", err)
	}
	defer cleanSM()

	devSaveInvVar, cleanSIV, err := getDevicePtr(e, saveInvVariance)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: getDevicePtr(saveInvVar): %w", err)
	}
	defer cleanSIV()

	dxElems := n * c * h * w
	devDX, err := e.pool.Alloc(e.deviceID, dxElems*f32Size)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: dx alloc: %w", err)
	}

	devDScale, err := e.pool.Alloc(e.deviceID, c*f32Size)
	if err != nil {
		e.pool.Free(e.deviceID, devDX, dxElems*f32Size)
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: dScale alloc: %w", err)
	}

	devDBias, err := e.pool.Alloc(e.deviceID, c*f32Size)
	if err != nil {
		e.pool.Free(e.deviceID, devDX, dxElems*f32Size)
		e.pool.Free(e.deviceID, devDScale, c*f32Size)
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: dBias alloc: %w", err)
	}

	if err := e.dnn.BatchNormBackward(
		devX, [4]int{n, c, h, w},
		devDY,
		devScale,
		c,
		devSaveMean, devSaveInvVar,
		devDX, devDScale, devDBias,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devDX, dxElems*f32Size)
		e.pool.Free(e.deviceID, devDScale, c*f32Size)
		e.pool.Free(e.deviceID, devDBias, c*f32Size)
		return nil, nil, nil, fmt.Errorf("CudnnBatchNormBackward: %w", err)
	}

	dx, err := makeGPUResult[T](e, xShape, devDX, dxElems)
	if err != nil {
		e.pool.Free(e.deviceID, devDScale, c*f32Size)
		e.pool.Free(e.deviceID, devDBias, c*f32Size)
		return nil, nil, nil, err
	}

	dScale, err := makeGPUResult[T](e, []int{c}, devDScale, c)
	if err != nil {
		e.pool.Free(e.deviceID, devDBias, c*f32Size)
		return nil, nil, nil, err
	}

	dBias, err := makeGPUResult[T](e, []int{c}, devDBias, c)
	if err != nil {
		return nil, nil, nil, err
	}

	return dx, dScale, dBias, nil
}

// CudnnActivationBackward computes the gradient of an activation function via cuDNN.
// y: forward output, dy: upstream gradient, x: original input.
// All must have the same shape. Returns dx with the same shape.
func (e *GPUEngine[T]) CudnnActivationBackward(
	_ context.Context,
	x, y, dy *tensor.TensorNumeric[T],
	mode gpuapi.ActivationMode,
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("CudnnActivationBackward: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("CudnnActivationBackward: only float32 supported")
	}
	e.setDevice()

	shape := x.Shape()
	numElems := 1
	for _, d := range shape {
		numElems *= d
	}

	n4, c4, h4, w4 := 1, 1, 1, 1
	switch len(shape) {
	case 4:
		n4, c4, h4, w4 = shape[0], shape[1], shape[2], shape[3]
	case 3:
		n4, c4, h4 = shape[0], shape[1], shape[2]
	case 2:
		n4, c4 = shape[0], shape[1]
	case 1:
		c4 = shape[0]
	default:
		c4 = numElems
	}

	devY, cleanY, err := getDevicePtr(e, y)
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationBackward: getDevicePtr(y): %w", err)
	}
	defer cleanY()

	devDY, cleanDY, err := getDevicePtr(e, dy)
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationBackward: getDevicePtr(dy): %w", err)
	}
	defer cleanDY()

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationBackward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	devDX, err := e.pool.Alloc(e.deviceID, numElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("CudnnActivationBackward: output alloc: %w", err)
	}

	if err := e.dnn.ActivationBackward(mode, devY, devDY, devX, devDX, [4]int{n4, c4, h4, w4}, e.stream); err != nil {
		e.pool.Free(e.deviceID, devDX, numElems*f32Size)
		return nil, fmt.Errorf("CudnnActivationBackward: %w", err)
	}

	return makeGPUResult[T](e, shape, devDX, numElems)
}

// CudnnPoolingBackward computes the gradient of 2D pooling via cuDNN.
// y: forward output [N,C,outH,outW], dy: upstream gradient [N,C,outH,outW],
// x: forward input [N,C,H,W]. Returns dx: [N,C,H,W].
func (e *GPUEngine[T]) CudnnPoolingBackward(
	_ context.Context,
	x, y, dy *tensor.TensorNumeric[T],
	mode gpuapi.PoolingMode,
	windowH, windowW, padH, padW, strideH, strideW int,
) (*tensor.TensorNumeric[T], error) {
	if e.dnn == nil {
		return nil, fmt.Errorf("CudnnPoolingBackward: DNN not available (build without -tags cuda)")
	}
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("CudnnPoolingBackward: only float32 supported")
	}
	e.setDevice()

	xShape := x.Shape()
	yShape := y.Shape()
	if len(xShape) != 4 {
		return nil, fmt.Errorf("CudnnPoolingBackward: x must be 4D, got %v", xShape)
	}
	if len(yShape) != 4 {
		return nil, fmt.Errorf("CudnnPoolingBackward: y must be 4D, got %v", yShape)
	}

	devY, cleanY, err := getDevicePtr(e, y)
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingBackward: getDevicePtr(y): %w", err)
	}
	defer cleanY()

	devDY, cleanDY, err := getDevicePtr(e, dy)
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingBackward: getDevicePtr(dy): %w", err)
	}
	defer cleanDY()

	devX, cleanX, err := getDevicePtr(e, x)
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingBackward: getDevicePtr(x): %w", err)
	}
	defer cleanX()

	dxElems := xShape[0] * xShape[1] * xShape[2] * xShape[3]
	devDX, err := e.pool.Alloc(e.deviceID, dxElems*f32Size)
	if err != nil {
		return nil, fmt.Errorf("CudnnPoolingBackward: output alloc: %w", err)
	}

	if err := e.dnn.PoolingBackward(
		mode,
		devY, devDY, [4]int{yShape[0], yShape[1], yShape[2], yShape[3]},
		devX, devDX, [4]int{xShape[0], xShape[1], xShape[2], xShape[3]},
		windowH, windowW, padH, padW, strideH, strideW,
		e.stream,
	); err != nil {
		e.pool.Free(e.deviceID, devDX, dxElems*f32Size)
		return nil, fmt.Errorf("CudnnPoolingBackward: %w", err)
	}

	return makeGPUResult[T](e, xShape, devDX, dxElems)
}
