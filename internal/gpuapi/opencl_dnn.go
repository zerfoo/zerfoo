package gpuapi

import (
	"fmt"
	"unsafe"
)

// OpenCLDNN implements the DNN interface for OpenCL.
// OpenCL has no standard DNN library (like cuDNN or MIOpen), so all
// operations return ErrNotSupported. The compute engine falls back to CPU.
type OpenCLDNN struct{}

// NewOpenCLDNN returns a new OpenCL DNN stub.
func NewOpenCLDNN() *OpenCLDNN {
	return &OpenCLDNN{}
}

var errOpenCLDNNNotSupported = fmt.Errorf("DNN operations not supported on OpenCL (no standard DNN library)")

func (d *OpenCLDNN) SetStream(_ Stream) error { return nil }
func (d *OpenCLDNN) Destroy() error            { return nil }

func (d *OpenCLDNN) ConvForward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) ConvBackwardData(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) ConvBackwardFilter(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _ [2]int,
	_ int,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) BatchNormForwardInference(
	_ unsafe.Pointer, _ [4]int,
	_, _, _, _ unsafe.Pointer,
	_ int,
	_ float64,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) BatchNormForwardTraining(
	_ unsafe.Pointer, _ [4]int,
	_, _ unsafe.Pointer,
	_ int,
	_, _ float64,
	_, _ unsafe.Pointer,
	_, _ unsafe.Pointer,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) BatchNormBackward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ unsafe.Pointer,
	_ int,
	_, _ unsafe.Pointer,
	_, _, _ unsafe.Pointer,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) ActivationForward(
	_ ActivationMode,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) ActivationBackward(
	_ ActivationMode,
	_ unsafe.Pointer, _ unsafe.Pointer,
	_ unsafe.Pointer, _ unsafe.Pointer,
	_ [4]int,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) PoolingForward(
	_ PoolingMode,
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ [4]int,
	_, _, _, _, _, _ int,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) PoolingBackward(
	_ PoolingMode,
	_ unsafe.Pointer, _ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer, _ unsafe.Pointer, _ [4]int,
	_, _, _, _, _, _ int,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) SoftmaxForward(
	_ unsafe.Pointer, _ [4]int,
	_ unsafe.Pointer,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

func (d *OpenCLDNN) AddTensor(
	_ float32,
	_ unsafe.Pointer, _ [4]int,
	_ float32,
	_ unsafe.Pointer, _ [4]int,
	_ Stream,
) error {
	return errOpenCLDNNNotSupported
}

// Compile-time interface assertion.
var _ DNN = (*OpenCLDNN)(nil)
