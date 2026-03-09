package gpuapi_test

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
)

// stubStream implements gpuapi.Stream for compile-time interface verification.
type stubStream struct{}

func (stubStream) Synchronize() error        { return nil }
func (stubStream) Destroy() error            { return nil }
func (stubStream) Ptr() unsafe.Pointer       { return nil }

var _ gpuapi.Stream = stubStream{}

// stubRuntime implements gpuapi.Runtime for compile-time interface verification.
type stubRuntime struct{}

func (stubRuntime) DeviceType() device.Type                                  { return device.CPU }
func (stubRuntime) SetDevice(int) error                                      { return nil }
func (stubRuntime) GetDeviceCount() (int, error)                             { return 0, nil }
func (stubRuntime) Malloc(int) (unsafe.Pointer, error)                       { return nil, nil }
func (stubRuntime) Free(unsafe.Pointer) error                                { return nil }
func (stubRuntime) Memcpy(_, _ unsafe.Pointer, _ int, _ gpuapi.MemcpyKind) error { return nil }
func (stubRuntime) MemcpyAsync(_, _ unsafe.Pointer, _ int, _ gpuapi.MemcpyKind, _ gpuapi.Stream) error {
	return nil
}
func (stubRuntime) MemcpyPeer(_ unsafe.Pointer, _ int, _ unsafe.Pointer, _ int, _ int) error {
	return nil
}
func (stubRuntime) CreateStream() (gpuapi.Stream, error) { return stubStream{}, nil }

var _ gpuapi.Runtime = stubRuntime{}

// stubBLAS implements gpuapi.BLAS for compile-time interface verification.
type stubBLAS struct{}

func (stubBLAS) Sgemm(_, _, _ int, _ float32, _, _ unsafe.Pointer, _ float32, _ unsafe.Pointer) error {
	return nil
}
func (stubBLAS) BFloat16Gemm(_, _, _ int, _ float32, _, _ unsafe.Pointer, _ float32, _ unsafe.Pointer) error {
	return nil
}
func (stubBLAS) SetStream(gpuapi.Stream) error { return nil }
func (stubBLAS) Destroy() error                { return nil }

var _ gpuapi.BLAS = stubBLAS{}

// stubDNN implements gpuapi.DNN for compile-time interface verification.
type stubDNN struct{}

func (stubDNN) ConvForward(_ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ unsafe.Pointer, _ [4]int, _ [2]int, _ [2]int, _ [2]int, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) ConvBackwardData(_ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ [4]int, _ [2]int, _ [2]int, _ [2]int, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) ConvBackwardFilter(_ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ [4]int, _ [2]int, _ [2]int, _ [2]int, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) BatchNormForwardInference(_ unsafe.Pointer, _ [4]int, _, _, _, _ unsafe.Pointer, _ int, _ float64, _ unsafe.Pointer, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) BatchNormForwardTraining(_ unsafe.Pointer, _ [4]int, _, _ unsafe.Pointer, _ int, _, _ float64, _, _, _, _ unsafe.Pointer, _ unsafe.Pointer, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) BatchNormBackward(_ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ unsafe.Pointer, _ int, _, _ unsafe.Pointer, _, _, _ unsafe.Pointer, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) ActivationForward(_ gpuapi.ActivationMode, _ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) ActivationBackward(_ gpuapi.ActivationMode, _, _, _, _ unsafe.Pointer, _ [4]int, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) PoolingForward(_ gpuapi.PoolingMode, _ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ [4]int, _, _, _, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) PoolingBackward(_ gpuapi.PoolingMode, _, _ unsafe.Pointer, _ [4]int, _, _ unsafe.Pointer, _ [4]int, _, _, _, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) SoftmaxForward(_ unsafe.Pointer, _ [4]int, _ unsafe.Pointer, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) AddTensor(_ float32, _ unsafe.Pointer, _ [4]int, _ float32, _ unsafe.Pointer, _ [4]int, _ gpuapi.Stream) error {
	return nil
}
func (stubDNN) SetStream(gpuapi.Stream) error { return nil }
func (stubDNN) Destroy() error                { return nil }

var _ gpuapi.DNN = stubDNN{}

// stubKernelRunner implements gpuapi.KernelRunner for compile-time interface verification.
type stubKernelRunner struct{}

func (stubKernelRunner) Add(_, _, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error { return nil }
func (stubKernelRunner) Sub(_, _, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error { return nil }
func (stubKernelRunner) Mul(_, _, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error { return nil }
func (stubKernelRunner) Div(_, _, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error { return nil }
func (stubKernelRunner) Pow(_, _, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error { return nil }
func (stubKernelRunner) Exp(_, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error    { return nil }
func (stubKernelRunner) Log(_, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error    { return nil }
func (stubKernelRunner) Sqrt(_, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error   { return nil }
func (stubKernelRunner) Rsqrt(_, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error  { return nil }
func (stubKernelRunner) Tanh(_, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error   { return nil }
func (stubKernelRunner) TanhPrime(_, _, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) AddScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) MulScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) DivScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) Fill(_ unsafe.Pointer, _ float32, _ int, _ gpuapi.Stream) error { return nil }
func (stubKernelRunner) SumAxis(_, _ unsafe.Pointer, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) Softmax(_, _ unsafe.Pointer, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) GemmQ4F32(_, _, _ unsafe.Pointer, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) AddBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) SubBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) MulBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) DivBroadcast(_, _, _ unsafe.Pointer, _, _, _, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) SubScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) PowScalar(_ unsafe.Pointer, _ float32, _ unsafe.Pointer, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) Transpose2D(_, _ unsafe.Pointer, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) TransposeND(_, _ unsafe.Pointer, _, _, _ []int32, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) Gather(_, _, _ unsafe.Pointer, _, _, _ int, _ gpuapi.Stream) error {
	return nil
}
func (stubKernelRunner) RMSNorm(_, _, _, _ unsafe.Pointer, _ float32, _, _ int, _ gpuapi.Stream) error {
	return nil
}

var _ gpuapi.KernelRunner = stubKernelRunner{}

// stubMemPool implements gpuapi.MemPool for compile-time interface verification.
type stubMemPool struct{}

func (stubMemPool) Alloc(int, int) (unsafe.Pointer, error)        { return nil, nil }
func (stubMemPool) Free(int, unsafe.Pointer, int)                 {}
func (stubMemPool) AllocManaged(int, int) (unsafe.Pointer, error) { return nil, nil }
func (stubMemPool) FreeManaged(int, unsafe.Pointer, int)          {}
func (stubMemPool) Drain() error                                  { return nil }
func (stubMemPool) Stats() (int, int)                             { return 0, 0 }

var _ gpuapi.MemPool = stubMemPool{}

func TestInterfaceStubs(t *testing.T) {
	// This test verifies that all stub types compile and satisfy their
	// respective interfaces. The var _ assertions above do the real work
	// at compile time; this function exists so `go test` has something to run.
	t.Log("all GRAL interface stubs satisfy their interfaces")
}
