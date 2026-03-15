//go:build opencl

package compute

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/zerfoo/zerfoo/internal/gpuapi"
	"github.com/zerfoo/zerfoo/log"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// OpenCLEngine is an OpenCL GPU-accelerated implementation of the Engine interface.
// It uses the same GRAL-based architecture as GPUEngine but creates OpenCL
// adapters (OpenCL runtime, CLBlast, OpenCL kernels) instead of CUDA ones.
//
// All compute methods currently delegate to CPUEngine for correctness.
// The GRAL OpenCL infrastructure (runtime, blas, dnn, kernels, pool, stream)
// is wired and ready for GPU-accelerated paths when hardware is available.
type OpenCLEngine[T tensor.Numeric] struct {
	cpu      *CPUEngine[T]
	runtime  *gpuapi.OpenCLRuntime
	blas     gpuapi.BLAS
	dnn      gpuapi.DNN
	kernels  gpuapi.KernelRunner
	pool     gpuapi.MemPool
	stream   gpuapi.Stream
	logger   log.Logger
	deviceID int

	oomFallbackCount atomic.Int64
}

// NewOpenCLEngine creates a new OpenCLEngine backed by OpenCL via GRAL.
// An optional deviceID selects the GPU (default 0).
// Call Close() when done to release all resources.
func NewOpenCLEngine[T tensor.Numeric](ops numeric.Arithmetic[T], deviceID ...int) (*OpenCLEngine[T], error) {
	dev := 0
	if len(deviceID) > 0 {
		dev = deviceID[0]
	}

	rt := gpuapi.NewOpenCLRuntime()
	if err := rt.SetDevice(dev); err != nil {
		return nil, fmt.Errorf("failed to set OpenCL device %d: %w", dev, err)
	}

	stream, err := rt.CreateStream()
	if err != nil {
		return nil, fmt.Errorf("failed to create OpenCL command queue: %w", err)
	}

	blas := gpuapi.NewOpenCLBlas(rt.CLQueue(), rt.CLContext())
	if err := blas.SetStream(stream); err != nil {
		_ = stream.Destroy()
		return nil, fmt.Errorf("failed to set CLBlast stream: %w", err)
	}

	kernels, err := gpuapi.NewOpenCLKernels(rt.CLContext(), rt.CLDevice(), rt.CLQueue())
	if err != nil {
		_ = stream.Destroy()
		_ = blas.Destroy()
		return nil, fmt.Errorf("failed to compile OpenCL kernels: %w", err)
	}

	dnn := gpuapi.NewOpenCLDNN()
	pool := gpuapi.NewOpenCLMemPool(rt)

	l := log.Nop()
	l.Info("opencl engine initialized", "device", dev, "pool", "enabled", "stream", "enabled")

	return &OpenCLEngine[T]{
		cpu:      NewCPUEngine(ops),
		runtime:  rt,
		blas:     blas,
		dnn:      dnn,
		kernels:  kernels,
		pool:     pool,
		stream:   stream,
		logger:   l,
		deviceID: dev,
	}, nil
}

// DeviceID returns the GPU device ID this engine is bound to.
func (e *OpenCLEngine[T]) DeviceID() int { return e.deviceID }

// SetLogger replaces the engine's logger.
func (e *OpenCLEngine[T]) SetLogger(l log.Logger) {
	if l == nil {
		l = log.Nop()
	}
	e.logger = l
	e.cpu.SetLogger(l)
}

// Close releases all OpenCL resources.
func (e *OpenCLEngine[T]) Close() error {
	if e.pool != nil {
		_ = e.pool.Drain()
	}
	if e.kernels != nil {
		e.kernels.Destroy()
	}
	if e.dnn != nil {
		_ = e.dnn.Destroy()
	}
	if e.stream != nil {
		_ = e.stream.Destroy()
	}
	if e.blas != nil {
		_ = e.blas.Destroy()
	}
	return nil
}

// OOMFallbackCount returns the number of OOM-triggered CPU fallbacks.
func (e *OpenCLEngine[T]) OOMFallbackCount() int64 {
	return e.oomFallbackCount.Load()
}

// --- Engine[T] interface implementation ---
// All compute methods delegate to CPUEngine for correctness. The GRAL OpenCL
// infrastructure is wired and ready for GPU-accelerated paths.

func (e *OpenCLEngine[T]) Ops() numeric.Arithmetic[T] { return e.cpu.Ops() }

func (e *OpenCLEngine[T]) UnaryOp(ctx context.Context, a *tensor.TensorNumeric[T], op func(T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.UnaryOp(ctx, a, op, dst...)
}

func (e *OpenCLEngine[T]) Add(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Add(ctx, a, b, dst...)
}

func (e *OpenCLEngine[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Sub(ctx, a, b, dst...)
}

func (e *OpenCLEngine[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Mul(ctx, a, b, dst...)
}

func (e *OpenCLEngine[T]) Div(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Div(ctx, a, b, dst...)
}

func (e *OpenCLEngine[T]) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.MatMul(ctx, a, b, dst...)
}

func (e *OpenCLEngine[T]) Transpose(ctx context.Context, a *tensor.TensorNumeric[T], axes []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Transpose(ctx, a, axes, dst...)
}

func (e *OpenCLEngine[T]) Sum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Sum(ctx, a, axis, keepDims, dst...)
}

func (e *OpenCLEngine[T]) Exp(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Exp(ctx, a, dst...)
}

func (e *OpenCLEngine[T]) Log(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Log(ctx, a, dst...)
}

func (e *OpenCLEngine[T]) Sin(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Sin(ctx, a, dst...)
}

func (e *OpenCLEngine[T]) Cos(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Cos(ctx, a, dst...)
}

func (e *OpenCLEngine[T]) Tanh(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Tanh(ctx, a, dst...)
}

func (e *OpenCLEngine[T]) TanhPrime(ctx context.Context, a, upstream *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.TanhPrime(ctx, a, upstream, dst...)
}

func (e *OpenCLEngine[T]) Pow(ctx context.Context, base, exponent *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Pow(ctx, base, exponent, dst...)
}

func (e *OpenCLEngine[T]) Zero(ctx context.Context, a *tensor.TensorNumeric[T]) error {
	return e.cpu.Zero(ctx, a)
}

func (e *OpenCLEngine[T]) Zeros(ctx context.Context, a *tensor.TensorNumeric[T], shape []int) error {
	return e.cpu.Zeros(ctx, a, shape)
}

func (e *OpenCLEngine[T]) Copy(ctx context.Context, dst, src *tensor.TensorNumeric[T]) error {
	return e.cpu.Copy(ctx, dst, src)
}

func (e *OpenCLEngine[T]) Gather(ctx context.Context, params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T]) error {
	return e.cpu.Gather(ctx, params, indices, output)
}

func (e *OpenCLEngine[T]) ScatterAdd(ctx context.Context, dEmbeddingTable *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[T]) error {
	return e.cpu.ScatterAdd(ctx, dEmbeddingTable, indices, dOut)
}

func (e *OpenCLEngine[T]) RandomUniform(ctx context.Context, t *tensor.TensorNumeric[T], minVal, maxVal T) error {
	return e.cpu.RandomUniform(ctx, t, minVal, maxVal)
}

func (e *OpenCLEngine[T]) Fill(ctx context.Context, t *tensor.TensorNumeric[T], value T) error {
	return e.cpu.Fill(ctx, t, value)
}

func (e *OpenCLEngine[T]) MulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.MulScalar(ctx, a, scalar, dst...)
}

func (e *OpenCLEngine[T]) DivScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.DivScalar(ctx, a, scalar, dst...)
}

func (e *OpenCLEngine[T]) Softmax(ctx context.Context, a *tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Softmax(ctx, a, axis, dst...)
}

func (e *OpenCLEngine[T]) ReduceSum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.ReduceSum(ctx, a, axis, keepDims, dst...)
}

func (e *OpenCLEngine[T]) AddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.AddScalar(ctx, a, scalar, dst...)
}

func (e *OpenCLEngine[T]) Sqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Sqrt(ctx, a, dst...)
}

func (e *OpenCLEngine[T]) Split(ctx context.Context, a *tensor.TensorNumeric[T], numSplits int, axis int) ([]*tensor.TensorNumeric[T], error) {
	return e.cpu.Split(ctx, a, numSplits, axis)
}

func (e *OpenCLEngine[T]) Concat(ctx context.Context, tensors []*tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Concat(ctx, tensors, axis, dst...)
}

func (e *OpenCLEngine[T]) Repeat(ctx context.Context, a *tensor.TensorNumeric[T], axis int, repetitions int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Repeat(ctx, a, axis, repetitions, dst...)
}

func (e *OpenCLEngine[T]) OneHot(ctx context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.OneHot(ctx, input, depth, dst...)
}

func (e *OpenCLEngine[T]) Reshape(ctx context.Context, a *tensor.TensorNumeric[T], shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Reshape(ctx, a, shape, dst...)
}

func (e *OpenCLEngine[T]) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.ReduceMean(ctx, a, axis, keepDims, dst...)
}

func (e *OpenCLEngine[T]) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.cpu.Rsqrt(ctx, a, dst...)
}

// Static type assertion: OpenCLEngine satisfies Engine.
var _ Engine[float32] = (*OpenCLEngine[float32])(nil)
