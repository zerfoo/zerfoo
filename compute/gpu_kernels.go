//go:build cuda

package compute

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
	"github.com/zerfoo/zerfoo/tensor"
)

const f32Size = int(unsafe.Sizeof(float32(0)))

// gpuBinaryOp transfers two equal-length float32 tensors to GPU, runs a kernel,
// and transfers the result back. Falls back to CPU for non-float32 types.
func gpuBinaryOp[T tensor.Numeric](
	e *GPUEngine[T],
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	kernelFn func(devA, devB, devC unsafe.Pointer, n int) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	// Only float32 uses GPU kernels.
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("GPU kernel: unsupported type %T", zero)
	}

	aData := a.Data()
	bData := b.Data()
	n := len(aData)

	if len(bData) != n {
		return nil, fmt.Errorf("GPU binary op: length mismatch %d vs %d", n, len(bData))
	}

	result, err := e.cpu.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}

	byteSize := n * f32Size

	devA, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devA) }()

	devB, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devB) }()

	devC, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devC) }()

	aF32 := *(*[]float32)(unsafe.Pointer(&aData))
	bF32 := *(*[]float32)(unsafe.Pointer(&bData))

	if err := cuda.Memcpy(devA, unsafe.Pointer(&aF32[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	if err := cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	if err := kernelFn(devA, devB, devC, n); err != nil {
		return nil, err
	}

	resultF32 := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&resultF32[0]), devC, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		return nil, err
	}

	resultT := *(*[]T)(unsafe.Pointer(&resultF32))
	result.SetData(resultT)

	return result, nil
}

// gpuUnaryOp transfers a float32 tensor to GPU, runs a kernel, and returns.
func gpuUnaryOp[T tensor.Numeric](
	e *GPUEngine[T],
	a *tensor.TensorNumeric[T],
	kernelFn func(devA, devC unsafe.Pointer, n int) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("GPU kernel: unsupported type %T", zero)
	}

	aData := a.Data()
	n := len(aData)

	result, err := e.cpu.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}

	byteSize := n * f32Size

	devA, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devA) }()

	devC, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devC) }()

	aF32 := *(*[]float32)(unsafe.Pointer(&aData))

	if err := cuda.Memcpy(devA, unsafe.Pointer(&aF32[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	if err := kernelFn(devA, devC, n); err != nil {
		return nil, err
	}

	resultF32 := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&resultF32[0]), devC, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		return nil, err
	}

	resultT := *(*[]T)(unsafe.Pointer(&resultF32))
	result.SetData(resultT)

	return result, nil
}

// gpuScalarOp transfers a tensor to GPU and applies a scalar kernel.
func gpuScalarOp[T tensor.Numeric](
	e *GPUEngine[T],
	a *tensor.TensorNumeric[T],
	scalar float32,
	kernelFn func(devA unsafe.Pointer, scalar float32, devC unsafe.Pointer, n int) error,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	var zero T
	if _, ok := any(zero).(float32); !ok {
		return nil, fmt.Errorf("GPU kernel: unsupported type %T", zero)
	}

	aData := a.Data()
	n := len(aData)

	result, err := e.cpu.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}

	byteSize := n * f32Size

	devA, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devA) }()

	devC, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devC) }()

	aF32 := *(*[]float32)(unsafe.Pointer(&aData))

	if err := cuda.Memcpy(devA, unsafe.Pointer(&aF32[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	if err := kernelFn(devA, scalar, devC, n); err != nil {
		return nil, err
	}

	resultF32 := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&resultF32[0]), devC, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		return nil, err
	}

	resultT := *(*[]T)(unsafe.Pointer(&resultF32))
	result.SetData(resultT)

	return result, nil
}

// isFloat32 checks if the generic type T is float32.
func isFloat32[T tensor.Numeric]() bool {
	var zero T
	_, ok := any(zero).(float32)

	return ok
}

// toFloat32 converts a T value to float32 via any.
func toFloat32[T tensor.Numeric](v T) float32 {
	return any(v).(float32)
}

// --- GPU-accelerated method overrides ---
// These replace the CPU fallbacks in gpu_engine.go for float32 types.
// For non-float32, they fall back to CPUEngine.

func (e *GPUEngine[T]) gpuAdd(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || !sameShape(a, b) {
		return e.cpu.Add(ctx, a, b, dst...)
	}

	return gpuBinaryOp(e, ctx, a, b, kernels.Add, dst...)
}

func (e *GPUEngine[T]) gpuSub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || !sameShape(a, b) {
		return e.cpu.Sub(ctx, a, b, dst...)
	}

	return gpuBinaryOp(e, ctx, a, b, kernels.Sub, dst...)
}

func (e *GPUEngine[T]) gpuMul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || !sameShape(a, b) {
		return e.cpu.Mul(ctx, a, b, dst...)
	}

	return gpuBinaryOp(e, ctx, a, b, kernels.Mul, dst...)
}

func (e *GPUEngine[T]) gpuDiv(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || !sameShape(a, b) {
		return e.cpu.Div(ctx, a, b, dst...)
	}

	return gpuBinaryOp(e, ctx, a, b, kernels.Div, dst...)
}

func (e *GPUEngine[T]) gpuPow(ctx context.Context, base, exponent *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || !sameShape(base, exponent) {
		return e.cpu.Pow(ctx, base, exponent, dst...)
	}

	return gpuBinaryOp(e, ctx, base, exponent, kernels.Pow, dst...)
}

func (e *GPUEngine[T]) gpuExp(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Exp(ctx, a, dst...)
	}

	return gpuUnaryOp(e, a, kernels.Exp, dst...)
}

func (e *GPUEngine[T]) gpuLog(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Log(ctx, a, dst...)
	}

	return gpuUnaryOp(e, a, kernels.Log, dst...)
}

func (e *GPUEngine[T]) gpuSqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Sqrt(ctx, a, dst...)
	}

	return gpuUnaryOp(e, a, kernels.Sqrt, dst...)
}

func (e *GPUEngine[T]) gpuRsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Rsqrt(ctx, a, dst...)
	}

	return gpuUnaryOp(e, a, kernels.Rsqrt, dst...)
}

func (e *GPUEngine[T]) gpuTanh(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Tanh(ctx, a, dst...)
	}

	return gpuUnaryOp(e, a, kernels.Tanh, dst...)
}

func (e *GPUEngine[T]) gpuTanhPrime(ctx context.Context, a, upstream *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() || !sameShape(a, upstream) {
		return e.cpu.TanhPrime(ctx, a, upstream, dst...)
	}

	return gpuBinaryOp(e, ctx, a, upstream, kernels.TanhPrime, dst...)
}

func (e *GPUEngine[T]) gpuAddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.AddScalar(ctx, a, scalar, dst...)
	}

	return gpuScalarOp(e, a, toFloat32(scalar), kernels.AddScalar, dst...)
}

func (e *GPUEngine[T]) gpuMulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.MulScalar(ctx, a, scalar, dst...)
	}

	return gpuScalarOp(e, a, toFloat32(scalar), kernels.MulScalar, dst...)
}

func (e *GPUEngine[T]) gpuDivScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.DivScalar(ctx, a, scalar, dst...)
	}

	return gpuScalarOp(e, a, toFloat32(scalar), kernels.DivScalar, dst...)
}

func (e *GPUEngine[T]) gpuFill(ctx context.Context, t *tensor.TensorNumeric[T], value T) error {
	if !isFloat32[T]() {
		return e.cpu.Fill(ctx, t, value)
	}

	data := t.Data()
	n := len(data)
	byteSize := n * f32Size

	devPtr, err := cuda.Malloc(byteSize)
	if err != nil {
		return err
	}

	defer func() { _ = cuda.Free(devPtr) }()

	if err := kernels.Fill(devPtr, toFloat32(value), n); err != nil {
		return err
	}

	resultF32 := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&resultF32[0]), devPtr, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		return err
	}

	resultT := *(*[]T)(unsafe.Pointer(&resultF32))
	t.SetData(resultT)

	return nil
}

func (e *GPUEngine[T]) gpuSum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Sum(ctx, a, axis, keepDims, dst...)
	}

	if a == nil {
		return nil, fmt.Errorf("Sum: input tensor must not be nil")
	}

	// Negative axis means sum over all elements -- fall back to CPU for simplicity.
	if axis < 0 {
		return e.cpu.Sum(ctx, a, axis, keepDims, dst...)
	}

	shape := a.Shape()
	rank := len(shape)

	if axis >= rank {
		return nil, fmt.Errorf("Sum: axis %d out of bounds for %d dimensions", axis, rank)
	}

	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}

	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}

	axisSize := shape[axis]
	numStripes := outer * inner

	// Compute output shape.
	var newShape []int
	if keepDims {
		newShape = make([]int, rank)
		for i, d := range shape {
			if i == axis {
				newShape[i] = 1
			} else {
				newShape[i] = d
			}
		}
	} else {
		for i, d := range shape {
			if i != axis {
				newShape = append(newShape, d)
			}
		}
		if len(newShape) == 0 {
			newShape = []int{1}
		}
	}

	result, err := e.cpu.getOrCreateDest(newShape, dst...)
	if err != nil {
		return nil, err
	}

	aData := a.Data()
	inByteSize := len(aData) * f32Size
	outByteSize := numStripes * f32Size

	devIn, err := cuda.Malloc(inByteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devIn) }()

	devOut, err := cuda.Malloc(outByteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devOut) }()

	aF32 := *(*[]float32)(unsafe.Pointer(&aData))

	if err := cuda.Memcpy(devIn, unsafe.Pointer(&aF32[0]), inByteSize, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	if err := kernels.SumAxis(devIn, devOut, outer, inner, axisSize); err != nil {
		return nil, err
	}

	resultF32 := make([]float32, numStripes)
	if err := cuda.Memcpy(unsafe.Pointer(&resultF32[0]), devOut, outByteSize, cuda.MemcpyDeviceToHost); err != nil {
		return nil, err
	}

	// The kernel outputs stripes in order [o0*inner+in0, o0*inner+in1, ...].
	// For keepDims=false, the output shape collapses the axis dimension.
	// The stripe ordering is consistent with the output layout: for axis reduction,
	// the output index (o, in) maps to linear index o*inner + in, which matches
	// the kernel's stripe ordering.
	resultT := *(*[]T)(unsafe.Pointer(&resultF32))
	result.SetData(resultT)

	return result, nil
}

func (e *GPUEngine[T]) gpuReduceSum(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.gpuSum(ctx, a, axis, keepDims, dst...)
}

func (e *GPUEngine[T]) gpuReduceMean(ctx context.Context, a *tensor.TensorNumeric[T], axis int, keepDims bool, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.ReduceMean(ctx, a, axis, keepDims, dst...)
	}

	// Negative axis: fall back to CPU (gpuSum also falls back).
	if axis < 0 {
		return e.cpu.ReduceMean(ctx, a, axis, keepDims, dst...)
	}

	shape := a.Shape()
	rank := len(shape)

	if axis >= rank {
		return nil, fmt.Errorf("ReduceMean: axis %d out of bounds for %d dimensions", axis, rank)
	}

	// ReduceMean = Sum / axisSize
	sumResult, err := e.gpuSum(ctx, a, axis, keepDims)
	if err != nil {
		return nil, err
	}

	divisor := any(float32(shape[axis])).(T)

	return e.gpuDivScalar(ctx, sumResult, divisor, dst...)
}

func (e *GPUEngine[T]) gpuSoftmax(ctx context.Context, a *tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if !isFloat32[T]() {
		return e.cpu.Softmax(ctx, a, axis, dst...)
	}

	if a == nil {
		return nil, fmt.Errorf("Softmax: input tensor must not be nil")
	}

	shape := a.Shape()
	rank := len(shape)

	if rank == 0 {
		return e.cpu.Softmax(ctx, a, axis, dst...)
	}

	if axis < 0 {
		axis = rank + axis
	}

	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("Softmax: axis %d out of bounds for %d dimensions", axis, rank)
	}

	result, err := e.cpu.getOrCreateDest(shape, dst...)
	if err != nil {
		return nil, err
	}

	aData := a.Data()
	n := len(aData)
	byteSize := n * f32Size

	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}

	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}

	axisSize := shape[axis]

	devIn, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devIn) }()

	devOut, err := cuda.Malloc(byteSize)
	if err != nil {
		return nil, err
	}

	defer func() { _ = cuda.Free(devOut) }()

	aF32 := *(*[]float32)(unsafe.Pointer(&aData))

	if err := cuda.Memcpy(devIn, unsafe.Pointer(&aF32[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		return nil, err
	}

	if err := kernels.Softmax(devIn, devOut, outer, inner, axisSize); err != nil {
		return nil, err
	}

	resultF32 := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&resultF32[0]), devOut, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		return nil, err
	}

	resultT := *(*[]T)(unsafe.Pointer(&resultF32))
	result.SetData(resultT)

	return result, nil
}

// sameShape checks if two tensors have the same shape (for non-broadcasting GPU path).
func sameShape[T tensor.Numeric](a, b *tensor.TensorNumeric[T]) bool {
	as := a.Shape()
	bs := b.Shape()

	if len(as) != len(bs) {
		return false
	}

	for i := range as {
		if as[i] != bs[i] {
			return false
		}
	}

	return true
}
