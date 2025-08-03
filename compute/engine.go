package compute

import (
	"context"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Engine defines the interface for a computation engine (e.g., CPU, GPU).
// All tensor operations should be routed through an Engine implementation to ensure
// hardware interoperability and optimized performance.
type Engine[T tensor.Numeric] interface {
	// Ops returns the numeric.Arithmetic operations for the engine's numeric type.
	Ops() numeric.Arithmetic[T]
	// UnaryOp applies a unary function `op` to each element of tensor `a`.
	// It returns a new tensor with the results.
	// Returns an error if the input tensor is nil.
	UnaryOp(ctx context.Context, a *tensor.Tensor[T], op func(T) T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Add performs element-wise addition of two tensors, with support for broadcasting.
	// It returns a new tensor with the results.
	// Returns an error if tensors are nil or their shapes are not compatible for broadcasting.
	Add(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Sub performs element-wise subtraction of two tensors, with support for broadcasting.
	// It returns a new tensor with the results.
	// Returns an error if tensors are nil or their shapes are not compatible for broadcasting.
	Sub(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Mul performs element-wise multiplication of two tensors, with support for broadcasting.
	// It returns a new tensor with the results.
	// Returns an error if tensors are nil or their shapes are not compatible for broadcasting.
	Mul(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Div performs element-wise division of two tensors, with support for broadcasting.
	// It returns a new tensor with the results.
	// Returns an error if tensors are nil or their shapes are not compatible for broadcasting.
	Div(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// MatMul performs matrix multiplication of two 2D tensors.
	// It returns a new tensor with the result.
	// Returns an error if the tensors are nil, not 2D, or their shapes are incompatible for matrix multiplication.
	MatMul(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Transpose transposes a tensor along the given axes.
	// It returns a new tensor with the result.
	// Returns an error if the tensor is nil or the axes are invalid.
	Transpose(ctx context.Context, a *tensor.Tensor[T], axes []int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Sum calculates the sum of elements along a specified axis.
	// A negative axis means summing along all axes, returning a scalar tensor.
	// If keepDims is true, the reduced dimensions are retained with size 1.
	// Returns a new tensor with the reduced shape.
	// Returns an error if the tensor is nil or the axis is out of bounds.
	Sum(ctx context.Context, a *tensor.Tensor[T], axis int, keepDims bool, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Exp computes the element-wise exponential of a tensor.
	Exp(ctx context.Context, a *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Log computes the element-wise natural logarithm of a tensor.
	Log(ctx context.Context, a *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Pow raises each element of a tensor to the power of the corresponding element in another tensor.
	Pow(ctx context.Context, base, exponent *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Zero sets all elements of a tensor to zero.
	Zero(ctx context.Context, a *tensor.Tensor[T]) error

	// Zeros fills the tensor with zeros. If a shape is provided, the tensor is reallocated to that shape.
	Zeros(ctx context.Context, a *tensor.Tensor[T], shape []int) error

	// Copy copies the data from one tensor to another.
	Copy(ctx context.Context, dst, src *tensor.Tensor[T]) error

	// Gather performs a gather operation.
	// output[i] = params[indices[i]]
	Gather(ctx context.Context, params *tensor.Tensor[T], indices *tensor.Tensor[int], output *tensor.Tensor[T]) error

	// ScatterAdd performs a scatter-add operation.
	// dEmbeddingTable[indices[i]] += dOut[i]
	ScatterAdd(ctx context.Context, dEmbeddingTable *tensor.Tensor[T], indices *tensor.Tensor[int], dOut *tensor.Tensor[T]) error

	// RandomUniform fills the tensor with random values from a uniform distribution.
	RandomUniform(ctx context.Context, t *tensor.Tensor[T], minVal, maxVal T) error

	// Fill fills the tensor with a scalar value.
	Fill(ctx context.Context, t *tensor.Tensor[T], value T) error

	// MulScalar performs element-wise multiplication of a tensor by a scalar.
	MulScalar(ctx context.Context, a *tensor.Tensor[T], scalar T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// DivScalar performs element-wise division of a tensor by a scalar.
	DivScalar(ctx context.Context, a *tensor.Tensor[T], scalar T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Softmax applies the softmax function to a tensor along a given axis.
	Softmax(ctx context.Context, a *tensor.Tensor[T], axis int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// ReduceSum calculates the sum of elements along a specified axis, similar to Sum but potentially with different
	// internal handling or optimizations for reduction operations.
	ReduceSum(ctx context.Context, a *tensor.Tensor[T], axis int, keepDims bool, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// AddScalar performs element-wise addition of a tensor by a scalar.
	AddScalar(ctx context.Context, a *tensor.Tensor[T], scalar T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Sqrt computes the element-wise square root of a tensor.
	Sqrt(ctx context.Context, a *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Split splits a tensor into multiple tensors along a given axis.
	Split(ctx context.Context, a *tensor.Tensor[T], numSplits int, axis int) ([]*tensor.Tensor[T], error)

	// Concat concatenates a list of tensors along a given axis.
	Concat(ctx context.Context, tensors []*tensor.Tensor[T], axis int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Repeat repeats the input tensor along a given axis a specified number of times.
	Repeat(ctx context.Context, a *tensor.Tensor[T], axis int, repetitions int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// OneHot creates a one-hot encoding of the input tensor.
	OneHot(ctx context.Context, input *tensor.Tensor[int], depth int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)

	// Reshape changes the shape of a tensor without changing its data.
	Reshape(ctx context.Context, a *tensor.Tensor[T], shape []int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)
}