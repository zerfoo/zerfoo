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
	UnaryOp(ctx context.Context, a *tensor.TensorNumeric[T], op func(T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Add performs element-wise addition of two tensors, with support for broadcasting.
	// It returns a new tensor with the results.
	// Returns an error if tensors are nil or their shapes are not compatible for broadcasting.
	Add(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Sub performs element-wise subtraction of two tensors, with support for broadcasting.
	// It returns a new tensor with the results.
	// Returns an error if tensors are nil or their shapes are not compatible for broadcasting.
	Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Mul performs element-wise multiplication of two tensors, with support for broadcasting.
	// It returns a new tensor with the results.
	// Returns an error if tensors are nil or their shapes are not compatible for broadcasting.
	Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Div performs element-wise division of two tensors, with support for broadcasting.
	// It returns a new tensor with the results.
	// Returns an error if tensors are nil or their shapes are not compatible for broadcasting.
	Div(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// MatMul performs matrix multiplication of two 2D tensors.
	// It returns a new tensor with the result.
	// Returns an error if the tensors are nil, not 2D, or their shapes are incompatible for matrix multiplication.
	MatMul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Transpose transposes a tensor along the given axes.
	// It returns a new tensor with the result.
	// Returns an error if the tensor is nil or the axes are invalid.
	Transpose(ctx context.Context, a *tensor.TensorNumeric[T], axes []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Sum calculates the sum of elements along a specified axis.
	// A negative axis means summing along all axes, returning a scalar tensor.
	// If keepDims is true, the reduced dimensions are retained with size 1.
	// Returns a new tensor with the reduced shape.
	// Returns an error if the tensor is nil or the axis is out of bounds.
	Sum(
		ctx context.Context,
		a *tensor.TensorNumeric[T],
		axis int,
		keepDims bool,
		dst ...*tensor.TensorNumeric[T],
	) (*tensor.TensorNumeric[T], error)

	// Exp computes the element-wise exponential of a tensor.
	Exp(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Log computes the element-wise natural logarithm of a tensor.
	Log(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Pow raises each element of a tensor to the power of the corresponding element in another tensor.
	Pow(ctx context.Context, base, exponent *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Zero sets all elements of a tensor to zero.
	Zero(ctx context.Context, a *tensor.TensorNumeric[T]) error

	// Zeros fills the tensor with zeros. If a shape is provided, the tensor is reallocated to that shape.
	Zeros(ctx context.Context, a *tensor.TensorNumeric[T], shape []int) error

	// Copy copies the data from one tensor to another.
	Copy(ctx context.Context, dst, src *tensor.TensorNumeric[T]) error

	// Gather performs an embedding-style gather.
	// params must be 2D [vocab, dim].
	// indices may be 1D [N] or 2D [batch, seq].
	// output must be [indices..., dim], i.e., [N, dim] or [batch, seq, dim].
	Gather(ctx context.Context, params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T]) error

	// ScatterAdd performs a row-wise scatter-add for embeddings.
	// dEmbeddingTable must be [vocab, dim].
	// indices may be 1D [N] or multi-dim with flattened length N.
	// dOut must be [N, dim].
	// For each i in [0..N), it applies: dEmbeddingTable[indices[i], :] += dOut[i, :].
	ScatterAdd(ctx context.Context, dEmbeddingTable *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[T]) error

	// RandomUniform fills the tensor with random values from a uniform distribution.
	RandomUniform(ctx context.Context, t *tensor.TensorNumeric[T], minVal, maxVal T) error

	// Fill fills the tensor with a scalar value.
	Fill(ctx context.Context, t *tensor.TensorNumeric[T], value T) error

	// MulScalar performs element-wise multiplication of a tensor by a scalar.
	MulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// DivScalar performs element-wise division of a tensor by a scalar.
	DivScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Softmax applies the softmax function to a tensor along a given axis.
	Softmax(ctx context.Context, a *tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// ReduceSum calculates the sum of elements along a specified axis, similar to Sum but potentially with different
	// internal handling or optimizations for reduction operations.
	ReduceSum(
		ctx context.Context,
		a *tensor.TensorNumeric[T],
		axis int,
		keepDims bool,
		dst ...*tensor.TensorNumeric[T],
	) (*tensor.TensorNumeric[T], error)

	// AddScalar performs element-wise addition of a tensor by a scalar.
	AddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Sqrt computes the element-wise square root of a tensor.
	Sqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Split splits a tensor into multiple tensors along a given axis.
	Split(ctx context.Context, a *tensor.TensorNumeric[T], numSplits int, axis int) ([]*tensor.TensorNumeric[T], error)

	// Concat concatenates a list of tensors along a given axis.
	Concat(ctx context.Context, tensors []*tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Repeat repeats the input tensor along a given axis a specified number of times.
	Repeat(
		ctx context.Context,
		a *tensor.TensorNumeric[T],
		axis int,
		repetitions int,
		dst ...*tensor.TensorNumeric[T],
	) (*tensor.TensorNumeric[T], error)

	// OneHot creates a one-hot encoding of the input tensor.
	OneHot(ctx context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// Reshape changes the shape of a tensor without changing its data.
	Reshape(ctx context.Context, a *tensor.TensorNumeric[T], shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

	// ReduceMean calculates the mean of elements along a specified axis.
	ReduceMean(
		ctx context.Context,
		a *tensor.TensorNumeric[T],
		axis int,
		keepDims bool,
		dst ...*tensor.TensorNumeric[T],
	) (*tensor.TensorNumeric[T], error)

	// Rsqrt computes the element-wise reciprocal square root of a tensor.
	Rsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
}
