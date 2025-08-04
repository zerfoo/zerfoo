package compute

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"time"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// CPUEngine is a CPU-based implementation of the Engine interface.
type CPUEngine[T tensor.Numeric] struct {
	ops numeric.Arithmetic[T]
}

// NewCPUEngine creates a new CPUEngine.
func NewCPUEngine[T tensor.Numeric](ops numeric.Arithmetic[T]) *CPUEngine[T] {
	return &CPUEngine[T]{ops: ops}
}

// Ops returns the numeric.Arithmetic operations for the engine's numeric type.
func (e *CPUEngine[T]) Ops() numeric.Arithmetic[T] {
	return e.ops
}

func (e *CPUEngine[T]) getOrCreateDest(shape []int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(dst) > 0 && dst[0] != nil {
		if !reflect.DeepEqual(dst[0].Shape(), shape) {
			return nil, fmt.Errorf("destination tensor has incorrect shape: got %v, want %v", dst[0].Shape(), shape)
		}
		return dst[0], nil
	}
	return tensor.New[T](shape, nil)
}

// UnaryOp applies a unary operation to a tensor.
func (e *CPUEngine[T]) UnaryOp(_ context.Context, a *tensor.Tensor[T], op func(T) T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i, v := range a.Data() {
		result.Data()[i] = op(v)
	}
	return result, nil
}

// binaryOp performs element-wise binary operations with broadcasting support.
func (e *CPUEngine[T]) binaryOp(_ context.Context, a, b *tensor.Tensor[T], op func(T, T) T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}
	outputShape, broadcastA, broadcastB, err := tensor.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	result, err := e.getOrCreateDest(outputShape, dst...)
	if err != nil {
		return nil, err
	}
	aData := a.Data()
	bData := b.Data()
	rData := result.Data()
	for i := range rData {
		aIndex := tensor.BroadcastIndex(i, a.Shape(), outputShape, broadcastA)
		bIndex := tensor.BroadcastIndex(i, b.Shape(), outputShape, broadcastB)
		rData[i] = op(aData[aIndex], bData[bIndex])
	}
	return result, nil
}

// Add performs element-wise addition of two tensors.
func (e *CPUEngine[T]) Add(
	ctx context.Context,
	a, b *tensor.Tensor[T],
	dst ...*tensor.Tensor[T],
) (*tensor.Tensor[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Add, dst...)
}

// Sub performs element-wise subtraction of two tensors.
func (e *CPUEngine[T]) Sub(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Sub, dst...)
}

// Mul performs element-wise multiplication of two tensors.
func (e *CPUEngine[T]) Mul(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Mul, dst...)
}

// Div performs element-wise division of two tensors.
func (e *CPUEngine[T]) Div(_ context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}
	outputShape, broadcastA, broadcastB, err := tensor.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	result, err := e.getOrCreateDest(outputShape, dst...)
	if err != nil {
		return nil, err
	}
	aData := a.Data()
	bData := b.Data()
	rData := result.Data()
	for i := range rData {
		aIndex := tensor.BroadcastIndex(i, a.Shape(), outputShape, broadcastA)
		bIndex := tensor.BroadcastIndex(i, b.Shape(), outputShape, broadcastB)
		if e.ops.IsZero(bData[bIndex]) {
			return nil, fmt.Errorf("division by zero")
		}
		rData[i] = e.ops.Div(aData[aIndex], bData[bIndex])
	}
	return result, nil
}

// MatMul performs matrix multiplication of two tensors.
func (e *CPUEngine[T]) MatMul(_ context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}
	// Basic implementation for 2D matrices
	aShape := a.Shape()
	bShape := b.Shape()
	if len(aShape) != 2 || len(bShape) != 2 || aShape[1] != bShape[0] {
		return nil, fmt.Errorf("invalid shapes for matrix multiplication")
	}
	result, err := e.getOrCreateDest([]int{aShape[0], bShape[1]}, dst...)
	if err != nil {
		return nil, err
	}
	for i := 0; i < aShape[0]; i++ {
		for j := 0; j < bShape[1]; j++ {
			sum := e.ops.FromFloat64(0)
			for k := 0; k < aShape[1]; k++ {
				valA, _ := a.At(i, k)
				valB, _ := b.At(k, j)
				sum = e.ops.Add(sum, e.ops.Mul(valA, valB))
			}
			if err := result.Set(sum, i, j); err != nil {
				return nil, err
			}
		}
	}
	return result, nil
}

// Transpose transposes a 2D tensor.
func (e *CPUEngine[T]) Transpose(_ context.Context, a *tensor.Tensor[T], axes []int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	originalShape := a.Shape()
	if len(originalShape) != 2 {
		return nil, fmt.Errorf("transpose is only supported for 2D tensors, got %d dimensions", len(originalShape))
	}
	if len(axes) != len(originalShape) {
		return nil, fmt.Errorf("number of axes %d must match tensor dimensions %d", len(axes), len(originalShape))
	}

	newShape := make([]int, len(originalShape))
	for i, axis := range axes {
		if axis < 0 || axis >= len(originalShape) {
			return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, len(originalShape))
		}
		newShape[i] = originalShape[axis]
	}

	result, err := e.getOrCreateDest(newShape, dst...)
	if err != nil {
		return nil, err
	}

	// Create a mapping from new strides to old strides
	oldStrides := a.Strides()
	newStrides := result.Strides()

	// Iterate over all elements and copy them to the new positions
	for i := 0; i < a.Size(); i++ {
		oldCoords := make([]int, len(originalShape))
		linearIndex := i
		for j, stride := range oldStrides {
			oldCoords[j] = linearIndex / stride
			linearIndex %= stride
		}

		newCoords := make([]int, len(originalShape))
		for j, axis := range axes {
			newCoords[j] = oldCoords[axis]
		}

		newLinearIndex := 0
		for j, coord := range newCoords {
			newLinearIndex += coord * newStrides[j]
		}
		result.Data()[newLinearIndex] = a.Data()[i]
	}

	return result, nil
}

// Sum computes the sum of tensor elements along the specified axis.
// If keepDims is true, the reduced dimensions are retained with size 1.
// An optional destination tensor can be provided to store the result.
func (e *CPUEngine[T]) Sum(
	ctx context.Context,
	a *tensor.Tensor[T],
	axis int,
	keepDims bool,
	dst ...*tensor.Tensor[T],
) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}

	// A negative axis means sum over all axes.
	if axis < 0 {
		var sum T
		for _, v := range a.Data() {
			sum = e.ops.Add(sum, v)
		}
		shape := []int{1}
		if keepDims {
			shape = make([]int, a.Dims())
			for i := range shape {
				shape[i] = 1
			}
		}
		result, err := e.getOrCreateDest(shape, dst...)
		if err != nil {
			return nil, err
		}
		result.Data()[0] = sum
		return result, nil
	}

	shape := a.Shape()
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, len(shape))
	}

	newShape := make([]int, 0, len(shape))
	if keepDims {
		newShape = make([]int, len(shape))
		for i, dim := range shape {
			if i == axis {
				newShape[i] = 1
			} else {
				newShape[i] = dim
			}
		}
	} else {
		for i, dim := range shape {
			if i != axis {
				newShape = append(newShape, dim)
			}
		}
		if len(newShape) == 0 {
			newShape = []int{1}
		}
	}

	result, err := e.getOrCreateDest(newShape, dst...)
	if err != nil {
		return nil, err
	}
	if err := e.Zero(ctx, result); err != nil {
		return nil, err
	}

	aData := a.Data()
	rData := result.Data()
	aStrides := a.Strides()
	rStrides := result.Strides()

	for i, v := range aData {
		rIndex := 0
		temp := i
		for j, stride := range aStrides {
			coord := temp / stride
			temp %= stride
			if j < axis {
				rIndex += coord * rStrides[j]
			} else if j > axis {
				if keepDims {
					rIndex += coord * rStrides[j]
				} else {
					rIndex += coord * rStrides[j-1]
				}
			}
		}
		rData[rIndex] = e.ops.Add(rData[rIndex], v)
	}

	return result, nil
}

// Exp computes the element-wise exponential of a tensor.
func (e *CPUEngine[T]) Exp(_ context.Context, a *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i, v := range a.Data() {
		result.Data()[i] = e.ops.Exp(v)
	}
	return result, nil
}

// Log computes the element-wise natural logarithm of a tensor.
func (e *CPUEngine[T]) Log(_ context.Context, a *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i, v := range a.Data() {
		result.Data()[i] = e.ops.Log(v)
	}
	return result, nil
}

// Pow raises each element of a tensor to the power of the corresponding element in another tensor.
func (e *CPUEngine[T]) Pow(
	ctx context.Context,
	base, exponent *tensor.Tensor[T],
	dst ...*tensor.Tensor[T],
) (*tensor.Tensor[T], error) {
	return e.binaryOp(ctx, base, exponent, e.ops.Pow, dst...)
}

// Zero sets all elements of a tensor to zero.
func (e *CPUEngine[T]) Zero(_ context.Context, a *tensor.Tensor[T]) error {
	if a == nil {
		return fmt.Errorf("input tensor cannot be nil")
	}
	for i := range a.Data() {
		a.Data()[i] = e.ops.FromFloat64(0)
	}
	return nil
}

// Zeros fills the tensor with zeros. If a shape is provided, the tensor is reallocated to that shape.
func (e *CPUEngine[T]) Zeros(_ context.Context, a *tensor.Tensor[T], shape []int) error {
	if a == nil {
		return fmt.Errorf("input tensor cannot be nil")
	}

	if shape != nil && !reflect.DeepEqual(a.Shape(), shape) {
		// Reallocate the underlying data slice and update shape/strides
		newTensor, err := tensor.New[T](shape, nil)
		if err != nil {
			return err
		}
		a.SetData(newTensor.Data())
		a.SetShape(newTensor.Shape())
		a.SetStrides(newTensor.Strides())
	}

	for i := range a.Data() {
		a.Data()[i] = e.ops.FromFloat64(0)
	}
	return nil
}

// Copy copies the data from one tensor to another.
func (e *CPUEngine[T]) Copy(_ context.Context, dst, src *tensor.Tensor[T]) error {
	if dst == nil || src == nil {
		return fmt.Errorf("input tensors cannot be nil")
	}
	if !reflect.DeepEqual(dst.Shape(), src.Shape()) {
		return fmt.Errorf("tensors must have the same shape")
	}
	copy(dst.Data(), src.Data())
	return nil
}

// Gather performs a gather operation.
func (e *CPUEngine[T]) Gather(_ context.Context, params *tensor.Tensor[T], indices *tensor.Tensor[int], output *tensor.Tensor[T]) error {
	if params == nil || indices == nil || output == nil {
		return fmt.Errorf("input tensors cannot be nil")
	}

	if len(params.Shape()) != 2 {
		return fmt.Errorf("params must be a 2D tensor for Gather operation")
	}
	if len(indices.Shape()) != 2 {
		return fmt.Errorf("indices must be a 2D tensor for Gather operation")
	}

	vocabSize := params.Shape()[0]
	embeddingDim := params.Shape()[1]
	numIndices := indices.Shape()[1]

	expectedOutputShape := []int{indices.Shape()[0], numIndices, embeddingDim}
	if !reflect.DeepEqual(output.Shape(), expectedOutputShape) {
		return fmt.Errorf("output tensor has incorrect shape: got %v, want %v", output.Shape(), expectedOutputShape)
	}

	paramsData := params.Data()
	indicesData := indices.Data()
	outputData := output.Data()

	for i := 0; i < numIndices; i++ {
		idx := indicesData[i]
		if idx < 0 || idx >= vocabSize {
			return fmt.Errorf("index %d out of bounds for vocabulary size %d", idx, vocabSize)
		}
		// Copy the embedding vector for the current index
		copy(outputData[i*embeddingDim:(i+1)*embeddingDim], paramsData[idx*embeddingDim:(idx+1)*embeddingDim])
	}
	return nil
}

// ScatterAdd performs a scatter-add operation.
func (e *CPUEngine[T]) ScatterAdd(_ context.Context, dEmbeddingTable *tensor.Tensor[T], indices *tensor.Tensor[int], dOut *tensor.Tensor[T]) error {
	if dEmbeddingTable == nil || indices == nil || dOut == nil {
		return fmt.Errorf("input tensors cannot be nil")
	}

	if len(dEmbeddingTable.Shape()) != 2 {
		return fmt.Errorf("dEmbeddingTable must be a 2D tensor for ScatterAdd operation")
	}
	if len(indices.Shape()) != 2 {
		return fmt.Errorf("indices must be a 2D tensor for ScatterAdd operation")
	}

	vocabSize := dEmbeddingTable.Shape()[0]
	embeddingDim := dEmbeddingTable.Shape()[1]
	numIndices := indices.Shape()[1]

	// Ensure dEmbeddingTable is zeroed out before accumulation
	for i := range dEmbeddingTable.Data() {
		dEmbeddingTable.Data()[i] = e.ops.FromFloat64(0)
	}

	indicesData := indices.Data()
	dOutData := dOut.Data()
	dEmbeddingTableData := dEmbeddingTable.Data()

	for i := 0; i < numIndices; i++ {
		idx := indicesData[i]
		if idx < 0 || idx >= vocabSize {
			return fmt.Errorf("index %d out of bounds for vocabulary size %d", idx, vocabSize)
		}
		for j := 0; j < embeddingDim; j++ {
			currentVal := dEmbeddingTableData[idx*embeddingDim+j]
			gradVal := dOutData[i*embeddingDim+j]
			dEmbeddingTableData[idx*embeddingDim+j] = e.ops.Add(currentVal, gradVal)
		}
	}
	return nil
}

// RandomUniform fills the tensor with random values from a uniform distribution.
func (e *CPUEngine[T]) RandomUniform(_ context.Context, t *tensor.Tensor[T], minVal, maxVal T) error {
	if t == nil {
		return fmt.Errorf("input tensor cannot be nil")
	}

	// #nosec G404 - Using math/rand for ML weight initialization is acceptable
	// as cryptographic security is not required for neural network weight sampling
	src := rand.NewSource(time.Now().UnixNano())
	r := rand.New(src) //nolint:gosec

	data := t.Data()
	for i := range data {
		// Generate a random float64 between 0.0 and 1.0
		randFloat := r.Float64()

		// Scale and shift to the desired range [minVal, maxVal]
		scaledValue := float64(e.ops.Sub(maxVal, minVal))*randFloat + float64(minVal)

		// Convert back to type T
		data[i] = e.ops.FromFloat64(scaledValue)
	}
	return nil
}

// Fill fills the tensor with a scalar value.
func (e *CPUEngine[T]) Fill(_ context.Context, t *tensor.Tensor[T], value T) error {
	if t == nil {
		return fmt.Errorf("input tensor cannot be nil")
	}
	for i := range t.Data() {
		t.Data()[i] = value
	}
	return nil
}

// MulScalar performs element-wise multiplication of a tensor by a scalar.
func (e *CPUEngine[T]) MulScalar(_ context.Context, a *tensor.Tensor[T], scalar T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i, v := range a.Data() {
		result.Data()[i] = e.ops.Mul(v, scalar)
	}
	return result, nil
}

// DivScalar performs element-wise division of a tensor by a scalar.
func (e *CPUEngine[T]) DivScalar(_ context.Context, a *tensor.Tensor[T], scalar T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	if e.ops.IsZero(scalar) {
		return nil, fmt.Errorf("division by zero")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i, v := range a.Data() {
		result.Data()[i] = e.ops.Div(v, scalar)
	}
	return result, nil
}

// Softmax applies the softmax function to a tensor along a given axis.
func (e *CPUEngine[T]) Softmax(_ context.Context, a *tensor.Tensor[T], axis int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	shape := a.Shape()
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, len(shape))
	}

	result, err := e.getOrCreateDest(shape, dst...)
	if err != nil {
		return nil, err
	}

	// Calculate max for numerical stability
	maxVal := e.ops.FromFloat64(-math.MaxFloat64)
	for _, v := range a.Data() {
		if e.ops.GreaterThan(v, maxVal) {
			maxVal = v
		}
	}

	// Calculate exponentials and sum
	expSum := e.ops.FromFloat64(0)
	expData := make([]T, len(a.Data()))
	for i, v := range a.Data() {
		expVal := e.ops.Exp(e.ops.Sub(v, maxVal))
		expData[i] = expVal
		expSum = e.ops.Add(expSum, expVal)
	}

	// Normalize
	for i := range expData {
		result.Data()[i] = e.ops.Div(expData[i], expSum)
	}

	return result, nil
}

// ReduceSum calculates the sum of elements along a specified axis.
func (e *CPUEngine[T]) ReduceSum(
	ctx context.Context,
	a *tensor.Tensor[T],
	axis int,
	keepDims bool,
	dst ...*tensor.Tensor[T],
) (*tensor.Tensor[T], error) {
	return e.Sum(ctx, a, axis, keepDims, dst...)
}

// AddScalar performs element-wise addition of a tensor by a scalar.
func (e *CPUEngine[T]) AddScalar(_ context.Context, a *tensor.Tensor[T], scalar T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i, v := range a.Data() {
		result.Data()[i] = e.ops.Add(v, scalar)
	}
	return result, nil
}

// Sqrt computes the element-wise square root of a tensor.
func (e *CPUEngine[T]) Sqrt(_ context.Context, a *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i, v := range a.Data() {
		result.Data()[i] = e.ops.Sqrt(v)
	}
	return result, nil
}

// Split splits a tensor into multiple tensors along a given axis.
func (e *CPUEngine[T]) Split(_ context.Context, a *tensor.Tensor[T], numSplits int, axis int) ([]*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	shape := a.Shape()
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, len(shape))
	}
	if shape[axis]%numSplits != 0 {
		return nil, fmt.Errorf("dimension %d (size %d) is not divisible by numSplits %d", axis, shape[axis], numSplits)
	}

	splitSize := shape[axis] / numSplits
	var results []*tensor.Tensor[T]

	for i := 0; i < numSplits; i++ {
		ranges := make([][2]int, len(shape))
		for j := range shape {
			if j == axis {
				ranges[j] = [2]int{i * splitSize, (i + 1) * splitSize}
			} else {
				ranges[j] = [2]int{0, shape[j]}
			}
		}

		slice, err := a.Slice(ranges...)
		if err != nil {
			return nil, err
		}
		results = append(results, slice)
	}
	return results, nil
}

// Concat concatenates a list of tensors along a given axis.
func (e *CPUEngine[T]) Concat(_ context.Context, tensors []*tensor.Tensor[T], axis int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("no tensors provided for concatenation")
	}

	firstTensor := tensors[0]
	firstShape := firstTensor.Shape()
	if axis < 0 || axis >= len(firstShape) {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, len(firstShape))
	}

	// Calculate new shape and total size
	newShape := make([]int, len(firstShape))
	copy(newShape, firstShape)
	newShape[axis] = 0 // Sum of sizes along the concatenation axis

	for _, t := range tensors {
		currentShape := t.Shape()
		if len(currentShape) != len(firstShape) {
			return nil, fmt.Errorf("tensors must have the same number of dimensions for concatenation")
		}
		for i, dim := range currentShape {
			if i == axis {
				newShape[axis] += dim
			} else if dim != firstShape[i] {
				return nil, fmt.Errorf("dimensions must be equal except for the concatenation axis")
			}
		}
	}

	result, err := e.getOrCreateDest(newShape, dst...)
	if err != nil {
		return nil, err
	}

	currentOffset := 0
	for _, t := range tensors {
		// Calculate the size of the current tensor's slice along the concatenation axis
		sliceSize := 1
		for i, dim := range t.Shape() {
			if i != axis {
				sliceSize *= dim
			}
		}
		sliceSize *= t.Shape()[axis] // Total elements in the current tensor

		copy(result.Data()[currentOffset:currentOffset+sliceSize], t.Data())
		currentOffset += sliceSize
	}

	return result, nil
}

// OneHot creates a one-hot encoding of the input tensor.
func (e *CPUEngine[T]) OneHot(_ context.Context, input *tensor.Tensor[int], depth int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if input == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	if depth <= 0 {
		return nil, fmt.Errorf("depth must be positive")
	}

	outputShape := append(input.Shape(), depth)
	result, err := e.getOrCreateDest(outputShape, dst...)
	if err != nil {
		return nil, err
	}

	outputData := result.Data()
	inputData := input.Data()
	outputSize := result.Size()

	// Initialize all elements to zero
	for i := 0; i < outputSize; i++ {
		outputData[i] = e.ops.FromFloat64(0)
	}

	// Set the appropriate index to one
	for i, val := range inputData {
		if val < 0 || val >= depth {
			return nil, fmt.Errorf("index %d out of bounds for depth %d", val, depth)
		}
		// Calculate the base index for the current one-hot vector
		baseIndex := i * depth
		outputData[baseIndex+val] = e.ops.FromFloat64(1)
	}

	return result, nil
}

// Reshape changes the shape of a tensor without changing its data.
func (e *CPUEngine[T]) Reshape(_ context.Context, a *tensor.Tensor[T], shape []int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}

	// Check if the new shape is compatible with the existing data size
	currentSize := 1
	for _, dim := range a.Shape() {
		currentSize *= dim
	}
	newSize := 1
	for _, dim := range shape {
		newSize *= dim
	}

	if currentSize != newSize {
		return nil, fmt.Errorf("new shape %v is not compatible with current tensor size %d", shape, currentSize)
	}

	result, err := e.getOrCreateDest(shape, dst...)
	if err != nil {
		return nil, err
	}

	// If the destination is a new tensor, copy the data.
	// If it's the same tensor, its shape and strides will be updated by getOrCreateDest.
	if result != a {
		copy(result.Data(), a.Data())
	}

	return result, nil
}

// Repeat repeats the input tensor along a given axis a specified number of times.
func (e *CPUEngine[T]) Repeat(_ context.Context, a *tensor.Tensor[T], axis int, repetitions int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	shape := a.Shape()
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, len(shape))
	}
	if repetitions <= 0 {
		return nil, fmt.Errorf("repetitions must be positive")
	}

	newShape := make([]int, len(shape))
	copy(newShape, shape)
	newShape[axis] *= repetitions

	result, err := e.getOrCreateDest(newShape, dst...)
	if err != nil {
		return nil, err
	}

	// Calculate block size and number of blocks
	blockSize := 1
	for i := axis + 1; i < len(shape); i++ {
		blockSize *= shape[i]
	}
	numBlocks := a.Size() / (blockSize * shape[axis])

	// Fill the result tensor
	for i := 0; i < numBlocks; i++ {
		for r := 0; r < repetitions; r++ {
			for j := 0; j < shape[axis]; j++ {
				srcStart := i*shape[axis]*blockSize + j*blockSize
				dstStart := i*shape[axis]*blockSize*repetitions + r*shape[axis]*blockSize + j*blockSize
				copy(result.Data()[dstStart:dstStart+blockSize], a.Data()[srcStart:srcStart+blockSize])
			}
		}
	}

	return result, nil
}
