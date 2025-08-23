// Package compute implements tensor computation engines and operations.
package compute

import (
	"context"
	"errors"
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

// ReduceSum calculates the sum along a specified axis. It delegates to Sum.
func (e *CPUEngine[T]) ReduceSum(
    ctx context.Context,
    a *tensor.TensorNumeric[T],
    axis int,
    keepDims bool,
    dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
    return e.Sum(ctx, a, axis, keepDims, dst...)
}

// Softmax applies the softmax function along the specified axis.
// Implementation: exp(x) then divide by sum(exp(x)) along axis using broadcasting.
func (e *CPUEngine[T]) Softmax(
    ctx context.Context,
    a *tensor.TensorNumeric[T],
    axis int,
    dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
    if a == nil {
        return nil, errors.New("input tensor cannot be nil")
    }

    // Compute exponentials
    exps, err := e.Exp(ctx, a)
    if err != nil {
        return nil, err
    }

    // Sum along axis, keeping dimensions for broadcasting
    sums, err := e.Sum(ctx, exps, axis, true)
    if err != nil {
        return nil, err
    }

    // Divide exps by sums with broadcasting
    return e.Div(ctx, exps, sums, dst...)
}

// AddScalar performs element-wise addition of a tensor by a scalar.
func (e *CPUEngine[T]) AddScalar(_ context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	aData := a.Data()
	rData := result.Data()
	for i := 0; i < len(aData); i++ { //nolint:intrange // Classic index loop maintained
		rData[i] = e.ops.Add(aData[i], scalar)
	}

	return result, nil
}

// MulScalar performs element-wise multiplication of a tensor by a scalar.
func (e *CPUEngine[T]) MulScalar(_ context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	aData := a.Data()
	rData := result.Data()
	for i := 0; i < len(aData); i++ { //nolint:intrange // Classic index loop maintained
		rData[i] = e.ops.Mul(aData[i], scalar)
	}

	return result, nil
}

// Sqrt computes the element-wise square root of a tensor.
func (e *CPUEngine[T]) Sqrt(_ context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	aData := a.Data()
	rData := result.Data()
	for i := 0; i < len(aData); i++ { //nolint:intrange // Classic index loop maintained
		rData[i] = e.ops.FromFloat64(math.Sqrt(float64(aData[i])))
	}

	return result, nil
}

// Split splits a tensor into multiple tensors along a given axis.
func (e *CPUEngine[T]) Split(_ context.Context, a *tensor.TensorNumeric[T], numSplits int, axis int) ([]*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	shape := a.Shape()
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, len(shape))
	}
	if numSplits <= 0 || shape[axis]%numSplits != 0 {
		return nil, fmt.Errorf("cannot split dimension %d of size %d into %d parts", axis, shape[axis], numSplits)
	}
	part := shape[axis] / numSplits
	outShape := make([]int, len(shape))
	copy(outShape, shape)
	outShape[axis] = part

	// Precompute strides and block sizes
	strides := a.Strides()
	blockSize := 1
	for i := axis + 1; i < len(shape); i++ {
		blockSize *= shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ { //nolint:intrange // Classic index loop maintained
		outer *= shape[i]
	}

	results := make([]*tensor.TensorNumeric[T], numSplits)
	for s := 0; s < numSplits; s++ { //nolint:intrange // Classic index loop maintained
		t, err := tensor.New[T](outShape, nil)
		if err != nil {
			return nil, err
		}
		results[s] = t
	}

	// Copy slices
	data := a.Data()
	for o := 0; o < outer; o++ { //nolint:intrange // Classic index loop maintained
		for s := 0; s < numSplits; s++ { //nolint:intrange // Classic index loop maintained
			for j := 0; j < part; j++ { //nolint:intrange // Classic index loop maintained
				srcStart := o*shape[axis]*blockSize + (s*part+j)*blockSize
				dstStart := o*part*blockSize + j*blockSize
				copy(results[s].Data()[dstStart:dstStart+blockSize], data[srcStart:srcStart+blockSize])
			}
		}
	}

	_ = strides // kept for potential future optimization

	return results, nil
}

// Gather performs an embedding-style gather.
// params is expected to be [vocab, dim].
// indices may be 1D [N] or 2D [batch, seq].
// output must be [indices..., dim].
func (e *CPUEngine[T]) Gather(_ context.Context, params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T]) error {
    if params == nil || indices == nil || output == nil {
        return errors.New("params, indices, and output cannot be nil")
    }

    pShape := params.Shape()
    if len(pShape) != 2 {
        return fmt.Errorf("params must be 2D [vocab, dim], got %v", pShape)
    }
    vocab := pShape[0]
    dim := pShape[1]

    // Validate indices dims (allow 1D or 2D)
    iShape := indices.Shape()
    if len(iShape) != 1 && len(iShape) != 2 {
        return fmt.Errorf("indices must be 1D or 2D, got %dD shape %v", len(iShape), iShape)
    }

    // Expected output shape is indices shape with an extra trailing dim
    expectedOutShape := append(append([]int{}, iShape...), dim)
    if !reflect.DeepEqual(output.Shape(), expectedOutShape) {
        return fmt.Errorf("output shape %v must equal indices shape appended with dim %v; want %v", output.Shape(), dim, expectedOutShape)
    }

    idx := indices.Data()
    out := output.Data()

    // Number of positions in indices (flattened)
    n := 1
    for _, d := range iShape {
        n *= d
    }
    if n != len(idx) {
        return fmt.Errorf("indices data size mismatch: shape %v -> %d, but data length is %d", iShape, n, len(idx))
    }

    // For each index, copy the corresponding row vector of length dim
    pData := params.Data()
    for i := 0; i < n; i++ { //nolint:intrange // Classic index loop maintained
        row := idx[i]
        if row < 0 || row >= vocab {
            return fmt.Errorf("gather index %d out of bounds [0,%d)", row, vocab)
        }
        srcStart := row * dim
        dstStart := i * dim
        copy(out[dstStart:dstStart+dim], pData[srcStart:srcStart+dim])
    }

    return nil
}

// ScatterAdd performs a scatter-add over rows: dEmbeddingTable[indices[i], :] += dOut[i, :].
// dEmbeddingTable must be [vocab, dim]. dOut must be [N, dim]. indices can be [N] or [1, N] or [batch, seq] with N=batch*seq.
func (e *CPUEngine[T]) ScatterAdd(_ context.Context, dEmbeddingTable *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[T]) error {
    if dEmbeddingTable == nil || indices == nil || dOut == nil {
        return errors.New("inputs cannot be nil")
    }

    tShape := dEmbeddingTable.Shape()
    if len(tShape) != 2 {
        return fmt.Errorf("dEmbeddingTable must be 2D [vocab, dim], got %v", tShape)
    }
    vocab := tShape[0]
    dim := tShape[1]

    // Validate dOut shape [N, dim]
    gShape := dOut.Shape()
    if len(gShape) != 2 || gShape[1] != dim {
        return fmt.Errorf("dOut must be 2D [N, %d], got %v", dim, gShape)
    }
    n := gShape[0]

    // Flatten indices length and verify equals N
    iShape := indices.Shape()
    if len(iShape) == 0 {
        return fmt.Errorf("indices must have at least 1 dimension")
    }
    idxCount := 1
    for _, d := range iShape {
        idxCount *= d
    }
    if idxCount != n {
        return fmt.Errorf("indices flattened length %d must equal dOut rows %d", idxCount, n)
    }

    table := dEmbeddingTable.Data()
    idx := indices.Data()
    grad := dOut.Data()

    for i := 0; i < n; i++ { //nolint:intrange // Classic index loop maintained
        row := idx[i]
        if row < 0 || row >= vocab {
            return fmt.Errorf("scatter index %d out of bounds [0,%d)", row, vocab)
        }
        tStart := row * dim
        gStart := i * dim
        for j := 0; j < dim; j++ { //nolint:intrange // Classic index loop maintained
            table[tStart+j] = e.ops.Add(table[tStart+j], grad[gStart+j])
        }
    }

    return nil
}

// RandomUniform fills the tensor with values from a uniform distribution in [minVal, maxVal).
func (e *CPUEngine[T]) RandomUniform(_ context.Context, t *tensor.TensorNumeric[T], minVal, maxVal T) error {
	if t == nil {
		return errors.New("input tensor cannot be nil")
	}
	//nolint:gosec // G404: math/rand is acceptable for non-cryptographic ML initialization
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	minF := float64(minVal)
	maxF := float64(maxVal)
	if maxF < minF {
		return errors.New("maxVal must be >= minVal")
	}
	span := maxF - minF
	data := t.Data()
	for i := 0; i < len(data); i++ {
		v := minF + rng.Float64()*span
		data[i] = e.ops.FromFloat64(v)
	}

	return nil
}

// Fill sets all elements of the tensor to a scalar value.
func (e *CPUEngine[T]) Fill(_ context.Context, t *tensor.TensorNumeric[T], value T) error {
	if t == nil {
		return errors.New("input tensor cannot be nil")
	}
	data := t.Data()
	for i := 0; i < len(data); i++ {
		data[i] = value
	}

	return nil
}

// Zero sets all elements of a tensor to zero.
func (e *CPUEngine[T]) Zero(_ context.Context, a *tensor.TensorNumeric[T]) error {
	if a == nil {
		return errors.New("input tensor cannot be nil")
	}
	zero := e.ops.FromFloat64(0)
	data := a.Data()
	for i := 0; i < len(data); i++ {
		data[i] = zero
	}

	return nil
}

// Zeros fills the tensor with zeros. If a shape is provided, the tensor is reallocated to that shape.
func (e *CPUEngine[T]) Zeros(_ context.Context, a *tensor.TensorNumeric[T], shape []int) error {
	if a == nil {
		return errors.New("input tensor cannot be nil")
	}
	if shape != nil {
		// Reallocate to the new shape
		size := 1
		for _, d := range shape {
			if d < 0 {
				return fmt.Errorf("invalid shape dimension: %d; must be non-negative", d)
			}
			size *= d
		}
		data := make([]T, size)
		// compute strides
		strides := make([]int, len(shape))
		stride := 1
		for i := len(shape) - 1; i >= 0; i-- {
			strides[i] = stride
			stride *= shape[i]
		}
		a.SetShape(shape)
		a.SetStrides(strides)
		a.SetData(data)
	}

	return e.Zero(context.Background(), a)
}

// Copy copies the data from one tensor to another.
func (e *CPUEngine[T]) Copy(_ context.Context, dst, src *tensor.TensorNumeric[T]) error {
	if dst == nil || src == nil {
		return errors.New("tensors cannot be nil")
	}
	if !reflect.DeepEqual(dst.Shape(), src.Shape()) {
		return fmt.Errorf("shape mismatch: dst %v vs src %v", dst.Shape(), src.Shape())
	}
	copy(dst.Data(), src.Data())

	return nil
}

// DivScalar performs element-wise division of a tensor by a scalar.
func (e *CPUEngine[T]) DivScalar(_ context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	if e.ops.IsZero(scalar) {
		return nil, errors.New("division by zero")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	aData := a.Data()
	rData := result.Data()
	for i := 0; i < len(aData); i++ { //nolint:intrange // Classic index loop maintained
		rData[i] = e.ops.Div(aData[i], scalar)
	}

	return result, nil
}

// NewCPUEngine creates a new CPUEngine.
func NewCPUEngine[T tensor.Numeric](ops numeric.Arithmetic[T]) *CPUEngine[T] {
	return &CPUEngine[T]{ops: ops}
}

// Ops returns the numeric.Arithmetic operations for the engine's numeric type.
func (e *CPUEngine[T]) Ops() numeric.Arithmetic[T] {
	return e.ops
}

func (e *CPUEngine[T]) getOrCreateDest(shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(dst) > 0 && dst[0] != nil {
		if !reflect.DeepEqual(dst[0].Shape(), shape) {
			return nil, fmt.Errorf("destination tensor has incorrect shape: got %v, want %v", dst[0].Shape(), shape)
		}

		return dst[0], nil
	}

	return tensor.New[T](shape, nil)
}

// UnaryOp applies a unary operation to a tensor.
func (e *CPUEngine[T]) UnaryOp(_ context.Context, a *tensor.TensorNumeric[T], op func(T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
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
func (e *CPUEngine[T]) binaryOp(_ context.Context, a, b *tensor.TensorNumeric[T], op func(T, T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil || b == nil {
		return nil, errors.New("input tensors cannot be nil")
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
	for i := 0; i < len(rData); i++ {
		aIndex := tensor.BroadcastIndex(i, a.Shape(), outputShape, broadcastA)
		bIndex := tensor.BroadcastIndex(i, b.Shape(), outputShape, broadcastB)
		rData[i] = op(aData[aIndex], bData[bIndex])
	}

	return result, nil
}

// Add performs element-wise addition of two tensors.
func (e *CPUEngine[T]) Add(
	ctx context.Context,
	a, b *tensor.TensorNumeric[T],
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Add, dst...)
}

// Sub performs element-wise subtraction of two tensors.
func (e *CPUEngine[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Sub, dst...)
}

// Mul performs element-wise multiplication of two tensors.
func (e *CPUEngine[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Mul, dst...)
}

// Div performs element-wise division of two tensors.
func (e *CPUEngine[T]) Div(_ context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil || b == nil {
		return nil, errors.New("input tensors cannot be nil")
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
	for i := 0; i < len(rData); i++ {
		aIndex := tensor.BroadcastIndex(i, a.Shape(), outputShape, broadcastA)
		bIndex := tensor.BroadcastIndex(i, b.Shape(), outputShape, broadcastB)
		if e.ops.IsZero(bData[bIndex]) {
			return nil, errors.New("division by zero")
		}
		rData[i] = e.ops.Div(aData[aIndex], bData[bIndex])
	}

	return result, nil
}

// MatMul performs matrix multiplication of two tensors.
func (e *CPUEngine[T]) MatMul(_ context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil || b == nil {
		return nil, errors.New("input tensors cannot be nil")
	}

	aShape := a.Shape()
	bShape := b.Shape()

	// Both tensors must have at least 2 dimensions
	if len(aShape) < 2 || len(bShape) < 2 {
		return nil, errors.New("tensors must have at least 2 dimensions")
	}

	// Check if the inner dimensions are compatible for matrix multiplication
	// For a @ b, the last dimension of a must match the second-to-last dimension of b
	if aShape[len(aShape)-1] != bShape[len(bShape)-2] {
		return nil, fmt.Errorf("invalid shapes for matrix multiplication: a.Shape()=%v, b.Shape()=%v (inner dimensions %d != %d)",
			aShape, bShape, aShape[len(aShape)-1], bShape[len(bShape)-2])
	}

	// Handle broadcasting: if b is 2D and a is higher dimensional, broadcast b
	var outputShape []int
	var batchSize int

	if len(aShape) > len(bShape) {
		// Broadcasting case: a is [batch..., m, k], b is [k, n]
		outputShape = make([]int, len(aShape))
		copy(outputShape, aShape[:len(aShape)-1])               // Copy batch dims + m
		outputShape[len(outputShape)-1] = bShape[len(bShape)-1] // Set n

		batchSize = 1
		for i := range len(aShape) - 2 {
			batchSize *= aShape[i]
		}
	} else {
		// Same dimensions case
		for i := range len(aShape) - 2 {
			if aShape[i] != bShape[i] {
				return nil, errors.New("batch dimensions must be equal")
			}
		}

		outputShape = make([]int, len(aShape))
		copy(outputShape, aShape[:len(aShape)-2])
		outputShape[len(outputShape)-2] = aShape[len(aShape)-2]
		outputShape[len(outputShape)-1] = bShape[len(bShape)-1]

		batchSize = 1
		for i := range len(aShape) - 2 {
			batchSize *= aShape[i]
		}
	}

	result, err := e.getOrCreateDest(outputShape, dst...)
	if err != nil {
		return nil, err
	}

	m := aShape[len(aShape)-2]
	k := aShape[len(aShape)-1]
	n := bShape[len(bShape)-1]

	aData := a.Data()
	bData := b.Data()
	rData := result.Data()

	for i := range batchSize {
		aOffset := i * m * k
		rOffset := i * m * n

		// For broadcasting, b doesn't have batch dimension, so bOffset is always 0
		bOffset := 0
		if len(aShape) == len(bShape) {
			bOffset = i * k * n
		}

		for row := 0; row < m; row++ { //nolint:intrange // Classic index loop maintained
			for col := 0; col < n; col++ { //nolint:intrange // Classic index loop maintained
				sum := e.ops.FromFloat64(0)
				for inner := 0; inner < k; inner++ { //nolint:intrange // Classic index loop maintained
					valA := aData[aOffset+row*k+inner]
					valB := bData[bOffset+inner*n+col]
					sum = e.ops.Add(sum, e.ops.Mul(valA, valB))
				}
				rData[rOffset+row*n+col] = sum
			}
		}
	}

	return result, nil
}

// Transpose transposes a tensor.
func (e *CPUEngine[T]) Transpose(_ context.Context, a *tensor.TensorNumeric[T], axes []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	originalShape := a.Shape()
	if axes == nil {
		// Default transpose for 2D tensors
		if len(originalShape) != 2 {
			return nil, fmt.Errorf("default transpose is only supported for 2D tensors, got %d dimensions", len(originalShape))
		}
		axes = []int{1, 0}
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
	for i := 0; i < a.Size(); i++ { //nolint:intrange // Classic index loop maintained
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
	a *tensor.TensorNumeric[T],
	axis int,
	keepDims bool,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
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
	if axis < 0 {
		axis = len(shape) + axis
	}
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

	for i := 0; i < len(aData); i++ { //nolint:intrange // Classic index loop maintained
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
		rData[rIndex] = e.ops.Add(rData[rIndex], aData[i])
	}

	return result, nil
}

// Exp computes the element-wise exponential of a tensor.
func (e *CPUEngine[T]) Exp(_ context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(a.Data()); i++ { //nolint:intrange // Classic index loop maintained
		result.Data()[i] = e.ops.Exp(a.Data()[i])
	}

	return result, nil
}

// Log computes the element-wise natural logarithm of a tensor.
func (e *CPUEngine[T]) Log(_ context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(a.Data()); i++ { //nolint:intrange // Classic index loop maintained
		result.Data()[i] = e.ops.Log(a.Data()[i])
	}

	return result, nil
}

// Pow raises each element of a tensor to the power of the corresponding element in another tensor.
func (e *CPUEngine[T]) Pow(
	ctx context.Context,
	base, exponent *tensor.TensorNumeric[T],
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	return e.binaryOp(ctx, base, exponent, e.ops.Pow, dst...)
}

// Concat concatenates a list of tensors along a given axis.
func (e *CPUEngine[T]) Concat(_ context.Context, tensors []*tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(tensors) == 0 {
		return nil, errors.New("no tensors provided for concatenation")
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
			return nil, errors.New("tensors must have the same number of dimensions for concatenation")
		}
		for i, dim := range currentShape {
			if i == axis {
				newShape[axis] += dim
			} else if dim != firstShape[i] {
				return nil, errors.New("dimensions must be equal except for the concatenation axis")
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
func (e *CPUEngine[T]) OneHot(_ context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if input == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	if depth <= 0 {
		return nil, errors.New("depth must be positive")
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
	for i := 0; i < outputSize; i++ { //nolint:intrange // Classic index loop maintained
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
func (e *CPUEngine[T]) Reshape(_ context.Context, a *tensor.TensorNumeric[T], shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}

	// Calculate current tensor size
	currentSize := 1
	for _, dim := range a.Shape() {
		currentSize *= dim
	}

	// Handle -1 dimension inference
	inferredShape := make([]int, len(shape))
	copy(inferredShape, shape)

	inferIndex := -1
	knownSize := 1
	for i, dim := range shape {
		switch {
		case dim == -1:
			if inferIndex != -1 {
				return nil, errors.New("only one dimension can be -1")
			}
			inferIndex = i
		case dim <= 0:
			return nil, fmt.Errorf("invalid dimension size: %d", dim)
		default:
			knownSize *= dim
		}
	}

	if inferIndex != -1 {
		if currentSize%knownSize != 0 {
			return nil, fmt.Errorf("cannot infer dimension: tensor size %d not divisible by known dimensions %d", currentSize, knownSize)
		}
		inferredShape[inferIndex] = currentSize / knownSize
	}

	// Check if the new shape is compatible with the existing data size
	newSize := 1
	for _, dim := range inferredShape {
		newSize *= dim
	}

	if currentSize != newSize {
		return nil, fmt.Errorf("new shape %v is not compatible with current tensor size %d", inferredShape, currentSize)
	}

	result, err := e.getOrCreateDest(inferredShape, dst...)
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
func (e *CPUEngine[T]) Repeat(_ context.Context, a *tensor.TensorNumeric[T], axis int, repetitions int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	shape := a.Shape()
	if axis < 0 || axis >= len(shape) {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, len(shape))
	}
	if repetitions <= 0 {
		return nil, errors.New("repetitions must be positive")
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
	for i := range numBlocks {
		for r := range repetitions {
			for j := range shape[axis] {
				srcStart := i*shape[axis]*blockSize + j*blockSize
				dstStart := i*shape[axis]*blockSize*repetitions + r*shape[axis]*blockSize + j*blockSize
				copy(result.Data()[dstStart:dstStart+blockSize], a.Data()[srcStart:srcStart+blockSize])
			}
		}
	}

	return result, nil
}

// ReduceMean calculates the mean of elements along a specified axis.
func (e *CPUEngine[T]) ReduceMean(
	ctx context.Context,
	a *tensor.TensorNumeric[T],
	axis int,
	keepDims bool,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	sum, err := e.Sum(ctx, a, axis, keepDims, dst...)
	if err != nil {
		return nil, err
	}

	// Get the size of the dimension that was reduced
	var divisor T
	if axis >= 0 && axis < a.Dims() {
		divisor = e.ops.FromFloat64(float64(a.Shape()[axis]))
	} else {
		divisor = e.ops.FromFloat64(float64(a.Size()))
	}

	return e.DivScalar(ctx, sum, divisor, sum)
}

// Rsqrt computes the element-wise reciprocal square root of a tensor.
func (e *CPUEngine[T]) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.UnaryOp(ctx, a, func(v T) T {
		return e.ops.FromFloat64(1.0 / math.Sqrt(float64(v)))
	}, dst...)
}
