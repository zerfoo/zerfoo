// Package compute implements tensor computation engines and operations.
package compute

import (
	"context"
	"errors"
	"fmt"
	rand "math/rand/v2"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"

	float16 "github.com/zerfoo/float16"
	float8 "github.com/zerfoo/float8"
	"github.com/zerfoo/zerfoo/internal/xblas"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// CPUEngine is a CPU-based implementation of the Engine interface.
type CPUEngine[T tensor.Numeric] struct {
	ops numeric.Arithmetic[T]
}

// parallelFor splits [0,total) into chunks and runs fn(start,end) across workers.
// It avoids goroutine overhead for small ranges.
func parallelFor(total int, fn func(start, end int)) {
	const minPerG = 32768
	if total <= minPerG {
		fn(0, total)
		return
	}
	workers := runtime.GOMAXPROCS(0)
	// Cap workers to avoid tiny chunks
	maxW := total / minPerG
	if maxW < 1 {
		maxW = 1
	}
	if workers > maxW {
		workers = maxW
	}
	if workers < 1 {
		workers = 1
	}
	chunk := (total + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < total; start += chunk {
		end := start + chunk
		if end > total {
			end = total
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}

// Split splits a tensor into numSplits along the given axis.
// All splits are equal-sized; shape[axis] must be divisible by numSplits.
func (e *CPUEngine[T]) Split(_ context.Context, a *tensor.TensorNumeric[T], numSplits int, axis int) ([]*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	if numSplits <= 0 {
		return nil, errors.New("numSplits must be positive")
	}
	shape := a.Shape()
	rank := len(shape)
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, rank)
	}
	if shape[axis]%numSplits != 0 {
		return nil, fmt.Errorf("cannot split dimension %d (size %d) into %d equal parts", axis, shape[axis], numSplits)
	}

	part := shape[axis] / numSplits
	outShape := make([]int, rank)
	copy(outShape, shape)
	outShape[axis] = part

	// Allocate outputs
	outs := make([]*tensor.TensorNumeric[T], numSplits)
	for i := 0; i < numSplits; i++ { //nolint:intrange
		t, err := tensor.New[T](outShape, nil)
		if err != nil {
			return nil, err
		}
		outs[i] = t
	}

	// Compute block sizes for contiguous copies in row-major order
	blockSize := 1
	for i := axis + 1; i < rank; i++ {
		blockSize *= shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}

	srcData := a.Data()
	for i := 0; i < numSplits; i++ { //nolint:intrange
		dstData := outs[i].Data()
		for o := 0; o < outer; o++ { //nolint:intrange
			for j := 0; j < part; j++ { //nolint:intrange
				srcStart := o*shape[axis]*blockSize + (i*part+j)*blockSize
				dstStart := o*part*blockSize + j*blockSize
				copy(dstData[dstStart:dstStart+blockSize], srcData[srcStart:srcStart+blockSize])
			}
		}
	}

	return outs, nil
}

// ReduceSum delegates to Sum for reduction along an axis.
func (e *CPUEngine[T]) ReduceSum(
	ctx context.Context,
	a *tensor.TensorNumeric[T],
	axis int,
	keepDims bool,
	dst ...*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	return e.Sum(ctx, a, axis, keepDims, dst...)
}

// Gather performs an embedding-style gather.
// params must be 2D [vocab, dim].
// indices may be 1D [N] or 2D [batch, seq].
// output must be [indices..., dim], i.e., [N, dim] or [batch, seq, dim].
func (e *CPUEngine[T]) Gather(_ context.Context, params *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[T]) error {
	if params == nil || indices == nil || output == nil {
		return errors.New("params, indices, and output cannot be nil")
	}
	pShape := params.Shape()
	if len(pShape) != 2 {
		return fmt.Errorf("params must be 2D [vocab, dim], got shape %v", pShape)
	}
	vocab, dim := pShape[0], pShape[1]

	switch indices.Dims() {
	case 1:
		n := indices.Shape()[0]
		if !reflect.DeepEqual(output.Shape(), []int{n, dim}) {
			return fmt.Errorf("output shape must be [N, dim]=[%d, %d], got %v", n, dim, output.Shape())
		}
		idxData := indices.Data()
		outData := output.Data()
		parData := params.Data()
		for i := 0; i < n; i++ { //nolint:intrange
			idx := idxData[i]
			if idx < 0 || idx >= vocab {
				return fmt.Errorf("index %d out of bounds [0,%d)", idx, vocab)
			}
			copy(outData[i*dim:(i+1)*dim], parData[idx*dim:(idx+1)*dim])
		}
		return nil
	case 2:
		b, s := indices.Shape()[0], indices.Shape()[1]
		if !reflect.DeepEqual(output.Shape(), []int{b, s, dim}) {
			return fmt.Errorf("output shape must be [batch, seq, dim]=[%d, %d, %d], got %v", b, s, dim, output.Shape())
		}
		idxData := indices.Data()
		outData := output.Data()
		parData := params.Data()
		// flatten loop over N=b*s
		N := b * s
		for i := 0; i < N; i++ { //nolint:intrange
			idx := idxData[i]
			if idx < 0 || idx >= vocab {
				return fmt.Errorf("index %d out of bounds [0,%d)", idx, vocab)
			}
			copy(outData[i*dim:(i+1)*dim], parData[idx*dim:(idx+1)*dim])
		}
		return nil
	default:
		return fmt.Errorf("indices must be 1D or 2D, got %dD", indices.Dims())
	}
}

// ScatterAdd performs a row-wise scatter-add for embeddings.
// dEmbeddingTable must be [vocab, dim].
// indices may be 1D [N] or multi-dim with flattened length N.
// dOut must be [N, dim].
// For each i in [0..N), it applies: dEmbeddingTable[indices[i], :] += dOut[i, :].
func (e *CPUEngine[T]) ScatterAdd(_ context.Context, dEmbeddingTable *tensor.TensorNumeric[T], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[T]) error {
	if dEmbeddingTable == nil || indices == nil || dOut == nil {
		return errors.New("dEmbeddingTable, indices, and dOut cannot be nil")
	}
	tblShape := dEmbeddingTable.Shape()
	if len(tblShape) != 2 {
		return fmt.Errorf("dEmbeddingTable must be 2D [vocab, dim], got shape %v", tblShape)
	}
	vocab, dim := tblShape[0], tblShape[1]
	// Flattened N from indices
	N := 1
	for _, d := range indices.Shape() {
		N *= d
	}
	if !reflect.DeepEqual(dOut.Shape(), []int{N, dim}) {
		return fmt.Errorf("dOut shape must be [N, dim]=[%d, %d], got %v", N, dim, dOut.Shape())
	}
	idxData := indices.Data()
	outData := dOut.Data()
	tblData := dEmbeddingTable.Data()
	for i := 0; i < N; i++ { //nolint:intrange
		idx := idxData[i]
		if idx < 0 || idx >= vocab {
			return fmt.Errorf("index %d out of bounds [0,%d)", idx, vocab)
		}
		rowStart := idx * dim
		srcStart := i * dim
		for j := 0; j < dim; j++ { //nolint:intrange
			tblData[rowStart+j] = e.ops.Add(tblData[rowStart+j], outData[srcStart+j])
		}
	}
	return nil
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
	parallelFor(len(aData), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			rData[i] = e.ops.Add(aData[i], scalar)
		}
	})
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
	parallelFor(len(aData), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			rData[i] = e.ops.Mul(aData[i], scalar)
		}
	})
	return result, nil
}

// NewCPUEngine constructs a new CPUEngine for the given numeric operations.
func NewCPUEngine[T tensor.Numeric](ops numeric.Arithmetic[T]) *CPUEngine[T] {
	return &CPUEngine[T]{ops: ops}
}

// Ops returns the arithmetic ops for this engine.
func (e *CPUEngine[T]) Ops() numeric.Arithmetic[T] { return e.ops }

// getOrCreateDest ensures a destination tensor with the requested shape exists.
// If dst is provided, validates the shape and returns it; otherwise allocates a new tensor.
func (e *CPUEngine[T]) getOrCreateDest(shape []int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(dst) > 0 && dst[0] != nil {
		if !reflect.DeepEqual(dst[0].Shape(), shape) {
			return nil, fmt.Errorf("destination tensor has shape %v, expected %v", dst[0].Shape(), shape)
		}
		return dst[0], nil
	}
	out, err := tensor.New[T](shape, nil)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Zero sets all elements of tensor a to zero.
func (e *CPUEngine[T]) Zero(_ context.Context, a *tensor.TensorNumeric[T]) error {
	if a == nil {
		return errors.New("input tensor cannot be nil")
	}
	zero := e.ops.FromFloat64(0)
	data := a.Data()
	parallelFor(len(data), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			data[i] = zero
		}
	})
	return nil
}

// Zeros fills the tensor with zeros. If shape is provided, (re)allocates to that shape.
func (e *CPUEngine[T]) Zeros(ctx context.Context, a *tensor.TensorNumeric[T], shape []int) error {
	if a == nil {
		return errors.New("input tensor cannot be nil")
	}
	if shape != nil {
		// Allocate a fresh buffer with the requested shape
		tmp, err := tensor.New[T](shape, nil)
		if err != nil {
			return err
		}
		a.SetData(tmp.Data())
		a.SetShape(tmp.Shape())
		a.SetStrides(tmp.Strides())
	}
	return e.Zero(ctx, a)
}

// UnaryOp applies a unary element-wise operation.
func (e *CPUEngine[T]) UnaryOp(_ context.Context, a *tensor.TensorNumeric[T], op func(T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	aData := a.Data()
	rData := result.Data()
	parallelFor(len(aData), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			rData[i] = op(aData[i])
		}
	})
	return result, nil
}

// binaryOp performs a broadcasted binary element-wise operation.
func (e *CPUEngine[T]) binaryOp(_ context.Context, a, b *tensor.TensorNumeric[T], op func(T, T) T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil || b == nil {
		return nil, errors.New("input tensors cannot be nil")
	}
	outShape, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	result, err := e.getOrCreateDest(outShape, dst...)
	if err != nil {
		return nil, err
	}

	// Expand shapes/strides to out rank
	R := len(outShape)
	aShape, aStrides := expandShapeStrides(a.Shape(), makeStrides(a.Shape()), R)
	bShape, bStrides := expandShapeStrides(b.Shape(), makeStrides(b.Shape()), R)
	outStrides := makeStrides(outShape)

	aData := a.Data()
	bData := b.Data()
	rData := result.Data()

	total := 1
	for _, d := range outShape {
		total *= d
	}

	parallelFor(total, func(start, end int) {
		for lin := start; lin < end; lin++ { //nolint:intrange
			// Decode linear index into coords
			idx := lin
			offA := 0
			offB := 0
			for i := 0; i < R; i++ { //nolint:intrange
				stride := outStrides[i]
				coord := 0
				if stride != 0 {
					coord = idx / stride
					idx %= stride
				}
				if aShape[i] != 1 {
					offA += coord * aStrides[i]
				}
				if bShape[i] != 1 {
					offB += coord * bStrides[i]
				}
			}
			rData[lin] = op(aData[offA], bData[offB])
		}
	})

	return result, nil
}

// Add performs element-wise addition with broadcasting.
func (e *CPUEngine[T]) Add(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Add, dst...)
}

// Sub performs element-wise subtraction with broadcasting.
func (e *CPUEngine[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Sub, dst...)
}

// Mul performs element-wise multiplication with broadcasting.
func (e *CPUEngine[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.binaryOp(ctx, a, b, e.ops.Mul, dst...)
}

// Div performs element-wise division with broadcasting. For integer types, division by zero returns an error.
func (e *CPUEngine[T]) Div(_ context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil || b == nil {
		return nil, errors.New("input tensors cannot be nil")
	}
	outShape, err := broadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	result, err := e.getOrCreateDest(outShape, dst...)
	if err != nil {
		return nil, err
	}

	R := len(outShape)
	aShape, aStrides := expandShapeStrides(a.Shape(), a.Strides(), R)
	bShape, bStrides := expandShapeStrides(b.Shape(), b.Strides(), R)
	outStrides := makeStrides(outShape)

	aData := a.Data()
	bData := b.Data()
	rData := result.Data()

	// Determine if T is an integer type
	var zeroT T
	isInt := false
	switch any(zeroT).(type) {
	case int, int8, int16, int32, int64, uint, uint32, uint64:
		isInt = true
	default:
		isInt = false
	}

	total := 1
	for _, d := range outShape {
		total *= d
	}

	// Track division-by-zero across workers; return error after completion if any
	var divZeroFound atomic.Bool

	parallelFor(total, func(start, end int) {
		for lin := start; lin < end; lin++ { //nolint:intrange
			// Decode linear index
			idx := lin
			offA := 0
			offB := 0
			for i := 0; i < R; i++ { //nolint:intrange
				stride := outStrides[i]
				coord := 0
				if stride != 0 {
					coord = idx / stride
					idx %= stride
				}
				if aShape[i] != 1 {
					offA += coord * aStrides[i]
				}
				if bShape[i] != 1 {
					offB += coord * bStrides[i]
				}
			}
			if isInt {
				if bData[offB] == zeroT {
					divZeroFound.Store(true)
					// Skip writing to keep semantics consistent when returning error
					continue
				}
			}
			rData[lin] = e.ops.Div(aData[offA], bData[offB])
		}
	})

	if isInt && divZeroFound.Load() {
		return nil, errors.New("division by zero")
	}

	return result, nil
}

// Copy copies src into dst; shapes must match.
func (e *CPUEngine[T]) Copy(_ context.Context, dst, src *tensor.TensorNumeric[T]) error {
	if dst == nil || src == nil {
		return errors.New("input tensors cannot be nil")
	}
	if !reflect.DeepEqual(dst.Shape(), src.Shape()) {
		return fmt.Errorf("shape mismatch: dst %v vs src %v", dst.Shape(), src.Shape())
	}
	copy(dst.Data(), src.Data())
	return nil
}

// DivScalar divides a tensor by a scalar value element-wise.
func (e *CPUEngine[T]) DivScalar(_ context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	// Integer divide-by-zero guard
	var zeroT T
	switch any(zeroT).(type) {
	case int, int8, int16, int32, int64, uint, uint32, uint64:
		if scalar == zeroT {
			return nil, errors.New("division by zero")
		}
	}
	result, err := e.getOrCreateDest(a.Shape(), dst...)
	if err != nil {
		return nil, err
	}
	aData := a.Data()
	rData := result.Data()
	parallelFor(len(aData), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			rData[i] = e.ops.Div(aData[i], scalar)
		}
	})
	return result, nil
}

// RandomUniform fills t with random values between minVal and maxVal.
func (e *CPUEngine[T]) RandomUniform(_ context.Context, t *tensor.TensorNumeric[T], minVal, maxVal T) error {
	if t == nil {
		return errors.New("input tensor cannot be nil")
	}
	min := float64(minVal)
	max := float64(maxVal)
	if max < min {
		min, max = max, min
	}
	span := max - min
	data := t.Data()
	parallelFor(len(data), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			r := rand.Float64()*span + min
			data[i] = e.ops.FromFloat64(r)
		}
	})
	return nil
}

// Fill sets all elements of t to value.
func (e *CPUEngine[T]) Fill(_ context.Context, t *tensor.TensorNumeric[T], value T) error {
	if t == nil {
		return errors.New("input tensor cannot be nil")
	}
	data := t.Data()
	parallelFor(len(data), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			data[i] = value
		}
	})
	return nil
}

// Helper: compute row-major strides for a shape.
func makeStrides(shape []int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// Helper: broadcast two shapes following NumPy rules.
func broadcastShapes(a, b []int) ([]int, error) {
	// Align right
	maxRank := len(a)
	if len(b) > maxRank {
		maxRank = len(b)
	}
	out := make([]int, maxRank)
	for i := 0; i < maxRank; i++ {
		da := 1
		db := 1
		if i >= maxRank-len(a) {
			da = a[i-(maxRank-len(a))]
		}
		if i >= maxRank-len(b) {
			db = b[i-(maxRank-len(b))]
		}
		switch {
		case da == db:
			out[i] = da
		case da == 1:
			out[i] = db
		case db == 1:
			out[i] = da
		default:
			return nil, fmt.Errorf("shapes %v and %v are not broadcastable", a, b)
		}
	}
	return out, nil
}

// Helper: expand shape and strides to a target rank (left-padding with size 1 and stride 0).
func expandShapeStrides(shape, strides []int, rank int) ([]int, []int) {
	pad := rank - len(shape)
	if pad < 0 {
		pad = 0
	}
	es := make([]int, rank)
	est := make([]int, rank)
	for i := 0; i < pad; i++ {
		es[i] = 1
		est[i] = 0
	}
	for i := pad; i < rank; i++ {
		es[i] = shape[i-pad]
		est[i] = strides[i-pad]
		if es[i] == 1 {
			est[i] = 0 // broadcasting dimension
		}
	}
	return es, est
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
		for i := 0; i < len(aShape)-2; i++ {
			batchSize *= aShape[i]
		}
	} else {
		// Same dimensions case
		for i := 0; i < len(aShape)-2; i++ {
			if aShape[i] != bShape[i] {
				return nil, errors.New("batch dimensions must be equal")
			}
		}

		outputShape = make([]int, len(aShape))
		copy(outputShape, aShape[:len(aShape)-2])
		outputShape[len(outputShape)-2] = aShape[len(aShape)-2]
		outputShape[len(outputShape)-1] = bShape[len(bShape)-1]

		batchSize = 1
		for i := 0; i < len(aShape)-2; i++ {
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

	// Use xblas adapter: f32/f64 direct; f16/f8 via convert->sgemm->convert
	switch any(*new(T)).(type) {
	case float32:
		aF := any(aData).([]float32)
		bF := any(bData).([]float32)
		rF := any(rData).([]float32)
		for i := 0; i < batchSize; i++ { //nolint:intrange
			aOffset := i * m * k
			cOffset := i * m * n
			bOffset := 0
			if len(aShape) == len(bShape) {
				bOffset = i * k * n
			}
			xblas.GemmF32(m, n, k,
				aF[aOffset:aOffset+m*k],
				bF[bOffset:bOffset+k*n],
				rF[cOffset:cOffset+m*n],
			)
		}
	case float64:
		aD := any(aData).([]float64)
		bD := any(bData).([]float64)
		rD := any(rData).([]float64)
		for i := 0; i < batchSize; i++ { //nolint:intrange
			aOffset := i * m * k
			cOffset := i * m * n
			bOffset := 0
			if len(aShape) == len(bShape) {
				bOffset = i * k * n
			}
			xblas.GemmF64(m, n, k,
				aD[aOffset:aOffset+m*k],
				bD[bOffset:bOffset+k*n],
				rD[cOffset:cOffset+m*n],
			)
		}
	case float16.Float16:
		aH := any(aData).([]float16.Float16)
		bH := any(bData).([]float16.Float16)
		rH := any(rData).([]float16.Float16)
		for i := 0; i < batchSize; i++ { //nolint:intrange
			aOffset := i * m * k
			cOffset := i * m * n
			bOffset := 0
			if len(aShape) == len(bShape) {
				bOffset = i * k * n
			}
			xblas.GemmF16(m, n, k,
				aH[aOffset:aOffset+m*k],
				bH[bOffset:bOffset+k*n],
				rH[cOffset:cOffset+m*n],
			)
		}
	case float8.Float8:
		aE := any(aData).([]float8.Float8)
		bE := any(bData).([]float8.Float8)
		rE := any(rData).([]float8.Float8)
		for i := 0; i < batchSize; i++ { //nolint:intrange
			aOffset := i * m * k
			cOffset := i * m * n
			bOffset := 0
			if len(aShape) == len(bShape) {
				bOffset = i * k * n
			}
			xblas.GemmF8(m, n, k,
				aE[aOffset:aOffset+m*k],
				bE[bOffset:bOffset+k*n],
				rE[cOffset:cOffset+m*n],
			)
		}
	default:
		// Fallback to naive implementation for other types
		for i := 0; i < batchSize; i++ {
			aOffset := i * m * k
			rOffset := i * m * n
			bOffset := 0
			if len(aShape) == len(bShape) {
				bOffset = i * k * n
			}
			for row := 0; row < m; row++ { //nolint:intrange
				for col := 0; col < n; col++ { //nolint:intrange
					sum := e.ops.FromFloat64(0)
					for inner := 0; inner < k; inner++ { //nolint:intrange
						valA := aData[aOffset+row*k+inner]
						valB := bData[bOffset+inner*n+col]
						sum = e.ops.Add(sum, e.ops.Mul(valA, valB))
					}
					rData[rOffset+row*n+col] = sum
				}
			}
		}
	}

	return result, nil
}

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

  // Use compact strides to match Data() layout (important for views)
  aStrides := makeStrides(originalShape)
  rStrides := makeStrides(newShape)
  aData := a.Data()
  rData := result.Data()

  // Build inverse permutation: invAxes[oldAxis] = newAxisIndex
  invAxes := make([]int, len(axes))
  for newAxis, oldAxis := range axes {
    invAxes[oldAxis] = newAxis
  }

  total := 1
  for _, d := range originalShape {
    total *= d
  }

  parallelFor(total, func(start, end int) {
    for lin := start; lin < end; lin++ { //nolint:intrange
      // Decode lin into old coordinates using compact strides, and map directly to new linear index
      idx := lin
      newLin := 0
      for dim := 0; dim < len(originalShape); dim++ { //nolint:intrange
        stride := aStrides[dim]
        coord := 0
        if stride != 0 {
          coord = idx / stride
          idx %= stride
        }
        newDim := invAxes[dim]
        newLin += coord * rStrides[newDim]
      }
      rData[newLin] = aData[lin]
    }
  })

  return result, nil
}

// Sum computes the sum of tensor elements along the specified axis.
// If keepDims is true, the reduced dimensions are retained with size 1.
// An optional destination tensor can be provided to store the result.
func (e *CPUEngine[T]) Sum(
  _ context.Context,
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
    for i := range shape {
      if i != axis {
        newShape = append(newShape, shape[i])
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

  // Use compact strides to match Data() layout and parallelize over independent stripes
  aData := a.Data()
  rData := result.Data()
  aStrides := makeStrides(shape)
  rStrides := makeStrides(newShape)

  // Compute block sizes
  inner := 1
  for i := axis + 1; i < len(shape); i++ {
    inner *= shape[i]
  }
  outer := 1
  for i := 0; i < axis; i++ {
    outer *= shape[i]
  }
  axisSize := shape[axis]

  stripes := outer * inner // each stripe maps to exactly one output index
  parallelFor(stripes, func(start, end int) {
    for s := start; s < end; s++ { //nolint:intrange
      o := 0
      in := 0
      if inner != 0 {
        o = s / inner
        in = s % inner
      }
      base := o*axisSize*inner + in
      step := inner

      // Compute output linear index for this stripe by decoding base index
      rIndex := 0
      tmp := base
      for j := 0; j < len(shape); j++ { //nolint:intrange
        stride := aStrides[j]
        coord := 0
        if stride != 0 {
          coord = tmp / stride
          tmp %= stride
        }
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

      // Reduce along the axis for this stripe
      sum := e.ops.FromFloat64(0)
      for k := 0; k < axisSize; k++ { //nolint:intrange
        idx := base + k*step
        sum = e.ops.Add(sum, aData[idx])
      }
      rData[rIndex] = sum
    }
  })

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
	aData := a.Data()
	rData := result.Data()
	parallelFor(len(aData), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			rData[i] = e.ops.Exp(aData[i])
		}
	})

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
	aData := a.Data()
	rData := result.Data()
	parallelFor(len(aData), func(start, end int) {
		for i := start; i < end; i++ { //nolint:intrange
			rData[i] = e.ops.Log(aData[i])
		}
	})

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

	first := tensors[0]
	rank := len(first.Shape())
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, rank)
	}

	// Validate shapes and build output shape
	outShape := make([]int, rank)
	copy(outShape, first.Shape())
	outShape[axis] = 0
	for _, t := range tensors {
		s := t.Shape()
		if len(s) != rank {
			return nil, errors.New("tensors must have the same number of dimensions for concatenation")
		}
		for i, d := range s {
			if i == axis {
				outShape[axis] += d
			} else if d != first.Shape()[i] {
				return nil, errors.New("dimensions must be equal except for the concatenation axis")
			}
		}
	}

	out, err := e.getOrCreateDest(outShape, dst...)
	if err != nil {
		return nil, err
	}

	// Compute block sizes
	blockSize := 1
	for i := axis + 1; i < rank; i++ {
		blockSize *= outShape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= outShape[i]
	}

	outData := out.Data()
	axisOffset := 0 // running offset along concatenation axis in output
	for _, t := range tensors {
		ts := t.Shape()
		tAxis := ts[axis]
		tData := t.Data()

		for o := 0; o < outer; o++ { //nolint:intrange
			for j := 0; j < tAxis; j++ { //nolint:intrange
				srcStart := o*tAxis*blockSize + j*blockSize
				dstStart := o*outShape[axis]*blockSize + (axisOffset+j)*blockSize
				copy(outData[dstStart:dstStart+blockSize], tData[srcStart:srcStart+blockSize])
			}
		}
		axisOffset += tAxis
	}

	return out, nil
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
		return e.ops.Div(e.ops.FromFloat64(1), e.ops.Sqrt(v))
	}, dst...)
}

// Sqrt computes the element-wise square root of a tensor.
func (e *CPUEngine[T]) Sqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.UnaryOp(ctx, a, e.ops.Sqrt, dst...)
}

// Softmax applies the softmax function to a tensor along a given axis.
// If axis is negative, it is interpreted relative to the last axis (e.g., -1 means last axis).
func (e *CPUEngine[T]) Softmax(_ context.Context, a *tensor.TensorNumeric[T], axis int, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if a == nil {
		return nil, errors.New("input tensor cannot be nil")
	}
	shape := a.Shape()
	rank := len(shape)
	if rank == 0 {
		// Softmax of a scalar is 1
		out, err := e.getOrCreateDest(shape, dst...)
		if err != nil {
			return nil, err
		}
		out.Data()[0] = e.ops.FromFloat64(1)
		return out, nil
	}
	if axis < 0 {
		axis = rank + axis
	}
	if axis < 0 || axis >= rank {
		return nil, fmt.Errorf("axis %d is out of bounds for tensor with %d dimensions", axis, rank)
	}

	out, err := e.getOrCreateDest(shape, dst...)
	if err != nil {
		return nil, err
	}

	aData := a.Data()
	oData := out.Data()

	// Compute sizes for block iteration
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= shape[i]
	}
	axisSize := shape[axis]

	// Iterate blocks; within each (outer, inner) pair we process a stripe across axis
	for o := 0; o < outer; o++ { //nolint:intrange
		for in := 0; in < inner; in++ { //nolint:intrange
			base := o*axisSize*inner + in
			step := inner

			// 1) Find max for numerical stability
			maxVal := aData[base]
			for k := 1; k < axisSize; k++ { //nolint:intrange
				idx := base + k*step
				if e.ops.GreaterThan(aData[idx], maxVal) {
					maxVal = aData[idx]
				}
			}

			// 2) Compute exponentials and sum
			sum := e.ops.FromFloat64(0)
			for k := 0; k < axisSize; k++ { //nolint:intrange
				idx := base + k*step
				shifted := e.ops.Sub(aData[idx], maxVal)
				ex := e.ops.Exp(shifted)
				oData[idx] = ex
				sum = e.ops.Add(sum, ex)
			}

			// 3) Normalize
			for k := 0; k < axisSize; k++ { //nolint:intrange
				idx := base + k*step
				oData[idx] = e.ops.Div(oData[idx], sum)
			}
		}
	}

	return out, nil
}
