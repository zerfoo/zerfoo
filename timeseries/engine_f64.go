package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// matMulEngine performs matrix multiplication via the compute engine.
// a: [M][K], b: [K][N] -> result: [M][N].
// Converts float64 inputs to float32 tensors, calls engine.MatMul, and
// converts the float32 result back to float64.
func matMulEngine(engine compute.Engine[float32], ctx context.Context, a, b [][]float64) ([][]float64, error) {
	rows := len(a)
	if rows == 0 {
		return nil, nil
	}
	inner := len(a[0])
	cols := len(b[0])

	aFlat := make([]float32, rows*inner)
	bFlat := make([]float32, inner*cols)

	for i, row := range a {
		off := i * inner
		for j, v := range row {
			aFlat[off+j] = float32(v)
		}
	}
	for i, row := range b {
		off := i * cols
		for j, v := range row {
			bFlat[off+j] = float32(v)
		}
	}

	aTensor, err := tensor.New[float32]([]int{rows, inner}, aFlat)
	if err != nil {
		return nil, fmt.Errorf("matMulEngine: create a tensor: %w", err)
	}
	bTensor, err := tensor.New[float32]([]int{inner, cols}, bFlat)
	if err != nil {
		return nil, fmt.Errorf("matMulEngine: create b tensor: %w", err)
	}

	cTensor, err := engine.MatMul(ctx, aTensor, bTensor)
	if err != nil {
		return nil, fmt.Errorf("matMulEngine: matmul: %w", err)
	}

	cData := cTensor.Data()
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		off := i * cols
		for j := 0; j < cols; j++ {
			result[i][j] = float64(cData[off+j])
		}
	}
	return result, nil
}

// linearF64Engine computes x @ W + b using the engine for the MatMul.
// x: [n][inDim], W: [inDim*outDim] (row-major), b: [outDim].
// The matrix multiplication is performed via engine.MatMul in float32;
// bias addition remains in float64.
func linearF64Engine(engine compute.Engine[float32], ctx context.Context, x [][]float64, w, b []float64, inDim, outDim int) ([][]float64, error) {
	n := len(x)

	// Reshape flat w [inDim*outDim] into [inDim][outDim] for matMulEngine.
	wMat := make([][]float64, inDim)
	for i := 0; i < inDim; i++ {
		wMat[i] = w[i*outDim : (i+1)*outDim]
	}

	out, err := matMulEngine(engine, ctx, x, wMat)
	if err != nil {
		return nil, err
	}

	// Add bias in float64.
	for i := 0; i < n; i++ {
		for j := 0; j < outDim; j++ {
			out[i][j] += b[j]
		}
	}
	return out, nil
}
