package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Tile repeats a tensor along each dimension according to the repeats tensor.
// ONNX Tile op: output[i] = input[i % input_shape[dim]] for each dim.
type Tile[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (t *Tile[T]) OpType() string                  { return "Tile" }
func (t *Tile[T]) Attributes() map[string]any       { return nil }
func (t *Tile[T]) OutputShape() []int               { return nil }
func (t *Tile[T]) Parameters() []*graph.Parameter[T] { return nil }

func (t *Tile[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Tile requires 2 inputs (data, repeats), got %d", len(inputs))
	}
	data := inputs[0]
	repeatsData := inputs[1].Data()
	inShape := data.Shape()

	if len(repeatsData) != len(inShape) {
		return nil, fmt.Errorf("Tile: repeats length %d != input rank %d", len(repeatsData), len(inShape))
	}

	repeats := make([]int, len(repeatsData))
	outShape := make([]int, len(inShape))
	totalSize := 1
	for i, r := range repeatsData {
		repeats[i] = int(r)
		if repeats[i] <= 0 {
			return nil, fmt.Errorf("Tile: repeat[%d] = %d must be positive", i, repeats[i])
		}
		outShape[i] = inShape[i] * repeats[i]
		totalSize *= outShape[i]
	}

	inData := data.Data()
	out := make([]T, totalSize)

	// For each output index, compute the corresponding input index using modulo.
	rank := len(inShape)
	inStrides := make([]int, rank)
	outStrides := make([]int, rank)
	inStrides[rank-1] = 1
	outStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		inStrides[i] = inStrides[i+1] * inShape[i+1]
		outStrides[i] = outStrides[i+1] * outShape[i+1]
	}

	for i := 0; i < totalSize; i++ {
		out[i] = inData[tileSourceIndex(i, rank, inShape, outStrides, inStrides)]
	}

	return tensor.New(outShape, out)
}

// tileSourceIndex decomposes a flat output index into per-dimension
// coordinates and maps each coordinate back into the input via modulo.
func tileSourceIndex(flatIdx, rank int, inShape, outStrides, inStrides []int) int {
	inIdx := 0
	rem := flatIdx
	for d := 0; d < rank; d++ {
		coord := rem / outStrides[d]
		rem %= outStrides[d]
		inIdx += (coord % inShape[d]) * inStrides[d]
	}
	return inIdx
}

func (t *Tile[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("Tile backward not implemented")
}

// BuildTile constructs a Tile node from attributes.
func BuildTile[T tensor.Numeric](
	engine compute.Engine[T], _ numeric.Arithmetic[T], _ string,
	_ map[string]*graph.Parameter[T], _ map[string]any,
) (graph.Node[T], error) {
	return &Tile[T]{engine: engine}, nil
}

var _ graph.Node[float32] = (*Tile[float32])(nil)
