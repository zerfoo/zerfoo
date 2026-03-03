package normalization

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// BatchNormalization implements inference-mode batch normalization.
// Forward expects 5 inputs: X, scale, B, mean, var (in that order).
// Formula per element at channel c:
//
//	y = scale[c] * (X - mean[c]) / sqrt(var[c] + epsilon) + B[c]
//
// scale, B, mean, and var must have shape [C] matching X's channel dimension.
type BatchNormalization[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	epsilon     T
	outputShape []int
}

// NewBatchNormalization creates a BatchNormalization layer.
func NewBatchNormalization[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], epsilon T) *BatchNormalization[T] {
	return &BatchNormalization[T]{engine: engine, ops: ops, epsilon: epsilon}
}

// Forward computes batch normalization in inference mode.
// inputs: [X, scale, B, mean, var].
func (b *BatchNormalization[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 5 {
		return nil, fmt.Errorf("BatchNormalization requires 5 inputs (X, scale, B, mean, var), got %d", len(inputs))
	}
	X, scale, bias, mean, variance := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]

	xShape := X.Shape()
	if len(xShape) < 2 {
		return nil, fmt.Errorf("BatchNormalization: X must have at least rank 2, got shape %v", xShape)
	}

	c := xShape[1]
	scaleData := scale.Data()
	biasData := bias.Data()
	meanData := mean.Data()
	varData := variance.Data()

	if len(scaleData) != c || len(biasData) != c || len(meanData) != c || len(varData) != c {
		return nil, fmt.Errorf("BatchNormalization: scale/B/mean/var must have length %d (C), got %d/%d/%d/%d",
			c, len(scaleData), len(biasData), len(meanData), len(varData))
	}

	// Pre-compute per-channel normalization factors: scale[c] / sqrt(var[c] + eps).
	factors := make([]T, c)
	shifts := make([]T, c)
	for ci := range c {
		std := b.ops.Sqrt(b.ops.Add(varData[ci], b.epsilon))
		factors[ci] = b.ops.Div(scaleData[ci], std)
		// shift = bias[c] - scale[c] * mean[c] / sqrt(var[c] + eps)
		shifts[ci] = b.ops.Sub(biasData[ci], b.ops.Mul(factors[ci], meanData[ci]))
	}

	xData := X.Data()
	outData := make([]T, len(xData))

	// Spatial size = product of dims after dim 1.
	spatialSize := 1
	for _, d := range xShape[2:] {
		spatialSize *= d
	}

	n := xShape[0]
	for ni := range n {
		for ci := range c {
			base := ni*c*spatialSize + ci*spatialSize
			f := factors[ci]
			s := shifts[ci]
			for i := range spatialSize {
				outData[base+i] = b.ops.Add(b.ops.Mul(f, xData[base+i]), s)
			}
		}
	}

	out, err := tensor.New[T](xShape, outData)
	if err != nil {
		return nil, fmt.Errorf("BatchNormalization: failed to create output tensor: %w", err)
	}
	b.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only).
func (b *BatchNormalization[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "BatchNormalization".
func (b *BatchNormalization[T]) OpType() string { return "BatchNormalization" }

// Attributes returns the layer configuration.
func (b *BatchNormalization[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"epsilon": b.epsilon}
}

// OutputShape returns the output shape (populated after Forward).
func (b *BatchNormalization[T]) OutputShape() []int { return b.outputShape }

// Parameters returns nil.
func (b *BatchNormalization[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildBatchNormalization constructs a BatchNormalization layer for the registry.
// Reads "epsilon" (float32 or float64) from attributes; defaults to 1e-5.
func BuildBatchNormalization[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	eps := 1e-5
	if v, ok := attributes["epsilon"]; ok {
		switch e := v.(type) {
		case float64:
			eps = e
		case float32:
			eps = float64(e)
		}
	}
	return NewBatchNormalization(engine, ops, ops.FromFloat64(eps)), nil
}

// Statically assert that BatchNormalization implements graph.Node.
var _ graph.Node[float32] = (*BatchNormalization[float32])(nil)
