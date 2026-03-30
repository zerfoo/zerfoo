package timeseries

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// VariateProjection projects each variate of a multivariate time series
// independently, then concatenates with a learned frequency embedding.
// This follows the Moirai-2 any-variate input projection design, supporting
// arbitrary numbers of variates with potentially different lengths via
// padding and attention masks.
//
// Each variate is projected from its time dimension to embedDim using a
// shared linear projection. A learnable frequency embedding is added per
// variate to encode variate identity.
type VariateProjection[T tensor.Numeric] struct {
	name     string
	engine   compute.Engine[T]
	ops      numeric.Arithmetic[T]
	proj     *graph.Parameter[T] // [inputDim, embedDim]
	bias     *graph.Parameter[T] // [embedDim]
	freqEmb  *graph.Parameter[T] // [maxVariates, embedDim]
	inputDim int
	embedDim int
	maxVar   int
}

// NewVariateProjection creates a new any-variate input projection layer.
//
// Parameters:
//   - name: layer name
//   - engine: compute engine for tensor operations
//   - ops: arithmetic operations for the numeric type
//   - inputDim: length of each variate's time series
//   - embedDim: output embedding dimension per variate
//   - maxVariates: maximum number of variates supported (for frequency embedding table)
func NewVariateProjection[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputDim, embedDim, maxVariates int,
) (*VariateProjection[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if inputDim <= 0 {
		return nil, fmt.Errorf("inputDim must be positive, got %d", inputDim)
	}
	if embedDim <= 0 {
		return nil, fmt.Errorf("embedDim must be positive, got %d", embedDim)
	}
	if maxVariates <= 0 {
		return nil, fmt.Errorf("maxVariates must be positive, got %d", maxVariates)
	}

	// Xavier initialization for projection weights.
	scale := math.Sqrt(2.0 / float64(inputDim+embedDim))
	projData := make([]T, inputDim*embedDim)
	for i := range projData {
		projData[i] = T(rand.NormFloat64() * scale)
	}
	projTensor, err := tensor.New[T]([]int{inputDim, embedDim}, projData)
	if err != nil {
		return nil, err
	}
	proj, err := graph.NewParameter[T](name+"_proj", projTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	// Zero-initialized bias.
	biasTensor, err := tensor.New[T]([]int{embedDim}, make([]T, embedDim))
	if err != nil {
		return nil, err
	}
	bias, err := graph.NewParameter[T](name+"_bias", biasTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	// Learnable frequency embedding for variate identity.
	freqData := make([]T, maxVariates*embedDim)
	freqScale := math.Sqrt(1.0 / float64(embedDim))
	for i := range freqData {
		freqData[i] = T(rand.NormFloat64() * freqScale)
	}
	freqTensor, err := tensor.New[T]([]int{maxVariates, embedDim}, freqData)
	if err != nil {
		return nil, err
	}
	freqEmb, err := graph.NewParameter[T](name+"_freq_emb", freqTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	return &VariateProjection[T]{
		name:     name,
		engine:   engine,
		ops:      ops,
		proj:     proj,
		bias:     bias,
		freqEmb:  freqEmb,
		inputDim: inputDim,
		embedDim: embedDim,
		maxVar:   maxVariates,
	}, nil
}

// OpType returns the operation type of the layer.
func (vp *VariateProjection[T]) OpType() string {
	return "VariateProjection"
}

// Attributes returns the attributes of the layer.
func (vp *VariateProjection[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_dim":    vp.inputDim,
		"embed_dim":    vp.embedDim,
		"max_variates": vp.maxVar,
	}
}

// OutputShape returns the output shape of the layer.
func (vp *VariateProjection[T]) OutputShape() []int {
	return []int{-1, -1, vp.embedDim} // [batch, numVariates, embedDim]
}

// Forward projects each variate independently and adds frequency embeddings.
//
// Input shape: [batch, numVariates, inputDim]
//   - Each variate is a time series of length inputDim.
//   - Variates shorter than inputDim should be zero-padded by the caller.
//   - numVariates must be <= maxVariates.
//
// Output shape: [batch, numVariates, embedDim]
//
// Optional second input: attention mask [batch, numVariates] with 1.0 for
// valid variates and 0.0 for padded variates. When provided, padded variate
// outputs are zeroed out.
func (vp *VariateProjection[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 || len(inputs) > 2 {
		return nil, fmt.Errorf("VariateProjection requires 1 or 2 inputs (data [, mask]), got %d", len(inputs))
	}

	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("VariateProjection input must be 3D [batch, numVariates, inputDim], got shape %v", shape)
	}

	batch := shape[0]
	numVar := shape[1]
	inDim := shape[2]

	if inDim != vp.inputDim {
		return nil, fmt.Errorf("input last dim = %d, want %d", inDim, vp.inputDim)
	}
	if numVar > vp.maxVar {
		return nil, fmt.Errorf("numVariates %d exceeds maxVariates %d", numVar, vp.maxVar)
	}

	// Reshape [batch, numVar, inputDim] -> [batch*numVar, inputDim] for projection.
	flat, err := vp.engine.Reshape(ctx, x, []int{batch * numVar, vp.inputDim})
	if err != nil {
		return nil, fmt.Errorf("flatten variates: %w", err)
	}

	// Project: [batch*numVar, inputDim] @ [inputDim, embedDim] -> [batch*numVar, embedDim]
	projected, err := vp.engine.MatMul(ctx, flat, vp.proj.Value)
	if err != nil {
		return nil, fmt.Errorf("variate projection: %w", err)
	}

	// Add bias: manually broadcast [embedDim] across [batch*numVar, embedDim].
	biasData := vp.bias.Value.Data()
	biasExpData := make([]T, batch*numVar*vp.embedDim)
	for row := range batch * numVar {
		copy(biasExpData[row*vp.embedDim:(row+1)*vp.embedDim], biasData)
	}
	biasExpanded, err := tensor.New[T]([]int{batch * numVar, vp.embedDim}, biasExpData)
	if err != nil {
		return nil, fmt.Errorf("broadcast bias: %w", err)
	}
	projected, err = vp.engine.Add(ctx, projected, biasExpanded)
	if err != nil {
		return nil, fmt.Errorf("add bias: %w", err)
	}

	// Reshape to [batch, numVar, embedDim].
	output, err := vp.engine.Reshape(ctx, projected, []int{batch, numVar, vp.embedDim})
	if err != nil {
		return nil, fmt.Errorf("reshape projected: %w", err)
	}

	// Add frequency embedding for variate identity.
	// Slice freqEmb to [numVar, embedDim], then broadcast to [batch, numVar, embedDim].
	freqData := vp.freqEmb.Value.Data()
	freqSlice := make([]T, numVar*vp.embedDim)
	copy(freqSlice, freqData[:numVar*vp.embedDim])
	freqTensor, err := tensor.New[T]([]int{numVar, vp.embedDim}, freqSlice)
	if err != nil {
		return nil, fmt.Errorf("slice freq embedding: %w", err)
	}

	// Broadcast [numVar, embedDim] -> [batch, numVar, embedDim] by repeating.
	freqBroadcast := make([]T, batch*numVar*vp.embedDim)
	for b := range batch {
		copy(freqBroadcast[b*numVar*vp.embedDim:(b+1)*numVar*vp.embedDim], freqTensor.Data())
	}
	freqBroad, err := tensor.New[T]([]int{batch, numVar, vp.embedDim}, freqBroadcast)
	if err != nil {
		return nil, fmt.Errorf("broadcast freq embedding: %w", err)
	}

	output, err = vp.engine.Add(ctx, output, freqBroad)
	if err != nil {
		return nil, fmt.Errorf("add freq embedding: %w", err)
	}

	// Apply attention mask if provided.
	if len(inputs) == 2 {
		mask := inputs[1]
		maskShape := mask.Shape()
		if len(maskShape) != 2 || maskShape[0] != batch || maskShape[1] != numVar {
			return nil, fmt.Errorf("mask shape must be [%d, %d], got %v", batch, numVar, maskShape)
		}

		// Expand mask [batch, numVar] -> [batch, numVar, embedDim] and multiply.
		maskData := mask.Data()
		maskExpanded := make([]T, batch*numVar*vp.embedDim)
		for b := range batch {
			for v := range numVar {
				m := maskData[b*numVar+v]
				base := (b*numVar + v) * vp.embedDim
				for d := range vp.embedDim {
					maskExpanded[base+d] = m
				}
			}
		}
		maskTensor, err := tensor.New[T]([]int{batch, numVar, vp.embedDim}, maskExpanded)
		if err != nil {
			return nil, fmt.Errorf("expand mask: %w", err)
		}
		output, err = vp.engine.Mul(ctx, output, maskTensor)
		if err != nil {
			return nil, fmt.Errorf("apply mask: %w", err)
		}
	}

	return output, nil
}

// Parameters returns the trainable parameters.
func (vp *VariateProjection[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{vp.proj, vp.bias, vp.freqEmb}
}

// SetName sets the name of the layer.
func (vp *VariateProjection[T]) SetName(name string) {
	vp.name = name
	vp.proj.Name = name + "_proj"
	vp.bias.Name = name + "_bias"
	vp.freqEmb.Name = name + "_freq_emb"
}

// Name returns the name of the layer.
func (vp *VariateProjection[T]) Name() string {
	return vp.name
}
