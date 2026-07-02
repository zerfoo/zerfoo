package timeseries

import (
	"context"
	"fmt"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// PatchEmbed splits a 1D time series into non-overlapping patches and
// projects each patch to embed_dim using a learned linear projection.
type PatchEmbed[T tensor.Numeric] struct {
	name      string
	engine    compute.Engine[T]
	ops       numeric.Arithmetic[T]
	proj      *graph.Parameter[T] // [patch_size, embed_dim]
	patchSize int
	embedDim  int
}

// NewPatchEmbed creates a new PatchEmbed layer.
func NewPatchEmbed[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	patchSize, embedDim int,
) (*PatchEmbed[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if patchSize <= 0 {
		return nil, fmt.Errorf("patch_size must be positive, got %d", patchSize)
	}
	if embedDim <= 0 {
		return nil, fmt.Errorf("embed_dim must be positive, got %d", embedDim)
	}

	data := make([]T, patchSize*embedDim)
	for i := range data {
		data[i] = T(rand.Float32())
	}
	projTensor, err := tensor.New[T]([]int{patchSize, embedDim}, data)
	if err != nil {
		return nil, err
	}
	proj, err := graph.NewParameter[T](name+"_proj", projTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	return &PatchEmbed[T]{
		name:      name,
		engine:    engine,
		ops:       ops,
		proj:      proj,
		patchSize: patchSize,
		embedDim:  embedDim,
	}, nil
}

// OpType returns the operation type of the layer.
func (pe *PatchEmbed[T]) OpType() string {
	return "PatchEmbed"
}

// Attributes returns the attributes of the layer.
func (pe *PatchEmbed[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"patch_size": pe.patchSize,
		"embed_dim":  pe.embedDim,
	}
}

// OutputShape returns the output shape of the layer.
func (pe *PatchEmbed[T]) OutputShape() []int {
	return []int{-1, -1, pe.embedDim} // [batch, num_patches, embed_dim]
}

// Forward takes [batch, seq_len] input and returns [batch, num_patches, embed_dim].
// seq_len is padded with zeros if not divisible by PatchSize.
func (pe *PatchEmbed[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("PatchEmbed requires exactly one input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("PatchEmbed input must be 2D [batch, seq_len], got shape %v", shape)
	}

	batch := shape[0]
	seqLen := shape[1]

	// Pad if seq_len is not divisible by patch_size.
	if seqLen%pe.patchSize != 0 {
		paddedLen := ((seqLen + pe.patchSize - 1) / pe.patchSize) * pe.patchSize
		padded, err := tensor.New[T]([]int{batch, paddedLen}, make([]T, batch*paddedLen))
		if err != nil {
			return nil, err
		}
		// Copy original data into padded tensor.
		srcData := x.Data()
		dstData := padded.Data()
		for b := 0; b < batch; b++ {
			copy(dstData[b*paddedLen:b*paddedLen+seqLen], srcData[b*seqLen:(b+1)*seqLen])
		}
		x = padded
		seqLen = paddedLen
	}

	numPatches := seqLen / pe.patchSize

	// Reshape [batch, seq_len] -> [batch * num_patches, patch_size]
	reshaped, err := pe.engine.Reshape(ctx, x, []int{batch * numPatches, pe.patchSize})
	if err != nil {
		return nil, fmt.Errorf("reshape to patches: %w", err)
	}

	// Project: [batch * num_patches, patch_size] @ [patch_size, embed_dim] -> [batch * num_patches, embed_dim]
	projected, err := pe.engine.MatMul(ctx, reshaped, pe.proj.Value)
	if err != nil {
		return nil, fmt.Errorf("patch projection: %w", err)
	}

	// Reshape [batch * num_patches, embed_dim] -> [batch, num_patches, embed_dim]
	output, err := pe.engine.Reshape(ctx, projected, []int{batch, numPatches, pe.embedDim})
	if err != nil {
		return nil, fmt.Errorf("reshape to output: %w", err)
	}

	return output, nil
}

// Backward computes the gradients for the patch embedding layer.
func (pe *PatchEmbed[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("PatchEmbed requires exactly one input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// Determine padded length.
	paddedLen := seqLen
	if seqLen%pe.patchSize != 0 {
		paddedLen = ((seqLen + pe.patchSize - 1) / pe.patchSize) * pe.patchSize
		padded, err := tensor.New[T]([]int{batch, paddedLen}, make([]T, batch*paddedLen))
		if err != nil {
			return nil, err
		}
		srcData := x.Data()
		dstData := padded.Data()
		for b := 0; b < batch; b++ {
			copy(dstData[b*paddedLen:b*paddedLen+seqLen], srcData[b*seqLen:(b+1)*seqLen])
		}
		x = padded
	}
	numPatches := paddedLen / pe.patchSize

	// outputGradient: [batch, num_patches, embed_dim]
	// Reshape to [batch * num_patches, embed_dim]
	gradReshaped, err := pe.engine.Reshape(ctx, outputGradient, []int{batch * numPatches, pe.embedDim})
	if err != nil {
		return nil, err
	}

	// Gradient w.r.t. projection weights:
	// dW = patches^T @ gradReshaped
	// patches: [batch * num_patches, patch_size]
	patches, err := pe.engine.Reshape(ctx, x, []int{batch * numPatches, pe.patchSize})
	if err != nil {
		return nil, err
	}
	patchesT, err := pe.engine.Transpose(ctx, patches, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dw, err := pe.engine.MatMul(ctx, patchesT, gradReshaped)
	if err != nil {
		return nil, err
	}
	pe.proj.Gradient, err = pe.engine.Add(ctx, pe.proj.Gradient, dw, pe.proj.Gradient)
	if err != nil {
		return nil, err
	}

	// Gradient w.r.t. input:
	// dx_patches = gradReshaped @ proj^T -> [batch * num_patches, patch_size]
	projT, err := pe.engine.Transpose(ctx, pe.proj.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dxPatches, err := pe.engine.MatMul(ctx, gradReshaped, projT)
	if err != nil {
		return nil, err
	}

	// Reshape back to [batch, paddedLen]
	dxPadded, err := pe.engine.Reshape(ctx, dxPatches, []int{batch, paddedLen})
	if err != nil {
		return nil, err
	}

	// If we padded, slice back to original seq_len.
	if paddedLen != seqLen {
		dxData := dxPadded.Data()
		origData := make([]T, batch*seqLen)
		for b := 0; b < batch; b++ {
			copy(origData[b*seqLen:(b+1)*seqLen], dxData[b*paddedLen:b*paddedLen+seqLen])
		}
		dx, err := tensor.New[T]([]int{batch, seqLen}, origData)
		if err != nil {
			return nil, err
		}
		return []*tensor.TensorNumeric[T]{dx}, nil
	}

	return []*tensor.TensorNumeric[T]{dxPadded}, nil
}

// Parameters returns the trainable parameters.
func (pe *PatchEmbed[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{pe.proj}
}

// SetName sets the name of the layer.
func (pe *PatchEmbed[T]) SetName(name string) {
	pe.name = name
	pe.proj.Name = name + "_proj"
}

// Name returns the name of the layer.
func (pe *PatchEmbed[T]) Name() string {
	return pe.name
}
