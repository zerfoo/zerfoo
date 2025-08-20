package transformer

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// GemmaStack implements a stack of TransformerBlocks with local-global attention scheduling.
type GemmaStack[T tensor.Numeric] struct {
	layers      []graph.Node[T]
	outputShape []int
}

// NewGemmaStack creates a new GemmaStack.
func NewGemmaStack[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads, ffnDim int,
	epsilon T,
	base float64,
	maxSeqLen, numLayers, localWindowSize, globalInterval int,
) (*GemmaStack[T], error) {
	var layers []graph.Node[T]
	for i := 0; i < numLayers; i++ {
		var block graph.Node[T]
		var err error
		if (i+1)%globalInterval == 0 {
			attn, err := attention.NewGlobalSelfAttention[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, epsilon, base, maxSeqLen)
			if err != nil {
				return nil, fmt.Errorf("failed to create layer %d: %w", i, err)
			}
			// Scale RoPE embeddings for global attention layers
			if err := attn.ScaleRope(context.Background(), 0.5); err != nil {
				return nil, fmt.Errorf("failed to scale rope for layer %d: %w", i, err)
			}
			block = attn
		} else {
			block, err = attention.NewLocalSlidingWindowAttention[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, localWindowSize, epsilon, base, maxSeqLen)
		}
		if err != nil {
			return nil, fmt.Errorf("failed to create layer %d: %w", i, err)
		}
		layers = append(layers, block)
	}

	return &GemmaStack[T]{
		layers: layers,
	}, nil
}

// OutputShape returns the output shape of the GemmaStack.
func (s *GemmaStack[T]) OutputShape() []int {
	return s.outputShape
}

// Parameters returns the parameters of the GemmaStack.
func (s *GemmaStack[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	for _, layer := range s.layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

// Forward computes the forward pass of the GemmaStack.
func (s *GemmaStack[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	x := inputs[0]
	for _, layer := range s.layers {
		var err error
		x, err = layer.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
	}
	s.outputShape = x.Shape()
	return x, nil
}

// Backward computes the backward pass of the GemmaStack.
func (s *GemmaStack[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	// This is a simplified backward pass. A real implementation would need to
	// correctly chain the gradients through the layers.
	return nil, fmt.Errorf("GemmaStack backward pass not yet implemented")
}
