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
		var attentionLayer graph.Node[T]
		var err error
		if (i+1)%globalInterval == 0 {
			attn, err := attention.NewGlobalAttention[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, base, maxSeqLen)
			if err != nil {
				return nil, fmt.Errorf("failed to create global attention layer for block %d: %w", i, err)
			}
			// Scale RoPE embeddings for global attention layers
			var iface interface{} = attn
			if scaler, ok := iface.(attention.RopeScaler[T]); ok {
				if err := scaler.ScaleRope(context.Background(), 0.5); err != nil {
					return nil, fmt.Errorf("failed to scale rope for layer %d: %w", i, err)
				}
			}
			attentionLayer = attn
		} else {
			attentionLayer, err = attention.NewLocalAttention[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, localWindowSize, base, maxSeqLen)
		}
		if err != nil {
			return nil, fmt.Errorf("failed to create attention layer for block %d: %w", i, err)
		}

		block, err := NewTransformerBlock[T](engine, ops, modelDim, ffnDim, epsilon, attentionLayer)
		if err != nil {
			return nil, fmt.Errorf("failed to create transformer block %d: %w", i, err)
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
	var err error
	d := dOut
	for i := len(s.layers) - 1; i >= 0; i-- {
		var input *tensor.Tensor[T]
		if i > 0 {
			// This is a simplification. We should have cached the input to each layer.
			// For now, we assume the input to the stack is the input to the first layer.
			// This will not work for a real training scenario.
			input = inputs[0]
		} else {
			input = inputs[0]
		}
		grads, err := s.layers[i].Backward(ctx, d, input)
		if err != nil {
			return nil, err
		}
		d = grads[0]
	}
	return []*tensor.Tensor[T]{d}, err
}
