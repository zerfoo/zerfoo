// Package embeddings provides neural network embedding layers for the Zerfoo ML framework.
package embeddings

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/components"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// TokenEmbedding converts token IDs into dense vector representations.
type TokenEmbedding[T tensor.Numeric] struct {
	engine       compute.Engine[T]
	vocabSize    int // Size of the vocabulary
	embeddingDim int // Dimension of the embedding vectors

	// Trainable parameter: the embedding table
	embeddingTable *graph.Parameter[T]

	// Cached input for backward pass
	inputTokenIDs *tensor.TensorNumeric[int] // Input is int tensor for token IDs
	outputShape   []int
}

// TokenEmbeddingOptions holds configuration options for TokenEmbedding layers.
type TokenEmbeddingOptions[T tensor.Numeric] struct {
	Initializer components.WeightInitializer[T]
}

// TokenEmbeddingOption is a functional option for configuring TokenEmbedding layers.
type TokenEmbeddingOption[T tensor.Numeric] func(*TokenEmbeddingOptions[T])

// WithTokenEmbeddingInitializer sets a custom weight initializer for the embedding table.
func WithTokenEmbeddingInitializer[T tensor.Numeric](initializer components.WeightInitializer[T]) TokenEmbeddingOption[T] {
	return func(opts *TokenEmbeddingOptions[T]) {
		opts.Initializer = initializer
	}
}

// NewTokenEmbedding creates a new TokenEmbedding layer.
// vocabSize: The size of the vocabulary (number of unique tokens).
// embeddingDim: The dimension of the embedding vectors.
func NewTokenEmbedding[T tensor.Numeric](engine compute.Engine[T], vocabSize, embeddingDim int, options ...TokenEmbeddingOption[T]) (*TokenEmbedding[T], error) {
	if vocabSize <= 0 {
		return nil, fmt.Errorf("vocabSize must be positive, got %d", vocabSize)
	}

	if embeddingDim <= 0 {
		return nil, fmt.Errorf("embeddingDim must be positive, got %d", embeddingDim)
	}

	// Apply functional options
	opts := &TokenEmbeddingOptions[T]{}
	for _, option := range options {
		option(opts)
	}

	// Initialize the embedding table (vocabSize x embeddingDim)
	// This will be a trainable parameter.
	embeddingTableTensor, err := tensor.New[T]([]int{vocabSize, embeddingDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding table tensor: %w", err)
	}

	// Initialize embedding table with the provided initializer or default
	if opts.Initializer != nil {
		weights, err := opts.Initializer.Initialize(vocabSize, embeddingDim)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize embedding table with custom initializer: %w", err)
		}
		// Copy weights to tensor
		copy(embeddingTableTensor.Data(), weights)
	} else {
		// Default initialization: uniform random values
		if err := engine.RandomUniform(context.Background(), embeddingTableTensor, engine.Ops().FromFloat64(-0.05), engine.Ops().FromFloat64(0.05)); err != nil {
			return nil, fmt.Errorf("failed to initialize embedding table: %w", err)
		}
	}

	embeddingTable, err := graph.NewParameter[T]("embedding_table", embeddingTableTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding parameter: %w", err)
	}

	return &TokenEmbedding[T]{
		engine:         engine,
		vocabSize:      vocabSize,
		embeddingDim:   embeddingDim,
		embeddingTable: embeddingTable,
	}, nil
}

// NewTokenEmbeddingFromParam creates a new TokenEmbedding layer from an existing embedding table.
func NewTokenEmbeddingFromParam[T tensor.Numeric](engine compute.Engine[T], embeddingTable *graph.Parameter[T]) (*TokenEmbedding[T], error) {
	shape := embeddingTable.Value.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("embedding table must have 2 dimensions, got %d", len(shape))
	}

	return &TokenEmbedding[T]{
		engine:         engine,
		vocabSize:      shape[0],
		embeddingDim:   shape[1],
		embeddingTable: embeddingTable,
	}, nil
}

// OutputShape returns the output shape of the embedding layer.
func (te *TokenEmbedding[T]) OutputShape() []int {
	return te.outputShape
}

// Parameters returns the trainable embedding table.
func (te *TokenEmbedding[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{te.embeddingTable}
}

// Forward performs the embedding lookup.
// Input: A tensor of token IDs (T type).
// Output: A tensor of embedding vectors (T type).
func (te *TokenEmbedding[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TokenEmbedding expects 1 input, got %d", len(inputs))
	}
	// Accept indices as the same numeric type T and convert to int indices internally.
	tokenIDsT := inputs[0]
	inputShape := tokenIDsT.Shape()
	// Convert token IDs to int tensor (flattened indices)
	flatSize := 1
	for _, d := range inputShape {
		flatSize *= d
	}
	intData := make([]int, flatSize)
	dataT := tokenIDsT.Data()
	for i := 0; i < flatSize; i++ {
		switch v := any(dataT[i]).(type) {
		case float32:
			intData[i] = int(v)
		case float64:
			intData[i] = int(v)
		default:
			return nil, fmt.Errorf("TokenEmbedding requires input indices convertible to int; unsupported element type %T", v)
		}
	}
	tokenIDs, err := tensor.New[int](inputShape, intData)
	if err != nil {
		return nil, err
	}
	te.inputTokenIDs = tokenIDs // Cache for backward (original shape)

	// Prepare 1D indices for gather: [N]
	flatIDs, err := tensor.New[int]([]int{flatSize}, intData)
	if err != nil {
		return nil, err
	}

	// Output shape is inputShape + [embeddingDim]
	te.outputShape = append(append([]int{}, inputShape...), te.embeddingDim)

	// Allocate flat output [N, embeddingDim]
	outputFlat, err := tensor.New[T]([]int{flatSize, te.embeddingDim}, nil)
	if err != nil {
		return nil, err
	}

	// Perform embedding lookup with 1D indices => [N, dim]
	if err := te.engine.Gather(ctx, te.embeddingTable.Value, flatIDs, outputFlat); err != nil {
		return nil, err
	}

	// Reshape to [inputShape..., dim]
	output, err := te.engine.Reshape(ctx, outputFlat, te.outputShape)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for the embedding table.
func (te *TokenEmbedding[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The gradient for the embedding table is a sparse update.
	// For each token ID in inputTokenIDs, we add the corresponding dOut slice
	// to the gradient accumulator of that embedding vector in the embeddingTable.

	// Initialize gradient for embedding table to zeros
	dEmbeddingTable, err := tensor.New[T](te.embeddingTable.Value.Shape(), nil)
	if err != nil {
		return nil, err
	}

	if err := te.engine.Zeros(ctx, dEmbeddingTable, te.embeddingTable.Value.Shape()); err != nil { // Assuming Zeros is available
		return nil, err
	}

	// Scatter-add operation: add dOut slices to dEmbeddingTable based on token IDs
	// Reshape outputGradient from [inputShape..., embedding_dim] to [N, embedding_dim]
	ogShape := outputGradient.Shape()
	if len(ogShape) < 2 {
		return nil, fmt.Errorf("outputGradient must have at least 2 dims, got %v", ogShape)
	}
	embDim := ogShape[len(ogShape)-1]
	N := 1
	for i := 0; i < len(ogShape)-1; i++ {
		N *= ogShape[i]
	}
	reshapedDOut, err := te.engine.Reshape(ctx, outputGradient, []int{N, embDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape outputGradient for ScatterAdd: %w", err)
	}

	// Flatten inputTokenIDs from inputShape to [N]
	idsShape := te.inputTokenIDs.Shape()
	total := 1
	for _, d := range idsShape {
		total *= d
	}
	flatIDsData := make([]int, total)
	copy(flatIDsData, te.inputTokenIDs.Data())
	reshapedInputTokenIDs, err := tensor.New[int]([]int{total}, flatIDsData)
	if err != nil {
		return nil, fmt.Errorf("failed to build flattened inputTokenIDs for ScatterAdd: %w", err)
	}

	if err := te.engine.ScatterAdd(ctx, dEmbeddingTable, reshapedInputTokenIDs, reshapedDOut); err != nil { // Assuming ScatterAdd is available
		return nil, err
	}

	if err := te.embeddingTable.AddGradient(dEmbeddingTable); err != nil {
		return nil, err
	}

	// Embedding layer typically does not pass gradients back to its input (token IDs are discrete).
	// So, return nil for input gradients.
	return nil, nil
}

// OpType returns the operation type of the TokenEmbedding layer.
func (te *TokenEmbedding[T]) OpType() string {
	return "TokenEmbedding"
}

// Attributes returns the attributes of the TokenEmbedding layer.
func (te *TokenEmbedding[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"vocab_size":    te.vocabSize,
		"embedding_dim": te.embeddingDim,
	}
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*TokenEmbedding[float32])(nil)
