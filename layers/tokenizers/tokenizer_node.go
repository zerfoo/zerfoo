package tokenizers

import (
	"context"
	"fmt"
	"strings"

	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// TokenizerNode converts a tensor of strings into a tensor of integer token IDs.
// NOTE: This implementation assumes a flexible Node interface that can handle
// different tensor types, not one strictly tied to numerics.
type TokenizerNode struct {
	vocab      map[string]int32
	unkTokenID int32
}

// NewTokenizerNode creates a new node for tokenization.
// The vocabulary maps string tokens to their integer IDs.
// unkTokenID is the ID to use for tokens not found in the vocabulary.
func NewTokenizerNode(vocab map[string]int32, unkTokenID int32) *TokenizerNode {
	return &TokenizerNode{
		vocab:      vocab,
		unkTokenID: unkTokenID,
	}
}

// Forward performs the tokenization.
// It expects a single input: a 1D TensorString.
// It outputs a 2D TensorNumeric[int32] with shape [1, sequence_length].
func (n *TokenizerNode) Forward(ctx context.Context, inputs ...tensor.Tensor) (tensor.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TokenizerNode expects 1 input, got %d", len(inputs))
	}

	input, ok := inputs[0].(*tensor.TensorString)
	if !ok {
		return nil, fmt.Errorf("TokenizerNode expects a TensorString input, got %T", inputs[0])
	}

	// For simplicity, this tokenizer handles one string at a time (batch size 1).
	// A more advanced version would handle a batch of strings.
	if len(input.Shape()) != 1 || input.Shape()[0] != 1 {
		return nil, fmt.Errorf("TokenizerNode currently expects a 1D tensor with a single string, shape [1]")
	}

	rawText := input.Data()[0]
	tokens := strings.Fields(strings.ToLower(rawText)) // Simple whitespace and lowercase tokenization

	tokenIDs := make([]int32, len(tokens))
	for i, token := range tokens {
		if id, found := n.vocab[token]; found {
			tokenIDs[i] = id
		} else {
			tokenIDs[i] = n.unkTokenID
		}
	}

	// Output shape is [1, sequence_length] to represent a single batch item.
	outputShape := []int{1, len(tokenIDs)}

	return tensor.New[int32](outputShape, tokenIDs)
}

// Backward is not implemented for TokenizerNode as it is not a differentiable operation.
func (n *TokenizerNode) Backward(ctx context.Context, mode types.BackwardMode, outputGradient tensor.Tensor) ([]tensor.Tensor, error) {
	return nil, fmt.Errorf("tokenization is not a differentiable operation")
}

// OutputShape returns the shape of the output tensor.
// Since the sequence length is dynamic, we can represent it with -1.
func (n *TokenizerNode) OutputShape() []int {
	return []int{1, -1} // [batch_size, dynamic_sequence_length]
}

// OpType returns the type of the node.
func (n *TokenizerNode) OpType() string {
	return "Tokenizer"
}

// Attributes returns no attributes for this node.
func (n *TokenizerNode) Attributes() map[string]any {
	return nil
}
