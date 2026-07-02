// Package speculative implements speculative decoding strategies for
// accelerating autoregressive text generation.
package speculative

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// ForwardFunc runs a model forward pass on the given input tokens and returns
// logits shaped [1, seqLen, vocabSize]. The implementation decides how many
// transformer layers to execute — callers use a partial-layer function for
// drafting and the full model for verification.
type ForwardFunc[T tensor.Numeric] func(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)

// SelfDraft implements self-speculative decoding: the same model is used for
// both drafting and verification. Drafting runs only the first N/2 layers
// (early exit), producing cheap approximate tokens. Verification runs the
// full model to accept or reject draft tokens.
//
// This avoids loading a separate draft model, reducing memory at the cost of
// draft quality (typically alpha > 0.4 for well-trained models).
type SelfDraft[T tensor.Numeric] struct {
	draftFn    ForwardFunc[T] // partial-layer forward (first N/2 layers)
	verifyFn   ForwardFunc[T] // full-model forward (all N layers)
	vocabSize  int
	numLayers  int // total layers in the full model
	draftDepth int // number of layers used for drafting (N/2)
}

// NewSelfDraft creates a SelfDraft speculative decoder.
//
// Parameters:
//   - draftFn: forward function using only the first draftDepth layers
//   - verifyFn: forward function using all layers
//   - vocabSize: model vocabulary size
//   - numLayers: total transformer layers in the full model
//   - draftDepth: number of layers to use for drafting (typically numLayers/2)
func NewSelfDraft[T tensor.Numeric](
	draftFn, verifyFn ForwardFunc[T],
	vocabSize, numLayers, draftDepth int,
) *SelfDraft[T] {
	if draftDepth <= 0 {
		draftDepth = numLayers / 2
	}
	if draftDepth >= numLayers {
		draftDepth = numLayers / 2
	}
	if draftDepth <= 0 {
		draftDepth = 1
	}
	return &SelfDraft[T]{
		draftFn:    draftFn,
		verifyFn:   verifyFn,
		vocabSize:  vocabSize,
		numLayers:  numLayers,
		draftDepth: draftDepth,
	}
}

// DraftDepth returns the number of layers used for drafting.
func (sd *SelfDraft[T]) DraftDepth() int { return sd.draftDepth }

// Generate produces K draft tokens using the partial-layer forward function.
// Each draft step feeds the previous draft token back as input. The returned
// slice contains up to K token IDs.
func (sd *SelfDraft[T]) Generate(ctx context.Context, tokens []int, K int) ([]int, error) {
	if K <= 0 {
		return nil, nil
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("self_draft: empty input tokens")
	}

	draft := make([]int, 0, K)
	nextInput := tokens[len(tokens)-1]

	for range K {
		if err := ctx.Err(); err != nil {
			break
		}

		inputTensor, err := idsToTensor[T]([]int{nextInput})
		if err != nil {
			return nil, fmt.Errorf("self_draft: create input tensor: %w", err)
		}

		logits, err := sd.draftFn(ctx, inputTensor)
		if err != nil {
			return nil, fmt.Errorf("self_draft: draft forward: %w", err)
		}

		token := greedyArgmax(logits)
		draft = append(draft, token)
		nextInput = token
	}

	return draft, nil
}

// Verify checks draft tokens against the full model. It runs the full model
// on the draft token sequence and returns the number of accepted tokens
// (where the full model's greedy prediction matches the draft).
func (sd *SelfDraft[T]) Verify(ctx context.Context, draftTokens []int) (accepted int, correction int, err error) {
	if len(draftTokens) == 0 {
		return 0, -1, nil
	}

	inputTensor, err := idsToTensor[T](draftTokens)
	if err != nil {
		return 0, -1, fmt.Errorf("self_draft: create verify tensor: %w", err)
	}

	logits, err := sd.verifyFn(ctx, inputTensor)
	if err != nil {
		return 0, -1, fmt.Errorf("self_draft: verify forward: %w", err)
	}

	shape := logits.Shape()
	if len(shape) != 3 {
		return 0, -1, fmt.Errorf("self_draft: expected 3D logits, got shape %v", shape)
	}
	seqLen := shape[1]
	vocabSize := shape[2]
	data := logits.Data()

	// Compare: at position i, the full model predicts what comes after
	// draftTokens[i]. If that matches draftTokens[i+1], accept.
	accepted = 0
	correction = -1
	for i := 0; i < len(draftTokens) && i < seqLen; i++ {
		offset := i * vocabSize
		targetToken := argmaxSlice(data[offset : offset+vocabSize])

		if i == len(draftTokens)-1 {
			// Last position: the full model's prediction is the "bonus" token.
			accepted = len(draftTokens)
			correction = targetToken
			return accepted, correction, nil
		}

		if targetToken != draftTokens[i+1] {
			// Mismatch: accept tokens up to i, use target's prediction as correction.
			accepted = i + 1
			correction = targetToken
			return accepted, correction, nil
		}
	}

	accepted = len(draftTokens)
	return accepted, correction, nil
}

// AcceptanceRate measures the fraction of draft tokens accepted by the full
// model (alpha). It generates K draft tokens from the prompt, then verifies
// them. Returns alpha in [0, 1].
func (sd *SelfDraft[T]) AcceptanceRate(ctx context.Context, prompt []int, K int) (float64, error) {
	if K <= 0 {
		K = 4
	}

	draftTokens, err := sd.Generate(ctx, prompt, K)
	if err != nil {
		return 0, fmt.Errorf("self_draft: generate draft: %w", err)
	}

	if len(draftTokens) == 0 {
		return 0, nil
	}

	accepted, _, err := sd.Verify(ctx, draftTokens)
	if err != nil {
		return 0, fmt.Errorf("self_draft: verify draft: %w", err)
	}

	return float64(accepted) / float64(len(draftTokens)), nil
}

// idsToTensor converts token IDs to a [1, seqLen] input tensor.
func idsToTensor[T tensor.Numeric](ids []int) (*tensor.TensorNumeric[T], error) {
	data := make([]T, len(ids))
	for i, id := range ids {
		data[i] = T(id)
	}
	return tensor.New([]int{1, len(ids)}, data)
}

// greedyArgmax returns the argmax of the last position in a [1, seqLen, vocab] tensor.
func greedyArgmax[T tensor.Numeric](logits *tensor.TensorNumeric[T]) int {
	shape := logits.Shape()
	vocabSize := shape[2]
	seqLen := shape[1]
	data := logits.Data()
	lastStart := (seqLen - 1) * vocabSize
	return argmaxSlice(data[lastStart : lastStart+vocabSize])
}

// argmaxSlice returns the index of the maximum value in a slice.
func argmaxSlice[T tensor.Numeric](data []T) int {
	maxIdx := 0
	maxVal := data[0]
	for i := 1; i < len(data); i++ {
		if data[i] > maxVal {
			maxVal = data[i]
			maxIdx = i
		}
	}
	return maxIdx
}
