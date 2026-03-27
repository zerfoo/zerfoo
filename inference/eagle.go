package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// EAGLEConfig holds configuration for EAGLE-style self-speculative decoding.
type EAGLEConfig struct {
	NumDraftTokens int // number of draft tokens to generate per step
	HiddenDim      int // hidden dimension of the model
}

// BuildEAGLEHead constructs an EAGLEHead layer from an engine and config.
// The returned head can be used with GenerateDraftTokens to produce draft
// tokens from the penultimate transformer layer's output.
func BuildEAGLEHead[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	config EAGLEConfig,
) (*core.EAGLEHead[T], error) {
	if config.HiddenDim <= 0 {
		return nil, fmt.Errorf("EAGLEConfig: HiddenDim must be positive, got %d", config.HiddenDim)
	}
	if config.NumDraftTokens <= 0 {
		return nil, fmt.Errorf("EAGLEConfig: NumDraftTokens must be positive, got %d", config.NumDraftTokens)
	}
	return core.NewEAGLEHead[T](engine, ops, config.HiddenDim)
}

// lmHeadForward applies the LM head weight to hidden states to produce logits.
// hiddenStates shape: [batch, 1, hidden], weight shape: [vocab, hidden].
// Returns logits of shape [batch, 1, vocab].
func lmHeadForward[T tensor.Numeric](
	ctx context.Context,
	engine compute.Engine[T],
	hiddenStates *tensor.TensorNumeric[T],
	lmHeadWeight *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	shape := hiddenStates.Shape()
	batch, seqLen, hidden := shape[0], shape[1], shape[2]

	flat, err := engine.Reshape(ctx, hiddenStates, []int{batch * seqLen, hidden})
	if err != nil {
		return nil, fmt.Errorf("eagle lm head reshape: %w", err)
	}

	// weight is [vocab, hidden], compute flat * weight^T = [batch*seq, vocab]
	if tb, ok := engine.(compute.TransposeBMatMuler[T]); ok {
		out, err := tb.MatMulTransposeB(ctx, flat, lmHeadWeight)
		if err != nil {
			return nil, fmt.Errorf("eagle lm head matmul: %w", err)
		}
		vocabSize := lmHeadWeight.Shape()[0]
		return engine.Reshape(ctx, out, []int{batch, seqLen, vocabSize})
	}

	wT, err := engine.Transpose(ctx, lmHeadWeight, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("eagle lm head transpose: %w", err)
	}
	out, err := engine.MatMul(ctx, flat, wT)
	if err != nil {
		return nil, fmt.Errorf("eagle lm head matmul: %w", err)
	}
	vocabSize := lmHeadWeight.Shape()[0]
	return engine.Reshape(ctx, out, []int{batch, seqLen, vocabSize})
}

// argmaxLastPos returns the argmax token ID from the last sequence position
// of a [batch, seqLen, vocab] logits tensor (batch index 0).
func argmaxLastPos[T tensor.Numeric](logits *tensor.TensorNumeric[T]) int {
	shape := logits.Shape()
	vocabSize := shape[2]
	seqLen := shape[1]
	data := logits.Data()
	offset := (seqLen - 1) * vocabSize

	maxIdx := 0
	maxVal := data[offset]
	for i := 1; i < vocabSize; i++ {
		if data[offset+i] > maxVal {
			maxVal = data[offset+i]
			maxIdx = i
		}
	}
	return maxIdx
}

// GenerateDraftTokens uses an EAGLEHead to autoregressively generate draft
// token IDs from the penultimate transformer layer's hidden state.
//
// It feeds penultimateFeatures (shape [1, 1, hidden]) through the EAGLEHead,
// applies the LM head weight to get logits, takes argmax for a draft token,
// and repeats numDrafts times. Each iteration feeds the previous EAGLEHead
// output back as input for the next draft.
//
// Returns a slice of numDrafts draft token IDs.
func GenerateDraftTokens[T tensor.Numeric](
	ctx context.Context,
	eagleHead *core.EAGLEHead[T],
	engine compute.Engine[T],
	penultimateFeatures *tensor.TensorNumeric[T],
	lmHeadWeight *tensor.TensorNumeric[T],
	numDrafts int,
) ([]int, error) {
	if numDrafts <= 0 {
		return nil, fmt.Errorf("GenerateDraftTokens: numDrafts must be positive, got %d", numDrafts)
	}

	featureShape := penultimateFeatures.Shape()
	if len(featureShape) != 3 {
		return nil, fmt.Errorf("GenerateDraftTokens: penultimateFeatures must be 3D [batch, seq, hidden], got %dD", len(featureShape))
	}

	draftTokens := make([]int, 0, numDrafts)
	currentFeatures := penultimateFeatures

	for i := range numDrafts {
		// Feed through EAGLEHead to predict next hidden state.
		nextHidden, err := eagleHead.Forward(ctx, currentFeatures)
		if err != nil {
			return nil, fmt.Errorf("GenerateDraftTokens: step %d forward: %w", i, err)
		}

		// Apply LM head to get logits.
		logits, err := lmHeadForward(ctx, engine, nextHidden, lmHeadWeight)
		if err != nil {
			return nil, fmt.Errorf("GenerateDraftTokens: step %d lm head: %w", i, err)
		}

		// Argmax to get draft token.
		tokenID := argmaxLastPos(logits)
		draftTokens = append(draftTokens, tokenID)

		// Use the predicted hidden state as input for next iteration.
		currentFeatures = nextHidden
	}

	return draftTokens, nil
}
