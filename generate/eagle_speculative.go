package generate

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/ztensor/compute"
	tokenizer "github.com/zerfoo/ztoken"
	"github.com/zerfoo/ztensor/tensor"
)

// EAGLEForwardResult holds both the logits and penultimate hidden states
// returned by a target model forward pass for EAGLE speculative decoding.
type EAGLEForwardResult[T tensor.Numeric] struct {
	Logits              *tensor.TensorNumeric[T] // [1, seqLen, vocabSize]
	PenultimateFeatures *tensor.TensorNumeric[T] // [1, seqLen, hiddenDim]
}

// EAGLEForwardFunc runs a target model forward pass and returns both logits
// and penultimate layer hidden states. The caller is responsible for wiring
// this to capture the output of the second-to-last transformer layer.
type EAGLEForwardFunc[T tensor.Numeric] func(ctx context.Context, input *tensor.TensorNumeric[T]) (*EAGLEForwardResult[T], error)

// EAGLEGenerator implements EAGLE-style self-speculative decoding. It uses
// the target model for verification and a lightweight EAGLEHead to draft
// tokens from the target's penultimate layer features — no separate draft
// model is needed.
//
// The decode loop:
//  1. Run target forward, capture penultimate features and logits.
//  2. Feed penultimate features to EAGLEHead for N draft tokens.
//  3. Verify all N draft tokens in a single batched target forward pass.
//  4. Accept the matching prefix, reject the rest.
//  5. Adaptively adjust N based on acceptance rate.
type EAGLEGenerator[T tensor.Numeric] struct {
	forwardFn    EAGLEForwardFunc[T]
	eagleHead    *core.EAGLEHead[T]
	tokenizer    tokenizer.Tokenizer
	engine       compute.Engine[T]
	lmHeadWeight *tensor.TensorNumeric[T]
	config       ModelConfig
	draftLen     int
	adaptive     bool
}

// NewEAGLEGenerator creates an EAGLE speculative generator.
//
// Parameters:
//   - forwardFn: target model forward that returns logits + penultimate features
//   - eagleHead: lightweight FFN that predicts next hidden state from penultimate features
//   - tok: tokenizer for encoding prompts and decoding output
//   - engine: compute engine for tensor operations
//   - lmHeadWeight: LM head weight tensor [vocabSize, hiddenDim] for draft token projection
//   - cfg: model configuration
//   - draftLen: initial number of draft tokens per step (default 4)
func NewEAGLEGenerator[T tensor.Numeric](
	forwardFn EAGLEForwardFunc[T],
	eagleHead *core.EAGLEHead[T],
	tok tokenizer.Tokenizer,
	engine compute.Engine[T],
	lmHeadWeight *tensor.TensorNumeric[T],
	cfg ModelConfig,
	draftLen int,
) *EAGLEGenerator[T] {
	if draftLen <= 0 {
		draftLen = 4
	}
	return &EAGLEGenerator[T]{
		forwardFn:    forwardFn,
		eagleHead:    eagleHead,
		tokenizer:    tok,
		engine:       engine,
		lmHeadWeight: lmHeadWeight,
		config:       cfg,
		draftLen:     draftLen,
		adaptive:     true,
	}
}

// WithAdaptive enables or disables adaptive draft length adjustment.
// When enabled (default), the draft length is adjusted based on acceptance rate.
func (eg *EAGLEGenerator[T]) WithAdaptive(enabled bool) *EAGLEGenerator[T] {
	eg.adaptive = enabled
	return eg
}

// Generate produces text from a prompt using EAGLE speculative decoding with
// greedy sampling. Generates identical output to vanilla autoregressive
// decoding (greedy).
func (eg *EAGLEGenerator[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error) {
	if sc.MaxNewTokens <= 0 {
		sc.MaxNewTokens = 256
	}

	promptIDs, err := eg.tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("eagle: encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return "", fmt.Errorf("eagle: prompt produced no tokens")
	}

	if eg.config.BOSTokenID > 0 {
		promptIDs = append([]int{eg.config.BOSTokenID}, promptIDs...)
	}

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[eg.config.EOSTokenID] = true

	// Prefill: run the full prompt through the target model.
	prefillTensor, err := tokenIDsToTensor[T](promptIDs)
	if err != nil {
		return "", fmt.Errorf("eagle: create prefill tensor: %w", err)
	}

	result, err := eg.forwardFn(ctx, prefillTensor)
	if err != nil {
		return "", fmt.Errorf("eagle: prefill forward: %w", err)
	}

	// Sample first token from target (greedy).
	firstToken := logitsArgmaxLastPos(result.Logits)
	if stopSet[firstToken] {
		return "", nil
	}

	generatedIDs := []int{firstToken}
	lastPenultimate := result.PenultimateFeatures

	// Running state for incremental stop-string checking.
	var runningDecoded string
	var decodedCount int

	var tracker *adaptiveDraftLen
	if eg.adaptive {
		tracker = newAdaptiveDraftLen(eg.draftLen, 1, 8, 32)
	}

	for len(generatedIDs) < sc.MaxNewTokens {
		if err := ctx.Err(); err != nil {
			break
		}

		currentDraftLen := eg.draftLen
		if tracker != nil {
			currentDraftLen = tracker.Current()
		}
		draftN := min(currentDraftLen, sc.MaxNewTokens-len(generatedIDs))

		// Draft phase: use EAGLEHead to generate draft tokens from
		// the target's penultimate features.
		draftTokens, dErr := eg.generateDraftTokens(ctx, lastPenultimate, draftN)
		if dErr != nil {
			return "", fmt.Errorf("eagle: generate draft tokens: %w", dErr)
		}

		if len(draftTokens) == 0 {
			break
		}

		// Verify phase: target processes all draft tokens in one batched
		// forward pass. We get logits + penultimate features for the next step.
		verifyTensor, tErr := tokenIDsToTensor[T](draftTokens)
		if tErr != nil {
			return "", fmt.Errorf("eagle: create verify tensor: %w", tErr)
		}

		verifyResult, fErr := eg.forwardFn(ctx, verifyTensor)
		if fErr != nil {
			return "", fmt.Errorf("eagle: verify forward: %w", fErr)
		}

		// Accept/reject: compare target's greedy output with draft tokens.
		accepted, bonusToken := verifyDraftTokens(verifyResult.Logits, draftTokens, stopSet)

		// Emit accepted tokens and bonus token.
		var stopped bool
		generatedIDs, stopped = emitVerified(accepted, bonusToken, generatedIDs, sc.MaxNewTokens, stopSet)
		if stopped {
			break
		}

		// Record acceptance rate for adaptive draft length.
		if tracker != nil {
			tracker.Record(len(accepted), len(draftTokens))
		}

		// Update penultimate features for the next draft step.
		// Use the features from the last accepted position in the verify result.
		lastPenultimate, err = eg.extractLastPosition(verifyResult.PenultimateFeatures, len(accepted)-1)
		if err != nil {
			return "", fmt.Errorf("eagle: extract penultimate features: %w", err)
		}

		// Check stop strings.
		if len(sc.StopStrings) > 0 {
			if stopped, text := incrementalCheckStop(eg.tokenizer, generatedIDs, sc.StopStrings, &runningDecoded, &decodedCount); stopped {
				return text, nil
			}
		}
	}

	if len(generatedIDs) == 0 {
		return "", nil
	}

	output, err := eg.tokenizer.Decode(generatedIDs)
	if err != nil {
		return "", fmt.Errorf("eagle: decode output: %w", err)
	}
	return output, nil
}

// generateDraftTokens uses the EAGLEHead to autoregressively generate draft
// token IDs from the penultimate transformer layer's hidden state.
func (eg *EAGLEGenerator[T]) generateDraftTokens(
	ctx context.Context,
	penultimateFeatures *tensor.TensorNumeric[T],
	numDrafts int,
) ([]int, error) {
	if numDrafts <= 0 {
		return nil, fmt.Errorf("eagle: numDrafts must be positive, got %d", numDrafts)
	}

	draftTokens := make([]int, 0, numDrafts)
	currentFeatures := penultimateFeatures

	for i := range numDrafts {
		nextHidden, err := eg.eagleHead.Forward(ctx, currentFeatures)
		if err != nil {
			return nil, fmt.Errorf("eagle: draft step %d forward: %w", i, err)
		}

		logits, err := eagleLMHeadForward(ctx, eg.engine, nextHidden, eg.lmHeadWeight)
		if err != nil {
			return nil, fmt.Errorf("eagle: draft step %d lm head: %w", i, err)
		}

		tokenID := logitsArgmaxLastPos(logits)
		draftTokens = append(draftTokens, tokenID)
		currentFeatures = nextHidden
	}

	return draftTokens, nil
}

// eagleLMHeadForward applies the LM head weight to hidden states to produce logits.
// hiddenStates shape: [batch, 1, hidden], weight shape: [vocab, hidden].
// Returns logits of shape [batch, 1, vocab].
func eagleLMHeadForward[T tensor.Numeric](
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

// extractLastPosition extracts a single sequence position from a
// [1, seqLen, hidden] tensor, returning a [1, 1, hidden] tensor.
func (eg *EAGLEGenerator[T]) extractLastPosition(
	features *tensor.TensorNumeric[T],
	pos int,
) (*tensor.TensorNumeric[T], error) {
	shape := features.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("expected 3D features, got %dD", len(shape))
	}
	seqLen := shape[1]
	hidden := shape[2]

	if pos < 0 {
		pos = 0
	}
	if pos >= seqLen {
		pos = seqLen - 1
	}

	data := features.Data()
	start := pos * hidden
	end := start + hidden
	posData := make([]T, hidden)
	copy(posData, data[start:end])

	return tensor.New([]int{1, 1, hidden}, posData)
}

