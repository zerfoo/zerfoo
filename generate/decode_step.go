package generate

import (
	"context"
	"fmt"
	"strings"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// decodeStepResult holds the output of a single autoregressive decode step.
type decodeStepResult struct {
	// Token is the sampled token ID.
	Token int
	// Stop is true when the sampled token is in the stop set.
	Stop bool
}

// runDecodeStep executes a single autoregressive decode step: resets the arena
// pool, updates the reused tensor, runs graph forward (or compiled plan), and
// samples the next token. It handles plan compilation on the first decode step.
//
// decodeBuf must be a single-element slice backing tokenTensor (for in-place update).
// prevToken is the token to feed as input for this step.
func (gen *Generator[T]) runDecodeStep(
	ctx context.Context,
	genCtx context.Context,
	tokenTensor *tensor.TensorNumeric[T],
	decodeBuf []T,
	prevToken int,
	sc SamplingConfig,
	generatedIDs []int,
	stopSet map[int]bool,
) (decodeStepResult, error) {
	// Reset arena pool between tokens so intermediates are reclaimed.
	if resetter, ok := gen.engine.(compute.PoolResetter); ok {
		resetter.ResetPool()
	}

	// Update the reused tensor's value in-place.
	decodeBuf[0] = T(prevToken)

	var logits *tensor.TensorNumeric[T]
	var err error
	if p := gen.plan.Load(); p != nil {
		logits, err = p.Run(genCtx, tokenTensor)
	} else {
		logits, err = gen.graph.Forward(genCtx, tokenTensor)
		if err == nil {
			gen.compileGraph(genCtx, tokenTensor)
		}
	}
	if err != nil {
		return decodeStepResult{}, fmt.Errorf("decode forward: %w", err)
	}

	nextToken, err := gen.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		return decodeStepResult{}, fmt.Errorf("sample: %w", err)
	}

	return decodeStepResult{
		Token: nextToken,
		Stop:  stopSet[nextToken],
	}, nil
}

// verifyDraftTokens compares target logits against draft tokens for speculative
// decoding. It returns the accepted draft tokens and a bonus token (-1 if none).
// This is the shared implementation used by SpeculativeGenerator, EAGLEGenerator,
// and Generator.generateSpeculative.
func verifyDraftTokens[T tensor.Numeric](
	targetLogits *tensor.TensorNumeric[T],
	draftTokens []int,
	stopSet map[int]bool,
) (accepted []int, bonusToken int) {
	shape := targetLogits.Shape()
	vocabSize := shape[2]
	seqLen := shape[1]
	data := targetLogits.Data()

	accepted = make([]int, 0, len(draftTokens))
	bonusToken = -1

	for i, dt := range draftTokens {
		if i >= seqLen {
			break
		}

		offset := i * vocabSize
		targetToken := sliceArgmax(data[offset : offset+vocabSize])

		switch {
		case i == len(draftTokens)-1:
			// Last draft position: accept the draft token and use target's
			// prediction as the bonus token for the next step.
			accepted = append(accepted, dt)
			if !stopSet[dt] {
				bonusToken = targetToken
			}
		case targetToken == draftTokens[i+1]:
			// Target agrees with what draft predicted next. Accept current token.
			accepted = append(accepted, dt)
		default:
			// Target disagrees. Accept this token but use target's next-token
			// prediction instead of draft's.
			accepted = append(accepted, dt)
			bonusToken = targetToken
			return accepted, bonusToken
		}
	}

	return accepted, bonusToken
}

// emitVerifiedResult holds the outcome of emitting verified speculative tokens.
type emitVerifiedResult struct {
	// Emitted is the token IDs that were appended to generatedIDs.
	Emitted []int
	// Stop is true when generation should halt (stop token or max reached).
	Stop bool
}

// emitVerified appends accepted draft tokens and the bonus token to
// generatedIDs, stopping on any stop-set token or when maxNewTokens is
// reached. This is the shared emit loop used by SpeculativeGenerator,
// EAGLEGenerator, and Generator.generateSpeculative.
func emitVerified(
	accepted []int,
	bonusToken int,
	generatedIDs []int,
	maxNewTokens int,
	stopSet map[int]bool,
) (updatedIDs []int, stop bool) {
	for _, tok := range accepted {
		if stopSet[tok] {
			return generatedIDs, true
		}
		generatedIDs = append(generatedIDs, tok)
		if len(generatedIDs) >= maxNewTokens {
			return generatedIDs, true
		}
	}
	if bonusToken >= 0 {
		if stopSet[bonusToken] {
			return generatedIDs, true
		}
		generatedIDs = append(generatedIDs, bonusToken)
	}
	if len(generatedIDs) >= maxNewTokens {
		return generatedIDs, true
	}
	return generatedIDs, false
}

// sliceArgmax returns the index of the maximum value in a slice.
func sliceArgmax[T tensor.Numeric](data []T) int {
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

// logitsArgmaxLastPos returns the argmax token ID from the last sequence
// position of a [batch, seqLen, vocab] logits tensor (batch index 0).
func logitsArgmaxLastPos[T tensor.Numeric](logits *tensor.TensorNumeric[T]) int {
	shape := logits.Shape()
	vocabSize := shape[2]
	seqLen := shape[1]
	data := logits.Data()
	offset := (seqLen - 1) * vocabSize
	return sliceArgmax(data[offset : offset+vocabSize])
}

// tokenIDsToTensor converts token IDs to a [1, seqLen] input tensor.
func tokenIDsToTensor[T tensor.Numeric](ids []int) (*tensor.TensorNumeric[T], error) {
	data := make([]T, len(ids))
	for i, id := range ids {
		data[i] = T(id)
	}
	return tensor.New([]int{1, len(ids)}, data)
}

// incrementalCheckStop checks if the decoded generated tokens contain any stop
// string. It maintains a running decoded string across calls to avoid
// re-decoding all tokens on every step (which would be O(n^2) over a generation).
//
// This is the shared implementation used by SpeculativeGenerator, EAGLEGenerator,
// and Generator. The tok parameter provides the tokenizer for decoding.
func incrementalCheckStop(
	tok interface {
		Decode([]int) (string, error)
	},
	generatedIDs []int,
	stopStrings []string,
	prevDecoded *string,
	prevCount *int,
) (bool, string) {
	if len(generatedIDs) == *prevCount {
		return false, ""
	}

	if *prevCount > 0 {
		overlapIDs := generatedIDs[*prevCount-1:]
		overlapDecoded, err := tok.Decode(overlapIDs)
		if err != nil {
			return false, ""
		}
		singleDecoded, err := tok.Decode(generatedIDs[*prevCount-1 : *prevCount])
		if err != nil {
			return false, ""
		}
		fragment := overlapDecoded[len(singleDecoded):]
		*prevDecoded += fragment
	} else {
		decoded, err := tok.Decode(generatedIDs)
		if err != nil {
			return false, ""
		}
		*prevDecoded = decoded
	}
	*prevCount = len(generatedIDs)

	for _, ss := range stopStrings {
		if idx := strings.Index(*prevDecoded, ss); idx >= 0 {
			return true, (*prevDecoded)[:idx]
		}
	}
	return false, ""
}
