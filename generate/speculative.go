package generate

import (
	"context"
	"fmt"
	"strings"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	tokenizer "github.com/zerfoo/ztoken"
	"github.com/zerfoo/ztensor/tensor"
)

// SpeculativeGenerator implements speculative decoding using a small draft
// model and a large target model. The draft model proposes N tokens greedily,
// then the target model verifies all N in a single batched forward pass.
// Accepted tokens are emitted; on first mismatch the target's token is used.
type SpeculativeGenerator[T tensor.Numeric] struct {
	draftGraph  *graph.Graph[T]
	targetGraph *graph.Graph[T]
	tokenizer   tokenizer.Tokenizer
	engine      compute.Engine[T]
	draftCfg    ModelConfig
	targetCfg   ModelConfig
	draftLen    int // initial draft tokens per step (default 4)
	adaptive    bool
}

// NewSpeculativeGenerator creates a speculative generator with separate draft
// and target model graphs. draftLen controls how many tokens the draft model
// proposes per verification step (typically 2-8).
func NewSpeculativeGenerator[T tensor.Numeric](
	draftGraph, targetGraph *graph.Graph[T],
	tok tokenizer.Tokenizer,
	engine compute.Engine[T],
	draftCfg, targetCfg ModelConfig,
	draftLen int,
) *SpeculativeGenerator[T] {
	if draftLen <= 0 {
		draftLen = 4
	}
	return &SpeculativeGenerator[T]{
		draftGraph:  draftGraph,
		targetGraph: targetGraph,
		tokenizer:   tok,
		engine:      engine,
		draftCfg:    draftCfg,
		targetCfg:   targetCfg,
		draftLen:    draftLen,
		adaptive:    true,
	}
}

// WithAdaptive enables or disables adaptive draft length adjustment.
// When enabled (default), the draft length is adjusted based on acceptance rate.
func (sg *SpeculativeGenerator[T]) WithAdaptive(enabled bool) *SpeculativeGenerator[T] {
	sg.adaptive = enabled
	return sg
}

// Generate produces text from a prompt using speculative decoding with greedy
// sampling. The draft model proposes tokens, the target model verifies them.
func (sg *SpeculativeGenerator[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error) {
	if sc.MaxNewTokens <= 0 {
		sc.MaxNewTokens = 256
	}

	promptIDs, err := sg.tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[sg.targetCfg.EOSTokenID] = true

	// Create KV caches for both models.
	draftCache := NewKVCache[T](sg.draftCfg.NumLayers, sg.draftCfg.MaxSeqLen)
	targetCache := NewKVCache[T](sg.targetCfg.NumLayers, sg.targetCfg.MaxSeqLen)

	draftCtx := WithCache(ctx, CacheProvider[T](draftCache))
	targetCtx := WithCache(ctx, CacheProvider[T](targetCache))

	// Prefill both models with the prompt.
	prefillTensor, err := sg.idsToTensor(promptIDs)
	if err != nil {
		return "", fmt.Errorf("create prefill tensor: %w", err)
	}

	_, err = sg.draftGraph.Forward(draftCtx, prefillTensor)
	if err != nil {
		return "", fmt.Errorf("draft prefill: %w", err)
	}

	targetLogits, err := sg.targetGraph.Forward(targetCtx, prefillTensor)
	if err != nil {
		return "", fmt.Errorf("target prefill: %w", err)
	}

	// Sample first token from target.
	firstToken := sg.greedyArgmax(targetLogits)
	if stopSet[firstToken] {
		return "", nil
	}

	generatedIDs := []int{firstToken}
	nextDraftInput := firstToken

	var tracker *adaptiveDraftLen
	if sg.adaptive {
		tracker = newAdaptiveDraftLen(sg.draftLen, 1, 8, 32)
	}

	for len(generatedIDs) < sc.MaxNewTokens {
		if err := ctx.Err(); err != nil {
			break
		}

		currentDraftLen := sg.draftLen
		if tracker != nil {
			currentDraftLen = tracker.Current()
		}
		draftN := min(currentDraftLen, sc.MaxNewTokens-len(generatedIDs))

		// Draft phase: generate draftN tokens greedily.
		draftTokens := make([]int, 0, draftN)
		draftInput := nextDraftInput

		for range draftN {
			tokenTensor, tErr := sg.idsToTensor([]int{draftInput})
			if tErr != nil {
				return "", fmt.Errorf("draft token tensor: %w", tErr)
			}

			draftLogits, fErr := sg.draftGraph.Forward(draftCtx, tokenTensor)
			if fErr != nil {
				return "", fmt.Errorf("draft forward: %w", fErr)
			}

			draftToken := sg.greedyArgmax(draftLogits)
			draftTokens = append(draftTokens, draftToken)

			if stopSet[draftToken] {
				break
			}
			draftInput = draftToken
		}

		if len(draftTokens) == 0 {
			break
		}

		// Verify phase: target processes all draft tokens in one forward pass.
		verifyTensor, tErr := sg.idsToTensor(draftTokens)
		if tErr != nil {
			return "", fmt.Errorf("verify tensor: %w", tErr)
		}

		verifyLogits, fErr := sg.targetGraph.Forward(targetCtx, verifyTensor)
		if fErr != nil {
			return "", fmt.Errorf("target verify forward: %w", fErr)
		}

		// Accept/reject: compare target's greedy output with draft tokens.
		// Target logits at position i predict what comes AFTER draft token i.
		// We already accepted the "previous" token (firstToken or last accepted).
		// Now we check: does target agree with the draft for the remaining?
		accepted, bonusToken := sg.verifyTokens(verifyLogits, draftTokens, stopSet)

		// Emit accepted draft tokens.
		stopped := false
		for _, tok := range accepted {
			if stopSet[tok] {
				stopped = true
				break
			}
			generatedIDs = append(generatedIDs, tok)
			if len(generatedIDs) >= sc.MaxNewTokens {
				stopped = true
				break
			}
		}
		if stopped {
			break
		}

		// Emit the bonus token (target's correction or next token).
		if bonusToken >= 0 {
			if stopSet[bonusToken] {
				break
			}
			generatedIDs = append(generatedIDs, bonusToken)
		}

		if len(generatedIDs) >= sc.MaxNewTokens {
			break
		}

		// Record acceptance rate for adaptive draft length.
		if tracker != nil {
			tracker.Record(len(accepted), len(draftTokens))
		}

		// Roll back caches if tokens were rejected.
		// The target cache has the verified tokens + all draft tokens.
		// The draft cache has the prompt + generated + all draft tokens.
		// Truncate both to the correct position: prompt + accepted + bonus.
		correctSeqLen := len(promptIDs) + len(generatedIDs)
		if draftCache.SeqLen() > correctSeqLen {
			draftCache.Truncate(correctSeqLen)
		}
		if targetCache.SeqLen() > correctSeqLen {
			targetCache.Truncate(correctSeqLen)
		}

		if len(generatedIDs) > 0 {
			nextDraftInput = generatedIDs[len(generatedIDs)-1]
		}

		// Check stop strings.
		if len(sc.StopStrings) > 0 {
			if stopped, text := sg.checkStop(generatedIDs, sc.StopStrings); stopped {
				return text, nil
			}
		}
	}

	if len(generatedIDs) == 0 {
		return "", nil
	}

	result, err := sg.tokenizer.Decode(generatedIDs)
	if err != nil {
		return "", fmt.Errorf("decode output: %w", err)
	}
	return result, nil
}

// verifyTokens compares target logits against draft tokens.
// Returns the accepted draft tokens and a bonus token (-1 if none).
func (sg *SpeculativeGenerator[T]) verifyTokens(
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

		// Target's greedy prediction at position i.
		offset := i * vocabSize
		targetToken := sg.argmaxSlice(data[offset : offset+vocabSize])

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

// greedyArgmax returns the argmax of the last position in a [1, seqLen, vocab] tensor.
func (sg *SpeculativeGenerator[T]) greedyArgmax(logits *tensor.TensorNumeric[T]) int {
	shape := logits.Shape()
	vocabSize := shape[2]
	seqLen := shape[1]
	data := logits.Data()
	lastStart := (seqLen - 1) * vocabSize
	return sg.argmaxSlice(data[lastStart : lastStart+vocabSize])
}

// argmaxSlice returns the index of the maximum value in a slice.
func (sg *SpeculativeGenerator[T]) argmaxSlice(data []T) int {
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

// idsToTensor converts token IDs to a [1, seqLen] input tensor.
func (sg *SpeculativeGenerator[T]) idsToTensor(ids []int) (*tensor.TensorNumeric[T], error) {
	data := make([]T, len(ids))
	for i, id := range ids {
		data[i] = T(id)
	}
	return tensor.New([]int{1, len(ids)}, data)
}

// checkStop checks if the decoded generated tokens contain any stop string.
func (sg *SpeculativeGenerator[T]) checkStop(generatedIDs []int, stopStrings []string) (bool, string) {
	decoded, err := sg.tokenizer.Decode(generatedIDs)
	if err != nil {
		return false, ""
	}
	for _, ss := range stopStrings {
		if idx := strings.Index(decoded, ss); idx >= 0 {
			return true, decoded[:idx]
		}
	}
	return false, ""
}
