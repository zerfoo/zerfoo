package generate

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// WithPJRTPlan configures the generator to use a pre-compiled PJRTPlan for
// inference. When set, the generator uses RunPrefill/RunDecode instead of
// the standard graph Forward and ExecutionPlan paths. The KV cache is managed
// internally by the PJRTPlan.
func WithPJRTPlan[T tensor.Numeric](plan *graph.PJRTPlan[T]) GeneratorOption {
	return func(o *generatorOptions) {
		o.pjrtPlan = plan
	}
}

// pjrtGenerate runs autoregressive generation using a PJRTPlan.
// It calls RunPrefill for the prompt, then RunDecode per token.
// Sampling is done on the CPU-side logits returned by each step.
func (s *InferenceSession[T]) pjrtGenerate(ctx context.Context, plan *graph.PJRTPlan[T], promptIDs []int, sc SamplingConfig) (string, error) {
	plan.Reset()

	// Build the input slot map for prefill: the first input slot receives the
	// token ID tensor.
	if len(plan.InputSlots) == 0 {
		return "", fmt.Errorf("pjrt generate: plan has no input slots")
	}
	tokenSlot := plan.InputSlots[0]

	// Prefill: full prompt.
	prefillData := make([]T, len(promptIDs))
	for i, id := range promptIDs {
		prefillData[i] = T(id)
	}
	prefillTensor, err := tensor.New[T]([]int{1, len(promptIDs)}, prefillData)
	if err != nil {
		return "", fmt.Errorf("pjrt prefill tensor: %w", err)
	}

	logits, err := plan.RunPrefill(ctx, map[int]*tensor.TensorNumeric[T]{tokenSlot: prefillTensor})
	if err != nil {
		return "", fmt.Errorf("pjrt prefill: %w", err)
	}

	s.prepareStopSet(sc.StopTokenIDs)
	stopSet := s.stopSet

	s.prepareGeneratedIDs(sc.MaxNewTokens)
	generatedIDs := s.generatedIDs[:0]

	nextToken, err := s.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		return "", fmt.Errorf("pjrt sample after prefill: %w", err)
	}

	if stopSet[nextToken] {
		return "", nil
	}
	generatedIDs = append(generatedIDs, nextToken)

	// Decode loop: single token at a time.
	if plan.HasKVCache() && plan.DecodeExec != nil {
		decodeBuf := []T{T(nextToken)}
		for range sc.MaxNewTokens - 1 {
			if err := ctx.Err(); err != nil {
				break
			}

			decodeBuf[0] = T(nextToken)
			decodeTensor, err := tensor.New[T]([]int{1, 1}, decodeBuf)
			if err != nil {
				return "", fmt.Errorf("pjrt decode tensor: %w", err)
			}

			logits, err = plan.RunDecode(ctx, map[int]*tensor.TensorNumeric[T]{tokenSlot: decodeTensor})
			if err != nil {
				return "", fmt.Errorf("pjrt decode: %w", err)
			}

			nextToken, err = s.sampleFromLogits(logits, sc, generatedIDs)
			if err != nil {
				return "", fmt.Errorf("pjrt sample: %w", err)
			}

			if stopSet[nextToken] {
				break
			}
			generatedIDs = append(generatedIDs, nextToken)
		}
	}

	// Record token usage for billing middleware.
	if usage := TokenUsageFromContext(ctx); usage != nil {
		usage.SetPromptTokens(len(promptIDs))
		usage.SetCompletionTokens(len(generatedIDs))
	}

	if len(generatedIDs) == 0 {
		return "", nil
	}

	result, err := s.tokenizer.Decode(generatedIDs)
	if err != nil {
		return "", fmt.Errorf("pjrt decode output: %w", err)
	}
	return result, nil
}

// pjrtGenerateStream runs streaming generation using a PJRTPlan.
func (s *InferenceSession[T]) pjrtGenerateStream(ctx context.Context, plan *graph.PJRTPlan[T], promptIDs []int, sc SamplingConfig, stream TokenStream) error {
	plan.Reset()

	if len(plan.InputSlots) == 0 {
		return fmt.Errorf("pjrt stream: plan has no input slots")
	}
	tokenSlot := plan.InputSlots[0]

	// Prefill.
	prefillData := make([]T, len(promptIDs))
	for i, id := range promptIDs {
		prefillData[i] = T(id)
	}
	prefillTensor, err := tensor.New[T]([]int{1, len(promptIDs)}, prefillData)
	if err != nil {
		return fmt.Errorf("pjrt prefill tensor: %w", err)
	}

	logits, err := plan.RunPrefill(ctx, map[int]*tensor.TensorNumeric[T]{tokenSlot: prefillTensor})
	if err != nil {
		return fmt.Errorf("pjrt prefill: %w", err)
	}

	s.prepareStopSet(sc.StopTokenIDs)
	stopSet := s.stopSet

	s.prepareGeneratedIDs(sc.MaxNewTokens)
	generatedIDs := s.generatedIDs[:0]
	prevDecoded := ""

	nextToken, err := s.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		return fmt.Errorf("pjrt sample after prefill: %w", err)
	}

	if stopSet[nextToken] {
		return stream.OnToken("", true)
	}
	generatedIDs = append(generatedIDs, nextToken)

	if emitErr := s.emitToken(generatedIDs, &prevDecoded, sc.StopStrings, stream); emitErr != nil {
		if isStopStringErr(emitErr) {
			return nil
		}
		return emitErr
	}

	// Decode loop.
	if plan.HasKVCache() && plan.DecodeExec != nil {
		decodeBuf := []T{T(nextToken)}
		for range sc.MaxNewTokens - 1 {
			if err := ctx.Err(); err != nil {
				break
			}

			decodeBuf[0] = T(nextToken)
			decodeTensor, err := tensor.New[T]([]int{1, 1}, decodeBuf)
			if err != nil {
				return fmt.Errorf("pjrt decode tensor: %w", err)
			}

			logits, err = plan.RunDecode(ctx, map[int]*tensor.TensorNumeric[T]{tokenSlot: decodeTensor})
			if err != nil {
				return fmt.Errorf("pjrt decode: %w", err)
			}

			nextToken, err = s.sampleFromLogits(logits, sc, generatedIDs)
			if err != nil {
				return fmt.Errorf("pjrt sample: %w", err)
			}

			if stopSet[nextToken] {
				break
			}
			generatedIDs = append(generatedIDs, nextToken)

			if emitErr := s.emitToken(generatedIDs, &prevDecoded, sc.StopStrings, stream); emitErr != nil {
				if isStopStringErr(emitErr) {
					return nil
				}
				return emitErr
			}
		}
	}

	if usage := TokenUsageFromContext(ctx); usage != nil {
		usage.SetPromptTokens(len(promptIDs))
		usage.SetCompletionTokens(len(generatedIDs))
	}

	return stream.OnToken("", true)
}

// isStopStringErr checks if the error is the errStopString sentinel.
func isStopStringErr(err error) bool {
	return err == errStopString
}
