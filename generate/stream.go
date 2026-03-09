package generate

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// errStopString is a sentinel indicating that a stop string was matched
// and the done signal was already sent to the stream.
var errStopString = errors.New("stop string matched")

// TokenStream receives tokens as they are generated.
type TokenStream interface {
	// OnToken is called for each decoded token during streaming generation.
	// When done is true, generation is complete (token may be empty).
	// Returning a non-nil error stops generation.
	OnToken(token string, done bool) error
}

// TokenStreamFunc adapts a function to the TokenStream interface.
type TokenStreamFunc func(token string, done bool) error

// OnToken implements TokenStream.
func (f TokenStreamFunc) OnToken(token string, done bool) error {
	return f(token, done)
}

// GenerateStream produces text from a prompt, delivering each token to the
// stream as it is generated. The final output matches what Generate would return.
func (gen *Generator[T]) GenerateStream(ctx context.Context, prompt string, sc SamplingConfig, stream TokenStream) error {
	if sc.MaxNewTokens <= 0 {
		sc.MaxNewTokens = 256
	}

	promptIDs, err := gen.tokenizer.Encode(prompt)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return fmt.Errorf("prompt produced no tokens")
	}

	cache := NewKVCache[T](gen.config.NumLayers, gen.config.MaxSeqLen)
	genCtx := WithKVCache(ctx, cache)

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[gen.config.EOSTokenID] = true

	generatedIDs := make([]int, 0, sc.MaxNewTokens)
	prevDecoded := ""

	// Prefill.
	prefillTensor, err := gen.idsToTensor(promptIDs)
	if err != nil {
		return fmt.Errorf("create prefill tensor: %w", err)
	}

	logits, err := gen.graph.Forward(genCtx, prefillTensor)
	if err != nil {
		return fmt.Errorf("prefill forward: %w", err)
	}

	nextToken, err := gen.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		return fmt.Errorf("sample after prefill: %w", err)
	}

	if stopSet[nextToken] {
		return stream.OnToken("", true)
	}
	generatedIDs = append(generatedIDs, nextToken)

	// Emit incremental token.
	if emitErr := gen.emitToken(generatedIDs, &prevDecoded, sc.StopStrings, stream); emitErr != nil {
		if errors.Is(emitErr, errStopString) {
			return nil
		}
		return emitErr
	}

	// Autoregressive decode loop.
	for range sc.MaxNewTokens - 1 {
		if err := ctx.Err(); err != nil {
			break
		}

		tokenTensor, tErr := gen.idsToTensor([]int{nextToken})
		if tErr != nil {
			return fmt.Errorf("create token tensor: %w", tErr)
		}

		if p := gen.plan.Load(); p != nil {
			logits, err = p.Run(genCtx, tokenTensor)
		} else {
			logits, err = gen.graph.Forward(genCtx, tokenTensor)
			if err == nil {
				gen.compileGraph(genCtx, tokenTensor)
			}
		}
		if err != nil {
			return fmt.Errorf("decode forward: %w", err)
		}

		nextToken, err = gen.sampleFromLogits(logits, sc, generatedIDs)
		if err != nil {
			return fmt.Errorf("sample: %w", err)
		}

		if stopSet[nextToken] {
			break
		}
		generatedIDs = append(generatedIDs, nextToken)

		if emitErr := gen.emitToken(generatedIDs, &prevDecoded, sc.StopStrings, stream); emitErr != nil {
			if errors.Is(emitErr, errStopString) {
				return nil
			}
			return emitErr
		}
	}

	return stream.OnToken("", true)
}

// emitToken decodes the full generated sequence, computes the incremental
// difference from the previous decoding, and emits it to the stream.
// It also checks for stop strings and returns a sentinel error if found.
func (gen *Generator[T]) emitToken(
	generatedIDs []int,
	prevDecoded *string,
	stopStrings []string,
	stream TokenStream,
) error {
	decoded, err := gen.tokenizer.Decode(generatedIDs)
	if err != nil {
		return fmt.Errorf("decode token: %w", err)
	}

	// Check stop strings.
	for _, ss := range stopStrings {
		if idx := strings.Index(decoded, ss); idx >= 0 {
			// Emit any text before the stop string that hasn't been emitted yet.
			remaining := decoded[:idx]
			if len(remaining) > len(*prevDecoded) {
				delta := remaining[len(*prevDecoded):]
				if err := stream.OnToken(delta, false); err != nil {
					return err
				}
			}
			if err := stream.OnToken("", true); err != nil {
				return err
			}
			return errStopString
		}
	}

	// Emit incremental text.
	if len(decoded) > len(*prevDecoded) {
		delta := decoded[len(*prevDecoded):]
		*prevDecoded = decoded
		return stream.OnToken(delta, false)
	}
	*prevDecoded = decoded
	return nil
}

// Statically verify TokenStreamFunc implements TokenStream.
var _ TokenStream = TokenStreamFunc(nil)
