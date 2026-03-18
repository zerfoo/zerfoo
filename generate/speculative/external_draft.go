package speculative

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// ExternalDraft uses a smaller external model to generate draft tokens for
// speculative decoding. The draft and target models share a compute engine
// and block manager so GPU memory is not duplicated.
type ExternalDraft[T tensor.Numeric] struct {
	draftGraph *graph.Graph[T]
	engine     compute.Engine[T]
	blockPool  *generate.BlockPool[T]
	config     generate.ModelConfig
}

// NewExternalDraft creates an ExternalDraft that uses draftGraph as the draft
// model. The engine and blockPool are shared with the target model.
// blockPool may be nil if paged KV caching is not used.
func NewExternalDraft[T tensor.Numeric](
	draftGraph *graph.Graph[T],
	engine compute.Engine[T],
	blockPool *generate.BlockPool[T],
	config generate.ModelConfig,
) *ExternalDraft[T] {
	return &ExternalDraft[T]{
		draftGraph: draftGraph,
		engine:     engine,
		blockPool:  blockPool,
		config:     config,
	}
}

// Generate runs K greedy decoding steps on the draft model, starting from
// the given input tokens. It returns up to K draft token IDs and their
// corresponding log probabilities. Generation stops early if an EOS token
// is produced. The returned slices have equal length (<= K).
func (ed *ExternalDraft[T]) Generate(ctx context.Context, tokens []int32, K int) ([]int32, []float32, error) {
	if K <= 0 {
		return nil, nil, fmt.Errorf("K must be positive, got %d", K)
	}
	if len(tokens) == 0 {
		return nil, nil, fmt.Errorf("tokens must be non-empty")
	}

	// Create a KV cache for the draft model.
	cache := generate.NewKVCache[T](ed.config.NumLayers, ed.config.MaxSeqLen)
	draftCtx := generate.WithCache(ctx, generate.CacheProvider[T](cache))

	// Prefill: run all input tokens through the draft model.
	prefillTensor, err := intsToTensor[T](int32sToInts(tokens))
	if err != nil {
		return nil, nil, fmt.Errorf("create prefill tensor: %w", err)
	}

	logits, err := ed.draftGraph.Forward(draftCtx, prefillTensor)
	if err != nil {
		return nil, nil, fmt.Errorf("draft prefill: %w", err)
	}

	// Sample first draft token from last position of prefill logits.
	firstToken, firstLogProb, err := externalGreedyWithLogProb[T](logits)
	if err != nil {
		return nil, nil, fmt.Errorf("sample after prefill: %w", err)
	}

	draftTokens := make([]int32, 0, K)
	draftLogProbs := make([]float32, 0, K)

	draftTokens = append(draftTokens, firstToken)
	draftLogProbs = append(draftLogProbs, firstLogProb)

	if firstToken == int32(ed.config.EOSTokenID) {
		return draftTokens, draftLogProbs, nil
	}

	// Autoregressive draft loop for remaining K-1 tokens.
	nextToken := firstToken
	for range K - 1 {
		if err := ctx.Err(); err != nil {
			break
		}

		tokenTensor, tErr := intsToTensor[T]([]int{int(nextToken)})
		if tErr != nil {
			return nil, nil, fmt.Errorf("draft token tensor: %w", tErr)
		}

		logits, err = ed.draftGraph.Forward(draftCtx, tokenTensor)
		if err != nil {
			return nil, nil, fmt.Errorf("draft forward: %w", err)
		}

		tok, lp, sErr := externalGreedyWithLogProb[T](logits)
		if sErr != nil {
			return nil, nil, fmt.Errorf("sample draft token: %w", sErr)
		}

		draftTokens = append(draftTokens, tok)
		draftLogProbs = append(draftLogProbs, lp)

		if tok == int32(ed.config.EOSTokenID) {
			break
		}
		nextToken = tok
	}

	return draftTokens, draftLogProbs, nil
}

// externalGreedyWithLogProb extracts the last position from a
// [1, seqLen, vocabSize] logits tensor, finds the argmax token, and computes
// its log probability via log-softmax.
func externalGreedyWithLogProb[T tensor.Numeric](logits *tensor.TensorNumeric[T]) (int32, float32, error) {
	shape := logits.Shape()
	if len(shape) != 3 {
		return 0, 0, fmt.Errorf("expected 3D logits [batch, seq, vocab], got shape %v", shape)
	}

	vocabSize := shape[2]
	seqLen := shape[1]
	data := logits.Data()

	lastStart := (seqLen - 1) * vocabSize
	if lastStart+vocabSize > len(data) {
		return 0, 0, fmt.Errorf("logits data too short: %d < %d", len(data), lastStart+vocabSize)
	}

	slice := data[lastStart : lastStart+vocabSize]

	// Argmax.
	maxIdx := 0
	maxVal := float64(slice[0])
	for i := 1; i < vocabSize; i++ {
		v := float64(slice[i])
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}

	// Log-softmax with log-sum-exp trick for numerical stability.
	sumExp := 0.0
	for i := 0; i < vocabSize; i++ {
		sumExp += math.Exp(float64(slice[i]) - maxVal)
	}
	logProb := float32(-math.Log(sumExp))

	return int32(maxIdx), logProb, nil
}

// intsToTensor converts int token IDs to a [1, seqLen] input tensor.
func intsToTensor[T tensor.Numeric](ids []int) (*tensor.TensorNumeric[T], error) {
	data := make([]T, len(ids))
	for i, id := range ids {
		data[i] = T(id)
	}
	return tensor.New([]int{1, len(ids)}, data)
}


// int32sToInts converts a slice of int32 to a slice of int.
func int32sToInts(ids []int32) []int {
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}
