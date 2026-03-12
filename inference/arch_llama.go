package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// buildLlamaGraph constructs a computation graph for the Llama architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// The Llama architecture is:
//
//	Embed -> [RMSNorm -> GQA -> Add -> RMSNorm -> FFN(SiLU-gate) -> Add] x N -> RMSNorm -> LMHead
func buildLlamaGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	embedWeight, ok := tensors["model.embed_tokens.weight"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %q", "model.embed_tokens.weight")
	}

	// Llama can tie lm_head to embedding weights.
	lmHeadWeight, ok := tensors["lm_head.weight"]
	if !ok {
		lmHeadWeight = embedWeight
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, transformerGraphOpts{})
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}

// lmHeadNode projects hidden states to vocabulary logits.
// weight shape: [vocabSize, hiddenDim].
// input shape: [batch, seqLen, hiddenDim].
// output shape: [batch, seqLen, vocabSize].
type lmHeadNode[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	weight     *tensor.TensorNumeric[T]
	weightT    *tensor.TensorNumeric[T] // pre-transposed weight [hiddenDim, vocabSize]
	softcapVal float32                  // if > 0, apply softcapping: cap * tanh(logit/cap)
}

func (h *lmHeadNode[T]) OpType() string                  { return "LMHead" }
func (h *lmHeadNode[T]) Attributes() map[string]any       { return nil }
func (h *lmHeadNode[T]) OutputShape() []int               { return nil }
func (h *lmHeadNode[T]) Parameters() []*graph.Parameter[T] { return nil }

// EmbeddedFrozen returns the LM head weight so the compiler registers it as
// a frozen slot during graph compilation.
func (h *lmHeadNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	if h.weight == nil {
		return nil
	}
	return []*tensor.TensorNumeric[T]{h.weight}
}

func (h *lmHeadNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	batch, seqLen, hiddenDim := shape[0], shape[1], shape[2]

	flat, err := h.engine.Reshape(ctx, input, []int{batch * seqLen, hiddenDim})
	if err != nil {
		return nil, err
	}

	// Pre-transpose weight once and cache for subsequent calls.
	if h.weightT == nil {
		wT, tErr := h.engine.Transpose(ctx, h.weight, []int{1, 0})
		if tErr != nil {
			return nil, tErr
		}
		h.weightT = wT
	}

	out, err := h.engine.MatMul(ctx, flat, h.weightT)
	if err != nil {
		return nil, err
	}

	vocabSize := h.weight.Shape()[0]
	result, err := h.engine.Reshape(ctx, out, []int{batch, seqLen, vocabSize})
	if err != nil {
		return nil, err
	}

	// Apply logit softcapping: cap * tanh(logit / cap).
	if h.softcapVal > 0 {
		data := result.Data()
		cap := float64(h.softcapVal)
		for i := range data {
			data[i] = T(cap * math.Tanh(float64(data[i])/cap))
		}
	}

	return result, nil
}

func (h *lmHeadNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// embeddingLookupNode converts token IDs [batch, seqLen] to embeddings
// [batch, seqLen, hiddenDim] by looking up rows in the weight table.
// Optionally scales embeddings by a constant factor.
type embeddingLookupNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	weight *tensor.TensorNumeric[T] // [vocabSize, hiddenDim]
	scale  float32                  // 0 means no scaling
}

func (e *embeddingLookupNode[T]) OpType() string                  { return "EmbeddingLookup" }
func (e *embeddingLookupNode[T]) Attributes() map[string]any       { return nil }
func (e *embeddingLookupNode[T]) OutputShape() []int               { return nil }
func (e *embeddingLookupNode[T]) Parameters() []*graph.Parameter[T] { return nil }

// EmbeddedFrozen returns the embedding weight so the compiler registers it as
// a frozen slot during graph compilation.
func (e *embeddingLookupNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	if e.weight == nil {
		return nil
	}
	return []*tensor.TensorNumeric[T]{e.weight}
}

func (e *embeddingLookupNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	ids := input.Data()
	hiddenDim := e.weight.Shape()[1]

	seqLen := 1
	for _, d := range shape {
		seqLen *= d
	}

	out := make([]T, seqLen*hiddenDim)

	// Fast path: dequantize only the needed rows from Q8 storage instead
	// of materializing the entire embedding table.
	type rangeDeq interface {
		DequantizeRange(dst []float32, start, count int)
	}
	if q8, ok := any(e.weight.GetStorage()).(rangeDeq); ok {
		row := make([]float32, hiddenDim)
		for i := range seqLen {
			id := int(ids[i])
			q8.DequantizeRange(row, id*hiddenDim, hiddenDim)
			for j := range hiddenDim {
				out[i*hiddenDim+j] = T(row[j])
			}
		}
	} else {
		embData := e.weight.Data()
		for i := range seqLen {
			id := int(ids[i])
			for j := range hiddenDim {
				out[i*hiddenDim+j] = embData[id*hiddenDim+j]
			}
		}
	}

	if e.scale > 0 {
		s := T(e.scale)
		for i := range out {
			out[i] *= s
		}
	}

	batch := shape[0]
	sl := seqLen / batch
	return tensor.New([]int{batch, sl, hiddenDim}, out)
}

func (e *embeddingLookupNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Static interface assertions.
var _ graph.EmbeddedFrozenProvider[float32] = (*lmHeadNode[float32])(nil)
var _ graph.EmbeddedFrozenProvider[float32] = (*embeddingLookupNode[float32])(nil)
