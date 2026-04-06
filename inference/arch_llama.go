package inference

import (
	"context"
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// tanhf32 computes tanh(x) using a rational approximation entirely in float32.
// Accurate to ~3e-5 for |x| < 4.5 (sufficient for logit softcapping).
// For |x| >= 4.5, returns +-1.
func tanhf32(x float32) float32 {
	if x > 4.5 {
		return 1
	}
	if x < -4.5 {
		return -1
	}
	x2 := x * x
	return x * (27 + x2) / (27 + 9*x2)
}

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
	tl := newTensorLookup(tensors)

	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	// Llama can tie lm_head to embedding weights.
	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, transformerGraphOpts{
		residual: ResidualConfigFromGGUF(cfg.ResidualMode, cfg.AttnResNumBlocks),
	})
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
	softcapVal float32 // if > 0, apply softcapping: cap * tanh(logit/cap)
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

	// Use MatMulTransposeB to compute flat * weight^T directly, avoiding an
	// explicit Transpose allocation. Caching a transposed tensor caused a
	// use-after-free: the graph's ref-counting released the tensor while the
	// cache still held a reference, resulting in a null device pointer and
	// cuBLAS status 7 (INTERNAL_ERROR).
	//
	// Q4 storage uses virtual transpose (shape swap only) so Q4 GEMV reads
	// blocks in native order -- use regular MatMul for quantized weights.
	var out *tensor.TensorNumeric[T]
	if _, isQ4 := any(h.weight.GetStorage()).(*tensor.Q4Storage); isQ4 {
		ws := h.weight.Shape()
		wT, tErr := tensor.NewWithStorage[T]([]int{ws[1], ws[0]}, h.weight.GetStorage())
		if tErr != nil {
			return nil, tErr
		}
		out, err = h.engine.MatMul(ctx, flat, wT)
	} else if _, isQ4K := any(h.weight.GetStorage()).(*tensor.Q4KStorage); isQ4K {
		ws := h.weight.Shape()
		wT, tErr := tensor.NewWithStorage[T]([]int{ws[1], ws[0]}, h.weight.GetStorage())
		if tErr != nil {
			return nil, tErr
		}
		out, err = h.engine.MatMul(ctx, flat, wT)
	} else if _, isMmap := any(h.weight.GetStorage()).(*tensor.MmapStorage); isMmap {
		// MmapStorage: use MatMulTransposeB when available, falling back
		// to explicit Transpose. Virtual transpose (shape swap) would work
		// but the per-op matMulMmap dispatch handles B-weight correctly.
		if tb, ok := h.engine.(compute.TransposeBMatMuler[T]); ok {
			out, err = tb.MatMulTransposeB(ctx, flat, h.weight)
		} else {
			wT, tErr := h.engine.Transpose(ctx, h.weight, []int{1, 0})
			if tErr != nil {
				return nil, tErr
			}
			out, err = h.engine.MatMul(ctx, flat, wT)
		}
	} else if tb, ok := h.engine.(compute.TransposeBMatMuler[T]); ok {
		out, err = tb.MatMulTransposeB(ctx, flat, h.weight)
	} else {
		wT, tErr := h.engine.Transpose(ctx, h.weight, []int{1, 0})
		if tErr != nil {
			return nil, tErr
		}
		out, err = h.engine.MatMul(ctx, flat, wT)
	}
	if err != nil {
		return nil, err
	}

	vocabSize := h.weight.Shape()[0]
	result, err := h.engine.Reshape(ctx, out, []int{batch, seqLen, vocabSize})
	if err != nil {
		return nil, err
	}

	// Convert FP16 logits to F32 for sampling. When the forward pass runs in
	// FP16, the MatMul output uses Float16Storage. Sampling expects F32, so
	// convert here at the boundary before returning logits.
	if _, isFP16 := any(result.GetStorage()).(*tensor.Float16Storage); isFP16 {
		if conv, ok := any(h.engine).(compute.FP16ToF32Converter); ok {
			if f32t, ok := any(result).(*tensor.TensorNumeric[float32]); ok {
				if converted, cErr := conv.ConvertFP16ToF32(f32t); cErr == nil {
					result = any(converted).(*tensor.TensorNumeric[T])
				}
			}
		}
	}

	// Apply logit softcapping: cap * tanh(logit / cap).
	if h.softcapVal > 0 {
		// GPU path: use engine operations to keep data on device.
		if _, ok := result.GetStorage().(*tensor.GPUStorage[T]); ok {
			invCap := T(1.0 / float64(h.softcapVal))
			scaled, scErr := h.engine.MulScalar(ctx, result, invCap)
			if scErr != nil {
				return nil, scErr
			}
			tanhed, tErr := h.engine.Tanh(ctx, scaled)
			if tErr != nil {
				return nil, tErr
			}
			result, err = h.engine.MulScalar(ctx, tanhed, T(h.softcapVal))
			if err != nil {
				return nil, err
			}
		} else {
			// CPU path: fast float32 tanh approximation parallelized across cores.
			data := result.Data()
			n := len(data)
			invCap := float32(1.0 / float64(h.softcapVal))
			cap32 := h.softcapVal
			const chunkSize = 4096
			nChunks := (n + chunkSize - 1) / chunkSize
			if nChunks <= 1 {
				for i := range data {
					data[i] = T(cap32 * tanhf32(float32(data[i])*invCap))
				}
			} else {
				var wg sync.WaitGroup
				wg.Add(nChunks)
				for c := range nChunks {
					start := c * chunkSize
					end := start + chunkSize
					if end > n {
						end = n
					}
					go func() {
						defer wg.Done()
						for i := start; i < end; i++ {
							data[i] = T(cap32 * tanhf32(float32(data[i])*invCap))
						}
					}()
				}
				wg.Wait()
			}
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

func (e *embeddingLookupNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	ids := input.Data()
	hiddenDim := e.weight.Shape()[1]

	seqLen := 1
	for _, d := range shape {
		seqLen *= d
	}

	batch := shape[0]
	sl := seqLen / batch

	vocabSize := e.weight.Shape()[0]

	// GPU path: use engine.Gather when weight has GPU storage or Q8 with GPUPtr.
	isGPUStorage := false
	if _, ok := e.weight.GetStorage().(*tensor.GPUStorage[T]); ok {
		isGPUStorage = true
	}
	if qs, ok := any(e.weight.GetStorage()).(*tensor.Q8Storage); ok {
		if ptr, _, _ := qs.GPUPtr(); ptr != nil {
			isGPUStorage = true
		}
	}
	if isGPUStorage {
		intIDs := make([]int, seqLen)
		for i := range intIDs {
			id := int(ids[i])
			if id < 0 || id >= vocabSize {
				return nil, fmt.Errorf("token ID %d out of range [0, %d)", id, vocabSize)
			}
			intIDs[i] = id
		}
		idxTensor, err := tensor.New([]int{seqLen}, intIDs)
		if err != nil {
			return nil, fmt.Errorf("embedding lookup: create index tensor: %w", err)
		}
		outTensor, err := tensor.New[T]([]int{seqLen, hiddenDim}, nil)
		if err != nil {
			return nil, fmt.Errorf("embedding lookup: create output tensor: %w", err)
		}
		if err := e.engine.Gather(ctx, e.weight, idxTensor, outTensor); err != nil {
			return nil, fmt.Errorf("embedding lookup: gather: %w", err)
		}
		if e.scale > 0 {
			outTensor, err = e.engine.MulScalar(ctx, outTensor, T(e.scale))
			if err != nil {
				return nil, fmt.Errorf("embedding lookup: scale: %w", err)
			}
		}
		return e.engine.Reshape(ctx, outTensor, []int{batch, sl, hiddenDim})
	}

	// CPU path: direct row lookup.
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
			if id < 0 || id >= vocabSize {
				return nil, fmt.Errorf("token ID %d out of range [0, %d)", id, vocabSize)
			}
			q8.DequantizeRange(row, id*hiddenDim, hiddenDim)
			for j := range hiddenDim {
				out[i*hiddenDim+j] = T(row[j])
			}
		}
	} else {
		embData := e.weight.Data()
		for i := range seqLen {
			id := int(ids[i])
			if id < 0 || id >= vocabSize {
				return nil, fmt.Errorf("token ID %d out of range [0, %d)", id, vocabSize)
			}
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

	return tensor.New([]int{batch, sl, hiddenDim}, out)
}

func (e *embeddingLookupNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Static interface assertions.
var _ graph.EmbeddedFrozenProvider[float32] = (*lmHeadNode[float32])(nil)
var _ graph.EmbeddedFrozenProvider[float32] = (*embeddingLookupNode[float32])(nil)
