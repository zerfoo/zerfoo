package attention

import (
	"context"
	"fmt"
	"math"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// negInfValue returns a large negative value (-1e9) for floating point types.
func negInfValue[T tensor.Numeric]() T {
	var zero T
	if p, ok := any(&zero).(*float32); ok {
		*p = -1e9
		return zero
	}
	if p, ok := any(&zero).(*float64); ok {
		*p = -1e9
		return zero
	}
	return zero
}

// ScaledDotProductAttention implements the scaled dot-product attention mechanism.
type ScaledDotProductAttention[T tensor.Numeric] struct {
	engine        compute.Engine[T]
	headDim       float64 // Dimension of each head, used for scaling
	numQueryHeads int     // Number of query heads (for flash decode GQA dispatch)
	numKVHeads    int     // Number of KV heads (for flash decode GQA dispatch)
	causal        bool    // if true, apply causal masking to attention scores

	// Cached tensors for backward pass. SDPA is not itself a graph.Node
	// (its Forward takes q/k/v/mask and its Backward is called by the
	// owning attention node with nil inputs), so q/k/v and the attention
	// weights are SAVE-class intermediates: they are registered with the
	// save-for-backward contract (ztensor ADR 006) via the Saver the
	// owning node fans in, pinning arena-backed storage until the owning
	// node's Backward has consumed them.
	q                *tensor.TensorNumeric[T]
	k                *tensor.TensorNumeric[T]
	v                *tensor.TensorNumeric[T]
	attentionWeights *tensor.TensorNumeric[T]
	saver            graph.Saver[T] // fanned in by the owning attention node; nil outside a Graph

	// Persistent flash-kernel scratch (zerfoo#870, docs/lore.md L-0006).
	// tryFlashForward and tryFlashDecode allocate GPU buffers directly via
	// tensor.NewGPUStorage (bypassing the engine arena, since they operate on
	// bare device pointers below the engine abstraction). A per-call
	// malloc+free of those buffers is safe in eager execution but corrupts
	// training under training.CaptureReplayRunner: a CUDA graph bakes in the
	// literal device address a kernel launch reads/writes, so freeing that
	// address (via defer or GC finalizer) lets an unrelated later allocation
	// reuse it while the graph still replays against it -- observed as an
	// illegal memory access around replay #141/511. Caching the buffers here,
	// keyed to this SDPA node's lifetime (which spans the whole capture-replay
	// life of the graph it belongs to), means they are allocated once and
	// reused for every call instead of being freed per call.
	flashFwdOut        gpuScratchBuffer[T]
	flashDecOut        gpuScratchBuffer[T]
	flashDecPartialO   gpuScratchBuffer[T]
	flashDecPartialLSE gpuScratchBuffer[float32]
}

// SetSaver implements graph.SaverAware. Owning nodes (AttentionHead,
// GroupedQueryAttention, ...) fan their Saver into SDPA so its cached
// tensors are attributed to the owning node and released after its
// Backward returns.
func (sdpa *ScaledDotProductAttention[T]) SetSaver(sv graph.Saver[T]) {
	sdpa.saver = sv
}

// SetCausal enables or disables causal (lower-triangular) masking.
func (sdpa *ScaledDotProductAttention[T]) SetCausal(causal bool) {
	sdpa.causal = causal
}

// ScaledDotProductAttentionOptions holds configuration options for ScaledDotProductAttention.
type ScaledDotProductAttentionOptions[T tensor.Numeric] struct {
	bidirectional bool // when true, causal masking is disabled
	numQueryHeads int  // query head count for flash decode dispatch
	numKVHeads    int  // KV head count for flash decode dispatch
}

// ScaledDotProductAttentionOption applies an option to ScaledDotProductAttentionOptions.
type ScaledDotProductAttentionOption[T tensor.Numeric] func(*ScaledDotProductAttentionOptions[T])

// WithBidirectional returns an option that disables causal masking, allowing
// every position to attend to every other position. This is required for
// encoder-style models such as BERT.
func WithBidirectional[T tensor.Numeric]() ScaledDotProductAttentionOption[T] {
	return func(o *ScaledDotProductAttentionOptions[T]) {
		o.bidirectional = true
	}
}

// WithHeadCounts sets the query and KV head counts, enabling the split-KV
// flash decode kernel for autoregressive decode with GQA support.
func WithHeadCounts[T tensor.Numeric](numQueryHeads, numKVHeads int) ScaledDotProductAttentionOption[T] {
	return func(o *ScaledDotProductAttentionOptions[T]) {
		o.numQueryHeads = numQueryHeads
		o.numKVHeads = numKVHeads
	}
}

// NewScaledDotProductAttention creates a new ScaledDotProductAttention layer.
func NewScaledDotProductAttention[T tensor.Numeric](engine compute.Engine[T], headDim int, opts ...ScaledDotProductAttentionOption[T]) *ScaledDotProductAttention[T] {
	options := &ScaledDotProductAttentionOptions[T]{}
	for _, opt := range opts {
		opt(options)
	}

	return &ScaledDotProductAttention[T]{
		engine:        engine,
		headDim:       float64(headDim),
		numQueryHeads: options.numQueryHeads,
		numKVHeads:    options.numKVHeads,
	}
}

// NewBidirectionalSDPA creates a ScaledDotProductAttention layer with causal
// masking disabled. All positions attend to all other positions, which is the
// attention pattern used by encoder models such as BERT.
func NewBidirectionalSDPA[T tensor.Numeric](engine compute.Engine[T], headDim int, opts ...ScaledDotProductAttentionOption[T]) *ScaledDotProductAttention[T] {
	return NewScaledDotProductAttention(engine, headDim, append(opts, WithBidirectional[T]())...)
}

// Forward computes the scaled dot-product attention.
// Q, K, V are expected to be 3D tensors (batch_size, seq_len, head_dim).
// mask is an optional 4D tensor (batch_size, num_heads, seq_len_q, seq_len_k).
func (sdpa *ScaledDotProductAttention[T]) Forward(ctx context.Context, q, k, v, mask *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Cache inputs for backward pass. q/k/v are upstream intermediates of
	// the owning node (projections), so they must be pinned too.
	sdpa.q = q
	sdpa.k = k
	sdpa.v = v
	// Invalidate the cached attention weights from any PREVIOUS step. The
	// fused paths below (flash decode / flash forward) return early without
	// setting attentionWeights; Backward treats a non-nil cache as
	// authoritative and would otherwise consume the previous step's tensor --
	// arena-backed, unpinned, and typically reclaimed by a per-step pool
	// Reset -- yielding deterministically wrong gradients from the second
	// step of a training loop onward (Backward itself re-populates this
	// cache when it recomputes after a fused forward, which is what arms the
	// staleness). Always clearing here makes Backward recompute from the
	// freshly pinned q/k of THIS forward.
	sdpa.attentionWeights = nil
	if sdpa.saver != nil {
		sdpa.saver.SaveForBackward(q, k, v)
	}

	// Resolve the engine's GPU stream (compute.StreamProvider, proxy-unwrapped)
	// once, so both the flash-decode and flash-forward kernels launch
	// stream-ordered after the engine ops that produced Q/K/V. Launching on a
	// private stream races with in-flight producers and was observed to silently
	// corrupt training (Wolf CrossAsset GB10; zerfoo#865/#866).
	var engStream unsafe.Pointer
	realEng := compute.Engine[T](sdpa.engine)
	if proxy, ok := sdpa.engine.(*compute.EngineProxy[T]); ok {
		realEng = proxy.Real()
	}
	if sp, ok := realEng.(compute.StreamProvider); ok {
		engStream = sp.Stream()
	}

	// Try split-KV flash decode for single-query autoregressive decode.
	// This path handles the seqLen_Q==1 case that tryFlashForward rejects.
	// Scratch buffers are cached on sdpa (zerfoo#870) so they are
	// replay-stable under CUDA-graph capture instead of being freed per call.
	if mask == nil && sdpa.numQueryHeads > 0 && sdpa.numKVHeads > 0 {
		if result, err := tryFlashDecode(
			q, k, v, int(sdpa.headDim), sdpa.numQueryHeads, sdpa.numKVHeads, engStream,
			&sdpa.flashDecOut, &sdpa.flashDecPartialO, &sdpa.flashDecPartialLSE,
		); result != nil || err != nil {
			return result, err
		}
	}

	// Try fused flash attention when no arbitrary mask is provided.
	// Flash attention handles causal masking internally via the causal flag.
	if mask == nil {
		if result, err := tryFlashForward(q, k, v, int(sdpa.headDim), sdpa.causal, engStream, &sdpa.flashFwdOut); result != nil || err != nil {
			return result, err
		}
	}

	// 1. MatMul Q and K^T
	// (batch, seq_len_q, head_dim) x (batch, head_dim, seq_len_k) -> (batch, seq_len_q, seq_len_k)
	// Use MatMulTransposeB when available to avoid explicit Transpose allocation + kernel.
	var (
		attentionScores *tensor.TensorNumeric[T]
		err             error
	)
	if tb, ok := sdpa.engine.(compute.TransposeBMatMuler[T]); ok {
		attentionScores, err = tb.MatMulTransposeB(ctx, q, k)
	} else {
		var kTransposed *tensor.TensorNumeric[T]
		kTransposed, err = sdpa.engine.Transpose(ctx, k, []int{0, 2, 1})
		if err != nil {
			return nil, err
		}
		attentionScores, err = sdpa.engine.MatMul(ctx, q, kTransposed, nil)
	}
	if err != nil {
		return nil, err
	}

	// 2. Scale attention scores and apply softmax
	// Compute head dimension robustly to avoid division by zero
	d := sdpa.headDim
	if d <= 0 {
		// Fallback to deriving from Q's last dimension
		if q == nil || len(q.Shape()) < 3 {
			return nil, fmt.Errorf("ScaledDotProductAttention: invalid Q shape %v to infer head dimension", q.Shape())
		}
		d = float64(q.Shape()[2])
	}
	if d <= 0 {
		return nil, fmt.Errorf("ScaledDotProductAttention: headDim must be > 0, got %v", d)
	}
	// Determine whether masking will intervene between scaling and softmax.
	needsMasking := mask != nil || (sdpa.causal && len(attentionScores.Shape()) >= 2 && attentionScores.Shape()[len(attentionScores.Shape())-2] > 1)
	scale := float32(1.0 / math.Sqrt(d))

	// Fused softmax+V multiply for decode (seqQ=1).
	// Combines scale, softmax, and V matmul in a single kernel launch.
	if q.Shape()[1] == 1 && !needsMasking {
		realEng := compute.Engine[T](sdpa.engine)
		if proxy, ok := sdpa.engine.(*compute.EngineProxy[T]); ok {
			realEng = proxy.Real()
		}
		if fuser, ok := realEng.(compute.FusedSoftmaxVMulProvider[T]); ok {
			fusedOut, fusedErr := fuser.GPUFusedSoftmaxVMul(attentionScores, v, scale)
			if fusedErr == nil {
				return fusedOut, nil
			}
		}
	}

	var attentionWeights *tensor.TensorNumeric[T]
	if !needsMasking {
		// Fused scaled softmax: single kernel replaces MulScalar + Softmax.
		realEngine := compute.Engine[T](sdpa.engine)
		if proxy, ok := sdpa.engine.(*compute.EngineProxy[T]); ok {
			realEngine = proxy.Real()
		}
		if provider, ok := realEngine.(compute.FusedScaledSoftmaxProvider[T]); ok {
			out, fusedErr := provider.GPUScaledSoftmax(attentionScores, scale, -1)
			if fusedErr == nil {
				attentionWeights = out
			}
		}
	}

	if attentionWeights == nil {
		// Fallback: separate MulScalar + optional masking + Softmax.
		scaleFactor := sdpa.engine.Ops().FromFloat64(1.0 / math.Sqrt(d))

		scaledAttentionScores, err := sdpa.engine.MulScalar(ctx, attentionScores, scaleFactor, nil)
		if err != nil {
			return nil, err
		}

		// 3. Apply mask (explicit 4D mask or causal)
		if mask != nil {
			batchSize := q.Shape()[0]
			numHeads := mask.Shape()[1]
			seqLenQ := q.Shape()[1]
			seqLenK := k.Shape()[1]

			reshapedScores, err := sdpa.engine.Reshape(ctx, scaledAttentionScores, []int{batchSize / numHeads, numHeads, seqLenQ, seqLenK})
			if err != nil {
				return nil, err
			}

			maskedScores, err := sdpa.engine.Add(ctx, reshapedScores, mask, nil)
			if err != nil {
				return nil, err
			}

			scaledAttentionScores, err = sdpa.engine.Reshape(ctx, maskedScores, []int{batchSize, seqLenQ, seqLenK})
			if err != nil {
				return nil, err
			}
		} else if sdpa.causal {
			// Apply causal masking directly to 3D scores (batch, seqQ, seqK).
			// Set positions where q_pos < k_pos to -inf.
			shape := scaledAttentionScores.Shape()
			seqQ := shape[1]
			seqK := shape[2]
			// During decode (seqQ == 1), every cached position is visible
			// (offset = seqK - 1, so ki <= seqK-1 == qi+offset for all ki).
			// Skip masking entirely to avoid a costly .Data() D2H copy on GPU tensors.
			if seqQ > 1 {
				// Build a causal mask tensor [1, seqQ, seqK] with 0 for visible
				// positions and -inf for future positions, then add it to scores.
				// This avoids .Data() which causes a D2H copy on GPU tensors and
				// leaves the GPU-side data unmasked (the root cause of the GPU
				// inference regression where the model ignored causal ordering).
				batch := shape[0]
				offset := seqK - seqQ
				negInf := negInfValue[T]()
				maskData := make([]T, seqQ*seqK)
				for qi := range seqQ {
					for ki := range seqK {
						if ki > qi+offset {
							maskData[qi*seqK+ki] = negInf
						}
					}
				}
				causalMask, maskErr := tensor.New[T]([]int{1, seqQ, seqK}, maskData)
				if maskErr != nil {
					return nil, fmt.Errorf("causal mask: %w", maskErr)
				}
				// Broadcast [1, seqQ, seqK] across [batch, seqQ, seqK].
				_ = batch // broadcast handled by engine.Add
				scaledAttentionScores, err = sdpa.engine.Add(ctx, scaledAttentionScores, causalMask)
				if err != nil {
					return nil, fmt.Errorf("causal mask add: %w", err)
				}
			}
		}

		// 4. Apply Softmax
		attentionWeights, err = sdpa.engine.Softmax(ctx, scaledAttentionScores, -1, nil) // Softmax along the last dimension
		if err != nil {
			return nil, err
		}
	}

	sdpa.attentionWeights = attentionWeights // Cache for backward pass
	if sdpa.saver != nil {
		sdpa.saver.SaveForBackward(attentionWeights)
	}

	// 5. MatMul attention weights and V
	// (batch, seq_len_q, seq_len_k) x (batch, seq_len_k, head_dim) -> (batch, seq_len_q, head_dim)
	output, err := sdpa.engine.MatMul(ctx, attentionWeights, v, nil)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for ScaledDotProductAttention.
// dOut is the gradient from the subsequent layer.
func (sdpa *ScaledDotProductAttention[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut, _, _, _ *tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// When the forward pass used the fused flash attention kernel, it computed
	// the output directly without materializing intermediate attention weights.
	// Recompute them here from cached Q, K so the backward math is correct.
	if sdpa.attentionWeights == nil {
		var scores *tensor.TensorNumeric[T]
		var recomputeErr error
		if tb, ok := sdpa.engine.(compute.TransposeBMatMuler[T]); ok {
			scores, recomputeErr = tb.MatMulTransposeB(ctx, sdpa.q, sdpa.k)
		} else {
			var kt *tensor.TensorNumeric[T]
			kt, recomputeErr = sdpa.engine.Transpose(ctx, sdpa.k, []int{0, 2, 1})
			if recomputeErr == nil {
				scores, recomputeErr = sdpa.engine.MatMul(ctx, sdpa.q, kt, nil)
			}
		}
		if recomputeErr != nil {
			return nil, fmt.Errorf("SDPA backward: recompute attention scores: %w", recomputeErr)
		}
		d := sdpa.headDim
		if d <= 0 && sdpa.q != nil && len(sdpa.q.Shape()) >= 3 {
			d = float64(sdpa.q.Shape()[2])
		}
		scaleFactor := sdpa.engine.Ops().FromFloat64(1.0 / math.Sqrt(d))
		scaled, scaleErr := sdpa.engine.MulScalar(ctx, scores, scaleFactor, nil)
		if scaleErr != nil {
			return nil, fmt.Errorf("SDPA backward: scale scores: %w", scaleErr)
		}
		sdpa.attentionWeights, recomputeErr = sdpa.engine.Softmax(ctx, scaled, -1, nil)
		if recomputeErr != nil {
			return nil, fmt.Errorf("SDPA backward: softmax recompute: %w", recomputeErr)
		}
	}

	// 1. Gradient w.r.t. V
	attentionWeightsTransposed, err := sdpa.engine.Transpose(ctx, sdpa.attentionWeights, []int{0, 2, 1})
	if err != nil {
		return nil, err
	}

	dV, err := sdpa.engine.MatMul(ctx, attentionWeightsTransposed, dOut, nil)
	if err != nil {
		return nil, err
	}

	// 2. Gradient w.r.t. attention_weights
	vTransposed, err := sdpa.engine.Transpose(ctx, sdpa.v, []int{0, 2, 1})
	if err != nil {
		return nil, err
	}

	dAttentionWeights, err := sdpa.engine.MatMul(ctx, dOut, vTransposed, nil)
	if err != nil {
		return nil, err
	}

	// 3. Gradient w.r.t. scaled_attention_scores (through softmax)
	// dL/dx = (dL/dy - sum(dL/dy * y)) * y
	mul, err := sdpa.engine.Mul(ctx, dAttentionWeights, sdpa.attentionWeights)
	if err != nil {
		return nil, err
	}

	// Use explicit last axis index instead of -1 because ReduceSum treats
	// negative axes as "sum over all axes" rather than counting from the end.
	lastAxis := len(mul.Shape()) - 1
	sum, err := sdpa.engine.ReduceSum(ctx, mul, lastAxis, true)
	if err != nil {
		return nil, err
	}

	sub, err := sdpa.engine.Sub(ctx, dAttentionWeights, sum)
	if err != nil {
		return nil, err
	}

	dScaledAttentionScores, err := sdpa.engine.Mul(ctx, sub, sdpa.attentionWeights)
	if err != nil {
		return nil, err
	}

	// 4. Gradient w.r.t. attention_scores (through scaling)
	// Use the same robust head dimension computation as in Forward
	d := sdpa.headDim
	if d <= 0 {
		if sdpa.q == nil || len(sdpa.q.Shape()) < 3 {
			return nil, fmt.Errorf("ScaledDotProductAttention: cannot infer headDim in Backward; cached Q is invalid")
		}
		d = float64(sdpa.q.Shape()[2])
	}
	scaleFactor := sdpa.engine.Ops().FromFloat64(1.0 / math.Sqrt(d))

	dAttentionScores, err := sdpa.engine.MulScalar(ctx, dScaledAttentionScores, scaleFactor, nil)
	if err != nil {
		return nil, err
	}

	// 5. Gradient w.r.t. Q and K
	dQ, err := sdpa.engine.MatMul(ctx, dAttentionScores, sdpa.k, nil)
	if err != nil {
		return nil, err
	}

	dAttentionScoresTransposed, err := sdpa.engine.Transpose(ctx, dAttentionScores, []int{0, 2, 1})
	if err != nil {
		return nil, err
	}

	dK, err := sdpa.engine.MatMul(ctx, dAttentionScoresTransposed, sdpa.q, nil)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dQ, dK, dV}, nil
}
