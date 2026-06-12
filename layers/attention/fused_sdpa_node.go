// Package attention exposes a graph.Node wrapper around ScaledDotProductAttention
// so consumers (e.g., Wolf cross-attention) can compose fused SDPA via
// graph.Builder without duplicating the underlying math.
package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// FusedSDPA wraps ScaledDotProductAttention as a graph.Node[T]. It accepts
// (Q, K, V) or (Q, K, V, mask) inputs in Forward and delegates to the inner
// SDPA. Backward delegates to the inner SDPA and pads a nil gradient slot for
// the mask input when one was supplied so input/grad indexing stays aligned.
type FusedSDPA[T tensor.Numeric] struct {
	inner *ScaledDotProductAttention[T]

	// hadMask records whether the most recent Forward received a mask input,
	// so Backward can return a matching gradient slot count.
	hadMask bool

	// outputShape is cached from the most recent Forward.
	outputShape []int
}

// FusedSDPAOption configures a FusedSDPA at construction time.
type FusedSDPAOption[T tensor.Numeric] func(*fusedSDPAOptions[T])

type fusedSDPAOptions[T tensor.Numeric] struct {
	bidirectional bool
	numQueryHeads int
	numKVHeads    int
}

// WithFusedSDPABidirectional disables causal masking (encoder-style attention).
func WithFusedSDPABidirectional[T tensor.Numeric]() FusedSDPAOption[T] {
	return func(o *fusedSDPAOptions[T]) { o.bidirectional = true }
}

// WithFusedSDPAHeadCounts sets query/KV head counts so the inner SDPA can
// dispatch to the split-KV flash decode kernel where applicable.
func WithFusedSDPAHeadCounts[T tensor.Numeric](numQueryHeads, numKVHeads int) FusedSDPAOption[T] {
	return func(o *fusedSDPAOptions[T]) {
		o.numQueryHeads = numQueryHeads
		o.numKVHeads = numKVHeads
	}
}

// NewFusedSDPA constructs a FusedSDPA graph.Node wrapping a
// ScaledDotProductAttention configured with the given engine, head dimension,
// and options.
func NewFusedSDPA[T tensor.Numeric](engine compute.Engine[T], headDim int, opts ...FusedSDPAOption[T]) *FusedSDPA[T] {
	cfg := &fusedSDPAOptions[T]{}
	for _, opt := range opts {
		opt(cfg)
	}

	innerOpts := []ScaledDotProductAttentionOption[T]{}
	if cfg.bidirectional {
		innerOpts = append(innerOpts, WithBidirectional[T]())
	}
	if cfg.numQueryHeads > 0 || cfg.numKVHeads > 0 {
		innerOpts = append(innerOpts, WithHeadCounts[T](cfg.numQueryHeads, cfg.numKVHeads))
	}

	sdpa := NewScaledDotProductAttention(engine, headDim, innerOpts...)
	if !cfg.bidirectional {
		sdpa.SetCausal(true)
	}

	return &FusedSDPA[T]{inner: sdpa}
}

// NewFusedSDPAFrom wraps an existing ScaledDotProductAttention as a graph.Node.
// Useful when callers want to share state with an already-configured layer.
func NewFusedSDPAFrom[T tensor.Numeric](sdpa *ScaledDotProductAttention[T]) *FusedSDPA[T] {
	return &FusedSDPA[T]{inner: sdpa}
}

// SetSaver implements graph.SaverAware by fanning the graph's Saver into the
// inner SDPA. The inner SDPA caches Q/K/V and the attention weights in
// Forward and consumes them in Backward (which is called with nil inputs),
// so they are SAVE-class intermediates: without this wiring they would be
// unpinned arena intermediates on GPU engines, and downstream forward ops
// could overwrite them before Backward runs (zerfoo#864, the #842 class).
func (n *FusedSDPA[T]) SetSaver(s graph.Saver[T]) { n.inner.SetSaver(s) }

// OpType implements graph.Node.
func (n *FusedSDPA[T]) OpType() string { return "FusedSDPA" }

// Attributes implements graph.Node and reports configuration knobs needed to
// reconstruct the op (head dimension and causal flag).
func (n *FusedSDPA[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"head_dim": n.inner.headDim,
		"causal":   n.inner.causal,
	}
}

// Forward delegates to the inner SDPA. inputs must be (Q, K, V) or (Q, K, V, mask).
func (n *FusedSDPA[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 3 && len(inputs) != 4 {
		return nil, fmt.Errorf("FusedSDPA: expected 3 or 4 inputs (Q, K, V[, mask]), got %d", len(inputs))
	}
	q, k, v := inputs[0], inputs[1], inputs[2]
	var mask *tensor.TensorNumeric[T]
	if len(inputs) == 4 {
		mask = inputs[3]
	}
	n.hadMask = mask != nil

	out, err := n.inner.Forward(ctx, q, k, v, mask)
	if err != nil {
		return nil, err
	}
	if out != nil {
		n.outputShape = append(n.outputShape[:0], out.Shape()...)
	}
	return out, nil
}

// Backward delegates to the inner SDPA. The inner returns gradients for
// [Q, K, V]; we append a nil gradient for mask when one was supplied so the
// returned slice aligns 1:1 with the inputs passed to Forward.
func (n *FusedSDPA[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 0 && len(inputs) != 3 && len(inputs) != 4 {
		return nil, fmt.Errorf("FusedSDPA: backward expected 0, 3, or 4 inputs, got %d", len(inputs))
	}
	grads, err := n.inner.Backward(ctx, mode, outputGradient, nil, nil, nil)
	if err != nil {
		return nil, err
	}
	if n.hadMask || len(inputs) == 4 {
		grads = append(grads, nil)
	}
	return grads, nil
}

// Parameters implements graph.Node. SDPA has no trainable parameters of its own.
func (n *FusedSDPA[T]) Parameters() []*graph.Parameter[T] { return nil }

// OutputShape returns the cached shape of the most recent Forward, or nil.
func (n *FusedSDPA[T]) OutputShape() []int { return n.outputShape }
