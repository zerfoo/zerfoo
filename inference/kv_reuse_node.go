// Package inference: KVReuseNode bridges a donor attention layer's K or V
// port (shape [B, numKVHeads, S, headDim]) to the layout expected by a
// downstream GroupedQueryAttention layer configured with WithExternalKV
// (shape [B, S, numKVHeads*headDim]).
//
// The node exists to make the donor -> shared-KV edge an explicit graph node
// for readability, impact tracing, and CUDA graph capture (ADR-087). Builders
// wire donor.KPort()/VPort() through a KVReuseNode before passing the result
// as inputs[1]/inputs[2] to the consumer GQA's Forward.
package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// KVReuseNode is a thin pass-through graph node that reads a donor GQA
// layer's K or V port (produced in [B, numKVHeads, S, headDim] layout) and
// emits the K/V in the [B, S, numKVHeads*headDim] layout consumed by the
// downstream consumer GQA's Forward when WithExternalKV is active.
//
// The node owns no parameters, has no gradient of its own, and defers to the
// donor port for source semantics. Forward must be called AFTER the donor
// layer's Forward pass; otherwise the donor port will error.
type KVReuseNode[T tensor.Numeric] struct {
	engine           compute.Engine[T]
	donor            graph.Node[T]
	numKVHeads       int
	headDim          int
	isKey            bool
	lastOutputShape  []int
}

// NewKVReuseNode constructs a KVReuseNode that wraps the given donor port
// (typically a result of donorGQA.KPort() or donorGQA.VPort()).
//
// numKVHeads and headDim describe the donor layer's K/V geometry; the node
// flattens the head and headDim axes to produce the consumer layer's
// external-K/V input layout.
func NewKVReuseNode[T tensor.Numeric](
	engine compute.Engine[T],
	donor graph.Node[T],
	numKVHeads, headDim int,
	isKey bool,
) (*KVReuseNode[T], error) {
	if engine == nil {
		return nil, fmt.Errorf("KVReuseNode: engine must not be nil")
	}
	if donor == nil {
		return nil, fmt.Errorf("KVReuseNode: donor must not be nil")
	}
	if numKVHeads <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("KVReuseNode: numKVHeads and headDim must be positive (got %d, %d)", numKVHeads, headDim)
	}
	return &KVReuseNode[T]{
		engine:     engine,
		donor:      donor,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		isKey:      isKey,
	}, nil
}

// OpType identifies the node for graph dumps and debugging.
func (n *KVReuseNode[T]) OpType() string {
	if n.isKey {
		return "KVReuseNode.K"
	}
	return "KVReuseNode.V"
}

// Attributes returns the node's static attributes.
func (n *KVReuseNode[T]) Attributes() map[string]interface{} {
	kind := "v"
	if n.isKey {
		kind = "k"
	}
	return map[string]interface{}{
		"port":         kind,
		"num_kv_heads": n.numKVHeads,
		"head_dim":     n.headDim,
	}
}

// Forward pulls the donor's cached K/V tensor (shape [B, numKVHeads, S, headDim]),
// transposes it to [B, S, numKVHeads, headDim], and reshapes to
// [B, S, numKVHeads*headDim] -- the layout expected by GroupedQueryAttention
// Forward when WithExternalKV is active. Inputs to this node are ignored.
func (n *KVReuseNode[T]) Forward(ctx context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	src, err := n.donor.Forward(ctx)
	if err != nil {
		return nil, fmt.Errorf("KVReuseNode: donor port Forward: %w", err)
	}
	shape := src.Shape()
	if len(shape) != 4 {
		return nil, fmt.Errorf("KVReuseNode: donor tensor rank = %d, want 4 ([B, numKVHeads, S, headDim])", len(shape))
	}
	if shape[1] != n.numKVHeads || shape[3] != n.headDim {
		return nil, fmt.Errorf("KVReuseNode: donor shape %v incompatible with numKVHeads=%d headDim=%d",
			shape, n.numKVHeads, n.headDim)
	}
	batch, seqLen := shape[0], shape[2]

	// [B, numKV, S, headDim] -> [B, S, numKV, headDim]
	transposed, err := n.engine.Transpose(ctx, src, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("KVReuseNode: transpose: %w", err)
	}
	// [B, S, numKV, headDim] -> [B, S, numKV*headDim]
	out, err := n.engine.Reshape(ctx, transposed, []int{batch, seqLen, n.numKVHeads * n.headDim})
	if err != nil {
		return nil, fmt.Errorf("KVReuseNode: reshape: %w", err)
	}
	n.lastOutputShape = out.Shape()
	return out, nil
}

// Backward is a no-op: gradients for shared-KV flow back through the donor
// GQA layer, not through this adapter. In external-KV mode the consumer GQA's
// Backward is itself unsupported (see ADR-087 implementation notes).
func (n *KVReuseNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Parameters returns nil -- the node carries no trainable state of its own.
func (n *KVReuseNode[T]) Parameters() []*graph.Parameter[T] { return nil }

// OutputShape returns the last produced output shape, or nil before the first
// Forward call (the donor must run first to determine batch/seqLen).
func (n *KVReuseNode[T]) OutputShape() []int {
	return n.lastOutputShape
}

// Statically assert KVReuseNode implements graph.Node.
var _ graph.Node[float32] = (*KVReuseNode[float32])(nil)
