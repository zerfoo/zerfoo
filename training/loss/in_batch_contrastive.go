package loss

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// InBatchContrastive trains pairs of query and document vectors with in-batch
// negatives. Target is a [batch, batch] probability matrix: each row must sum
// to one and can assign weight to multiple relevant documents. The caller
// supplies vectors in the same representation used at inference; no implicit
// pooling or normalization occurs here.
//
// Logits[i,j] = dot(query[i], document[j]) / temperature.
// The loss is mean cross entropy over query rows.
type InBatchContrastive[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	temperature float64
	ce          *CrossEntropyLossOneHot[T]
	query       *tensor.TensorNumeric[T]
	document    *tensor.TensorNumeric[T]
}

func NewInBatchContrastive[T tensor.Numeric](engine compute.Engine[T], temperature float64) (*InBatchContrastive[T], error) {
	if engine == nil || math.IsNaN(temperature) || math.IsInf(temperature, 0) || temperature <= 0 {
		return nil, fmt.Errorf("contrastive loss requires an engine and positive finite temperature")
	}
	return &InBatchContrastive[T]{engine: engine, temperature: temperature, ce: NewCrossEntropyLossOneHot(engine)}, nil
}

func (c *InBatchContrastive[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 3 || inputs[0] == nil || inputs[1] == nil || inputs[2] == nil {
		return nil, fmt.Errorf("contrastive loss expects query, document, and target tensors")
	}
	q, d, target := inputs[0], inputs[1], inputs[2]
	qs, ds, ts := q.Shape(), d.Shape(), target.Shape()
	if len(qs) != 2 || len(ds) != 2 || len(ts) != 2 || qs[0] <= 0 || qs[1] <= 0 || qs[0] != ds[0] || qs[1] != ds[1] || ts[0] != qs[0] || ts[1] != qs[0] {
		return nil, fmt.Errorf("contrastive loss expects query/document [batch, dim] and target [batch, batch], got %v, %v, %v", qs, ds, ts)
	}
	dT, err := c.engine.Transpose(ctx, d, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("transpose documents: %w", err)
	}
	logits, err := c.engine.MatMul(ctx, q, dT)
	if err != nil {
		return nil, fmt.Errorf("compute similarity: %w", err)
	}
	logits, err = c.engine.DivScalar(ctx, logits, c.engine.Ops().FromFloat64(c.temperature))
	if err != nil {
		return nil, fmt.Errorf("scale similarity: %w", err)
	}
	loss, err := c.ce.Forward(ctx, logits, target)
	if err != nil {
		return nil, fmt.Errorf("contrastive cross entropy: %w", err)
	}
	c.query, c.document = q, d
	return loss, nil
}

func (c *InBatchContrastive[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if c.query == nil || c.document == nil {
		return nil, fmt.Errorf("contrastive loss backward before forward")
	}
	grads, err := c.ce.Backward(ctx, mode, dOut)
	if err != nil {
		return nil, fmt.Errorf("contrastive cross entropy backward: %w", err)
	}
	dLogits, err := c.engine.DivScalar(ctx, grads[0], c.engine.Ops().FromFloat64(c.temperature))
	if err != nil {
		return nil, fmt.Errorf("scale contrastive gradient: %w", err)
	}
	dQuery, err := c.engine.MatMul(ctx, dLogits, c.document)
	if err != nil {
		return nil, fmt.Errorf("query gradient: %w", err)
	}
	dLogitsT, err := c.engine.Transpose(ctx, dLogits, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("transpose contrastive gradient: %w", err)
	}
	dDocument, err := c.engine.MatMul(ctx, dLogitsT, c.query)
	if err != nil {
		return nil, fmt.Errorf("document gradient: %w", err)
	}
	return []*tensor.TensorNumeric[T]{dQuery, dDocument, nil}, nil
}

func (c *InBatchContrastive[T]) OutputShape() []int { return []int{1} }
func (c *InBatchContrastive[T]) OpType() string     { return "InBatchContrastive" }
func (c *InBatchContrastive[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"temperature": c.temperature}
}
func (c *InBatchContrastive[T]) Parameters() []*graph.Parameter[T] { return nil }

var _ graph.Node[float32] = (*InBatchContrastive[float32])(nil)
