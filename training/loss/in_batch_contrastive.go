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
// negatives. Target is a [queries, documents] nonnegative weight matrix;
// rows may contain multiple positives or be all zero. The caller
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
	target      *tensor.TensorNumeric[T]
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
	if len(qs) != 2 || len(ds) != 2 || len(ts) != 2 || qs[0] <= 0 || ds[0] <= 0 || qs[1] <= 0 || qs[1] != ds[1] || ts[0] != qs[0] || ts[1] != ds[0] {
		return nil, fmt.Errorf("contrastive loss expects query [queries, dim], document [documents, dim], and target [queries, documents], got %v, %v, %v", qs, ds, ts)
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
	c.query, c.document, c.target = q, d, target
	return loss, nil
}

func (c *InBatchContrastive[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if c.query == nil || c.document == nil {
		return nil, fmt.Errorf("contrastive loss backward before forward")
	}
	// For weighted targets, dCE/dlogits = softmax*sum(target row)-target.
	// The shared CE node assumes each target row sums to one, so derive the
	// gradient here. An all-zero row then has zero loss and zero gradient.
	rowSums, err := c.engine.ReduceSum(ctx, c.target, 1, false)
	if err != nil {
		return nil, fmt.Errorf("sum contrastive targets: %w", err)
	}
	rowSums, err = c.engine.Reshape(ctx, rowSums, []int{c.query.Shape()[0], 1})
	if err != nil {
		return nil, fmt.Errorf("reshape contrastive targets: %w", err)
	}
	weighted, err := c.engine.Mul(ctx, c.ce.SoftmaxOutput(), rowSums, nil)
	if err != nil {
		return nil, fmt.Errorf("weight contrastive probabilities: %w", err)
	}
	grad, err := c.engine.Sub(ctx, weighted, c.target, nil)
	if err != nil {
		return nil, fmt.Errorf("subtract contrastive targets: %w", err)
	}
	grad, err = c.engine.MulScalar(ctx, grad, c.engine.Ops().FromFloat64(1.0/float64(c.query.Shape()[0])))
	if err != nil {
		return nil, fmt.Errorf("average contrastive gradient: %w", err)
	}
	grad, err = c.engine.Mul(ctx, grad, dOut, nil)
	if err != nil {
		return nil, fmt.Errorf("scale contrastive upstream gradient: %w", err)
	}
	dLogits, err := c.engine.DivScalar(ctx, grad, c.engine.Ops().FromFloat64(c.temperature))
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
