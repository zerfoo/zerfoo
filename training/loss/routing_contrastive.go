package loss

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// RoutingContrastive computes an auxiliary contrastive loss over routing scores
// from SparseRoutedAttention. It encourages routing diversity (different heads
// attend to different document regions) and routing specificity (each head
// specializes on a subset of documents).
//
// Input: routing scores tensor with shape [batch, numHeads, seqLen].
//
// The loss is the mean pairwise cosine similarity between head routing
// distributions. Minimizing this pushes heads apart so they specialize on
// different sequence regions.
//
// Loss = scale * mean(cosineSim(head_i, head_j)) for all i < j.
type RoutingContrastive[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	scale  T

	// Cached input for backward pass.
	scores *tensor.TensorNumeric[T]
}

// NewRoutingContrastive creates a new contrastive routing loss.
// scale controls the loss magnitude (default recommendation: 0.01).
func NewRoutingContrastive[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], scale float64) *RoutingContrastive[T] {
	return &RoutingContrastive[T]{
		engine: engine,
		ops:    ops,
		scale:  ops.FromFloat64(scale),
	}
}

// Forward computes the contrastive routing loss.
//
// Inputs: exactly one tensor of shape [batch, numHeads, seqLen].
// Returns a scalar loss tensor of shape [1].
func (rc *RoutingContrastive[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RoutingContrastive: expected 1 input (routing scores), got %d", len(inputs))
	}

	scores := inputs[0]
	shape := scores.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("RoutingContrastive: expected 3D input [batch, numHeads, seqLen], got %dD", len(shape))
	}

	rc.scores = scores

	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	data := scores.Data()
	ops := rc.ops
	eps := ops.FromFloat64(1e-12)

	if numHeads < 2 {
		return tensor.New[T]([]int{1}, []T{ops.FromFloat64(0)})
	}

	numPairs := numHeads * (numHeads - 1) / 2
	invPairs := ops.FromFloat64(1.0 / float64(numPairs*batch))

	var totalSim T
	for b := range batch {
		for i := range numHeads {
			for j := i + 1; j < numHeads; j++ {
				var dotIJ, normISq, normJSq T
				baseI := (b*numHeads + i) * seqLen
				baseJ := (b*numHeads + j) * seqLen
				for s := range seqLen {
					vi := data[baseI+s]
					vj := data[baseJ+s]
					dotIJ = ops.Add(dotIJ, ops.Mul(vi, vj))
					normISq = ops.Add(normISq, ops.Mul(vi, vi))
					normJSq = ops.Add(normJSq, ops.Mul(vj, vj))
				}
				denom := ops.Add(ops.Sqrt(ops.Mul(normISq, normJSq)), eps)
				sim := ops.Div(dotIJ, denom)
				totalSim = ops.Add(totalSim, sim)
			}
		}
	}

	loss := ops.Mul(rc.scale, ops.Mul(totalSim, invPairs))
	return tensor.New[T]([]int{1}, []T{loss})
}

// Backward computes gradients of the contrastive routing loss with respect to
// the routing scores input.
//
// d(loss)/d(scores[b,h,s]) = scale / (numPairs*batch) *
//
//	sum over j!=h of d(cosineSim(h,j))/d(scores[b,h,s])
//
// where d(cos(a,b))/d(a_i) = (b_i / (|a|*|b|)) - cos(a,b) * (a_i / |a|^2)
func (rc *RoutingContrastive[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	scores := rc.scores
	if len(inputs) > 0 {
		scores = inputs[0]
	}
	if scores == nil {
		return nil, fmt.Errorf("RoutingContrastive: no cached scores for backward")
	}

	shape := scores.Shape()
	batch := shape[0]
	numHeads := shape[1]
	seqLen := shape[2]
	data := scores.Data()
	ops := rc.ops
	eps := ops.FromFloat64(1e-12)

	gradData := make([]T, len(data))

	if numHeads < 2 {
		grad, err := tensor.New[T](shape, gradData)
		if err != nil {
			return nil, err
		}
		return []*tensor.TensorNumeric[T]{grad}, nil
	}

	numPairs := numHeads * (numHeads - 1) / 2
	coeff := ops.Mul(rc.scale, ops.FromFloat64(1.0/float64(numPairs*batch)))

	for b := range batch {
		// Precompute squared norms for each head.
		normsSq := make([]T, numHeads)
		for h := range numHeads {
			var n T
			base := (b*numHeads + h) * seqLen
			for s := range seqLen {
				v := data[base+s]
				n = ops.Add(n, ops.Mul(v, v))
			}
			normsSq[h] = n
		}

		for i := range numHeads {
			for j := i + 1; j < numHeads; j++ {
				normI := ops.Sqrt(normsSq[i])
				normJ := ops.Sqrt(normsSq[j])
				denom := ops.Add(ops.Mul(normI, normJ), eps)
				normISqEps := ops.Add(normsSq[i], eps)
				normJSqEps := ops.Add(normsSq[j], eps)

				baseI := (b*numHeads + i) * seqLen
				baseJ := (b*numHeads + j) * seqLen

				// Compute dot product.
				var dot T
				for s := range seqLen {
					dot = ops.Add(dot, ops.Mul(data[baseI+s], data[baseJ+s]))
				}
				cosVal := ops.Div(dot, denom)

				// Gradient contributions.
				for s := range seqLen {
					vi := data[baseI+s]
					vj := data[baseJ+s]

					// d(cos)/d(a_s) = b_s/(|a|*|b|) - cos * a_s/|a|^2
					dI := ops.Mul(coeff, ops.Sub(
						ops.Div(vj, denom),
						ops.Mul(cosVal, ops.Div(vi, normISqEps)),
					))
					// d(cos)/d(b_s) = a_s/(|a|*|b|) - cos * b_s/|b|^2
					dJ := ops.Mul(coeff, ops.Sub(
						ops.Div(vi, denom),
						ops.Mul(cosVal, ops.Div(vj, normJSqEps)),
					))

					gradData[baseI+s] = ops.Add(gradData[baseI+s], dI)
					gradData[baseJ+s] = ops.Add(gradData[baseJ+s], dJ)
				}
			}
		}
	}

	grad, err := tensor.New[T](shape, gradData)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{grad}, nil
}

// OutputShape returns the output shape of the loss (scalar).
func (rc *RoutingContrastive[T]) OutputShape() []int {
	return []int{1}
}

// OpType returns the operation type identifier.
func (rc *RoutingContrastive[T]) OpType() string {
	return "RoutingContrastive"
}

// Attributes returns the layer configuration.
func (rc *RoutingContrastive[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"scale": rc.scale,
	}
}

// Parameters returns nil (no trainable parameters).
func (rc *RoutingContrastive[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Statically assert that RoutingContrastive implements graph.Node.
var _ graph.Node[float32] = (*RoutingContrastive[float32])(nil)
