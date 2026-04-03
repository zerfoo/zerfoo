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
			headI, err := tensor.New[T]([]int{seqLen}, data[(b*numHeads+i)*seqLen:(b*numHeads+i+1)*seqLen])
			if err != nil {
				return nil, err
			}
			normISqT, err := rc.engine.Mul(ctx, headI, headI)
			if err != nil {
				return nil, err
			}
			normISqSum, err := rc.engine.ReduceSum(ctx, normISqT, 0, false)
			if err != nil {
				return nil, err
			}
			normISq := normISqSum.Data()[0]

			for j := i + 1; j < numHeads; j++ {
				headJ, err := tensor.New[T]([]int{seqLen}, data[(b*numHeads+j)*seqLen:(b*numHeads+j+1)*seqLen])
				if err != nil {
					return nil, err
				}

				// Dot product via engine.Mul + engine.ReduceSum.
				prod, err := rc.engine.Mul(ctx, headI, headJ)
				if err != nil {
					return nil, err
				}
				dotT, err := rc.engine.ReduceSum(ctx, prod, 0, false)
				if err != nil {
					return nil, err
				}
				dotIJ := dotT.Data()[0]

				normJSqT, err := rc.engine.Mul(ctx, headJ, headJ)
				if err != nil {
					return nil, err
				}
				normJSqSum, err := rc.engine.ReduceSum(ctx, normJSqT, 0, false)
				if err != nil {
					return nil, err
				}
				normJSq := normJSqSum.Data()[0]

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
func (rc *RoutingContrastive[T]) Backward(ctx context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
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
		// Precompute squared norms for each head via engine ops.
		normsSq := make([]T, numHeads)
		heads := make([]*tensor.TensorNumeric[T], numHeads)
		for h := range numHeads {
			base := (b*numHeads + h) * seqLen
			head, err := tensor.New[T]([]int{seqLen}, data[base:base+seqLen])
			if err != nil {
				return nil, err
			}
			heads[h] = head
			sq, err := rc.engine.Mul(ctx, head, head)
			if err != nil {
				return nil, err
			}
			nT, err := rc.engine.ReduceSum(ctx, sq, 0, false)
			if err != nil {
				return nil, err
			}
			normsSq[h] = nT.Data()[0]
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

				// Dot product via engine ops.
				prod, err := rc.engine.Mul(ctx, heads[i], heads[j])
				if err != nil {
					return nil, err
				}
				dotT, err := rc.engine.ReduceSum(ctx, prod, 0, false)
				if err != nil {
					return nil, err
				}
				cosVal := ops.Div(dotT.Data()[0], denom)

				// Gradient via engine ops: coeffTensor * (headJ/denom - cosVal * headI/normISqEps)
				denomT, err := tensor.New[T]([]int{seqLen}, scalarFill(seqLen, denom))
				if err != nil {
					return nil, err
				}
				cosValT, err := tensor.New[T]([]int{seqLen}, scalarFill(seqLen, cosVal))
				if err != nil {
					return nil, err
				}
				coeffT, err := tensor.New[T]([]int{seqLen}, scalarFill(seqLen, coeff))
				if err != nil {
					return nil, err
				}

				// dI = coeff * (headJ/denom - cosVal * headI/normISqEps)
				normISqEpsT, err := tensor.New[T]([]int{seqLen}, scalarFill(seqLen, normISqEps))
				if err != nil {
					return nil, err
				}
				term1I, err := rc.engine.Div(ctx, heads[j], denomT)
				if err != nil {
					return nil, err
				}
				scaledI, err := rc.engine.Div(ctx, heads[i], normISqEpsT)
				if err != nil {
					return nil, err
				}
				cosScaledI, err := rc.engine.Mul(ctx, cosValT, scaledI)
				if err != nil {
					return nil, err
				}
				diffI, err := rc.engine.Sub(ctx, term1I, cosScaledI)
				if err != nil {
					return nil, err
				}
				dI, err := rc.engine.Mul(ctx, coeffT, diffI)
				if err != nil {
					return nil, err
				}

				// dJ = coeff * (headI/denom - cosVal * headJ/normJSqEps)
				normJSqEpsT, err := tensor.New[T]([]int{seqLen}, scalarFill(seqLen, normJSqEps))
				if err != nil {
					return nil, err
				}
				term1J, err := rc.engine.Div(ctx, heads[i], denomT)
				if err != nil {
					return nil, err
				}
				scaledJ, err := rc.engine.Div(ctx, heads[j], normJSqEpsT)
				if err != nil {
					return nil, err
				}
				cosScaledJ, err := rc.engine.Mul(ctx, cosValT, scaledJ)
				if err != nil {
					return nil, err
				}
				diffJ, err := rc.engine.Sub(ctx, term1J, cosScaledJ)
				if err != nil {
					return nil, err
				}
				dJ, err := rc.engine.Mul(ctx, coeffT, diffJ)
				if err != nil {
					return nil, err
				}

				// Accumulate gradients.
				dIData := dI.Data()
				dJData := dJ.Data()
				for s := range seqLen {
					gradData[baseI+s] = ops.Add(gradData[baseI+s], dIData[s])
					gradData[baseJ+s] = ops.Add(gradData[baseJ+s], dJData[s])
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

// scalarFill returns a slice of length n filled with value v.
func scalarFill[T tensor.Numeric](n int, v T) []T {
	s := make([]T, n)
	for i := range s {
		s[i] = v
	}
	return s
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
