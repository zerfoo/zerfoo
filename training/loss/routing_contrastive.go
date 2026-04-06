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
//
// Uses engine.MatMul to compute the gram matrix (all pairwise dot products)
// in a single operation per batch element, replacing O(numHeads^2) loops.
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
	ops := rc.ops
	eps := ops.FromFloat64(1e-12)

	if numHeads < 2 {
		return tensor.New[T]([]int{1}, []T{ops.FromFloat64(0)})
	}

	numPairs := numHeads * (numHeads - 1) / 2
	invPairs := ops.FromFloat64(1.0 / float64(numPairs*batch))

	// Split along batch axis: [batch, numHeads, seqLen] → batch * [1, numHeads, seqLen]
	batchSlices, err := rc.engine.Split(ctx, scores, batch, 0)
	if err != nil {
		return nil, fmt.Errorf("RoutingContrastive: split batch: %w", err)
	}

	var totalSimScalar T
	for _, bSlice := range batchSlices {
		// Reshape to [numHeads, seqLen]
		headsMat, err := rc.engine.Reshape(ctx, bSlice, []int{numHeads, seqLen})
		if err != nil {
			return nil, err
		}

		// Gram matrix: heads @ heads^T → [numHeads, numHeads]
		headsT, err := rc.engine.Transpose(ctx, headsMat, []int{1, 0})
		if err != nil {
			return nil, err
		}
		gram, err := rc.engine.MatMul(ctx, headsMat, headsT)
		if err != nil {
			return nil, err
		}

		// Squared norms per head: element-wise square + reduce.
		headsSq, err := rc.engine.Mul(ctx, headsMat, headsMat)
		if err != nil {
			return nil, err
		}
		normsSq, err := rc.engine.ReduceSum(ctx, headsSq, 1, false) // [numHeads]
		if err != nil {
			return nil, err
		}

		// Norm products matrix: sqrt(normsSq_i) * sqrt(normsSq_j) via outer product.
		// norms = sqrt(normsSq) → [numHeads]
		norms, err := rc.engine.UnaryOp(ctx, normsSq, ops.Sqrt)
		if err != nil {
			return nil, err
		}
		normsCol, err := rc.engine.Reshape(ctx, norms, []int{numHeads, 1})
		if err != nil {
			return nil, err
		}
		normsRow, err := rc.engine.Reshape(ctx, norms, []int{1, numHeads})
		if err != nil {
			return nil, err
		}
		normProd, err := rc.engine.MatMul(ctx, normsCol, normsRow) // [numHeads, numHeads]
		if err != nil {
			return nil, err
		}

		// Cosine similarity matrix = gram / (normProd + eps)
		normProdEps, err := rc.engine.AddScalar(ctx, normProd, eps)
		if err != nil {
			return nil, err
		}
		cosSim, err := rc.engine.Div(ctx, gram, normProdEps)
		if err != nil {
			return nil, err
		}

		// Sum of upper triangle = (sum_all - trace) / 2
		// For cosine similarity, trace = numHeads (each head with itself = 1.0)
		sumAll, err := rc.engine.ReduceSum(ctx, cosSim, 1, false) // [numHeads]
		if err != nil {
			return nil, err
		}
		sumTotal, err := rc.engine.ReduceSum(ctx, sumAll, 0, false) // scalar [1]
		if err != nil {
			return nil, err
		}
		totalVal := sumTotal.Data()[0]
		trace := ops.FromFloat64(float64(numHeads))
		upperTriSum := ops.Div(ops.Sub(totalVal, trace), ops.FromFloat64(2.0))
		totalSimScalar = ops.Add(totalSimScalar, upperTriSum)
	}

	loss := ops.Mul(rc.scale, ops.Mul(totalSimScalar, invPairs))
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
