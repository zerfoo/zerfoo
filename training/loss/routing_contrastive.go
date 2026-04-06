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
	ops := rc.ops
	eps := ops.FromFloat64(1e-12)

	if numHeads < 2 {
		grad, err := tensor.New[T](shape, make([]T, batch*numHeads*seqLen))
		if err != nil {
			return nil, err
		}
		return []*tensor.TensorNumeric[T]{grad}, nil
	}

	numPairs := numHeads * (numHeads - 1) / 2
	coeff := ops.Mul(rc.scale, ops.FromFloat64(1.0/float64(numPairs*batch)))

	// Split along batch axis.
	batchSlices, err := rc.engine.Split(ctx, scores, batch, 0)
	if err != nil {
		return nil, fmt.Errorf("RoutingContrastive backward: split batch: %w", err)
	}

	batchGrads := make([]*tensor.TensorNumeric[T], batch)

	for b := range batch {
		// Reshape [1, numHeads, seqLen] → [numHeads, seqLen].
		headsMat, err := rc.engine.Reshape(ctx, batchSlices[b], []int{numHeads, seqLen})
		if err != nil {
			return nil, err
		}

		// Split into individual head tensors: numHeads * [1, seqLen].
		headSlices, err := rc.engine.Split(ctx, headsMat, numHeads, 0)
		if err != nil {
			return nil, err
		}
		heads := make([]*tensor.TensorNumeric[T], numHeads)
		for h := range numHeads {
			heads[h], err = rc.engine.Reshape(ctx, headSlices[h], []int{seqLen})
			if err != nil {
				return nil, err
			}
		}

		// Compute per-head squared norms via engine.
		headsSq, err := rc.engine.Mul(ctx, headsMat, headsMat)
		if err != nil {
			return nil, err
		}
		normsSqT, err := rc.engine.ReduceSum(ctx, headsSq, 1, false)
		if err != nil {
			return nil, err
		}
		normsSqData := normsSqT.Data()

		// Pairwise dot products via MatMul.
		headsT, err := rc.engine.Transpose(ctx, headsMat, []int{1, 0})
		if err != nil {
			return nil, err
		}
		dotMat, err := rc.engine.MatMul(ctx, headsMat, headsT)
		if err != nil {
			return nil, err
		}
		dotData := dotMat.Data()

		// Initialize per-head gradient accumulators as zero tensors.
		headGrads := make([]*tensor.TensorNumeric[T], numHeads)
		for h := range numHeads {
			headGrads[h], err = tensor.New[T]([]int{seqLen}, make([]T, seqLen))
			if err != nil {
				return nil, err
			}
		}

		for i := range numHeads {
			for j := i + 1; j < numHeads; j++ {
				normI := ops.Sqrt(normsSqData[i])
				normJ := ops.Sqrt(normsSqData[j])
				denom := ops.Add(ops.Mul(normI, normJ), eps)
				normISqEps := ops.Add(normsSqData[i], eps)
				normJSqEps := ops.Add(normsSqData[j], eps)

				cosVal := ops.Div(dotData[i*numHeads+j], denom)

				// dI = coeff * (headJ/denom - cosVal * headI/normISqEps)
				term1I, err := rc.engine.DivScalar(ctx, heads[j], denom)
				if err != nil {
					return nil, err
				}
				scaledI, err := rc.engine.DivScalar(ctx, heads[i], normISqEps)
				if err != nil {
					return nil, err
				}
				cosScaledI, err := rc.engine.MulScalar(ctx, scaledI, cosVal)
				if err != nil {
					return nil, err
				}
				diffI, err := rc.engine.Sub(ctx, term1I, cosScaledI)
				if err != nil {
					return nil, err
				}
				dI, err := rc.engine.MulScalar(ctx, diffI, coeff)
				if err != nil {
					return nil, err
				}

				// dJ = coeff * (headI/denom - cosVal * headJ/normJSqEps)
				term1J, err := rc.engine.DivScalar(ctx, heads[i], denom)
				if err != nil {
					return nil, err
				}
				scaledJ, err := rc.engine.DivScalar(ctx, heads[j], normJSqEps)
				if err != nil {
					return nil, err
				}
				cosScaledJ, err := rc.engine.MulScalar(ctx, scaledJ, cosVal)
				if err != nil {
					return nil, err
				}
				diffJ, err := rc.engine.Sub(ctx, term1J, cosScaledJ)
				if err != nil {
					return nil, err
				}
				dJ, err := rc.engine.MulScalar(ctx, diffJ, coeff)
				if err != nil {
					return nil, err
				}

				// Accumulate gradients via engine.Add.
				headGrads[i], err = rc.engine.Add(ctx, headGrads[i], dI)
				if err != nil {
					return nil, err
				}
				headGrads[j], err = rc.engine.Add(ctx, headGrads[j], dJ)
				if err != nil {
					return nil, err
				}
			}
		}

		// Concatenate head gradients and reshape to [1, numHeads, seqLen].
		batchGradFlat, err := rc.engine.Concat(ctx, headGrads, 0)
		if err != nil {
			return nil, err
		}
		batchGrads[b], err = rc.engine.Reshape(ctx, batchGradFlat, []int{1, numHeads, seqLen})
		if err != nil {
			return nil, err
		}
	}

	// Concatenate batch gradients into [batch, numHeads, seqLen].
	grad, err := rc.engine.Concat(ctx, batchGrads, 0)
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
