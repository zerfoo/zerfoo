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
	ops := rc.ops
	eps := ops.FromFloat64(1e-12)

	if numHeads < 2 {
		return tensor.New[T]([]int{1}, []T{ops.FromFloat64(0)})
	}

	numPairs := numHeads * (numHeads - 1) / 2
	invPairs := ops.FromFloat64(1.0 / float64(numPairs*batch))

	var totalSim T
	for b := range batch {
		// Extract [numHeads, seqLen] sub-matrix for this batch element.
		base := b * numHeads * seqLen
		headsMat, err := tensor.New[T]([]int{numHeads, seqLen}, scores.Data()[base:base+numHeads*seqLen])
		if err != nil {
			return nil, err
		}

		// Transpose to [seqLen, numHeads].
		headsT, err := rc.engine.Transpose(ctx, headsMat, []int{1, 0})
		if err != nil {
			return nil, err
		}

		// MatMul: [numHeads, seqLen] x [seqLen, numHeads] = [numHeads, numHeads] dot products.
		dotMat, err := rc.engine.MatMul(ctx, headsMat, headsT)
		if err != nil {
			return nil, err
		}

		// Compute per-head squared norms via engine: element-wise square then ReduceSum along axis 1.
		headsSq, err := rc.engine.Mul(ctx, headsMat, headsMat)
		if err != nil {
			return nil, err
		}
		normsSq, err := rc.engine.ReduceSum(ctx, headsSq, 1, false)
		if err != nil {
			return nil, err
		}
		norms, err := rc.engine.Sqrt(ctx, normsSq)
		if err != nil {
			return nil, err
		}

		// Extract computed values to sum upper-triangle cosine similarities.
		dotData := dotMat.Data()
		normsData := norms.Data()

		for i := range numHeads {
			for j := i + 1; j < numHeads; j++ {
				dotIJ := dotData[i*numHeads+j]
				denom := ops.Add(ops.Mul(normsData[i], normsData[j]), eps)
				totalSim = ops.Add(totalSim, ops.Div(dotIJ, denom))
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
	ops := rc.ops
	eps := ops.FromFloat64(1e-12)

	totalElems := batch * numHeads * seqLen
	gradData := make([]T, totalElems)

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
		// Extract [numHeads, seqLen] sub-matrix for this batch element.
		base := b * numHeads * seqLen
		headsMat, err := tensor.New[T]([]int{numHeads, seqLen}, scores.Data()[base:base+numHeads*seqLen])
		if err != nil {
			return nil, err
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

		// Transpose and MatMul for all pairwise dot products.
		headsT, err := rc.engine.Transpose(ctx, headsMat, []int{1, 0})
		if err != nil {
			return nil, err
		}
		dotMat, err := rc.engine.MatMul(ctx, headsMat, headsT)
		if err != nil {
			return nil, err
		}
		dotData := dotMat.Data()

		// Extract per-head 1D tensors for gradient computation.
		heads := make([]*tensor.TensorNumeric[T], numHeads)
		for h := range numHeads {
			hBase := h * seqLen
			heads[h], err = tensor.New[T]([]int{seqLen}, headsMat.Data()[hBase:hBase+seqLen])
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

				baseI := base + i*seqLen
				baseJ := base + j*seqLen

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
