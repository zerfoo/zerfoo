package attention

import (
	"context"
	"math"
	"sort"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// NSAFineSelection implements the fine-grained token selection path of
// Natively Sparse Attention (NSA). For each query position it scores all
// KV positions via Q*K^T / sqrt(d), selects the top-f tokens by score,
// and computes softmax attention only over those selected tokens.
type NSAFineSelection[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	topTokens  int // f — number of tokens to keep per query position
	numHeads   int
	numKVHeads int
	headDim    int
}

// NewNSAFineSelection creates a new NSAFineSelection layer.
//
// Parameters:
//   - engine: compute engine for tensor operations
//   - topTokens: number of tokens to select per query position (f)
//   - numHeads: number of query heads
//   - numKVHeads: number of key/value heads
//   - headDim: dimension of each head
func NewNSAFineSelection[T tensor.Numeric](
	engine compute.Engine[T],
	topTokens, numHeads, numKVHeads, headDim int,
) *NSAFineSelection[T] {
	return &NSAFineSelection[T]{
		engine:     engine,
		topTokens:  topTokens,
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
	}
}

// Forward computes the fine-grained token selection attention.
//
// Inputs:
//   - Q: [batch, numHeads, seqQ, headDim]
//   - K: [batch, numKVHeads, seqKV, headDim]
//   - V: [batch, numKVHeads, seqKV, headDim]
//
// Returns:
//   - output: [batch, numHeads, seqQ, headDim]
func (n *NSAFineSelection[T]) Forward(
	ctx context.Context,
	Q, K, V *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	qShape := Q.Shape()
	kShape := K.Shape()

	batch := qShape[0]
	numHeads := qShape[1]
	seqQ := qShape[2]
	headDim := qShape[3]
	seqKV := kShape[2]

	// GQA repeat factor: how many query heads share each KV head.
	gqaFactor := numHeads / n.numKVHeads

	// Clamp topTokens to available KV length.
	f := n.topTokens
	if f > seqKV {
		f = seqKV
	}

	scale := 1.0 / math.Sqrt(float64(headDim))

	qData := Q.Data()
	kData := K.Data()
	vData := V.Data()

	outSize := batch * numHeads * seqQ * headDim
	out := make([]T, outSize)

	for b := range batch {
		for h := range numHeads {
			kvh := h / gqaFactor // which KV head this query head maps to

			for qi := range seqQ {
				// Compute scores: Q[b,h,qi,:] dot K[b,kvh,ki,:] for all ki.
				qOff := ((b*numHeads+h)*seqQ + qi) * headDim
				scores := make([]float64, seqKV)
				for ki := range seqKV {
					kOff := ((b*n.numKVHeads+kvh)*seqKV + ki) * headDim
					var dot float64
					for d := range headDim {
						dot += float64(qData[qOff+d]) * float64(kData[kOff+d])
					}
					scores[ki] = dot * scale
				}

				// Select top-f indices by score.
				indices := make([]int, seqKV)
				for i := range indices {
					indices[i] = i
				}
				sort.Slice(indices, func(a, b int) bool {
					return scores[indices[a]] > scores[indices[b]]
				})
				topIndices := indices[:f]

				// Softmax over selected scores.
				maxScore := scores[topIndices[0]]
				for _, idx := range topIndices[1:] {
					if scores[idx] > maxScore {
						maxScore = scores[idx]
					}
				}
				expSum := 0.0
				weights := make([]float64, f)
				for i, idx := range topIndices {
					w := math.Exp(scores[idx] - maxScore)
					weights[i] = w
					expSum += w
				}
				for i := range weights {
					weights[i] /= expSum
				}

				// Weighted sum of V at selected indices.
				oOff := ((b*numHeads+h)*seqQ + qi) * headDim
				for i, idx := range topIndices {
					vOff := ((b*n.numKVHeads+kvh)*seqKV + idx) * headDim
					w := weights[i]
					for d := range headDim {
						out[oOff+d] += T(w * float64(vData[vOff+d]))
					}
				}
			}
		}
	}

	return tensor.New[T]([]int{batch, numHeads, seqQ, headDim}, out)
}

// SelectedIndices computes and returns the top-f token indices per query
// position without computing the full attention output. This is useful for
// testing that the selection logic matches manual computation.
//
// Returns indices with shape [batch, numHeads, seqQ, topTokens] (sorted
// descending by score).
func (n *NSAFineSelection[T]) SelectedIndices(
	Q, K *tensor.TensorNumeric[T],
) []int {
	qShape := Q.Shape()
	kShape := K.Shape()

	batch := qShape[0]
	numHeads := qShape[1]
	seqQ := qShape[2]
	headDim := qShape[3]
	seqKV := kShape[2]

	gqaFactor := numHeads / n.numKVHeads

	f := n.topTokens
	if f > seqKV {
		f = seqKV
	}

	scale := 1.0 / math.Sqrt(float64(headDim))

	qData := Q.Data()
	kData := K.Data()

	result := make([]int, batch*numHeads*seqQ*f)

	for b := range batch {
		for h := range numHeads {
			kvh := h / gqaFactor

			for qi := range seqQ {
				qOff := ((b*numHeads+h)*seqQ + qi) * headDim
				scores := make([]float64, seqKV)
				for ki := range seqKV {
					kOff := ((b*n.numKVHeads+kvh)*seqKV + ki) * headDim
					var dot float64
					for d := range headDim {
						dot += float64(qData[qOff+d]) * float64(kData[kOff+d])
					}
					scores[ki] = dot * scale
				}

				indices := make([]int, seqKV)
				for i := range indices {
					indices[i] = i
				}
				sort.Slice(indices, func(a, b int) bool {
					return scores[indices[a]] > scores[indices[b]]
				})

				base := ((b*numHeads+h)*seqQ + qi) * f
				copy(result[base:base+f], indices[:f])
			}
		}
	}

	return result
}
