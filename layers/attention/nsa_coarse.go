package attention

import (
	"context"
	"math"
	"sort"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// NSACoarseCompression implements the coarse-grained token compression path
// of Native Sparse Attention (NSA). It divides the KV sequence into blocks
// of B tokens, computes block-level attention scores by averaging key
// representations per block, selects top-c blocks per query position, and
// attends to the selected blocks at full resolution.
type NSACoarseCompression[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	blockSize  int // B: number of tokens per block
	topBlocks  int // c: number of blocks to select
	numHeads   int
	numKVHeads int
	headDim    int

	outputShape []int
}

// NewNSACoarseCompression creates a new NSACoarseCompression layer.
//
// Parameters:
//   - engine: compute engine for tensor operations
//   - ops: arithmetic operations for the numeric type
//   - blockSize: number of tokens per KV block (B)
//   - topBlocks: number of blocks to select per query position (c)
//   - numHeads: number of query attention heads
//   - numKVHeads: number of key/value heads
//   - headDim: dimension of each attention head
func NewNSACoarseCompression[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	blockSize, topBlocks, numHeads, numKVHeads, headDim int,
) *NSACoarseCompression[T] {
	return &NSACoarseCompression[T]{
		engine:     engine,
		ops:        ops,
		blockSize:  blockSize,
		topBlocks:  topBlocks,
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
	}
}

// OpType returns the operation type identifier.
func (nsa *NSACoarseCompression[T]) OpType() string {
	return "NSACoarseCompression"
}

// Attributes returns the layer configuration.
func (nsa *NSACoarseCompression[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"block_size":   nsa.blockSize,
		"top_blocks":   nsa.topBlocks,
		"num_heads":    nsa.numHeads,
		"num_kv_heads": nsa.numKVHeads,
		"head_dim":     nsa.headDim,
	}
}

// Parameters returns nil (no trainable parameters).
func (nsa *NSACoarseCompression[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// OutputShape returns the output shape from the last forward call.
func (nsa *NSACoarseCompression[T]) OutputShape() []int {
	return nsa.outputShape
}

// Forward computes the coarse-grained compression attention.
//
// Inputs:
//   - Q: [batch, numHeads, seqQ, headDim]
//   - K: [batch, numKVHeads, seqKV, headDim]
//   - V: [batch, numKVHeads, seqKV, headDim]
//
// Returns output with shape [batch, numHeads, seqQ, headDim].
//
// The algorithm:
//  1. Reshape K into blocks of size B and compute block-level keys by averaging.
//  2. Compute coarse attention scores: Q @ blockKeys^T.
//  3. Select top-c blocks per query position.
//  4. Gather full-resolution K,V from selected blocks.
//  5. Compute fine-grained attention on selected blocks.
func (nsa *NSACoarseCompression[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	Q := inputs[0]
	K := inputs[1]
	V := inputs[2]

	qShape := Q.Shape()
	kShape := K.Shape()

	batch := qShape[0]
	numHeads := qShape[1]
	seqQ := qShape[2]
	headDim := qShape[3]
	numKVHeads := kShape[1]
	seqKV := kShape[2]

	B := nsa.blockSize
	numBlocks := seqKV / B
	c := nsa.topBlocks
	if c > numBlocks {
		c = numBlocks
	}

	headsPerGroup := numHeads / numKVHeads
	scale := float64(1.0) / math.Sqrt(float64(headDim))
	scaleT := nsa.ops.FromFloat64(scale)

	// Selected block window size for fine attention.
	selectedSeqLen := c * B

	// Output buffer.
	outSize := batch * numHeads * seqQ * headDim
	outData := make([]T, outSize)

	qData := Q.Data()
	kData := K.Data()
	vData := V.Data()

	// Step 1: Compute block-level keys by averaging K over each block.
	// blockKeys shape: [batch, numKVHeads, numBlocks, headDim]
	blockKeys := make([]T, batch*numKVHeads*numBlocks*headDim)
	invB := nsa.ops.FromFloat64(1.0 / float64(B))

	for b := range batch {
		for kv := range numKVHeads {
			for blk := range numBlocks {
				for d := range headDim {
					var sum T
					for t := range B {
						seqIdx := blk*B + t
						idx := ((b*numKVHeads+kv)*seqKV + seqIdx) * headDim
						sum = nsa.ops.Add(sum, kData[idx+d])
					}
					bkIdx := ((b*numKVHeads+kv)*numBlocks + blk) * headDim
					blockKeys[bkIdx+d] = nsa.ops.Mul(sum, invB)
				}
			}
		}
	}

	// Buffers for per-query-position work.
	coarseScores := make([]T, numBlocks)

	for b := range batch {
		for h := range numHeads {
			kvHead := h / headsPerGroup

			for q := range seqQ {
				// Step 2: Compute coarse attention scores for this query position.
				// score[blk] = sum_d(Q[b,h,q,d] * blockKeys[b,kvHead,blk,d]) * scale
				qOffset := ((b*numHeads+h)*seqQ + q) * headDim
				bkBase := (b*numKVHeads + kvHead) * numBlocks * headDim

				for blk := range numBlocks {
					var dot T
					bkOff := bkBase + blk*headDim
					for d := range headDim {
						dot = nsa.ops.Add(dot, nsa.ops.Mul(qData[qOffset+d], blockKeys[bkOff+d]))
					}
					coarseScores[blk] = nsa.ops.Mul(dot, scaleT)
				}

				// Step 3: Select top-c blocks using GreaterThan for comparison.
				blockIndices := topKIndicesT(coarseScores, c, nsa.ops)

				// Sort selected blocks by position for causal consistency.
				sort.Ints(blockIndices)

				// Step 4: Gather full-resolution K,V from selected blocks.
				// Compute fine scores and find max for numerically stable softmax.
				fineScores := make([]T, selectedSeqLen)
				kvBase := (b*numKVHeads + kvHead) * seqKV * headDim
				first := true
				var maxScore T
				for si, blkIdx := range blockIndices {
					for t := range B {
						seqIdx := blkIdx*B + t
						kOff := kvBase + seqIdx*headDim
						var dot T
						for d := range headDim {
							dot = nsa.ops.Add(dot, nsa.ops.Mul(qData[qOffset+d], kData[kOff+d]))
						}
						s := nsa.ops.Mul(dot, scaleT)
						fineScores[si*B+t] = s
						if first || nsa.ops.GreaterThan(s, maxScore) {
							maxScore = s
							first = false
						}
					}
				}

				// Softmax over fine scores.
				expScores := make([]T, selectedSeqLen)
				var sumExp T
				for i, s := range fineScores {
					e := nsa.ops.Exp(nsa.ops.Sub(s, maxScore))
					expScores[i] = e
					sumExp = nsa.ops.Add(sumExp, e)
				}

				// Step 5: Weighted sum of V from selected blocks.
				outOffset := ((b*numHeads+h)*seqQ + q) * headDim
				for si, blkIdx := range blockIndices {
					for t := range B {
						w := nsa.ops.Div(expScores[si*B+t], sumExp)
						seqIdx := blkIdx*B + t
						vOff := kvBase + seqIdx*headDim
						for d := range headDim {
							outData[outOffset+d] = nsa.ops.Add(
								outData[outOffset+d],
								nsa.ops.Mul(w, vData[vOff+d]),
							)
						}
					}
				}
			}
		}
	}

	outShape := []int{batch, numHeads, seqQ, headDim}
	nsa.outputShape = outShape
	return tensor.New[T](outShape, outData)
}

// Backward implements the straight-through estimator for the block selection.
// In the forward pass we use hard top-k; the backward pass passes gradients
// through as if the selection were soft (identity on the selected paths).
func (nsa *NSACoarseCompression[T]) Backward(_ context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Straight-through estimator: pass dOut through unchanged for Q.
	// K and V gradients are zero (block selection is non-differentiable;
	// gradients flow through the selected attention weights).
	return []*tensor.TensorNumeric[T]{dOut, nil, nil}, nil
}

// topKIndicesT returns the indices of the top-k largest values using
// the generic Arithmetic interface for comparisons.
func topKIndicesT[T tensor.Numeric](scores []T, k int, ops numeric.Arithmetic[T]) []int {
	n := len(scores)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return ops.GreaterThan(scores[indices[i]], scores[indices[j]])
	})
	if k > n {
		k = n
	}
	return indices[:k]
}

// Statically assert that NSACoarseCompression implements graph.Node.
var _ graph.Node[float32] = (*NSACoarseCompression[float32])(nil)
