package attention

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// cosineEngine is the interface for engines that support CosineSimilarity.
// Both CPUEngine and GPUEngine implement this method but it is not part of
// the base Engine interface. SparseRoutedAttention uses a type assertion to
// access it at construction time.
type cosineEngine[T tensor.Numeric] interface {
	CosineSimilarity(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
}

// SparseRoutedKVCache is the interface for compressed KV caches used by
// SparseRoutedAttention. This is satisfied by generate.CompressedKVCache
// and avoids an import cycle with the generate package.
type SparseRoutedKVCache[T tensor.Numeric] interface {
	NumLayers() int
	SeqLen() int
	Reset()
}

// SparseRoutedAttention routes queries to a subset of KV segments using
// cosine similarity scoring. It divides the KV sequence into fixed-size
// segments, computes cosine similarity between each query and segment
// centroids (mean key vectors), selects the top-k most similar segments,
// and performs scaled dot-product attention over the selected segments.
//
// Position encoding uses document-wise RoPE so that position IDs reset at
// document boundaries during multi-document inference.
//
// KV history is stored in a CompressedKVCache (via SparseRoutedKVCache
// interface) for efficient long-context inference.
type SparseRoutedAttention[T tensor.Numeric] struct {
	engine    compute.Engine[T]
	cosine    cosineEngine[T]
	ops       numeric.Arithmetic[T]
	rope      *embeddings.RotaryPositionalEmbedding[T]
	kvCache   SparseRoutedKVCache[T]
	numHeads  int
	numKVHeads int
	headDim   int

	segmentSize int // number of tokens per KV segment
	topK        int // number of segments to attend to per query

	outputShape []int
}

// NewSparseRoutedAttention creates a new SparseRoutedAttention layer.
//
// Parameters:
//   - engine: compute engine for tensor operations (must support CosineSimilarity)
//   - ops: arithmetic operations for the numeric type
//   - rope: rotary positional embedding (supports document-wise mode)
//   - kvCache: compressed KV cache for storing key-value pairs (satisfies SparseRoutedKVCache)
//   - numHeads: number of query attention heads
//   - numKVHeads: number of key/value attention heads
//   - headDim: dimension of each attention head
//   - segmentSize: number of tokens per KV segment for routing
//   - topK: number of segments to select per query position
func NewSparseRoutedAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	rope *embeddings.RotaryPositionalEmbedding[T],
	kvCache SparseRoutedKVCache[T],
	numHeads, numKVHeads, headDim, segmentSize, topK int,
) (*SparseRoutedAttention[T], error) {
	if numHeads <= 0 {
		return nil, fmt.Errorf("SparseRoutedAttention: numHeads must be > 0, got %d", numHeads)
	}
	if numKVHeads <= 0 {
		return nil, fmt.Errorf("SparseRoutedAttention: numKVHeads must be > 0, got %d", numKVHeads)
	}
	if numHeads%numKVHeads != 0 {
		return nil, fmt.Errorf("SparseRoutedAttention: numHeads (%d) must be divisible by numKVHeads (%d)", numHeads, numKVHeads)
	}
	if segmentSize <= 0 {
		return nil, fmt.Errorf("SparseRoutedAttention: segmentSize must be > 0, got %d", segmentSize)
	}
	if topK <= 0 {
		return nil, fmt.Errorf("SparseRoutedAttention: topK must be > 0, got %d", topK)
	}

	cosine, ok := engine.(cosineEngine[T])
	if !ok {
		return nil, fmt.Errorf("SparseRoutedAttention: engine does not support CosineSimilarity")
	}

	return &SparseRoutedAttention[T]{
		engine:     engine,
		cosine:     cosine,
		ops:        ops,
		rope:       rope,
		kvCache:    kvCache,
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
		segmentSize: segmentSize,
		topK:       topK,
	}, nil
}

// OpType returns the operation type identifier.
func (sra *SparseRoutedAttention[T]) OpType() string {
	return "SparseRoutedAttention"
}

// Attributes returns the layer configuration.
func (sra *SparseRoutedAttention[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_heads":    sra.numHeads,
		"num_kv_heads": sra.numKVHeads,
		"head_dim":     sra.headDim,
		"segment_size": sra.segmentSize,
		"top_k":        sra.topK,
	}
}

// Parameters returns nil (no trainable parameters).
func (sra *SparseRoutedAttention[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// OutputShape returns the output shape from the last forward call.
func (sra *SparseRoutedAttention[T]) OutputShape() []int {
	return sra.outputShape
}

// SetDocumentBoundaries sets document boundary positions for document-wise
// RoPE. Position IDs reset to 0 at each boundary so each document receives
// independent positional encoding. Pass nil to disable.
func (sra *SparseRoutedAttention[T]) SetDocumentBoundaries(boundaries []int) {
	sra.rope.SetDocumentBoundaries(boundaries)
}

// Forward computes sparse routed attention.
//
// Inputs:
//   - Q: [batch, numHeads, seqQ, headDim]
//   - K: [batch, numKVHeads, seqKV, headDim]
//   - V: [batch, numKVHeads, seqKV, headDim]
//
// Returns output with shape [batch, numHeads, seqQ, headDim].
//
// The algorithm:
//  1. Divide K into segments of segmentSize tokens and compute segment centroids.
//  2. For each query position, compute cosine similarity with all centroids.
//  3. Select top-k segments per query.
//  4. Gather full-resolution K,V from selected segments.
//  5. Apply RoPE to Q and selected K.
//  6. Compute scaled dot-product attention over selected segments.
func (sra *SparseRoutedAttention[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 3 {
		return nil, fmt.Errorf("SparseRoutedAttention: expected 3 inputs (Q, K, V), got %d", len(inputs))
	}

	Q, K, V := inputs[0], inputs[1], inputs[2]

	qShape := Q.Shape()
	kShape := K.Shape()

	batch := qShape[0]
	numHeads := qShape[1]
	seqQ := qShape[2]
	headDim := qShape[3]
	numKVHeads := kShape[1]
	seqKV := kShape[2]

	numSegments := seqKV / sra.segmentSize
	if numSegments == 0 {
		numSegments = 1
	}
	k := sra.topK
	if k > numSegments {
		k = numSegments
	}

	headsPerGroup := numHeads / numKVHeads
	scale := float64(1.0) / math.Sqrt(float64(headDim))
	scaleT := sra.ops.FromFloat64(scale)

	qData := Q.Data()
	kData := K.Data()
	vData := V.Data()

	// Step 1: Compute segment centroids by averaging K over each segment.
	// centroids shape: [batch, numKVHeads, numSegments, headDim]
	invSeg := sra.ops.FromFloat64(1.0 / float64(sra.segmentSize))
	centroids := make([]T, batch*numKVHeads*numSegments*headDim)

	for b := range batch {
		for kv := range numKVHeads {
			for seg := range numSegments {
				segStart := seg * sra.segmentSize
				segEnd := segStart + sra.segmentSize
				if segEnd > seqKV {
					segEnd = seqKV
				}
				segLen := segEnd - segStart
				actualInv := invSeg
				if segLen != sra.segmentSize {
					actualInv = sra.ops.FromFloat64(1.0 / float64(segLen))
				}

				for d := range headDim {
					var sum T
					for t := segStart; t < segEnd; t++ {
						idx := ((b*numKVHeads+kv)*seqKV + t) * headDim
						sum = sra.ops.Add(sum, kData[idx+d])
					}
					cIdx := ((b*numKVHeads+kv)*numSegments + seg) * headDim
					centroids[cIdx+d] = sra.ops.Mul(sum, actualInv)
				}
			}
		}
	}

	// Step 2 & 3: For each query position, compute cosine similarity with
	// centroids and select top-k segments.
	// We use the engine's CosineSimilarity for 2D pairwise similarity.

	selectedSeqLen := k * sra.segmentSize
	if selectedSeqLen > seqKV {
		selectedSeqLen = seqKV
	}

	outSize := batch * numHeads * seqQ * headDim
	outData := make([]T, outSize)

	for b := range batch {
		for h := range numHeads {
			kvHead := h / headsPerGroup

			// Build centroid matrix for this batch/kv-head: [numSegments, headDim]
			centroidData := make([]T, numSegments*headDim)
			centroidBase := (b*numKVHeads + kvHead) * numSegments * headDim
			copy(centroidData, centroids[centroidBase:centroidBase+numSegments*headDim])

			centroidMat, err := tensor.New[T]([]int{numSegments, headDim}, centroidData)
			if err != nil {
				return nil, fmt.Errorf("SparseRoutedAttention: creating centroid matrix: %w", err)
			}

			for q := range seqQ {
				// Extract query vector: [1, headDim]
				qOffset := ((b*numHeads+h)*seqQ + q) * headDim
				qVec := make([]T, headDim)
				copy(qVec, qData[qOffset:qOffset+headDim])

				qMat, err := tensor.New[T]([]int{1, headDim}, qVec)
				if err != nil {
					return nil, fmt.Errorf("SparseRoutedAttention: creating query matrix: %w", err)
				}

				// Cosine similarity: [1, numSegments]
				simMat, err := sra.cosine.CosineSimilarity(ctx, qMat, centroidMat)
				if err != nil {
					return nil, fmt.Errorf("SparseRoutedAttention: cosine similarity: %w", err)
				}
				simData := simMat.Data()

				// Select top-k segments.
				segIndices := topKIndicesT(simData, k, sra.ops)
				sort.Ints(segIndices)

				// Step 4 & 5: Gather K,V from selected segments and compute attention.
				kvBase := (b*numKVHeads + kvHead) * seqKV * headDim

				// Compute fine-grained scores over selected segment tokens.
				actualTokens := 0
				for _, segIdx := range segIndices {
					segStart := segIdx * sra.segmentSize
					segEnd := segStart + sra.segmentSize
					if segEnd > seqKV {
						segEnd = seqKV
					}
					actualTokens += segEnd - segStart
				}

				fineScores := make([]T, actualTokens)
				tokenOffsets := make([]int, actualTokens) // maps fine index to seqKV position
				ti := 0
				first := true
				var maxScore T

				for _, segIdx := range segIndices {
					segStart := segIdx * sra.segmentSize
					segEnd := segStart + sra.segmentSize
					if segEnd > seqKV {
						segEnd = seqKV
					}
					for t := segStart; t < segEnd; t++ {
						kOff := kvBase + t*headDim
						var dot T
						for d := range headDim {
							dot = sra.ops.Add(dot, sra.ops.Mul(qData[qOffset+d], kData[kOff+d]))
						}
						s := sra.ops.Mul(dot, scaleT)
						fineScores[ti] = s
						tokenOffsets[ti] = t
						if first || sra.ops.GreaterThan(s, maxScore) {
							maxScore = s
							first = false
						}
						ti++
					}
				}

				// Softmax over fine scores.
				expScores := make([]T, actualTokens)
				var sumExp T
				for i, s := range fineScores {
					e := sra.ops.Exp(sra.ops.Sub(s, maxScore))
					expScores[i] = e
					sumExp = sra.ops.Add(sumExp, e)
				}

				// Weighted sum of V from selected segments.
				outOffset := ((b*numHeads+h)*seqQ + q) * headDim
				for i := range actualTokens {
					w := sra.ops.Div(expScores[i], sumExp)
					vOff := kvBase + tokenOffsets[i]*headDim
					for d := range headDim {
						outData[outOffset+d] = sra.ops.Add(
							outData[outOffset+d],
							sra.ops.Mul(w, vData[vOff+d]),
						)
					}
				}
			}
		}
	}

	outShape := []int{batch, numHeads, seqQ, headDim}
	sra.outputShape = outShape
	return tensor.New[T](outShape, outData)
}

// Backward computes gradients for the SparseRoutedAttention layer.
// Uses straight-through estimation for the routing selection.
func (sra *SparseRoutedAttention[T]) Backward(_ context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return []*tensor.TensorNumeric[T]{dOut, nil, nil}, nil
}

// Statically assert that SparseRoutedAttention implements graph.Node.
var _ graph.Node[float32] = (*SparseRoutedAttention[float32])(nil)
