package residual

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BlockAttnRes implements Block Attention Residuals (arXiv:2603.15031).
// Partitions L layers into N blocks of S layers each.
// Intra-block: standard residual accumulation (sum of layer outputs).
// Inter-block: softmax attention over N block-level representations.
//
// Forward implements Fig 2 from the paper:
//  1. Stack block representations + partial block into value matrix V
//  2. Apply RMSNorm to get keys K
//  3. Compute logits = query^T * K (dot product)
//  4. alpha = softmax(logits)
//  5. h = sum(alpha_i * V_i)
type BlockAttnRes[T tensor.Numeric] struct {
	engine    compute.Engine[T]
	ops       numeric.Arithmetic[T]
	blockSize int
	norm      *normalization.RMSNorm[T]
}

// NewBlockAttnRes creates a new BlockAttnRes layer.
//
// Parameters:
//   - engine: the compute engine for all arithmetic
//   - ops: arithmetic operations for type T
//   - blockSize: number of layers per block (S)
//   - modelDim: hidden dimension size (for RMSNorm initialization)
//   - epsilon: small constant for RMSNorm numerical stability
func NewBlockAttnRes[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], blockSize, modelDim int, epsilon T) (*BlockAttnRes[T], error) {
	if blockSize <= 0 {
		return nil, fmt.Errorf("BlockAttnRes: blockSize must be positive, got %d", blockSize)
	}

	norm, err := normalization.NewRMSNorm[T]("block_attn_res_norm", engine, ops, modelDim, normalization.WithRMSNormEpsilon[T](epsilon))
	if err != nil {
		return nil, fmt.Errorf("BlockAttnRes: create RMSNorm: %w", err)
	}

	return &BlockAttnRes[T]{
		engine:    engine,
		ops:       ops,
		blockSize: blockSize,
		norm:      norm,
	}, nil
}

// BlockSize returns the number of layers per block.
func (b *BlockAttnRes[T]) BlockSize() int {
	return b.blockSize
}

// Parameters returns the learnable parameters.
// The RMSNorm gain is initialized to unit (ones) and not trained by BlockAttnRes,
// but is exposed for completeness.
func (b *BlockAttnRes[T]) Parameters() []*graph.Parameter[T] {
	return b.norm.Parameters()
}

// Forward computes the block attention residual.
//
// Parameters:
//   - ctx: context for cancellation
//   - query: current layer hidden state [dim] or [1, dim]
//   - blocks: completed block representations, each [dim] or [1, dim]
//   - partialBlock: sum of layer outputs in the current (incomplete) block, [dim] or [1, dim]
//
// Returns the weighted combination of all block representations via softmax attention.
func (b *BlockAttnRes[T]) Forward(
	ctx context.Context,
	query *tensor.TensorNumeric[T],
	blocks []*tensor.TensorNumeric[T],
	partialBlock *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if query == nil {
		return nil, fmt.Errorf("BlockAttnRes: query must not be nil")
	}
	if partialBlock == nil {
		return nil, fmt.Errorf("BlockAttnRes: partialBlock must not be nil")
	}

	// Build the list of all block representations (completed + partial).
	allBlocks := make([]*tensor.TensorNumeric[T], 0, len(blocks)+1)
	allBlocks = append(allBlocks, blocks...)
	allBlocks = append(allBlocks, partialBlock)

	n := len(allBlocks)

	// Ensure all blocks are 2D [1, dim] for stacking.
	for i, blk := range allBlocks {
		if len(blk.Shape()) == 1 {
			reshaped, err := b.engine.Reshape(ctx, blk, []int{1, blk.Shape()[0]})
			if err != nil {
				return nil, fmt.Errorf("BlockAttnRes: reshape block %d: %w", i, err)
			}
			allBlocks[i] = reshaped
		}
	}

	// Step (a): Stack blocks into V matrix [n, dim].
	v, err := b.engine.Concat(ctx, allBlocks, 0)
	if err != nil {
		return nil, fmt.Errorf("BlockAttnRes: concat blocks: %w", err)
	}

	// Step (b): Apply RMSNorm to get keys K [n, dim].
	k, err := b.norm.Forward(ctx, v)
	if err != nil {
		return nil, fmt.Errorf("BlockAttnRes: norm keys: %w", err)
	}

	// Ensure query is 2D [1, dim].
	q := query
	if len(q.Shape()) == 1 {
		q, err = b.engine.Reshape(ctx, q, []int{1, q.Shape()[0]})
		if err != nil {
			return nil, fmt.Errorf("BlockAttnRes: reshape query: %w", err)
		}
	}

	// Step (c): Compute logits = query * K^T -> [1, n].
	kT, err := b.engine.Transpose(ctx, k, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("BlockAttnRes: transpose keys: %w", err)
	}

	logits, err := b.engine.MatMul(ctx, q, kT)
	if err != nil {
		return nil, fmt.Errorf("BlockAttnRes: matmul logits: %w", err)
	}

	// Step (d): alpha = softmax(logits) along last dim -> [1, n].
	alpha, err := b.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return nil, fmt.Errorf("BlockAttnRes: softmax: %w", err)
	}

	// Step (e): h = alpha * V -> [1, dim].
	// alpha is [1, n], V is [n, dim] -> matmul gives [1, dim].
	h, err := b.engine.MatMul(ctx, alpha, v)
	if err != nil {
		return nil, fmt.Errorf("BlockAttnRes: matmul output: %w", err)
	}

	// Reshape back to match query shape.
	if len(query.Shape()) == 1 {
		dim := query.Shape()[0]
		h, err = b.engine.Reshape(ctx, h, []int{dim})
		if err != nil {
			return nil, fmt.Errorf("BlockAttnRes: reshape output: %w", err)
		}
	}

	_ = n // used implicitly through concat/matmul shapes
	return h, nil
}

// AttentionWeights computes and returns the softmax attention weights over blocks.
// This is useful for inspection/debugging. Returns weights [1, n] where n = len(blocks) + 1.
func (b *BlockAttnRes[T]) AttentionWeights(
	ctx context.Context,
	query *tensor.TensorNumeric[T],
	blocks []*tensor.TensorNumeric[T],
	partialBlock *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if query == nil || partialBlock == nil {
		return nil, fmt.Errorf("BlockAttnRes: query and partialBlock must not be nil")
	}

	allBlocks := make([]*tensor.TensorNumeric[T], 0, len(blocks)+1)
	allBlocks = append(allBlocks, blocks...)
	allBlocks = append(allBlocks, partialBlock)

	for i, blk := range allBlocks {
		if len(blk.Shape()) == 1 {
			reshaped, err := b.engine.Reshape(ctx, blk, []int{1, blk.Shape()[0]})
			if err != nil {
				return nil, err
			}
			allBlocks[i] = reshaped
		}
	}

	v, err := b.engine.Concat(ctx, allBlocks, 0)
	if err != nil {
		return nil, err
	}

	k, err := b.norm.Forward(ctx, v)
	if err != nil {
		return nil, err
	}

	q := query
	if len(q.Shape()) == 1 {
		q, err = b.engine.Reshape(ctx, q, []int{1, q.Shape()[0]})
		if err != nil {
			return nil, err
		}
	}

	kT, err := b.engine.Transpose(ctx, k, []int{1, 0})
	if err != nil {
		return nil, err
	}

	logits, err := b.engine.MatMul(ctx, q, kT)
	if err != nil {
		return nil, err
	}

	return b.engine.Softmax(ctx, logits, -1)
}
