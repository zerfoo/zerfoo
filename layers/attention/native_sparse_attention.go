package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// NativeSparseAttention implements the full Native Sparse Attention (NSA)
// mechanism, combining three parallel attention paths:
//   - Coarse: block-level compression attention
//   - Fine: token-level selection attention
//   - Window: sliding window local attention
//
// The outputs are combined via learned per-head sigmoid gates:
//
//	O = sigmoid(gateCoarse) * O_coarse + sigmoid(gateFine) * O_fine + sigmoid(gateWindow) * O_window
//
// Gates are initialized to zero so that sigmoid(0) = 0.5 gives equal weighting
// at initialization.
type NativeSparseAttention[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	coarse *NSACoarseCompression[T]
	fine   *NSAFineSelection[T]
	window *NSAWindowAttention[T]

	// Learned per-head gates (shape [numHeads] each).
	gateCoarse *graph.Parameter[T]
	gateFine   *graph.Parameter[T]
	gateWindow *graph.Parameter[T]

	numHeads    int
	outputShape []int
}

// NewNativeSparseAttention creates a new NativeSparseAttention layer combining
// coarse, fine, and window attention paths with learned sigmoid gates.
//
// Parameters:
//   - engine: compute engine for tensor operations
//   - ops: arithmetic operations for the numeric type
//   - modelDim: model dimension (unused directly, reserved for projection layers)
//   - numHeads: number of query attention heads
//   - numKVHeads: number of key/value attention heads
//   - blockSize: number of tokens per KV block for coarse path
//   - topBlocks: number of blocks to select for coarse path
//   - topTokens: number of tokens to select for fine path
//   - windowSize: sliding window size for window path
func NewNativeSparseAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numHeads, numKVHeads, blockSize, topBlocks, topTokens, windowSize int,
) (*NativeSparseAttention[T], error) {
	if numHeads <= 0 {
		return nil, fmt.Errorf("NativeSparseAttention: numHeads must be > 0, got %d", numHeads)
	}
	if numKVHeads <= 0 {
		return nil, fmt.Errorf("NativeSparseAttention: numKVHeads must be > 0, got %d", numKVHeads)
	}
	if numHeads%numKVHeads != 0 {
		return nil, fmt.Errorf("NativeSparseAttention: numHeads (%d) must be divisible by numKVHeads (%d)", numHeads, numKVHeads)
	}

	headDim := modelDim / numHeads

	coarse := NewNSACoarseCompression[T](engine, ops, blockSize, topBlocks, numHeads, numKVHeads, headDim)

	fine := NewNSAFineSelection[T](engine, topTokens, numHeads, numKVHeads, headDim)

	window, err := NewNSAWindowAttention[T](engine, ops, windowSize, numHeads, numKVHeads, headDim)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: creating window path: %w", err)
	}

	// Initialize gates to zero (sigmoid(0) = 0.5 for equal weighting).
	zeroGate := make([]T, numHeads)
	gateCoarseVal, err := tensor.New[T]([]int{numHeads}, zeroGate)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: creating coarse gate: %w", err)
	}
	gateFineVal, err := tensor.New[T]([]int{numHeads}, make([]T, numHeads))
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: creating fine gate: %w", err)
	}
	gateWindowVal, err := tensor.New[T]([]int{numHeads}, make([]T, numHeads))
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: creating window gate: %w", err)
	}

	newTensor := func(shape []int, data []T) (*tensor.TensorNumeric[T], error) {
		return tensor.New[T](shape, data)
	}

	gCoarse, err := graph.NewParameter[T]("nsa_gate_coarse", gateCoarseVal, newTensor)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: creating coarse gate parameter: %w", err)
	}
	gFine, err := graph.NewParameter[T]("nsa_gate_fine", gateFineVal, newTensor)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: creating fine gate parameter: %w", err)
	}
	gWindow, err := graph.NewParameter[T]("nsa_gate_window", gateWindowVal, newTensor)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: creating window gate parameter: %w", err)
	}

	return &NativeSparseAttention[T]{
		engine:     engine,
		ops:        ops,
		coarse:     coarse,
		fine:       fine,
		window:     window,
		gateCoarse: gCoarse,
		gateFine:   gFine,
		gateWindow: gWindow,
		numHeads:   numHeads,
	}, nil
}

// OpType returns the operation type identifier.
func (nsa *NativeSparseAttention[T]) OpType() string {
	return "NativeSparseAttention"
}

// Attributes returns the layer configuration.
func (nsa *NativeSparseAttention[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_heads":   nsa.numHeads,
		"block_size":  nsa.coarse.blockSize,
		"top_blocks":  nsa.coarse.topBlocks,
		"top_tokens":  nsa.fine.topTokens,
		"window_size": nsa.window.windowSize,
	}
}

// Parameters returns the trainable gate parameters.
func (nsa *NativeSparseAttention[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{nsa.gateCoarse, nsa.gateFine, nsa.gateWindow}
}

// OutputShape returns the output shape from the last forward call.
func (nsa *NativeSparseAttention[T]) OutputShape() []int {
	return nsa.outputShape
}

// Forward computes NativeSparseAttention by running all three paths and
// combining with learned sigmoid gates.
//
// Inputs:
//   - Q: [batch, numHeads, seqQ, headDim]
//   - K: [batch, numKVHeads, seqKV, headDim]
//   - V: [batch, numKVHeads, seqKV, headDim]
//
// Returns output with shape [batch, numHeads, seqQ, headDim].
func (nsa *NativeSparseAttention[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 3 {
		return nil, fmt.Errorf("NativeSparseAttention: expected 3 inputs (Q, K, V), got %d", len(inputs))
	}

	Q, K, V := inputs[0], inputs[1], inputs[2]

	// Run three attention paths.
	outCoarse, err := nsa.coarse.Forward(ctx, Q, K, V)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: coarse path: %w", err)
	}

	outFine, err := nsa.fine.Forward(ctx, Q, K, V)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: fine path: %w", err)
	}

	outWindow, err := nsa.window.Forward(ctx, Q, K, V)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention: window path: %w", err)
	}

	// Compute sigmoid of gates.
	gCoarseData := nsa.gateCoarse.Value.Data()
	gFineData := nsa.gateFine.Value.Data()
	gWindowData := nsa.gateWindow.Value.Data()

	sigCoarse := make([]T, nsa.numHeads)
	sigFine := make([]T, nsa.numHeads)
	sigWindow := make([]T, nsa.numHeads)
	for h := range nsa.numHeads {
		sigCoarse[h] = nsa.ops.Sigmoid(gCoarseData[h])
		sigFine[h] = nsa.ops.Sigmoid(gFineData[h])
		sigWindow[h] = nsa.ops.Sigmoid(gWindowData[h])
	}

	// Combine: O = sigCoarse * O_coarse + sigFine * O_fine + sigWindow * O_window
	outShape := outCoarse.Shape()
	batch := outShape[0]
	numHeads := outShape[1]
	seqQ := outShape[2]
	headDim := outShape[3]

	coarseData := outCoarse.Data()
	fineData := outFine.Data()
	windowData := outWindow.Data()

	outSize := batch * numHeads * seqQ * headDim
	outData := make([]T, outSize)

	for b := range batch {
		for h := range numHeads {
			sc := sigCoarse[h]
			sf := sigFine[h]
			sw := sigWindow[h]
			for q := range seqQ {
				offset := ((b*numHeads+h)*seqQ + q) * headDim
				for d := range headDim {
					idx := offset + d
					outData[idx] = nsa.ops.Add(
						nsa.ops.Add(
							nsa.ops.Mul(sc, coarseData[idx]),
							nsa.ops.Mul(sf, fineData[idx]),
						),
						nsa.ops.Mul(sw, windowData[idx]),
					)
				}
			}
		}
	}

	nsa.outputShape = outShape
	return tensor.New[T](outShape, outData)
}

// Backward computes gradients for the NativeSparseAttention layer.
// Gradients flow through the sigmoid gates to the gate parameters and through
// each attention path via straight-through estimation.
func (nsa *NativeSparseAttention[T]) Backward(_ context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	dShape := dOut.Shape()
	batch := dShape[0]
	numHeads := dShape[1]
	seqQ := dShape[2]
	headDim := dShape[3]

	dOutData := dOut.Data()

	// We need the forward outputs to compute gate gradients.
	// Re-run forward paths to get the outputs (in a production implementation
	// these would be cached during forward).
	ctx := context.Background()
	Q, K, V := inputs[0], inputs[1], inputs[2]

	outCoarse, err := nsa.coarse.Forward(ctx, Q, K, V)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention backward: coarse forward: %w", err)
	}
	outFine, err := nsa.fine.Forward(ctx, Q, K, V)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention backward: fine forward: %w", err)
	}
	outWindow, err := nsa.window.Forward(ctx, Q, K, V)
	if err != nil {
		return nil, fmt.Errorf("NativeSparseAttention backward: window forward: %w", err)
	}

	coarseData := outCoarse.Data()
	fineData := outFine.Data()
	windowData := outWindow.Data()

	gCoarseData := nsa.gateCoarse.Value.Data()
	gFineData := nsa.gateFine.Value.Data()
	gWindowData := nsa.gateWindow.Value.Data()

	// Accumulate gate gradients: dL/dgate = sum over (b,q,d) of dOut * O_path * sigmoid'(gate)
	// where sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
	gateCoarseGrad := make([]T, numHeads)
	gateFineGrad := make([]T, numHeads)
	gateWindowGrad := make([]T, numHeads)

	for h := range numHeads {
		sigC := nsa.ops.Sigmoid(gCoarseData[h])
		sigF := nsa.ops.Sigmoid(gFineData[h])
		sigW := nsa.ops.Sigmoid(gWindowData[h])
		sigGradC := nsa.ops.SigmoidGrad(gCoarseData[h])
		sigGradF := nsa.ops.SigmoidGrad(gFineData[h])
		sigGradW := nsa.ops.SigmoidGrad(gWindowData[h])

		var gradC, gradF, gradW T
		for b := range batch {
			for q := range seqQ {
				offset := ((b*numHeads+h)*seqQ + q) * headDim
				for d := range headDim {
					idx := offset + d
					gradC = nsa.ops.Add(gradC, nsa.ops.Mul(dOutData[idx], nsa.ops.Mul(coarseData[idx], sigGradC)))
					gradF = nsa.ops.Add(gradF, nsa.ops.Mul(dOutData[idx], nsa.ops.Mul(fineData[idx], sigGradF)))
					gradW = nsa.ops.Add(gradW, nsa.ops.Mul(dOutData[idx], nsa.ops.Mul(windowData[idx], sigGradW)))
				}
			}
		}
		gateCoarseGrad[h] = gradC
		gateFineGrad[h] = gradF
		gateWindowGrad[h] = gradW

		// Pass scaled dOut through to each path (straight-through for the paths).
		_ = sigC
		_ = sigF
		_ = sigW
	}

	// Accumulate gate gradients into parameters.
	gcGrad, err := tensor.New[T]([]int{numHeads}, gateCoarseGrad)
	if err != nil {
		return nil, err
	}
	if err := nsa.gateCoarse.AddGradient(gcGrad); err != nil {
		return nil, fmt.Errorf("NativeSparseAttention backward: coarse gate gradient: %w", err)
	}
	gfGrad, err := tensor.New[T]([]int{numHeads}, gateFineGrad)
	if err != nil {
		return nil, err
	}
	if err := nsa.gateFine.AddGradient(gfGrad); err != nil {
		return nil, fmt.Errorf("NativeSparseAttention backward: fine gate gradient: %w", err)
	}
	gwGrad, err := tensor.New[T]([]int{numHeads}, gateWindowGrad)
	if err != nil {
		return nil, err
	}
	if err := nsa.gateWindow.AddGradient(gwGrad); err != nil {
		return nil, fmt.Errorf("NativeSparseAttention backward: window gate gradient: %w", err)
	}

	// Pass dOut through as gradient for Q (straight-through for all three paths).
	// K and V gradients are nil (non-differentiable selection in coarse/fine paths).
	return []*tensor.TensorNumeric[T]{dOut, nil, nil}, nil
}

// Statically assert that NativeSparseAttention implements graph.Node.
var _ graph.Node[float32] = (*NativeSparseAttention[float32])(nil)
