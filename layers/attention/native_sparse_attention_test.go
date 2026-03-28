package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestNativeSparseAttention_Forward_OutputShape(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, numHeads, numKVHeads, headDim := 1, 4, 2, 8
	seqQ, seqKV := 4, 8
	modelDim := numHeads * headDim

	nsa, err := NewNativeSparseAttention[float32](
		engine, ops, modelDim, numHeads, numKVHeads,
		2,  // blockSize
		2,  // topBlocks
		4,  // topTokens
		4,  // windowSize
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention: %v", err)
	}

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	out, err := nsa.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	wantShape := []int{batch, numHeads, seqQ, headDim}
	gotShape := out.Shape()
	if len(gotShape) != len(wantShape) {
		t.Fatalf("output rank: got %d, want %d", len(gotShape), len(wantShape))
	}
	for i := range wantShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
		}
	}
}

func TestNativeSparseAttention_GatesInitializedToZero(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	numHeads := 4
	modelDim := numHeads * 8

	nsa, err := NewNativeSparseAttention[float32](
		engine, ops, modelDim, numHeads, 2,
		2, 2, 4, 4,
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention: %v", err)
	}

	params := nsa.Parameters()
	if len(params) != 3 {
		t.Fatalf("expected 3 parameters, got %d", len(params))
	}

	for _, p := range params {
		data := p.Value.Data()
		for i, v := range data {
			if v != 0 {
				t.Errorf("parameter %s[%d]: got %v, want 0", p.Name, i, v)
			}
		}
	}
}

func TestNativeSparseAttention_EqualWeightingAtInit(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, numHeads, numKVHeads, headDim := 1, 2, 2, 4
	seqQ, seqKV := 2, 8
	modelDim := numHeads * headDim

	nsa, err := NewNativeSparseAttention[float32](
		engine, ops, modelDim, numHeads, numKVHeads,
		2, 2, 4, 4,
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention: %v", err)
	}

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	out, err := nsa.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// With gates at zero, sigmoid(0) = 0.5, so each path contributes equally.
	// Run the three paths individually and verify the combination.
	coarse := NewNSACoarseCompression[float32](engine, ops, 2, 2, numHeads, numKVHeads, headDim)
	fine := NewNSAFineSelection[float32](engine, 4, numHeads, numKVHeads, headDim)
	window, err := NewNSAWindowAttention[float32](engine, ops, 4, numHeads, numKVHeads, headDim)
	if err != nil {
		t.Fatalf("NewNSAWindowAttention: %v", err)
	}

	outC, err := coarse.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("coarse Forward: %v", err)
	}
	outF, err := fine.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("fine Forward: %v", err)
	}
	outW, err := window.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("window Forward: %v", err)
	}

	outData := out.Data()
	cData := outC.Data()
	fData := outF.Data()
	wData := outW.Data()

	for i := range outData {
		want := float32(0.5)*(cData[i]) + float32(0.5)*(fData[i]) + float32(0.5)*(wData[i])
		got := outData[i]
		if diff := math.Abs(float64(got - want)); diff > 1e-5 {
			t.Errorf("output[%d]: got %v, want %v (diff %v)", i, got, want, diff)
		}
	}
}

func TestNativeSparseAttention_Backward_GradientsFlow(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, numHeads, numKVHeads, headDim := 1, 2, 2, 4
	seqQ, seqKV := 2, 8
	modelDim := numHeads * headDim

	nsa, err := NewNativeSparseAttention[float32](
		engine, ops, modelDim, numHeads, numKVHeads,
		2, 2, 4, 4,
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention: %v", err)
	}

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	// Forward to populate state.
	_, err = nsa.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Create dOut with all ones.
	dOutSize := batch * numHeads * seqQ * headDim
	dOutData := make([]float32, dOutSize)
	for i := range dOutData {
		dOutData[i] = 1.0
	}
	dOut, err := tensor.New[float32]([]int{batch, numHeads, seqQ, headDim}, dOutData)
	if err != nil {
		t.Fatalf("create dOut: %v", err)
	}

	// Clear gradients before backward.
	for _, p := range nsa.Parameters() {
		p.ClearGradient()
	}

	grads, err := nsa.Backward(ctx, types.FullBackprop, dOut, Q, K, V)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	// Should return 3 gradients (dQ, dK, dV).
	if len(grads) != 3 {
		t.Fatalf("expected 3 gradients, got %d", len(grads))
	}

	// dQ should not be nil (straight-through).
	if grads[0] == nil {
		t.Fatal("dQ gradient is nil")
	}

	// Gate gradients should be non-zero (dOut is all ones, outputs are non-zero).
	for _, p := range nsa.Parameters() {
		gradData := p.Gradient.Data()
		allZero := true
		for _, v := range gradData {
			if v != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("parameter %s has all-zero gradient after backward", p.Name)
		}
	}
}

func TestNativeSparseAttention_Metadata(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	nsa, err := NewNativeSparseAttention[float32](
		engine, ops, 32, 4, 2, 2, 2, 4, 4,
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention: %v", err)
	}

	if nsa.OpType() != "NativeSparseAttention" {
		t.Errorf("OpType: got %q, want %q", nsa.OpType(), "NativeSparseAttention")
	}

	attrs := nsa.Attributes()
	if attrs["num_heads"] != 4 {
		t.Errorf("num_heads: got %v, want 4", attrs["num_heads"])
	}
	if attrs["block_size"] != 2 {
		t.Errorf("block_size: got %v, want 2", attrs["block_size"])
	}
	if attrs["top_blocks"] != 2 {
		t.Errorf("top_blocks: got %v, want 2", attrs["top_blocks"])
	}
	if attrs["top_tokens"] != 4 {
		t.Errorf("top_tokens: got %v, want 4", attrs["top_tokens"])
	}
	if attrs["window_size"] != 4 {
		t.Errorf("window_size: got %v, want 4", attrs["window_size"])
	}
}

func TestNativeSparseAttention_InvalidArgs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := NewNativeSparseAttention[float32](engine, ops, 32, 0, 2, 2, 2, 4, 4)
	if err == nil {
		t.Error("expected error for numHeads=0")
	}

	_, err = NewNativeSparseAttention[float32](engine, ops, 32, 4, 0, 2, 2, 4, 4)
	if err == nil {
		t.Error("expected error for numKVHeads=0")
	}

	_, err = NewNativeSparseAttention[float32](engine, ops, 32, 4, 3, 2, 2, 4, 4)
	if err == nil {
		t.Error("expected error for numHeads not divisible by numKVHeads")
	}

	_, err = NewNativeSparseAttention[float32](engine, ops, 32, 4, 2, 2, 2, 4, 0)
	if err == nil {
		t.Error("expected error for windowSize=0")
	}
}

func TestNativeSparseAttention_ForwardTooFewInputs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	nsa, err := NewNativeSparseAttention[float32](engine, ops, 32, 4, 2, 2, 2, 4, 4)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention: %v", err)
	}

	Q, _ := tensor.New[float32]([]int{1, 4, 2, 8}, make([]float32, 64))
	_, err = nsa.Forward(context.Background(), Q)
	if err == nil {
		t.Error("expected error for too few inputs")
	}
}

func TestNativeSparseAttention_DegenerateFullSelection_ApproximatesDense(t *testing.T) {
	// When topBlocks covers all blocks, topTokens covers all tokens, and the
	// window covers the entire KV sequence, each path should approximate dense
	// causal attention. The gated combination (sigmoid(0)=0.5 for each) should
	// still produce a non-zero output that is consistent across paths.
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, numHeads, numKVHeads, headDim := 1, 2, 2, 4
	seqQ, seqKV := 4, 4
	modelDim := numHeads * headDim
	blockSize := 1

	nsa, err := NewNativeSparseAttention[float32](
		engine, ops, modelDim, numHeads, numKVHeads,
		blockSize,
		seqKV,  // topBlocks = all blocks (seqKV / blockSize = seqKV)
		seqKV,  // topTokens = all tokens
		seqKV,  // windowSize = full sequence
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention: %v", err)
	}

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	out, err := nsa.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// When all tokens are selected, each path sees the entire KV.
	// Run individual paths to verify they produce valid outputs.
	coarse := NewNSACoarseCompression[float32](engine, ops, blockSize, seqKV, numHeads, numKVHeads, headDim)
	fine := NewNSAFineSelection[float32](engine, seqKV, numHeads, numKVHeads, headDim)
	window, err := NewNSAWindowAttention[float32](engine, ops, seqKV, numHeads, numKVHeads, headDim)
	if err != nil {
		t.Fatalf("NewNSAWindowAttention: %v", err)
	}

	outC, err := coarse.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("coarse Forward: %v", err)
	}
	outF, err := fine.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("fine Forward: %v", err)
	}
	outW, err := window.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("window Forward: %v", err)
	}

	// The fine path with topTokens=seqKV selects all tokens, which is standard
	// scaled dot-product attention. The coarse path with blockSize=1 and
	// topBlocks=seqKV also selects all tokens. Both should approximate the
	// window path (which is standard causal attention with full window).
	// Check that fine and coarse outputs are close to each other.
	cData := outC.Data()
	fData := outF.Data()
	wData := outW.Data()

	const tol = 1e-4
	for i := range fData {
		// Fine (non-causal top-k with all tokens) and coarse (block top-k with
		// all blocks at blockSize=1) should match closely.
		if diff := math.Abs(float64(fData[i] - cData[i])); diff > tol {
			t.Errorf("fine vs coarse[%d]: fine=%f, coarse=%f, diff=%e", i, fData[i], cData[i], diff)
		}
	}

	// Combined output should equal 0.5*coarse + 0.5*fine + 0.5*window.
	outData := out.Data()
	for i := range outData {
		want := float32(0.5)*cData[i] + float32(0.5)*fData[i] + float32(0.5)*wData[i]
		if diff := math.Abs(float64(outData[i] - want)); diff > 1e-5 {
			t.Errorf("combined[%d]: got %f, want %f, diff=%e", i, outData[i], want, diff)
		}
	}

	// All paths should produce non-zero output.
	for _, d := range [][]float32{cData, fData, wData} {
		allZero := true
		for _, v := range d {
			if v != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Error("one of the paths produced all-zero output in degenerate full-selection case")
		}
	}
}

func TestNativeSparseAttention_DifferentInputsProduceDifferentOutputs(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, numHeads, numKVHeads, headDim := 1, 2, 2, 4
	seqQ, seqKV := 2, 8
	modelDim := numHeads * headDim

	nsa, err := NewNativeSparseAttention[float32](
		engine, ops, modelDim, numHeads, numKVHeads,
		2, 2, 4, 4,
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention: %v", err)
	}

	// First set of inputs.
	Q1, K1, V1 := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)
	out1, err := nsa.Forward(ctx, Q1, K1, V1)
	if err != nil {
		t.Fatalf("Forward (input 1): %v", err)
	}

	// Second set of inputs with different data.
	qSize := batch * numHeads * seqQ * headDim
	kSize := batch * numKVHeads * seqKV * headDim
	qData2 := make([]float32, qSize)
	kData2 := make([]float32, kSize)
	vData2 := make([]float32, kSize)
	for i := range qData2 {
		qData2[i] = float32(i%3)*0.5 + 1.0
	}
	for i := range kData2 {
		kData2[i] = float32(i%7)*0.3 - 0.5
	}
	for i := range vData2 {
		vData2[i] = float32(i%13) * 0.2
	}
	Q2, err := tensor.New[float32]([]int{batch, numHeads, seqQ, headDim}, qData2)
	if err != nil {
		t.Fatalf("create Q2: %v", err)
	}
	K2, err := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, kData2)
	if err != nil {
		t.Fatalf("create K2: %v", err)
	}
	V2, err := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, vData2)
	if err != nil {
		t.Fatalf("create V2: %v", err)
	}

	out2, err := nsa.Forward(ctx, Q2, K2, V2)
	if err != nil {
		t.Fatalf("Forward (input 2): %v", err)
	}

	// Outputs should differ.
	d1 := out1.Data()
	d2 := out2.Data()
	if len(d1) != len(d2) {
		t.Fatalf("output sizes differ: %d vs %d", len(d1), len(d2))
	}

	allSame := true
	for i := range d1 {
		if d1[i] != d2[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("different inputs produced identical outputs")
	}
}

func TestNativeSparseAttention_GateWeightsAffectOutput(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	batch, numHeads, numKVHeads, headDim := 1, 2, 2, 4
	seqQ, seqKV := 2, 8
	modelDim := numHeads * headDim

	// Create two NSA instances with different gate values.
	nsa1, err := NewNativeSparseAttention[float32](
		engine, ops, modelDim, numHeads, numKVHeads,
		2, 2, 4, 4,
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention (1): %v", err)
	}

	nsa2, err := NewNativeSparseAttention[float32](
		engine, ops, modelDim, numHeads, numKVHeads,
		2, 2, 4, 4,
	)
	if err != nil {
		t.Fatalf("NewNativeSparseAttention (2): %v", err)
	}

	// Set nsa2's coarse gate to a large positive value (sigmoid -> ~1),
	// and fine/window gates to large negative values (sigmoid -> ~0).
	// This should make nsa2's output dominated by the coarse path.
	for i := range numHeads {
		nsa2.gateCoarse.Value.Data()[i] = 10.0 // sigmoid(10) ~ 1.0
		nsa2.gateFine.Value.Data()[i] = -10.0   // sigmoid(-10) ~ 0.0
		nsa2.gateWindow.Value.Data()[i] = -10.0  // sigmoid(-10) ~ 0.0
	}

	Q, K, V := makeQKV(t, batch, numHeads, numKVHeads, seqQ, seqKV, headDim)

	out1, err := nsa1.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward (nsa1): %v", err)
	}

	out2, err := nsa2.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("Forward (nsa2): %v", err)
	}

	// Outputs should differ because the gate weights are different.
	d1 := out1.Data()
	d2 := out2.Data()

	allSame := true
	for i := range d1 {
		if d1[i] != d2[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("different gate weights produced identical outputs")
	}

	// Verify nsa2 output is close to just the coarse path output
	// (since fine and window gates are suppressed).
	coarse := NewNSACoarseCompression[float32](engine, ops, 2, 2, numHeads, numKVHeads, headDim)
	outC, err := coarse.Forward(ctx, Q, K, V)
	if err != nil {
		t.Fatalf("coarse Forward: %v", err)
	}
	cData := outC.Data()

	// nsa2 output ~ sigmoid(10)*coarse + sigmoid(-10)*fine + sigmoid(-10)*window
	// ~ 1.0*coarse + 0.0*fine + 0.0*window = coarse
	const tol = 1e-3
	for i := range d2 {
		if diff := math.Abs(float64(d2[i] - cData[i])); diff > tol {
			t.Errorf("gate-weighted output[%d]: got %f, want ~%f (coarse only), diff=%e", i, d2[i], cData[i], diff)
		}
	}
}

// makeQKV creates deterministic Q, K, V tensors for testing.
func makeQKV(t *testing.T, batch, numHeads, numKVHeads, seqQ, seqKV, headDim int) (
	*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32],
) {
	t.Helper()

	qSize := batch * numHeads * seqQ * headDim
	kSize := batch * numKVHeads * seqKV * headDim

	qData := make([]float32, qSize)
	kData := make([]float32, kSize)
	vData := make([]float32, kSize)
	for i := range qData {
		qData[i] = float32(i%7) * 0.1
	}
	for i := range kData {
		kData[i] = float32(i%5) * 0.1
	}
	for i := range vData {
		vData[i] = float32(i%11) * 0.1
	}

	Q, err := tensor.New[float32]([]int{batch, numHeads, seqQ, headDim}, qData)
	if err != nil {
		t.Fatalf("create Q: %v", err)
	}
	K, err := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, kData)
	if err != nil {
		t.Fatalf("create K: %v", err)
	}
	V, err := tensor.New[float32]([]int{batch, numKVHeads, seqKV, headDim}, vData)
	if err != nil {
		t.Fatalf("create V: %v", err)
	}

	return Q, K, V
}
