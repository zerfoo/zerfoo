package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
)

func TestGroupedQueryAttention_Forward_Shape(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 2
	seqLen := 7
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2

	gqa, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		modelDim,
		numQueryHeads,
		numKeyValueHeads,
		WithRopeBase[float32](10000.0),
		WithMaxSeqLen[float32](seqLen),
	)
	if err != nil {
		t.Fatalf("failed to construct GQA: %v", err)
	}

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("failed creating input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%13) / 10.0 // simple deterministic data
	}

	out, err := gqa.Forward(context.Background(), inp)
	if err != nil {
		t.Fatalf("GQA Forward failed: %v", err)
	}

	expected := []int{batchSize, seqLen, modelDim}
	if !testutils.IntSliceEqual(expected, out.Shape()) {
		t.Fatalf("unexpected output shape: got %v want %v", out.Shape(), expected)
	}

	// Sanity: output should be finite numbers (no NaN/Inf) for a few positions
	for _, idx := range []int{0, len(out.Data()) / 2, len(out.Data()) - 1} {
		v := out.Data()[idx]
		if !(v == v) { // NaN check
			t.Fatalf("output contains NaN at idx %d", idx)
		}
	}
}

func TestGroupedQueryAttention_Forward_NoRoPE(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	tests := []struct {
		name             string
		batchSize        int
		seqLen           int
		modelDim         int
		numQueryHeads    int
		numKeyValueHeads int
	}{
		{
			name:             "MHA no RoPE",
			batchSize:        1,
			seqLen:           5,
			modelDim:         16,
			numQueryHeads:    4,
			numKeyValueHeads: 4,
		},
		{
			name:             "GQA no RoPE",
			batchSize:        2,
			seqLen:           7,
			modelDim:         16,
			numQueryHeads:    4,
			numKeyValueHeads: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gqa, err := NewGroupedQueryAttention[float32](
				engine,
				numeric.Float32Ops{},
				tt.modelDim,
				tt.numQueryHeads,
				tt.numKeyValueHeads,
				WithNoRoPE[float32](),
			)
			if err != nil {
				t.Fatalf("failed to construct GQA: %v", err)
			}

			if gqa.rope != nil {
				t.Fatal("expected rope to be nil when WithNoRoPE is set")
			}

			inp, err := tensor.New[float32]([]int{tt.batchSize, tt.seqLen, tt.modelDim}, nil)
			if err != nil {
				t.Fatalf("failed creating input: %v", err)
			}
			for i := range inp.Data() {
				inp.Data()[i] = float32(i%13) / 10.0
			}

			out, err := gqa.Forward(context.Background(), inp)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			expected := []int{tt.batchSize, tt.seqLen, tt.modelDim}
			if !testutils.IntSliceEqual(expected, out.Shape()) {
				t.Fatalf("unexpected output shape: got %v want %v", out.Shape(), expected)
			}

			for i, v := range out.Data() {
				if v != v { // NaN check
					t.Fatalf("output contains NaN at idx %d", i)
				}
			}
		})
	}
}

func TestGroupedQueryAttention_KEqV_Forward(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 1
	seqLen := 5
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2

	gqa, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		modelDim,
		numQueryHeads,
		numKeyValueHeads,
		WithRopeBase[float32](10000.0),
		WithMaxSeqLen[float32](seqLen),
	)
	if err != nil {
		t.Fatalf("failed to construct GQA: %v", err)
	}

	gqa.SetKEqV(true)

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("failed creating input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%13) / 10.0
	}

	out, err := gqa.Forward(context.Background(), inp)
	if err != nil {
		t.Fatalf("KEqV Forward failed: %v", err)
	}

	expected := []int{batchSize, seqLen, modelDim}
	if !testutils.IntSliceEqual(expected, out.Shape()) {
		t.Fatalf("unexpected output shape: got %v want %v", out.Shape(), expected)
	}

	for i, v := range out.Data() {
		if v != v {
			t.Fatalf("output contains NaN at idx %d", i)
		}
	}
}

func TestGroupedQueryAttention_KEqV_MatchesIdenticalWeights(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 1
	seqLen := 5
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2

	// Create the K=V GQA.
	gqaKEqV, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		modelDim,
		numQueryHeads,
		numKeyValueHeads,
		WithRopeBase[float32](10000.0),
		WithMaxSeqLen[float32](seqLen),
	)
	if err != nil {
		t.Fatalf("failed to construct K=V GQA: %v", err)
	}
	gqaKEqV.SetKEqV(true)

	// Create the separate GQA sharing the same Q, K, O weights but with V = K weights.
	// Copy K weight data into V weight so both paths compute the same thing.
	gqaSeparate, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		modelDim,
		numQueryHeads,
		numKeyValueHeads,
		WithRopeBase[float32](10000.0),
		WithMaxSeqLen[float32](seqLen),
	)
	if err != nil {
		t.Fatalf("failed to construct separate GQA: %v", err)
	}

	// Copy all weights from K=V GQA to separate GQA so they share the same
	// random initialization, then set separate's V weight = K weight.
	copyDenseWeights(t, gqaSeparate.wq, gqaKEqV.wq)
	copyDenseWeights(t, gqaSeparate.wk, gqaKEqV.wk)
	copyDenseWeights(t, gqaSeparate.wo, gqaKEqV.wo)
	copyDenseWeights(t, gqaSeparate.wv, gqaKEqV.wk) // V = K

	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("failed creating input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%13) / 10.0
	}

	outKEqV, err := gqaKEqV.Forward(context.Background(), inp)
	if err != nil {
		t.Fatalf("KEqV Forward failed: %v", err)
	}

	outSep, err := gqaSeparate.Forward(context.Background(), inp)
	if err != nil {
		t.Fatalf("Separate Forward failed: %v", err)
	}

	if !testutils.IntSliceEqual(outKEqV.Shape(), outSep.Shape()) {
		t.Fatalf("shape mismatch: K=V %v vs separate %v", outKEqV.Shape(), outSep.Shape())
	}

	const tol = 1e-5
	for i := range outKEqV.Data() {
		diff := outKEqV.Data()[i] - outSep.Data()[i]
		if diff < -tol || diff > tol {
			t.Fatalf("output mismatch at idx %d: K=V=%f separate=%f diff=%f",
				i, outKEqV.Data()[i], outSep.Data()[i], diff)
		}
	}
}

// copyDenseWeights copies parameter data from src Dense to dst Dense.
func copyDenseWeights(t *testing.T, dst, src *core.Dense[float32]) {
	t.Helper()
	srcParams := src.Parameters()
	dstParams := dst.Parameters()
	for i, dp := range dstParams {
		if i < len(srcParams) {
			copy(dp.Value.Data(), srcParams[i].Value.Data())
		}
	}
}

func TestGroupedQueryAttention_NoRoPE_SetDocumentBoundaries(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	gqa, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		16, 4, 4,
		WithNoRoPE[float32](),
	)
	if err != nil {
		t.Fatalf("failed to construct GQA: %v", err)
	}

	// Should not panic when rope is nil.
	gqa.SetDocumentBoundaries([]int{0, 3})
	gqa.SetDocumentBoundaries(nil)
}

// TestGroupedQueryAttention_ExternalKV_MatchesInternal verifies that a GQA
// configured with WithExternalKV() produces outputs numerically identical to
// an internal-KV GQA when fed the same pre-computed K/V tensors. See ADR-087.
func TestGroupedQueryAttention_ExternalKV_MatchesInternal(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	batchSize := 2
	seqLen := 5
	modelDim := 16
	numQueryHeads := 4
	numKeyValueHeads := 2

	// Reference: standard GQA with internal K/V projection.
	gqaInternal, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		modelDim,
		numQueryHeads,
		numKeyValueHeads,
		WithRopeBase[float32](10000.0),
		WithMaxSeqLen[float32](seqLen),
	)
	if err != nil {
		t.Fatalf("failed to construct internal GQA: %v", err)
	}

	// Under-test: GQA with externalKV mode. Must not instantiate wk/wv.
	gqaExternal, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		modelDim,
		numQueryHeads,
		numKeyValueHeads,
		WithRopeBase[float32](10000.0),
		WithMaxSeqLen[float32](seqLen),
		WithExternalKV[float32](),
	)
	if err != nil {
		t.Fatalf("failed to construct external GQA: %v", err)
	}
	if !gqaExternal.ExternalKV() {
		t.Fatalf("ExternalKV() = false, expected true")
	}
	if gqaExternal.wk != nil {
		t.Fatalf("externalKV GQA must not instantiate wk")
	}
	if gqaExternal.wv != nil {
		t.Fatalf("externalKV GQA must not instantiate wv")
	}

	// Share Q and output weights so only the K/V source differs.
	copyDenseWeights(t, gqaExternal.wq, gqaInternal.wq)
	copyDenseWeights(t, gqaExternal.wo, gqaInternal.wo)

	// Deterministic input.
	inp, err := tensor.New[float32]([]int{batchSize, seqLen, modelDim}, nil)
	if err != nil {
		t.Fatalf("failed creating input: %v", err)
	}
	for i := range inp.Data() {
		inp.Data()[i] = float32(i%13) / 10.0
	}

	ctx := context.Background()

	// Run internal GQA and capture the K/V projections it computed.
	outInternal, err := gqaInternal.Forward(ctx, inp)
	if err != nil {
		t.Fatalf("internal GQA Forward failed: %v", err)
	}
	kProj := gqaInternal.kProj
	vProj := gqaInternal.vProj
	if kProj == nil || vProj == nil {
		t.Fatalf("internal GQA did not cache kProj/vProj")
	}

	// Validate K/V shape matches the documented contract.
	expectedKVShape := []int{batchSize, seqLen, numKeyValueHeads * (modelDim / numQueryHeads)}
	if !testutils.IntSliceEqual(kProj.Shape(), expectedKVShape) {
		t.Fatalf("unexpected K shape: got %v want %v", kProj.Shape(), expectedKVShape)
	}

	// Run external GQA with the same K/V fed as inputs[1]/[2].
	outExternal, err := gqaExternal.Forward(ctx, inp, kProj, vProj)
	if err != nil {
		t.Fatalf("external GQA Forward failed: %v", err)
	}

	if !testutils.IntSliceEqual(outInternal.Shape(), outExternal.Shape()) {
		t.Fatalf("shape mismatch: internal %v vs external %v",
			outInternal.Shape(), outExternal.Shape())
	}

	const tol = 1e-5
	for i := range outInternal.Data() {
		diff := outInternal.Data()[i] - outExternal.Data()[i]
		if math.Abs(float64(diff)) > tol {
			t.Fatalf("output mismatch at idx %d: internal=%f external=%f diff=%f",
				i, outInternal.Data()[i], outExternal.Data()[i], diff)
		}
	}
}

// TestGroupedQueryAttention_ExternalKV_RejectsKEqV verifies the mutual
// exclusivity of WithExternalKV and SetKEqV.
func TestGroupedQueryAttention_ExternalKV_RejectsKEqV(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	gqa, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		16, 4, 2,
		WithNoRoPE[float32](),
		WithExternalKV[float32](),
	)
	if err != nil {
		t.Fatalf("failed to construct external GQA: %v", err)
	}

	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic when SetKEqV is called on externalKV GQA")
		}
	}()
	gqa.SetKEqV(true)
}

// TestGroupedQueryAttention_ExternalKV_RejectsShortInputs verifies Forward
// requires hidden+K+V inputs in external mode.
func TestGroupedQueryAttention_ExternalKV_RejectsShortInputs(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	gqa, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		16, 4, 2,
		WithNoRoPE[float32](),
		WithExternalKV[float32](),
	)
	if err != nil {
		t.Fatalf("failed to construct external GQA: %v", err)
	}

	inp, err := tensor.New[float32]([]int{1, 3, 16}, nil)
	if err != nil {
		t.Fatalf("failed creating input: %v", err)
	}
	if _, err := gqa.Forward(context.Background(), inp); err == nil {
		t.Fatalf("expected error when externalKV Forward is called without K/V inputs")
	}
}

func TestGroupedQueryAttention_NoRoPE_ScaleRope(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	gqa, err := NewGroupedQueryAttention[float32](
		engine,
		numeric.Float32Ops{},
		16, 4, 4,
		WithNoRoPE[float32](),
	)
	if err != nil {
		t.Fatalf("failed to construct GQA: %v", err)
	}

	// Should return nil when rope is nil.
	if err := gqa.ScaleRope(context.Background(), 2.0); err != nil {
		t.Fatalf("ScaleRope with nil rope should return nil, got: %v", err)
	}
}
