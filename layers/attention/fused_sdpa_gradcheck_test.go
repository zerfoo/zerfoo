package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/testing/gradcheck"
)

// Registers FusedSDPA with ztensor's shared gradcheck harness (plan T135.4,
// UC-GH-001). fused_sdpa_node_test.go already asserts fused-vs-unfused
// forward/backward EQUIVALENCE (the wrapper matches the inner
// ScaledDotProductAttention bit-for-bit), but nothing previously verified
// that the underlying analytic Backward -- the softmax-Jacobian chain rule
// in scaled_dot_product_attention.go -- is itself a correct Jacobian of
// Forward. This closes that gap with float64 central-difference gradcheck
// against Q/K/V, covering both the causal and bidirectional (encoder-style)
// masking branches.
func TestFusedSDPA_GradcheckCausal(t *testing.T) {
	const batch, seqQ, seqK, headDim = 2, 3, 3, 4
	op := gradcheck.OpInfo{
		Name: "FusedSDPA_Causal",
		Seed: 10,
		Make: func(e compute.Engine[float64]) (graph.Node[float64], error) {
			return NewFusedSDPA[float64](e, headDim), nil
		},
		InputShapes: [][]int{
			{batch, seqQ, headDim},
			{batch, seqK, headDim},
			{batch, seqK, headDim},
		},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatalf("gradcheck mechanical failure: %v", err)
	}
	if !report.OK() {
		t.Fatalf("FusedSDPA (causal) gradcheck failed:\n%s", report)
	}
}

func TestFusedSDPA_GradcheckBidirectional(t *testing.T) {
	const batch, seqQ, seqK, headDim = 2, 3, 3, 4
	op := gradcheck.OpInfo{
		Name: "FusedSDPA_Bidirectional",
		Seed: 20,
		Make: func(e compute.Engine[float64]) (graph.Node[float64], error) {
			return NewFusedSDPA[float64](e, headDim, WithFusedSDPABidirectional[float64]()), nil
		},
		InputShapes: [][]int{
			{batch, seqQ, headDim},
			{batch, seqK, headDim},
			{batch, seqK, headDim},
		},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatalf("gradcheck mechanical failure: %v", err)
	}
	if !report.OK() {
		t.Fatalf("FusedSDPA (bidirectional) gradcheck failed:\n%s", report)
	}
}

// TestFusedSDPA_GradcheckDecode covers the seqQ==1 single-query decode shape,
// which Forward routes through the fused softmax+V-matmul short-circuit
// (compute.FusedSoftmaxVMulProvider) on GPU engines and the "skip causal mask
// entirely" branch on all engines (scaled_dot_product_attention.go: "every
// cached position is visible"). On this float64 CPU engine no fused provider
// is present, but the skip-masking-for-decode Forward branch is still
// exercised, and its Backward must produce the same Jacobian as any other
// causal shape.
func TestFusedSDPA_GradcheckDecode(t *testing.T) {
	const batch, seqQ, seqK, headDim = 2, 1, 4, 4
	op := gradcheck.OpInfo{
		Name: "FusedSDPA_Decode",
		Seed: 30,
		Make: func(e compute.Engine[float64]) (graph.Node[float64], error) {
			return NewFusedSDPA[float64](e, headDim), nil
		},
		InputShapes: [][]int{
			{batch, seqQ, headDim},
			{batch, seqK, headDim},
			{batch, seqK, headDim},
		},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatalf("gradcheck mechanical failure: %v", err)
	}
	if !report.OK() {
		t.Fatalf("FusedSDPA (decode, seqQ=1) gradcheck failed:\n%s", report)
	}
}
