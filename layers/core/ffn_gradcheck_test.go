package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/testing/gradcheck"
)

// Registers FFN's two activation paths (SwiGLU and GELU) with ztensor's
// shared gradcheck harness (plan T135.4, UC-GH-001). Both paths compose the
// GPU-fused fast path (compute.FusedSwiGLUProvider / geluForward) with a
// CPU/unfused fallback in Forward, but Backward always differentiates the
// UNFUSED decomposition regardless of which forward path executed --
// TestFFN_Backward only asserted "runs without erroring" ("Skip weight
// gradient checks for now") and never verified the analytic gradient
// against a numerical reference. This closes that gap: gradcheck compares
// FFN.Backward's analytic Jacobian (inputs AND all six Dense weight/bias
// parameters) against float64 central differences.
func TestFFN_GradcheckSwiGLU(t *testing.T) {
	op := gradcheck.OpInfo{
		Name: "FFN_SwiGLU",
		Seed: 1,
		Make: func(e compute.Engine[float64]) (graph.Node[float64], error) {
			return NewFFN[float64]("gc_ffn_swiglu", e, numeric.Float64Ops{}, 3, 4, 2, WithSwiGLU[float64]())
		},
		InputShapes: [][]int{{2, 3}},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatalf("gradcheck mechanical failure: %v", err)
	}
	if !report.OK() {
		t.Fatalf("FFN (SwiGLU) gradcheck failed:\n%s", report)
	}
}

func TestFFN_GradcheckGELU(t *testing.T) {
	op := gradcheck.OpInfo{
		Name: "FFN_GELU",
		Seed: 2,
		Make: func(e compute.Engine[float64]) (graph.Node[float64], error) {
			return NewFFN[float64]("gc_ffn_gelu", e, numeric.Float64Ops{}, 3, 4, 2, WithGELU[float64]())
		},
		InputShapes: [][]int{{2, 3}},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatalf("gradcheck mechanical failure: %v", err)
	}
	if !report.OK() {
		t.Fatalf("FFN (GELU) gradcheck failed:\n%s", report)
	}
}

// TestFFN_GradcheckSwiGLU_NoBias covers the no-bias configuration used by
// several GGUF architectures (bias parameters absent from Parameters()).
func TestFFN_GradcheckSwiGLU_NoBias(t *testing.T) {
	op := gradcheck.OpInfo{
		Name: "FFN_SwiGLU_NoBias",
		Seed: 3,
		Make: func(e compute.Engine[float64]) (graph.Node[float64], error) {
			return NewFFN[float64]("gc_ffn_swiglu_nobias", e, numeric.Float64Ops{}, 3, 4, 2,
				WithSwiGLU[float64](), WithFFNNoBias[float64]())
		},
		InputShapes: [][]int{{2, 3}},
	}
	report, err := op.Run(context.Background())
	if err != nil {
		t.Fatalf("gradcheck mechanical failure: %v", err)
	}
	if !report.OK() {
		t.Fatalf("FFN (SwiGLU, no bias) gradcheck failed:\n%s", report)
	}
}
