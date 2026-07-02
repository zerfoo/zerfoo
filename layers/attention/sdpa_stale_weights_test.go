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

// fakeFusedVMulEngine wraps the CPU engine and implements
// compute.FusedSoftmaxVMulProvider so SDPA's fused decode early-return path
// (which skips populating the attentionWeights cache) is reachable in CI
// without a GPU. The math matches the contract: softmax(scores*scale) @ V.
type fakeFusedVMulEngine struct {
	compute.Engine[float32]
}

func (f *fakeFusedVMulEngine) GPUFusedSoftmaxVMul(scores, v *tensor.TensorNumeric[float32], scale float32) (*tensor.TensorNumeric[float32], error) {
	ctx := context.Background()
	scaled, err := f.Engine.MulScalar(ctx, scores, scale)
	if err != nil {
		return nil, err
	}
	w, err := f.Engine.Softmax(ctx, scaled, -1)
	if err != nil {
		return nil, err
	}
	return f.Engine.MatMul(ctx, w, v)
}

// TestSDPA_BackwardAfterFusedForward_NotStale is the regression test for the
// stale attentionWeights cache: a fused forward (flash / fused softmax-V)
// returns early WITHOUT setting attentionWeights, while Backward repopulates
// the cache when it recomputes. Before the fix, the SECOND step's Backward
// saw the FIRST step's cached weights (non-nil) and consumed them instead of
// recomputing from the current step's Q/K -- on a GPU arena that memory has
// been pool-Reset, producing deterministically wrong gradients from step 2
// of every training loop (observed in-situ: Wolf CrossAsset GB10, fold-0
// acc 0.7042 -> 0.4948). Here the stale weights are merely from different
// inputs, which is enough to make the gradients diverge from a fresh
// reference.
func TestSDPA_BackwardAfterFusedForward_NotStale(t *testing.T) {
	cpu := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	fake := &fakeFusedVMulEngine{Engine: cpu}
	ctx := context.Background()
	const headDim = 4

	mk := func(vals []float32, shape ...int) *tensor.TensorNumeric[float32] {
		tt, err := tensor.New[float32](shape, vals)
		if err != nil {
			t.Fatal(err)
		}
		return tt
	}
	// seqQ = 1 so the fused softmax-V early return fires; seqKV = 2.
	q1 := mk([]float32{0.4, -0.2, 0.1, 0.3}, 1, 1, headDim)
	k1 := mk([]float32{0.2, 0.1, -0.3, 0.5, -0.1, 0.4, 0.2, -0.2}, 1, 2, headDim)
	v1 := mk([]float32{1, 2, 3, 4, 5, 6, 7, 8}, 1, 2, headDim)

	q2 := mk([]float32{-0.5, 0.3, 0.2, -0.1}, 1, 1, headDim)
	k2 := mk([]float32{-0.2, 0.4, 0.1, -0.5, 0.3, -0.4, -0.2, 0.2}, 1, 2, headDim)
	v2 := mk([]float32{8, 7, 6, 5, 4, 3, 2, 1}, 1, 2, headDim)

	dOut := mk([]float32{1, -1, 0.5, 2}, 1, 1, headDim)

	// Step 1 on the shared SDPA: fused forward + backward (backward
	// recomputes and, as a side effect, populates the weights cache).
	shared := NewBidirectionalSDPA[float32](fake, headDim)
	if _, err := shared.Forward(ctx, q1, k1, v1, nil); err != nil {
		t.Fatalf("step-1 forward: %v", err)
	}
	if _, err := shared.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil); err != nil {
		t.Fatalf("step-1 backward: %v", err)
	}

	// Step 2 on the shared SDPA with DIFFERENT inputs.
	if _, err := shared.Forward(ctx, q2, k2, v2, nil); err != nil {
		t.Fatalf("step-2 forward: %v", err)
	}
	got, err := shared.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil)
	if err != nil {
		t.Fatalf("step-2 backward: %v", err)
	}

	// Reference: a fresh SDPA computing step 2 alone.
	fresh := NewBidirectionalSDPA[float32](fake, headDim)
	if _, err := fresh.Forward(ctx, q2, k2, v2, nil); err != nil {
		t.Fatalf("ref forward: %v", err)
	}
	want, err := fresh.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil)
	if err != nil {
		t.Fatalf("ref backward: %v", err)
	}

	names := []string{"dQ", "dK", "dV"}
	for gi := range want {
		g, w := got[gi].Data(), want[gi].Data()
		for i := range w {
			if d := math.Abs(float64(g[i] - w[i])); d > 1e-6 {
				t.Fatalf("%s[%d] stale-cache divergence: got %v want %v (step-2 backward consumed step-1 attention weights)", names[gi], i, g[i], w[i])
			}
		}
	}
}
