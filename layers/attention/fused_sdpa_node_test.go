package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// TestFusedSDPA_NodeContract verifies the graph.Node interface is satisfied
// and the metadata is well-formed.
func TestFusedSDPA_NodeContract(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := NewFusedSDPA[float32](engine, 4)

	// Compile-time interface check.
	var _ graph.Node[float32] = node

	if node.OpType() != "FusedSDPA" {
		t.Fatalf("OpType: want FusedSDPA, got %s", node.OpType())
	}
	attrs := node.Attributes()
	if _, ok := attrs["head_dim"]; !ok {
		t.Fatalf("Attributes missing head_dim: %v", attrs)
	}
	if _, ok := attrs["causal"]; !ok {
		t.Fatalf("Attributes missing causal: %v", attrs)
	}
	if got := node.Parameters(); got != nil {
		t.Fatalf("Parameters: want nil, got %v", got)
	}
}

// TestFusedSDPA_BadInputCount ensures the wrapper rejects malformed Forward
// arities.
func TestFusedSDPA_BadInputCount(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := NewFusedSDPA[float32](engine, 4)

	_, err := node.Forward(context.Background())
	if err == nil {
		t.Fatalf("Forward with 0 inputs: want error, got nil")
	}

	q := fusedSDPAMustTensor[float32](t, []int{1, 2, 4}, []float32{1, 0, 0, 0, 0, 1, 0, 0})
	_, err = node.Forward(context.Background(), q, q)
	if err == nil {
		t.Fatalf("Forward with 2 inputs: want error, got nil")
	}
}

func TestFusedSDPA_ForwardBackwardEquivalence_Float32(t *testing.T) {
	runEquivalenceFloat32(t, false /*causal*/, false /*withMask*/, 1e-6, 1e-5)
	runEquivalenceFloat32(t, true /*causal*/, false /*withMask*/, 1e-6, 1e-5)
	runEquivalenceFloat32(t, false /*causal*/, true /*withMask*/, 1e-6, 1e-5)
}

func TestFusedSDPA_ForwardBackwardEquivalence_Float64(t *testing.T) {
	runEquivalenceFloat64(t, false, false, 1e-12, 1e-10)
	runEquivalenceFloat64(t, true, false, 1e-12, 1e-10)
	runEquivalenceFloat64(t, false, true, 1e-12, 1e-10)
}

func runEquivalenceFloat32(t *testing.T, causal, withMask bool, fwdTol, bwdTol float32) {
	t.Helper()
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	const batch, seqQ, seqK, headDim = 2, 3, 3, 4
	q := fusedSDPAMustTensor[float32](t, []int{batch, seqQ, headDim}, fusedSDPARandSliceF32(batch*seqQ*headDim, 11))
	k := fusedSDPAMustTensor[float32](t, []int{batch, seqK, headDim}, fusedSDPARandSliceF32(batch*seqK*headDim, 22))
	v := fusedSDPAMustTensor[float32](t, []int{batch, seqK, headDim}, fusedSDPARandSliceF32(batch*seqK*headDim, 33))

	var mask *tensor.TensorNumeric[float32]
	if withMask {
		// 4D mask: (batchSize/numHeads, numHeads, seqQ, seqK). Use numHeads = batch.
		md := make([]float32, batch*seqQ*seqK)
		mask = fusedSDPAMustTensor[float32](t, []int{1, batch, seqQ, seqK}, md)
	}

	// Reference path: bare ScaledDotProductAttention.
	refOpts := []ScaledDotProductAttentionOption[float32]{}
	if !causal {
		refOpts = append(refOpts, WithBidirectional[float32]())
	}
	ref := NewScaledDotProductAttention(engine, headDim, refOpts...)
	if causal {
		ref.SetCausal(true)
	}
	refOut, err := ref.Forward(ctx, q, k, v, mask)
	if err != nil {
		t.Fatalf("ref forward: %v", err)
	}

	// Wrapper path.
	nodeOpts := []FusedSDPAOption[float32]{}
	if !causal {
		nodeOpts = append(nodeOpts, WithFusedSDPABidirectional[float32]())
	}
	node := NewFusedSDPA[float32](engine, headDim, nodeOpts...)

	var inputs []*tensor.TensorNumeric[float32]
	if withMask {
		inputs = []*tensor.TensorNumeric[float32]{q, k, v, mask}
	} else {
		inputs = []*tensor.TensorNumeric[float32]{q, k, v}
	}
	nodeOut, err := node.Forward(ctx, inputs...)
	if err != nil {
		t.Fatalf("node forward: %v", err)
	}

	fusedSDPAAssertCloseF32(t, "forward", refOut, nodeOut, fwdTol)

	// OutputShape sanity.
	if got := node.OutputShape(); !fusedSDPAIntSliceEq(got, refOut.Shape()) {
		t.Fatalf("OutputShape: want %v, got %v", refOut.Shape(), got)
	}

	// Backward.
	dOut := fusedSDPAMustTensor[float32](t, refOut.Shape(), fusedSDPARandSliceF32(fusedSDPAProd(refOut.Shape()), 44))
	refGrads, err := ref.Backward(ctx, types.FullBackprop, dOut, q, k, v)
	if err != nil {
		t.Fatalf("ref backward: %v", err)
	}
	nodeGrads, err := node.Backward(ctx, types.FullBackprop, dOut, inputs...)
	if err != nil {
		t.Fatalf("node backward: %v", err)
	}
	if withMask {
		if len(nodeGrads) != 4 {
			t.Fatalf("wrapper backward: want 4 grads (mask slot), got %d", len(nodeGrads))
		}
		if nodeGrads[3] != nil {
			t.Fatalf("wrapper backward: mask grad slot should be nil")
		}
	} else if len(nodeGrads) != 3 {
		t.Fatalf("wrapper backward: want 3 grads, got %d", len(nodeGrads))
	}

	for i, name := range []string{"dQ", "dK", "dV"} {
		fusedSDPAAssertCloseF32(t, "backward "+name, refGrads[i], nodeGrads[i], bwdTol)
	}
}

func runEquivalenceFloat64(t *testing.T, causal, withMask bool, fwdTol, bwdTol float64) {
	t.Helper()
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})

	const batch, seqQ, seqK, headDim = 2, 3, 3, 4
	q := fusedSDPAMustTensor[float64](t, []int{batch, seqQ, headDim}, fusedSDPARandSliceF64(batch*seqQ*headDim, 11))
	k := fusedSDPAMustTensor[float64](t, []int{batch, seqK, headDim}, fusedSDPARandSliceF64(batch*seqK*headDim, 22))
	v := fusedSDPAMustTensor[float64](t, []int{batch, seqK, headDim}, fusedSDPARandSliceF64(batch*seqK*headDim, 33))

	var mask *tensor.TensorNumeric[float64]
	if withMask {
		md := make([]float64, batch*seqQ*seqK)
		mask = fusedSDPAMustTensor[float64](t, []int{1, batch, seqQ, seqK}, md)
	}

	refOpts := []ScaledDotProductAttentionOption[float64]{}
	if !causal {
		refOpts = append(refOpts, WithBidirectional[float64]())
	}
	ref := NewScaledDotProductAttention(engine, headDim, refOpts...)
	if causal {
		ref.SetCausal(true)
	}
	refOut, err := ref.Forward(ctx, q, k, v, mask)
	if err != nil {
		t.Fatalf("ref forward: %v", err)
	}

	nodeOpts := []FusedSDPAOption[float64]{}
	if !causal {
		nodeOpts = append(nodeOpts, WithFusedSDPABidirectional[float64]())
	}
	node := NewFusedSDPA[float64](engine, headDim, nodeOpts...)

	var inputs []*tensor.TensorNumeric[float64]
	if withMask {
		inputs = []*tensor.TensorNumeric[float64]{q, k, v, mask}
	} else {
		inputs = []*tensor.TensorNumeric[float64]{q, k, v}
	}
	nodeOut, err := node.Forward(ctx, inputs...)
	if err != nil {
		t.Fatalf("node forward: %v", err)
	}

	fusedSDPAAssertCloseF64(t, "forward", refOut, nodeOut, fwdTol)

	dOut := fusedSDPAMustTensor[float64](t, refOut.Shape(), fusedSDPARandSliceF64(fusedSDPAProd(refOut.Shape()), 44))
	refGrads, err := ref.Backward(ctx, types.FullBackprop, dOut, q, k, v)
	if err != nil {
		t.Fatalf("ref backward: %v", err)
	}
	nodeGrads, err := node.Backward(ctx, types.FullBackprop, dOut, inputs...)
	if err != nil {
		t.Fatalf("node backward: %v", err)
	}
	for i, name := range []string{"dQ", "dK", "dV"} {
		fusedSDPAAssertCloseF64(t, "backward "+name, refGrads[i], nodeGrads[i], bwdTol)
	}
}

// --- helpers ---

func fusedSDPAMustTensor[T tensor.Numeric](t *testing.T, shape []int, data []T) *tensor.TensorNumeric[T] {
	t.Helper()
	tn, err := tensor.New[T](shape, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return tn
}

func fusedSDPAIntSliceEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func fusedSDPAProd(s []int) int {
	p := 1
	for _, v := range s {
		p *= v
	}
	return p
}

// fusedSDPARandSliceF32/F64 produce deterministic small floats in [-1, 1] using a tiny
// LCG seeded by the supplied seed. Avoids math/rand to keep tests hermetic.
func fusedSDPARandSliceF32(n int, seed uint32) []float32 {
	out := make([]float32, n)
	state := seed | 1
	for i := range out {
		state = state*1664525 + 1013904223
		// map to [-1, 1]
		out[i] = float32(int32(state)) / float32(math.MaxInt32)
	}
	return out
}

func fusedSDPARandSliceF64(n int, seed uint64) []float64 {
	out := make([]float64, n)
	state := seed | 1
	for i := range out {
		state = state*6364136223846793005 + 1442695040888963407
		out[i] = float64(int64(state)) / float64(math.MaxInt64)
	}
	return out
}

func fusedSDPAAssertCloseF32(t *testing.T, what string, want, got *tensor.TensorNumeric[float32], tol float32) {
	t.Helper()
	if !fusedSDPAIntSliceEq(want.Shape(), got.Shape()) {
		t.Fatalf("%s shape: want %v, got %v", what, want.Shape(), got.Shape())
	}
	a := want.Data()
	b := got.Data()
	if len(a) != len(b) {
		t.Fatalf("%s data length: want %d, got %d", what, len(a), len(b))
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			t.Fatalf("%s [%d]: want %v, got %v (diff %v > tol %v)", what, i, a[i], b[i], d, tol)
		}
	}
}

func fusedSDPAAssertCloseF64(t *testing.T, what string, want, got *tensor.TensorNumeric[float64], tol float64) {
	t.Helper()
	if !fusedSDPAIntSliceEq(want.Shape(), got.Shape()) {
		t.Fatalf("%s shape: want %v, got %v", what, want.Shape(), got.Shape())
	}
	a := want.Data()
	b := got.Data()
	if len(a) != len(b) {
		t.Fatalf("%s data length: want %d, got %d", what, len(a), len(b))
	}
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			t.Fatalf("%s [%d]: want %v, got %v (diff %v > tol %v)", what, i, a[i], b[i], d, tol)
		}
	}
}

// TestFusedSDPA_SaverAware verifies FusedSDPA implements graph.SaverAware and
// fans the Saver into the inner SDPA so its cached Q/K/V and attention
// weights are save-for-backward pinned (zerfoo#864). Without this, the inner
// SDPA's cached forward tensors are unpinned arena intermediates on GPU
// engines -- the zerfoo#842 corruption class.
func TestFusedSDPA_SaverAware(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node := NewFusedSDPA[float32](engine, 4, WithFusedSDPABidirectional[float32]())

	// Compile-time + runtime interface check.
	sa, ok := any(node).(graph.SaverAware[float32])
	if !ok {
		t.Fatal("FusedSDPA does not implement graph.SaverAware")
	}

	rec := &recordingSaver[float32]{}
	sa.SetSaver(rec)

	q, _ := tensor.New[float32]([]int{1, 2, 4}, make([]float32, 8))
	k, _ := tensor.New[float32]([]int{1, 2, 4}, make([]float32, 8))
	v, _ := tensor.New[float32]([]int{1, 2, 4}, make([]float32, 8))
	if _, err := node.Forward(context.Background(), q, k, v); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// Inner SDPA must have registered q, k, v (and the attention weights on
	// the discrete CPU path) with the saver.
	if rec.saved < 3 {
		t.Fatalf("SaveForBackward registered %d tensors, want >= 3 (q, k, v)", rec.saved)
	}
}

// recordingSaver counts SaveForBackward registrations.
type recordingSaver[T tensor.Numeric] struct{ saved int }

func (r *recordingSaver[T]) SaveForBackward(ts ...*tensor.TensorNumeric[T]) {
	r.saved += len(ts)
}
