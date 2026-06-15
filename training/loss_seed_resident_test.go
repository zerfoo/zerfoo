package training

import (
	"context"
	"testing"

	core "github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestLossSeedDeviceResident is the regression gate for issue #875.
//
// #872 fixed the loss-backward seed value (d(loss)/d(loss) = 1 instead of L),
// but built the seed via a fresh host-backed tensor.New on EVERY
// ComputeGradients call. On a GPU engine that host ones tensor is host->device
// cudaMemcpy'd each step; CaptureReplayRunner records ComputeGradients inside a
// CUDA-graph stream-capture region, and a host->device copy on the legacy
// stream during capture is illegal ("operation not permitted when stream is
// capturing"), so capture-on training crashed.
//
// The fix caches one DEVICE-RESIDENT ones seed per loss shape on the strategy's
// accumulator, built once (during an eager warmup step, before capture begins)
// and reused every step. A full capture-on assertion needs a GPU/GraphCapturer
// engine and is out of scope here (the Wolf consumer re-benches capture-on on
// the GB10 after release). The CPU-testable invariant that makes the capture
// fix true is: the seed is allocated ONCE and reused -- no per-call host
// allocation happens inside ComputeGradients. This test asserts that invariant
// via pointer identity of the cached seed tensor (and its backing storage)
// across multiple ComputeGradients calls, plus that the seed value is 1.0.
func TestLossSeedDeviceResident(t *testing.T) {
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)

	const (
		batch = 2
		in    = 4
		out   = 3
	)
	lin, err := core.NewLinear[float64]("lin", engine, ops, in, out)
	if err != nil {
		t.Fatalf("NewLinear: %v", err)
	}
	wd := lin.Parameters()[0].Value.Data()
	for i := range wd {
		wd[i] = 0.05 * float64(i+1)
	}

	b := graph.NewBuilder[float64](engine)
	inNode := b.Input([]int{batch, in})
	b.AddNode(lin, inNode)
	g, err := b.Build(lin)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	lossNode := loss.NewCrossEntropyLoss[float64](engine)

	inputData := make([]float64, batch*in)
	for i := range inputData {
		inputData[i] = 0.1*float64(i) - 0.3
	}
	input, err := tensor.New[float64]([]int{batch, in}, inputData)
	if err != nil {
		t.Fatalf("input tensor: %v", err)
	}
	targets, err := tensor.New[float64]([]int{batch}, []float64{2, 0})
	if err != nil {
		t.Fatalf("targets tensor: %v", err)
	}
	batchData := Batch[float64]{
		Inputs:  map[graph.Node[float64]]*tensor.TensorNumeric[float64]{inNode: input},
		Targets: targets,
	}

	ctx := context.Background()
	strategy := NewDefaultBackpropStrategy[float64]()

	// Step 1: populates the seed cache (this is the eager warmup step that, on
	// a GPU engine, runs OUTSIDE any capture region).
	if _, err := strategy.ComputeGradients(ctx, g, lossNode, batchData); err != nil {
		t.Fatalf("ComputeGradients (step 1): %v", err)
	}

	if len(strategy.grads.seeds) != 1 {
		t.Fatalf("after step 1: len(seeds) = %d, want 1 (exactly one cached seed)", len(strategy.grads.seeds))
	}
	var firstSeed *tensor.TensorNumeric[float64]
	for _, s := range strategy.grads.seeds {
		firstSeed = s
	}
	if firstSeed == nil {
		t.Fatal("after step 1: cached seed is nil")
	}

	// The seed value must be exactly 1.0 (preserve the #872 fix): a seed of L
	// would scale every gradient by the loss value.
	for i, v := range firstSeed.Data() {
		if v != 1.0 {
			t.Fatalf("seed[%d] = %v, want 1.0 (#872): loss.Backward must be seeded with d(loss)/d(loss)=1, not the loss value", i, v)
		}
	}
	// Scalar loss -> [1] seed.
	if got := firstSeed.Shape(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("seed shape = %v, want [1]", got)
	}

	firstStorage := firstSeed.GetStorage()

	// Steps 2 and 3: must REUSE the same cached seed tensor and its backing
	// storage -- no per-call host allocation. This is the property that makes
	// CUDA-graph capture legal: capture-step ComputeGradients touches an
	// already-resident buffer and enqueues no host->device copy.
	for step := 2; step <= 3; step++ {
		if _, err := strategy.ComputeGradients(ctx, g, lossNode, batchData); err != nil {
			t.Fatalf("ComputeGradients (step %d): %v", step, err)
		}
		if len(strategy.grads.seeds) != 1 {
			t.Fatalf("after step %d: len(seeds) = %d, want 1 (no new seed allocated)", step, len(strategy.grads.seeds))
		}
		got := strategy.grads.seeds[shapeKey(firstSeed.Shape())]
		if got != firstSeed {
			t.Fatalf("after step %d: cached seed tensor changed (got %p, want %p): seed must be reused, not reallocated per step (#875)", step, got, firstSeed)
		}
		if got.GetStorage() != firstStorage {
			t.Fatalf("after step %d: cached seed storage changed: the seed must be device-resident and stable across steps (#875)", step)
		}
	}
}

// TestLossSeedDeviceResident_NilEngine verifies the engine-less graph path
// (parameter-fixture graphs) still produces a correct 1.0 seed and caches it,
// exercising buildOnesSeed's nil-engine fallback.
func TestLossSeedDeviceResident_NilEngine(t *testing.T) {
	var acc gradAccumulator[float64]
	ref, err := tensor.New[float64]([]int{1}, []float64{0})
	if err != nil {
		t.Fatalf("ref tensor: %v", err)
	}

	first, err := acc.seedFor(nil, ref)
	if err != nil {
		t.Fatalf("seedFor (nil engine): %v", err)
	}
	if d := first.Data(); len(d) != 1 || d[0] != 1.0 {
		t.Fatalf("nil-engine seed = %v, want [1]", first.Data())
	}

	second, err := acc.seedFor(nil, ref)
	if err != nil {
		t.Fatalf("seedFor (nil engine, second call): %v", err)
	}
	if second != first {
		t.Fatalf("nil-engine seed not cached: got %p, want %p", second, first)
	}
}
