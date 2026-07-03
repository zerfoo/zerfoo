package training_test

import (
	"context"
	"math"
	"math/rand/v2"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestCaptureReplayGradientDivergence878 is the regression fixture for
// zerfoo#878: CUDA-graph capture-replay TRAINING silently produces wrong
// gradients. On the GB10 (v1.50.2+) the capture-on path runs without crashing
// (captures==1, replays==steps-warmup) yet the losses ASCEND ~10-20x and the
// model collapses to the majority class, while the byte-identical EAGER
// configuration converges. The hypothesised cause is that the device-resident
// loss seed / persistent gradient-accumulator state (training.gradAccumulator,
// built once via engine.Fill) is captured into the graph with stale or aliased
// state and replayed against corrupted memory. See:
// https://github.com/zerfoo/zerfoo/issues/878
//
// The fixture trains three identically-initialised copies of a tiny synthetic
// model on the SAME constant batch from the SAME seed:
//
//  1. baseline -- capture DISABLED, no arena resets (the known-good loop);
//  2. eager    -- capture DISABLED, consumer-realistic per-step
//     engine.ResetPool (the per-sample/per-epoch pattern of real trainers);
//  3. capture  -- capture ENABLED, per-step engine.ResetPool.
//
// All three run identical math on identical data, and GB10 kernels are
// deterministic, so all three loss trajectories must MATCH step for step.
// The fixture FAILS if capture-on ascends while the baseline converges (the
// coarse Wolf-scale signature) OR if either reset trajectory drifts from the
// baseline beyond fp slack (the sharp signature: a gradient computed from
// aliased/stale state -- AdamW's approximate scale invariance can hide a
// mis-scaled gradient from the coarse check while training silently runs on
// corrupted state). It is the RED proof that #878 is present, and turns
// GREEN once the root-cause fix lands (all paths then converge alike).
//
// Logits are shaped [ns, 3] to match the 3-class target shape from the issue
// ([ns,3] and [B*ns,3] were hit identically -> framework-level, not
// consumer-specific). The loss node is CrossEntropyLossOneHot, the only
// cross-entropy loss in this repo that is device-pure (no .Data() reads, no
// host math) and therefore legal inside a capture region.
//
// GATING (both conditions must hold or the test skips):
//
//   - A real CUDA GPU engine must be constructible at runtime AND expose the
//     GraphCapturer interface. On CPU-only hosts (e.g. darwin dev machines)
//     the capture runner is a transparent passthrough, so there is nothing to
//     diverge and the fixture would be meaningless -- we skip.
//   - ZERFOO_RUN_878_FIXTURE=1 must be set. The fixture trains 3x40 steps on
//     the GPU and requires the ZERFOO_UNSAFE_CAPTURE_TRAINING override, so it
//     stays opt-in rather than part of the standing gate's default sweep.
//     (Historically it was also expected RED until the #878 root-cause fix --
//     the allocation-stable loss seed, training/grad_accum.go buildOnesSeed --
//     landed; it is GREEN on GB10 since that fix and now serves as the
//     regression gate for it.)
//
// The test also sets ZERFOO_UNSAFE_CAPTURE_TRAINING=1: once E129's loud-fail
// gate (T129.2) lands, constructing a capture-replay training runner will error
// unless that override is set. Setting it here keeps the fixture runnable
// through the gate so it can still serve as the Phase 1 red/green proof.
func TestCaptureReplayGradientDivergence878(t *testing.T) {
	if os.Getenv("ZERFOO_RUN_878_FIXTURE") != "1" {
		t.Skip("zerfoo#878 regression fixture is opt-in (GPU training run); set ZERFOO_RUN_878_FIXTURE=1 to run it")
	}

	ops := numeric.Float32Ops{}
	engine, err := compute.NewGPUEngine[float32](ops)
	if err != nil {
		t.Skipf("zerfoo#878 fixture requires a CUDA GPU engine (GraphCapturer); none available: %v", err)
	}
	defer func() { _ = engine.Close() }()

	// Once the T129.2 loud-fail gate lands, capture-replay training refuses to
	// run without this override. Set it so the red proof keeps working.
	t.Setenv("ZERFOO_UNSAFE_CAPTURE_TRAINING", "1")

	const (
		ns       = 8 // number of "samples" -> logits [ns, 3]
		classes  = 3
		inDim    = 6
		dModel   = 16
		warmup   = 3
		nSteps   = 40
		lr       = 1e-2
		dataSeed = 20260702
		initSeed = 878
	)

	// One fixed synthetic batch, reused every step (the capture-replay
	// "stable operands" contract: the same input/target tensors on every Step,
	// so no per-step host copy is enqueued inside the captured region).
	inputData, onehotData := synthBatch878(ns, inDim, classes, dataSeed)

	// Ground-truth baseline: capture DISABLED, no arena resets. This is the
	// plain, known-good training loop the other two trajectories must match.
	baseLosses, baseRunner := run878Trajectory(t, engine, ops,
		inputData, onehotData, ns, inDim, dModel, classes, warmup, lr, nSteps, initSeed, false, false)
	defer func() { _ = baseRunner.Close() }()

	// Eager reference: capture DISABLED, consumer-realistic per-step
	// engine.ResetPool. Identical math to the baseline; a divergence here
	// means eager training state is aliased by arena reuse.
	eagerLosses, eagerRunner := run878Trajectory(t, engine, ops,
		inputData, onehotData, ns, inDim, dModel, classes, warmup, lr, nSteps, initSeed, false, true)
	defer func() { _ = eagerRunner.Close() }()
	if eagerRunner.Enabled() {
		t.Fatalf("eager reference runner should have capture disabled but reports Enabled()==true")
	}

	// Capture-on: identical model/seed/data, capture ENABLED, per-step resets.
	captureLosses, captureRunner := run878Trajectory(t, engine, ops,
		inputData, onehotData, ns, inDim, dModel, classes, warmup, lr, nSteps, initSeed, true, true)
	defer func() { _ = captureRunner.Close() }()
	if !captureRunner.Enabled() {
		t.Skip("engine does not implement GraphCapturer; capture-replay inactive, nothing to prove")
	}

	baseInit, baseFinal := edgeMean(baseLosses, 3)
	eagerInit, eagerFinal := edgeMean(eagerLosses, 3)
	capInit, capFinal := edgeMean(captureLosses, 3)

	t.Logf("baseline: init=%.4f final=%.4f (no resets, capture off)", baseInit, baseFinal)
	t.Logf("eager   : init=%.4f final=%.4f (captures=%d replays=%d)",
		eagerInit, eagerFinal, eagerRunner.CapturesPerformed(), eagerRunner.ReplaysPerformed())
	t.Logf("capture : init=%.4f final=%.4f (captures=%d replays=%d)",
		capInit, capFinal, captureRunner.CapturesPerformed(), captureRunner.ReplaysPerformed())

	// Precondition: the baseline must actually train. If it does not, the
	// fixture is broken (bad LR, degenerate data, ...) and cannot prove a
	// state-aliasing divergence -- fail loudly rather than emit a false red.
	if !(baseFinal < baseInit*0.9) {
		t.Fatalf("baseline did not converge (init=%.4f final=%.4f); "+
			"fixture cannot isolate the #878 divergence", baseInit, baseFinal)
	}

	// The #878 signature, coarse form: capture-on ascends (or lands far above
	// the baseline) while the plain loop descends. The real Wolf-scale bug was
	// a 10-20x blow-up; keep this assertion for its diagnostic message.
	diverged := capFinal > capInit*1.2 || capFinal > baseFinal*3.0
	if diverged {
		t.Errorf("zerfoo#878 REPRODUCED: capture-replay training diverged "+
			"(loss init=%.4f -> final=%.4f) while the identical baseline converged "+
			"(loss init=%.4f -> final=%.4f). captures=%d replays=%d. "+
			"See https://github.com/zerfoo/zerfoo/issues/878",
			capInit, capFinal, baseInit, baseFinal,
			captureRunner.CapturesPerformed(), captureRunner.ReplaysPerformed())
	}

	// The #878 signature, sharp form: all three trajectories run identical
	// math on identical data from identical initial weights, so they must
	// MATCH step for step. The kernels are deterministic (verified: with no
	// resets, capture-on and eager produce bit-identical loss trajectories on
	// GB10), so any per-step drift beyond tiny fp slack means a gradient was
	// computed from aliased/stale state -- exactly the silent corruption class
	// this fixture guards. AdamW's approximate scale invariance can otherwise
	// hide a mis-scaled gradient from the coarse convergence check above.
	const trajTol = 1e-4
	if i, d := maxAbsDiff(baseLosses, eagerLosses); d > trajTol {
		t.Errorf("zerfoo#878 REPRODUCED (eager): per-step ResetPool changed the eager "+
			"loss trajectory (max |diff|=%.6f at step %d: baseline=%.6f eager=%.6f); "+
			"cross-step training state is aliased by arena reuse", d, i, baseLosses[i], eagerLosses[i])
	}
	if i, d := maxAbsDiff(baseLosses, captureLosses); d > trajTol {
		t.Errorf("zerfoo#878 REPRODUCED (capture): capture-replay loss trajectory diverged "+
			"from the baseline (max |diff|=%.6f at step %d: baseline=%.6f capture=%.6f); "+
			"the captured graph is replaying against aliased/stale state", d, i, baseLosses[i], captureLosses[i])
	}
}

// maxAbsDiff returns the index and magnitude of the largest element-wise
// absolute difference between two equal-length trajectories.
func maxAbsDiff(a, b []float32) (int, float64) {
	idx, max := 0, 0.0
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > max {
			idx, max = i, d
		}
	}
	return idx, max
}

// run878Trajectory builds a fresh, deterministically-initialised tiny MLP
// (Dense -> Sigmoid -> Dense) that emits [ns, classes] logits, wires a
// device-pure one-hot cross-entropy loss, and trains it for nSteps through a
// CaptureReplayRunner (capture toggled by captureEnabled). It returns the
// per-step loss trajectory and the runner (for its counters / Enabled state).
//
// Only public APIs are used: graph.Builder, core.Dense, activations.Sigmoid,
// training.DefaultBackpropStrategy, training.CaptureReplayRunner,
// loss.CrossEntropyLossOneHot, optimizer.AdamW.
func run878Trajectory(
	t *testing.T,
	engine *compute.GPUEngine[float32],
	ops numeric.Float32Ops,
	inputData, onehotData []float32,
	ns, inDim, dModel, classes, warmup int,
	lr float32,
	nSteps int,
	initSeed uint64,
	captureEnabled bool,
	resetPool bool,
) ([]float32, *training.CaptureReplayRunner[float32]) {
	t.Helper()
	ctx := context.Background()

	// Build the model graph. Identical seed -> identical initial weights across
	// the eager and capture trajectories.
	b := graph.NewBuilder[float32](engine)
	input := b.Input([]int{ns, inDim})

	l1, err := core.NewDense[float32]("l1", engine, ops, inDim, dModel)
	if err != nil {
		t.Fatalf("NewDense l1: %v", err)
	}
	h := b.AddNode(l1, input)
	// The nonlinearity must be DEVICE-PURE (engine ops only) to satisfy the
	// CaptureReplayRunner contract. ReLU/BaseActivation is not: it routes
	// through engine.UnaryOp, which GPUEngine delegates to the CPU engine --
	// a host D2H read plus a CPU-resident result mid-graph, both illegal
	// inside a capture region (observed live on GB10: "operation would make
	// the legacy stream depend on a capturing blocking stream"). Sigmoid is
	// composed of engine primitives (Exp, AddScalar, Div), all with native
	// f32 GPU kernels, so the walk stays on-device.
	sigmoid := activations.NewSigmoid[float32](engine, ops)
	h = b.AddNode(sigmoid, h)
	head, err := core.NewDense[float32]("head", engine, ops, dModel, classes)
	if err != nil {
		t.Fatalf("NewDense head: %v", err)
	}
	logits := b.AddNode(head, h)

	g, err := b.Build(logits)
	if err != nil {
		t.Fatalf("build graph: %v", err)
	}

	// Deterministic Xavier-ish init.
	rng := rand.New(rand.NewPCG(initSeed, initSeed))
	for _, p := range g.Parameters() {
		data := p.Value.Data()
		scale := float32(math.Sqrt(2.0 / float64(len(data)+1)))
		for i := range data {
			data[i] = (rng.Float32()*2 - 1) * scale
		}
	}

	// Stable operands, reused every Step.
	inputT, err := tensor.New[float32]([]int{ns, inDim}, append([]float32(nil), inputData...))
	if err != nil {
		t.Fatalf("new input tensor: %v", err)
	}
	onehotT, err := tensor.New[float32]([]int{ns, classes}, append([]float32(nil), onehotData...))
	if err != nil {
		t.Fatalf("new one-hot target tensor: %v", err)
	}

	// Pre-upload weights and operands to device-resident storage with stable
	// pointers -- required for capture-replay (parameters must be device-resident
	// and the optimizer updates them in place; operands must not be re-copied
	// from the host inside the captured region).
	var toUpload []*tensor.TensorNumeric[float32]
	for _, p := range g.Parameters() {
		toUpload = append(toUpload, p.Value)
	}
	toUpload = append(toUpload, inputT, onehotT)
	if err := engine.UploadWeights(toUpload); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	strategy := training.NewDefaultBackpropStrategy[float32]()
	strategy.SetEngine(engine)
	lossNode := loss.NewCrossEntropyLossOneHot[float32](engine)
	opt := optimizer.NewAdamW[float32](engine, lr, 0.9, 0.999, 1e-8, 0.0)

	// Toggle capture for this trajectory. NewCaptureReplayRunner reads
	// ZERFOO_DISABLE_CUDA_GRAPH exactly once, at construction: set it for the
	// eager reference (disabled -> passthrough), clear it for capture.
	if captureEnabled {
		t.Setenv(training.DisableCUDAGraphEnv, "")
	} else {
		t.Setenv(training.DisableCUDAGraphEnv, "1")
	}
	runner, rerr := training.NewCaptureReplayRunner[float32](strategy, engine, warmup)
	if rerr != nil {
		t.Fatalf("NewCaptureReplayRunner (captureEnabled=%v): %v", captureEnabled, rerr)
	}

	batch := training.Batch[float32]{
		Inputs:  map[graph.Node[float32]]*tensor.TensorNumeric[float32]{input: inputT},
		Targets: onehotT,
	}

	losses := make([]float32, nSteps)
	for step := 0; step < nSteps; step++ {
		lossVal, serr := runner.Step(ctx, g, lossNode, batch)
		if serr != nil {
			t.Fatalf("step %d: runner.Step (captureEnabled=%v): %v", step, captureEnabled, serr)
		}
		if math.IsNaN(float64(lossVal)) || math.IsInf(float64(lossVal), 0) {
			t.Fatalf("step %d: non-finite loss %.4f (captureEnabled=%v)", step, lossVal, captureEnabled)
		}
		losses[step] = lossVal

		if oerr := opt.Step(ctx, g.Parameters()); oerr != nil {
			t.Fatalf("step %d: optimizer.Step: %v", step, oerr)
		}

		// Consumer-realistic per-step arena reset (the per-sample/per-epoch
		// ResetPool pattern of real training loops; ztensor#167's arena reset
		// floor makes it legal while a captured graph is live). This is the
		// trigger for the #878 mechanism: any cached cross-step training state
		// the strategy holds MUST live in allocation-stable non-arena storage,
		// or the reset recycles it behind the cache -- and, under capture, the
		// graph's baked device address for it becomes permanently aliased with
		// a per-step intermediate. Without this reset the fixture cannot
		// distinguish a correct implementation from one that merely never
		// exercises arena reuse.
		if resetPool {
			engine.ResetPool()
		}
	}

	return losses, runner
}

// synthBatch878 builds a fixed synthetic classification batch: random input
// features in [-1,1]*0.1 and one deterministic class label per row encoded as
// a one-hot [ns, classes] tensor. The same (seed) yields the same batch, so
// both trajectories train on identical data.
func synthBatch878(ns, inDim, classes int, seed uint64) (inputData, onehotData []float32) {
	rng := rand.New(rand.NewPCG(seed, seed))
	inputData = make([]float32, ns*inDim)
	for i := range inputData {
		inputData[i] = (rng.Float32()*2 - 1) * 0.1
	}
	onehotData = make([]float32, ns*classes)
	for row := 0; row < ns; row++ {
		label := rng.IntN(classes)
		onehotData[row*classes+label] = 1
	}
	return inputData, onehotData
}

// edgeMean returns the mean of the first k and the mean of the last k elements
// of losses, used to characterise the start vs end of a trajectory robustly to
// per-step noise.
func edgeMean(losses []float32, k int) (initMean, finalMean float32) {
	if k > len(losses) {
		k = len(losses)
	}
	var head, tail float32
	for i := 0; i < k; i++ {
		head += losses[i]
		tail += losses[len(losses)-1-i]
	}
	return head / float32(k), tail / float32(k)
}
