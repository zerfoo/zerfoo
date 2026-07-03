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
// The fixture trains two identically-initialised copies of a tiny synthetic
// model on the SAME constant batch from the SAME seed: one through the
// capture-replay runner with capture DISABLED (the eager reference) and one
// with capture ENABLED. It then compares the loss trajectories and FAILS if
// the capture-on trajectory diverges upward while the eager trajectory
// converges -- i.e. it is the RED proof that #878 is present, and turns GREEN
// once the Phase 1 root-cause fix lands (both paths must then converge alike).
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
//   - ZERFOO_RUN_878_FIXTURE=1 must be set. This fixture is EXPECTED RED on
//     GPU until zerfoo#878 is fixed in Phase 1, so the env gate keeps standing
//     validation/CI runs green while preserving the red proof for the E131
//     GB10 job and the Phase 1 fix loop to run on demand.
//
// The test also sets ZERFOO_UNSAFE_CAPTURE_TRAINING=1: once E129's loud-fail
// gate (T129.2) lands, constructing a capture-replay training runner will error
// unless that override is set. Setting it here keeps the fixture runnable
// through the gate so it can still serve as the Phase 1 red/green proof.
func TestCaptureReplayGradientDivergence878(t *testing.T) {
	if os.Getenv("ZERFOO_RUN_878_FIXTURE") != "1" {
		t.Skip("zerfoo#878 fixture is opt-in and expected RED on GPU until the Phase 1 fix; set ZERFOO_RUN_878_FIXTURE=1 to run it")
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

	// Eager reference: identical runner with capture DISABLED (transparent
	// passthrough to the backprop strategy).
	eagerLosses, eagerRunner := run878Trajectory(t, engine, ops,
		inputData, onehotData, ns, inDim, dModel, classes, warmup, lr, nSteps, initSeed, false)
	defer func() { _ = eagerRunner.Close() }()
	if eagerRunner.Enabled() {
		t.Fatalf("eager reference runner should have capture disabled but reports Enabled()==true")
	}

	// Capture-on: identical model/seed/data, capture ENABLED.
	captureLosses, captureRunner := run878Trajectory(t, engine, ops,
		inputData, onehotData, ns, inDim, dModel, classes, warmup, lr, nSteps, initSeed, true)
	defer func() { _ = captureRunner.Close() }()
	if !captureRunner.Enabled() {
		t.Skip("engine does not implement GraphCapturer; capture-replay inactive, nothing to prove")
	}

	eagerInit, eagerFinal := edgeMean(eagerLosses, 3)
	capInit, capFinal := edgeMean(captureLosses, 3)

	t.Logf("eager   : init=%.4f final=%.4f (captures=%d replays=%d)",
		eagerInit, eagerFinal, eagerRunner.CapturesPerformed(), eagerRunner.ReplaysPerformed())
	t.Logf("capture : init=%.4f final=%.4f (captures=%d replays=%d)",
		capInit, capFinal, captureRunner.CapturesPerformed(), captureRunner.ReplaysPerformed())

	// Precondition: the eager reference must actually train. If it does not,
	// the fixture is broken (bad LR, degenerate data, ...) and cannot prove a
	// capture-specific divergence -- fail loudly rather than emit a false red.
	if !(eagerFinal < eagerInit*0.9) {
		t.Fatalf("eager reference did not converge (init=%.4f final=%.4f); "+
			"fixture cannot isolate the #878 capture-replay divergence", eagerInit, eagerFinal)
	}

	// The #878 signature: capture-on ascends (or lands far above the eager
	// baseline) while eager descends. Tolerances are generous -- the real bug
	// is a 10-20x blow-up, not a subtle numeric drift. A correct capture path
	// converges like eager and this assertion passes (fixture turns GREEN).
	diverged := capFinal > capInit*1.2 || capFinal > eagerFinal*3.0
	if diverged {
		t.Errorf("zerfoo#878 REPRODUCED: capture-replay training diverged "+
			"(loss init=%.4f -> final=%.4f) while the identical eager config converged "+
			"(loss init=%.4f -> final=%.4f). captures=%d replays=%d. "+
			"This fixture is expected RED on GPU until the Phase 1 root-cause fix; "+
			"see https://github.com/zerfoo/zerfoo/issues/878",
			capInit, capFinal, eagerInit, eagerFinal,
			captureRunner.CapturesPerformed(), captureRunner.ReplaysPerformed())
	}
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
