// Package training: CUDA-graph capture-replay for repeated identical
// training steps.
//
// Capture-replay TRAINING is currently gated off. zerfoo#878 showed that
// enabling CUDA-graph capture on the training walk silently produces wrong
// gradients (losses ascend 10-20x while the identical eager config
// converges), so NewCaptureReplayRunner refuses to build a capture-enabled
// runner unless ZERFOO_UNSAFE_CAPTURE_TRAINING=1 is set to acknowledge the
// hazard. Eager/passthrough construction (no GraphCapturer engine, or
// capture disabled via ZERFOO_DISABLE_CUDA_GRAPH) is unaffected and needs no
// override. This gate is containment only; the root-cause fix is tracked in
// zerfoo#878. It does not touch inference-side capture (generate/,
// inference/), which is a separate, unaffected path.
package training

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// DisableCUDAGraphEnv is the environment variable that disables CUDA-graph
// capture-replay when set to a non-empty value. It is read ONCE, at
// NewCaptureReplayRunner time -- a debug toggle, not a hot-path read.
const DisableCUDAGraphEnv = "ZERFOO_DISABLE_CUDA_GRAPH"

// UnsafeCaptureTrainingEnv is the environment variable that overrides the
// zerfoo#878 loud-fail gate. Constructing a capture-ENABLED training runner
// returns ErrCaptureTrainingDisabled unless this is set to "1". It is read
// ONCE, at NewCaptureReplayRunner time.
const UnsafeCaptureTrainingEnv = "ZERFOO_UNSAFE_CAPTURE_TRAINING"

// ErrCaptureTrainingDisabled is returned by NewCaptureReplayRunner when the
// runner would enable CUDA-graph capture on the training walk but the
// ZERFOO_UNSAFE_CAPTURE_TRAINING=1 override is not set. Capture-replay
// training is disabled pending the zerfoo#878 silent-gradient-divergence fix.
var ErrCaptureTrainingDisabled = errors.New(
	"capture-replay training is disabled pending zerfoo#878 (silent gradient divergence); " +
		"set ZERFOO_UNSAFE_CAPTURE_TRAINING=1 to override at your own risk")

// CaptureReplayRunner drives a per-step training loop (forward + loss +
// backward + gradient accumulation) through CUDA-graph capture-replay on
// engines that implement compute.GraphCapturer:
//
//	step 0 .. warmup-1 : normal eager walk (allocates persistent gradient
//	                     accumulators, lazy engine workspaces, ...)
//	step warmup        : the walk runs inside BeginCapture/EndCapture and
//	                     is recorded into a CUDA graph, then the graph is
//	                     launched once so the step actually executes
//	step warmup+1 ..   : ReplayGraph only -- one graph launch replaces the
//	                     entire per-step kernel-launch sequence
//
// Counters: CapturesPerformed increments once per capture (one per runner
// when capture succeeds), ReplaysPerformed once per graph launch (including
// the launch immediately after capture). For an N-step run the expected
// steady state is CapturesPerformed == 1 and ReplaysPerformed == N-warmup.
//
// Contract (the caller owns all of these; violating any of them silently
// corrupts training, because graph replay re-executes the recorded kernels
// with the recorded device pointers):
//
//   - Stable operands: every tensor in batch (inputs and targets) must be
//     the SAME tensor on every Step call. Per-step data is staged by
//     copying into those tensors (engine.Copy / memcpy into device storage)
//     BEFORE Step; the copies run outside the captured region.
//   - Static shapes: the graph, loss node, and batch shapes must be
//     identical on every Step.
//   - Device-pure walk: every node Forward/Backward and the loss node must
//     issue engine ops only -- no .Data() reads, no host math on tensor
//     contents, no per-call host tensor creation feeding engine ops. Nodes
//     that read tensor values on the host (e.g. loss nodes that compute the
//     loss value on the CPU) cannot be captured.
//   - No host-driven randomness: dropout masks or any host-RNG-dependent
//     values are frozen at capture time. Capture-replay is only valid when
//     such nodes are disabled.
//   - Stable parameter/gradient storage: parameters must be device-resident
//     with stable pointers (e.g. compute.WeightUploader), and the optimizer
//     must update weights and zero gradients IN PLACE (never reallocate or
//     repoint storage).
//
// The loss value returned by Step is read from the loss tensor recorded at
// capture time, AFTER the graph has executed (ReplayGraph synchronizes the
// stream), so it is exact for every step.
type CaptureReplayRunner[T tensor.Numeric] struct {
	strategy *DefaultBackpropStrategy[T]
	gc       compute.GraphCapturer

	enabled     bool
	warmupSteps int
	stepsSeen   uint64

	captured   bool
	handle     compute.GraphHandle
	lossTensor *tensor.TensorNumeric[T]

	capturesPerformed uint64
	replaysPerformed  uint64
}

// NewCaptureReplayRunner constructs a CaptureReplayRunner around strategy
// and engine. Capture is enabled only when the engine implements
// compute.GraphCapturer and ZERFOO_DISABLE_CUDA_GRAPH is unset/empty (read
// once, here). warmupSteps is the number of eager steps to run before
// capturing; values < 1 are clamped to 1 (the first step must run eagerly
// to allocate persistent gradient accumulators and lazy workspaces).
//
// When capture is disabled the runner is a transparent passthrough to
// strategy.ComputeGradients and the counters stay at zero; that construction
// path always succeeds.
//
// Loud-fail gate (zerfoo#878): capture-enabled TRAINING silently produces
// wrong gradients, so when this constructor would enable capture (a
// GraphCapturer engine and ZERFOO_DISABLE_CUDA_GRAPH unset) it returns
// ErrCaptureTrainingDisabled unless ZERFOO_UNSAFE_CAPTURE_TRAINING=1 is set.
// Enablement is fully decided here (the env reads and the interface assertion
// all happen at construction), so this is the correct and only gate point --
// Step never re-checks, keeping the hot path clean.
func NewCaptureReplayRunner[T tensor.Numeric](
	strategy *DefaultBackpropStrategy[T],
	engine compute.Engine[T],
	warmupSteps int,
) (*CaptureReplayRunner[T], error) {
	if warmupSteps < 1 {
		warmupSteps = 1
	}
	r := &CaptureReplayRunner[T]{
		strategy:    strategy,
		warmupSteps: warmupSteps,
	}
	gc, ok := engine.(compute.GraphCapturer)
	if !ok {
		return r, nil
	}
	if os.Getenv(DisableCUDAGraphEnv) != "" {
		return r, nil
	}
	// Capture would be ENABLED from here: gate it behind the zerfoo#878
	// override before wiring up the capturer.
	if os.Getenv(UnsafeCaptureTrainingEnv) != "1" {
		return nil, ErrCaptureTrainingDisabled
	}
	r.gc = gc
	r.enabled = true
	return r, nil
}

// Enabled reports whether capture-replay is active (GraphCapturer engine
// and not disabled via ZERFOO_DISABLE_CUDA_GRAPH).
func (r *CaptureReplayRunner[T]) Enabled() bool { return r.enabled }

// CapturesPerformed returns the number of CUDA-graph captures performed.
func (r *CaptureReplayRunner[T]) CapturesPerformed() uint64 { return r.capturesPerformed }

// ReplaysPerformed returns the number of CUDA-graph launches performed.
func (r *CaptureReplayRunner[T]) ReplaysPerformed() uint64 { return r.replaysPerformed }

// Step runs one training step. Depending on runner state this is an eager
// walk (disabled or warmup), a capture (recording the walk into a CUDA
// graph, then launching it once), or a replay (one graph launch).
func (r *CaptureReplayRunner[T]) Step(
	ctx context.Context,
	g *graph.Graph[T],
	loss graph.Node[T],
	batch Batch[T],
) (T, error) {
	var zero T

	if !r.enabled {
		return r.strategy.ComputeGradients(ctx, g, loss, batch)
	}

	// Replay steady state: one graph launch re-executes the recorded
	// forward + loss + backward + gradient accumulation. ReplayGraph
	// synchronizes the stream, so the loss tensor holds this step's value.
	if r.captured {
		if err := r.gc.ReplayGraph(r.handle); err != nil {
			return zero, fmt.Errorf("training: capture-replay: replay: %w", err)
		}
		r.replaysPerformed++
		return r.lossTensor.Data()[0], nil
	}

	// Warmup: eager walks allocate the persistent gradient accumulators
	// (issue #850) and any lazily-initialized engine workspaces so the
	// capture region performs no foreign allocations.
	if r.stepsSeen < uint64(r.warmupSteps) {
		r.stepsSeen++
		return r.strategy.ComputeGradients(ctx, g, loss, batch)
	}

	// Capture: record the walk into a CUDA graph. Stream capture RECORDS
	// the kernels without executing them, so the loss readback must wait
	// until the graph has launched.
	if err := r.gc.BeginCapture(); err != nil {
		return zero, fmt.Errorf("training: capture-replay: begin capture: %w", err)
	}
	lossTensor, walkErr := r.strategy.ComputeGradientsTensor(ctx, g, loss, batch)
	handle, endErr := r.gc.EndCapture()
	if walkErr != nil {
		if endErr == nil {
			_ = r.gc.DestroyGraph(handle)
		}
		return zero, fmt.Errorf("training: capture-replay: captured walk: %w", walkErr)
	}
	if endErr != nil {
		return zero, fmt.Errorf("training: capture-replay: end capture: %w", endErr)
	}
	r.handle = handle
	r.lossTensor = lossTensor
	r.captured = true
	r.capturesPerformed++

	// ztensor#167 / ADR 007: the captured graph's kernels reference frozen
	// device addresses for every buffer the step touches, including the
	// save-for-backward intermediates that this capture step's own Backward
	// already unpinned (ADR 006 pins are released when the node's Backward
	// returns). Those addresses are then reissued by a later per-epoch
	// ResetPool + arena free-list reuse, and the replayed graph reads the
	// corrupted memory -- gradients collapse toward zero from the next step
	// (GB10 CrossAsset fold-0 0.6047 vs the CPU baseline 0.7257). Reserve the
	// captured graph's whole arena footprint for its replay lifetime by raising
	// the arena reset floor to the capture high-water. This mirrors the proven
	// decode-loop capture path (generate/generator.go onCaptured), which has
	// always done this and is therefore correct under capture-replay.
	if ap, ok := any(r.gc).(interface{ ArenaUsedBytes() int }); ok {
		if asf, ok2 := any(r.gc).(interface{ SetArenaResetFloor(int) }); ok2 {
			asf.SetArenaResetFloor(ap.ArenaUsedBytes())
		}
	}

	// The capture recorded this step but did not execute it: launch the
	// graph once so step `warmup` actually happens.
	if err := r.gc.ReplayGraph(r.handle); err != nil {
		return zero, fmt.Errorf("training: capture-replay: post-capture launch: %w", err)
	}
	r.replaysPerformed++
	return r.lossTensor.Data()[0], nil
}

// Close destroys the captured graph, if any. The runner must not be used
// after Close.
func (r *CaptureReplayRunner[T]) Close() error {
	if !r.captured {
		return nil
	}
	r.captured = false
	return r.gc.DestroyGraph(r.handle)
}
