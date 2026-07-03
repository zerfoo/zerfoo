package training

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// fakeCaptureEngine wraps a real CPU engine (every compute op delegates to
// it) and implements compute.GraphCapturer with instrumented stubs, so the
// CaptureReplayRunner state machine can be exercised without CUDA hardware.
type fakeCaptureEngine struct {
	compute.Engine[float32]

	beginCalls   int
	endCalls     int
	replayCalls  int
	destroyCalls int

	capturing bool
}

func newFakeCaptureEngine() *fakeCaptureEngine {
	return &fakeCaptureEngine{
		Engine: compute.NewCPUEngine[float32](numeric.Float32Ops{}),
	}
}

func (f *fakeCaptureEngine) BeginCapture() error {
	f.beginCalls++
	f.capturing = true
	return nil
}

func (f *fakeCaptureEngine) EndCapture() (compute.GraphHandle, error) {
	f.endCalls++
	f.capturing = false
	return compute.GraphHandle{}, nil
}

func (f *fakeCaptureEngine) ReplayGraph(_ compute.GraphHandle) error {
	f.replayCalls++
	return nil
}

func (f *fakeCaptureEngine) DestroyGraph(_ compute.GraphHandle) error {
	f.destroyCalls++
	return nil
}

var _ compute.GraphCapturer = (*fakeCaptureEngine)(nil)

// captureFixture builds a minimal trainable graph (the grad_accum bias
// fixture without an arena) plus a stable batch, mirroring the runner's
// stable-operand contract.
func captureFixture(t *testing.T) (*graph.Graph[float32], graph.Node[float32], Batch[float32]) {
	t.Helper()
	g, in, _ := wolfHazardFixture(t, false, nil)
	input, err := tensor.New([]int{2}, []float32{1, 1})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	targets, err := tensor.New([]int{2}, []float32{1, 10})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return g, in, Batch[float32]{
		Inputs:  map[graph.Node[float32]]*tensor.TensorNumeric[float32]{in: input},
		Targets: targets,
	}
}

// TestCaptureReplayRunner_CounterSchedule is the S9.1.1 acceptance test:
// for an N-step run with W warmup steps, capturesPerformed must equal 1 and
// replaysPerformed must equal N-W, with warmup steps counted in neither.
func TestCaptureReplayRunner_CounterSchedule(t *testing.T) {
	tests := []struct {
		name         string
		warmup       int
		steps        int
		wantCaptures uint64
		wantReplays  uint64
		wantBegin    int
	}{
		{name: "default warmup 1", warmup: 1, steps: 5, wantCaptures: 1, wantReplays: 4, wantBegin: 1},
		{name: "warmup 2", warmup: 2, steps: 6, wantCaptures: 1, wantReplays: 4, wantBegin: 1},
		{name: "warmup clamped from 0", warmup: 0, steps: 3, wantCaptures: 1, wantReplays: 2, wantBegin: 1},
		{name: "all warmup no capture", warmup: 4, steps: 3, wantCaptures: 0, wantReplays: 0, wantBegin: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newFakeCaptureEngine()
			strategy := NewDefaultBackpropStrategy[float32]()
			runner, err := NewCaptureReplayRunner(strategy, eng, tt.warmup)
			if err != nil {
				t.Fatalf("NewCaptureReplayRunner: %v", err)
			}
			if !runner.Enabled() {
				t.Fatal("runner should be enabled with a GraphCapturer engine")
			}

			g, _, batch := captureFixture(t)
			loss := &passthroughLoss{}
			ctx := context.Background()
			for i := 0; i < tt.steps; i++ {
				if _, err := runner.Step(ctx, g, loss, batch); err != nil {
					t.Fatalf("Step %d: %v", i, err)
				}
			}

			if got := runner.CapturesPerformed(); got != tt.wantCaptures {
				t.Errorf("CapturesPerformed = %d, want %d", got, tt.wantCaptures)
			}
			if got := runner.ReplaysPerformed(); got != tt.wantReplays {
				t.Errorf("ReplaysPerformed = %d, want %d", got, tt.wantReplays)
			}
			if eng.beginCalls != tt.wantBegin {
				t.Errorf("BeginCapture calls = %d, want %d", eng.beginCalls, tt.wantBegin)
			}
			if eng.endCalls != tt.wantBegin {
				t.Errorf("EndCapture calls = %d, want %d", eng.endCalls, tt.wantBegin)
			}
			if uint64(eng.replayCalls) != tt.wantReplays {
				t.Errorf("ReplayGraph calls = %d, want %d", eng.replayCalls, tt.wantReplays)
			}
			if eng.capturing {
				t.Error("engine left in capturing state")
			}
		})
	}
}

// TestCaptureReplayRunner_DisabledByEnv: ZERFOO_DISABLE_CUDA_GRAPH is a
// construction-time toggle -- the runner becomes a passthrough and never
// touches the capture API.
func TestCaptureReplayRunner_DisabledByEnv(t *testing.T) {
	t.Setenv(DisableCUDAGraphEnv, "1")

	eng := newFakeCaptureEngine()
	strategy := NewDefaultBackpropStrategy[float32]()
	runner, err := NewCaptureReplayRunner(strategy, eng, 1)
	if err != nil {
		t.Fatalf("NewCaptureReplayRunner: %v", err)
	}
	if runner.Enabled() {
		t.Fatal("runner must be disabled when ZERFOO_DISABLE_CUDA_GRAPH is set")
	}

	g, _, batch := captureFixture(t)
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		if _, err := runner.Step(ctx, g, &passthroughLoss{}, batch); err != nil {
			t.Fatalf("Step %d: %v", i, err)
		}
	}
	if runner.CapturesPerformed() != 0 || runner.ReplaysPerformed() != 0 {
		t.Errorf("counters = %d/%d, want 0/0 when disabled",
			runner.CapturesPerformed(), runner.ReplaysPerformed())
	}
	if eng.beginCalls != 0 || eng.replayCalls != 0 {
		t.Errorf("capture API touched while disabled: begin=%d replay=%d",
			eng.beginCalls, eng.replayCalls)
	}
}

// TestCaptureReplayRunner_NonCapturerEngine: a plain engine (no
// GraphCapturer) yields a disabled passthrough runner.
func TestCaptureReplayRunner_NonCapturerEngine(t *testing.T) {
	cpu := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	strategy := NewDefaultBackpropStrategy[float32]()
	runner, err := NewCaptureReplayRunner[float32](strategy, cpu, 1)
	if err != nil {
		t.Fatalf("NewCaptureReplayRunner: %v", err)
	}
	if runner.Enabled() {
		t.Fatal("runner must be disabled for engines without GraphCapturer")
	}

	g, _, batch := captureFixture(t)
	if _, err := runner.Step(context.Background(), g, &passthroughLoss{}, batch); err != nil {
		t.Fatalf("Step: %v", err)
	}
	if runner.CapturesPerformed() != 0 || runner.ReplaysPerformed() != 0 {
		t.Errorf("counters = %d/%d, want 0/0 for non-capturer engine",
			runner.CapturesPerformed(), runner.ReplaysPerformed())
	}
}

// TestCaptureReplayRunner_Close destroys the captured graph exactly once.
func TestCaptureReplayRunner_Close(t *testing.T) {
	eng := newFakeCaptureEngine()
	strategy := NewDefaultBackpropStrategy[float32]()
	runner, err := NewCaptureReplayRunner(strategy, eng, 1)
	if err != nil {
		t.Fatalf("NewCaptureReplayRunner: %v", err)
	}

	g, _, batch := captureFixture(t)
	ctx := context.Background()
	for i := 0; i < 3; i++ {
		if _, err := runner.Step(ctx, g, &passthroughLoss{}, batch); err != nil {
			t.Fatalf("Step %d: %v", i, err)
		}
	}
	if err := runner.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := runner.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if eng.destroyCalls != 1 {
		t.Errorf("DestroyGraph calls = %d, want 1 (Close must be idempotent)", eng.destroyCalls)
	}
}

// TestCaptureReplayRunner_CaptureEnabledNoOverrideNeeded is the T133.4
// regression test for the removed containment gate: a GraphCapturer engine
// with ZERFOO_DISABLE_CUDA_GRAPH unset builds a capture-enabled runner with
// no override env var required (the ZERFOO_UNSAFE_CAPTURE_TRAINING gate from
// T129.2 is gone now that zerfoo#878's root cause is fixed, T133.3).
func TestCaptureReplayRunner_CaptureEnabledNoOverrideNeeded(t *testing.T) {
	eng := newFakeCaptureEngine()
	strategy := NewDefaultBackpropStrategy[float32]()
	runner, err := NewCaptureReplayRunner(strategy, eng, 1)
	if err != nil {
		t.Fatalf("NewCaptureReplayRunner: %v", err)
	}
	if !runner.Enabled() {
		t.Fatal("runner must be capture-enabled for a GraphCapturer engine with no env overrides set")
	}

	g, _, batch := captureFixture(t)
	ctx := context.Background()
	const steps = 4
	for i := 0; i < steps; i++ {
		if _, serr := runner.Step(ctx, g, &passthroughLoss{}, batch); serr != nil {
			t.Fatalf("Step %d: %v", i, serr)
		}
	}
	if got := runner.CapturesPerformed(); got != 1 {
		t.Errorf("CapturesPerformed = %d, want 1", got)
	}
	if got := runner.ReplaysPerformed(); got != uint64(steps-1) {
		t.Errorf("ReplaysPerformed = %d, want %d", got, steps-1)
	}
}
