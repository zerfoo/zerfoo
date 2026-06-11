package training

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ---------------------------------------------------------------------------
// Issue #850 regression tests: Parameter.Gradient must not be backed by arena
// memory across a per-sample arena reset (the Wolf crossasset schedule:
// forward+backward per sample, ResetPool per sample, optimizer step per
// batch). The harness below mirrors ztensor's host-backed cuda.ArenaPool
// poison-test pattern (TestSaveForBackward_WolfHazard_*) without importing
// ztensor internals: a bump allocator whose Reset fills unpinned spans with
// NaN -- the ZTENSOR_ARENA_POISON=1 semantics that named this bug on the GB10.
// ---------------------------------------------------------------------------

// testArena is a host-backed bump allocator with poison-on-reset and
// per-span pin refcounts.
type testArena struct {
	buf   []float32
	off   int
	spans []*arenaSpan
}

type arenaSpan struct {
	start, n int
	pins     int
}

func newTestArena(elems int) *testArena {
	return &testArena{buf: make([]float32, elems)}
}

func (a *testArena) alloc(tb testing.TB, n int) *arenaSpan {
	tb.Helper()
	if a.off+n > len(a.buf) {
		tb.Fatalf("testArena: out of memory (off=%d, n=%d, cap=%d)", a.off, n, len(a.buf))
	}
	s := &arenaSpan{start: a.off, n: n}
	a.off += n
	a.spans = append(a.spans, s)
	return s
}

// Reset rewinds the arena and poisons every unpinned span with NaN
// (ZTENSOR_ARENA_POISON semantics). Pinned spans raise the rewind floor,
// matching the arena's raise-the-floor Reset.
func (a *testArena) Reset() {
	nan := float32(math.NaN())
	floor := 0
	kept := a.spans[:0]
	for _, s := range a.spans {
		if s.pins > 0 {
			if end := s.start + s.n; end > floor {
				floor = end
			}
			kept = append(kept, s)
			continue
		}
		for i := s.start; i < s.start+s.n; i++ {
			a.buf[i] = nan
		}
	}
	a.spans = kept
	a.off = floor
}

// arenaStorage backs a tensor with a testArena span, implementing
// tensor.PinnableStorage exactly like ztensor's GPUStorage does for
// arena-backed device memory -- this is what arenaBackedStorage detects.
type arenaStorage struct {
	a    *testArena
	span *arenaSpan
}

func (s *arenaStorage) Len() int                { return s.span.n }
func (s *arenaStorage) Slice() []float32        { return s.a.buf[s.span.start : s.span.start+s.span.n] }
func (s *arenaStorage) Set(d []float32)         { copy(s.Slice(), d) }
func (s *arenaStorage) DeviceType() device.Type { return device.CPU }
func (s *arenaStorage) PinForBackward() bool    { s.span.pins++; return true }
func (s *arenaStorage) UnpinForBackward() {
	if s.span.pins > 0 {
		s.span.pins--
	}
}

var (
	_ tensor.Storage[float32] = (*arenaStorage)(nil)
	_ tensor.PinnableStorage  = (*arenaStorage)(nil)
)

// arenaBiasNode mirrors layers/core/bias.go's Backward: it OVERWRITES its
// parameter's Gradient with a freshly engine-allocated tensor every call --
// on the GPU engine that allocation comes from the arena, modeled here by
// testArena. With useArena=false it allocates plain GC-owned host tensors
// (the CPU engine behavior).
type arenaBiasNode struct {
	param        *graph.Parameter[float32]
	extra        *graph.Parameter[float32] // a parameter that never receives a gradient
	arena        *testArena
	useArena     bool
	tb           testing.TB
	lastAssigned *tensor.TensorNumeric[float32]
}

func (n *arenaBiasNode) OpType() string                     { return "ArenaBias" }
func (n *arenaBiasNode) Attributes() map[string]interface{} { return nil }
func (n *arenaBiasNode) OutputShape() []int                 { return n.param.Value.Shape() }

func (n *arenaBiasNode) Parameters() []*graph.Parameter[float32] {
	ps := []*graph.Parameter[float32]{n.param}
	if n.extra != nil {
		ps = append(ps, n.extra)
	}
	return ps
}

func (n *arenaBiasNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return inputs[0], nil
}

func (n *arenaBiasNode) Backward(_ context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	gradVals := outputGradient.Data()

	var g *tensor.TensorNumeric[float32]
	if n.useArena {
		span := n.arena.alloc(n.tb, len(gradVals))
		st := &arenaStorage{a: n.arena, span: span}
		var err error
		g, err = tensor.NewWithStorage([]int{len(gradVals)}, tensor.Storage[float32](st))
		if err != nil {
			return nil, err
		}
		copy(st.Slice(), gradVals)
	} else {
		var err error
		g, err = tensor.New([]int{len(gradVals)}, append([]float32(nil), gradVals...))
		if err != nil {
			return nil, err
		}
	}

	// The bug under test: layers assign the engine-op result (an arena
	// tensor on GPU) directly to the persistent Parameter.Gradient field.
	n.param.Gradient = g
	n.lastAssigned = g

	return []*tensor.TensorNumeric[float32]{outputGradient}, nil
}

// passthroughLoss returns a scalar loss and propagates the targets tensor as
// the initial gradient, so tests control per-sample gradients exactly.
type passthroughLoss struct{}

func (l *passthroughLoss) OpType() string                          { return "PassthroughLoss" }
func (l *passthroughLoss) Attributes() map[string]interface{}      { return nil }
func (l *passthroughLoss) OutputShape() []int                      { return []int{1} }
func (l *passthroughLoss) Parameters() []*graph.Parameter[float32] { return nil }

func (l *passthroughLoss) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return tensor.New([]int{1}, []float32{0})
}

func (l *passthroughLoss) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	// inputs are (modelOutput, targets); the targets tensor is the gradient.
	return []*tensor.TensorNumeric[float32]{inputs[1]}, nil
}

// wolfHazardFixture builds a 1-node graph around an arenaBiasNode.
func wolfHazardFixture(t *testing.T, useArena bool, arena *testArena) (*graph.Graph[float32], graph.Node[float32], *arenaBiasNode) {
	t.Helper()

	value, err := tensor.New([]int{2}, []float32{0, 0})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	param, err := graph.NewParameter("bias", value, tensor.New[float32])
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}

	extraValue, err := tensor.New([]int{2}, []float32{0, 0})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	extra, err := graph.NewParameter("never_used", extraValue, tensor.New[float32])
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	// A parameter the active graph never writes: the hook must tolerate a
	// nil gradient.
	extra.Gradient = nil

	node := &arenaBiasNode{param: param, extra: extra, arena: arena, useArena: useArena, tb: t}

	b := graph.NewBuilder[float32](nil)
	in := b.Input([]int{2})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return g, in, node
}

// runWolfHazardSchedule runs the Wolf crossasset schedule: per sample, one
// forward+backward through the strategy followed by an arena Reset; targets
// for sample k are [k+1, 10*(k+1)].
func runWolfHazardSchedule(t *testing.T, strategy GradientStrategy[float32], g *graph.Graph[float32], in graph.Node[float32], arena *testArena, samples int) {
	t.Helper()
	ctx := context.Background()
	loss := &passthroughLoss{}

	for k := 0; k < samples; k++ {
		input, err := tensor.New([]int{2}, []float32{1, 1})
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		targets, err := tensor.New([]int{2}, []float32{float32(k + 1), float32(10 * (k + 1))})
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		batch := Batch[float32]{
			Inputs:  map[graph.Node[float32]]*tensor.TensorNumeric[float32]{in: input},
			Targets: targets,
		}
		if _, err := strategy.ComputeGradients(ctx, g, loss, batch); err != nil {
			t.Fatalf("ComputeGradients (sample %d): %v", k, err)
		}
		// Wolf's per-sample ResetPool: recycle (and poison) the arena.
		arena.Reset()
	}
}

// analytic sum of per-sample grads for runWolfHazardSchedule with 3 samples:
// [1+2+3, 10+20+30] = [6, 60].
var wantSum3 = []float32{6, 60}

// TestPersistentGradAccum_WolfHazard_Fixed: WITH the hook, the accumulated
// gradient survives per-sample arena resets and equals the analytic sum of
// per-sample gradients.
func TestPersistentGradAccum_WolfHazard_Fixed(t *testing.T) {
	arena := newTestArena(1024)
	g, in, node := wolfHazardFixture(t, true, arena)
	strategy := NewDefaultBackpropStrategy[float32]()

	runWolfHazardSchedule(t, strategy, g, in, arena, 3)

	got := node.param.Gradient.Data()
	for i, v := range got {
		if math.IsNaN(float64(v)) {
			t.Fatalf("param.Gradient[%d] = NaN: accumulated gradient was poisoned by arena reset", i)
		}
		if v != wantSum3[i] {
			t.Fatalf("param.Gradient[%d] = %v, want %v (analytic sum of per-sample grads)", i, v, wantSum3[i])
		}
	}

	// The gradient now lives in the persistent accumulator, not the arena.
	if node.param.Gradient == node.lastAssigned {
		t.Fatal("param.Gradient still points at the layer-assigned arena tensor")
	}
	if arenaBackedStorage(node.param.Gradient) {
		t.Fatal("param.Gradient is still arena-backed after the hook")
	}
	// The nil-gradient parameter was tolerated and left untouched.
	if node.extra.Gradient != nil {
		t.Fatalf("nil-gradient parameter was modified: %v", node.extra.Gradient)
	}
	// Exactly one accumulator was allocated.
	if len(strategy.grads.accums) != 1 {
		t.Fatalf("len(accums) = %d, want 1", len(strategy.grads.accums))
	}
}

// TestPersistentGradAccum_WolfHazard_WithoutHookPoisoned asserts the OLD
// behavior is detectable: without the hook, the same schedule leaves
// Parameter.Gradient pointing at poisoned (NaN) arena memory. This is the
// exact failure ZTENSOR_ARENA_POISON named on the GB10 (issue #850).
func TestPersistentGradAccum_WolfHazard_WithoutHookPoisoned(t *testing.T) {
	arena := newTestArena(1024)
	g, in, node := wolfHazardFixture(t, true, arena)

	// noHookStrategy reproduces pre-fix behavior: computeGradientsCommon
	// with a nil accumulator.
	ctx := context.Background()
	loss := &passthroughLoss{}
	for k := 0; k < 3; k++ {
		input, _ := tensor.New([]int{2}, []float32{1, 1})
		targets, _ := tensor.New([]int{2}, []float32{float32(k + 1), float32(10 * (k + 1))})
		batch := Batch[float32]{
			Inputs:  map[graph.Node[float32]]*tensor.TensorNumeric[float32]{in: input},
			Targets: targets,
		}
		if _, err := computeGradientsCommon[float32](ctx, g, loss, batch, types.FullBackprop, nil); err != nil {
			t.Fatalf("computeGradientsCommon (sample %d): %v", k, err)
		}
		arena.Reset()
	}

	got := node.param.Gradient.Data()
	for i, v := range got {
		if !math.IsNaN(float64(v)) {
			t.Fatalf("param.Gradient[%d] = %v, want NaN: without the hook the gradient must read poisoned arena memory", i, v)
		}
	}
}

// TestPersistentGradAccum_WithEngine: the engine fast path (in-place
// engine.Add with dst=accumulator) produces the same analytic sum and never
// swaps the accumulator's storage.
func TestPersistentGradAccum_WithEngine(t *testing.T) {
	arena := newTestArena(1024)
	g, in, node := wolfHazardFixture(t, true, arena)
	strategy := NewDefaultBackpropStrategy[float32]()
	strategy.SetEngine(compute.NewCPUEngine[float32](numeric.Float32Ops{}))

	runWolfHazardSchedule(t, strategy, g, in, arena, 3)

	got := node.param.Gradient.Data()
	for i, v := range got {
		if v != wantSum3[i] {
			t.Fatalf("param.Gradient[%d] = %v, want %v", i, v, wantSum3[i])
		}
	}
	accum := strategy.grads.accums[node.param]
	if accum == nil {
		t.Fatal("no accumulator allocated")
	}
	if node.param.Gradient != accum {
		t.Fatal("param.Gradient does not point at the persistent accumulator")
	}
}

// TestEngineAdd_PreservesAccumulatorStorageIdentity: engine.Add with
// dst=accumulator must write IN PLACE into the provided dst storage and not
// swap it -- the property the hook's fast path depends on.
func TestEngineAdd_PreservesAccumulatorStorageIdentity(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	accum, err := tensor.New([]int{3}, []float32{1, 2, 3})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	grad, err := tensor.New([]int{3}, []float32{10, 20, 30})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	before := accum.GetStorage()
	res, err := engine.Add(ctx, accum, grad, accum)
	if err != nil {
		t.Fatalf("engine.Add: %v", err)
	}
	if res != accum {
		t.Fatal("engine.Add returned a different tensor than dst")
	}
	if res.GetStorage() != before {
		t.Fatal("engine.Add swapped dst's storage; accumulation requires in-place writes")
	}
	want := []float32{11, 22, 33}
	for i, v := range res.Data() {
		if v != want[i] {
			t.Fatalf("res[%d] = %v, want %v", i, v, want[i])
		}
	}

	// Same property through the hook's addInto.
	acc := &gradAccumulator[float32]{engine: engine}
	if err := acc.addInto(ctx, accum, grad, "p"); err != nil {
		t.Fatalf("addInto: %v", err)
	}
	if accum.GetStorage() != before {
		t.Fatal("addInto swapped the accumulator's storage")
	}
	want = []float32{21, 42, 63}
	for i, v := range accum.Data() {
		if v != want[i] {
			t.Fatalf("accum[%d] = %v, want %v", i, v, want[i])
		}
	}
}

// TestPersistentGradAccum_CPUNoOp: on a pure CPU path (no arena-backed
// storage anywhere) the hook is a no-op -- no accumulator is allocated and
// Parameter.Gradient is exactly the tensor the layer assigned, preserving
// pre-fix CPU semantics byte for byte.
func TestPersistentGradAccum_CPUNoOp(t *testing.T) {
	g, in, node := wolfHazardFixture(t, false, nil)
	strategy := NewDefaultBackpropStrategy[float32]()

	ctx := context.Background()
	loss := &passthroughLoss{}
	for k := 0; k < 2; k++ {
		input, _ := tensor.New([]int{2}, []float32{1, 1})
		targets, _ := tensor.New([]int{2}, []float32{float32(k + 1), float32(10 * (k + 1))})
		batch := Batch[float32]{
			Inputs:  map[graph.Node[float32]]*tensor.TensorNumeric[float32]{in: input},
			Targets: targets,
		}
		if _, err := strategy.ComputeGradients(ctx, g, loss, batch); err != nil {
			t.Fatalf("ComputeGradients (sample %d): %v", k, err)
		}
	}

	if len(strategy.grads.accums) != 0 {
		t.Fatalf("len(accums) = %d, want 0: CPU gradients must not allocate accumulators", len(strategy.grads.accums))
	}
	if node.param.Gradient != node.lastAssigned {
		t.Fatal("param.Gradient was repointed on the CPU path; hook must be a no-op")
	}
	// CPU overwrite semantics preserved: last sample's gradient, not a sum.
	want := []float32{2, 20}
	for i, v := range node.param.Gradient.Data() {
		if v != want[i] {
			t.Fatalf("param.Gradient[%d] = %v, want %v (unchanged CPU overwrite semantics)", i, v, want[i])
		}
	}
}

// TestPersistentGradAccum_OptimizerZeroReuse: the optimizer zeroes
// Parameter.Gradient in place after Step (zerfoo#845 stepMixedV zeroes via
// storage Set). The hook must reuse the SAME persistent buffer for the next
// batch, and the next batch's accumulated value must not include the
// previous batch.
func TestPersistentGradAccum_OptimizerZeroReuse(t *testing.T) {
	arena := newTestArena(4096)
	g, in, node := wolfHazardFixture(t, true, arena)
	strategy := NewDefaultBackpropStrategy[float32]()

	// Batch 1: 3 samples.
	runWolfHazardSchedule(t, strategy, g, in, arena, 3)

	accum := node.param.Gradient
	storageBefore := accum.GetStorage()

	// Optimizer end-of-batch: zero the gradient IN PLACE via storage Set
	// (the zerfoo#845 contract).
	zeroed := accum.Data()
	for i := range zeroed {
		zeroed[i] = 0
	}
	accum.GetStorage().Set(zeroed)

	// Batch 2: same 3-sample schedule.
	runWolfHazardSchedule(t, strategy, g, in, arena, 3)

	if node.param.Gradient != accum {
		t.Fatal("hook allocated a new accumulator instead of reusing the zeroed persistent buffer")
	}
	if node.param.Gradient.GetStorage() != storageBefore {
		t.Fatal("accumulator storage changed across batches")
	}
	for i, v := range node.param.Gradient.Data() {
		if v != wantSum3[i] {
			t.Fatalf("batch-2 param.Gradient[%d] = %v, want %v (zeroed accumulator must restart the sum)", i, v, wantSum3[i])
		}
	}
}

// TestPersistentGradAccum_InPlaceLayerSkipped: a layer (or optimizer) that
// writes in place into the persistent buffer the hook installed must not be
// double-counted: the hook skips when Parameter.Gradient already IS the
// accumulator.
func TestPersistentGradAccum_InPlaceLayerSkipped(t *testing.T) {
	ctx := context.Background()
	acc := &gradAccumulator[float32]{}

	arena := newTestArena(64)
	span := arena.alloc(t, 2)
	st := &arenaStorage{a: arena, span: span}
	gradT, err := tensor.NewWithStorage([]int{2}, tensor.Storage[float32](st))
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}
	copy(st.Slice(), []float32{3, 4})

	value, _ := tensor.New([]int{2}, []float32{0, 0})
	p, err := graph.NewParameter("w", value, tensor.New[float32])
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	p.Gradient = gradT

	node := &arenaBiasNode{param: p}
	b := graph.NewBuilder[float32](nil)
	in := b.Input([]int{2})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	// First capture: migrates the arena gradient into a fresh accumulator.
	if err := acc.capture(ctx, g); err != nil {
		t.Fatalf("capture: %v", err)
	}
	accum := acc.accums[p]
	if accum == nil || p.Gradient != accum {
		t.Fatal("first capture did not install the accumulator")
	}

	// An in-place writer (AddGradient-style layer, optimizer clip with dst)
	// mutates the accumulator directly; p.Gradient still IS the accumulator.
	if err := p.AddGradient(mustNewTensor(t, []float32{1, 1})); err != nil {
		t.Fatalf("AddGradient: %v", err)
	}

	// Second capture must not double-count or reallocate.
	if err := acc.capture(ctx, g); err != nil {
		t.Fatalf("capture: %v", err)
	}
	if p.Gradient != accum {
		t.Fatal("second capture replaced the accumulator")
	}
	want := []float32{4, 5}
	for i, v := range p.Gradient.Data() {
		if v != want[i] {
			t.Fatalf("p.Gradient[%d] = %v, want %v (in-place writes must not be re-accumulated)", i, v, want[i])
		}
	}
}

func mustNewTensor(t *testing.T, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tt, err := tensor.New([]int{len(data)}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return tt
}
