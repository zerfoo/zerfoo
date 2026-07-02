package training

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// arenaAddStyleNode mirrors layers/core/linear.go's gradient accumulation.
//
// FIXED (dst-form) semantics, as converted in this PR:
//
//	p.Gradient, err = engine.Add(ctx, p.Gradient, dw, p.Gradient)
//
// modeled engine behavior: the FIRST call sees the host-initialized gradient
// as dst, and the GPU engine re-homes the result into the arena (fresh arena
// tensor holding old+delta); once the capture hook has installed its
// persistent accumulator, dst is stable non-arena storage and the add is in
// place, preserving storage identity.
//
// LEGACY (no-dst) semantics, the pre-fix bug:
//
//	p.Gradient, err = engine.Add(ctx, p.Gradient, dw)
//
// every call allocates a fresh arena tensor holding old+delta. Combined with
// the capture hook (which then re-adds that history into the accumulator)
// this double-counts: accum becomes 2*accum + delta per sample and explodes
// exponentially -- the Inf observed on the GB10 (zerfoo#850 follow-up).
type arenaAddStyleNode struct {
	param   *graph.Parameter[float32]
	arena   *testArena
	tb      testing.TB
	dstForm bool // true = fixed in-place form; false = legacy no-dst form
	calls   int
}

func (n *arenaAddStyleNode) OpType() string                     { return "ArenaAddStyle" }
func (n *arenaAddStyleNode) Attributes() map[string]interface{} { return nil }
func (n *arenaAddStyleNode) OutputShape() []int                 { return n.param.Value.Shape() }
func (n *arenaAddStyleNode) Parameters() []*graph.Parameter[float32] {
	return []*graph.Parameter[float32]{n.param}
}

func (n *arenaAddStyleNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return inputs[0], nil
}

func (n *arenaAddStyleNode) Backward(_ context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	n.calls++
	delta := outputGradient.Data()
	old := n.param.Gradient.Data() // host copy or backing slice; values only

	if n.dstForm && n.calls > 1 {
		// In-place add into the existing (hook-installed accumulator)
		// storage: storage identity preserved, no arena allocation.
		dst := n.param.Gradient.Data()
		for i := range dst {
			dst[i] = old[i] + delta[i]
		}
		return []*tensor.TensorNumeric[float32]{outputGradient}, nil
	}

	// Re-homed / legacy path: a fresh arena tensor holding old+delta.
	span := n.arena.alloc(n.tb, len(delta))
	st := &arenaStorage{a: n.arena, span: span}
	g, err := tensor.NewWithStorage([]int{len(delta)}, tensor.Storage[float32](st))
	if err != nil {
		return nil, err
	}
	buf := st.Slice()
	for i := range buf {
		buf[i] = old[i] + delta[i]
	}
	n.param.Gradient = g
	return []*tensor.TensorNumeric[float32]{outputGradient}, nil
}

func addStyleFixture(t *testing.T, arena *testArena, dstForm bool) (*graph.Graph[float32], graph.Node[float32], *arenaAddStyleNode) {
	t.Helper()
	value, err := tensor.New([]int{2}, []float32{0, 0})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	param, err := graph.NewParameter("linear_w", value, tensor.New[float32])
	if err != nil {
		t.Fatalf("NewParameter: %v", err)
	}
	node := &arenaAddStyleNode{param: param, arena: arena, tb: t, dstForm: dstForm}
	b := graph.NewBuilder[float32](nil)
	in := b.Input([]int{2})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	return g, in, node
}

// TestGradAccum_AddStyleDstForm_NoDoubleCount proves the linear.go dst-form
// conversion: an add-style layer under the Wolf hazard schedule (per-sample
// arena Reset) accumulates to exactly the analytic per-sample sum -- no
// double-count, no poison.
func TestGradAccum_AddStyleDstForm_NoDoubleCount(t *testing.T) {
	arena := newTestArena(64)
	g, in, node := addStyleFixture(t, arena, true)
	strategy := NewDefaultBackpropStrategy[float32]()
	runWolfHazardSchedule(t, strategy, g, in, arena, 3)

	got := node.param.Gradient.Data()
	want := []float32{6, 60} // [1+2+3, 10+20+30]
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("accumulated gradient[%d] = %v, want %v (full: %v)", i, got[i], want[i], got)
		}
	}
}

// TestGradAccum_AddStyleLegacyNoDst_DoubleCounts documents WHY the dst-form
// conversion is required: the legacy no-dst pattern hands the capture hook a
// fresh arena tensor already containing the accumulated history, which the
// hook re-adds -- the result diverges from the analytic sum (exponentially in
// the sample count; Inf on real batch sizes).
func TestGradAccum_AddStyleLegacyNoDst_DoubleCounts(t *testing.T) {
	arena := newTestArena(64)
	g, in, node := addStyleFixture(t, arena, false)
	strategy := NewDefaultBackpropStrategy[float32]()
	runWolfHazardSchedule(t, strategy, g, in, arena, 3)

	got := node.param.Gradient.Data()
	want := []float32{6, 60}
	if got[0] == want[0] && got[1] == want[1] {
		t.Fatalf("legacy no-dst pattern unexpectedly produced the analytic sum %v -- the double-count this test documents has been silently fixed elsewhere; update the test", got)
	}
}
