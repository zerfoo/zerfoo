package training

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// Bug 11 follow-up: engineFor selects how the persistent accumulator adds a
// gradient. An explicitly configured engine (SetEngine) always wins; without
// one, only the fully device-resident f32 case may derive the graph's own
// engine -- every host-backed case must take the host fallback (nil), which
// is the behavior all existing CPU-path tests rely on. The positive
// GPU-derivation branch requires *tensor.GPUStorage and is exercised on real
// CUDA hardware (the Wolf verify schedule); off-GPU it is covered by the
// f32+GPUStorage gating below returning nil for everything else.
func TestGradAccumEngineFor(t *testing.T) {
	t.Parallel()

	cpuEngine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	newHostTensor := func(t *testing.T) *tensor.TensorNumeric[float32] {
		t.Helper()
		tn, err := tensor.New([]int{2}, []float32{1, 2})
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		return tn
	}

	buildGraph := func(t *testing.T, eng compute.Engine[float32]) *graph.Graph[float32] {
		t.Helper()
		b := graph.NewBuilder[float32](eng)
		in := b.Input([]int{2})
		g, err := b.Build(in)
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		return g
	}

	t.Run("explicit engine wins", func(t *testing.T) {
		t.Parallel()
		a := &gradAccumulator[float32]{}
		a.setEngine(cpuEngine)
		g := buildGraph(t, nil)
		got := a.engineFor(g, newHostTensor(t), newHostTensor(t))
		if got != compute.Engine[float32](cpuEngine) {
			t.Fatalf("engineFor = %v, want the explicitly configured engine", got)
		}
	})

	t.Run("host-backed tensors never derive the graph engine", func(t *testing.T) {
		t.Parallel()
		a := &gradAccumulator[float32]{}
		g := buildGraph(t, cpuEngine)
		if got := a.engineFor(g, newHostTensor(t), newHostTensor(t)); got != nil {
			t.Fatalf("engineFor = %v, want nil (host fallback) for host-backed tensors", got)
		}
	})

	t.Run("non-f32 never derives the graph engine", func(t *testing.T) {
		t.Parallel()
		eng64 := compute.NewCPUEngine[float64](numeric.Float64Ops{})
		b := graph.NewBuilder[float64](eng64)
		in := b.Input([]int{2})
		g, err := b.Build(in)
		if err != nil {
			t.Fatalf("Build: %v", err)
		}
		tn, err := tensor.New([]int{2}, []float64{1, 2})
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		a := &gradAccumulator[float64]{}
		if got := a.engineFor(g, tn, tn); got != nil {
			t.Fatalf("engineFor = %v, want nil (host fallback) for float64", got)
		}
	})

	t.Run("nil graph engine derives nothing", func(t *testing.T) {
		t.Parallel()
		a := &gradAccumulator[float32]{}
		g := buildGraph(t, nil)
		if got := a.engineFor(g, newHostTensor(t), newHostTensor(t)); got != nil {
			t.Fatalf("engineFor = %v, want nil when the graph has no engine", got)
		}
	})
}
