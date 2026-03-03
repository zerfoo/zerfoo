package gather

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// errIntEngine wraps a real int engine and injects errors on specific calls.
// We use an int engine so that indices can be passed directly as inputs (same type T=int).
type errIntEngine struct {
	compute.Engine[int]
	calls   map[string]int
	failOn  map[string]int
	failErr error
}

func newErrIntEngine(failOn map[string]int) *errIntEngine {
	return &errIntEngine{
		Engine:  compute.NewCPUEngine[int](numeric.IntOps{}),
		calls:   make(map[string]int),
		failOn:  failOn,
		failErr: fmt.Errorf("injected error"),
	}
}

func (e *errIntEngine) check(op string) error {
	e.calls[op]++
	if n, ok := e.failOn[op]; ok && e.calls[op] >= n {
		return e.failErr
	}
	return nil
}

func (e *errIntEngine) ScatterAdd(ctx context.Context, table *tensor.TensorNumeric[int], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[int]) error {
	if err := e.check("ScatterAdd"); err != nil {
		return err
	}
	return e.Engine.ScatterAdd(ctx, table, indices, dOut)
}

func (e *errIntEngine) Gather(ctx context.Context, params *tensor.TensorNumeric[int], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[int]) error {
	if err := e.check("Gather"); err != nil {
		return err
	}
	return e.Engine.Gather(ctx, params, indices, output)
}

// ---------- Backward ScatterAdd error ----------

func TestBackward_ScatterAddError(t *testing.T) {
	eng := newErrIntEngine(map[string]int{"ScatterAdd": 1})
	g := New[int](eng)
	ctx := context.Background()

	params, _ := tensor.New[int]([]int{4, 3}, nil)
	indices, _ := tensor.New[int]([]int{1, 2}, []int{0, 2})
	dOut, _ := tensor.New[int]([]int{2, 3}, make([]int, 6))

	_, err := g.Backward(ctx, types.FullBackprop, dOut, params, indices)
	if err == nil {
		t.Error("expected error from ScatterAdd")
	}
}

// ---------- Forward engine.Gather error ----------

func TestForward_GatherError(t *testing.T) {
	eng := newErrIntEngine(map[string]int{"Gather": 1})
	g := New[int](eng)
	ctx := context.Background()

	params, _ := tensor.New[int]([]int{4, 3}, nil)
	indices, _ := tensor.New[int]([]int{1, 2}, []int{0, 2})

	_, err := g.Forward(ctx, params, indices)
	if err == nil {
		t.Error("expected error from engine.Gather")
	}
}

// ---------- BuildGather with weight patterns ----------

func TestBuildGather_NoWeights(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// No parameters at all -> creates dummy tensor
	node, err := BuildGather(eng, ops, "layer/Gather", nil, nil)
	if err != nil {
		t.Fatalf("BuildGather failed: %v", err)
	}
	g, ok := node.(*Gather[float32])
	if !ok {
		t.Fatal("expected *Gather[float32]")
	}
	if !g.HasEmbeddedWeights() {
		t.Error("expected dummy embedded weights")
	}
}

func TestBuildGather_WithNamedWeight(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	wTensor, _ := tensor.New[float32]([]int{10, 4}, nil)
	param, _ := graph.NewParameter("layer/Gather.weight", wTensor, tensor.New[float32])

	params := map[string]*graph.Parameter[float32]{
		"layer/Gather.weight": param,
	}

	node, err := BuildGather(eng, ops, "layer/Gather", params, nil)
	if err != nil {
		t.Fatalf("BuildGather failed: %v", err)
	}
	g := node.(*Gather[float32])
	if !g.HasEmbeddedWeights() {
		t.Error("expected embedded weights from .weight pattern")
	}
}

func TestBuildGather_WithGenericWeightKey(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	wTensor, _ := tensor.New[float32]([]int{8, 3}, nil)
	param, _ := graph.NewParameter("some_weight_param", wTensor, tensor.New[float32])

	// Key contains "weight" -> matched by the generic contains check
	params := map[string]*graph.Parameter[float32]{
		"some_weight_param": param,
	}

	node, err := BuildGather(eng, ops, "other_name", params, nil)
	if err != nil {
		t.Fatalf("BuildGather failed: %v", err)
	}
	g := node.(*Gather[float32])
	if !g.HasEmbeddedWeights() {
		t.Error("expected embedded weights from generic weight key")
	}
}
