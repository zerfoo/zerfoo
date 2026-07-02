package optimizer

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestSWA_InterfaceAssertion(t *testing.T) {
	var _ Optimizer[float32] = (*SWA[float32])(nil)
}

func TestSWA_Step_DelegatesToInner(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	inner := &setOptimizer[float32]{value: 42.0}
	swa := NewSWA[float32](inner, engine, 0)

	value, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])

	if err := swa.Step(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatalf("Step: %v", err)
	}

	for i, v := range param.Value.Data() {
		if v != 42.0 {
			t.Errorf("param[%d] = %f, want 42.0", i, v)
		}
	}
}

func TestSWA_UpdateAverage_SkipsBeforeStartEpoch(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	inner := &noopOptimizer[float32]{}
	swa := NewSWA[float32](inner, engine, 5)

	value, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])

	for epoch := 0; epoch < 5; epoch++ {
		if err := swa.UpdateAverage(ctx, []*graph.Parameter[float32]{param}, epoch); err != nil {
			t.Fatalf("UpdateAverage epoch %d: %v", epoch, err)
		}
	}

	if swa.NAveraged() != 0 {
		t.Errorf("nAveraged = %d, want 0", swa.NAveraged())
	}
}

func TestSWA_UpdateAverage_ComputesRunningMean(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	inner := &noopOptimizer[float32]{}
	swa := NewSWA[float32](inner, engine, 0)

	value, _ := tensor.New[float32]([]int{1}, []float32{2.0})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])

	// Epoch 0: value=2.0 -> avg=2.0 (first init)
	if err := swa.UpdateAverage(ctx, []*graph.Parameter[float32]{param}, 0); err != nil {
		t.Fatalf("UpdateAverage epoch 0: %v", err)
	}

	// Epoch 1: value=4.0 -> avg = 2.0 + (4.0-2.0)/2 = 3.0
	param.Value.SetData([]float32{4.0})
	if err := swa.UpdateAverage(ctx, []*graph.Parameter[float32]{param}, 1); err != nil {
		t.Fatalf("UpdateAverage epoch 1: %v", err)
	}

	// Epoch 2: value=6.0 -> avg = 3.0 + (6.0-3.0)/3 = 4.0
	param.Value.SetData([]float32{6.0})
	if err := swa.UpdateAverage(ctx, []*graph.Parameter[float32]{param}, 2); err != nil {
		t.Fatalf("UpdateAverage epoch 2: %v", err)
	}

	// Expected: mean(2, 4, 6) = 4.0
	avg := swa.avgParams[param].Data()[0]
	if math.Abs(float64(avg)-4.0) > 1e-5 {
		t.Errorf("avg = %f, want 4.0", avg)
	}
	if swa.NAveraged() != 3 {
		t.Errorf("nAveraged = %d, want 3", swa.NAveraged())
	}
}

func TestSWA_SwapWeights_ExchangesParams(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	inner := &noopOptimizer[float32]{}
	swa := NewSWA[float32](inner, engine, 0)

	value, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])

	// Initialize avg
	if err := swa.UpdateAverage(ctx, []*graph.Parameter[float32]{param}, 0); err != nil {
		t.Fatalf("UpdateAverage: %v", err)
	}

	// Manually set avg to different values
	swa.avgParams[param].SetData([]float32{5.0, 6.0})

	origValue := make([]float32, 2)
	copy(origValue, param.Value.Data())

	// Swap: param should get avg values
	if err := swa.SwapWeights(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatalf("SwapWeights: %v", err)
	}
	if param.Value.Data()[0] != 5.0 || param.Value.Data()[1] != 6.0 {
		t.Errorf("after swap: param = %v, want [5.0, 6.0]", param.Value.Data())
	}

	// Double swap: should restore original values
	if err := swa.SwapWeights(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatalf("SwapWeights back: %v", err)
	}
	for i, v := range param.Value.Data() {
		if math.Abs(float64(v-origValue[i])) > 1e-6 {
			t.Errorf("after double swap: param[%d] = %f, want %f", i, v, origValue[i])
		}
	}
}

func TestSWA_NAveraged_Increments(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	inner := &noopOptimizer[float32]{}
	swa := NewSWA[float32](inner, engine, 0)

	value, _ := tensor.New[float32]([]int{1}, []float32{1.0})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])

	for i := 0; i < 5; i++ {
		if got := swa.NAveraged(); got != i {
			t.Errorf("before epoch %d: NAveraged = %d, want %d", i, got, i)
		}
		if err := swa.UpdateAverage(ctx, []*graph.Parameter[float32]{param}, i); err != nil {
			t.Fatalf("UpdateAverage epoch %d: %v", i, err)
		}
	}
	if got := swa.NAveraged(); got != 5 {
		t.Errorf("final NAveraged = %d, want 5", got)
	}
}
