package optimizer

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestEMA_InterfaceAssertion(t *testing.T) {
	var _ Optimizer[float32] = (*EMA[float32])(nil)
}

func TestEMA_Step_InitializesShadow(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Use a no-op inner optimizer
	inner := &noopOptimizer[float32]{}
	ema := NewEMA[float32](inner, engine, 0.99)

	value, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])
	// Set a gradient so inner optimizer doesn't skip
	grad, _ := tensor.New[float32]([]int{2}, []float32{0.0, 0.0})
	param.Gradient = grad

	err := ema.Step(ctx, []*graph.Parameter[float32]{param})
	if err != nil {
		t.Fatalf("Step: %v", err)
	}

	// Shadow should exist and be a copy of param value
	shadow, ok := ema.shadow[param]
	if !ok {
		t.Fatal("shadow not initialized")
	}
	for i, v := range shadow.Data() {
		if v != param.Value.Data()[i] {
			t.Errorf("shadow[%d] = %f, want %f", i, v, param.Value.Data()[i])
		}
	}
}

func TestEMA_Step_ShadowDiverges(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Inner optimizer that sets params to constant values
	inner := &setOptimizer[float32]{value: 10.0}
	ema := NewEMA[float32](inner, engine, 0.9)

	value, _ := tensor.New[float32]([]int{1}, []float32{0.0})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])
	grad, _ := tensor.New[float32]([]int{1}, []float32{1.0})
	param.Gradient = grad

	// First step: inner sets value to 10.0, shadow initializes to 10.0
	if err := ema.Step(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatalf("Step 1: %v", err)
	}

	// Second step: inner sets value to 10.0 again, shadow = 0.9*10 + 0.1*10 = 10.0
	grad2, _ := tensor.New[float32]([]int{1}, []float32{1.0})
	param.Gradient = grad2
	if err := ema.Step(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatalf("Step 2: %v", err)
	}

	// Shadow should be 10.0 (converged)
	shadowVal := ema.shadow[param].Data()[0]
	if math.Abs(float64(shadowVal)-10.0) > 0.01 {
		t.Errorf("shadow = %f, want ~10.0", shadowVal)
	}
}

func TestEMA_SwapShadow_SwapBack(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	inner := &noopOptimizer[float32]{}
	ema := NewEMA[float32](inner, engine, 0.5)

	value, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])
	grad, _ := tensor.New[float32]([]int{2}, []float32{0.0, 0.0})
	param.Gradient = grad

	// Initialize shadow
	if err := ema.Step(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatalf("Step: %v", err)
	}

	// Manually set shadow to different values
	ema.shadow[param].Data()[0] = 5.0
	ema.shadow[param].Data()[1] = 6.0
	ema.shadow[param].SetData(ema.shadow[param].Data())

	origValue := make([]float32, 2)
	copy(origValue, param.Value.Data())

	// Swap: param should get shadow values
	if err := ema.SwapShadow(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatalf("SwapShadow: %v", err)
	}
	if param.Value.Data()[0] != 5.0 || param.Value.Data()[1] != 6.0 {
		t.Errorf("after swap: param = %v, want [5.0, 6.0]", param.Value.Data())
	}

	// SwapBack: should restore original values
	if err := ema.SwapBack(ctx, []*graph.Parameter[float32]{param}); err != nil {
		t.Fatalf("SwapBack: %v", err)
	}
	for i, v := range param.Value.Data() {
		if math.Abs(float64(v-origValue[i])) > 1e-6 {
			t.Errorf("after swapback: param[%d] = %f, want %f", i, v, origValue[i])
		}
	}
}
