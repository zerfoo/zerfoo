// Package hrm_test contains tests for the HRM layers.
package hrm_test

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/hrm"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func newLModule(t *testing.T) *hrm.LModule[float32] {
	t.Helper()

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 32
	numHeads := 2

	attn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create attention: %v", err)
	}

	m, err := hrm.NewLModule[float32](engine, ops, modelDim, ffnDim, attn)
	if err != nil {
		t.Fatalf("failed to create LModule: %v", err)
	}

	return m
}

func TestNewLModule(t *testing.T) {
	m := newLModule(t)

	if m.Block == nil {
		t.Error("LModule.Block is nil")
	}

	if m.HiddenState == nil {
		t.Error("LModule.HiddenState is nil")
	}

	if len(m.Parameters()) == 0 {
		t.Error("LModule has no parameters")
	}
}

func TestLModule_OpType(t *testing.T) {
	m := newLModule(t)
	if got := m.OpType(); got != "LModule" {
		t.Errorf("OpType() = %q, want %q", got, "LModule")
	}
}

func TestLModule_Attributes(t *testing.T) {
	m := newLModule(t)

	attrs := m.Attributes()
	if attrs == nil {
		t.Fatal("Attributes() returned nil")
	}

	dim, ok := attrs["model_dim"]
	if !ok {
		t.Fatal("Attributes() missing model_dim")
	}

	if dim != 16 {
		t.Errorf("model_dim = %v, want 16", dim)
	}
}

func TestLModule_OutputShape(t *testing.T) {
	m := newLModule(t)

	// OutputShape delegates to Block.OutputShape which delegates to attention.
	// May return nil before forward is called. Just verify it doesn't panic.
	_ = m.OutputShape()
}

func TestLModule_Forward(t *testing.T) {
	m := newLModule(t)
	ctx := context.Background()

	// 2D inputs: [batch=1, model_dim=16]
	hState, _ := tensor.New[float32]([]int{1, 16}, nil)
	projectedInput, _ := tensor.New[float32]([]int{1, 16}, nil)

	output, err := m.Forward(ctx, hState, projectedInput)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if got := output.Shape(); got[0] != 1 || got[1] != 16 {
		t.Errorf("output shape = %v, want [1, 16]", got)
	}

	if m.HiddenState != output {
		t.Error("HiddenState not updated after Forward")
	}
}

func TestLModule_Forward3D(t *testing.T) {
	m := newLModule(t)
	ctx := context.Background()

	hState, _ := tensor.New[float32]([]int{1, 1, 16}, nil)
	projectedInput, _ := tensor.New[float32]([]int{1, 1, 16}, nil)

	output, err := m.Forward(ctx, hState, projectedInput)
	if err != nil {
		t.Fatalf("Forward with 3D input failed: %v", err)
	}

	if len(output.Shape()) != 3 {
		t.Errorf("expected 3D output, got %dD: %v", len(output.Shape()), output.Shape())
	}
}

func TestLModule_ForwardTooFewInputs(t *testing.T) {
	m := newLModule(t)
	ctx := context.Background()

	tests := []struct {
		name   string
		inputs []*tensor.TensorNumeric[float32]
	}{
		{"zero inputs", nil},
		{"one input", func() []*tensor.TensorNumeric[float32] {
			t, _ := tensor.New[float32]([]int{1, 16}, nil)
			return []*tensor.TensorNumeric[float32]{t}
		}()},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := m.Forward(ctx, tc.inputs...)
			if err == nil {
				t.Error("Forward with too few inputs should fail")
			}
		})
	}
}

func TestLModule_Backward(t *testing.T) {
	m := newLModule(t)
	ctx := context.Background()

	hState, _ := tensor.New[float32]([]int{1, 16}, nil)
	projectedInput, _ := tensor.New[float32]([]int{1, 16}, nil)

	_, err := m.Forward(ctx, hState, projectedInput)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32]([]int{1, 16}, nil)

	grads, err := m.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	// LModule returns 2 gradients: [dHState, dProjectedInput]
	if len(grads) != 2 {
		t.Fatalf("Backward returned %d gradients, want 2", len(grads))
	}

	for i, g := range grads {
		if got := g.Shape(); got[0] != 1 || got[1] != 16 {
			t.Errorf("gradient[%d] shape = %v, want [1, 16]", i, got)
		}
	}
}

func TestLModule_Backward3D(t *testing.T) {
	m := newLModule(t)
	ctx := context.Background()

	hState, _ := tensor.New[float32]([]int{1, 1, 16}, nil)
	projectedInput, _ := tensor.New[float32]([]int{1, 1, 16}, nil)

	_, err := m.Forward(ctx, hState, projectedInput)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32]([]int{1, 1, 16}, nil)

	grads, err := m.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward with 3D failed: %v", err)
	}

	if len(grads) != 2 {
		t.Fatalf("Backward returned %d gradients, want 2", len(grads))
	}

	for i, g := range grads {
		if len(g.Shape()) != 3 {
			t.Errorf("gradient[%d] expected 3D, got %dD: %v", i, len(g.Shape()), g.Shape())
		}
	}
}

func TestLModule_GraphNodeInterface(t *testing.T) {
	m := newLModule(t)
	var _ graph.Node[float32] = m
}

func TestLModule_MultipleForwardSteps(t *testing.T) {
	m := newLModule(t)
	ctx := context.Background()

	for step := 0; step < 3; step++ {
		hState, _ := tensor.New[float32]([]int{1, 16}, nil)
		projectedInput, _ := tensor.New[float32]([]int{1, 16}, nil)

		output, err := m.Forward(ctx, hState, projectedInput)
		if err != nil {
			t.Fatalf("step %d: Forward failed: %v", step, err)
		}

		if output == nil {
			t.Fatalf("step %d: output is nil", step)
		}
	}
}
