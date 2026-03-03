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

func newHModule(t *testing.T) *hrm.HModule[float32] {
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

	m, err := hrm.NewHModule[float32](engine, ops, modelDim, ffnDim, attn)
	if err != nil {
		t.Fatalf("failed to create HModule: %v", err)
	}

	return m
}

func TestNewHModule(t *testing.T) {
	m := newHModule(t)

	if m.Block == nil {
		t.Error("HModule.Block is nil")
	}

	if m.HiddenState == nil {
		t.Error("HModule.HiddenState is nil")
	}

	if len(m.Parameters()) == 0 {
		t.Error("HModule has no parameters")
	}
}

func TestHModule_OpType(t *testing.T) {
	m := newHModule(t)
	if got := m.OpType(); got != "HModule" {
		t.Errorf("OpType() = %q, want %q", got, "HModule")
	}
}

func TestHModule_Attributes(t *testing.T) {
	m := newHModule(t)

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

func TestHModule_OutputShape(t *testing.T) {
	m := newHModule(t)

	// OutputShape delegates to Block.OutputShape which delegates to attention.
	// May return nil before forward is called. Just verify it doesn't panic.
	_ = m.OutputShape()
}

func TestHModule_Forward(t *testing.T) {
	m := newHModule(t)
	ctx := context.Background()

	// 2D input: [batch=1, model_dim=16]
	input, err := tensor.New[float32]([]int{1, 16}, nil)
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	output, err := m.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Output should match input shape (2D → squeeze back to 2D)
	if got := output.Shape(); got[0] != 1 || got[1] != 16 {
		t.Errorf("output shape = %v, want [1, 16]", got)
	}

	// HiddenState should be updated
	if m.HiddenState != output {
		t.Error("HiddenState not updated after Forward")
	}
}

func TestHModule_Forward3D(t *testing.T) {
	m := newHModule(t)
	ctx := context.Background()

	// Reset hidden state to 3D to test the non-squeeze path
	m.HiddenState, _ = tensor.New[float32]([]int{1, 1, 16}, nil)

	input, err := tensor.New[float32]([]int{1, 1, 16}, nil)
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	output, err := m.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward with 3D input failed: %v", err)
	}

	if len(output.Shape()) != 3 {
		t.Errorf("expected 3D output, got %dD: %v", len(output.Shape()), output.Shape())
	}
}

func TestHModule_ForwardNoInputs(t *testing.T) {
	m := newHModule(t)
	ctx := context.Background()

	_, err := m.Forward(ctx)
	if err == nil {
		t.Error("Forward with no inputs should fail")
	}
}

func TestHModule_Backward(t *testing.T) {
	m := newHModule(t)
	ctx := context.Background()

	// Must call Forward first to cache intermediates.
	input, _ := tensor.New[float32]([]int{1, 16}, nil)

	_, err := m.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Backward with gradient matching output shape.
	dOut, _ := tensor.New[float32]([]int{1, 16}, nil)

	grads, err := m.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	if len(grads) == 0 {
		t.Fatal("Backward returned no gradients")
	}

	// Gradient should match input shape.
	if got := grads[0].Shape(); got[0] != 1 || got[1] != 16 {
		t.Errorf("gradient shape = %v, want [1, 16]", got)
	}
}

func TestHModule_Backward3D(t *testing.T) {
	m := newHModule(t)
	ctx := context.Background()

	// Reset hidden state to 3D
	m.HiddenState, _ = tensor.New[float32]([]int{1, 1, 16}, nil)

	input, _ := tensor.New[float32]([]int{1, 1, 16}, nil)

	_, err := m.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32]([]int{1, 1, 16}, nil)

	grads, err := m.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward with 3D failed: %v", err)
	}

	if len(grads) == 0 {
		t.Fatal("Backward returned no gradients")
	}

	if len(grads[0].Shape()) != 3 {
		t.Errorf("expected 3D gradient, got %dD: %v", len(grads[0].Shape()), grads[0].Shape())
	}
}

func TestHModule_GraphNodeInterface(t *testing.T) {
	m := newHModule(t)

	// Verify the static assertion compiles — this is a compile-time check,
	// but we also verify at runtime.
	var _ graph.Node[float32] = m
}

func TestHModule_MultipleForwardSteps(t *testing.T) {
	m := newHModule(t)
	ctx := context.Background()

	// Run multiple forward steps to verify hidden state recurrence.
	for step := 0; step < 3; step++ {
		input, _ := tensor.New[float32]([]int{1, 16}, nil)

		output, err := m.Forward(ctx, input)
		if err != nil {
			t.Fatalf("step %d: Forward failed: %v", step, err)
		}

		if output == nil {
			t.Fatalf("step %d: output is nil", step)
		}
	}
}
