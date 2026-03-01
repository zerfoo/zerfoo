package regularization

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// newTestDropout is a helper that creates an engine, ops, and Dropout for tests.
func newTestDropout(rate float32) (*Dropout[float32], compute.Engine[float32]) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	d := NewDropout(engine, ops, rate)
	return d, engine
}

func TestDropout_Forward_EvalMode(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestDropout(0.5)
	// Default mode is eval (training = false).

	inputData := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	input, err := tensor.New([]int{2, 3}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	// In eval mode the output must be the exact same tensor.
	if output != input {
		t.Error("expected output to be the same pointer as input in eval mode")
	}

	outData := output.Data()
	for i, v := range outData {
		if v != inputData[i] {
			t.Errorf("element %d: got %v, want %v", i, v, inputData[i])
		}
	}
}

func TestDropout_Forward_TrainingMode(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestDropout(0.5)
	d.SetTraining(true)

	// Use a large tensor so the random mask is very unlikely to be all-zero or all-nonzero.
	size := 1000
	inputData := make([]float32, size)
	for i := range inputData {
		inputData[i] = 1.0
	}

	input, err := tensor.New([]int{size}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	// Verify output shape matches input.
	if len(output.Shape()) != 1 || output.Shape()[0] != size {
		t.Fatalf("output shape mismatch: got %v, want [%d]", output.Shape(), size)
	}

	outData := output.Data()
	zeros := 0
	scale := float32(1.0 / (1.0 - 0.5)) // == 2.0
	for _, v := range outData {
		if v == 0 {
			zeros++
		} else if v != scale {
			// Surviving elements must equal input * scale.
			t.Errorf("non-zero element: got %v, want %v", v, scale)
		}
	}

	// With rate=0.5 and 1000 elements, having zero dropped elements is astronomically unlikely.
	if zeros == 0 {
		t.Error("expected some elements to be zeroed in training mode with rate=0.5")
	}
	if zeros == size {
		t.Error("expected some elements to survive in training mode with rate=0.5")
	}
}

func TestDropout_Forward_RateZero(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestDropout(0.0)
	d.SetTraining(true)

	inputData := []float32{1.0, 2.0, 3.0, 4.0}
	input, err := tensor.New([]int{4}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	output, err := d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	// With rate=0, scale = 1/(1-0) = 1, and all elements survive.
	outData := output.Data()
	for i, v := range outData {
		if v != inputData[i] {
			t.Errorf("element %d: got %v, want %v (rate=0 should preserve all values)", i, v, inputData[i])
		}
	}
}

func TestDropout_Backward_TrainingMode(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestDropout(0.5)
	d.SetTraining(true)

	size := 100
	inputData := make([]float32, size)
	for i := range inputData {
		inputData[i] = float32(i + 1)
	}
	input, err := tensor.New([]int{size}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	// Run Forward to generate and cache the mask.
	_, err = d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	// Create an upstream gradient of all ones.
	dOutData := make([]float32, size)
	for i := range dOutData {
		dOutData[i] = 1.0
	}
	dOut, err := tensor.New([]int{size}, dOutData)
	if err != nil {
		t.Fatalf("failed to create dOut tensor: %v", err)
	}

	grads, err := d.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient tensor, got %d", len(grads))
	}

	dInputData := grads[0].Data()
	maskData := d.mask.Data()
	scale := float32(1.0 / (1.0 - 0.5))
	for i, grad := range dInputData {
		if maskData[i] == 0 {
			if grad != 0 {
				t.Errorf("element %d: mask=0 but grad=%v, want 0", i, grad)
			}
		} else {
			// dOut[i] * mask[i] = 1.0 * scale
			if grad != scale {
				t.Errorf("element %d: mask=%v, grad=%v, want %v", i, maskData[i], grad, scale)
			}
		}
	}
}

func TestDropout_Backward_EvalMode(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestDropout(0.5)
	// Default is eval mode.

	inputData := []float32{1.0, 2.0, 3.0}
	input, err := tensor.New([]int{3}, inputData)
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	dOutData := []float32{0.1, 0.2, 0.3}
	dOut, err := tensor.New([]int{3}, dOutData)
	if err != nil {
		t.Fatalf("failed to create dOut tensor: %v", err)
	}

	grads, err := d.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward returned error: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient tensor, got %d", len(grads))
	}

	// In eval mode, dOut should pass through unchanged.
	if grads[0] != dOut {
		t.Error("expected gradient to be the same pointer as dOut in eval mode")
	}
}

func TestDropout_OpType(t *testing.T) {
	d, _ := newTestDropout(0.5)

	if got := d.OpType(); got != "Dropout" {
		t.Errorf("OpType() = %q, want %q", got, "Dropout")
	}
}

func TestDropout_Attributes(t *testing.T) {
	d, _ := newTestDropout(0.3)

	attrs := d.Attributes()
	if attrs == nil {
		t.Fatal("Attributes() returned nil")
	}

	rate, ok := attrs["rate"]
	if !ok {
		t.Fatal("Attributes() missing 'rate' key")
	}

	rateVal, ok := rate.(float32)
	if !ok {
		t.Fatalf("rate attribute has type %T, want float32", rate)
	}

	if rateVal != 0.3 {
		t.Errorf("rate = %v, want 0.3", rateVal)
	}
}

func TestDropout_Parameters(t *testing.T) {
	d, _ := newTestDropout(0.5)

	params := d.Parameters()
	if len(params) != 0 {
		t.Errorf("Parameters() returned %d parameters, want 0", len(params))
	}
}

func TestDropout_InvalidInputCount(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestDropout(0.5)

	// Zero inputs.
	_, err := d.Forward(ctx)
	if err == nil {
		t.Error("expected error for zero inputs, got nil")
	}

	// Two inputs.
	t1, _ := tensor.New([]int{2}, []float32{1, 2})
	t2, _ := tensor.New([]int{2}, []float32{3, 4})
	_, err = d.Forward(ctx, t1, t2)
	if err == nil {
		t.Error("expected error for two inputs, got nil")
	}

	// Also test Backward with wrong count.
	dOut, _ := tensor.New([]int{2}, []float32{1, 1})
	_, err = d.Backward(ctx, types.FullBackprop, dOut)
	if err == nil {
		t.Error("expected error for zero inputs in Backward, got nil")
	}

	_, err = d.Backward(ctx, types.FullBackprop, dOut, t1, t2)
	if err == nil {
		t.Error("expected error for two inputs in Backward, got nil")
	}
}

func TestDropout_SetTraining(t *testing.T) {
	d, _ := newTestDropout(0.5)

	// Default should be false.
	if d.IsTraining() {
		t.Error("expected IsTraining() to be false by default")
	}

	d.SetTraining(true)
	if !d.IsTraining() {
		t.Error("expected IsTraining() to be true after SetTraining(true)")
	}

	d.SetTraining(false)
	if d.IsTraining() {
		t.Error("expected IsTraining() to be false after SetTraining(false)")
	}
}

// TestDropout_OutputShape verifies OutputShape returns the shape from the last Forward call.
func TestDropout_OutputShape(t *testing.T) {
	ctx := context.Background()
	d, _ := newTestDropout(0.5)

	// Before any Forward call, OutputShape should be nil.
	if s := d.OutputShape(); s != nil {
		t.Errorf("OutputShape before Forward: got %v, want nil", s)
	}

	input, err := tensor.New([]int{3, 4}, make([]float32, 12))
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	_, err = d.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	shape := d.OutputShape()
	expected := []int{3, 4}
	if len(shape) != len(expected) {
		t.Fatalf("OutputShape length: got %d, want %d", len(shape), len(expected))
	}
	for i, dim := range shape {
		if dim != expected[i] {
			t.Errorf("OutputShape[%d] = %d, want %d", i, dim, expected[i])
		}
	}
}

// TestDropout_BuildDropout verifies the registry builder function.
func TestDropout_BuildDropout(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	// Valid attributes.
	node, err := BuildDropout(engine, ops, "test", nil, map[string]any{
		"rate": float64(0.3),
	})
	if err != nil {
		t.Fatalf("BuildDropout returned error: %v", err)
	}
	if node == nil {
		t.Fatal("BuildDropout returned nil node")
	}
	if node.OpType() != "Dropout" {
		t.Errorf("OpType() = %q, want %q", node.OpType(), "Dropout")
	}

	// Missing rate attribute.
	_, err = BuildDropout(engine, ops, "test", nil, map[string]any{})
	if err == nil {
		t.Error("expected error for missing rate attribute")
	}

	// Wrong type for rate.
	_, err = BuildDropout(engine, ops, "test", nil, map[string]any{
		"rate": "not a number",
	})
	if err == nil {
		t.Error("expected error for non-float64 rate attribute")
	}
}

// Compile-time interface check.
var _ graph.Node[float32] = (*Dropout[float32])(nil)
