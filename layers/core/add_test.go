// Package core contains tests for the core layers.
package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestAdd(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	a, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	b, _ := tensor.New[float32]([]int{1, 4}, []float32{5, 6, 7, 8})

	add := NewAdd[float32](engine)

	// Test Forward pass
	output, err := add.Forward(context.Background(), a, b)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	expectedOutput := []float32{6, 8, 10, 12}
	testutils.AssertFloat32SliceApproxEqual(t, expectedOutput, output.Data(), 1e-6, "Forward pass output incorrect")

	// Test Backward pass
	dOut, _ := tensor.New[float32]([]int{1, 4}, []float32{1, 1, 1, 1})

	grads, err := add.Backward(context.Background(), dOut)
	if err != nil {
		t.Fatalf("Backward pass failed: %v", err)
	}

	if len(grads) != 2 {
		t.Fatalf("Expected 2 gradients, but got %d", len(grads))
	}

	testutils.AssertFloat32SliceApproxEqual(t, dOut.Data(), grads[0].Data(), 1e-6, "Backward pass gradient for input 1 incorrect")
	testutils.AssertFloat32SliceApproxEqual(t, dOut.Data(), grads[1].Data(), 1e-6, "Backward pass gradient for input 2 incorrect")
}

// Statically assert that the type implements the graph.Node interface.
