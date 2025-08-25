package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
	"github.com/zerfoo/zerfoo/types"
)

// Analytic check: for degree=2 with bias on 2 features, terms are
// [1, x, y, x^2, x*y, y^2] (order may vary). With upstream grad = 1 for all terms,
// dL/dx = 1 + 2x + y, dL/dy = 1 + x + 2y.
func TestPolynomialExpansion_BackwardValues_Deg2WithBias(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	poly, err := NewPolynomialExpansion("poly", engine, ops, 2, WithPolynomialDegree[float32](2), WithPolynomialBias[float32](true))
	testutils.AssertNoError(t, err, "new poly")

	in, err := tensor.New([]int{1, 2}, []float32{2.0, 3.0})
	testutils.AssertNoError(t, err, "new input")

	out, err := poly.Forward(context.Background(), in)
	testutils.AssertNoError(t, err, "forward")

	// upstream grad = ones
	gOutData := make([]float32, len(out.Data()))
	for i := range gOutData {
		gOutData[i] = 1.0
	}
	gOut, err := tensor.New(out.Shape(), gOutData)
	testutils.AssertNoError(t, err, "new gout")

	grads, err := poly.Backward(context.Background(), types.FullBackprop, gOut, in)
	testutils.AssertNoError(t, err, "backward")
	testutils.AssertEqual(t, 1, len(grads), "grads len")

	gIn := grads[0]
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{1, 2}, gIn.Shape()), "grad shape")

	// Expected analytic gradients at x=2, y=3
	// dL/dx = 1 + 2x + y = 1 + 4 + 3 = 8
	// dL/dy = 1 + x + 2y = 1 + 2 + 6 = 9
	expected := []float32{8.0, 9.0}
	testutils.AssertFloat32SliceApproxEqual(t, expected, gIn.Data(), 1e-5, "grad values")
}
