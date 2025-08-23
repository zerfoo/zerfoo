package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestConcat_ForwardAxis1(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	layer := NewConcat[float32](engine, 1)

	in1, err := tensor.New[float32]([]int{2, 3}, []float32{
		0, 1, 2,
		3, 4, 5,
	})
	testutils.AssertNoError(t, err, "create in1")
	in2, err := tensor.New[float32]([]int{2, 2}, []float32{
		100, 101,
		102, 103,
	})
	testutils.AssertNoError(t, err, "create in2")

	out, err := layer.Forward(context.Background(), in1, in2)
	testutils.AssertNoError(t, err, "forward")
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{2, 5}, out.Shape()), "shape mismatch")

	expected := []float32{
		0, 1, 2, 100, 101,
		3, 4, 5, 102, 103,
	}
	testutils.AssertFloat32SliceApproxEqual(t, expected, out.Data(), 0, "data mismatch")
}

func TestConcat_BackwardAxis1(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	layer := NewConcat[float32](engine, 1)

	in1, _ := tensor.New[float32]([]int{2, 3}, []float32{
		0, 1, 2,
		3, 4, 5,
	})
	in2, _ := tensor.New[float32]([]int{2, 2}, []float32{
		100, 101,
		102, 103,
	})
	// Upstream gradient for concatenated output [2,5]
	gOut, _ := tensor.New[float32]([]int{2, 5}, []float32{
		10, 11, 12, 13, 14,
		20, 21, 22, 23, 24,
	})

	grads, err := layer.Backward(context.Background(), gOut, in1, in2)
	testutils.AssertNoError(t, err, "backward")
	testutils.AssertEqual(t, 2, len(grads), "grads len")
	testutils.AssertTrue(t, testutils.IntSliceEqual(in1.Shape(), grads[0].Shape()), "grad1 shape")
	testutils.AssertTrue(t, testutils.IntSliceEqual(in2.Shape(), grads[1].Shape()), "grad2 shape")

	expG1 := []float32{
		10, 11, 12,
		20, 21, 22,
	}
	expG2 := []float32{
		13, 14,
		23, 24,
	}

	testutils.AssertFloat32SliceApproxEqual(t, expG1, grads[0].Data(), 0, "grad1 data")
	testutils.AssertFloat32SliceApproxEqual(t, expG2, grads[1].Data(), 0, "grad2 data")
}

func TestConcat_ForwardAxis0(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	layer := NewConcat[float32](engine, 0)

	in1, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 2, 3})
	in2, _ := tensor.New[float32]([]int{2, 3}, []float32{4, 5, 6, 7, 8, 9})

	out, err := layer.Forward(context.Background(), in1, in2)
	testutils.AssertNoError(t, err, "forward axis0")
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{3, 3}, out.Shape()), "shape")

	expected := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	testutils.AssertFloat32SliceApproxEqual(t, expected, out.Data(), 0, "data")
}

func TestConcat_BackwardAxis0(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	layer := NewConcat[float32](engine, 0)

	in1, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 2, 3})
	in2, _ := tensor.New[float32]([]int{2, 3}, []float32{4, 5, 6, 7, 8, 9})
	gOut, _ := tensor.New[float32]([]int{3, 3}, []float32{
		10, 11, 12,
		20, 21, 22,
		30, 31, 32,
	})

	grads, err := layer.Backward(context.Background(), gOut, in1, in2)
	testutils.AssertNoError(t, err, "backward axis0")
	testutils.AssertEqual(t, 2, len(grads), "grads len")

	expG1 := []float32{10, 11, 12}
	expG2 := []float32{20, 21, 22, 30, 31, 32}

	testutils.AssertFloat32SliceApproxEqual(t, expG1, grads[0].Data(), 0, "grad1 data")
	testutils.AssertFloat32SliceApproxEqual(t, expG2, grads[1].Data(), 0, "grad2 data")
}
