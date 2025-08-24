package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestConcat_Forward_NegativeAxis(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	// axis=-1 on 2D is equivalent to axis=1 (last dim)
	layerNeg := NewConcat[float32](engine, -1)
	layerPos := NewConcat[float32](engine, 1)

	in1, _ := tensor.New([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	in2, _ := tensor.New([]int{2, 2}, []float32{
		7, 8,
		9, 10,
	})

	outNeg, err := layerNeg.Forward(context.Background(), in1, in2)
	testutils.AssertNoError(t, err, "forward neg axis")
	outPos, err := layerPos.Forward(context.Background(), in1, in2)
	testutils.AssertNoError(t, err, "forward pos axis")

	testutils.AssertTrue(t, testutils.IntSliceEqual(outPos.Shape(), outNeg.Shape()), "shape equal")
	testutils.AssertFloat32SliceApproxEqual(t, outPos.Data(), outNeg.Data(), 0, "data equal")
}

func TestConcat_Backward_NegativeAxis(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	layer := NewConcat[float32](engine, -1)

	in1, _ := tensor.New([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	in2, _ := tensor.New([]int{2, 2}, []float32{
		7, 8,
		9, 10,
	})
	// gOut for concatenated [2,5]
	gOut, _ := tensor.New([]int{2, 5}, []float32{
		10, 11, 12, 13, 14,
		20, 21, 22, 23, 24,
	})

	grads, err := layer.Backward(context.Background(), gOut, in1, in2)
	testutils.AssertNoError(t, err, "backward neg axis")
	testutils.AssertEqual(t, 2, len(grads), "grads len")

	expG1 := []float32{10, 11, 12, 20, 21, 22}
	expG2 := []float32{13, 14, 23, 24}
	testutils.AssertFloat32SliceApproxEqual(t, expG1, grads[0].Data(), 0, "grad1")
	testutils.AssertFloat32SliceApproxEqual(t, expG2, grads[1].Data(), 0, "grad2")
}

func TestConcat_Forward_ThreeInputs(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	layer := NewConcat[float32](engine, 1)

	in1, _ := tensor.New([]int{1, 2}, []float32{1, 2})
	in2, _ := tensor.New([]int{1, 3}, []float32{3, 4, 5})
	in3, _ := tensor.New([]int{1, 1}, []float32{6})

	out, err := layer.Forward(context.Background(), in1, in2, in3)
	testutils.AssertNoError(t, err, "forward three inputs")
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{1, 6}, out.Shape()), "shape")
	expected := []float32{1, 2, 3, 4, 5, 6}
	testutils.AssertFloat32SliceApproxEqual(t, expected, out.Data(), 0, "data")
}

func TestConcat_Backward_ThreeInputs(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	layer := NewConcat[float32](engine, 1)

	in1, _ := tensor.New([]int{2, 1}, []float32{1, 2})
	in2, _ := tensor.New([]int{2, 2}, []float32{3, 4, 5, 6})
	in3, _ := tensor.New([]int{2, 1}, []float32{7, 8})
	gOut, _ := tensor.New([]int{2, 4}, []float32{
		10, 11, 12, 13,
		20, 21, 22, 23,
	})

	grads, err := layer.Backward(context.Background(), gOut, in1, in2, in3)
	testutils.AssertNoError(t, err, "backward three inputs")
	testutils.AssertEqual(t, 3, len(grads), "grads len")

	expG1 := []float32{10, 20}
	expG2 := []float32{11, 12, 21, 22}
	expG3 := []float32{13, 23}
	testutils.AssertFloat32SliceApproxEqual(t, expG1, grads[0].Data(), 0, "g1")
	testutils.AssertFloat32SliceApproxEqual(t, expG2, grads[1].Data(), 0, "g2")
	testutils.AssertFloat32SliceApproxEqual(t, expG3, grads[2].Data(), 0, "g3")
}

func TestConcat_SingleInput_Passthrough(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	layer := NewConcat[float32](engine, 0)

	in1, _ := tensor.New([]int{2, 2}, []float32{1, 2, 3, 4})
	out, err := layer.Forward(context.Background(), in1)
	testutils.AssertNoError(t, err, "forward single input")
	testutils.AssertTrue(t, testutils.IntSliceEqual(in1.Shape(), out.Shape()), "shape")
	testutils.AssertFloat32SliceApproxEqual(t, in1.Data(), out.Data(), 0, "data")

	gOut, _ := tensor.New(out.Shape(), []float32{10, 11, 12, 13})
	grads, err := layer.Backward(context.Background(), gOut, in1)
	testutils.AssertNoError(t, err, "backward single input")
	testutils.AssertEqual(t, 1, len(grads), "grads len")
	testutils.AssertFloat32SliceApproxEqual(t, gOut.Data(), grads[0].Data(), 0, "grad passthrough")
}
