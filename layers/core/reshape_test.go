package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestReshape_StaticShape(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	r := NewReshape(engine, []int{2, 3})

	input, err := tensor.New[float32]([]int{6}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	out, err := r.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := out.Shape()
	if len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Errorf("shape = %v, want [2 3]", got)
	}
}

func TestReshape_DynamicShapeFromSecondInput(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	// Static shape is nil — shape comes from second input.
	r := NewReshape[float32](engine, nil)

	data, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("tensor.New data: %v", err)
	}

	// Shape tensor: [3, 2]
	shapeTensor, err := tensor.New[float32]([]int{2}, []float32{3, 2})
	if err != nil {
		t.Fatalf("tensor.New shape: %v", err)
	}

	out, err := r.Forward(context.Background(), data, shapeTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := out.Shape()
	if len(got) != 2 || got[0] != 3 || got[1] != 2 {
		t.Errorf("shape = %v, want [3 2]", got)
	}
}

func TestReshape_DynamicShapeWithInferredDim(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	r := NewReshape[float32](engine, nil)

	data, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("tensor.New data: %v", err)
	}

	// Shape tensor: [3, -1] → should infer [3, 2]
	shapeTensor, err := tensor.New[float32]([]int{2}, []float32{3, -1})
	if err != nil {
		t.Fatalf("tensor.New shape: %v", err)
	}

	out, err := r.Forward(context.Background(), data, shapeTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := out.Shape()
	if len(got) != 2 || got[0] != 3 || got[1] != 2 {
		t.Errorf("shape = %v, want [3 2]", got)
	}
}

func TestReshape_DynamicShapeWithZeroDim(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	r := NewReshape[float32](engine, nil)

	data, err := tensor.New[float32]([]int{2, 3, 4}, make([]float32, 24))
	if err != nil {
		t.Fatalf("tensor.New data: %v", err)
	}

	// Shape tensor: [0, -1] → 0 copies dim 0 from input (2), -1 infers 12 → [2, 12]
	shapeTensor, err := tensor.New[float32]([]int{2}, []float32{0, -1})
	if err != nil {
		t.Fatalf("tensor.New shape: %v", err)
	}

	out, err := r.Forward(context.Background(), data, shapeTensor)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := out.Shape()
	if len(got) != 2 || got[0] != 2 || got[1] != 12 {
		t.Errorf("shape = %v, want [2 12]", got)
	}
}
