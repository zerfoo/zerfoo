package compute

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestCPUEngine_Gather_1D2D(t *testing.T) {
	engine := NewCPUEngine[float64](numeric.Float64Ops{})

	vocab, dim := 4, 3
	paramsData := []float64{
		0.1, 0.2, 0.3,
		1.0, 1.1, 1.2,
		2.0, 2.1, 2.2,
		3.0, 3.1, 3.2,
	}
	params, _ := tensor.New[float64]([]int{vocab, dim}, paramsData)

	ctx := context.Background()

	// 1D indices
	idx1D, _ := tensor.New[int]([]int{3}, []int{1, 3, 0})
	out1D, _ := tensor.New[float64]([]int{3, dim}, make([]float64, 3*dim))
	if err := engine.Gather(ctx, params, idx1D, out1D); err != nil {
		t.Fatalf("Gather 1D returned error: %v", err)
	}
	expected1D := []float64{
		// row 1
		1.0, 1.1, 1.2,
		// row 3
		3.0, 3.1, 3.2,
		// row 0
		0.1, 0.2, 0.3,
	}
	if !reflect.DeepEqual(out1D.Data(), expected1D) {
		t.Fatalf("Gather 1D mismatch. got %v want %v", out1D.Data(), expected1D)
	}

	// 2D indices [2,2] => output [2,2,dim]
	idx2D, _ := tensor.New[int]([]int{2, 2}, []int{
		2, 0,
		3, 1,
	})
	out2D, _ := tensor.New[float64]([]int{2, 2, dim}, make([]float64, 2*2*dim))
	if err := engine.Gather(ctx, params, idx2D, out2D); err != nil {
		t.Fatalf("Gather 2D returned error: %v", err)
	}
	expected2D := []float64{
		// (0,0) -> row 2
		2.0, 2.1, 2.2,
		// (0,1) -> row 0
		0.1, 0.2, 0.3,
		// (1,0) -> row 3
		3.0, 3.1, 3.2,
		// (1,1) -> row 1
		1.0, 1.1, 1.2,
	}
	if !reflect.DeepEqual(out2D.Data(), expected2D) {
		t.Fatalf("Gather 2D mismatch. got %v want %v", out2D.Data(), expected2D)
	}
}

func TestCPUEngine_ScatterAdd_1D2D(t *testing.T) {
	engine := NewCPUEngine[float64](numeric.Float64Ops{})
	ctx := context.Background()

	vocab, dim := 4, 3

	// 1D indices case
	dTable1, _ := tensor.New[float64]([]int{vocab, dim}, make([]float64, vocab*dim))
	indices1, _ := tensor.New[int]([]int{3}, []int{0, 2, 2})
	dOut1, _ := tensor.New[float64]([]int{3, dim}, []float64{
		0.1, 0.2, 0.3,
		0.5, 0.5, 0.5,
		1.0, 1.0, 1.0,
	})
	if err := engine.ScatterAdd(ctx, dTable1, indices1, dOut1); err != nil {
		t.Fatalf("ScatterAdd 1D returned error: %v", err)
	}
	expectedTable1 := []float64{
		// row 0
		0.1, 0.2, 0.3,
		// row 1
		0, 0, 0,
		// row 2 (0.5+1.0)
		1.5, 1.5, 1.5,
		// row 3
		0, 0, 0,
	}
	if !reflect.DeepEqual(dTable1.Data(), expectedTable1) {
		t.Fatalf("ScatterAdd 1D table mismatch. got %v want %v", dTable1.Data(), expectedTable1)
	}

	// 2D indices [1,2] with dOut [2,dim]
	dTable2, _ := tensor.New[float64]([]int{vocab, dim}, make([]float64, vocab*dim))
	indices2, _ := tensor.New[int]([]int{1, 2}, []int{1, 0})
	dOut2, _ := tensor.New[float64]([]int{2, dim}, []float64{
		0.2, 0.2, 0.2,
		0.3, 0.3, 0.3,
	})
	if err := engine.ScatterAdd(ctx, dTable2, indices2, dOut2); err != nil {
		t.Fatalf("ScatterAdd 2D returned error: %v", err)
	}
	expectedTable2 := []float64{
		// row 0 gets second row of dOut2
		0.3, 0.3, 0.3,
		// row 1 gets first row of dOut2
		0.2, 0.2, 0.2,
		// row 2
		0, 0, 0,
		// row 3
		0, 0, 0,
	}
	if !reflect.DeepEqual(dTable2.Data(), expectedTable2) {
		t.Fatalf("ScatterAdd 2D table mismatch. got %v want %v", dTable2.Data(), expectedTable2)
	}
}
