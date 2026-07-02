package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestGemm_Forward(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	tests := []struct {
		name      string
		alpha     float64
		beta      float64
		transA    bool
		transB    bool
		aShape    []int
		aData     []float32
		bShape    []int
		bData     []float32
		cShape    []int
		cData     []float32
		wantShape []int
		wantData  []float32
	}{
		{
			name:  "basic 2x3 * 3x2",
			alpha: 1.0, beta: 0.0,
			aShape: []int{2, 3}, aData: []float32{1, 2, 3, 4, 5, 6},
			bShape: []int{3, 2}, bData: []float32{1, 2, 3, 4, 5, 6},
			wantShape: []int{2, 2},
			wantData:  []float32{22, 28, 49, 64},
		},
		{
			name:  "with bias vector",
			alpha: 1.0, beta: 1.0,
			aShape: []int{2, 3}, aData: []float32{1, 2, 3, 4, 5, 6},
			bShape: []int{3, 2}, bData: []float32{1, 2, 3, 4, 5, 6},
			cShape: []int{2}, cData: []float32{10, 20},
			wantShape: []int{2, 2},
			wantData:  []float32{32, 48, 59, 84},
		},
		{
			name:  "transB",
			alpha: 1.0, beta: 0.0, transB: true,
			aShape: []int{2, 3}, aData: []float32{1, 2, 3, 4, 5, 6},
			bShape: []int{2, 3}, bData: []float32{1, 2, 3, 4, 5, 6},
			wantShape: []int{2, 2},
			wantData:  []float32{14, 32, 32, 77},
		},
		{
			name:  "transA",
			alpha: 1.0, beta: 0.0, transA: true,
			aShape: []int{3, 2}, aData: []float32{1, 4, 2, 5, 3, 6},
			bShape: []int{3, 2}, bData: []float32{1, 2, 3, 4, 5, 6},
			wantShape: []int{2, 2},
			wantData:  []float32{22, 28, 49, 64},
		},
		{
			name:  "alpha scaling",
			alpha: 2.0, beta: 0.0,
			aShape: []int{1, 2}, aData: []float32{1, 2},
			bShape: []int{2, 1}, bData: []float32{3, 4},
			wantShape: []int{1, 1},
			wantData:  []float32{22}, // 2*(1*3+2*4) = 22
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node := &Gemm[float32]{
				engine: eng, ops: ops,
				alpha: tc.alpha, beta: tc.beta,
				transA: tc.transA, transB: tc.transB,
			}
			a, _ := tensor.New[float32](tc.aShape, tc.aData)
			b, _ := tensor.New[float32](tc.bShape, tc.bData)
			var inputs []*tensor.TensorNumeric[float32]
			inputs = append(inputs, a, b)
			if tc.cData != nil {
				c, _ := tensor.New[float32](tc.cShape, tc.cData)
				inputs = append(inputs, c)
			}

			out, err := node.Forward(context.Background(), inputs...)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			gotShape := out.Shape()
			if len(gotShape) != len(tc.wantShape) {
				t.Fatalf("shape = %v, want %v", gotShape, tc.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tc.wantShape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], tc.wantShape[i])
				}
			}
			got := out.Data()
			for i := range got {
				if math.Abs(float64(got[i]-tc.wantData[i])) > 1e-4 {
					t.Errorf("data[%d] = %v, want %v", i, got[i], tc.wantData[i])
				}
			}
		})
	}
}

func TestGemm_WrongInputCount(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	node := &Gemm[float32]{engine: eng, ops: ops, alpha: 1, beta: 0}
	a, _ := tensor.New[float32]([]int{2, 2}, nil)
	_, err := node.Forward(context.Background(), a)
	if err == nil {
		t.Error("expected error for 1 input")
	}
}

func TestGemm_OpType(t *testing.T) {
	node := &Gemm[float32]{}
	if got := node.OpType(); got != "Gemm" {
		t.Errorf("OpType() = %q, want %q", got, "Gemm")
	}
}
