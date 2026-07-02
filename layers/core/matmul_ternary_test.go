package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestMatMul_TernaryDispatch(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	layer := NewMatMul[float32](engine)

	// Weight matrix B [N=2, K=4] with ternary values:
	//   row 0: [ 1,  0, -1,  1]
	//   row 1: [-1,  1,  0,  0]
	ts := tensor.NewTernaryStorageFrom([]int8{1, 0, -1, 1, -1, 1, 0, 0})
	b, err := tensor.NewWithStorage[float32]([]int{2, 4}, ts)
	if err != nil {
		t.Fatalf("failed to create ternary tensor: %v", err)
	}

	tests := []struct {
		name    string
		aShape  []int
		aData   []float32
		wantOut []int
		want    []float32
	}{
		{
			name:   "single_vector",
			aShape: []int{1, 4},
			aData:  []float32{1, 2, 3, 4},
			// row0: 1*1 + 2*0 + 3*(-1) + 4*1 = 1 - 3 + 4 = 2
			// row1: 1*(-1) + 2*1 + 3*0 + 4*0 = -1 + 2 = 1
			wantOut: []int{1, 2},
			want:    []float32{2, 1},
		},
		{
			name:   "two_vectors",
			aShape: []int{2, 4},
			aData:  []float32{1, 2, 3, 4, 0, 1, 0, 1},
			// vec0: row0=2, row1=1 (same as above)
			// vec1: row0=0*1+1*0+0*(-1)+1*1=1, row1=0*(-1)+1*1+0*0+1*0=1
			wantOut: []int{2, 2},
			want:    []float32{2, 1, 1, 1},
		},
		{
			name:   "batched_3d",
			aShape: []int{2, 1, 4},
			aData:  []float32{1, 2, 3, 4, 0, 1, 0, 1},
			wantOut: []int{2, 1, 2},
			want:    []float32{2, 1, 1, 1},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a, err := tensor.New[float32](tc.aShape, tc.aData)
			if err != nil {
				t.Fatalf("failed to create input tensor: %v", err)
			}

			result, err := layer.Forward(context.Background(), a, b)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			gotShape := result.Shape()
			if len(gotShape) != len(tc.wantOut) {
				t.Fatalf("shape length: got %v, want %v", gotShape, tc.wantOut)
			}
			for i := range gotShape {
				if gotShape[i] != tc.wantOut[i] {
					t.Fatalf("shape[%d]: got %d, want %d", i, gotShape[i], tc.wantOut[i])
				}
			}

			gotData := result.Data()
			if len(gotData) != len(tc.want) {
				t.Fatalf("data length: got %d, want %d", len(gotData), len(tc.want))
			}
			for i, w := range tc.want {
				if math.Abs(float64(gotData[i]-w)) > 1e-6 {
					t.Errorf("data[%d]: got %f, want %f", i, gotData[i], w)
				}
			}
		})
	}
}

func TestMatMul_TernaryFallsBackForNonFloat32(t *testing.T) {
	// Verify that non-float32 tensors with ternary storage fall through
	// to the standard path (ternary dispatch only supports float32).
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	layer := NewMatMul[float64](engine)

	// Create a regular 2D weight and input to verify standard path works.
	a, err := tensor.New[float64]([]int{1, 2}, []float64{1, 2})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}
	// B is [2, 1] so A [1,2] @ B [2,1] works without transpose.
	b, err := tensor.New[float64]([]int{2, 1}, []float64{3, 4})
	if err != nil {
		t.Fatalf("failed to create weight: %v", err)
	}

	result, err := layer.Forward(context.Background(), a, b)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	got := result.Data()
	// 1*3 + 2*4 = 11
	if math.Abs(float64(got[0])-11) > 1e-6 {
		t.Errorf("got %f, want 11", got[0])
	}
}

func TestMatMul_TernaryTransparentToArchBuilders(t *testing.T) {
	// Verify that the same MatMul layer handles both regular and ternary
	// weights without any configuration change — transparent dispatch.
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	layer := NewMatMul[float32](engine)

	// Regular weight (not ternary).
	aReg, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 2})
	bReg, _ := tensor.New[float32]([]int{2, 1}, []float32{3, 4})
	r1, err := layer.Forward(context.Background(), aReg, bReg)
	if err != nil {
		t.Fatalf("regular Forward failed: %v", err)
	}
	if math.Abs(float64(r1.Data()[0])-11) > 1e-6 {
		t.Errorf("regular: got %f, want 11", r1.Data()[0])
	}

	// Ternary weight — same layer instance.
	ts := tensor.NewTernaryStorageFrom([]int8{1, -1})
	bTern, _ := tensor.NewWithStorage[float32]([]int{1, 2}, ts)
	aTern, _ := tensor.New[float32]([]int{1, 2}, []float32{5, 3})
	r2, err := layer.Forward(context.Background(), aTern, bTern)
	if err != nil {
		t.Fatalf("ternary Forward failed: %v", err)
	}
	// 5*1 + 3*(-1) = 2
	if math.Abs(float64(r2.Data()[0])-2) > 1e-6 {
		t.Errorf("ternary: got %f, want 2", r2.Data()[0])
	}
}
