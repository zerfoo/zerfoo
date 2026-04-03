package functional

import (
	"context"
	"math"
	"testing"
)

func TestMultiHeadAttentionBackward_NumericalGradient(t *testing.T) {
	ctx := context.Background()
	eps := 1e-5
	tol := 1e-3

	cases := []struct {
		name   string
		shape  []int
		nHeads int
	}{
		{"4x8_2heads", []int{4, 8}, 2},
		{"8x16_4heads", []int{8, 16}, 4},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			engine, ops := newF64Engine()
			seqLen, dModel := tc.shape[0], tc.shape[1]
			n := seqLen * dModel

			// Create input data with varied values.
			qData := make([]float64, n)
			kData := make([]float64, n)
			vData := make([]float64, n)
			for i := range n {
				qData[i] = float64(i)*0.1 - float64(n)*0.05
				kData[i] = float64(n-i)*0.1 - float64(n)*0.05
				vData[i] = float64(i%7)*0.2 - 0.6
			}

			// Upstream gradient.
			dOutData := make([]float64, n)
			for i := range n {
				dOutData[i] = float64(i+1) * 0.01
			}

			qT := makeTensor(t, tc.shape, qData)
			kT := makeTensor(t, tc.shape, kData)
			vT := makeTensor(t, tc.shape, vData)
			dOutT := makeTensor(t, tc.shape, dOutData)

			// Analytical gradients.
			dQ, dK, dV, err := MultiHeadAttentionBackward(ctx, engine, ops, dOutT, qT, kT, vT, tc.nHeads)
			if err != nil {
				t.Fatalf("MultiHeadAttentionBackward: %v", err)
			}
			dQData := dQ.Data()
			dKData := dK.Data()
			dVData := dV.Data()

			// loss = sum(dOutput * MHA(q, k, v))
			loss := func(q, k, v []float64) float64 {
				qt := makeTensor(t, tc.shape, q)
				kt := makeTensor(t, tc.shape, k)
				vt := makeTensor(t, tc.shape, v)
				out, err := MultiHeadAttention(ctx, engine, qt, kt, vt, tc.nHeads)
				if err != nil {
					t.Fatalf("MultiHeadAttention: %v", err)
				}
				return dot64(dOutData, out.Data())
			}

			// Check dQ numerically.
			for i := 0; i < n; i++ {
				plus := make([]float64, n)
				copy(plus, qData)
				plus[i] += eps
				minus := make([]float64, n)
				copy(minus, qData)
				minus[i] -= eps
				numerical := (loss(plus, kData, vData) - loss(minus, kData, vData)) / (2 * eps)
				if math.Abs(dQData[i]-numerical) > tol {
					t.Errorf("dQ[%d]: analytical=%v, numerical=%v, diff=%v",
						i, dQData[i], numerical, math.Abs(dQData[i]-numerical))
				}
			}

			// Check dK numerically.
			for i := 0; i < n; i++ {
				plus := make([]float64, n)
				copy(plus, kData)
				plus[i] += eps
				minus := make([]float64, n)
				copy(minus, kData)
				minus[i] -= eps
				numerical := (loss(qData, plus, vData) - loss(qData, minus, vData)) / (2 * eps)
				if math.Abs(dKData[i]-numerical) > tol {
					t.Errorf("dK[%d]: analytical=%v, numerical=%v, diff=%v",
						i, dKData[i], numerical, math.Abs(dKData[i]-numerical))
				}
			}

			// Check dV numerically.
			for i := 0; i < n; i++ {
				plus := make([]float64, n)
				copy(plus, vData)
				plus[i] += eps
				minus := make([]float64, n)
				copy(minus, vData)
				minus[i] -= eps
				numerical := (loss(qData, kData, plus) - loss(qData, kData, minus)) / (2 * eps)
				if math.Abs(dVData[i]-numerical) > tol {
					t.Errorf("dV[%d]: analytical=%v, numerical=%v, diff=%v",
						i, dVData[i], numerical, math.Abs(dVData[i]-numerical))
				}
			}
		})
	}
}
