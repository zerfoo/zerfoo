package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func calculateExpectedNormalizedTensor(input *tensor.TensorNumeric[float64], epsilon float64) *tensor.TensorNumeric[float64] {
	data := input.Data()

	sumSq := 0.0
	for _, v := range data {
		sumSq += v * v
	}

	rms := math.Sqrt(sumSq/float64(input.Size()) + epsilon)

	normalizedData := make([]float64, input.Size())
	for i, v := range data {
		normalizedData[i] = v / rms
	}

	newTensor, _ := tensor.New[float64](input.Shape(), normalizedData)

	return newTensor
}

func TestQKNorm(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})

	tests := []struct {
		name      string
		qInput    *tensor.TensorNumeric[float64]
		kInput    *tensor.TensorNumeric[float64]
		epsilon   float64
		expectErr bool
	}{
		{
			name: "Simple 1D tensors",
			qInput: func() *tensor.TensorNumeric[float64] {
				t, _ := tensor.New[float64]([]int{3}, []float64{1.0, 2.0, 3.0})

				return t
			}(),
			kInput: func() *tensor.TensorNumeric[float64] {
				t, _ := tensor.New[float64]([]int{3}, []float64{4.0, 5.0, 6.0})

				return t
			}(),
			epsilon:   1e-5,
			expectErr: false,
		},
		{
			name: "2D tensors",
			qInput: func() *tensor.TensorNumeric[float64] {
				t, _ := tensor.New[float64]([]int{2, 2}, []float64{1.0, 2.0, 3.0, 4.0})

				return t
			}(),
			kInput: func() *tensor.TensorNumeric[float64] {
				t, _ := tensor.New[float64]([]int{2, 2}, []float64{5.0, 6.0, 7.0, 8.0})

				return t
			}(),
			epsilon:   1e-5,
			expectErr: false,
		},
		{
			name: "Mismatched shapes",
			qInput: func() *tensor.TensorNumeric[float64] {
				t, _ := tensor.New[float64]([]int{3}, []float64{1.0, 2.0, 3.0})

				return t
			}(),
			kInput: func() *tensor.TensorNumeric[float64] {
				t, _ := tensor.New[float64]([]int{2}, []float64{4.0, 5.0})

				return t
			}(),
			epsilon:   1e-5,
			expectErr: true,
		},
		{
			name:   "Nil Q input",
			qInput: nil,
			kInput: func() *tensor.TensorNumeric[float64] {
				t, _ := tensor.New[float64]([]int{3}, []float64{4.0, 5.0, 6.0})

				return t
			}(),
			epsilon:   1e-5,
			expectErr: true,
		},
		{
			name: "Nil K input",
			qInput: func() *tensor.TensorNumeric[float64] {
				t, _ := tensor.New[float64]([]int{3}, []float64{1.0, 2.0, 3.0})

				return t
			}(),
			kInput:    nil,
			epsilon:   1e-5,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actualQ, actualK, err := QKNorm(ctx, engine, tt.qInput, tt.kInput, tt.epsilon)

			if tt.expectErr {
				testutils.AssertError(t, err, "expected error")
			} else {
				testutils.AssertNoError(t, err, "unexpected error")
				testutils.AssertNotNil(t, actualQ, "actualQ should not be nil")
				testutils.AssertNotNil(t, actualK, "actualK should not be nil")

				expectedQ := calculateExpectedNormalizedTensor(tt.qInput, tt.epsilon)
				expectedK := calculateExpectedNormalizedTensor(tt.kInput, tt.epsilon)

				if !testutils.CompareTensorsApprox(t, actualQ, expectedQ, tt.epsilon) {
					t.Errorf("QKNorm(%s) Q output mismatch", tt.name)
				}

				if !testutils.CompareTensorsApprox(t, actualK, expectedK, tt.epsilon) {
					t.Errorf("QKNorm(%s) K output mismatch", tt.name)
				}
			}
		})
	}
}
