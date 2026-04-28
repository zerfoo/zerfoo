package activations_test

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// TestSoftmaxBackward_MatchesFunctional asserts that the Softmax layer's
// Backward produces the same gradient as functional.SoftmaxBackward when
// fed the same softmax output and upstream gradient. The two
// implementations live in separate packages (to avoid an import cycle),
// so this test pins them to the same numerical contract.
func TestSoftmaxBackward_MatchesFunctional(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	cases := []struct {
		name  string
		shape []int
		in    []float32
		dOut  []float32
	}{
		{
			name:  "1x3_simple",
			shape: []int{1, 3},
			in:    []float32{1, 2, 3},
			dOut:  []float32{0.5, -0.25, 0.75},
		},
		{
			name:  "2x4_mixed",
			shape: []int{2, 4},
			in:    []float32{-1, 0, 1, 2, 3, -2, 0.5, 1.5},
			dOut:  []float32{1, 0, 0, 0, 0.1, 0.2, 0.3, 0.4},
		},
		{
			name:  "3x2_negatives",
			shape: []int{3, 2},
			in:    []float32{-3, -1, 0, 0, 5, 10},
			dOut:  []float32{0.1, -0.1, 0.2, -0.2, 0.3, -0.3},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			input, err := tensor.New[float32](tc.shape, append([]float32(nil), tc.in...))
			if err != nil {
				t.Fatalf("input tensor: %v", err)
			}
			dOut, err := tensor.New[float32](tc.shape, append([]float32(nil), tc.dOut...))
			if err != nil {
				t.Fatalf("dOut tensor: %v", err)
			}

			// Layer path.
			sm := activations.NewSoftmax[float32](engine, -1)
			y, err := sm.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Softmax.Forward: %v", err)
			}
			grads, err := sm.Backward(ctx, types.FullBackprop, dOut)
			if err != nil {
				t.Fatalf("Softmax.Backward: %v", err)
			}
			if len(grads) != 1 {
				t.Fatalf("expected 1 gradient from layer, got %d", len(grads))
			}
			gotLayer := grads[0].Data()

			// Functional path, fed the same softmax output.
			ref, err := functional.SoftmaxBackward[float32](ctx, engine, ops, dOut, y)
			if err != nil {
				t.Fatalf("functional.SoftmaxBackward: %v", err)
			}
			gotFn := ref.Data()

			if len(gotLayer) != len(gotFn) {
				t.Fatalf("length mismatch: layer=%d functional=%d", len(gotLayer), len(gotFn))
			}
			const tol = 1e-6
			for i := range gotLayer {
				if math.Abs(float64(gotLayer[i]-gotFn[i])) > tol {
					t.Errorf("dInput[%d]: layer=%v functional=%v (|delta|=%v > tol=%v)",
						i, gotLayer[i], gotFn[i],
						math.Abs(float64(gotLayer[i]-gotFn[i])), tol)
				}
			}
		})
	}
}
