package loss

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// oneHotEncode builds the [positions, classes] one-hot float encoding of
// integer labels, the caller-side preparation CrossEntropyLossOneHot expects.
func oneHotEncode(t *testing.T, labels []int, classes int) *tensor.TensorNumeric[float32] {
	t.Helper()
	data := make([]float32, len(labels)*classes)
	for i, l := range labels {
		data[i*classes+l] = 1
	}
	oh, err := tensor.New([]int{len(labels), classes}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	return oh
}

// TestCrossEntropyLossOneHot_MatchesCrossEntropyLoss: the device-op loss
// must reproduce the host-float64 CrossEntropyLoss: identical gradients
// (same Sub/MulScalar/Mul chain over the same softmax) and a loss value
// equal up to element-type rounding.
func TestCrossEntropyLossOneHot_MatchesCrossEntropyLoss(t *testing.T) {
	tests := []struct {
		name   string
		shape  []int
		logits []float32
		labels []int
	}{
		{
			name:   "wolf shape 4x3",
			shape:  []int{4, 3},
			logits: []float32{2, -1, 0.5, -3, 4, 0, 0.1, 0.1, 0.1, -2, -2, 5},
			labels: []int{0, 1, 2, 2},
		},
		{
			name:   "skewed logits",
			shape:  []int{2, 3},
			logits: []float32{30, -30, 0, -25, 25, 1},
			labels: []int{2, 0},
		},
		{
			name:   "single position",
			shape:  []int{1, 3},
			logits: []float32{0.3, 0.2, 0.5},
			labels: []int{1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

			preds, err := tensor.New(tt.shape, tt.logits)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}
			intTargets := make([]float32, len(tt.labels))
			for i, l := range tt.labels {
				intTargets[i] = float32(l)
			}
			labelT, err := tensor.New([]int{len(tt.labels)}, intTargets)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}
			onehot := oneHotEncode(t, tt.labels, tt.shape[1])

			ref := NewCrossEntropyLoss[float32](engine)
			refLoss, err := ref.Forward(ctx, preds, labelT)
			if err != nil {
				t.Fatalf("reference Forward: %v", err)
			}

			oh := NewCrossEntropyLossOneHot[float32](engine)
			gotLoss, err := oh.Forward(ctx, preds, onehot)
			if err != nil {
				t.Fatalf("one-hot Forward: %v", err)
			}
			if got := gotLoss.Shape(); len(got) != 1 || got[0] != 1 {
				t.Fatalf("loss shape = %v, want [1]", got)
			}

			refVal := float64(refLoss.Data()[0])
			gotVal := float64(gotLoss.Data()[0])
			if math.Abs(refVal-gotVal) > 1e-5*math.Max(1, math.Abs(refVal)) {
				t.Errorf("loss = %v, reference = %v", gotVal, refVal)
			}

			// Backward parity with the SAME upstream gradient (the strategy
			// passes the loss tensor itself as dOut).
			refGrads, err := ref.Backward(ctx, types.FullBackprop, refLoss)
			if err != nil {
				t.Fatalf("reference Backward: %v", err)
			}
			gotGrads, err := oh.Backward(ctx, types.FullBackprop, refLoss)
			if err != nil {
				t.Fatalf("one-hot Backward: %v", err)
			}
			if len(gotGrads) != 2 || gotGrads[1] != nil {
				t.Fatalf("Backward must return [grad, nil], got %d grads", len(gotGrads))
			}
			refG := refGrads[0].Data()
			gotG := gotGrads[0].Data()
			if len(refG) != len(gotG) {
				t.Fatalf("grad length %d != reference %d", len(gotG), len(refG))
			}
			for i := range refG {
				if diff := math.Abs(float64(refG[i] - gotG[i])); diff > 1e-7 {
					t.Errorf("grad[%d] = %v, reference %v (diff %g)", i, gotG[i], refG[i], diff)
				}
			}
		})
	}
}

// TestCrossEntropyLossOneHot_SaturatedClassStaysFinite: a class probability
// that underflows to zero must produce a finite loss (the logFloor guard),
// never -Inf/NaN -- the failure mode that motivated the host-float64 fused
// log-softmax in CrossEntropyLoss.
func TestCrossEntropyLossOneHot_SaturatedClassStaysFinite(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Target class logit is ~200 below the max: softmax(target) underflows
	// to exactly 0 in float32.
	preds, err := tensor.New([]int{1, 3}, []float32{200, 0, 0})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	onehot := oneHotEncode(t, []int{1}, 3)

	oh := NewCrossEntropyLossOneHot[float32](engine)
	lossT, err := oh.Forward(ctx, preds, onehot)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	v := float64(lossT.Data()[0])
	if math.IsInf(v, 0) || math.IsNaN(v) {
		t.Fatalf("loss = %v, want finite (logFloor guard)", v)
	}
	if v <= 0 {
		t.Fatalf("loss = %v, want large positive", v)
	}
}

// TestCrossEntropyLossOneHot_InputValidation rejects malformed inputs.
func TestCrossEntropyLossOneHot_InputValidation(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	oh := NewCrossEntropyLossOneHot[float32](engine)

	preds, err := tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	badTargets, err := tensor.New([]int{2, 2}, []float32{1, 0, 0, 1})
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	if _, err := oh.Forward(ctx, preds); err == nil {
		t.Error("Forward with 1 input must fail")
	}
	if _, err := oh.Forward(ctx, preds, badTargets); err == nil {
		t.Error("Forward with mismatched target shape must fail")
	}
	if _, err := oh.Backward(ctx, types.FullBackprop, nil); err == nil {
		t.Error("Backward before Forward must fail")
	}
}
