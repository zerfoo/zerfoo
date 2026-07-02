package loss

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ---------------------------------------------------------------------------
// CorrLoss Forward engine error paths
// ---------------------------------------------------------------------------

func TestCorrLoss_Forward_EngineErrors(t *testing.T) {
	// CorrLoss.Forward calls engine ops in this order:
	// 1. ReduceMean (predictions)
	// 2. ReduceMean (targets)
	// 3. AddScalar (center predictions)
	// 4. AddScalar (center targets)
	// 5. Mul (pc*tc)
	// 6. Mul (pc*pc)
	// 7. Mul (tc*tc)
	// 8. Sum (pcTc)
	// 9. Sum (pcPc)
	// 10. Sum (tcTc)
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"ReduceMean_1st", map[string]int{"ReduceMean": 1}},
		{"ReduceMean_2nd", map[string]int{"ReduceMean": 2}},
		{"AddScalar_1st", map[string]int{"AddScalar": 1}},
		{"AddScalar_2nd", map[string]int{"AddScalar": 2}},
		{"Mul_1st_pcTc", map[string]int{"Mul": 1}},
		{"Mul_2nd_pcPc", map[string]int{"Mul": 2}},
		{"Mul_3rd_tcTc", map[string]int{"Mul": 3}},
		{"Sum_1st_sumPT", map[string]int{"Sum": 1}},
		{"Sum_2nd_sumPP", map[string]int{"Sum": 2}},
		{"Sum_3rd_sumTT", map[string]int{"Sum": 3}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fe := newFailingEngine(tc.failOn)
			ops := numeric.Float32Ops{}
			cl := NewCorrLoss[float32](fe, ops)

			preds, _ := tensor.New[float32]([]int{4}, []float32{0.1, 0.3, 0.7, 0.9})
			targets, _ := tensor.New[float32]([]int{4}, []float32{0.0, 0.2, 0.8, 1.0})

			_, err := cl.Forward(context.Background(), preds, targets)
			if err == nil {
				t.Errorf("expected error from %s", tc.name)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// CorrLoss Backward engine error paths
// ---------------------------------------------------------------------------

func TestCorrLoss_Backward_EngineErrors(t *testing.T) {
	// CorrLoss.Backward calls engine ops in this order:
	// 1. ReduceMean (preds)
	// 2. ReduceMean (targs)
	// 3. AddScalar (center preds)
	// 4. AddScalar (center targs)
	// 5. Mul (pc*tc)
	// 6. Mul (pc*pc)
	// 7. Mul (tc*tc)
	// 8. Sum (pcTc)
	// 9. Sum (pcPc)
	// 10. Sum (tcTc)
	// 11. MulScalar (term1)
	// 12. MulScalar (term2)
	// 13. Add (term1 + term2)
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"Backward_ReduceMean_1st", map[string]int{"ReduceMean": 1}},
		{"Backward_ReduceMean_2nd", map[string]int{"ReduceMean": 2}},
		{"Backward_AddScalar_1st", map[string]int{"AddScalar": 1}},
		{"Backward_AddScalar_2nd", map[string]int{"AddScalar": 2}},
		{"Backward_Mul_1st", map[string]int{"Mul": 1}},
		{"Backward_Mul_2nd", map[string]int{"Mul": 2}},
		{"Backward_Mul_3rd", map[string]int{"Mul": 3}},
		{"Backward_Sum_1st", map[string]int{"Sum": 1}},
		{"Backward_Sum_2nd", map[string]int{"Sum": 2}},
		{"Backward_Sum_3rd", map[string]int{"Sum": 3}},
		{"Backward_MulScalar_1st", map[string]int{"MulScalar": 1}},
		{"Backward_MulScalar_2nd", map[string]int{"MulScalar": 2}},
		{"Backward_Add_gradPred", map[string]int{"Add": 1}},
	}

	preds, _ := tensor.New[float32]([]int{4}, []float32{0.1, 0.3, 0.7, 0.9})
	targets, _ := tensor.New[float32]([]int{4}, []float32{0.0, 0.2, 0.8, 1.0})
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fe := newFailingEngine(tc.failOn)
			ops := numeric.Float32Ops{}
			cl := NewCorrLoss[float32](fe, ops)

			_, err := cl.Backward(context.Background(), types.FullBackprop, dOut, preds, targets)
			if err == nil {
				t.Errorf("expected error from %s", tc.name)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// MSE Forward engine error paths
// ---------------------------------------------------------------------------

func TestMSE_Forward_EngineErrors(t *testing.T) {
	// MSE.Forward calls: Sub, Mul
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"Sub_diff", map[string]int{"Sub": 1}},
		{"Mul_squared", map[string]int{"Mul": 1}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fe := newFailingEngine(tc.failOn)
			ops := numeric.Float32Ops{}
			mse := NewMSE[float32](fe, ops)

			preds, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
			targets, _ := tensor.New[float32]([]int{3}, []float32{1, 1, 1})

			_, err := mse.Forward(context.Background(), preds, targets)
			if err == nil {
				t.Errorf("expected error from %s", tc.name)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// MSE Backward engine error paths
// ---------------------------------------------------------------------------

func TestMSE_Backward_EngineErrors(t *testing.T) {
	// MSE.Backward calls: Sub, Mul
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"Sub_diff", map[string]int{"Sub": 1}},
		{"Mul_gradPred", map[string]int{"Mul": 1}},
	}

	preds, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	targets, _ := tensor.New[float32]([]int{3}, []float32{1, 1, 1})
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fe := newFailingEngine(tc.failOn)
			ops := numeric.Float32Ops{}
			mse := NewMSE[float32](fe, ops)

			_, err := mse.Backward(context.Background(), types.FullBackprop, dOut, preds, targets)
			if err == nil {
				t.Errorf("expected error from %s", tc.name)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// CrossEntropyLoss unsupported target type
// ---------------------------------------------------------------------------

func TestCrossEntropyLoss_Forward_WrongInputCount(t *testing.T) {
	fe := newFailingEngine(nil)
	cel := NewCrossEntropyLoss[float32](fe)

	pred, _ := tensor.New[float32]([]int{2, 3}, []float32{0.1, 0.2, 0.7, 0.3, 0.3, 0.4})
	_, err := cel.Forward(context.Background(), pred)
	if err == nil {
		t.Error("expected error for wrong input count")
	}
}
