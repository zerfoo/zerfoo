package timeseries

import (
	"math"
	"testing"
)

func TestClipGradients(t *testing.T) {
	t.Run("clips when norm exceeds", func(t *testing.T) {
		grads := []float64{3.0, 4.0} // norm = 5
		clipGradients(grads, 2.5)
		norm := math.Sqrt(grads[0]*grads[0] + grads[1]*grads[1])
		if math.Abs(norm-2.5) > 1e-10 {
			t.Errorf("expected clipped norm 2.5, got %v", norm)
		}
		// Direction should be preserved: ratio 3:4.
		if math.Abs(grads[0]/grads[1]-0.75) > 1e-10 {
			t.Errorf("clipping changed direction: grads = %v", grads)
		}
	})

	t.Run("no-op when under", func(t *testing.T) {
		grads := []float64{1.0, 2.0} // norm ~ 2.236
		orig := make([]float64, len(grads))
		copy(orig, grads)
		clipGradients(grads, 5.0)
		for i := range grads {
			if grads[i] != orig[i] {
				t.Errorf("grads[%d] changed from %v to %v", i, orig[i], grads[i])
			}
		}
	})

	t.Run("disabled when maxNorm zero", func(t *testing.T) {
		grads := []float64{100.0, 200.0}
		orig := make([]float64, len(grads))
		copy(orig, grads)
		clipGradients(grads, 0)
		for i := range grads {
			if grads[i] != orig[i] {
				t.Errorf("grads[%d] changed from %v to %v", i, orig[i], grads[i])
			}
		}
	})
}

func TestAdamWUpdate(t *testing.T) {
	// One step of AdamW on a single parameter.
	p := 1.0
	params := []*float64{&p}
	grads := []float64{0.1}
	state := newAdamWState(1)
	cfg := TrainConfig{
		Beta1:       0.9,
		Beta2:       0.999,
		Epsilon:     1e-8,
		WeightDecay: 0.01,
	}
	lr := 0.001

	adamWUpdate(params, grads, state, lr, cfg, 1.0)

	// Compute expected value manually.
	m := 0.9*0.0 + 0.1*0.1   // 0.01
	v := 0.999*0.0 + 0.001*0.01 // 0.00001
	mHat := m / (1 - 0.9)    // 0.1
	vHat := v / (1 - 0.999)  // 0.01
	expected := 1.0 - lr*(mHat/(math.Sqrt(vHat)+1e-8)+0.01*1.0)

	if math.Abs(p-expected) > 1e-12 {
		t.Errorf("adamWUpdate: got %v, want %v", p, expected)
	}

	// Verify moments were stored.
	if math.Abs(state.m[0]-m) > 1e-15 {
		t.Errorf("m[0] = %v, want %v", state.m[0], m)
	}
	if math.Abs(state.v[0]-v) > 1e-15 {
		t.Errorf("v[0] = %v, want %v", state.v[0], v)
	}
}

func TestMSELossFlat(t *testing.T) {
	pred := []float64{1.0, 2.0, 3.0}
	target := []float64{1.5, 2.5, 3.5}

	loss, dPred := mseLossFlat(pred, target)

	// Each diff = -0.5, diff^2 = 0.25, sum = 0.75, MSE = 0.25.
	expectedLoss := 0.25
	if math.Abs(loss-expectedLoss) > 1e-12 {
		t.Errorf("loss = %v, want %v", loss, expectedLoss)
	}

	// dPred[i] = 2*(pred[i]-target[i])/n = 2*(-0.5)/3 = -1/3.
	for i, d := range dPred {
		expected := -1.0 / 3.0
		if math.Abs(d-expected) > 1e-12 {
			t.Errorf("dPred[%d] = %v, want %v", i, d, expected)
		}
	}
}

