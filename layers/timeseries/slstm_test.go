package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestSLSTM_OutputShape(t *testing.T) {
	engine := makeEngine()
	inputDim, hiddenDim := 4, 3
	slstm, err := NewSLSTM[float32](engine, inputDim, hiddenDim)
	if err != nil {
		t.Fatalf("NewSLSTM: %v", err)
	}

	batch := 2
	x, _ := tensor.New[float32]([]int{batch, inputDim}, make([]float32, batch*inputDim))
	hPrev, _ := tensor.New[float32]([]int{batch, hiddenDim}, make([]float32, batch*hiddenDim))
	cPrev, _ := tensor.New[float32]([]int{batch, hiddenDim}, make([]float32, batch*hiddenDim))
	nPrev, _ := tensor.New[float32]([]int{batch, hiddenDim}, make([]float32, batch*hiddenDim))

	// Initialise nPrev to 1 to avoid division by zero.
	for i := range nPrev.Data() {
		nPrev.Data()[i] = 1.0
	}

	h, c, n, _, err := slstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	for _, pair := range []struct {
		name string
		t    *tensor.TensorNumeric[float32]
	}{
		{"h", h}, {"c", c}, {"n", n},
	} {
		got := pair.t.Shape()
		want := []int{batch, hiddenDim}
		if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
			t.Errorf("%s shape = %v, want %v", pair.name, got, want)
		}
	}
}

func TestSLSTM_ManualComputation(t *testing.T) {
	// Verify against a hand-computed example with inputDim=1, hiddenDim=1, batch=1.
	engine := makeEngine()
	slstm, err := NewSLSTM[float32](engine, 1, 1)
	if err != nil {
		t.Fatalf("NewSLSTM: %v", err)
	}

	// Set known weights: all W=1, all R=0.5, all biases=0.
	for _, p := range []*tensor.TensorNumeric[float32]{
		slstm.Wi.Value, slstm.Wf.Value, slstm.Wz.Value, slstm.Wo.Value,
	} {
		p.Data()[0] = 1.0
	}
	for _, p := range []*tensor.TensorNumeric[float32]{
		slstm.Ri.Value, slstm.Rf.Value, slstm.Rz.Value, slstm.Ro.Value,
	} {
		p.Data()[0] = 0.5
	}
	for _, p := range []*tensor.TensorNumeric[float32]{
		slstm.Bi.Value, slstm.Bf.Value, slstm.Bz.Value, slstm.Bo.Value,
	} {
		p.Data()[0] = 0.0
	}

	// Input: x=0.5, hPrev=0.1, cPrev=0.2, nPrev=1.0
	x, _ := tensor.New[float32]([]int{1, 1}, []float32{0.5})
	hPrev, _ := tensor.New[float32]([]int{1, 1}, []float32{0.1})
	cPrev, _ := tensor.New[float32]([]int{1, 1}, []float32{0.2})
	nPrev, _ := tensor.New[float32]([]int{1, 1}, []float32{1.0})

	h, c, n, _, err := slstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Hand computation (stabilized form, mPrev=0):
	// pre = W*x + R*h + b = 1*0.5 + 0.5*0.1 + 0 = 0.55  (same for all gates)
	// m = max(preF + mPrev, preI) = max(0.55 + 0, 0.55) = 0.55
	// iGate = exp(preI - m) = exp(0) = 1
	// fGate = exp(preF + mPrev - m) = exp(0) = 1
	pre := 1.0*0.5 + 0.5*0.1
	iGate := 1.0                                // exp(preI - m)
	fGate := 1.0                                // exp(preF + mPrev - m)
	zVal := math.Tanh(pre)                      // tanh(0.55)
	oGate := 1.0 / (1.0 + math.Exp(-pre))      // sigmoid(0.55)

	wantN := fGate*1.0 + iGate                  // f*nPrev + i = 2
	wantC := fGate*0.2 + iGate*zVal             // f*cPrev + i*z
	wantH := oGate * (wantC / wantN)            // o * (c/n)

	tol := 1e-4
	if diff := math.Abs(float64(h.Data()[0]) - wantH); diff > tol {
		t.Errorf("h = %f, want %f (diff %e)", h.Data()[0], wantH, diff)
	}
	if diff := math.Abs(float64(c.Data()[0]) - wantC); diff > tol {
		t.Errorf("c = %f, want %f (diff %e)", c.Data()[0], wantC, diff)
	}
	if diff := math.Abs(float64(n.Data()[0]) - wantN); diff > tol {
		t.Errorf("n = %f, want %f (diff %e)", n.Data()[0], wantN, diff)
	}
}

func TestSLSTM_ExponentialGatingClamp(t *testing.T) {
	// Verify that large pre-activations are clamped and don't produce Inf/NaN.
	engine := makeEngine()
	slstm, err := NewSLSTM[float32](engine, 1, 1)
	if err != nil {
		t.Fatalf("NewSLSTM: %v", err)
	}

	// Set very large weights to force pre-activations way beyond safe range.
	slstm.Wi.Value.Data()[0] = 1000.0
	slstm.Wf.Value.Data()[0] = 1000.0
	slstm.Wz.Value.Data()[0] = 0.0
	slstm.Wo.Value.Data()[0] = 0.0
	for _, p := range []*tensor.TensorNumeric[float32]{
		slstm.Ri.Value, slstm.Rf.Value, slstm.Rz.Value, slstm.Ro.Value,
	} {
		p.Data()[0] = 0.0
	}
	for _, p := range []*tensor.TensorNumeric[float32]{
		slstm.Bi.Value, slstm.Bf.Value, slstm.Bz.Value, slstm.Bo.Value,
	} {
		p.Data()[0] = 0.0
	}

	x, _ := tensor.New[float32]([]int{1, 1}, []float32{1.0})
	hPrev, _ := tensor.New[float32]([]int{1, 1}, []float32{0.0})
	cPrev, _ := tensor.New[float32]([]int{1, 1}, []float32{0.0})
	nPrev, _ := tensor.New[float32]([]int{1, 1}, []float32{1.0})

	h, c, n, _, err := slstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// All outputs must be finite (no Inf, no NaN).
	for _, pair := range []struct {
		name string
		val  float32
	}{
		{"h", h.Data()[0]},
		{"c", c.Data()[0]},
		{"n", n.Data()[0]},
	} {
		v := float64(pair.val)
		if math.IsInf(v, 0) || math.IsNaN(v) {
			t.Errorf("%s = %f, expected finite value", pair.name, pair.val)
		}
	}

	// Stabilized form: with Wi=Wf=1000 clamped to maxGatePreAct, both preI and
	// preF equal maxGatePreAct, so m = max(preF+0, preI) = maxGatePreAct and
	// both gates = exp(0) = 1. With nPrev=1 and cPrev=0:
	//   n = 1*1 + 1 = 2
	//   c = 1*0 + 1*tanh(0) = 0     (Wz=0, x=1 → preZ=0)
	//   h = sigmoid(0) * (0/2) = 0
	// The key invariant — no Inf/NaN — is checked above. Here we verify the
	// gates are bounded to 1 (paper stabilization property).
	expectedN := 2.0
	if diff := math.Abs(float64(n.Data()[0]) - expectedN); diff > 1e-4 {
		t.Errorf("n = %e, want %e (stabilized gates bounded to 1)", n.Data()[0], expectedN)
	}
}

func TestSLSTM_BatchIndependence(t *testing.T) {
	engine := makeEngine()
	inputDim, hiddenDim := 3, 2
	slstm, err := NewSLSTM[float32](engine, inputDim, hiddenDim)
	if err != nil {
		t.Fatalf("NewSLSTM: %v", err)
	}

	// Create batched input (batch=2).
	xData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	hData := []float32{0.01, 0.02, 0.03, 0.04}
	cData := []float32{0.1, 0.2, 0.3, 0.4}
	nData := []float32{1.0, 1.0, 1.0, 1.0}

	x, _ := tensor.New[float32]([]int{2, inputDim}, xData)
	hPrev, _ := tensor.New[float32]([]int{2, hiddenDim}, hData)
	cPrev, _ := tensor.New[float32]([]int{2, hiddenDim}, cData)
	nPrev, _ := tensor.New[float32]([]int{2, hiddenDim}, nData)

	hBatch, cBatch, nBatch, _, err := slstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward batched: %v", err)
	}

	// Run each batch element individually and compare.
	for b := 0; b < 2; b++ {
		xSingle, _ := tensor.New[float32]([]int{1, inputDim}, xData[b*inputDim:(b+1)*inputDim])
		hSingle, _ := tensor.New[float32]([]int{1, hiddenDim}, hData[b*hiddenDim:(b+1)*hiddenDim])
		cSingle, _ := tensor.New[float32]([]int{1, hiddenDim}, cData[b*hiddenDim:(b+1)*hiddenDim])
		nSingle, _ := tensor.New[float32]([]int{1, hiddenDim}, nData[b*hiddenDim:(b+1)*hiddenDim])

		hs, cs, ns, _, err := slstm.Forward(context.Background(), xSingle, hSingle, cSingle, nSingle, nil)
		if err != nil {
			t.Fatalf("Forward single batch %d: %v", b, err)
		}

		for j := 0; j < hiddenDim; j++ {
			idx := b*hiddenDim + j
			if math.Abs(float64(hBatch.Data()[idx]-hs.Data()[j])) > 1e-5 {
				t.Errorf("batch %d, h[%d]: batched=%f, single=%f", b, j, hBatch.Data()[idx], hs.Data()[j])
			}
			if math.Abs(float64(cBatch.Data()[idx]-cs.Data()[j])) > 1e-5 {
				t.Errorf("batch %d, c[%d]: batched=%f, single=%f", b, j, cBatch.Data()[idx], cs.Data()[j])
			}
			if math.Abs(float64(nBatch.Data()[idx]-ns.Data()[j])) > 1e-5 {
				t.Errorf("batch %d, n[%d]: batched=%f, single=%f", b, j, nBatch.Data()[idx], ns.Data()[j])
			}
		}
	}
}

func TestNewSLSTM_InvalidArgs(t *testing.T) {
	engine := makeEngine()

	tests := []struct {
		name      string
		inputDim  int
		hiddenDim int
	}{
		{"zero inputDim", 0, 4},
		{"negative inputDim", -1, 4},
		{"zero hiddenDim", 3, 0},
		{"negative hiddenDim", 3, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSLSTM[float32](engine, tt.inputDim, tt.hiddenDim)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestSLSTM_ForwardInputValidation(t *testing.T) {
	engine := makeEngine()
	slstm, err := NewSLSTM[float32](engine, 3, 2)
	if err != nil {
		t.Fatalf("NewSLSTM: %v", err)
	}

	ctx := context.Background()
	good := func(shape []int, size int) *tensor.TensorNumeric[float32] {
		t, _ := tensor.New[float32](shape, make([]float32, size))
		return t
	}

	t.Run("wrong x rank", func(t *testing.T) {
		x := good([]int{1, 2, 3}, 6)
		h := good([]int{1, 2}, 2)
		c := good([]int{1, 2}, 2)
		n := good([]int{1, 2}, 2)
		_, _, _, _, err := slstm.Forward(ctx, x, h, c, n, nil)
		if err == nil {
			t.Error("expected error for 3D x")
		}
	})

	t.Run("wrong x inputDim", func(t *testing.T) {
		x := good([]int{1, 5}, 5)
		h := good([]int{1, 2}, 2)
		c := good([]int{1, 2}, 2)
		n := good([]int{1, 2}, 2)
		_, _, _, _, err := slstm.Forward(ctx, x, h, c, n, nil)
		if err == nil {
			t.Error("expected error for wrong inputDim")
		}
	})

	t.Run("wrong hPrev shape", func(t *testing.T) {
		x := good([]int{1, 3}, 3)
		h := good([]int{1, 5}, 5)
		c := good([]int{1, 2}, 2)
		n := good([]int{1, 2}, 2)
		_, _, _, _, err := slstm.Forward(ctx, x, h, c, n, nil)
		if err == nil {
			t.Error("expected error for wrong hPrev shape")
		}
	})
}

func TestSLSTM_OpTypeAndAttributes(t *testing.T) {
	engine := makeEngine()
	slstm, err := NewSLSTM[float32](engine, 4, 3)
	if err != nil {
		t.Fatalf("NewSLSTM: %v", err)
	}
	if got := slstm.OpType(); got != "SLSTM" {
		t.Errorf("OpType = %q, want %q", got, "SLSTM")
	}
	attrs := slstm.Attributes()
	if attrs["input_dim"] != 4 {
		t.Errorf("input_dim = %v, want 4", attrs["input_dim"])
	}
	if attrs["hidden_dim"] != 3 {
		t.Errorf("hidden_dim = %v, want 3", attrs["hidden_dim"])
	}
}

func TestSLSTM_Parameters(t *testing.T) {
	engine := makeEngine()
	slstm, err := NewSLSTM[float32](engine, 3, 2)
	if err != nil {
		t.Fatalf("NewSLSTM: %v", err)
	}

	params := slstm.Parameters()
	// 4 input weights + 4 recurrent weights + 4 biases = 12 parameters.
	if len(params) != 12 {
		t.Errorf("expected 12 parameters, got %d", len(params))
	}

	names := make(map[string]bool)
	for _, p := range params {
		names[p.Name] = true
	}
	for _, want := range []string{
		"slstm_Wi", "slstm_Wf", "slstm_Wz", "slstm_Wo",
		"slstm_Ri", "slstm_Rf", "slstm_Rz", "slstm_Ro",
		"slstm_bi", "slstm_bf", "slstm_bz", "slstm_bo",
	} {
		if !names[want] {
			t.Errorf("missing parameter %q", want)
		}
	}
}
