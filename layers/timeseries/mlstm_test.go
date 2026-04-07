package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestMLSTM_OutputShape(t *testing.T) {
	engine := makeEngine()
	inputDim, hiddenDim := 4, 3
	mlstm, err := NewMLSTM[float32](engine, inputDim, hiddenDim)
	if err != nil {
		t.Fatalf("NewMLSTM: %v", err)
	}

	batch := 2
	x, _ := tensor.New[float32]([]int{batch, inputDim}, make([]float32, batch*inputDim))
	hPrev, _ := tensor.New[float32]([]int{batch, hiddenDim}, make([]float32, batch*hiddenDim))
	cPrev, _ := tensor.New[float32]([]int{batch, hiddenDim, hiddenDim}, make([]float32, batch*hiddenDim*hiddenDim))
	nPrev, _ := tensor.New[float32]([]int{batch, hiddenDim}, make([]float32, batch*hiddenDim))

	h, c, n, _, err := mlstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	if got := h.Shape(); len(got) != 2 || got[0] != batch || got[1] != hiddenDim {
		t.Errorf("h shape = %v, want [%d, %d]", got, batch, hiddenDim)
	}
	if got := c.Shape(); len(got) != 3 || got[0] != batch || got[1] != hiddenDim || got[2] != hiddenDim {
		t.Errorf("C shape = %v, want [%d, %d, %d]", got, batch, hiddenDim, hiddenDim)
	}
	if got := n.Shape(); len(got) != 2 || got[0] != batch || got[1] != hiddenDim {
		t.Errorf("n shape = %v, want [%d, %d]", got, batch, hiddenDim)
	}
}

func TestMLSTM_OuterProductUpdate(t *testing.T) {
	// Verify the covariance memory update on a small matrix with known weights.
	// inputDim=2, hiddenDim=2, batch=1.
	engine := makeEngine()
	mlstm, err := NewMLSTM[float32](engine, 2, 2)
	if err != nil {
		t.Fatalf("NewMLSTM: %v", err)
	}

	// Set known weights.
	// Wk = [[1, 0], [0, 1]] (identity) → k = x
	mlstm.Wk.Value.Data()[0] = 1
	mlstm.Wk.Value.Data()[1] = 0
	mlstm.Wk.Value.Data()[2] = 0
	mlstm.Wk.Value.Data()[3] = 1

	// Wv = [[1, 0], [0, 1]] (identity) → v = x
	mlstm.Wv.Value.Data()[0] = 1
	mlstm.Wv.Value.Data()[1] = 0
	mlstm.Wv.Value.Data()[2] = 0
	mlstm.Wv.Value.Data()[3] = 1

	// Wq = [[1, 0], [0, 1]] (identity) → q = x
	mlstm.Wq.Value.Data()[0] = 1
	mlstm.Wq.Value.Data()[1] = 0
	mlstm.Wq.Value.Data()[2] = 0
	mlstm.Wq.Value.Data()[3] = 1

	// Gate weights: all zero so pre-activations come only from bias.
	mlstm.Wi.Value.Data()[0] = 0
	mlstm.Wi.Value.Data()[1] = 0
	mlstm.Wf.Value.Data()[0] = 0
	mlstm.Wf.Value.Data()[1] = 0
	mlstm.Wo.Value.Data()[0] = 0
	mlstm.Wo.Value.Data()[1] = 0

	// Biases: i=0 → exp(0)=1, f=0 → exp(0)=1, o=0 → sigmoid(0)=0.5
	mlstm.Bi.Value.Data()[0] = 0
	mlstm.Bi.Value.Data()[1] = 0
	mlstm.Bf.Value.Data()[0] = 0
	mlstm.Bf.Value.Data()[1] = 0
	mlstm.Bo.Value.Data()[0] = 0
	mlstm.Bo.Value.Data()[1] = 0

	// Input: x = [1, 2]
	x, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 2})
	hPrev, _ := tensor.New[float32]([]int{1, 2}, []float32{0, 0})

	// cPrev = zero matrix [1, 2, 2]
	cPrev, _ := tensor.New[float32]([]int{1, 2, 2}, []float32{0, 0, 0, 0})
	// nPrev = [0, 0]
	nPrev, _ := tensor.New[float32]([]int{1, 2}, []float32{0, 0})

	_, cNew, nNew, _, err := mlstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// k = [1, 2], v = [1, 2]
	// i = exp(0) = 1, f = exp(0) = 1
	// C = 1 * 0 + 1 * v*k^T = [[1*1, 1*2], [2*1, 2*2]] = [[1, 2], [2, 4]]
	wantC := []float64{1, 2, 2, 4}
	cData := cNew.Data()
	for i, want := range wantC {
		if diff := math.Abs(float64(cData[i]) - want); diff > 1e-4 {
			t.Errorf("C[%d] = %f, want %f (diff %e)", i, cData[i], want, diff)
		}
	}

	// n = 1 * 0 + 1 * k = [1, 2]
	wantN := []float64{1, 2}
	nNewData := nNew.Data()
	for i, want := range wantN {
		if diff := math.Abs(float64(nNewData[i]) - want); diff > 1e-4 {
			t.Errorf("n[%d] = %f, want %f (diff %e)", i, nNewData[i], want, diff)
		}
	}
}

func TestMLSTM_ManualComputation(t *testing.T) {
	// Full hand-computed example: inputDim=2, hiddenDim=2, batch=1.
	engine := makeEngine()
	mlstm, err := NewMLSTM[float32](engine, 2, 2)
	if err != nil {
		t.Fatalf("NewMLSTM: %v", err)
	}

	// Set identity projections.
	mlstm.Wk.Value.Data()[0] = 1
	mlstm.Wk.Value.Data()[1] = 0
	mlstm.Wk.Value.Data()[2] = 0
	mlstm.Wk.Value.Data()[3] = 1

	mlstm.Wv.Value.Data()[0] = 1
	mlstm.Wv.Value.Data()[1] = 0
	mlstm.Wv.Value.Data()[2] = 0
	mlstm.Wv.Value.Data()[3] = 1

	mlstm.Wq.Value.Data()[0] = 1
	mlstm.Wq.Value.Data()[1] = 0
	mlstm.Wq.Value.Data()[2] = 0
	mlstm.Wq.Value.Data()[3] = 1

	// Gate weights = 0, biases: i=0.5, f=-0.5, o=1.0
	mlstm.Wi.Value.Data()[0] = 0
	mlstm.Wi.Value.Data()[1] = 0
	mlstm.Wf.Value.Data()[0] = 0
	mlstm.Wf.Value.Data()[1] = 0
	mlstm.Wo.Value.Data()[0] = 0
	mlstm.Wo.Value.Data()[1] = 0

	mlstm.Bi.Value.Data()[0] = 0.5
	mlstm.Bi.Value.Data()[1] = 0
	mlstm.Bf.Value.Data()[0] = -0.5
	mlstm.Bf.Value.Data()[1] = 0
	mlstm.Bo.Value.Data()[0] = 1.0
	mlstm.Bo.Value.Data()[1] = 0

	// x = [0.3, 0.7], cPrev = [[0.1, 0.2], [0.3, 0.4]], nPrev = [1, 1]
	x, _ := tensor.New[float32]([]int{1, 2}, []float32{0.3, 0.7})
	hPrev, _ := tensor.New[float32]([]int{1, 2}, []float32{0, 0})
	cPrev, _ := tensor.New[float32]([]int{1, 2, 2}, []float32{0.1, 0.2, 0.3, 0.4})
	nPrev, _ := tensor.New[float32]([]int{1, 2}, []float32{1.0, 1.0})

	hNew, cNew, nNew, _, err := mlstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Hand computation (stabilized form, mPrev=0):
	// k = [0.3, 0.7], v = [0.3, 0.7], q = [0.3, 0.7]
	// preI = 0 + 0.5 = 0.5 (bias[0]), preF = 0 + (-0.5) = -0.5, preO = 0 + 1.0 = 1.0
	// m = max(preF + mPrev, preI) = max(-0.5 + 0, 0.5) = 0.5
	// iGate = exp(preI - m) = exp(0) = 1
	// fGate = exp(preF + mPrev - m) = exp(-0.5 + 0 - 0.5) = exp(-1)
	iGate := 1.0
	fGate := math.Exp(-1.0)
	oGate := 1.0 / (1.0 + math.Exp(-1.0))

	// C = f * C_prev + i * (v * k^T)
	// v*k^T = [[0.3*0.3, 0.3*0.7], [0.7*0.3, 0.7*0.7]] = [[0.09, 0.21], [0.21, 0.49]]
	wantC := [4]float64{
		fGate*0.1 + iGate*0.09,
		fGate*0.2 + iGate*0.21,
		fGate*0.3 + iGate*0.21,
		fGate*0.4 + iGate*0.49,
	}

	// n = f * nPrev + i * k = [f*1 + i*0.3, f*1 + i*0.7]
	wantN := [2]float64{
		fGate*1.0 + iGate*0.3,
		fGate*1.0 + iGate*0.7,
	}

	// C * q: row 0 = wantC[0]*0.3 + wantC[1]*0.7, row 1 = wantC[2]*0.3 + wantC[3]*0.7
	cq0 := wantC[0]*0.3 + wantC[1]*0.7
	cq1 := wantC[2]*0.3 + wantC[3]*0.7

	// n^T * q = wantN[0]*0.3 + wantN[1]*0.7
	nq := wantN[0]*0.3 + wantN[1]*0.7
	denom := math.Max(math.Abs(nq), 1.0)

	wantH := [2]float64{
		oGate * cq0 / denom,
		oGate * cq1 / denom,
	}

	tol := 1e-4
	cData := cNew.Data()
	for i, want := range wantC {
		if diff := math.Abs(float64(cData[i]) - want); diff > tol {
			t.Errorf("C[%d] = %f, want %f (diff %e)", i, cData[i], want, diff)
		}
	}

	nNewData := nNew.Data()
	for i, want := range wantN {
		if diff := math.Abs(float64(nNewData[i]) - want); diff > tol {
			t.Errorf("n[%d] = %f, want %f (diff %e)", i, nNewData[i], want, diff)
		}
	}

	hData := hNew.Data()
	for i, want := range wantH {
		if diff := math.Abs(float64(hData[i]) - want); diff > tol {
			t.Errorf("h[%d] = %f, want %f (diff %e)", i, hData[i], want, diff)
		}
	}
}

func TestMLSTM_ExponentialGatingClamp(t *testing.T) {
	engine := makeEngine()
	mlstm, err := NewMLSTM[float32](engine, 1, 2)
	if err != nil {
		t.Fatalf("NewMLSTM: %v", err)
	}

	// Set large gate weights to force overflow without clamping.
	mlstm.Wi.Value.Data()[0] = 1000.0
	mlstm.Wf.Value.Data()[0] = 1000.0
	mlstm.Wo.Value.Data()[0] = 0.0

	// Projections = identity-like.
	mlstm.Wk.Value.Data()[0] = 1
	mlstm.Wk.Value.Data()[1] = 0
	mlstm.Wv.Value.Data()[0] = 1
	mlstm.Wv.Value.Data()[1] = 0
	mlstm.Wq.Value.Data()[0] = 1
	mlstm.Wq.Value.Data()[1] = 0

	for _, p := range []*tensor.TensorNumeric[float32]{
		mlstm.Bi.Value, mlstm.Bf.Value, mlstm.Bo.Value,
	} {
		for i := range p.Data() {
			p.Data()[i] = 0
		}
	}

	x, _ := tensor.New[float32]([]int{1, 1}, []float32{1.0})
	hPrev, _ := tensor.New[float32]([]int{1, 2}, []float32{0, 0})
	cPrev, _ := tensor.New[float32]([]int{1, 2, 2}, make([]float32, 4))
	nPrev, _ := tensor.New[float32]([]int{1, 2}, []float32{1, 1})

	h, c, n, _, err := mlstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// All outputs must be finite.
	for _, val := range h.Data() {
		if v := float64(val); math.IsInf(v, 0) || math.IsNaN(v) {
			t.Errorf("h contains non-finite value: %f", val)
		}
	}
	for _, val := range c.Data() {
		if v := float64(val); math.IsInf(v, 0) || math.IsNaN(v) {
			t.Errorf("C contains non-finite value: %f", val)
		}
	}
	for _, val := range n.Data() {
		if v := float64(val); math.IsInf(v, 0) || math.IsNaN(v) {
			t.Errorf("n contains non-finite value: %f", val)
		}
	}
}

func TestMLSTM_BatchIndependence(t *testing.T) {
	engine := makeEngine()
	inputDim, hiddenDim := 3, 2
	mlstm, err := NewMLSTM[float32](engine, inputDim, hiddenDim)
	if err != nil {
		t.Fatalf("NewMLSTM: %v", err)
	}

	// Create batched input (batch=2).
	xData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	hData := []float32{0.01, 0.02, 0.03, 0.04}
	cData := []float32{
		0.1, 0.2, 0.3, 0.4, // batch 0: 2x2 matrix
		0.5, 0.6, 0.7, 0.8, // batch 1: 2x2 matrix
	}
	nData := []float32{1.0, 1.0, 1.0, 1.0}

	x, _ := tensor.New[float32]([]int{2, inputDim}, xData)
	hPrev, _ := tensor.New[float32]([]int{2, hiddenDim}, hData)
	cPrev, _ := tensor.New[float32]([]int{2, hiddenDim, hiddenDim}, cData)
	nPrev, _ := tensor.New[float32]([]int{2, hiddenDim}, nData)

	hBatch, cBatch, nBatch, _, err := mlstm.Forward(context.Background(), x, hPrev, cPrev, nPrev, nil)
	if err != nil {
		t.Fatalf("Forward batched: %v", err)
	}

	// Run each batch element individually and compare.
	d := hiddenDim
	for b := 0; b < 2; b++ {
		xSingle, _ := tensor.New[float32]([]int{1, inputDim}, xData[b*inputDim:(b+1)*inputDim])
		hSingle, _ := tensor.New[float32]([]int{1, hiddenDim}, hData[b*hiddenDim:(b+1)*hiddenDim])
		cSingle, _ := tensor.New[float32]([]int{1, hiddenDim, hiddenDim}, cData[b*d*d:(b+1)*d*d])
		nSingle, _ := tensor.New[float32]([]int{1, hiddenDim}, nData[b*hiddenDim:(b+1)*hiddenDim])

		hs, cs, ns, _, err := mlstm.Forward(context.Background(), xSingle, hSingle, cSingle, nSingle, nil)
		if err != nil {
			t.Fatalf("Forward single batch %d: %v", b, err)
		}

		for j := 0; j < hiddenDim; j++ {
			idx := b*hiddenDim + j
			if math.Abs(float64(hBatch.Data()[idx]-hs.Data()[j])) > 1e-5 {
				t.Errorf("batch %d, h[%d]: batched=%f, single=%f", b, j, hBatch.Data()[idx], hs.Data()[j])
			}
		}
		for i := 0; i < d*d; i++ {
			batchIdx := b*d*d + i
			if math.Abs(float64(cBatch.Data()[batchIdx]-cs.Data()[i])) > 1e-5 {
				t.Errorf("batch %d, C[%d]: batched=%f, single=%f", b, i, cBatch.Data()[batchIdx], cs.Data()[i])
			}
		}
		for j := 0; j < hiddenDim; j++ {
			idx := b*hiddenDim + j
			if math.Abs(float64(nBatch.Data()[idx]-ns.Data()[j])) > 1e-5 {
				t.Errorf("batch %d, n[%d]: batched=%f, single=%f", b, j, nBatch.Data()[idx], ns.Data()[j])
			}
		}
	}
}

func TestNewMLSTM_InvalidArgs(t *testing.T) {
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
			_, err := NewMLSTM[float32](engine, tt.inputDim, tt.hiddenDim)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestMLSTM_ForwardInputValidation(t *testing.T) {
	engine := makeEngine()
	mlstm, err := NewMLSTM[float32](engine, 3, 2)
	if err != nil {
		t.Fatalf("NewMLSTM: %v", err)
	}

	ctx := context.Background()

	t.Run("wrong x inputDim", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 5}, make([]float32, 5))
		h, _ := tensor.New[float32]([]int{1, 2}, make([]float32, 2))
		c, _ := tensor.New[float32]([]int{1, 2, 2}, make([]float32, 4))
		n, _ := tensor.New[float32]([]int{1, 2}, make([]float32, 2))
		_, _, _, _, err := mlstm.Forward(ctx, x, h, c, n, nil)
		if err == nil {
			t.Error("expected error for wrong inputDim")
		}
	})

	t.Run("wrong hPrev shape", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 3}, make([]float32, 3))
		h, _ := tensor.New[float32]([]int{1, 5}, make([]float32, 5))
		c, _ := tensor.New[float32]([]int{1, 2, 2}, make([]float32, 4))
		n, _ := tensor.New[float32]([]int{1, 2}, make([]float32, 2))
		_, _, _, _, err := mlstm.Forward(ctx, x, h, c, n, nil)
		if err == nil {
			t.Error("expected error for wrong hPrev shape")
		}
	})

	t.Run("wrong cPrev shape", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 3}, make([]float32, 3))
		h, _ := tensor.New[float32]([]int{1, 2}, make([]float32, 2))
		c, _ := tensor.New[float32]([]int{1, 3, 3}, make([]float32, 9))
		n, _ := tensor.New[float32]([]int{1, 2}, make([]float32, 2))
		_, _, _, _, err := mlstm.Forward(ctx, x, h, c, n, nil)
		if err == nil {
			t.Error("expected error for wrong cPrev shape")
		}
	})

	t.Run("wrong nPrev shape", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 3}, make([]float32, 3))
		h, _ := tensor.New[float32]([]int{1, 2}, make([]float32, 2))
		c, _ := tensor.New[float32]([]int{1, 2, 2}, make([]float32, 4))
		n, _ := tensor.New[float32]([]int{1, 5}, make([]float32, 5))
		_, _, _, _, err := mlstm.Forward(ctx, x, h, c, n, nil)
		if err == nil {
			t.Error("expected error for wrong nPrev shape")
		}
	})
}

func TestMLSTM_OpTypeAndAttributes(t *testing.T) {
	engine := makeEngine()
	mlstm, err := NewMLSTM[float32](engine, 4, 3)
	if err != nil {
		t.Fatalf("NewMLSTM: %v", err)
	}
	if got := mlstm.OpType(); got != "MLSTM" {
		t.Errorf("OpType = %q, want %q", got, "MLSTM")
	}
	attrs := mlstm.Attributes()
	if attrs["input_dim"] != 4 {
		t.Errorf("input_dim = %v, want 4", attrs["input_dim"])
	}
	if attrs["hidden_dim"] != 3 {
		t.Errorf("hidden_dim = %v, want 3", attrs["hidden_dim"])
	}
}

func TestMLSTM_Parameters(t *testing.T) {
	engine := makeEngine()
	mlstm, err := NewMLSTM[float32](engine, 3, 2)
	if err != nil {
		t.Fatalf("NewMLSTM: %v", err)
	}

	params := mlstm.Parameters()
	// 3 projections + 3 gate weights + 3 biases = 9 parameters.
	if len(params) != 9 {
		t.Errorf("expected 9 parameters, got %d", len(params))
	}

	names := make(map[string]bool)
	for _, p := range params {
		names[p.Name] = true
	}
	for _, want := range []string{
		"mlstm_Wk", "mlstm_Wv", "mlstm_Wq",
		"mlstm_Wi", "mlstm_Wf", "mlstm_Wo",
		"mlstm_bi", "mlstm_bf", "mlstm_bo",
	} {
		if !names[want] {
			t.Errorf("missing parameter %q", want)
		}
	}
}
