package timeseries

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// maxGatePreAct is the clamp threshold for gate pre-activations before exp()
// to prevent floating-point overflow. exp(20) ≈ 4.85e8 which is safe for
// float32; exp(89) would overflow.
const maxGatePreAct = 20.0

// SLSTM implements the sLSTM cell from the xLSTM paper (Beck et al., 2024).
//
// The sLSTM extends the classical LSTM with exponential gating and a scalar
// normalizer state that stabilises the cell when input and forget gates use
// exp() instead of sigmoid():
//
//	i_t = exp(Wi*x_t + Ri*h_{t-1} + bi)   — exponential input gate
//	f_t = exp(Wf*x_t + Rf*h_{t-1} + bf)   — exponential forget gate
//	z_t = tanh(Wz*x_t + Rz*h_{t-1} + bz)  — cell input
//	o_t = σ(Wo*x_t + Ro*h_{t-1} + bo)      — output gate (sigmoid)
//	n_t = f_t * n_{t-1} + i_t               — normalizer state
//	c_t = f_t * c_{t-1} + i_t * z_t         — cell state
//	h_t = o_t * (c_t / n_t)                 — hidden state
//
// Gate pre-activations for i and f are clamped to [-maxGatePreAct, maxGatePreAct]
// before applying exp() to prevent overflow.
type SLSTM[T tensor.Float] struct {
	engine compute.Engine[T]

	// Weight matrices: input projections [inputDim, hiddenDim]
	Wi *graph.Parameter[T]
	Wf *graph.Parameter[T]
	Wz *graph.Parameter[T]
	Wo *graph.Parameter[T]

	// Weight matrices: recurrent projections [hiddenDim, hiddenDim]
	Ri *graph.Parameter[T]
	Rf *graph.Parameter[T]
	Rz *graph.Parameter[T]
	Ro *graph.Parameter[T]

	// Biases [hiddenDim]
	Bi *graph.Parameter[T]
	Bf *graph.Parameter[T]
	Bz *graph.Parameter[T]
	Bo *graph.Parameter[T]

	inputDim  int
	hiddenDim int
}

// NewSLSTM creates a new sLSTM cell.
//
// Parameters:
//   - engine: the compute engine for tensor operations
//   - inputDim: dimensionality of the input vector at each time step
//   - hiddenDim: dimensionality of the hidden state
func NewSLSTM[T tensor.Float](engine compute.Engine[T], inputDim, hiddenDim int) (*SLSTM[T], error) {
	if inputDim <= 0 {
		return nil, fmt.Errorf("inputDim must be positive, got %d", inputDim)
	}
	if hiddenDim <= 0 {
		return nil, fmt.Errorf("hiddenDim must be positive, got %d", hiddenDim)
	}

	// Xavier/Glorot uniform initialization scale.
	inputScale := 1.0 / math.Sqrt(float64(inputDim))
	recurrentScale := 1.0 / math.Sqrt(float64(hiddenDim))

	makeInputWeight := func(name string) (*graph.Parameter[T], error) {
		data := make([]T, inputDim*hiddenDim)
		for i := range data {
			data[i] = T((rand.Float64()*2 - 1) * inputScale)
		}
		t, err := tensor.New[T]([]int{inputDim, hiddenDim}, data)
		if err != nil {
			return nil, fmt.Errorf("create %s tensor: %w", name, err)
		}
		return graph.NewParameter[T](name, t, tensor.New[T])
	}

	makeRecurrentWeight := func(name string) (*graph.Parameter[T], error) {
		data := make([]T, hiddenDim*hiddenDim)
		for i := range data {
			data[i] = T((rand.Float64()*2 - 1) * recurrentScale)
		}
		t, err := tensor.New[T]([]int{hiddenDim, hiddenDim}, data)
		if err != nil {
			return nil, fmt.Errorf("create %s tensor: %w", name, err)
		}
		return graph.NewParameter[T](name, t, tensor.New[T])
	}

	makeBias := func(name string) (*graph.Parameter[T], error) {
		data := make([]T, hiddenDim) // zero-initialised
		t, err := tensor.New[T]([]int{hiddenDim}, data)
		if err != nil {
			return nil, fmt.Errorf("create %s tensor: %w", name, err)
		}
		return graph.NewParameter[T](name, t, tensor.New[T])
	}

	wi, err := makeInputWeight("slstm_Wi")
	if err != nil {
		return nil, err
	}
	wf, err := makeInputWeight("slstm_Wf")
	if err != nil {
		return nil, err
	}
	wz, err := makeInputWeight("slstm_Wz")
	if err != nil {
		return nil, err
	}
	wo, err := makeInputWeight("slstm_Wo")
	if err != nil {
		return nil, err
	}

	ri, err := makeRecurrentWeight("slstm_Ri")
	if err != nil {
		return nil, err
	}
	rf, err := makeRecurrentWeight("slstm_Rf")
	if err != nil {
		return nil, err
	}
	rz, err := makeRecurrentWeight("slstm_Rz")
	if err != nil {
		return nil, err
	}
	ro, err := makeRecurrentWeight("slstm_Ro")
	if err != nil {
		return nil, err
	}

	bi, err := makeBias("slstm_bi")
	if err != nil {
		return nil, err
	}
	bf, err := makeBias("slstm_bf")
	if err != nil {
		return nil, err
	}
	bz, err := makeBias("slstm_bz")
	if err != nil {
		return nil, err
	}
	bo, err := makeBias("slstm_bo")
	if err != nil {
		return nil, err
	}

	return &SLSTM[T]{
		engine:    engine,
		Wi:        wi,
		Wf:        wf,
		Wz:        wz,
		Wo:        wo,
		Ri:        ri,
		Rf:        rf,
		Rz:        rz,
		Ro:        ro,
		Bi:        bi,
		Bf:        bf,
		Bz:        bz,
		Bo:        bo,
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
	}, nil
}

// OpType returns the operation type of the layer.
func (s *SLSTM[T]) OpType() string {
	return "SLSTM"
}

// Attributes returns the attributes of the layer.
func (s *SLSTM[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_dim":  s.inputDim,
		"hidden_dim": s.hiddenDim,
	}
}

// OutputShape returns the output shape of the layer.
func (s *SLSTM[T]) OutputShape() []int {
	return []int{-1, s.hiddenDim} // [batch, hiddenDim]
}

// Forward performs a single sLSTM time step.
//
// Inputs:
//   - x:     input vector        [batch, inputDim]
//   - hPrev: previous hidden state [batch, hiddenDim]
//   - cPrev: previous cell state   [batch, hiddenDim]
//   - nPrev: previous normalizer   [batch, hiddenDim]
//
// Returns (h, c, n) — the new hidden state, cell state, and normalizer.
func (s *SLSTM[T]) Forward(
	ctx context.Context,
	x, hPrev, cPrev, nPrev *tensor.TensorNumeric[T],
) (h, c, n *tensor.TensorNumeric[T], err error) {
	// Validate input shapes.
	xShape := x.Shape()
	if len(xShape) != 2 || xShape[1] != s.inputDim {
		return nil, nil, nil, fmt.Errorf("SLSTM: x must be [batch, %d], got shape %v", s.inputDim, xShape)
	}
	batch := xShape[0]

	for _, pair := range []struct {
		name string
		t    *tensor.TensorNumeric[T]
	}{
		{"hPrev", hPrev},
		{"cPrev", cPrev},
		{"nPrev", nPrev},
	} {
		shape := pair.t.Shape()
		if len(shape) != 2 || shape[0] != batch || shape[1] != s.hiddenDim {
			return nil, nil, nil, fmt.Errorf("SLSTM: %s must be [%d, %d], got shape %v",
				pair.name, batch, s.hiddenDim, shape)
		}
	}

	// Load parameter data slices for manual computation.
	wiData := s.Wi.Value.Data()
	wfData := s.Wf.Value.Data()
	wzData := s.Wz.Value.Data()
	woData := s.Wo.Value.Data()
	riData := s.Ri.Value.Data()
	rfData := s.Rf.Value.Data()
	rzData := s.Rz.Value.Data()
	roData := s.Ro.Value.Data()
	biData := s.Bi.Value.Data()
	bfData := s.Bf.Value.Data()
	bzData := s.Bz.Value.Data()
	boData := s.Bo.Value.Data()

	xData := x.Data()
	hData := hPrev.Data()
	cData := cPrev.Data()
	nData := nPrev.Data()

	hOut := make([]T, batch*s.hiddenDim)
	cOut := make([]T, batch*s.hiddenDim)
	nOut := make([]T, batch*s.hiddenDim)

	for b := 0; b < batch; b++ {
		xOff := b * s.inputDim
		hOff := b * s.hiddenDim

		for j := 0; j < s.hiddenDim; j++ {
			// Compute pre-activations: W*x + R*h + b
			var preI, preF, preZ, preO float64

			// Input projection: W[inputDim, hiddenDim] — x[b,:] dot W[:,j]
			for k := 0; k < s.inputDim; k++ {
				xk := float64(xData[xOff+k])
				preI += xk * float64(wiData[k*s.hiddenDim+j])
				preF += xk * float64(wfData[k*s.hiddenDim+j])
				preZ += xk * float64(wzData[k*s.hiddenDim+j])
				preO += xk * float64(woData[k*s.hiddenDim+j])
			}

			// Recurrent projection: R[hiddenDim, hiddenDim] — h[b,:] dot R[:,j]
			for k := 0; k < s.hiddenDim; k++ {
				hk := float64(hData[hOff+k])
				preI += hk * float64(riData[k*s.hiddenDim+j])
				preF += hk * float64(rfData[k*s.hiddenDim+j])
				preZ += hk * float64(rzData[k*s.hiddenDim+j])
				preO += hk * float64(roData[k*s.hiddenDim+j])
			}

			// Add biases.
			preI += float64(biData[j])
			preF += float64(bfData[j])
			preZ += float64(bzData[j])
			preO += float64(boData[j])

			// Clamp i and f pre-activations before exp().
			preI = clampFloat(preI, -maxGatePreAct, maxGatePreAct)
			preF = clampFloat(preF, -maxGatePreAct, maxGatePreAct)

			// Gate activations.
			iGate := math.Exp(preI)          // exponential input gate
			fGate := math.Exp(preF)          // exponential forget gate
			zVal := math.Tanh(preZ)          // cell input
			oGate := 1.0 / (1.0 + math.Exp(-preO)) // sigmoid output gate

			// State updates.
			cPrevVal := float64(cData[hOff+j])
			nPrevVal := float64(nData[hOff+j])

			nNew := fGate*nPrevVal + iGate
			cNew := fGate*cPrevVal + iGate*zVal

			// Hidden state: o * (c / n). Guard against division by zero.
			var hNew float64
			if nNew > 1e-12 {
				hNew = oGate * (cNew / nNew)
			}

			hOut[hOff+j] = T(hNew)
			cOut[hOff+j] = T(cNew)
			nOut[hOff+j] = T(nNew)
		}
	}

	h, err = tensor.New[T]([]int{batch, s.hiddenDim}, hOut)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: create h tensor: %w", err)
	}
	c, err = tensor.New[T]([]int{batch, s.hiddenDim}, cOut)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: create c tensor: %w", err)
	}
	n, err = tensor.New[T]([]int{batch, s.hiddenDim}, nOut)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: create n tensor: %w", err)
	}
	return h, c, n, nil
}

// Parameters returns all learnable parameters of the sLSTM cell.
func (s *SLSTM[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{
		s.Wi, s.Wf, s.Wz, s.Wo,
		s.Ri, s.Rf, s.Rz, s.Ro,
		s.Bi, s.Bf, s.Bz, s.Bo,
	}
}

// clampFloat clamps v to [lo, hi].
func clampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
