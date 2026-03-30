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

// MLSTM implements the mLSTM (Matrix LSTM) cell from the xLSTM paper
// (Beck et al., 2024, arXiv:2405.04517).
//
// The mLSTM replaces the scalar cell state of a classical LSTM with a matrix
// cell state (covariance memory) updated via outer products of key and value
// projections:
//
//	k_t = W_k * x_t                              — key projection
//	v_t = W_v * x_t                              — value projection
//	q_t = W_q * x_t                              — query projection
//	i_t = exp(clamp(w_i * x_t + b_i))            — exponential input gate (scalar per head)
//	f_t = exp(clamp(w_f * x_t + b_f))            — exponential forget gate (scalar per head)
//	C_t = f_t * C_{t-1} + i_t * (v_t * k_t^T)   — matrix cell state (outer product update)
//	n_t = f_t * n_{t-1} + i_t * k_t              — normalizer vector
//	h_t = o_t * (C_t * q_t) / max(|n_t^T * q_t|, 1) — hidden state
//	o_t = σ(w_o * x_t + b_o)                     — output gate (sigmoid)
//
// Gate pre-activations for i and f are clamped to [-maxGatePreAct, maxGatePreAct]
// before applying exp() to prevent overflow.
type MLSTM[T tensor.Float] struct {
	engine compute.Engine[T]

	// Weight matrices: projections [inputDim, hiddenDim]
	Wk *graph.Parameter[T]
	Wv *graph.Parameter[T]
	Wq *graph.Parameter[T]

	// Gate weights: [inputDim] (scalar gate per hidden unit)
	Wi *graph.Parameter[T]
	Wf *graph.Parameter[T]
	Wo *graph.Parameter[T]

	// Gate biases: [hiddenDim]
	Bi *graph.Parameter[T]
	Bf *graph.Parameter[T]
	Bo *graph.Parameter[T]

	inputDim  int
	hiddenDim int
}

// NewMLSTM creates a new mLSTM cell.
//
// Parameters:
//   - engine: the compute engine for tensor operations
//   - inputDim: dimensionality of the input vector at each time step
//   - hiddenDim: dimensionality of the hidden/key/value/query space
func NewMLSTM[T tensor.Float](engine compute.Engine[T], inputDim, hiddenDim int) (*MLSTM[T], error) {
	if inputDim <= 0 {
		return nil, fmt.Errorf("inputDim must be positive, got %d", inputDim)
	}
	if hiddenDim <= 0 {
		return nil, fmt.Errorf("hiddenDim must be positive, got %d", hiddenDim)
	}

	inputScale := 1.0 / math.Sqrt(float64(inputDim))

	makeProjection := func(name string) (*graph.Parameter[T], error) {
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

	makeGateWeight := func(name string) (*graph.Parameter[T], error) {
		data := make([]T, inputDim)
		for i := range data {
			data[i] = T((rand.Float64()*2 - 1) * inputScale)
		}
		t, err := tensor.New[T]([]int{inputDim}, data)
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

	wk, err := makeProjection("mlstm_Wk")
	if err != nil {
		return nil, err
	}
	wv, err := makeProjection("mlstm_Wv")
	if err != nil {
		return nil, err
	}
	wq, err := makeProjection("mlstm_Wq")
	if err != nil {
		return nil, err
	}

	wi, err := makeGateWeight("mlstm_Wi")
	if err != nil {
		return nil, err
	}
	wf, err := makeGateWeight("mlstm_Wf")
	if err != nil {
		return nil, err
	}
	wo, err := makeGateWeight("mlstm_Wo")
	if err != nil {
		return nil, err
	}

	bi, err := makeBias("mlstm_bi")
	if err != nil {
		return nil, err
	}
	bf, err := makeBias("mlstm_bf")
	if err != nil {
		return nil, err
	}
	bo, err := makeBias("mlstm_bo")
	if err != nil {
		return nil, err
	}

	return &MLSTM[T]{
		engine:    engine,
		Wk:        wk,
		Wv:        wv,
		Wq:        wq,
		Wi:        wi,
		Wf:        wf,
		Wo:        wo,
		Bi:        bi,
		Bf:        bf,
		Bo:        bo,
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
	}, nil
}

// OpType returns the operation type of the layer.
func (m *MLSTM[T]) OpType() string {
	return "MLSTM"
}

// Attributes returns the attributes of the layer.
func (m *MLSTM[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_dim":  m.inputDim,
		"hidden_dim": m.hiddenDim,
	}
}

// OutputShape returns the output shape of the layer.
func (m *MLSTM[T]) OutputShape() []int {
	return []int{-1, m.hiddenDim} // [batch, hiddenDim]
}

// Forward performs a single mLSTM time step.
//
// Inputs:
//   - x:     input vector          [batch, inputDim]
//   - hPrev: previous hidden state [batch, hiddenDim]
//   - cPrev: previous cell state   [batch, hiddenDim, hiddenDim] (matrix memory)
//   - nPrev: previous normalizer   [batch, hiddenDim]
//
// Returns (h, C, n) — the new hidden state, matrix cell state, and normalizer.
func (m *MLSTM[T]) Forward(
	ctx context.Context,
	x, hPrev *tensor.TensorNumeric[T],
	cPrev *tensor.TensorNumeric[T],
	nPrev *tensor.TensorNumeric[T],
) (h *tensor.TensorNumeric[T], cOut *tensor.TensorNumeric[T], nOut *tensor.TensorNumeric[T], err error) {
	// Validate input shapes.
	xShape := x.Shape()
	if len(xShape) != 2 || xShape[1] != m.inputDim {
		return nil, nil, nil, fmt.Errorf("MLSTM: x must be [batch, %d], got shape %v", m.inputDim, xShape)
	}
	batch := xShape[0]

	hShape := hPrev.Shape()
	if len(hShape) != 2 || hShape[0] != batch || hShape[1] != m.hiddenDim {
		return nil, nil, nil, fmt.Errorf("MLSTM: hPrev must be [%d, %d], got shape %v",
			batch, m.hiddenDim, hShape)
	}

	cShape := cPrev.Shape()
	if len(cShape) != 3 || cShape[0] != batch || cShape[1] != m.hiddenDim || cShape[2] != m.hiddenDim {
		return nil, nil, nil, fmt.Errorf("MLSTM: cPrev must be [%d, %d, %d], got shape %v",
			batch, m.hiddenDim, m.hiddenDim, cShape)
	}

	nShape := nPrev.Shape()
	if len(nShape) != 2 || nShape[0] != batch || nShape[1] != m.hiddenDim {
		return nil, nil, nil, fmt.Errorf("MLSTM: nPrev must be [%d, %d], got shape %v",
			batch, m.hiddenDim, nShape)
	}

	// Load parameter data.
	wkData := m.Wk.Value.Data()
	wvData := m.Wv.Value.Data()
	wqData := m.Wq.Value.Data()
	wiData := m.Wi.Value.Data()
	wfData := m.Wf.Value.Data()
	woData := m.Wo.Value.Data()
	biData := m.Bi.Value.Data()
	bfData := m.Bf.Value.Data()
	boData := m.Bo.Value.Data()

	xData := x.Data()
	cData := cPrev.Data()
	nData := nPrev.Data()

	d := m.hiddenDim
	hOutData := make([]T, batch*d)
	cOutData := make([]T, batch*d*d)
	nOutData := make([]T, batch*d)

	for b := 0; b < batch; b++ {
		xOff := b * m.inputDim

		// Compute key, value, query projections: W[inputDim, hiddenDim]
		kt := make([]float64, d)
		vt := make([]float64, d)
		qt := make([]float64, d)
		for j := 0; j < d; j++ {
			for k := 0; k < m.inputDim; k++ {
				xk := float64(xData[xOff+k])
				kt[j] += xk * float64(wkData[k*d+j])
				vt[j] += xk * float64(wvData[k*d+j])
				qt[j] += xk * float64(wqData[k*d+j])
			}
		}

		// Compute scalar gate pre-activations: dot(w, x) + b
		// Gates are scalar per batch element (single value controlling the whole update).
		var preI, preF, preO float64
		for k := 0; k < m.inputDim; k++ {
			xk := float64(xData[xOff+k])
			preI += xk * float64(wiData[k])
			preF += xk * float64(wfData[k])
			preO += xk * float64(woData[k])
		}

		// The bias is per hidden dim but for the scalar gate formulation in the
		// paper, we use a single scalar gate. We sum the bias as a scalar offset.
		// However, looking at the paper more carefully, the gates i_t and f_t are
		// scalar per head. We treat the whole cell as one head, so one scalar gate.
		// We use biData[0] as the scalar bias for simplicity.
		preI += float64(biData[0])
		preF += float64(bfData[0])
		preO += float64(boData[0])

		// Clamp and apply gate activations.
		preI = clampFloat(preI, -maxGatePreAct, maxGatePreAct)
		preF = clampFloat(preF, -maxGatePreAct, maxGatePreAct)

		iGate := math.Exp(preI)
		fGate := math.Exp(preF)
		oGate := 1.0 / (1.0 + math.Exp(-preO))

		// Update matrix cell state: C_t = f_t * C_{t-1} + i_t * (v_t * k_t^T)
		cOff := b * d * d
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				prevC := float64(cData[cOff+i*d+j])
				outerProd := vt[i] * kt[j]
				cOutData[cOff+i*d+j] = T(fGate*prevC + iGate*outerProd)
			}
		}

		// Update normalizer: n_t = f_t * n_{t-1} + i_t * k_t
		nOff := b * d
		for j := 0; j < d; j++ {
			prevN := float64(nData[nOff+j])
			nOutData[nOff+j] = T(fGate*prevN + iGate*kt[j])
		}

		// Compute C_t * q_t (matrix-vector product)
		cq := make([]float64, d)
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				cq[i] += float64(cOutData[cOff+i*d+j]) * qt[j]
			}
		}

		// Compute normalizer denominator: max(|n_t^T * q_t|, 1)
		var nq float64
		for j := 0; j < d; j++ {
			nq += float64(nOutData[nOff+j]) * qt[j]
		}
		denom := math.Max(math.Abs(nq), 1.0)

		// Hidden state: h_t = o_t * (C_t * q_t) / denom
		hOff := b * d
		for j := 0; j < d; j++ {
			hOutData[hOff+j] = T(oGate * cq[j] / denom)
		}
	}

	h, err = tensor.New[T]([]int{batch, d}, hOutData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("MLSTM: create h tensor: %w", err)
	}
	cOut, err = tensor.New[T]([]int{batch, d, d}, cOutData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("MLSTM: create C tensor: %w", err)
	}
	nOut, err = tensor.New[T]([]int{batch, d}, nOutData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("MLSTM: create n tensor: %w", err)
	}
	return h, cOut, nOut, nil
}

// Parameters returns all learnable parameters of the mLSTM cell.
func (m *MLSTM[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{
		m.Wk, m.Wv, m.Wq,
		m.Wi, m.Wf, m.Wo,
		m.Bi, m.Bf, m.Bo,
	}
}
