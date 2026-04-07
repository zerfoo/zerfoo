package timeseries

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/layers/functional"
)

// MLSTM implements the mLSTM (Matrix LSTM) cell from the xLSTM paper
// (Beck et al., 2024, arXiv:2405.04517) using the paper's *stabilized*
// exponential-gating formulation to prevent overflow over long sequences.
//
// The mLSTM replaces the scalar cell state of a classical LSTM with a matrix
// cell state (covariance memory) updated via outer products of key and value
// projections. With the stabilizer m_t both gates are bounded to (0, 1]:
//
//	k_t  = W_k * x_t                              — key projection
//	v_t  = W_v * x_t                              — value projection
//	q_t  = W_q * x_t                              — query projection
//	preI = w_i * x_t + b_i
//	preF = w_f * x_t + b_f
//	m_t  = max(preF + m_{t-1}, preI)              — running-max stabilizer
//	i_t  = exp(preI - m_t)                        ∈ (0, 1]
//	f_t  = exp(preF + m_{t-1} - m_t)              ∈ (0, 1]
//	C_t  = f_t * C_{t-1} + i_t * (v_t * k_t^T)   — matrix cell state
//	n_t  = f_t * n_{t-1} + i_t * k_t              — normalizer vector
//	o_t  = σ(w_o * x_t + b_o)                     — output gate (sigmoid)
//	h_t  = o_t * (C_t * q_t) / max(|n_t^T * q_t|, 1)
//
// Gate pre-activations are still clamped to [-maxGatePreAct, maxGatePreAct]
// as a safety net for pathological weight inits.
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

// Forward performs a single mLSTM time step using the paper's stabilized form.
//
// Inputs:
//   - x:     input vector          [batch, inputDim]
//   - hPrev: previous hidden state [batch, hiddenDim]
//   - cPrev: previous cell state   [batch, hiddenDim, hiddenDim] (matrix memory)
//   - nPrev: previous normalizer   [batch, hiddenDim]
//   - mPrev: previous stabilizer   [batch, 1]; nil is treated as zero (use nil
//     for the first timestep of a sequence)
//
// Returns (h, C, n, m) — the new hidden, matrix cell, normalizer, and
// stabilizer states. Callers must thread m through the sequence alongside C
// and n.
func (m *MLSTM[T]) Forward(
	ctx context.Context,
	x, hPrev *tensor.TensorNumeric[T],
	cPrev *tensor.TensorNumeric[T],
	nPrev *tensor.TensorNumeric[T],
	mPrev *tensor.TensorNumeric[T],
) (h *tensor.TensorNumeric[T], cOut *tensor.TensorNumeric[T], nOut *tensor.TensorNumeric[T], mOut *tensor.TensorNumeric[T], err error) {
	// Validate input shapes.
	xShape := x.Shape()
	if len(xShape) != 2 || xShape[1] != m.inputDim {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: x must be [batch, %d], got shape %v", m.inputDim, xShape)
	}
	batch := xShape[0]

	hShape := hPrev.Shape()
	if len(hShape) != 2 || hShape[0] != batch || hShape[1] != m.hiddenDim {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: hPrev must be [%d, %d], got shape %v",
			batch, m.hiddenDim, hShape)
	}

	cShape := cPrev.Shape()
	if len(cShape) != 3 || cShape[0] != batch || cShape[1] != m.hiddenDim || cShape[2] != m.hiddenDim {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: cPrev must be [%d, %d, %d], got shape %v",
			batch, m.hiddenDim, m.hiddenDim, cShape)
	}

	nShape := nPrev.Shape()
	if len(nShape) != 2 || nShape[0] != batch || nShape[1] != m.hiddenDim {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: nPrev must be [%d, %d], got shape %v",
			batch, m.hiddenDim, nShape)
	}

	eng := m.engine
	ops := eng.Ops()
	d := m.hiddenDim

	// Key, value, query projections via engine MatMul.
	// W is [inputDim, hiddenDim], x is [batch, inputDim] → [batch, hiddenDim]
	kt, err := eng.MatMul(ctx, x, m.Wk.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(x, Wk): %w", err)
	}
	vt, err := eng.MatMul(ctx, x, m.Wv.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(x, Wv): %w", err)
	}
	qt, err := eng.MatMul(ctx, x, m.Wq.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(x, Wq): %w", err)
	}

	// Scalar gate pre-activations: dot(x, w) + b[0] per batch element.
	// Wi is [inputDim], reshape to [inputDim, 1] for MatMul → [batch, 1]
	wiCol, err := eng.Reshape(ctx, m.Wi.Value, []int{m.inputDim, 1})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Reshape(Wi): %w", err)
	}
	preI, err := eng.MatMul(ctx, x, wiCol)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(x, Wi): %w", err)
	}
	preI, err = eng.AddScalar(ctx, preI, m.Bi.Value.Data()[0])
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: AddScalar(preI, bi): %w", err)
	}

	wfCol, err := eng.Reshape(ctx, m.Wf.Value, []int{m.inputDim, 1})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Reshape(Wf): %w", err)
	}
	preF, err := eng.MatMul(ctx, x, wfCol)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(x, Wf): %w", err)
	}
	preF, err = eng.AddScalar(ctx, preF, m.Bf.Value.Data()[0])
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: AddScalar(preF, bf): %w", err)
	}

	woCol, err := eng.Reshape(ctx, m.Wo.Value, []int{m.inputDim, 1})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Reshape(Wo): %w", err)
	}
	preO, err := eng.MatMul(ctx, x, woCol)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(x, Wo): %w", err)
	}
	preO, err = eng.AddScalar(ctx, preO, m.Bo.Value.Data()[0])
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: AddScalar(preO, bo): %w", err)
	}

	// Stabilized exponential gating (xLSTM paper). The running-max state
	// m_t = max(preF + m_{t-1}, preI) keeps both gates bounded to (0, 1].
	// preI, preF, mPrev all have shape [batch, 1].
	clampOp := func(v T) T {
		return T(clampFloat(float64(v), -maxGatePreAct, maxGatePreAct))
	}
	preIClamped, err := eng.UnaryOp(ctx, preI, clampOp)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: clamp(preI): %w", err)
	}
	preFClamped, err := eng.UnaryOp(ctx, preF, clampOp)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: clamp(preF): %w", err)
	}

	// preFPlusMPrev = preFClamped + mPrev (mPrev=nil → zero).
	preFPlusMPrev := preFClamped
	if mPrev != nil {
		mShape := mPrev.Shape()
		if len(mShape) != 2 || mShape[0] != batch || mShape[1] != 1 {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: mPrev must be [%d, 1], got shape %v", batch, mShape)
		}
		preFPlusMPrev, err = eng.Add(ctx, preFClamped, mPrev)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: Add(preFClamped, mPrev): %w", err)
		}
	}

	// m_t = elementwise max(preFPlusMPrev, preIClamped).
	// Implemented as a + relu(b - a) since the engine has no BinaryMax.
	bMinusA, err := eng.Sub(ctx, preFPlusMPrev, preIClamped)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Sub for max: %w", err)
	}
	reluBMinusA, err := eng.UnaryOp(ctx, bMinusA, func(v T) T {
		if float64(v) > 0 {
			return v
		}
		return 0
	})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: relu for max: %w", err)
	}
	mOut, err = eng.Add(ctx, preIClamped, reluBMinusA)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Add for max: %w", err)
	}

	// i_t = exp(preI - m_t)  ∈ (0, 1]
	preIMinusM, err := eng.Sub(ctx, preIClamped, mOut)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Sub(preI, m): %w", err)
	}
	iGate, err := eng.UnaryOp(ctx, preIMinusM, func(v T) T { return T(math.Exp(float64(v))) })
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: exp(preI-m): %w", err)
	}

	// f_t = exp(preF + mPrev - m_t)  ∈ (0, 1]
	preFShift, err := eng.Sub(ctx, preFPlusMPrev, mOut)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Sub(preF+mPrev, m): %w", err)
	}
	fGate, err := eng.UnaryOp(ctx, preFShift, func(v T) T { return T(math.Exp(float64(v))) })
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: exp(preF+mPrev-m): %w", err)
	}

	// o = sigmoid(preO) — [batch, 1]
	oGate, err := functional.Sigmoid(ctx, eng, ops, preO)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Sigmoid(preO): %w", err)
	}

	// Update normalizer: n_t = f_t * n_{t-1} + i_t * k_t
	// fGate and iGate are [batch, 1], nPrev and kt are [batch, d].
	// Broadcasting: [batch, 1] * [batch, d] → [batch, d]
	fN, err := eng.Mul(ctx, fGate, nPrev)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Mul(f, nPrev): %w", err)
	}
	iK, err := eng.Mul(ctx, iGate, kt)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Mul(i, k): %w", err)
	}
	nOut, err = eng.Add(ctx, fN, iK)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: Add(fN, iK): %w", err)
	}

	// Update matrix cell state: C_t = f_t * C_{t-1} + i_t * (v_t * k_t^T)
	// The engine's MatMul only handles 2D, so we split along batch axis
	// and process each batch element separately.
	vtSlices, err := eng.Split(ctx, vt, batch, 0) // batch * [1, d]
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split vt: %w", err)
	}
	ktSlices, err := eng.Split(ctx, kt, batch, 0)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split kt: %w", err)
	}
	iSlices, err := eng.Split(ctx, iGate, batch, 0) // batch * [1, 1]
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split iGate: %w", err)
	}
	fSlices, err := eng.Split(ctx, fGate, batch, 0)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split fGate: %w", err)
	}
	cPrevSlices, err := eng.Split(ctx, cPrev, batch, 0) // batch * [1, d, d]
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split cPrev: %w", err)
	}

	cOutSlices := make([]*tensor.TensorNumeric[T], batch)
	for b := 0; b < batch; b++ {
		// Reshape v[b] to [d, 1] and k[b] to [1, d] for outer product.
		vb, err := eng.Reshape(ctx, vtSlices[b], []int{d, 1})
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: reshape vb: %w", err)
		}
		kb, err := eng.Reshape(ctx, ktSlices[b], []int{1, d})
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: reshape kb: %w", err)
		}

		outerProd, err := eng.MatMul(ctx, vb, kb) // [d, d]
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(v, k^T): %w", err)
		}

		// Extract scalar gate values (1-element tensors).
		ib := iSlices[b].Data()[0]
		fb := fSlices[b].Data()[0]

		cPrevB, err := eng.Reshape(ctx, cPrevSlices[b], []int{d, d})
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: reshape cPrevB: %w", err)
		}
		fC, err := eng.MulScalar(ctx, cPrevB, fb)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: MulScalar(cPrev, f): %w", err)
		}
		iVK, err := eng.MulScalar(ctx, outerProd, ib)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: MulScalar(outer, i): %w", err)
		}
		cNewB, err := eng.Add(ctx, fC, iVK) // [d, d]
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: Add(fC, iVK): %w", err)
		}
		// Reshape to [1, d, d] for concatenation
		cNewB3D, err := eng.Reshape(ctx, cNewB, []int{1, d, d})
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: reshape cNewB: %w", err)
		}
		cOutSlices[b] = cNewB3D
	}

	cOut, err = eng.Concat(ctx, cOutSlices, 0) // [batch, d, d]
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: concat C: %w", err)
	}

	// Compute h_t = o_t * (C_t * q_t) / max(|n_t^T * q_t|, 1)
	// Process per batch since C is 3D and q is 2D.
	qtSlices, err := eng.Split(ctx, qt, batch, 0) // batch * [1, d]
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split qt: %w", err)
	}
	nOutSlices, err := eng.Split(ctx, nOut, batch, 0)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split nOut: %w", err)
	}
	oSlices, err := eng.Split(ctx, oGate, batch, 0) // batch * [1, 1]
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split oGate: %w", err)
	}
	cOutSplits, err := eng.Split(ctx, cOut, batch, 0) // batch * [1, d, d]
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: split cOut: %w", err)
	}

	hSlices := make([]*tensor.TensorNumeric[T], batch)
	for b := 0; b < batch; b++ {
		cB, err := eng.Reshape(ctx, cOutSplits[b], []int{d, d})
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: reshape cB: %w", err)
		}
		qb, err := eng.Reshape(ctx, qtSlices[b], []int{d, 1})
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: reshape qb: %w", err)
		}

		cq, err := eng.MatMul(ctx, cB, qb) // [d, 1]
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(C, q): %w", err)
		}

		// n_t^T * q_t: dot product
		nb, err := eng.Reshape(ctx, nOutSlices[b], []int{1, d})
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: reshape nb: %w", err)
		}
		nqTensor, err := eng.MatMul(ctx, nb, qb) // [1, 1]
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: MatMul(n, q): %w", err)
		}
		nqVal := float64(nqTensor.Data()[0]) // scalar extraction
		denom := math.Max(math.Abs(nqVal), 1.0)

		ob := float64(oSlices[b].Data()[0]) // scalar extraction

		// h[b] = o * (C*q) / denom
		cqFlat, err := eng.Reshape(ctx, cq, []int{1, d}) // [1, d]
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: reshape cq: %w", err)
		}
		scaledCQ, err := eng.MulScalar(ctx, cqFlat, ops.FromFloat64(ob/denom))
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("MLSTM: scale h: %w", err)
		}
		hSlices[b] = scaledCQ
	}

	h, err = eng.Concat(ctx, hSlices, 0) // [batch, d]
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MLSTM: concat h: %w", err)
	}

	return h, cOut, nOut, mOut, nil
}

// Parameters returns all learnable parameters of the mLSTM cell.
func (m *MLSTM[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{
		m.Wk, m.Wv, m.Wq,
		m.Wi, m.Wf, m.Wo,
		m.Bi, m.Bf, m.Bo,
	}
}
