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

// maxGatePreAct is the clamp threshold for gate pre-activations before exp()
// to prevent floating-point overflow. exp(20) ≈ 4.85e8 which is safe for
// float32; exp(89) would overflow.
const maxGatePreAct = 20.0

// SLSTM implements the sLSTM cell from the xLSTM paper (Beck et al., 2024)
// using the paper's *stabilized* formulation to prevent overflow of the
// exponential gates over long sequences.
//
// Naive formulation (overflows in float32 for all but tiny sequences):
//
//	i_t = exp(Wi*x_t + Ri*h_{t-1} + bi)
//	f_t = exp(Wf*x_t + Rf*h_{t-1} + bf)
//
// Stabilized formulation (used here): a running max m_t is tracked so that
// both i_t and f_t are bounded to (0, 1]:
//
//	preI = Wi*x_t + Ri*h_{t-1} + bi
//	preF = Wf*x_t + Rf*h_{t-1} + bf
//	m_t  = max(preF + m_{t-1}, preI)          — new stabilizer
//	i_t  = exp(preI - m_t)                    — bounded in (0, 1]
//	f_t  = exp(preF + m_{t-1} - m_t)          — bounded in (0, 1]
//	z_t  = tanh(Wz*x_t + Rz*h_{t-1} + bz)
//	o_t  = σ(Wo*x_t + Ro*h_{t-1} + bo)
//	n_t  = f_t * n_{t-1} + i_t
//	c_t  = f_t * c_{t-1} + i_t * z_t
//	h_t  = o_t * (c_t / max(n_t, ε))
//
// The stabilizer m is threaded through time as part of the cell state.
// maxGatePreAct is kept as a large safety clamp on preI/preF only — it is
// not the primary overflow guard.
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

// Forward performs a single sLSTM time step using the paper's stabilized
// formulation.
//
// Inputs:
//   - x:     input vector         [batch, inputDim]
//   - hPrev: previous hidden state [batch, hiddenDim]
//   - cPrev: previous cell state   [batch, hiddenDim]
//   - nPrev: previous normalizer   [batch, hiddenDim]
//   - mPrev: previous stabilizer   [batch, hiddenDim]; nil is treated as zero
//     (use nil for the first timestep of a sequence)
//
// Returns (h, c, n, m) — the new hidden, cell, normalizer, and stabilizer
// states. Callers must thread m through the sequence alongside c and n.
func (s *SLSTM[T]) Forward(
	ctx context.Context,
	x, hPrev, cPrev, nPrev, mPrev *tensor.TensorNumeric[T],
) (h, c, n, m *tensor.TensorNumeric[T], err error) {
	// Validate input shapes.
	xShape := x.Shape()
	if len(xShape) != 2 || xShape[1] != s.inputDim {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: x must be [batch, %d], got shape %v", s.inputDim, xShape)
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
			return nil, nil, nil, nil, fmt.Errorf("SLSTM: %s must be [%d, %d], got shape %v",
				pair.name, batch, s.hiddenDim, shape)
		}
	}

	eng := s.engine
	ops := eng.Ops()

	// Compute pre-activations: W*x + R*h + b for each gate.
	// W is [inputDim, hiddenDim], x is [batch, inputDim] → MatMul(x, W) = [batch, hiddenDim]
	// R is [hiddenDim, hiddenDim], hPrev is [batch, hiddenDim] → MatMul(hPrev, R) = [batch, hiddenDim]
	// Bias is [hiddenDim], broadcast-added to [batch, hiddenDim].

	preI, err := eng.MatMul(ctx, x, s.Wi.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: MatMul(x, Wi): %w", err)
	}
	riH, err := eng.MatMul(ctx, hPrev, s.Ri.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: MatMul(h, Ri): %w", err)
	}
	preI, err = eng.Add(ctx, preI, riH)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preI, riH): %w", err)
	}
	preI, err = eng.Add(ctx, preI, s.Bi.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preI, Bi): %w", err)
	}

	preF, err := eng.MatMul(ctx, x, s.Wf.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: MatMul(x, Wf): %w", err)
	}
	rfH, err := eng.MatMul(ctx, hPrev, s.Rf.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: MatMul(h, Rf): %w", err)
	}
	preF, err = eng.Add(ctx, preF, rfH)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preF, rfH): %w", err)
	}
	preF, err = eng.Add(ctx, preF, s.Bf.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preF, Bf): %w", err)
	}

	preZ, err := eng.MatMul(ctx, x, s.Wz.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: MatMul(x, Wz): %w", err)
	}
	rzH, err := eng.MatMul(ctx, hPrev, s.Rz.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: MatMul(h, Rz): %w", err)
	}
	preZ, err = eng.Add(ctx, preZ, rzH)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preZ, rzH): %w", err)
	}
	preZ, err = eng.Add(ctx, preZ, s.Bz.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preZ, Bz): %w", err)
	}

	preO, err := eng.MatMul(ctx, x, s.Wo.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: MatMul(x, Wo): %w", err)
	}
	roH, err := eng.MatMul(ctx, hPrev, s.Ro.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: MatMul(h, Ro): %w", err)
	}
	preO, err = eng.Add(ctx, preO, roH)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preO, roH): %w", err)
	}
	preO, err = eng.Add(ctx, preO, s.Bo.Value)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preO, Bo): %w", err)
	}

	// Stabilized exponential gating (xLSTM paper). The running-max state
	// m_t = max(preF + m_{t-1}, preI) keeps both gates bounded to (0, 1]:
	//   i_t = exp(preI - m_t)
	//   f_t = exp(preF + m_{t-1} - m_t)
	//
	// mPrev=nil is treated as zero (first timestep). A large safety clamp is
	// still applied to preI/preF in case an unstable weight init produces
	// extreme pre-activations.
	clampOp := func(v T) T {
		return T(clampFloat(float64(v), -maxGatePreAct, maxGatePreAct))
	}
	preIClamped, err := eng.UnaryOp(ctx, preI, clampOp)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: clamp(preI): %w", err)
	}
	preFClamped, err := eng.UnaryOp(ctx, preF, clampOp)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: clamp(preF): %w", err)
	}

	// preFPlusMPrev = preFClamped + mPrev (mPrev=nil is treated as zero).
	preFPlusMPrev := preFClamped
	if mPrev != nil {
		mShape := mPrev.Shape()
		if len(mShape) != 2 || mShape[0] != batch || mShape[1] != s.hiddenDim {
			return nil, nil, nil, nil, fmt.Errorf("SLSTM: mPrev must be [%d, %d], got shape %v",
				batch, s.hiddenDim, mShape)
		}
		preFPlusMPrev, err = eng.Add(ctx, preFClamped, mPrev)
		if err != nil {
			return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(preFClamped, mPrev): %w", err)
		}
	}

	// m_t = elementwise max(preF + mPrev, preI).
	// Implemented as a = preI, b = preFPlusMPrev: max(a, b) = a + relu(b - a).
	bMinusA, err := eng.Sub(ctx, preFPlusMPrev, preIClamped)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Sub for max: %w", err)
	}
	reluBMinusA, err := eng.UnaryOp(ctx, bMinusA, func(v T) T {
		if float64(v) > 0 {
			return v
		}
		return 0
	})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: relu for max: %w", err)
	}
	m, err = eng.Add(ctx, preIClamped, reluBMinusA)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add for max: %w", err)
	}

	// i_t = exp(preI - m_t)  ∈ (0, 1]
	preIMinusM, err := eng.Sub(ctx, preIClamped, m)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Sub(preI, m): %w", err)
	}
	iGate, err := eng.UnaryOp(ctx, preIMinusM, func(v T) T { return T(math.Exp(float64(v))) })
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: exp(preI-m): %w", err)
	}

	// f_t = exp(preF + mPrev - m_t)  ∈ (0, 1]
	preFShift, err := eng.Sub(ctx, preFPlusMPrev, m)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Sub(preF+mPrev, m): %w", err)
	}
	fGate, err := eng.UnaryOp(ctx, preFShift, func(v T) T { return T(math.Exp(float64(v))) })
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: exp(preF+mPrev-m): %w", err)
	}

	// Cell input: z = tanh(preZ)
	zVal, err := eng.Tanh(ctx, preZ)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Tanh(preZ): %w", err)
	}

	// Output gate: o = sigmoid(preO)
	oGate, err := functional.Sigmoid(ctx, eng, ops, preO)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Sigmoid(preO): %w", err)
	}

	// n = f * nPrev + i
	fN, err := eng.Mul(ctx, fGate, nPrev)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Mul(f, nPrev): %w", err)
	}
	n, err = eng.Add(ctx, fN, iGate)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(fN, i): %w", err)
	}

	// c = f * cPrev + i * z
	fC, err := eng.Mul(ctx, fGate, cPrev)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Mul(f, cPrev): %w", err)
	}
	iZ, err := eng.Mul(ctx, iGate, zVal)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Mul(i, z): %w", err)
	}
	c, err = eng.Add(ctx, fC, iZ)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Add(fC, iZ): %w", err)
	}

	// h = o * (c / n), guarding against division by zero.
	// Replace any n < 1e-12 with 1e-12 to avoid division by zero.
	safeN, err := eng.UnaryOp(ctx, n, func(v T) T {
		if float64(v) < 1e-12 {
			return T(1e-12)
		}
		return v
	})
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: safeN: %w", err)
	}
	cDivN, err := eng.Div(ctx, c, safeN)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Div(c, n): %w", err)
	}
	h, err = eng.Mul(ctx, oGate, cDivN)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("SLSTM: Mul(o, c/n): %w", err)
	}

	return h, c, n, m, nil
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
