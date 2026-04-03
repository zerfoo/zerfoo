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

	eng := s.engine
	ops := eng.Ops()

	// Compute pre-activations: W*x + R*h + b for each gate.
	// W is [inputDim, hiddenDim], x is [batch, inputDim] → MatMul(x, W) = [batch, hiddenDim]
	// R is [hiddenDim, hiddenDim], hPrev is [batch, hiddenDim] → MatMul(hPrev, R) = [batch, hiddenDim]
	// Bias is [hiddenDim], broadcast-added to [batch, hiddenDim].

	preI, err := eng.MatMul(ctx, x, s.Wi.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: MatMul(x, Wi): %w", err)
	}
	riH, err := eng.MatMul(ctx, hPrev, s.Ri.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: MatMul(h, Ri): %w", err)
	}
	preI, err = eng.Add(ctx, preI, riH)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(preI, riH): %w", err)
	}
	preI, err = eng.Add(ctx, preI, s.Bi.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(preI, Bi): %w", err)
	}

	preF, err := eng.MatMul(ctx, x, s.Wf.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: MatMul(x, Wf): %w", err)
	}
	rfH, err := eng.MatMul(ctx, hPrev, s.Rf.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: MatMul(h, Rf): %w", err)
	}
	preF, err = eng.Add(ctx, preF, rfH)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(preF, rfH): %w", err)
	}
	preF, err = eng.Add(ctx, preF, s.Bf.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(preF, Bf): %w", err)
	}

	preZ, err := eng.MatMul(ctx, x, s.Wz.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: MatMul(x, Wz): %w", err)
	}
	rzH, err := eng.MatMul(ctx, hPrev, s.Rz.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: MatMul(h, Rz): %w", err)
	}
	preZ, err = eng.Add(ctx, preZ, rzH)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(preZ, rzH): %w", err)
	}
	preZ, err = eng.Add(ctx, preZ, s.Bz.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(preZ, Bz): %w", err)
	}

	preO, err := eng.MatMul(ctx, x, s.Wo.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: MatMul(x, Wo): %w", err)
	}
	roH, err := eng.MatMul(ctx, hPrev, s.Ro.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: MatMul(h, Ro): %w", err)
	}
	preO, err = eng.Add(ctx, preO, roH)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(preO, roH): %w", err)
	}
	preO, err = eng.Add(ctx, preO, s.Bo.Value)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(preO, Bo): %w", err)
	}

	// Clamp i and f pre-activations before exp() to prevent overflow.
	clampExpOp := func(v T) T {
		f := clampFloat(float64(v), -maxGatePreAct, maxGatePreAct)
		return T(math.Exp(f))
	}
	iGate, err := eng.UnaryOp(ctx, preI, clampExpOp)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: clampExp(preI): %w", err)
	}
	fGate, err := eng.UnaryOp(ctx, preF, clampExpOp)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: clampExp(preF): %w", err)
	}

	// Cell input: z = tanh(preZ)
	zVal, err := eng.Tanh(ctx, preZ)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Tanh(preZ): %w", err)
	}

	// Output gate: o = sigmoid(preO)
	oGate, err := functional.Sigmoid(ctx, eng, ops, preO)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Sigmoid(preO): %w", err)
	}

	// n = f * nPrev + i
	fN, err := eng.Mul(ctx, fGate, nPrev)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Mul(f, nPrev): %w", err)
	}
	n, err = eng.Add(ctx, fN, iGate)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(fN, i): %w", err)
	}

	// c = f * cPrev + i * z
	fC, err := eng.Mul(ctx, fGate, cPrev)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Mul(f, cPrev): %w", err)
	}
	iZ, err := eng.Mul(ctx, iGate, zVal)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Mul(i, z): %w", err)
	}
	c, err = eng.Add(ctx, fC, iZ)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Add(fC, iZ): %w", err)
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
		return nil, nil, nil, fmt.Errorf("SLSTM: safeN: %w", err)
	}
	cDivN, err := eng.Div(ctx, c, safeN)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Div(c, n): %w", err)
	}
	h, err = eng.Mul(ctx, oGate, cDivN)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("SLSTM: Mul(o, c/n): %w", err)
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
