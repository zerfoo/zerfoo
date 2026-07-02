// Package ssm implements state space model layers.
package ssm

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"

	"github.com/zerfoo/zerfoo/layers/core"
)

// DiscretizationMode controls how the continuous SSM (A, B) is discretized.
type DiscretizationMode int

const (
	// ZOH uses zero-order hold discretization (Mamba 1/2 default):
	//   Ā = exp(Δ * A)
	//   B̄ = Δ * B
	ZOH DiscretizationMode = iota

	// ExpTrap uses exponential-trapezoidal discretization (Mamba 3):
	//   Ā = exp(Δ * A)
	//   B̄ = Δ * (I + exp(Δ * A)) / 2 * B
	//
	// This gives richer system dynamics by taking a trapezoidal average of the
	// continuous-time B at both endpoints of the discretization interval.
	ExpTrap
)

func randomData[T tensor.Numeric](size int) []T {
	data := make([]T, size)
	for i := range data {
		data[i] = T(rand.Float32())
	}
	return data
}

// scaleLinearWeights re-initializes the weights of a Linear layer with
// centered Xavier-style initialization: N(0, scale^2). core.NewLinear
// uses rand.Float32() in [0,1) which has mean 0.5 — the positive bias
// causes activation explosion in deep networks. This function centers
// the distribution and applies the given scale.
func scaleLinearWeights[T tensor.Numeric](l *core.Linear[T], scale float64) {
	for _, p := range l.Parameters() {
		data := p.Value.Data()
		for i := range data {
			// Center: map [0,1) -> [-0.5, 0.5), then scale.
			data[i] = T((float64(data[i]) - 0.5) * scale * 2)
		}
	}
}

// MambaBlock implements the Mamba selective state space model block.
//
// Architecture (Mamba-1 style):
//   - Input projection: d_model -> 2*d_inner (split into x and z branches)
//   - Depthwise causal Conv1D on x branch (kernel_size=4, groups=d_inner)
//   - SiLU activation on x
//   - SSM parameter projection: x -> (dt, B, C)
//   - Selective scan: discretize A,B with softplus(dt), run parallel scan
//   - Gate: y * SiLU(z)
//   - Output projection: d_inner -> d_model
//
// Input shape:  [batch, seq_len, d_model]
// Output shape: [batch, seq_len, d_model]
type MambaBlock[T tensor.Numeric] struct {
	name   string
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Dimensions
	dModel  int
	dInner  int
	dState  int
	dtRank  int
	convKer int // conv1d kernel size

	// Discretization mode (default: ZOH)
	discMode DiscretizationMode

	// Projections
	inProj  *core.Linear[T] // d_model -> 2*d_inner
	xProj   *core.Linear[T] // d_inner -> dt_rank + 2*d_state
	dtProj  *core.Linear[T] // dt_rank -> d_inner
	outProj *core.Linear[T] // d_inner -> d_model

	// Conv1D weight: [d_inner, 1, conv_ker] (depthwise)
	convWeight *graph.Parameter[T]
	convBias   *graph.Parameter[T] // [d_inner] — conv1d bias

	// SSM parameters
	A *graph.Parameter[T] // [d_inner, d_state] — log-space initialization
	D *graph.Parameter[T] // [d_inner] — skip connection

	// Cached intermediates for backward
	lastInput    *tensor.TensorNumeric[T]
	cachedX      *tensor.TensorNumeric[T] // after conv + silu
	cachedZ      *tensor.TensorNumeric[T] // gate branch
	cachedSiluZ  *tensor.TensorNumeric[T] // silu(z)
	cachedDt     *tensor.TensorNumeric[T] // softplus(dt_proj output)
	cachedB      *tensor.TensorNumeric[T] // [batch, seq, d_state]
	cachedC      *tensor.TensorNumeric[T] // [batch, seq, d_state]
	cachedY      *tensor.TensorNumeric[T] // scan output before gating
	cachedStates *tensor.TensorNumeric[T] // [batch, seq, d_inner, d_state] all hidden states
	cachedXConv  *tensor.TensorNumeric[T] // x before silu (after conv)
	cachedXBCDt  *tensor.TensorNumeric[T] // x_proj output (before split)
	cachedDtRaw  *tensor.TensorNumeric[T] // dt before softplus
	cachedXPreConv *tensor.TensorNumeric[T] // x branch before conv1d
}

// MambaBlockOption is a functional option for configuring a MambaBlock.
type MambaBlockOption[T tensor.Numeric] func(*MambaBlock[T])

// WithDiscretizationMode sets the SSM discretization mode.
// Defaults to ZOH for backward compatibility.
func WithDiscretizationMode[T tensor.Numeric](mode DiscretizationMode) MambaBlockOption[T] {
	return func(m *MambaBlock[T]) {
		m.discMode = mode
	}
}

// NewMambaBlock creates a new MambaBlock.
//
// Parameters:
//   - dModel: input/output dimension
//   - dInner: inner SSM dimension (typically 2*dModel)
//   - dState: SSM state dimension (e.g. 16)
//   - dtRank: rank of dt projection (typically dModel/16 or ceil(dModel/16))
//   - convKer: depthwise conv1d kernel size (typically 4)
//   - opts: optional functional options (e.g. WithDiscretizationMode)
func NewMambaBlock[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	dModel, dInner, dState, dtRank, convKer int,
	opts ...MambaBlockOption[T],
) (*MambaBlock[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if dModel <= 0 || dInner <= 0 || dState <= 0 || dtRank <= 0 || convKer <= 0 {
		return nil, fmt.Errorf("all dimensions must be positive")
	}

	inProj, err := core.NewLinear[T](name+"_in_proj", engine, ops, dModel, 2*dInner)
	if err != nil {
		return nil, fmt.Errorf("creating in_proj: %w", err)
	}

	xProj, err := core.NewLinear[T](name+"_x_proj", engine, ops, dInner, dtRank+2*dState)
	if err != nil {
		return nil, fmt.Errorf("creating x_proj: %w", err)
	}

	dtProj, err := core.NewLinear[T](name+"_dt_proj", engine, ops, dtRank, dInner)
	if err != nil {
		return nil, fmt.Errorf("creating dt_proj: %w", err)
	}

	outProj, err := core.NewLinear[T](name+"_out_proj", engine, ops, dInner, dModel)
	if err != nil {
		return nil, fmt.Errorf("creating out_proj: %w", err)
	}

	// Conv1D weight: depthwise [d_inner, 1, conv_ker]
	convData := randomData[T](dInner * convKer)
	convTensor, err := tensor.New[T]([]int{dInner, 1, convKer}, convData)
	if err != nil {
		return nil, err
	}
	convWeight, err := graph.NewParameter[T](name+"_conv_weight", convTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	// Conv1D bias: [d_inner], initialized to zeros
	convBiasData := make([]T, dInner)
	convBiasTensor, err := tensor.New[T]([]int{dInner}, convBiasData)
	if err != nil {
		return nil, err
	}
	convBias, err := graph.NewParameter[T](name+"_conv_bias", convBiasTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	// A: initialized as -exp(linspace(log(1), log(d_state), d_inner*d_state))
	// Simplified: initialize A in log-space as negative values
	aData := make([]T, dInner*dState)
	for i := 0; i < dInner; i++ {
		for j := 0; j < dState; j++ {
			// log(j+1) gives values from log(1)=0 to log(d_state)
			aData[i*dState+j] = T(math.Log(float64(j + 1)))
		}
	}
	aTensor, err := tensor.New[T]([]int{dInner, dState}, aData)
	if err != nil {
		return nil, err
	}
	aParam, err := graph.NewParameter[T](name+"_A", aTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	// D: skip connection, initialized to ones
	dData := make([]T, dInner)
	for i := range dData {
		dData[i] = T(1.0)
	}
	dTensor, err := tensor.New[T]([]int{dInner}, dData)
	if err != nil {
		return nil, err
	}
	dParam, err := graph.NewParameter[T](name+"_D", dTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	// Apply Xavier-style initialization: scale weights by 1/sqrt(fan_in).
	// core.NewLinear initializes with rand.Float32() in [0,1), which causes
	// activation explosion for larger models (e.g. dModel=64, NLayers=2).
	scaleLinearWeights(inProj, 1.0/math.Sqrt(float64(dModel)))
	scaleLinearWeights(xProj, 1.0/math.Sqrt(float64(dInner)))
	scaleLinearWeights(outProj, 1.0/math.Sqrt(float64(dInner)))

	// dt_proj: initialize with very small weights so that softplus(output) ≈ 0.7.
	// This follows the Mamba paper's dt_init which targets dt ∈ [0.001, 0.1].
	// Without this, large dt values cause the SSM recurrence to explode through
	// the multiplicative dt*B*x interaction. See issue #158.
	scaleLinearWeights(dtProj, 0.01/math.Sqrt(float64(dtRank)))

	// Scale conv weights: depthwise conv has effective fan_in = conv_ker.
	convScale := T(1.0 / math.Sqrt(float64(convKer)))
	for i := range convData {
		convData[i] *= convScale
	}

	block := &MambaBlock[T]{
		name:       name,
		engine:     engine,
		ops:        ops,
		dModel:     dModel,
		dInner:     dInner,
		dState:     dState,
		dtRank:     dtRank,
		convKer:    convKer,
		inProj:     inProj,
		xProj:      xProj,
		dtProj:     dtProj,
		outProj:    outProj,
		convWeight: convWeight,
		convBias:   convBias,
		A:          aParam,
		D:          dParam,
		discMode:   ZOH, // default
	}
	for _, opt := range opts {
		opt(block)
	}
	return block, nil
}

// ScaleOutputProj scales the output projection weights by the given factor.
// Used by multi-layer models to apply residual scaling (1/sqrt(NLayers)).
func (m *MambaBlock[T]) ScaleOutputProj(scale float64) {
	for _, p := range m.outProj.Parameters() {
		data := p.Value.Data()
		for i := range data {
			data[i] = T(float64(data[i]) * scale)
		}
	}
}

func (m *MambaBlock[T]) OpType() string { return "MambaBlock" }

func (m *MambaBlock[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"d_model":     m.dModel,
		"d_inner":     m.dInner,
		"d_state":     m.dState,
		"dt_rank":     m.dtRank,
		"kernel_size": m.convKer,
	}
}

func (m *MambaBlock[T]) OutputShape() []int {
	return []int{-1, -1, m.dModel}
}

func (m *MambaBlock[T]) Name() string    { return m.name }
func (m *MambaBlock[T]) SetName(n string) { m.name = n }

// Forward computes the Mamba block forward pass.
// Input: [batch, seq_len, d_model]
// Output: [batch, seq_len, d_model]
func (m *MambaBlock[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MambaBlock requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("MambaBlock input must be 3D [batch, seq_len, d_model], got %v", shape)
	}

	batch := shape[0]
	seqLen := shape[1]
	m.lastInput = input

	// 1. Input projection: [batch, seq_len, d_model] -> [batch, seq_len, 2*d_inner]
	projected, err := m.inProj.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("in_proj forward: %w", err)
	}

	// 2. Split into x and z branches along last dim
	projData := projected.Data()
	xData := make([]T, batch*seqLen*m.dInner)
	zData := make([]T, batch*seqLen*m.dInner)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * (2 * m.dInner)
			copy(xData[(b*seqLen+s)*m.dInner:], projData[off:off+m.dInner])
			copy(zData[(b*seqLen+s)*m.dInner:], projData[off+m.dInner:off+2*m.dInner])
		}
	}
	xTensor, err := tensor.New[T]([]int{batch, seqLen, m.dInner}, xData)
	if err != nil {
		return nil, err
	}
	zTensor, err := tensor.New[T]([]int{batch, seqLen, m.dInner}, zData)
	if err != nil {
		return nil, err
	}
	m.cachedZ = zTensor
	m.cachedXPreConv = xTensor

	// 3. Causal depthwise Conv1D on x: [batch, seq_len, d_inner]
	// Operate per-channel with left-padding (causal)
	xConvData := make([]T, batch*seqLen*m.dInner)
	xFlatData := xTensor.Data()
	convW := m.convWeight.Value.Data() // [d_inner * conv_ker]
	convB := m.convBias.Value.Data()   // [d_inner]
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < m.dInner; d++ {
				sum := convB[d]
				for k := 0; k < m.convKer; k++ {
					// Causal: look back (left-padded)
					srcPos := s - (m.convKer - 1) + k
					if srcPos >= 0 && srcPos < seqLen {
						xVal := xFlatData[(b*seqLen+srcPos)*m.dInner+d]
						wVal := convW[d*m.convKer+k]
						sum = m.ops.Add(sum, m.ops.Mul(xVal, wVal))
					}
				}
				xConvData[(b*seqLen+s)*m.dInner+d] = sum
			}
		}
	}
	xConv, err := tensor.New[T]([]int{batch, seqLen, m.dInner}, xConvData)
	if err != nil {
		return nil, err
	}
	m.cachedXConv = xConv

	// 4. SiLU on x: silu(x) = x * sigmoid(x)
	xSilu, err := m.applySiLU(ctx, xConv)
	if err != nil {
		return nil, fmt.Errorf("silu on x: %w", err)
	}
	m.cachedX = xSilu

	// 5. SSM parameter projection: x -> [dt_rank, B, C]
	// x_proj: [batch, seq_len, d_inner] -> [batch, seq_len, dt_rank + 2*d_state]
	xBCDt, err := m.xProj.Forward(ctx, xSilu)
	if err != nil {
		return nil, fmt.Errorf("x_proj forward: %w", err)
	}
	m.cachedXBCDt = xBCDt

	// Split x_proj output into dt_rank, B, C
	xbcdtData := xBCDt.Data()
	dtRankData := make([]T, batch*seqLen*m.dtRank)
	bData := make([]T, batch*seqLen*m.dState)
	cData := make([]T, batch*seqLen*m.dState)
	projWidth := m.dtRank + 2*m.dState
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * projWidth
			copy(dtRankData[(b*seqLen+s)*m.dtRank:], xbcdtData[off:off+m.dtRank])
			copy(bData[(b*seqLen+s)*m.dState:], xbcdtData[off+m.dtRank:off+m.dtRank+m.dState])
			copy(cData[(b*seqLen+s)*m.dState:], xbcdtData[off+m.dtRank+m.dState:off+projWidth])
		}
	}
	dtRankTensor, err := tensor.New[T]([]int{batch, seqLen, m.dtRank}, dtRankData)
	if err != nil {
		return nil, err
	}

	B, err := tensor.New[T]([]int{batch, seqLen, m.dState}, bData)
	if err != nil {
		return nil, err
	}
	C, err := tensor.New[T]([]int{batch, seqLen, m.dState}, cData)
	if err != nil {
		return nil, err
	}
	m.cachedB = B
	m.cachedC = C

	// 6. dt projection: [batch, seq_len, dt_rank] -> [batch, seq_len, d_inner]
	dtRaw, err := m.dtProj.Forward(ctx, dtRankTensor)
	if err != nil {
		return nil, fmt.Errorf("dt_proj forward: %w", err)
	}
	m.cachedDtRaw = dtRaw

	// 7. Softplus on dt: softplus(x) = log(1 + exp(x))
	dt, err := m.applySoftplus(ctx, dtRaw)
	if err != nil {
		return nil, fmt.Errorf("softplus on dt: %w", err)
	}
	m.cachedDt = dt

	// 8. Selective scan
	y, states, err := m.selectiveScan(ctx, xSilu, dt, B, C, batch, seqLen)
	if err != nil {
		return nil, fmt.Errorf("selective scan: %w", err)
	}
	m.cachedY = y
	m.cachedStates = states

	// 9. Gate: y * silu(z)
	siluZ, err := m.applySiLU(ctx, zTensor)
	if err != nil {
		return nil, fmt.Errorf("silu on z: %w", err)
	}
	m.cachedSiluZ = siluZ

	gated, err := m.engine.Mul(ctx, y, siluZ)
	if err != nil {
		return nil, fmt.Errorf("gating: %w", err)
	}

	// 10. Output projection: [batch, seq_len, d_inner] -> [batch, seq_len, d_model]
	output, err := m.outProj.Forward(ctx, gated)
	if err != nil {
		return nil, fmt.Errorf("out_proj forward: %w", err)
	}

	return output, nil
}

// selectiveScan runs the SSM recurrence with selective (input-dependent) parameters.
//
// Discretization depends on m.discMode:
//
// ZOH (zero-order hold):
//
//	dA = exp(dt * A)          [batch, seq, d_inner, d_state]
//	dB = dt * B               [batch, seq, d_inner, d_state]
//
// ExpTrap (exponential-trapezoidal, Mamba 3):
//
//	dA = exp(dt * A)
//	dB = dt * (1 + dA) / 2 * B   (trapezoidal average at both endpoints)
//
// Recurrence (per batch, per d_inner channel):
//
//	h[t] = dA[t] * h[t-1] + dB[t] * x[t]
//	y[t] = C[t] . h[t] + D * x[t]
//
// Returns y [batch, seq, d_inner] and all states [batch, seq, d_inner, d_state].
func (m *MambaBlock[T]) selectiveScan(
	ctx context.Context,
	x, dt, B, C *tensor.TensorNumeric[T],
	batch, seqLen int,
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], error) {
	xData := x.Data()
	dtData := dt.Data()
	bDataSlice := B.Data()
	cDataSlice := C.Data()
	aData := m.A.Value.Data()     // [d_inner, d_state]
	dData := m.D.Value.Data()     // [d_inner]

	yData := make([]T, batch*seqLen*m.dInner)
	// Store all hidden states for backward
	statesData := make([]T, batch*seqLen*m.dInner*m.dState)

	for b := 0; b < batch; b++ {
		// h: [d_inner, d_state] — running hidden state
		h := make([]T, m.dInner*m.dState)

		for s := 0; s < seqLen; s++ {
			bsOff := b*seqLen + s

			for d := 0; d < m.dInner; d++ {
				xVal := xData[bsOff*m.dInner+d]
				dtVal := dtData[bsOff*m.dInner+d]

				var yVal T
				for n := 0; n < m.dState; n++ {
					// dA = exp(dt * A_log)
					// A is stored in log-space, so A_real = -exp(A_log)
					aLog := aData[d*m.dState+n]
					aReal := T(-math.Exp(float64(aLog)))
					dA := T(math.Exp(float64(m.ops.Mul(dtVal, aReal))))

					bVal := bDataSlice[bsOff*m.dState+n]
					var dB T
					switch m.discMode {
					case ExpTrap:
						// B̄ = Δ * (1 + exp(Δ*A)) / 2 * B
						dB = m.ops.Mul(m.ops.Mul(dtVal, T((1.0+float64(dA))/2.0)), bVal)
					default: // ZOH
						// B̄ = Δ * B
						dB = m.ops.Mul(dtVal, bVal)
					}

					// h[d,n] = dA * h[d,n] + dB * x[t,d]
					hIdx := d*m.dState + n
					h[hIdx] = m.ops.Add(m.ops.Mul(dA, h[hIdx]), m.ops.Mul(dB, xVal))

					// y[t,d] += C[t,n] * h[d,n]
					cVal := cDataSlice[bsOff*m.dState+n]
					yVal = m.ops.Add(yVal, m.ops.Mul(cVal, h[hIdx]))
				}

				// Skip connection: y += D * x
				yVal = m.ops.Add(yVal, m.ops.Mul(dData[d], xVal))
				yData[bsOff*m.dInner+d] = yVal
			}

			// Store state for backward
			stOff := (b*seqLen + s) * m.dInner * m.dState
			copy(statesData[stOff:stOff+m.dInner*m.dState], h)
		}
	}

	yTensor, err := tensor.New[T]([]int{batch, seqLen, m.dInner}, yData)
	if err != nil {
		return nil, nil, err
	}
	statesTensor, err := tensor.New[T]([]int{batch, seqLen, m.dInner, m.dState}, statesData)
	if err != nil {
		return nil, nil, err
	}

	return yTensor, statesTensor, nil
}

// applySiLU computes silu(x) = x * sigmoid(x) element-wise.
func (m *MambaBlock[T]) applySiLU(_ context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	out := make([]T, len(data))
	for i, v := range data {
		sig := T(1.0 / (1.0 + math.Exp(-float64(v))))
		out[i] = m.ops.Mul(v, sig)
	}
	return tensor.New[T](x.Shape(), out)
}

// applySoftplus computes softplus(x) = log(1 + exp(x)) element-wise.
func (m *MambaBlock[T]) applySoftplus(_ context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	out := make([]T, len(data))
	for i, v := range data {
		out[i] = T(math.Log(1.0 + math.Exp(float64(v))))
	}
	return tensor.New[T](x.Shape(), out)
}

// Backward computes gradients for the Mamba block using the chain rule.
func (m *MambaBlock[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MambaBlock requires exactly 1 input for backward, got %d", len(inputs))
	}

	shape := m.lastInput.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// 10. Backward through outProj
	gated, err := m.engine.Mul(ctx, m.cachedY, m.cachedSiluZ)
	if err != nil {
		return nil, err
	}
	dGated, err := m.outProj.Backward(ctx, mode, outputGradient, gated)
	if err != nil {
		return nil, err
	}
	dGatedTensor := dGated[0] // [batch, seq, d_inner]

	// 9. Backward through gate: gated = y * silu(z)
	// dY = dGated * silu(z)
	dY, err := m.engine.Mul(ctx, dGatedTensor, m.cachedSiluZ)
	if err != nil {
		return nil, err
	}
	// dSiluZ = dGated * y
	dSiluZ, err := m.engine.Mul(ctx, dGatedTensor, m.cachedY)
	if err != nil {
		return nil, err
	}

	// dZ = dSiluZ * silu'(z)
	dZ := m.siluBackward(m.cachedZ, dSiluZ)

	// 8. Backward through selective scan
	dX_scan, dDt, dB_ssm, dC_ssm := m.selectiveScanBackward(batch, seqLen, dY)

	// 7. Backward through softplus on dt: softplus'(x) = sigmoid(x)
	dDtRaw := m.softplusBackward(m.cachedDtRaw, dDt)

	// 6. Backward through dtProj
	dtRankData := make([]T, batch*seqLen*m.dtRank)
	xbcdtData := m.cachedXBCDt.Data()
	projWidth := m.dtRank + 2*m.dState
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * projWidth
			copy(dtRankData[(b*seqLen+s)*m.dtRank:], xbcdtData[off:off+m.dtRank])
		}
	}
	dtRankTensor, err := tensor.New[T]([]int{batch, seqLen, m.dtRank}, dtRankData)
	if err != nil {
		return nil, err
	}
	dDtRank, err := m.dtProj.Backward(ctx, mode, dDtRaw, dtRankTensor)
	if err != nil {
		return nil, err
	}

	// 5. Backward through xProj: reassemble gradient for [dt_rank, B, C]
	dXBCDtData := make([]T, batch*seqLen*projWidth)
	dDtRankData := dDtRank[0].Data()
	dBData := dB_ssm.Data()
	dCData := dC_ssm.Data()
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * projWidth
			copy(dXBCDtData[off:off+m.dtRank], dDtRankData[(b*seqLen+s)*m.dtRank:])
			copy(dXBCDtData[off+m.dtRank:off+m.dtRank+m.dState], dBData[(b*seqLen+s)*m.dState:])
			copy(dXBCDtData[off+m.dtRank+m.dState:off+projWidth], dCData[(b*seqLen+s)*m.dState:])
		}
	}
	dXBCDt, err := tensor.New[T]([]int{batch, seqLen, projWidth}, dXBCDtData)
	if err != nil {
		return nil, err
	}

	dXSilu, err := m.xProj.Backward(ctx, mode, dXBCDt, m.cachedX)
	if err != nil {
		return nil, err
	}

	// Add scan gradient to xProj gradient
	dXTotal, err := m.engine.Add(ctx, dXSilu[0], dX_scan)
	if err != nil {
		return nil, err
	}

	// 4. Backward through SiLU on x
	dXConv := m.siluBackward(m.cachedXConv, dXTotal)

	// 3. Backward through causal depthwise conv1d
	dXPreConv, dConvW := m.conv1dBackward(batch, seqLen, dXConv)

	// Accumulate conv weight gradient
	m.convWeight.Gradient, err = m.engine.Add(ctx, m.convWeight.Gradient, dConvW, m.convWeight.Gradient)
	if err != nil {
		return nil, err
	}

	// Accumulate conv bias gradient: dBias[d] = sum over batch,seq of dXConv[b,s,d]
	dXConvData := dXConv.Data()
	dConvBiasData := make([]T, m.dInner)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < m.dInner; d++ {
				dConvBiasData[d] = m.ops.Add(dConvBiasData[d], dXConvData[(b*seqLen+s)*m.dInner+d])
			}
		}
	}
	dConvBias, err := tensor.New[T]([]int{m.dInner}, dConvBiasData)
	if err != nil {
		return nil, err
	}
	if m.convBias.Gradient == nil {
		m.convBias.Gradient = dConvBias
	} else {
		m.convBias.Gradient, err = m.engine.Add(ctx, m.convBias.Gradient, dConvBias, m.convBias.Gradient)
		if err != nil {
			return nil, err
		}
	}

	// 2. Backward through split: reassemble [dx, dz] -> [batch, seq, 2*d_inner]
	dXPreConvData := dXPreConv.Data()
	dZData := dZ.Data()
	dProjData := make([]T, batch*seqLen*2*m.dInner)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * (2 * m.dInner)
			copy(dProjData[off:off+m.dInner], dXPreConvData[(b*seqLen+s)*m.dInner:])
			copy(dProjData[off+m.dInner:off+2*m.dInner], dZData[(b*seqLen+s)*m.dInner:])
		}
	}
	dProj, err := tensor.New[T]([]int{batch, seqLen, 2 * m.dInner}, dProjData)
	if err != nil {
		return nil, err
	}

	// 1. Backward through inProj
	dInput, err := m.inProj.Backward(ctx, mode, dProj, m.lastInput)
	if err != nil {
		return nil, err
	}

	return dInput, nil
}

// selectiveScanBackward computes gradients through the selective scan.
func (m *MambaBlock[T]) selectiveScanBackward(
	batch, seqLen int,
	dY *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], *tensor.TensorNumeric[T], *tensor.TensorNumeric[T]) {
	dYData := dY.Data()
	xData := m.cachedX.Data()
	dtData := m.cachedDt.Data()
	bDataSlice := m.cachedB.Data()
	cDataSlice := m.cachedC.Data()
	aData := m.A.Value.Data()
	dParamData := m.D.Value.Data()
	statesData := m.cachedStates.Data()

	dXData := make([]T, batch*seqLen*m.dInner)
	dDtData := make([]T, batch*seqLen*m.dInner)
	dBData := make([]T, batch*seqLen*m.dState)
	dCData := make([]T, batch*seqLen*m.dState)

	// Accumulate A and D gradients
	dAData := make([]T, m.dInner*m.dState)
	dDData := make([]T, m.dInner)

	for b := 0; b < batch; b++ {
		// dh: gradient w.r.t. hidden state, propagated backward in time
		dh := make([]T, m.dInner*m.dState)

		for s := seqLen - 1; s >= 0; s-- {
			bsOff := b*seqLen + s

			for d := 0; d < m.dInner; d++ {
				dyVal := dYData[bsOff*m.dInner+d]
				xVal := xData[bsOff*m.dInner+d]
				dtVal := dtData[bsOff*m.dInner+d]

				// D gradient: dD += dY * x
				dDData[d] = m.ops.Add(dDData[d], m.ops.Mul(dyVal, xVal))

				// dX from D skip: dX += dY * D
				dXData[bsOff*m.dInner+d] = m.ops.Add(dXData[bsOff*m.dInner+d], m.ops.Mul(dyVal, dParamData[d]))

				for n := 0; n < m.dState; n++ {
					hIdx := d*m.dState + n
					cVal := cDataSlice[bsOff*m.dState+n]
					hVal := statesData[(bsOff)*m.dInner*m.dState+hIdx]

					// dC: y[t,d] = sum_n C[t,n] * h[t,d,n], so dC[t,n] += dY[t,d] * h[t,d,n]
					dCData[bsOff*m.dState+n] = m.ops.Add(dCData[bsOff*m.dState+n], m.ops.Mul(dyVal, hVal))

					// dh from output: dh[d,n] += dY[t,d] * C[t,n]
					dh[hIdx] = m.ops.Add(dh[hIdx], m.ops.Mul(dyVal, cVal))

					// Compute dA, dB discretization values
					aLog := aData[d*m.dState+n]
					aReal := T(-math.Exp(float64(aLog)))
					dAVal := T(math.Exp(float64(m.ops.Mul(dtVal, aReal))))
					bVal := bDataSlice[bsOff*m.dState+n]
					dBVal := m.ops.Mul(dtVal, bVal)
					_ = dBVal

					// Get previous state
					var hPrev T
					if s > 0 {
						prevOff := (b*seqLen + s - 1) * m.dInner * m.dState
						hPrev = statesData[prevOff+hIdx]
					}

					// h[t] = dA * h[t-1] + dB * x[t]
					// dh/d(dt): need dt gradient
					// d(dA)/d(dt) = dA * A_real
					// d(dB)/d(dt) = B[t,n]
					ddA_ddt := m.ops.Mul(dAVal, aReal)
					ddB_ddt := bVal

					// dt gradient: dDt += dh * (d(dA)/d(dt) * h[t-1] + d(dB)/d(dt) * x)
					dDtData[bsOff*m.dInner+d] = m.ops.Add(dDtData[bsOff*m.dInner+d],
						m.ops.Mul(dh[hIdx], m.ops.Add(
							m.ops.Mul(ddA_ddt, hPrev),
							m.ops.Mul(ddB_ddt, xVal),
						)),
					)

					// dB: dh * dt * x
					dBData[bsOff*m.dState+n] = m.ops.Add(dBData[bsOff*m.dState+n],
						m.ops.Mul(dh[hIdx], m.ops.Mul(dtVal, xVal)),
					)

					// dX from SSM: dh * dt * B[t,n]
					dXData[bsOff*m.dInner+d] = m.ops.Add(dXData[bsOff*m.dInner+d],
						m.ops.Mul(dh[hIdx], m.ops.Mul(dtVal, bVal)),
					)

					// dA (log-space): dh * h[t-1] * dA * dt * A_real * exp(A_log)
					// d(loss)/d(A_log) = d(loss)/d(dA) * d(dA)/d(A_real) * d(A_real)/d(A_log)
					// d(dA)/d(A_real) = dA * dt
					// d(A_real)/d(A_log) = -exp(A_log)
					dAData[d*m.dState+n] = m.ops.Add(dAData[d*m.dState+n],
						m.ops.Mul(dh[hIdx], m.ops.Mul(hPrev, m.ops.Mul(dAVal, m.ops.Mul(dtVal, m.ops.Mul(aReal, T(-math.Exp(float64(aLog)))))))),
					)

					// Propagate dh backward: dh[t-1] += dh[t] * dA
					// But we need to update dh for the next iteration (s-1)
					// So multiply current dh by dA (the transition)
					dh[hIdx] = m.ops.Mul(dh[hIdx], dAVal)
				}
			}
		}
	}

	// Accumulate A and D parameter gradients
	dATensor, _ := tensor.New[T](m.A.Value.Shape(), dAData)
	if m.A.Gradient != nil {
		m.A.Gradient, _ = m.engine.Add(context.Background(), m.A.Gradient, dATensor)
	} else {
		m.A.Gradient = dATensor
	}

	dDTensor, _ := tensor.New[T](m.D.Value.Shape(), dDData)
	if m.D.Gradient != nil {
		m.D.Gradient, _ = m.engine.Add(context.Background(), m.D.Gradient, dDTensor)
	} else {
		m.D.Gradient = dDTensor
	}

	dX, _ := tensor.New[T]([]int{batch, seqLen, m.dInner}, dXData)
	dDt, _ := tensor.New[T]([]int{batch, seqLen, m.dInner}, dDtData)
	dBTensor, _ := tensor.New[T]([]int{batch, seqLen, m.dState}, dBData)
	dCTensor, _ := tensor.New[T]([]int{batch, seqLen, m.dState}, dCData)

	return dX, dDt, dBTensor, dCTensor
}

// siluBackward computes dInput given dOutput and the pre-activation input.
// silu(x) = x * sigmoid(x)
// silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
func (m *MambaBlock[T]) siluBackward(input, dOutput *tensor.TensorNumeric[T]) *tensor.TensorNumeric[T] {
	inData := input.Data()
	dOutData := dOutput.Data()
	result := make([]T, len(inData))
	for i, x := range inData {
		sig := T(1.0 / (1.0 + math.Exp(-float64(x))))
		grad := m.ops.Mul(sig, T(1.0+float64(x)*(1.0-float64(sig))))
		result[i] = m.ops.Mul(dOutData[i], grad)
	}
	t, _ := tensor.New[T](input.Shape(), result)
	return t
}

// softplusBackward computes gradient through softplus.
// softplus'(x) = sigmoid(x) = 1 / (1 + exp(-x))
func (m *MambaBlock[T]) softplusBackward(input, dOutput *tensor.TensorNumeric[T]) *tensor.TensorNumeric[T] {
	inData := input.Data()
	dOutData := dOutput.Data()
	result := make([]T, len(inData))
	for i, x := range inData {
		sig := T(1.0 / (1.0 + math.Exp(-float64(x))))
		result[i] = m.ops.Mul(dOutData[i], sig)
	}
	t, _ := tensor.New[T](input.Shape(), result)
	return t
}

// conv1dBackward computes gradients through the causal depthwise conv1d.
func (m *MambaBlock[T]) conv1dBackward(
	batch, seqLen int,
	dOutput *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T]) {
	dOutData := dOutput.Data()
	xData := m.cachedXPreConv.Data()
	convW := m.convWeight.Value.Data()

	dXData := make([]T, batch*seqLen*m.dInner)
	dWData := make([]T, m.dInner*m.convKer)

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < m.dInner; d++ {
				dOut := dOutData[(b*seqLen+s)*m.dInner+d]
				for k := 0; k < m.convKer; k++ {
					srcPos := s - (m.convKer - 1) + k
					if srcPos >= 0 && srcPos < seqLen {
						// dX[srcPos, d] += dOut * w[d, k]
						dXData[(b*seqLen+srcPos)*m.dInner+d] = m.ops.Add(
							dXData[(b*seqLen+srcPos)*m.dInner+d],
							m.ops.Mul(dOut, convW[d*m.convKer+k]),
						)
						// dW[d, k] += dOut * x[srcPos, d]
						dWData[d*m.convKer+k] = m.ops.Add(
							dWData[d*m.convKer+k],
							m.ops.Mul(dOut, xData[(b*seqLen+srcPos)*m.dInner+d]),
						)
					}
				}
			}
		}
	}

	dX, _ := tensor.New[T]([]int{batch, seqLen, m.dInner}, dXData)
	dW, _ := tensor.New[T](m.convWeight.Value.Shape(), dWData)
	return dX, dW
}

// Parameters returns all trainable parameters.
func (m *MambaBlock[T]) Parameters() []*graph.Parameter[T] {
	params := m.inProj.Parameters()
	params = append(params, m.convWeight)
	params = append(params, m.convBias)
	params = append(params, m.xProj.Parameters()...)
	params = append(params, m.dtProj.Parameters()...)
	params = append(params, m.A)
	params = append(params, m.D)
	params = append(params, m.outProj.Parameters()...)
	return params
}

// Statically assert that MambaBlock implements graph.Node.
var _ graph.Node[float32] = (*MambaBlock[float32])(nil)
