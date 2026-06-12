package ssm

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// MIMOMambaBlock implements a multi-input multi-output SSM block with multiple
// parallel state spaces (heads) and cross-channel mixing.
//
// Architecture:
//   - Input projection: d_model -> 2*d_inner (x and z branches)
//   - Depthwise causal Conv1D on x branch
//   - SiLU activation on x
//   - SSM parameter projection: x -> (dt, B, C) per head
//   - Multi-head selective scan: each head processes d_inner/num_heads channels
//     with its own A, D parameters and independent state space
//   - Cross-head mixing: linear projection across head outputs
//   - Gate: mixed_y * SiLU(z)
//   - Output projection: d_inner -> d_model
//
// The multi-head design allows different heads to specialize on different
// temporal patterns, similar to multi-head attention but for SSM recurrence.
//
// Input shape:  [batch, seq_len, d_model]
// Output shape: [batch, seq_len, d_model]
type MIMOMambaBlock[T tensor.Numeric] struct {
	name   string
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Dimensions
	dModel   int
	dInner   int
	dState   int
	dtRank   int
	convKer  int
	numHeads int
	headDim  int // dInner / numHeads

	// Discretization mode
	discMode DiscretizationMode

	// Projections
	inProj  *Linear[T] // d_model -> 2*d_inner
	xProj   *Linear[T] // d_inner -> dt_rank + 2*d_state*num_heads
	dtProj  *Linear[T] // dt_rank -> d_inner
	outProj *Linear[T] // d_inner -> d_model

	// Cross-head mixing: d_inner -> d_inner
	headMix *Linear[T]

	// Conv1D weight: [d_inner, 1, conv_ker]
	convWeight *graph.Parameter[T]

	// Per-head SSM parameters
	A []*graph.Parameter[T] // each [headDim, d_state]
	D []*graph.Parameter[T] // each [headDim]

	// Cached intermediates for backward
	lastInput      *tensor.TensorNumeric[T]
	cachedX        *tensor.TensorNumeric[T]
	cachedZ        *tensor.TensorNumeric[T]
	cachedSiluZ    *tensor.TensorNumeric[T]
	cachedDt       *tensor.TensorNumeric[T]
	cachedB        []*tensor.TensorNumeric[T] // per-head [batch, seq, d_state]
	cachedC        []*tensor.TensorNumeric[T] // per-head [batch, seq, d_state]
	cachedY        *tensor.TensorNumeric[T]   // after mixing
	cachedYRaw     *tensor.TensorNumeric[T]   // before mixing
	cachedStates   []*tensor.TensorNumeric[T] // per-head
	cachedXConv    *tensor.TensorNumeric[T]
	cachedXBCDt    *tensor.TensorNumeric[T]
	cachedDtRaw    *tensor.TensorNumeric[T]
	cachedXPreConv *tensor.TensorNumeric[T]
}

// MIMOMambaBlockOption is a functional option for MIMOMambaBlock.
type MIMOMambaBlockOption[T tensor.Numeric] func(*MIMOMambaBlock[T])

// WithMIMODiscretizationMode sets the SSM discretization mode.
func WithMIMODiscretizationMode[T tensor.Numeric](mode DiscretizationMode) MIMOMambaBlockOption[T] {
	return func(m *MIMOMambaBlock[T]) {
		m.discMode = mode
	}
}

// NewMIMOMambaBlock creates a new multi-head MIMO SSM block.
//
// Parameters:
//   - dModel: input/output dimension
//   - dInner: inner SSM dimension (must be divisible by numHeads)
//   - dState: SSM state dimension per head
//   - dtRank: rank of dt projection
//   - convKer: depthwise conv1d kernel size
//   - numHeads: number of parallel SSM heads
func NewMIMOMambaBlock[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	dModel, dInner, dState, dtRank, convKer, numHeads int,
	opts ...MIMOMambaBlockOption[T],
) (*MIMOMambaBlock[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if dModel <= 0 || dInner <= 0 || dState <= 0 || dtRank <= 0 || convKer <= 0 || numHeads <= 0 {
		return nil, fmt.Errorf("all dimensions must be positive")
	}
	if dInner%numHeads != 0 {
		return nil, fmt.Errorf("dInner (%d) must be divisible by numHeads (%d)", dInner, numHeads)
	}

	headDim := dInner / numHeads

	inProj, err := NewLinear[T](name+"_in_proj", engine, ops, dModel, 2*dInner)
	if err != nil {
		return nil, fmt.Errorf("creating in_proj: %w", err)
	}

	// xProj outputs dt_rank + 2*d_state*num_heads (separate B,C per head)
	xProjOut := dtRank + 2*dState*numHeads
	xProj, err := NewLinear[T](name+"_x_proj", engine, ops, dInner, xProjOut)
	if err != nil {
		return nil, fmt.Errorf("creating x_proj: %w", err)
	}

	dtProj, err := NewLinear[T](name+"_dt_proj", engine, ops, dtRank, dInner)
	if err != nil {
		return nil, fmt.Errorf("creating dt_proj: %w", err)
	}

	outProj, err := NewLinear[T](name+"_out_proj", engine, ops, dInner, dModel)
	if err != nil {
		return nil, fmt.Errorf("creating out_proj: %w", err)
	}

	// Cross-head mixing projection
	headMix, err := NewLinear[T](name+"_head_mix", engine, ops, dInner, dInner)
	if err != nil {
		return nil, fmt.Errorf("creating head_mix: %w", err)
	}

	// Conv1D weight
	convData := randomData[T](dInner * convKer)
	convTensor, err := tensor.New[T]([]int{dInner, 1, convKer}, convData)
	if err != nil {
		return nil, err
	}
	convWeight, err := graph.NewParameter[T](name+"_conv_weight", convTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	// Per-head A and D parameters
	aParams := make([]*graph.Parameter[T], numHeads)
	dParams := make([]*graph.Parameter[T], numHeads)
	for h := 0; h < numHeads; h++ {
		// A: log-space initialization
		aData := make([]T, headDim*dState)
		for i := 0; i < headDim; i++ {
			for j := 0; j < dState; j++ {
				aData[i*dState+j] = T(math.Log(float64(j + 1)))
			}
		}
		aTensor, err := tensor.New[T]([]int{headDim, dState}, aData)
		if err != nil {
			return nil, err
		}
		aParams[h], err = graph.NewParameter[T](fmt.Sprintf("%s_A_h%d", name, h), aTensor, tensor.New[T])
		if err != nil {
			return nil, err
		}

		// D: skip connection initialized to ones
		dData := make([]T, headDim)
		for i := range dData {
			dData[i] = T(1.0)
		}
		dTensor, err := tensor.New[T]([]int{headDim}, dData)
		if err != nil {
			return nil, err
		}
		dParams[h], err = graph.NewParameter[T](fmt.Sprintf("%s_D_h%d", name, h), dTensor, tensor.New[T])
		if err != nil {
			return nil, err
		}
	}

	block := &MIMOMambaBlock[T]{
		name:     name,
		engine:   engine,
		ops:      ops,
		dModel:   dModel,
		dInner:   dInner,
		dState:   dState,
		dtRank:   dtRank,
		convKer:  convKer,
		numHeads: numHeads,
		headDim:  headDim,
		discMode: ZOH,
		inProj:   inProj,
		xProj:    xProj,
		dtProj:   dtProj,
		outProj:  outProj,
		headMix:  headMix,
		convWeight: convWeight,
		A:        aParams,
		D:        dParams,
	}
	for _, opt := range opts {
		opt(block)
	}
	return block, nil
}

func (m *MIMOMambaBlock[T]) OpType() string { return "MIMOMambaBlock" }

func (m *MIMOMambaBlock[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"d_model":     m.dModel,
		"d_inner":     m.dInner,
		"d_state":     m.dState,
		"dt_rank":     m.dtRank,
		"kernel_size": m.convKer,
		"num_heads":   m.numHeads,
		"head_dim":    m.headDim,
	}
}

func (m *MIMOMambaBlock[T]) OutputShape() []int {
	return []int{-1, -1, m.dModel}
}

func (m *MIMOMambaBlock[T]) Name() string    { return m.name }
func (m *MIMOMambaBlock[T]) SetName(n string) { m.name = n }

// Forward computes the MIMO Mamba block forward pass.
// Input: [batch, seq_len, d_model]
// Output: [batch, seq_len, d_model]
func (m *MIMOMambaBlock[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MIMOMambaBlock requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("MIMOMambaBlock input must be 3D [batch, seq_len, d_model], got %v", shape)
	}

	batch := shape[0]
	seqLen := shape[1]
	m.lastInput = input

	// 1. Input projection: [batch, seq_len, d_model] -> [batch, seq_len, 2*d_inner]
	projected, err := m.inProj.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("in_proj forward: %w", err)
	}

	// 2. Split into x and z branches
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

	// 3. Causal depthwise Conv1D on x
	xConvData := make([]T, batch*seqLen*m.dInner)
	xFlatData := xTensor.Data()
	convW := m.convWeight.Value.Data()
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < m.dInner; d++ {
				var sum T
				for k := 0; k < m.convKer; k++ {
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

	// 4. SiLU on x
	xSilu, err := m.applySiLU(ctx, xConv)
	if err != nil {
		return nil, fmt.Errorf("silu on x: %w", err)
	}
	m.cachedX = xSilu

	// 5. SSM parameter projection: x -> [dt_rank, B_h0, C_h0, B_h1, C_h1, ...]
	xBCDt, err := m.xProj.Forward(ctx, xSilu)
	if err != nil {
		return nil, fmt.Errorf("x_proj forward: %w", err)
	}
	m.cachedXBCDt = xBCDt

	// Split into dt_rank and per-head B, C
	xbcdtData := xBCDt.Data()
	projWidth := m.dtRank + 2*m.dState*m.numHeads
	dtRankData := make([]T, batch*seqLen*m.dtRank)

	m.cachedB = make([]*tensor.TensorNumeric[T], m.numHeads)
	m.cachedC = make([]*tensor.TensorNumeric[T], m.numHeads)
	headBData := make([][]T, m.numHeads)
	headCData := make([][]T, m.numHeads)
	for h := 0; h < m.numHeads; h++ {
		headBData[h] = make([]T, batch*seqLen*m.dState)
		headCData[h] = make([]T, batch*seqLen*m.dState)
	}

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			idx := b*seqLen + s
			off := idx * projWidth
			copy(dtRankData[idx*m.dtRank:], xbcdtData[off:off+m.dtRank])
			bcOff := off + m.dtRank
			for h := 0; h < m.numHeads; h++ {
				hOff := bcOff + h*2*m.dState
				copy(headBData[h][idx*m.dState:], xbcdtData[hOff:hOff+m.dState])
				copy(headCData[h][idx*m.dState:], xbcdtData[hOff+m.dState:hOff+2*m.dState])
			}
		}
	}

	dtRankTensor, err := tensor.New[T]([]int{batch, seqLen, m.dtRank}, dtRankData)
	if err != nil {
		return nil, err
	}
	for h := 0; h < m.numHeads; h++ {
		m.cachedB[h], err = tensor.New[T]([]int{batch, seqLen, m.dState}, headBData[h])
		if err != nil {
			return nil, err
		}
		m.cachedC[h], err = tensor.New[T]([]int{batch, seqLen, m.dState}, headCData[h])
		if err != nil {
			return nil, err
		}
	}

	// 6. dt projection: [batch, seq_len, dt_rank] -> [batch, seq_len, d_inner]
	dtRaw, err := m.dtProj.Forward(ctx, dtRankTensor)
	if err != nil {
		return nil, fmt.Errorf("dt_proj forward: %w", err)
	}
	m.cachedDtRaw = dtRaw

	// 7. Softplus on dt
	dt, err := m.applySoftplus(ctx, dtRaw)
	if err != nil {
		return nil, fmt.Errorf("softplus on dt: %w", err)
	}
	m.cachedDt = dt

	// 8. Multi-head selective scan
	yData := make([]T, batch*seqLen*m.dInner)
	m.cachedStates = make([]*tensor.TensorNumeric[T], m.numHeads)

	xSiluData := xSilu.Data()
	dtData := dt.Data()

	for h := 0; h < m.numHeads; h++ {
		headY, headStates, err := m.headSelectiveScan(
			ctx, h, xSiluData, dtData, m.cachedB[h].Data(), m.cachedC[h].Data(),
			batch, seqLen,
		)
		if err != nil {
			return nil, fmt.Errorf("head %d selective scan: %w", h, err)
		}
		m.cachedStates[h] = headStates

		// Copy head output into the correct channels of yData
		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				idx := b*seqLen + s
				srcOff := idx * m.headDim
				dstOff := idx*m.dInner + h*m.headDim
				copy(yData[dstOff:dstOff+m.headDim], headY[srcOff:srcOff+m.headDim])
			}
		}
	}

	yRaw, err := tensor.New[T]([]int{batch, seqLen, m.dInner}, yData)
	if err != nil {
		return nil, err
	}
	m.cachedYRaw = yRaw

	// 9. Cross-head mixing: linear projection across all head outputs
	yMixed, err := m.headMix.Forward(ctx, yRaw)
	if err != nil {
		return nil, fmt.Errorf("head_mix forward: %w", err)
	}
	m.cachedY = yMixed

	// 10. Gate: y_mixed * silu(z)
	siluZ, err := m.applySiLU(ctx, zTensor)
	if err != nil {
		return nil, fmt.Errorf("silu on z: %w", err)
	}
	m.cachedSiluZ = siluZ

	gated, err := m.engine.Mul(ctx, yMixed, siluZ)
	if err != nil {
		return nil, fmt.Errorf("gating: %w", err)
	}

	// 11. Output projection: [batch, seq_len, d_inner] -> [batch, seq_len, d_model]
	output, err := m.outProj.Forward(ctx, gated)
	if err != nil {
		return nil, fmt.Errorf("out_proj forward: %w", err)
	}

	return output, nil
}

// headSelectiveScan runs the SSM recurrence for a single head.
// The head operates on channels [h*headDim : (h+1)*headDim] of the input.
// Returns flat y data [batch*seq*headDim] and states tensor [batch, seq, headDim, dState].
func (m *MIMOMambaBlock[T]) headSelectiveScan(
	_ context.Context,
	headIdx int,
	xData, dtData []T,
	bData, cData []T,
	batch, seqLen int,
) ([]T, *tensor.TensorNumeric[T], error) {
	aData := m.A[headIdx].Value.Data()
	dData := m.D[headIdx].Value.Data()
	chanOff := headIdx * m.headDim

	yData := make([]T, batch*seqLen*m.headDim)
	statesData := make([]T, batch*seqLen*m.headDim*m.dState)

	for b := 0; b < batch; b++ {
		h := make([]T, m.headDim*m.dState)

		for s := 0; s < seqLen; s++ {
			bsOff := b*seqLen + s

			for d := 0; d < m.headDim; d++ {
				globalD := chanOff + d
				xVal := xData[bsOff*m.dInner+globalD]
				dtVal := dtData[bsOff*m.dInner+globalD]

				var yVal T
				for n := 0; n < m.dState; n++ {
					aLog := aData[d*m.dState+n]
					aReal := T(-math.Exp(float64(aLog)))
					dA := T(math.Exp(float64(m.ops.Mul(dtVal, aReal))))

					bVal := bData[bsOff*m.dState+n]
					var dB T
					switch m.discMode {
					case ExpTrap:
						dB = m.ops.Mul(m.ops.Mul(dtVal, T((1.0+float64(dA))/2.0)), bVal)
					default:
						dB = m.ops.Mul(dtVal, bVal)
					}

					hIdx := d*m.dState + n
					h[hIdx] = m.ops.Add(m.ops.Mul(dA, h[hIdx]), m.ops.Mul(dB, xVal))

					cVal := cData[bsOff*m.dState+n]
					yVal = m.ops.Add(yVal, m.ops.Mul(cVal, h[hIdx]))
				}

				yVal = m.ops.Add(yVal, m.ops.Mul(dData[d], xVal))
				yData[bsOff*m.headDim+d] = yVal
			}

			stOff := bsOff * m.headDim * m.dState
			copy(statesData[stOff:stOff+m.headDim*m.dState], h)
		}
	}

	statesTensor, err := tensor.New[T]([]int{batch, seqLen, m.headDim, m.dState}, statesData)
	if err != nil {
		return nil, nil, err
	}

	return yData, statesTensor, nil
}

func (m *MIMOMambaBlock[T]) applySiLU(_ context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	out := make([]T, len(data))
	for i, v := range data {
		sig := T(1.0 / (1.0 + math.Exp(-float64(v))))
		out[i] = m.ops.Mul(v, sig)
	}
	return tensor.New[T](x.Shape(), out)
}

func (m *MIMOMambaBlock[T]) applySoftplus(_ context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	out := make([]T, len(data))
	for i, v := range data {
		out[i] = T(math.Log(1.0 + math.Exp(float64(v))))
	}
	return tensor.New[T](x.Shape(), out)
}

// Backward computes gradients for the MIMO Mamba block.
func (m *MIMOMambaBlock[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MIMOMambaBlock requires exactly 1 input for backward, got %d", len(inputs))
	}

	shape := m.lastInput.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// 11. Backward through outProj
	gated, err := m.engine.Mul(ctx, m.cachedY, m.cachedSiluZ)
	if err != nil {
		return nil, err
	}
	dGated, err := m.outProj.Backward(ctx, mode, outputGradient, gated)
	if err != nil {
		return nil, err
	}
	dGatedTensor := dGated[0]

	// 10. Backward through gate
	dYMixed, err := m.engine.Mul(ctx, dGatedTensor, m.cachedSiluZ)
	if err != nil {
		return nil, err
	}
	dSiluZ, err := m.engine.Mul(ctx, dGatedTensor, m.cachedY)
	if err != nil {
		return nil, err
	}
	dZ := m.siluBackward(m.cachedZ, dSiluZ)

	// 9. Backward through head mixing
	dYRaw, err := m.headMix.Backward(ctx, mode, dYMixed, m.cachedYRaw)
	if err != nil {
		return nil, err
	}
	dYRawData := dYRaw[0].Data()

	// 8. Backward through multi-head selective scan
	dXScanData := make([]T, batch*seqLen*m.dInner)
	dDtData := make([]T, batch*seqLen*m.dInner)
	dBAllData := make([][]T, m.numHeads)
	dCAllData := make([][]T, m.numHeads)

	for h := 0; h < m.numHeads; h++ {
		// Extract per-head dY
		headDY := make([]T, batch*seqLen*m.headDim)
		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				idx := b*seqLen + s
				srcOff := idx*m.dInner + h*m.headDim
				dstOff := idx * m.headDim
				copy(headDY[dstOff:dstOff+m.headDim], dYRawData[srcOff:srcOff+m.headDim])
			}
		}

		dXHead, dDtHead, dBHead, dCHead := m.headScanBackward(h, batch, seqLen, headDY)
		dBAllData[h] = dBHead
		dCAllData[h] = dCHead

		// Accumulate into global dX and dDt
		chanOff := h * m.headDim
		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				idx := b*seqLen + s
				for d := 0; d < m.headDim; d++ {
					globalD := chanOff + d
					dXScanData[idx*m.dInner+globalD] = m.ops.Add(
						dXScanData[idx*m.dInner+globalD],
						dXHead[idx*m.headDim+d],
					)
					dDtData[idx*m.dInner+globalD] = m.ops.Add(
						dDtData[idx*m.dInner+globalD],
						dDtHead[idx*m.headDim+d],
					)
				}
			}
		}
	}

	dX_scan, err := tensor.New[T]([]int{batch, seqLen, m.dInner}, dXScanData)
	if err != nil {
		return nil, err
	}
	dDt, err := tensor.New[T]([]int{batch, seqLen, m.dInner}, dDtData)
	if err != nil {
		return nil, err
	}

	// 7. Backward through softplus
	dDtRaw := m.softplusBackward(m.cachedDtRaw, dDt)

	// 6. Backward through dtProj
	dtRankData := make([]T, batch*seqLen*m.dtRank)
	xbcdtData := m.cachedXBCDt.Data()
	projWidth := m.dtRank + 2*m.dState*m.numHeads
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

	// 5. Backward through xProj: reassemble gradient for [dt_rank, B_h0, C_h0, ...]
	dXBCDtData := make([]T, batch*seqLen*projWidth)
	dDtRankData := dDtRank[0].Data()
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			idx := b*seqLen + s
			off := idx * projWidth
			copy(dXBCDtData[off:off+m.dtRank], dDtRankData[idx*m.dtRank:])
			bcOff := off + m.dtRank
			for h := 0; h < m.numHeads; h++ {
				hOff := bcOff + h*2*m.dState
				copy(dXBCDtData[hOff:hOff+m.dState], dBAllData[h][idx*m.dState:])
				copy(dXBCDtData[hOff+m.dState:hOff+2*m.dState], dCAllData[h][idx*m.dState:])
			}
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

	dXTotal, err := m.engine.Add(ctx, dXSilu[0], dX_scan)
	if err != nil {
		return nil, err
	}

	// 4. Backward through SiLU on x
	dXConv := m.siluBackward(m.cachedXConv, dXTotal)

	// 3. Backward through conv1d
	dXPreConv, dConvW := m.conv1dBackward(batch, seqLen, dXConv)
	if m.convWeight.Gradient != nil {
		m.convWeight.Gradient, err = m.engine.Add(ctx, m.convWeight.Gradient, dConvW, m.convWeight.Gradient)
		if err != nil {
			return nil, err
		}
	} else {
		m.convWeight.Gradient = dConvW
	}

	// 2. Reassemble [dx, dz]
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

// headScanBackward computes gradients for a single head's selective scan.
func (m *MIMOMambaBlock[T]) headScanBackward(
	headIdx, batch, seqLen int,
	dYData []T,
) (dXData, dDtData, dBData, dCData []T) {
	xData := m.cachedX.Data()
	dtFull := m.cachedDt.Data()
	bData := m.cachedB[headIdx].Data()
	cData := m.cachedC[headIdx].Data()
	aData := m.A[headIdx].Value.Data()
	dParamData := m.D[headIdx].Value.Data()
	statesData := m.cachedStates[headIdx].Data()
	chanOff := headIdx * m.headDim

	dXData = make([]T, batch*seqLen*m.headDim)
	dDtData = make([]T, batch*seqLen*m.headDim)
	dBData = make([]T, batch*seqLen*m.dState)
	dCData = make([]T, batch*seqLen*m.dState)

	dAData := make([]T, m.headDim*m.dState)
	dDParamData := make([]T, m.headDim)

	for b := 0; b < batch; b++ {
		dh := make([]T, m.headDim*m.dState)

		for s := seqLen - 1; s >= 0; s-- {
			bsOff := b*seqLen + s

			for d := 0; d < m.headDim; d++ {
				globalD := chanOff + d
				dyVal := dYData[bsOff*m.headDim+d]
				xVal := xData[bsOff*m.dInner+globalD]
				dtVal := dtFull[bsOff*m.dInner+globalD]

				dDParamData[d] = m.ops.Add(dDParamData[d], m.ops.Mul(dyVal, xVal))
				dXData[bsOff*m.headDim+d] = m.ops.Add(dXData[bsOff*m.headDim+d], m.ops.Mul(dyVal, dParamData[d]))

				for n := 0; n < m.dState; n++ {
					hIdx := d*m.dState + n
					cVal := cData[bsOff*m.dState+n]
					hVal := statesData[bsOff*m.headDim*m.dState+hIdx]

					dCData[bsOff*m.dState+n] = m.ops.Add(dCData[bsOff*m.dState+n], m.ops.Mul(dyVal, hVal))
					dh[hIdx] = m.ops.Add(dh[hIdx], m.ops.Mul(dyVal, cVal))

					aLog := aData[d*m.dState+n]
					aReal := T(-math.Exp(float64(aLog)))
					dAVal := T(math.Exp(float64(m.ops.Mul(dtVal, aReal))))
					bVal := bData[bsOff*m.dState+n]

					var hPrev T
					if s > 0 {
						prevOff := (b*seqLen + s - 1) * m.headDim * m.dState
						hPrev = statesData[prevOff+hIdx]
					}

					ddA_ddt := m.ops.Mul(dAVal, aReal)
					ddB_ddt := bVal

					dDtData[bsOff*m.headDim+d] = m.ops.Add(dDtData[bsOff*m.headDim+d],
						m.ops.Mul(dh[hIdx], m.ops.Add(
							m.ops.Mul(ddA_ddt, hPrev),
							m.ops.Mul(ddB_ddt, xVal),
						)),
					)

					dBData[bsOff*m.dState+n] = m.ops.Add(dBData[bsOff*m.dState+n],
						m.ops.Mul(dh[hIdx], m.ops.Mul(dtVal, xVal)),
					)

					dXData[bsOff*m.headDim+d] = m.ops.Add(dXData[bsOff*m.headDim+d],
						m.ops.Mul(dh[hIdx], m.ops.Mul(dtVal, bVal)),
					)

					dAData[d*m.dState+n] = m.ops.Add(dAData[d*m.dState+n],
						m.ops.Mul(dh[hIdx], m.ops.Mul(hPrev, m.ops.Mul(dAVal, m.ops.Mul(dtVal, m.ops.Mul(aReal, T(-math.Exp(float64(aLog)))))))),
					)

					dh[hIdx] = m.ops.Mul(dh[hIdx], dAVal)
				}
			}
		}
	}

	// Accumulate per-head A and D gradients
	dATensor, _ := tensor.New[T](m.A[headIdx].Value.Shape(), dAData)
	if m.A[headIdx].Gradient != nil {
		m.A[headIdx].Gradient, _ = m.engine.Add(context.Background(), m.A[headIdx].Gradient, dATensor)
	} else {
		m.A[headIdx].Gradient = dATensor
	}

	dDTensor, _ := tensor.New[T](m.D[headIdx].Value.Shape(), dDParamData)
	if m.D[headIdx].Gradient != nil {
		m.D[headIdx].Gradient, _ = m.engine.Add(context.Background(), m.D[headIdx].Gradient, dDTensor)
	} else {
		m.D[headIdx].Gradient = dDTensor
	}

	return dXData, dDtData, dBData, dCData
}

func (m *MIMOMambaBlock[T]) siluBackward(input, dOutput *tensor.TensorNumeric[T]) *tensor.TensorNumeric[T] {
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

func (m *MIMOMambaBlock[T]) softplusBackward(input, dOutput *tensor.TensorNumeric[T]) *tensor.TensorNumeric[T] {
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

func (m *MIMOMambaBlock[T]) conv1dBackward(
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
						dXData[(b*seqLen+srcPos)*m.dInner+d] = m.ops.Add(
							dXData[(b*seqLen+srcPos)*m.dInner+d],
							m.ops.Mul(dOut, convW[d*m.convKer+k]),
						)
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
func (m *MIMOMambaBlock[T]) Parameters() []*graph.Parameter[T] {
	params := m.inProj.Parameters()
	params = append(params, m.convWeight)
	params = append(params, m.xProj.Parameters()...)
	params = append(params, m.dtProj.Parameters()...)
	for h := 0; h < m.numHeads; h++ {
		params = append(params, m.A[h])
		params = append(params, m.D[h])
	}
	params = append(params, m.headMix.Parameters()...)
	params = append(params, m.outProj.Parameters()...)
	return params
}

// Statically assert that MIMOMambaBlock implements graph.Node.
var _ graph.Node[float32] = (*MIMOMambaBlock[float32])(nil)
