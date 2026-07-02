// Package ssm implements state space model layers.
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

	"github.com/zerfoo/zerfoo/layers/embeddings"
)

// ComplexSSMState implements complex-valued SSM state tracking using RoPE
// embeddings on the B and C matrices. The hidden state is split into pairs
// of dimensions that are treated as (real, imaginary) components. RoPE
// rotates each pair by a position-dependent angle, encoding temporal
// structure into the state without doubling memory.
//
// This follows the Mamba 3 design where B and C are rotated by RoPE
// before the selective scan, giving the recurrence complex-valued dynamics.
//
// Input shape:  [batch, seq_len, d_model]
// Output shape: [batch, seq_len, d_model]
type ComplexSSMState[T tensor.Numeric] struct {
	name   string
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Dimensions
	dModel  int
	dInner  int
	dState  int // must be even for complex pairing
	dtRank  int
	convKer int

	// Discretization mode
	discMode DiscretizationMode

	// RoPE for B and C matrices
	ropeB *embeddings.RotaryPositionalEmbedding[T]
	ropeC *embeddings.RotaryPositionalEmbedding[T]

	// BCNorm stabilization
	bcNormB *BCNorm[T]
	bcNormC *BCNorm[T]

	// Projections
	inProj  *Linear[T]
	xProj   *Linear[T]
	dtProj  *Linear[T]
	outProj *Linear[T]

	// Conv1D weight: [d_inner, 1, conv_ker] (depthwise)
	convWeight *graph.Parameter[T]

	// SSM parameters
	A *graph.Parameter[T] // [d_inner, d_state]
	D *graph.Parameter[T] // [d_inner]

	// Cached intermediates for backward
	lastInput      *tensor.TensorNumeric[T]
	cachedX        *tensor.TensorNumeric[T]
	cachedZ        *tensor.TensorNumeric[T]
	cachedSiluZ    *tensor.TensorNumeric[T]
	cachedDt       *tensor.TensorNumeric[T]
	cachedB        *tensor.TensorNumeric[T] // after RoPE
	cachedC        *tensor.TensorNumeric[T] // after RoPE
	cachedY        *tensor.TensorNumeric[T]
	cachedStates   *tensor.TensorNumeric[T]
	cachedXConv    *tensor.TensorNumeric[T]
	cachedXBCDt    *tensor.TensorNumeric[T]
	cachedDtRaw    *tensor.TensorNumeric[T]
	cachedXPreConv *tensor.TensorNumeric[T]
}

// Linear is a simple linear projection layer used by ComplexSSMState.
type Linear[T tensor.Numeric] struct {
	name    string
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	weights *graph.Parameter[T]
	inDim   int
	outDim  int
}

// NewLinear creates a simple linear projection.
func NewLinear[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inDim, outDim int) (*Linear[T], error) {
	data := randomData[T](inDim * outDim)
	scale := T(math.Sqrt(2.0 / float64(inDim+outDim)))
	for i := range data {
		data[i] = ops.Mul(data[i], scale)
	}
	wTensor, err := tensor.New[T]([]int{inDim, outDim}, data)
	if err != nil {
		return nil, err
	}
	param, err := graph.NewParameter[T](name+"_weights", wTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}
	return &Linear[T]{
		name:    name,
		engine:  engine,
		ops:     ops,
		weights: param,
		inDim:   inDim,
		outDim:  outDim,
	}, nil
}

// Forward computes y = x @ W for the linear projection.
func (l *Linear[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Linear requires 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	lastDim := shape[len(shape)-1]
	if lastDim != l.inDim {
		return nil, fmt.Errorf("input last dim %d != expected %d", lastDim, l.inDim)
	}

	totalRows := 1
	for _, d := range shape[:len(shape)-1] {
		totalRows *= d
	}

	inData := input.Data()
	wData := l.weights.Value.Data()
	outData := make([]T, totalRows*l.outDim)

	for r := 0; r < totalRows; r++ {
		for o := 0; o < l.outDim; o++ {
			var sum T
			for i := 0; i < l.inDim; i++ {
				sum = l.ops.Add(sum, l.ops.Mul(inData[r*l.inDim+i], wData[i*l.outDim+o]))
			}
			outData[r*l.outDim+o] = sum
		}
	}

	outShape := make([]int, len(shape))
	copy(outShape, shape)
	outShape[len(outShape)-1] = l.outDim
	return tensor.New[T](outShape, outData)
}

// Backward computes gradients for the linear layer.
func (l *Linear[T]) Backward(_ context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	totalRows := 1
	for _, d := range shape[:len(shape)-1] {
		totalRows *= d
	}

	inData := input.Data()
	wData := l.weights.Value.Data()
	dOutData := dOut.Data()

	dInData := make([]T, totalRows*l.inDim)
	for r := 0; r < totalRows; r++ {
		for i := 0; i < l.inDim; i++ {
			var sum T
			for o := 0; o < l.outDim; o++ {
				sum = l.ops.Add(sum, l.ops.Mul(dOutData[r*l.outDim+o], wData[i*l.outDim+o]))
			}
			dInData[r*l.inDim+i] = sum
		}
	}

	dWData := make([]T, l.inDim*l.outDim)
	for i := 0; i < l.inDim; i++ {
		for o := 0; o < l.outDim; o++ {
			var sum T
			for r := 0; r < totalRows; r++ {
				sum = l.ops.Add(sum, l.ops.Mul(inData[r*l.inDim+i], dOutData[r*l.outDim+o]))
			}
			dWData[i*l.outDim+o] = sum
		}
	}

	dW, _ := tensor.New[T](l.weights.Value.Shape(), dWData)
	if l.weights.Gradient != nil {
		l.weights.Gradient, _ = l.engine.Add(context.Background(), l.weights.Gradient, dW)
	} else {
		l.weights.Gradient = dW
	}

	dIn, _ := tensor.New[T](shape, dInData)
	return []*tensor.TensorNumeric[T]{dIn}, nil
}

// Parameters returns the trainable parameters.
func (l *Linear[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{l.weights}
}

// ComplexSSMStateOption is a functional option for ComplexSSMState.
type ComplexSSMStateOption[T tensor.Numeric] func(*ComplexSSMState[T])

// WithComplexDiscretizationMode sets the discretization mode.
func WithComplexDiscretizationMode[T tensor.Numeric](mode DiscretizationMode) ComplexSSMStateOption[T] {
	return func(c *ComplexSSMState[T]) {
		c.discMode = mode
	}
}

// NewComplexSSMState creates a new ComplexSSMState block.
//
// dState must be even since dimensions are paired as (real, imaginary)
// for complex-valued RoPE rotation.
func NewComplexSSMState[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	dModel, dInner, dState, dtRank, convKer int,
	maxSeqLen int,
	opts ...ComplexSSMStateOption[T],
) (*ComplexSSMState[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if dModel <= 0 || dInner <= 0 || dState <= 0 || dtRank <= 0 || convKer <= 0 {
		return nil, fmt.Errorf("all dimensions must be positive")
	}
	if dState%2 != 0 {
		return nil, fmt.Errorf("dState must be even for complex pairing, got %d", dState)
	}
	if maxSeqLen <= 0 {
		return nil, fmt.Errorf("maxSeqLen must be positive")
	}

	inProj, err := NewLinear[T](name+"_in_proj", engine, ops, dModel, 2*dInner)
	if err != nil {
		return nil, fmt.Errorf("creating in_proj: %w", err)
	}

	xProj, err := NewLinear[T](name+"_x_proj", engine, ops, dInner, dtRank+2*dState)
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

	convData := randomData[T](dInner * convKer)
	convTensor, err := tensor.New[T]([]int{dInner, 1, convKer}, convData)
	if err != nil {
		return nil, err
	}
	convWeight, err := graph.NewParameter[T](name+"_conv_weight", convTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	aData := make([]T, dInner*dState)
	for i := 0; i < dInner; i++ {
		for j := 0; j < dState; j++ {
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

	ctx := context.Background()
	ropeB, err := embeddings.NewRotaryPositionalEmbedding[T](ctx, engine, dState, maxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("creating RoPE for B: %w", err)
	}
	ropeC, err := embeddings.NewRotaryPositionalEmbedding[T](ctx, engine, dState, maxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("creating RoPE for C: %w", err)
	}

	bcNormB, err := NewBCNorm[T](name+"_bc_norm_B", engine, ops, dState)
	if err != nil {
		return nil, fmt.Errorf("creating BCNorm for B: %w", err)
	}
	bcNormC, err := NewBCNorm[T](name+"_bc_norm_C", engine, ops, dState)
	if err != nil {
		return nil, fmt.Errorf("creating BCNorm for C: %w", err)
	}

	block := &ComplexSSMState[T]{
		name:       name,
		engine:     engine,
		ops:        ops,
		dModel:     dModel,
		dInner:     dInner,
		dState:     dState,
		dtRank:     dtRank,
		convKer:    convKer,
		discMode:   ExpTrap, // default to ExpTrap for Mamba 3
		ropeB:      ropeB,
		ropeC:      ropeC,
		bcNormB:    bcNormB,
		bcNormC:    bcNormC,
		inProj:     inProj,
		xProj:      xProj,
		dtProj:     dtProj,
		outProj:    outProj,
		convWeight: convWeight,
		A:          aParam,
		D:          dParam,
	}
	for _, opt := range opts {
		opt(block)
	}
	return block, nil
}

func (c *ComplexSSMState[T]) OpType() string { return "ComplexSSMState" }

func (c *ComplexSSMState[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"d_model":     c.dModel,
		"d_inner":     c.dInner,
		"d_state":     c.dState,
		"dt_rank":     c.dtRank,
		"kernel_size": c.convKer,
		"complex":     true,
	}
}

func (c *ComplexSSMState[T]) OutputShape() []int {
	return []int{-1, -1, c.dModel}
}

func (c *ComplexSSMState[T]) Name() string    { return c.name }
func (c *ComplexSSMState[T]) SetName(n string) { c.name = n }

// Forward computes the complex-valued SSM forward pass.
// Input: [batch, seq_len, d_model]
// Output: [batch, seq_len, d_model]
func (c *ComplexSSMState[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("ComplexSSMState requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("ComplexSSMState input must be 3D [batch, seq_len, d_model], got %v", shape)
	}

	batch := shape[0]
	seqLen := shape[1]
	c.lastInput = input

	// 1. Input projection: [batch, seq_len, d_model] -> [batch, seq_len, 2*d_inner]
	projected, err := c.inProj.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("in_proj forward: %w", err)
	}

	// 2. Split into x and z branches
	projData := projected.Data()
	xData := make([]T, batch*seqLen*c.dInner)
	zData := make([]T, batch*seqLen*c.dInner)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * (2 * c.dInner)
			copy(xData[(b*seqLen+s)*c.dInner:], projData[off:off+c.dInner])
			copy(zData[(b*seqLen+s)*c.dInner:], projData[off+c.dInner:off+2*c.dInner])
		}
	}
	xTensor, err := tensor.New[T]([]int{batch, seqLen, c.dInner}, xData)
	if err != nil {
		return nil, err
	}
	zTensor, err := tensor.New[T]([]int{batch, seqLen, c.dInner}, zData)
	if err != nil {
		return nil, err
	}
	c.cachedZ = zTensor
	c.cachedXPreConv = xTensor

	// 3. Causal depthwise Conv1D on x
	xConvData := make([]T, batch*seqLen*c.dInner)
	xFlatData := xTensor.Data()
	convW := c.convWeight.Value.Data()
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < c.dInner; d++ {
				var sum T
				for k := 0; k < c.convKer; k++ {
					srcPos := s - (c.convKer - 1) + k
					if srcPos >= 0 && srcPos < seqLen {
						xVal := xFlatData[(b*seqLen+srcPos)*c.dInner+d]
						wVal := convW[d*c.convKer+k]
						sum = c.ops.Add(sum, c.ops.Mul(xVal, wVal))
					}
				}
				xConvData[(b*seqLen+s)*c.dInner+d] = sum
			}
		}
	}
	xConv, err := tensor.New[T]([]int{batch, seqLen, c.dInner}, xConvData)
	if err != nil {
		return nil, err
	}
	c.cachedXConv = xConv

	// 4. SiLU on x
	xSilu, err := c.applySiLU(ctx, xConv)
	if err != nil {
		return nil, fmt.Errorf("silu on x: %w", err)
	}
	c.cachedX = xSilu

	// 5. SSM parameter projection: x -> [dt_rank, B, C]
	xBCDt, err := c.xProj.Forward(ctx, xSilu)
	if err != nil {
		return nil, fmt.Errorf("x_proj forward: %w", err)
	}
	c.cachedXBCDt = xBCDt

	xbcdtData := xBCDt.Data()
	dtRankData := make([]T, batch*seqLen*c.dtRank)
	bData := make([]T, batch*seqLen*c.dState)
	cDataSlice := make([]T, batch*seqLen*c.dState)
	projWidth := c.dtRank + 2*c.dState
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * projWidth
			copy(dtRankData[(b*seqLen+s)*c.dtRank:], xbcdtData[off:off+c.dtRank])
			copy(bData[(b*seqLen+s)*c.dState:], xbcdtData[off+c.dtRank:off+c.dtRank+c.dState])
			copy(cDataSlice[(b*seqLen+s)*c.dState:], xbcdtData[off+c.dtRank+c.dState:off+projWidth])
		}
	}
	dtRankTensor, err := tensor.New[T]([]int{batch, seqLen, c.dtRank}, dtRankData)
	if err != nil {
		return nil, err
	}

	B, err := tensor.New[T]([]int{batch, seqLen, c.dState}, bData)
	if err != nil {
		return nil, err
	}
	C, err := tensor.New[T]([]int{batch, seqLen, c.dState}, cDataSlice)
	if err != nil {
		return nil, err
	}

	// 5a. Apply BCNorm to B and C for stabilization
	B, err = c.bcNormB.Forward(ctx, B)
	if err != nil {
		return nil, fmt.Errorf("BCNorm B: %w", err)
	}
	C, err = c.bcNormC.Forward(ctx, C)
	if err != nil {
		return nil, fmt.Errorf("BCNorm C: %w", err)
	}

	// 5b. Apply RoPE to B and C (complex rotation in state space)
	B, err = c.ropeB.Forward(ctx, B)
	if err != nil {
		return nil, fmt.Errorf("RoPE B: %w", err)
	}
	C, err = c.ropeC.Forward(ctx, C)
	if err != nil {
		return nil, fmt.Errorf("RoPE C: %w", err)
	}
	c.cachedB = B
	c.cachedC = C

	// 6. dt projection
	dtRaw, err := c.dtProj.Forward(ctx, dtRankTensor)
	if err != nil {
		return nil, fmt.Errorf("dt_proj forward: %w", err)
	}
	c.cachedDtRaw = dtRaw

	// 7. Softplus on dt
	dt, err := c.applySoftplus(ctx, dtRaw)
	if err != nil {
		return nil, fmt.Errorf("softplus on dt: %w", err)
	}
	c.cachedDt = dt

	// 8. Selective scan (uses B and C that were rotated by RoPE)
	y, states, err := c.selectiveScan(ctx, xSilu, dt, B, C, batch, seqLen)
	if err != nil {
		return nil, fmt.Errorf("selective scan: %w", err)
	}
	c.cachedY = y
	c.cachedStates = states

	// 9. Gate: y * silu(z)
	siluZ, err := c.applySiLU(ctx, zTensor)
	if err != nil {
		return nil, fmt.Errorf("silu on z: %w", err)
	}
	c.cachedSiluZ = siluZ

	gated, err := c.engine.Mul(ctx, y, siluZ)
	if err != nil {
		return nil, fmt.Errorf("gating: %w", err)
	}

	// 10. Output projection
	output, err := c.outProj.Forward(ctx, gated)
	if err != nil {
		return nil, fmt.Errorf("out_proj forward: %w", err)
	}

	return output, nil
}

// selectiveScan runs the SSM recurrence with complex-rotated B/C.
func (c *ComplexSSMState[T]) selectiveScan(
	ctx context.Context,
	x, dt, B, C *tensor.TensorNumeric[T],
	batch, seqLen int,
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], error) {
	xData := x.Data()
	dtData := dt.Data()
	bDataSlice := B.Data()
	cDataSlice := C.Data()
	aData := c.A.Value.Data()
	dData := c.D.Value.Data()

	yData := make([]T, batch*seqLen*c.dInner)
	statesData := make([]T, batch*seqLen*c.dInner*c.dState)

	for b := 0; b < batch; b++ {
		h := make([]T, c.dInner*c.dState)

		for s := 0; s < seqLen; s++ {
			bsOff := b*seqLen + s

			for d := 0; d < c.dInner; d++ {
				xVal := xData[bsOff*c.dInner+d]
				dtVal := dtData[bsOff*c.dInner+d]

				var yVal T
				for n := 0; n < c.dState; n++ {
					aLog := aData[d*c.dState+n]
					aReal := T(-math.Exp(float64(aLog)))
					dA := T(math.Exp(float64(c.ops.Mul(dtVal, aReal))))

					bVal := bDataSlice[bsOff*c.dState+n]
					var dB T
					switch c.discMode {
					case ExpTrap:
						dB = c.ops.Mul(c.ops.Mul(dtVal, T((1.0+float64(dA))/2.0)), bVal)
					default:
						dB = c.ops.Mul(dtVal, bVal)
					}

					hIdx := d*c.dState + n
					h[hIdx] = c.ops.Add(c.ops.Mul(dA, h[hIdx]), c.ops.Mul(dB, xVal))

					cVal := cDataSlice[bsOff*c.dState+n]
					yVal = c.ops.Add(yVal, c.ops.Mul(cVal, h[hIdx]))
				}

				yVal = c.ops.Add(yVal, c.ops.Mul(dData[d], xVal))
				yData[bsOff*c.dInner+d] = yVal
			}

			stOff := (b*seqLen + s) * c.dInner * c.dState
			copy(statesData[stOff:stOff+c.dInner*c.dState], h)
		}
	}

	yTensor, err := tensor.New[T]([]int{batch, seqLen, c.dInner}, yData)
	if err != nil {
		return nil, nil, err
	}
	statesTensor, err := tensor.New[T]([]int{batch, seqLen, c.dInner, c.dState}, statesData)
	if err != nil {
		return nil, nil, err
	}

	return yTensor, statesTensor, nil
}

func (c *ComplexSSMState[T]) applySiLU(_ context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	out := make([]T, len(data))
	for i, v := range data {
		sig := T(1.0 / (1.0 + math.Exp(-float64(v))))
		out[i] = c.ops.Mul(v, sig)
	}
	return tensor.New[T](x.Shape(), out)
}

func (c *ComplexSSMState[T]) applySoftplus(_ context.Context, x *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	out := make([]T, len(data))
	for i, v := range data {
		out[i] = T(math.Log(1.0 + math.Exp(float64(v))))
	}
	return tensor.New[T](x.Shape(), out)
}

// Backward computes gradients for the ComplexSSMState block.
func (c *ComplexSSMState[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("ComplexSSMState requires exactly 1 input for backward, got %d", len(inputs))
	}

	shape := c.lastInput.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// 10. Backward through outProj
	gated, err := c.engine.Mul(ctx, c.cachedY, c.cachedSiluZ)
	if err != nil {
		return nil, err
	}
	dGated, err := c.outProj.Backward(ctx, mode, outputGradient, gated)
	if err != nil {
		return nil, err
	}
	dGatedTensor := dGated[0]

	// 9. Backward through gate
	dY, err := c.engine.Mul(ctx, dGatedTensor, c.cachedSiluZ)
	if err != nil {
		return nil, err
	}
	dSiluZ, err := c.engine.Mul(ctx, dGatedTensor, c.cachedY)
	if err != nil {
		return nil, err
	}
	dZ := c.siluBackward(c.cachedZ, dSiluZ)

	// 8. Backward through selective scan
	dX_scan, dDt, dB_ssm, dC_ssm := c.selectiveScanBackward(batch, seqLen, dY)

	// 5b. Backward through RoPE on B and C
	dBRope, err := c.ropeB.Backward(ctx, mode, dB_ssm)
	if err != nil {
		return nil, fmt.Errorf("RoPE B backward: %w", err)
	}
	dCRope, err := c.ropeC.Backward(ctx, mode, dC_ssm)
	if err != nil {
		return nil, fmt.Errorf("RoPE C backward: %w", err)
	}

	// 5a. Backward through BCNorm
	dBNorm, err := c.bcNormB.Backward(ctx, dBRope[0])
	if err != nil {
		return nil, fmt.Errorf("BCNorm B backward: %w", err)
	}
	dCNorm, err := c.bcNormC.Backward(ctx, dCRope[0])
	if err != nil {
		return nil, fmt.Errorf("BCNorm C backward: %w", err)
	}

	// 7. Backward through softplus
	dDtRaw := c.softplusBackward(c.cachedDtRaw, dDt)

	// 6. Backward through dtProj
	dtRankData := make([]T, batch*seqLen*c.dtRank)
	xbcdtData := c.cachedXBCDt.Data()
	projWidth := c.dtRank + 2*c.dState
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * projWidth
			copy(dtRankData[(b*seqLen+s)*c.dtRank:], xbcdtData[off:off+c.dtRank])
		}
	}
	dtRankTensor, err := tensor.New[T]([]int{batch, seqLen, c.dtRank}, dtRankData)
	if err != nil {
		return nil, err
	}
	dDtRank, err := c.dtProj.Backward(ctx, mode, dDtRaw, dtRankTensor)
	if err != nil {
		return nil, err
	}

	// 5. Backward through xProj
	dXBCDtData := make([]T, batch*seqLen*projWidth)
	dDtRankData := dDtRank[0].Data()
	dBData := dBNorm.Data()
	dCData := dCNorm.Data()
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * projWidth
			copy(dXBCDtData[off:off+c.dtRank], dDtRankData[(b*seqLen+s)*c.dtRank:])
			copy(dXBCDtData[off+c.dtRank:off+c.dtRank+c.dState], dBData[(b*seqLen+s)*c.dState:])
			copy(dXBCDtData[off+c.dtRank+c.dState:off+projWidth], dCData[(b*seqLen+s)*c.dState:])
		}
	}
	dXBCDt, err := tensor.New[T]([]int{batch, seqLen, projWidth}, dXBCDtData)
	if err != nil {
		return nil, err
	}

	dXSilu, err := c.xProj.Backward(ctx, mode, dXBCDt, c.cachedX)
	if err != nil {
		return nil, err
	}

	dXTotal, err := c.engine.Add(ctx, dXSilu[0], dX_scan)
	if err != nil {
		return nil, err
	}

	// 4. Backward through SiLU
	dXConv := c.siluBackward(c.cachedXConv, dXTotal)

	// 3. Backward through conv1d
	dXPreConv, dConvW := c.conv1dBackward(batch, seqLen, dXConv)
	c.convWeight.Gradient, err = c.engine.Add(ctx, c.convWeight.Gradient, dConvW, c.convWeight.Gradient)
	if err != nil {
		return nil, err
	}

	// 2. Reassemble
	dXPreConvData := dXPreConv.Data()
	dZData := dZ.Data()
	dProjData := make([]T, batch*seqLen*2*c.dInner)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * (2 * c.dInner)
			copy(dProjData[off:off+c.dInner], dXPreConvData[(b*seqLen+s)*c.dInner:])
			copy(dProjData[off+c.dInner:off+2*c.dInner], dZData[(b*seqLen+s)*c.dInner:])
		}
	}
	dProj, err := tensor.New[T]([]int{batch, seqLen, 2 * c.dInner}, dProjData)
	if err != nil {
		return nil, err
	}

	// 1. Backward through inProj
	dInput, err := c.inProj.Backward(ctx, mode, dProj, c.lastInput)
	if err != nil {
		return nil, err
	}

	return dInput, nil
}

func (c *ComplexSSMState[T]) selectiveScanBackward(
	batch, seqLen int,
	dY *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], *tensor.TensorNumeric[T], *tensor.TensorNumeric[T]) {
	dYData := dY.Data()
	xData := c.cachedX.Data()
	dtData := c.cachedDt.Data()
	bDataSlice := c.cachedB.Data()
	cDataSlice := c.cachedC.Data()
	aData := c.A.Value.Data()
	dParamData := c.D.Value.Data()
	statesData := c.cachedStates.Data()

	dXData := make([]T, batch*seqLen*c.dInner)
	dDtData := make([]T, batch*seqLen*c.dInner)
	dBData := make([]T, batch*seqLen*c.dState)
	dCData := make([]T, batch*seqLen*c.dState)

	dAData := make([]T, c.dInner*c.dState)
	dDData := make([]T, c.dInner)

	for b := 0; b < batch; b++ {
		dh := make([]T, c.dInner*c.dState)

		for s := seqLen - 1; s >= 0; s-- {
			bsOff := b*seqLen + s

			for d := 0; d < c.dInner; d++ {
				dyVal := dYData[bsOff*c.dInner+d]
				xVal := xData[bsOff*c.dInner+d]
				dtVal := dtData[bsOff*c.dInner+d]

				dDData[d] = c.ops.Add(dDData[d], c.ops.Mul(dyVal, xVal))
				dXData[bsOff*c.dInner+d] = c.ops.Add(dXData[bsOff*c.dInner+d], c.ops.Mul(dyVal, dParamData[d]))

				for n := 0; n < c.dState; n++ {
					hIdx := d*c.dState + n
					cVal := cDataSlice[bsOff*c.dState+n]
					hVal := statesData[bsOff*c.dInner*c.dState+hIdx]

					dCData[bsOff*c.dState+n] = c.ops.Add(dCData[bsOff*c.dState+n], c.ops.Mul(dyVal, hVal))
					dh[hIdx] = c.ops.Add(dh[hIdx], c.ops.Mul(dyVal, cVal))

					aLog := aData[d*c.dState+n]
					aReal := T(-math.Exp(float64(aLog)))
					dAVal := T(math.Exp(float64(c.ops.Mul(dtVal, aReal))))
					bVal := bDataSlice[bsOff*c.dState+n]

					var hPrev T
					if s > 0 {
						prevOff := (b*seqLen + s - 1) * c.dInner * c.dState
						hPrev = statesData[prevOff+hIdx]
					}

					ddA_ddt := c.ops.Mul(dAVal, aReal)
					ddB_ddt := bVal

					dDtData[bsOff*c.dInner+d] = c.ops.Add(dDtData[bsOff*c.dInner+d],
						c.ops.Mul(dh[hIdx], c.ops.Add(
							c.ops.Mul(ddA_ddt, hPrev),
							c.ops.Mul(ddB_ddt, xVal),
						)),
					)

					dBData[bsOff*c.dState+n] = c.ops.Add(dBData[bsOff*c.dState+n],
						c.ops.Mul(dh[hIdx], c.ops.Mul(dtVal, xVal)),
					)

					dXData[bsOff*c.dInner+d] = c.ops.Add(dXData[bsOff*c.dInner+d],
						c.ops.Mul(dh[hIdx], c.ops.Mul(dtVal, bVal)),
					)

					dAData[d*c.dState+n] = c.ops.Add(dAData[d*c.dState+n],
						c.ops.Mul(dh[hIdx], c.ops.Mul(hPrev, c.ops.Mul(dAVal, c.ops.Mul(dtVal, c.ops.Mul(aReal, T(-math.Exp(float64(aLog)))))))),
					)

					dh[hIdx] = c.ops.Mul(dh[hIdx], dAVal)
				}
			}
		}
	}

	dATensor, _ := tensor.New[T](c.A.Value.Shape(), dAData)
	if c.A.Gradient != nil {
		c.A.Gradient, _ = c.engine.Add(context.Background(), c.A.Gradient, dATensor)
	} else {
		c.A.Gradient = dATensor
	}

	dDTensor, _ := tensor.New[T](c.D.Value.Shape(), dDData)
	if c.D.Gradient != nil {
		c.D.Gradient, _ = c.engine.Add(context.Background(), c.D.Gradient, dDTensor)
	} else {
		c.D.Gradient = dDTensor
	}

	dX, _ := tensor.New[T]([]int{batch, seqLen, c.dInner}, dXData)
	dDt, _ := tensor.New[T]([]int{batch, seqLen, c.dInner}, dDtData)
	dBTensor, _ := tensor.New[T]([]int{batch, seqLen, c.dState}, dBData)
	dCTensor, _ := tensor.New[T]([]int{batch, seqLen, c.dState}, dCData)

	return dX, dDt, dBTensor, dCTensor
}

func (c *ComplexSSMState[T]) siluBackward(input, dOutput *tensor.TensorNumeric[T]) *tensor.TensorNumeric[T] {
	inData := input.Data()
	dOutData := dOutput.Data()
	result := make([]T, len(inData))
	for i, x := range inData {
		sig := T(1.0 / (1.0 + math.Exp(-float64(x))))
		grad := c.ops.Mul(sig, T(1.0+float64(x)*(1.0-float64(sig))))
		result[i] = c.ops.Mul(dOutData[i], grad)
	}
	t, _ := tensor.New[T](input.Shape(), result)
	return t
}

func (c *ComplexSSMState[T]) softplusBackward(input, dOutput *tensor.TensorNumeric[T]) *tensor.TensorNumeric[T] {
	inData := input.Data()
	dOutData := dOutput.Data()
	result := make([]T, len(inData))
	for i, x := range inData {
		sig := T(1.0 / (1.0 + math.Exp(-float64(x))))
		result[i] = c.ops.Mul(dOutData[i], sig)
	}
	t, _ := tensor.New[T](input.Shape(), result)
	return t
}

func (c *ComplexSSMState[T]) conv1dBackward(
	batch, seqLen int,
	dOutput *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T]) {
	dOutData := dOutput.Data()
	xData := c.cachedXPreConv.Data()
	convW := c.convWeight.Value.Data()

	dXData := make([]T, batch*seqLen*c.dInner)
	dWData := make([]T, c.dInner*c.convKer)

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < c.dInner; d++ {
				dOut := dOutData[(b*seqLen+s)*c.dInner+d]
				for k := 0; k < c.convKer; k++ {
					srcPos := s - (c.convKer - 1) + k
					if srcPos >= 0 && srcPos < seqLen {
						dXData[(b*seqLen+srcPos)*c.dInner+d] = c.ops.Add(
							dXData[(b*seqLen+srcPos)*c.dInner+d],
							c.ops.Mul(dOut, convW[d*c.convKer+k]),
						)
						dWData[d*c.convKer+k] = c.ops.Add(
							dWData[d*c.convKer+k],
							c.ops.Mul(dOut, xData[(b*seqLen+srcPos)*c.dInner+d]),
						)
					}
				}
			}
		}
	}

	dX, _ := tensor.New[T]([]int{batch, seqLen, c.dInner}, dXData)
	dW, _ := tensor.New[T](c.convWeight.Value.Shape(), dWData)
	return dX, dW
}

// Parameters returns all trainable parameters.
func (c *ComplexSSMState[T]) Parameters() []*graph.Parameter[T] {
	params := c.inProj.Parameters()
	params = append(params, c.convWeight)
	params = append(params, c.xProj.Parameters()...)
	params = append(params, c.dtProj.Parameters()...)
	params = append(params, c.A)
	params = append(params, c.D)
	params = append(params, c.bcNormB.Parameters()...)
	params = append(params, c.bcNormC.Parameters()...)
	params = append(params, c.outProj.Parameters()...)
	return params
}

var _ graph.Node[float32] = (*ComplexSSMState[float32])(nil)
