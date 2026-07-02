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

// SSMLayer implements a State Space Model layer with diagonal state matrix.
//
// The continuous-time system is:
//
//	x'(t) = A*x(t) + B*u(t)   (state equation)
//	y(t)  = C*x(t) + D*u(t)   (output equation)
//
// For efficient computation, the layer uses a discretized recurrence with
// Zero-Order Hold (ZOH). Because A is diagonal (stored as a 1-D vector),
// all matrix operations reduce to element-wise products:
//
//	A_bar = exp(A * dt)
//	B_bar = (A_bar - 1) / A * B
//	x[k+1] = A_bar * x[k] + B_bar * u[k]
//	y[k]   = C * x[k] + D * u[k]
//
// This is the S4D/S5-style parameterisation used by IBM Granite FlowState
// for time-series forecasting.
type SSMLayer[T tensor.Float] struct {
	engine compute.Engine[T]

	// Learnable parameters.
	// A is stored in log-space so that the actual diagonal is -exp(A),
	// guaranteeing stability (negative real eigenvalues).
	A  *graph.Parameter[T] // [d_state]
	B  *graph.Parameter[T] // [d_state, d_input]
	C  *graph.Parameter[T] // [d_output, d_state]
	D  *graph.Parameter[T] // [d_output, d_input] feedthrough (skip connection)
	Dt *graph.Parameter[T] // [1] discretisation step size (log-space)

	dState  int
	dInput  int
	dOutput int
}

// NewSSMLayer creates a new State Space Model layer with diagonal state matrix.
//
// Parameters:
//   - engine: the compute engine for tensor operations
//   - dState: dimensionality of the hidden state
//   - dInput: dimensionality of the input at each time step
//   - dOutput: dimensionality of the output at each time step
func NewSSMLayer[T tensor.Float](engine compute.Engine[T], dState, dInput, dOutput int) (*SSMLayer[T], error) {
	if dState <= 0 {
		return nil, fmt.Errorf("dState must be positive, got %d", dState)
	}
	if dInput <= 0 {
		return nil, fmt.Errorf("dInput must be positive, got %d", dInput)
	}
	if dOutput <= 0 {
		return nil, fmt.Errorf("dOutput must be positive, got %d", dOutput)
	}

	// A: initialise in log-space so that -exp(A) gives a stable diagonal.
	// Use uniform random in [-1, 1] which maps to eigenvalues in [-e, -1/e].
	aData := make([]T, dState)
	for i := range aData {
		aData[i] = T(rand.Float64()*2 - 1)
	}
	aTensor, err := tensor.New[T]([]int{dState}, aData)
	if err != nil {
		return nil, fmt.Errorf("create A tensor: %w", err)
	}
	aParam, err := graph.NewParameter[T]("ssm_A", aTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("create A parameter: %w", err)
	}

	// B: [d_state, d_input] with Kaiming-style init.
	scale := 1.0 / math.Sqrt(float64(dInput))
	bData := make([]T, dState*dInput)
	for i := range bData {
		bData[i] = T(rand.Float64() * scale)
	}
	bTensor, err := tensor.New[T]([]int{dState, dInput}, bData)
	if err != nil {
		return nil, fmt.Errorf("create B tensor: %w", err)
	}
	bParam, err := graph.NewParameter[T]("ssm_B", bTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("create B parameter: %w", err)
	}

	// C: [d_output, d_state] with Kaiming-style init.
	scaleC := 1.0 / math.Sqrt(float64(dState))
	cData := make([]T, dOutput*dState)
	for i := range cData {
		cData[i] = T(rand.Float64() * scaleC)
	}
	cTensor, err := tensor.New[T]([]int{dOutput, dState}, cData)
	if err != nil {
		return nil, fmt.Errorf("create C tensor: %w", err)
	}
	cParam, err := graph.NewParameter[T]("ssm_C", cTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("create C parameter: %w", err)
	}

	// D: [d_output, d_input] feedthrough, initialised near zero.
	dData := make([]T, dOutput*dInput)
	for i := range dData {
		dData[i] = T(rand.Float64() * 0.01)
	}
	dTensor, err := tensor.New[T]([]int{dOutput, dInput}, dData)
	if err != nil {
		return nil, fmt.Errorf("create D tensor: %w", err)
	}
	dParam, err := graph.NewParameter[T]("ssm_D", dTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("create D parameter: %w", err)
	}

	// dt: [1] discretisation step, stored in log-space. Init to log(0.01).
	dtData := []T{T(math.Log(0.01))}
	dtTensor, err := tensor.New[T]([]int{1}, dtData)
	if err != nil {
		return nil, fmt.Errorf("create dt tensor: %w", err)
	}
	dtParam, err := graph.NewParameter[T]("ssm_dt", dtTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("create dt parameter: %w", err)
	}

	return &SSMLayer[T]{
		engine:  engine,
		A:       aParam,
		B:       bParam,
		C:       cParam,
		D:       dParam,
		Dt:      dtParam,
		dState:  dState,
		dInput:  dInput,
		dOutput: dOutput,
	}, nil
}

// OpType returns the operation type of the layer.
func (s *SSMLayer[T]) OpType() string {
	return "SSM"
}

// Attributes returns the attributes of the layer.
func (s *SSMLayer[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"d_state":  s.dState,
		"d_input":  s.dInput,
		"d_output": s.dOutput,
	}
}

// OutputShape returns the output shape of the layer.
func (s *SSMLayer[T]) OutputShape() []int {
	return []int{-1, -1, s.dOutput} // [batch, seq_len, d_output]
}

// Forward processes an input sequence through the SSM.
//
// Input shape:  [batch, seq_len, d_input]
// Output shape: [batch, seq_len, d_output]
//
// The method performs a sequential scan (one time step at a time).
// A parallel scan variant can be added later for GPU optimisation.
func (s *SSMLayer[T]) Forward(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("SSMLayer input must be 3D [batch, seq_len, d_input], got shape %v", shape)
	}
	batch, seqLen, dIn := shape[0], shape[1], shape[2]
	if dIn != s.dInput {
		return nil, fmt.Errorf("SSMLayer input d_input = %d, want %d", dIn, s.dInput)
	}

	// --- Discretise using engine ops ---
	eng := s.engine
	ops := eng.Ops()

	// dt = exp(log_dt)
	dtTensor, err := eng.Exp(ctx, s.Dt.Value) // [1]
	if err != nil {
		return nil, fmt.Errorf("SSM: exp(log_dt): %w", err)
	}

	// A_diag = -exp(A_log) — stable negative eigenvalues
	expA, err := eng.Exp(ctx, s.A.Value) // [dState]
	if err != nil {
		return nil, fmt.Errorf("SSM: exp(A): %w", err)
	}
	aDiag, err := eng.MulScalar(ctx, expA, ops.FromFloat64(-1)) // [dState]
	if err != nil {
		return nil, fmt.Errorf("SSM: negate exp(A): %w", err)
	}

	// A_bar = exp(A_diag * dt)
	dtScalar := dtTensor.Data()[0] // scalar extraction from 1-element tensor
	aDt, err := eng.MulScalar(ctx, aDiag, dtScalar)              // [dState]
	if err != nil {
		return nil, fmt.Errorf("SSM: aDiag*dt: %w", err)
	}
	aBarFlat, err := eng.Exp(ctx, aDt) // [dState]
	if err != nil {
		return nil, fmt.Errorf("SSM: exp(aDiag*dt): %w", err)
	}
	aBarTensor, err := eng.Reshape(ctx, aBarFlat, []int{s.dState, 1}) // [dState, 1]
	if err != nil {
		return nil, fmt.Errorf("SSM: reshape aBar: %w", err)
	}

	// B_bar = (A_bar - 1) / A_diag * B, with safe division for near-zero A_diag
	aBarMinus1, err := eng.AddScalar(ctx, aBarFlat, ops.FromFloat64(-1)) // [dState]
	if err != nil {
		return nil, fmt.Errorf("SSM: aBar-1: %w", err)
	}
	// Safe divide: clamp |aDiag| away from zero, fallback to dt for near-zero values
	aDiagSafe, err := eng.UnaryOp(ctx, aDiag, func(v T) T {
		if math.Abs(float64(v)) < 1e-12 {
			return ops.FromFloat64(-1e-12) // placeholder; scale will be overridden
		}
		return v
	})
	if err != nil {
		return nil, fmt.Errorf("SSM: safe aDiag: %w", err)
	}
	scaleVec, err := eng.Div(ctx, aBarMinus1, aDiagSafe) // [dState]
	if err != nil {
		return nil, fmt.Errorf("SSM: (aBar-1)/aDiag: %w", err)
	}
	// For near-zero aDiag elements, override scale with dt
	scaleVec, err = eng.UnaryOp(ctx, scaleVec, func(v T) T {
		if math.Abs(float64(v)) > 1e6 { // overflow guard from near-zero division
			return dtScalar
		}
		return v
	})
	if err != nil {
		return nil, fmt.Errorf("SSM: scale overflow guard: %w", err)
	}
	// B_bar = scale * B: broadcast [dState, 1] * [dState, dInput]
	scaleCol, err := eng.Reshape(ctx, scaleVec, []int{s.dState, 1})
	if err != nil {
		return nil, fmt.Errorf("SSM: reshape scale: %w", err)
	}
	bBarTensor, err := eng.Mul(ctx, scaleCol, s.B.Value) // [dState, dInput]
	if err != nil {
		return nil, fmt.Errorf("SSM: scale*B: %w", err)
	}

	inputData := input.Data()

	// Allocate output buffer: [batch, seq_len, d_output]
	outputData := make([]T, batch*seqLen*s.dOutput)

	// --- Sequential scan per batch element ---
	for b := 0; b < batch; b++ {
		// Hidden state x: [d_state, 1], initialised to zero.
		xTensor, err := tensor.New[T]([]int{s.dState, 1}, make([]T, s.dState))
		if err != nil {
			return nil, fmt.Errorf("create x tensor: %w", err)
		}

		for t := 0; t < seqLen; t++ {
			// u = input[b, t, :] as [d_input, 1]
			uOffset := b*seqLen*s.dInput + t*s.dInput
			uData := make([]T, s.dInput)
			copy(uData, inputData[uOffset:uOffset+s.dInput])
			uTensor, err := tensor.New[T]([]int{s.dInput, 1}, uData)
			if err != nil {
				return nil, fmt.Errorf("create u tensor: %w", err)
			}

			// B_bar * u → [d_state, 1]
			bBarU, err := s.engine.MatMul(ctx, bBarTensor, uTensor)
			if err != nil {
				return nil, fmt.Errorf("ssm B_bar*u MatMul: %w", err)
			}

			// A_bar ⊙ x (element-wise, diagonal A) → [d_state, 1]
			aBarX, err := s.engine.Mul(ctx, aBarTensor, xTensor)
			if err != nil {
				return nil, fmt.Errorf("ssm A_bar*x Mul: %w", err)
			}

			// x = A_bar⊙x + B_bar*u → [d_state, 1]
			xTensor, err = s.engine.Add(ctx, aBarX, bBarU)
			if err != nil {
				return nil, fmt.Errorf("ssm state update Add: %w", err)
			}

			// C * x → [d_output, 1]
			cx, err := s.engine.MatMul(ctx, s.C.Value, xTensor)
			if err != nil {
				return nil, fmt.Errorf("ssm C*x MatMul: %w", err)
			}

			// D * u → [d_output, 1]
			du, err := s.engine.MatMul(ctx, s.D.Value, uTensor)
			if err != nil {
				return nil, fmt.Errorf("ssm D*u MatMul: %w", err)
			}

			// y = C*x + D*u → [d_output, 1]
			y, err := s.engine.Add(ctx, cx, du)
			if err != nil {
				return nil, fmt.Errorf("ssm output Add: %w", err)
			}

			// Copy y into output buffer.
			yOffset := b*seqLen*s.dOutput + t*s.dOutput
			copy(outputData[yOffset:yOffset+s.dOutput], y.Data())
		}
	}

	return tensor.New[T]([]int{batch, seqLen, s.dOutput}, outputData)
}

// Parameters returns all learnable parameters of the SSM layer.
func (s *SSMLayer[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{s.A, s.B, s.C, s.D, s.Dt}
}
