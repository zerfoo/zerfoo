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

	// --- Discretise ---
	// dt = exp(log_dt)  (scalar)
	logDt := float64(s.Dt.Value.Data()[0])
	dt := math.Exp(logDt)

	// A_diag = -exp(A_log)  (stable negative eigenvalues)
	aLog := s.A.Value.Data()
	aDiag := make([]float64, s.dState)
	for i, v := range aLog {
		aDiag[i] = -math.Exp(float64(v))
	}

	// A_bar[i] = exp(A_diag[i] * dt)
	aBar := make([]float64, s.dState)
	for i := range aDiag {
		aBar[i] = math.Exp(aDiag[i] * dt)
	}

	// B_bar[i,j] = (A_bar[i] - 1) / A_diag[i] * B[i,j]
	// When A_diag[i] is very close to 0, use the limit: B_bar = dt * B.
	bData := s.B.Value.Data()
	bBar := make([]float64, s.dState*s.dInput)
	for i := 0; i < s.dState; i++ {
		var scale float64
		if math.Abs(aDiag[i]) < 1e-12 {
			scale = dt
		} else {
			scale = (aBar[i] - 1.0) / aDiag[i]
		}
		for j := 0; j < s.dInput; j++ {
			bBar[i*s.dInput+j] = scale * float64(bData[i*s.dInput+j])
		}
	}

	// C: [d_output, d_state]
	cData := s.C.Value.Data()
	// D: [d_output, d_input]
	dData := s.D.Value.Data()

	inputData := input.Data()

	// Allocate output buffer: [batch, seq_len, d_output]
	outputData := make([]T, batch*seqLen*s.dOutput)

	// --- Sequential scan per batch element ---
	for b := 0; b < batch; b++ {
		// Hidden state x: [d_state], initialised to zero.
		x := make([]float64, s.dState)

		for t := 0; t < seqLen; t++ {
			// u = input[b, t, :] — the input vector at this time step.
			uOffset := b*seqLen*s.dInput + t*s.dInput

			// x_new[i] = A_bar[i] * x[i] + sum_j(B_bar[i,j] * u[j])
			for i := 0; i < s.dState; i++ {
				val := aBar[i] * x[i]
				for j := 0; j < s.dInput; j++ {
					val += bBar[i*s.dInput+j] * float64(inputData[uOffset+j])
				}
				x[i] = val
			}

			// y[k] = C * x + D * u
			// y[k] is [d_output]
			yOffset := b*seqLen*s.dOutput + t*s.dOutput
			for o := 0; o < s.dOutput; o++ {
				var val float64
				// C[o, :] dot x
				for i := 0; i < s.dState; i++ {
					val += float64(cData[o*s.dState+i]) * x[i]
				}
				// D[o, :] dot u
				for j := 0; j < s.dInput; j++ {
					val += float64(dData[o*s.dInput+j]) * float64(inputData[uOffset+j])
				}
				outputData[yOffset+o] = T(val)
			}
		}
	}

	return tensor.New[T]([]int{batch, seqLen, s.dOutput}, outputData)
}

// Parameters returns all learnable parameters of the SSM layer.
func (s *SSMLayer[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{s.A, s.B, s.C, s.D, s.Dt}
}
