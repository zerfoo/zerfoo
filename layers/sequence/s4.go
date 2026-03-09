// Package sequence provides sequence modeling layers such as State Space Models.
package sequence

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// S4 implements a diagonal State Space Model (S4D variant).
//
// The continuous-time state space model is:
//
//	x'(t) = A x(t) + B u(t)
//	y(t)  = C x(t) + D u(t)
//
// With diagonal A, the discrete-time equations become element-wise:
//
//	x_k = a * x_{k-1} + b * u_k
//	y_k = sum(c * x_k) + d * u_k
//
// where a = exp(dt * A_diag) ensures stability when A_diag < 0.
//
// Input shape:  [batch, seq_len, input_dim]
// Output shape: [batch, seq_len, input_dim]
//
// Parameters (per input dimension, state_dim internal states):
//
//	A_log [input_dim, state_dim] - log(-A), parameterizing stable eigenvalues
//	B     [input_dim, state_dim] - input-to-state projection
//	C     [input_dim, state_dim] - state-to-output projection
//	D     [input_dim]            - skip connection
type S4[T tensor.Float] struct {
	name     string
	engine   compute.Engine[T]
	ops      numeric.Arithmetic[T]
	aLog     *graph.Parameter[T] // [input_dim, state_dim]
	b        *graph.Parameter[T] // [input_dim, state_dim]
	c        *graph.Parameter[T] // [input_dim, state_dim]
	d        *graph.Parameter[T] // [input_dim]
	inputDim int
	stateDim int
	// Saved for backward pass.
	lastInput  *tensor.TensorNumeric[T]
	lastStates *tensor.TensorNumeric[T] // [batch, input_dim, state_dim] final state
}

// NewS4 creates a new S4 layer with HiPPO-inspired initialization.
func NewS4[T tensor.Float](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputDim, stateDim int,
) (*S4[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if inputDim <= 0 {
		return nil, fmt.Errorf("input_dim must be positive, got %d", inputDim)
	}
	if stateDim <= 0 {
		return nil, fmt.Errorf("state_dim must be positive, got %d", stateDim)
	}

	// Initialize A_log with HiPPO-inspired values: a_n = -(n+1),
	// so A_log[d][n] = log(n+1) to parameterize A = -exp(A_log).
	aLogData := make([]T, inputDim*stateDim)
	for n := range stateDim {
		val := T(math.Log(float64(n + 1)))
		for d := range inputDim {
			aLogData[d*stateDim+n] = val
		}
	}

	// Initialize B, C with small random values; D with ones.
	bData := make([]T, inputDim*stateDim)
	cData := make([]T, inputDim*stateDim)
	scale := T(1.0 / math.Sqrt(float64(stateDim)))
	for i := range bData {
		bData[i] = T(rand.Float64()-0.5) * scale
		cData[i] = T(rand.Float64()-0.5) * scale
	}
	dData := make([]T, inputDim)
	for i := range dData {
		dData[i] = 1
	}

	aLogTensor, err := tensor.New[T]([]int{inputDim, stateDim}, aLogData)
	if err != nil {
		return nil, fmt.Errorf("create a_log tensor: %w", err)
	}
	bTensor, err := tensor.New[T]([]int{inputDim, stateDim}, bData)
	if err != nil {
		return nil, fmt.Errorf("create b tensor: %w", err)
	}
	cTensor, err := tensor.New[T]([]int{inputDim, stateDim}, cData)
	if err != nil {
		return nil, fmt.Errorf("create c tensor: %w", err)
	}
	dTensor, err := tensor.New[T]([]int{inputDim}, dData)
	if err != nil {
		return nil, fmt.Errorf("create d tensor: %w", err)
	}

	aLogParam, err := graph.NewParameter(name+"_a_log", aLogTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}
	bParam, err := graph.NewParameter(name+"_b", bTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}
	cParam, err := graph.NewParameter(name+"_c", cTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}
	dParam, err := graph.NewParameter(name+"_d", dTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	return &S4[T]{
		name:     name,
		engine:   engine,
		ops:      ops,
		aLog:     aLogParam,
		b:        bParam,
		c:        cParam,
		d:        dParam,
		inputDim: inputDim,
		stateDim: stateDim,
	}, nil
}

// OpType returns the operation type.
func (s *S4[T]) OpType() string { return "S4" }

// Attributes returns the layer attributes.
func (s *S4[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_dim": s.inputDim,
		"state_dim": s.stateDim,
	}
}

// OutputShape returns the output shape.
func (s *S4[T]) OutputShape() []int {
	return []int{-1, -1, s.inputDim}
}

// Parameters returns all trainable parameters.
func (s *S4[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{s.aLog, s.b, s.c, s.d}
}

// Forward runs the diagonal SSM scan over the sequence.
//
// Input: [batch, seq_len, input_dim]
// Output: [batch, seq_len, input_dim]
//
// All arithmetic is routed through engine primitives so the computation
// graph is fully traceable by the tracing compiler.
func (s *S4[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("S4 requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("S4 input must be 3D [batch, seq, dim], got %dD", len(shape))
	}

	batchSize := shape[0]
	seqLen := shape[1]
	dim := shape[2]

	s.lastInput = input

	// Compute discrete A = exp(-exp(a_log)) element-wise via engine primitives.
	// A_disc = exp(-exp(a_log))
	expALog, err := s.engine.Exp(ctx, s.aLog.Value) // exp(a_log)  [dim, state_dim]
	if err != nil {
		return nil, fmt.Errorf("s4 exp(a_log): %w", err)
	}
	negExpALog, err := s.engine.MulScalar(ctx, expALog, T(-1)) // -exp(a_log)
	if err != nil {
		return nil, fmt.Errorf("s4 neg exp(a_log): %w", err)
	}
	aDisc, err := s.engine.Exp(ctx, negExpALog) // exp(-exp(a_log))  [dim, state_dim]
	if err != nil {
		return nil, fmt.Errorf("s4 exp(-exp(a_log)): %w", err)
	}

	// Initialize state to zeros: [batch, dim, state_dim].
	stateData := make([]T, batchSize*dim*s.stateDim)
	state, err := tensor.New[T]([]int{batchSize, dim, s.stateDim}, stateData)
	if err != nil {
		return nil, fmt.Errorf("s4 init state: %w", err)
	}

	// Collect per-step outputs for concatenation.
	stepOutputs := make([]*tensor.TensorNumeric[T], seqLen)

	// Sequential scan over time steps.
	for t := range seqLen {
		// Extract u_t: input[:, t, :] -> [batch, dim].
		// Build a contiguous tensor from the input slice at timestep t.
		uData := make([]T, batchSize*dim)
		inData := input.Data()
		for b := range batchSize {
			copy(uData[b*dim:(b+1)*dim], inData[b*seqLen*dim+t*dim:b*seqLen*dim+t*dim+dim])
		}
		ut, err := tensor.New[T]([]int{batchSize, dim}, uData)
		if err != nil {
			return nil, fmt.Errorf("s4 extract u_t: %w", err)
		}

		// Expand u_t to [batch, dim, 1] for broadcasting with [dim, state_dim].
		utExp, err := s.engine.Reshape(ctx, ut, []int{batchSize, dim, 1})
		if err != nil {
			return nil, fmt.Errorf("s4 reshape u_t: %w", err)
		}

		// state = aDisc * state + b * u_t
		// aDisc is [dim, state_dim], state is [batch, dim, state_dim] -> broadcasts.
		aState, err := s.engine.Mul(ctx, aDisc, state)
		if err != nil {
			return nil, fmt.Errorf("s4 aDisc*state: %w", err)
		}
		// b is [dim, state_dim], utExp is [batch, dim, 1] -> broadcasts to [batch, dim, state_dim].
		bU, err := s.engine.Mul(ctx, s.b.Value, utExp)
		if err != nil {
			return nil, fmt.Errorf("s4 b*u_t: %w", err)
		}
		state, err = s.engine.Add(ctx, aState, bU)
		if err != nil {
			return nil, fmt.Errorf("s4 state update: %w", err)
		}

		// y_t = sum(c * state, axis=-1) + d * u_t
		// c is [dim, state_dim], state is [batch, dim, state_dim].
		cx, err := s.engine.Mul(ctx, s.c.Value, state)
		if err != nil {
			return nil, fmt.Errorf("s4 c*state: %w", err)
		}
		// Sum over state_dim (axis 2) -> [batch, dim].
		yt, err := s.engine.Sum(ctx, cx, 2, false)
		if err != nil {
			return nil, fmt.Errorf("s4 sum c*x: %w", err)
		}

		// d * u_t: d is [dim], ut is [batch, dim] -> broadcasts.
		du, err := s.engine.Mul(ctx, s.d.Value, ut)
		if err != nil {
			return nil, fmt.Errorf("s4 d*u_t: %w", err)
		}
		yt, err = s.engine.Add(ctx, yt, du)
		if err != nil {
			return nil, fmt.Errorf("s4 y_t + d*u_t: %w", err)
		}

		// Reshape y_t to [batch, 1, dim] for later concatenation along axis 1.
		yt, err = s.engine.Reshape(ctx, yt, []int{batchSize, 1, dim})
		if err != nil {
			return nil, fmt.Errorf("s4 reshape y_t: %w", err)
		}
		stepOutputs[t] = yt
	}

	// Save final state for backward.
	s.lastStates = state

	// Concatenate all step outputs along axis 1 -> [batch, seq_len, dim].
	if seqLen == 1 {
		return s.engine.Reshape(ctx, stepOutputs[0], []int{batchSize, seqLen, dim})
	}
	return s.engine.Concat(ctx, stepOutputs, 1)
}

// Backward computes gradients using full backpropagation through time (BPTT).
//
// The adjoint equation for the hidden state is:
//
//	dL/dx_k = dL/dy_k * C + dL/dx_{k+1} * A_disc
//
// We iterate backward from the last timestep, accumulating gradients for all parameters.
func (s *S4[T]) Backward(_ context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("S4 backward requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	batchSize := shape[0]
	seqLen := shape[1]
	dim := shape[2]

	gradData := outputGradient.Data()
	uData := input.Data()
	aLogData := s.aLog.Value.Data()
	bData := s.b.Value.Data()
	cData := s.c.Value.Data()
	dData := s.d.Value.Data()

	// Pre-compute discrete A.
	aDisc := make([]T, dim*s.stateDim)
	for i, v := range aLogData {
		aDisc[i] = T(math.Exp(-math.Exp(float64(v))))
	}

	// Forward pass: recompute all states for gradient calculation.
	// allStates[batch, t, d, n] indexed as [batch*seqLen*dim*stateDim + t*dim*stateDim + d*stateDim + n]
	allStates := make([]T, batchSize*seqLen*dim*s.stateDim)
	curState := make([]T, batchSize*dim*s.stateDim)
	for batch := range batchSize {
		for t := range seqLen {
			for d := range dim {
				u := uData[batch*seqLen*dim+t*dim+d]
				for n := range s.stateDim {
					idx := d*s.stateDim + n
					prevIdx := batch*dim*s.stateDim + idx
					stateVal := aDisc[idx]*curState[prevIdx] + bData[idx]*u
					allStates[batch*seqLen*dim*s.stateDim+t*dim*s.stateDim+d*s.stateDim+n] = stateVal
					curState[prevIdx] = stateVal
				}
			}
		}
	}

	// Backward pass: iterate from last timestep to first.
	inputGradData := make([]T, batchSize*seqLen*dim)
	daLog := make([]T, dim*s.stateDim)
	db := make([]T, dim*s.stateDim)
	dc := make([]T, dim*s.stateDim)
	dd := make([]T, dim)

	// Adjoint state: dL/dx_{t+1}[batch, d, n]
	dState := make([]T, batchSize*dim*s.stateDim)

	for batch := range batchSize {
		// Reset adjoint state for each batch element.
		for i := range dState[batch*dim*s.stateDim : (batch+1)*dim*s.stateDim] {
			dState[batch*dim*s.stateDim+i] = 0
		}

		for t := seqLen - 1; t >= 0; t-- {
			for d := range dim {
				dy := gradData[batch*seqLen*dim+t*dim+d]
				u := uData[batch*seqLen*dim+t*dim+d]

				// dL/dD[d] += dy * u
				dd[d] += dy * u

				// Accumulate input gradient: dy * D[d]
				var dInput T
				dInput += dy * dData[d]

				for n := range s.stateDim {
					idx := d*s.stateDim + n
					stateIdx := batch*dim*s.stateDim + idx
					stateVal := allStates[batch*seqLen*dim*s.stateDim+t*dim*s.stateDim+d*s.stateDim+n]

					// dL/dC[d,n] += dy * x_k[d,n]
					dc[idx] += dy * stateVal

					// Adjoint: dL/dx_k = dy * C[d,n] + dL/dx_{k+1} * A_disc[d,n]
					dxk := dy*cData[idx] + dState[stateIdx]*aDisc[idx]

					// dL/dB[d,n] += dxk * u
					db[idx] += dxk * u

					// dL/dinput += dxk * B[d,n]
					dInput += dxk * bData[idx]

					// dL/dA_disc[d,n] += dxk * x_{k-1}[d,n] + dL/dx_{k+1} * x_k * ???
					// For A_log gradient: chain rule through A_disc = exp(-exp(A_log))
					// dA_disc/dA_log = -exp(A_log) * A_disc
					if t > 0 {
						prevStateVal := allStates[batch*seqLen*dim*s.stateDim+(t-1)*dim*s.stateDim+d*s.stateDim+n]
						expALog := T(math.Exp(float64(aLogData[idx])))
						// dL/dA_log += (dxk * x_{k-1} + dState_next * x_k) * dA_disc/dA_log
						// But dxk already includes dState_next * A_disc, so:
						// dL/dA_disc at step t = dxk * x_{k-1}
						daLog[idx] += dxk * prevStateVal * (-expALog * aDisc[idx])
					}

					// Update adjoint state for step t-1.
					dState[stateIdx] = dxk
				}
				inputGradData[batch*seqLen*dim+t*dim+d] = dInput
			}
		}
	}

	// Accumulate parameter gradients.
	for i := range daLog {
		s.aLog.Gradient.Data()[i] += daLog[i]
		s.b.Gradient.Data()[i] += db[i]
		s.c.Gradient.Data()[i] += dc[i]
	}
	for i := range dd {
		s.d.Gradient.Data()[i] += dd[i]
	}

	inputGrad, err := tensor.New[T]([]int{batchSize, seqLen, dim}, inputGradData)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

func init() {
	model.RegisterLayer("S4", func(
		engine compute.Engine[float32],
		ops numeric.Arithmetic[float32],
		name string,
		params map[string]*graph.Parameter[float32],
		attributes map[string]interface{},
	) (graph.Node[float32], error) {
		inputDim, ok := attributes["input_dim"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'input_dim' for S4")
		}
		stateDim, ok := attributes["state_dim"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'state_dim' for S4")
		}

		// Restore from ZMF parameters if available.
		aLog := params[name+"_a_log"]
		b := params[name+"_b"]
		c := params[name+"_c"]
		d := params[name+"_d"]
		if aLog != nil && b != nil && c != nil && d != nil {
			return &S4[float32]{
				name:     name,
				engine:   engine,
				ops:      ops,
				aLog:     aLog,
				b:        b,
				c:        c,
				d:        d,
				inputDim: inputDim,
				stateDim: stateDim,
			}, nil
		}

		return NewS4[float32](name, engine, ops, inputDim, stateDim)
	})
}

// Compile-time interface check.
var _ graph.Node[float32] = (*S4[float32])(nil)
