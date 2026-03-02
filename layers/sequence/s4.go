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
func (s *S4[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
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

	// Compute discrete A = exp(-exp(a_log)) element-wise.
	// Since a_log parameterizes log(-A), discrete A = exp(-exp(a_log) * dt).
	// We use dt=1 for simplicity, so A_disc = exp(-exp(a_log)).
	aLogData := s.aLog.Value.Data()
	bData := s.b.Value.Data()
	cData := s.c.Value.Data()
	dData := s.d.Value.Data()
	uData := input.Data()

	// Pre-compute discrete A values.
	aDisc := make([]T, dim*s.stateDim)
	for i, v := range aLogData {
		aDisc[i] = T(math.Exp(-math.Exp(float64(v))))
	}

	// Run the scan.
	outputData := make([]T, batchSize*seqLen*dim)
	// State: [batch, dim, state_dim]
	state := make([]T, batchSize*dim*s.stateDim)

	for batch := range batchSize {
		for t := range seqLen {
			for d := range dim {
				u := uData[batch*seqLen*dim+t*dim+d]
				var y T
				for n := range s.stateDim {
					idx := d*s.stateDim + n
					stateIdx := batch*dim*s.stateDim + idx
					// x_k = a * x_{k-1} + b * u_k
					state[stateIdx] = aDisc[idx]*state[stateIdx] + bData[idx]*u
					// y += c * x_k
					y += cData[idx] * state[stateIdx]
				}
				// y += d * u (skip connection)
				y += dData[d] * u
				outputData[batch*seqLen*dim+t*dim+d] = y
			}
		}
	}

	// Save final state for backward.
	finalState, err := tensor.New[T]([]int{batchSize, dim, s.stateDim}, state)
	if err != nil {
		return nil, err
	}
	s.lastStates = finalState

	return tensor.New[T]([]int{batchSize, seqLen, dim}, outputData)
}

// Backward computes gradients using one-step approximation.
// Gradients are computed for the last time step only, treating the state as detached.
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

	// Recompute states for gradient calculation.
	// states[batch][t][dim*stateDim]
	allStates := make([]T, batchSize*seqLen*dim*s.stateDim)
	prevState := make([]T, batchSize*dim*s.stateDim)
	for batch := range batchSize {
		for t := range seqLen {
			for d := range dim {
				u := uData[batch*seqLen*dim+t*dim+d]
				for n := range s.stateDim {
					idx := d*s.stateDim + n
					prevIdx := batch*dim*s.stateDim + idx
					stateVal := aDisc[idx]*prevState[prevIdx] + bData[idx]*u
					allStates[batch*seqLen*dim*s.stateDim+t*dim*s.stateDim+d*s.stateDim+n] = stateVal
					prevState[prevIdx] = stateVal
				}
			}
		}
	}

	// Compute gradients.
	inputGradData := make([]T, batchSize*seqLen*dim)
	daLog := make([]T, dim*s.stateDim)
	db := make([]T, dim*s.stateDim)
	dc := make([]T, dim*s.stateDim)
	dd := make([]T, dim)

	for batch := range batchSize {
		for t := range seqLen {
			for d := range dim {
				dy := gradData[batch*seqLen*dim+t*dim+d]
				u := uData[batch*seqLen*dim+t*dim+d]

				// d/d(D) += dy * u
				dd[d] += dy * u

				// d/d(input) += dy * D[d]
				var dInput T
				dInput += dy * dData[d]

				for n := range s.stateDim {
					idx := d*s.stateDim + n
					stateVal := allStates[batch*seqLen*dim*s.stateDim+t*dim*s.stateDim+d*s.stateDim+n]

					// d/d(C) += dy * x_k
					dc[idx] += dy * stateVal

					// dL/dx_k = dy * C[d,n]
					dxk := dy * cData[idx]

					// d/d(B) += dxk * u
					db[idx] += dxk * u

					// d/d(input) += dxk * B[d,n]
					dInput += dxk * bData[idx]

					// d/d(A_log): A_disc = exp(-exp(a_log))
					// dA_disc/da_log = -exp(a_log) * exp(-exp(a_log)) = -exp(a_log) * A_disc
					// dL/da_log = dxk * x_{k-1} * dA_disc/da_log
					if t > 0 {
						prevState := allStates[batch*seqLen*dim*s.stateDim+(t-1)*dim*s.stateDim+d*s.stateDim+n]
						expALog := T(math.Exp(float64(aLogData[idx])))
						daLog[idx] += dxk * prevState * (-expALog * aDisc[idx])
					}
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
		_ map[string]*graph.Parameter[float32],
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
		return NewS4[float32](name, engine, ops, inputDim, stateDim)
	})
}

// Compile-time interface check.
var _ graph.Node[float32] = (*S4[float32])(nil)
