package crossasset

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// cpuAdamState wraps the canonical AdamW[float32] optimizer and the
// graph.Parameter list that maps 1-to-1 with the model's weight slices.
type cpuAdamState struct {
	opt    *optimizer.AdamW[float32]
	params []*graph.Parameter[float32]
}

// newAdamState creates a canonical AdamW optimizer and registers every model
// parameter as a graph.Parameter whose Value tensor aliases the model's
// []float32 slice (zero-copy).
func newAdamState(model *Model, lr float64) (*cpuAdamState, error) {
	const (
		beta1       float32 = 0.9
		beta2       float32 = 0.999
		eps         float32 = 1e-8
		weightDecay float32 = 0.01
	)

	opt := optimizer.NewAdamW[float32](cpuEngine, float32(lr), beta1, beta2, eps, weightDecay)
	opt.SetMaxGradNorm(1.0)

	var params []*graph.Parameter[float32]

	wrap := func(name string, data []float32) error {
		t, err := tensor.New[float32]([]int{len(data)}, data)
		if err != nil {
			return fmt.Errorf("crossasset: wrap param %s: %w", name, err)
		}
		p, err := graph.NewParameter[float32](name, t, tensor.New[float32])
		if err != nil {
			return fmt.Errorf("crossasset: new param %s: %w", name, err)
		}
		params = append(params, p)
		return nil
	}

	// Head.
	if err := wrap("headW", model.headW); err != nil {
		return nil, err
	}
	if err := wrap("headB", model.headB); err != nil {
		return nil, err
	}

	// Layers.
	for li := range model.layers {
		l := &model.layers[li]
		prefix := fmt.Sprintf("layer%d.", li)
		for _, kv := range []struct {
			name string
			data []float32
		}{
			{"qW", l.qW}, {"kW", l.kW}, {"vW", l.vW}, {"outW", l.outW},
			{"lnGamma", l.lnGamma}, {"lnBeta", l.lnBeta},
			{"ffnW1", l.ffnW1}, {"ffnB1", l.ffnB1},
			{"ffnW2", l.ffnW2}, {"ffnB2", l.ffnB2},
			{"ffnGamma", l.ffnGamma}, {"ffnBeta", l.ffnBeta},
		} {
			if err := wrap(prefix+kv.name, kv.data); err != nil {
				return nil, err
			}
		}
	}

	// Input projections.
	for s := range model.inputW {
		prefix := fmt.Sprintf("input%d.", s)
		if err := wrap(prefix+"W", model.inputW[s]); err != nil {
			return nil, err
		}
		if err := wrap(prefix+"B", model.inputB[s]); err != nil {
			return nil, err
		}
	}

	return &cpuAdamState{opt: opt, params: params}, nil
}

// adamWUpdateAll sets gradients on each registered parameter and delegates
// the weight update to the canonical AdamW[float32] optimizer.
func adamWUpdateAll(
	model *Model,
	dHeadW, dHeadB []float32,
	dLayers []layer,
	dInputW, dInputB [][]float32,
	state *cpuAdamState,
) error {
	// Collect gradient slices in the same order as params were registered.
	var grads [][]float32

	// Head.
	grads = append(grads, dHeadW, dHeadB)

	// Layers.
	for li := range dLayers {
		dl := &dLayers[li]
		grads = append(grads,
			dl.qW, dl.kW, dl.vW, dl.outW,
			dl.lnGamma, dl.lnBeta,
			dl.ffnW1, dl.ffnB1,
			dl.ffnW2, dl.ffnB2,
			dl.ffnGamma, dl.ffnBeta,
		)
	}

	// Input projections.
	for s := range dInputW {
		grads = append(grads, dInputW[s], dInputB[s])
	}

	if len(grads) != len(state.params) {
		return fmt.Errorf("crossasset: adamw: gradient count %d != param count %d", len(grads), len(state.params))
	}

	// Set gradient tensors on each parameter (zero-copy alias).
	for i, p := range state.params {
		g, err := tensor.New[float32]([]int{len(grads[i])}, grads[i])
		if err != nil {
			return fmt.Errorf("crossasset: adamw: wrap gradient %s: %w", p.Name, err)
		}
		p.Gradient = g
	}

	return state.opt.Step(context.Background(), state.params)
}
