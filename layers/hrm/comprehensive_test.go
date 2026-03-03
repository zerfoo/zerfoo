package hrm

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// ---------- errEngine: wraps real engine, injects errors by method call count ----------

type errEngine struct {
	compute.Engine[float32]
	calls   map[string]*atomic.Int64
	failOn  map[string]int64
	failErr error
}

func newErrEngine(real compute.Engine[float32], failOn map[string]int64) *errEngine {
	e := &errEngine{
		Engine:  real,
		calls:   make(map[string]*atomic.Int64),
		failOn:  failOn,
		failErr: errors.New("injected error"),
	}
	for k := range failOn {
		e.calls[k] = &atomic.Int64{}
	}
	return e
}

func (e *errEngine) check(op string) error {
	counter, ok := e.calls[op]
	if !ok {
		return nil
	}
	n := counter.Add(1)
	if thresh, ok := e.failOn[op]; ok && n >= thresh {
		return fmt.Errorf("%w: %s call %d", e.failErr, op, n)
	}
	return nil
}

func (e *errEngine) Add(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
}

func (e *errEngine) Reshape(ctx context.Context, a *tensor.TensorNumeric[float32], shape []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Reshape"); err != nil {
		return nil, err
	}
	return e.Engine.Reshape(ctx, a, shape, dst...)
}

// ---------- errAttention: wraps real attention, injects errors ----------

type errAttention struct {
	graph.Node[float32]
	forwardErr  error
	backwardErr error
}

func (a *errAttention) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if a.forwardErr != nil {
		return nil, a.forwardErr
	}
	return a.Node.Forward(ctx, inputs...)
}

func (a *errAttention) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	if a.backwardErr != nil {
		return nil, a.backwardErr
	}
	return a.Node.Backward(ctx, mode, dOut, inputs...)
}

// ---------- Constructor error paths ----------

func TestNewHModule_BlockError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	attn, _ := attention.NewGlobalAttention[float32](engine, ops, 16, 2, 2)
	// ffnDim=0 triggers NewTransformerBlock failure
	_, err := NewHModule[float32](engine, ops, 16, 0, attn)
	if err == nil {
		t.Error("expected error from NewHModule with ffnDim=0")
	}
}

func TestNewLModule_BlockError(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	attn, _ := attention.NewGlobalAttention[float32](engine, ops, 16, 2, 2)
	// ffnDim=0 triggers NewTransformerBlock failure
	_, err := NewLModule[float32](engine, ops, 16, 0, attn)
	if err == nil {
		t.Error("expected error from NewLModule with ffnDim=0")
	}
}

// ---------- HModule Forward error paths ----------

func TestHModule_Forward_AddError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	eng := newErrEngine(real, map[string]int64{"Add": 1})
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	m, err := NewHModule[float32](eng, ops, 16, 32, attn)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected Add error in HModule.Forward")
	}
}

func TestHModule_Forward_ReshapeTo3DError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	eng := newErrEngine(real, map[string]int64{"Reshape": 1})
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	m, err := NewHModule[float32](eng, ops, 16, 32, attn)
	if err != nil {
		t.Fatal(err)
	}

	// 2D input triggers reshape to 3D
	input, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected Reshape error in HModule.Forward")
	}
}

func TestHModule_Forward_BlockForwardError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	ea := &errAttention{Node: attn, forwardErr: errors.New("attn forward fail")}
	m, err := NewHModule[float32](real, ops, 16, 32, ea)
	if err != nil {
		t.Fatal(err)
	}

	// 3D input avoids module-level Reshape; Block.Forward fails at attention
	input, _ := tensor.New[float32]([]int{1, 1, 16}, nil)
	_, err = m.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected block forward error in HModule.Forward")
	}
}

func TestHModule_Forward_ReshapeBackTo2DError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	// Reshape failOn:2 → expand (call 1) succeeds, squeeze (call 2) fails
	eng := newErrEngine(real, map[string]int64{"Reshape": 2})
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	m, err := NewHModule[float32](eng, ops, 16, 32, attn)
	if err != nil {
		t.Fatal(err)
	}

	input, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected Reshape back to 2D error in HModule.Forward")
	}
}

// ---------- HModule Backward error paths ----------

func TestHModule_Backward_ReshapeErrors(t *testing.T) {
	tests := []struct {
		name      string
		threshold int64 // Reshape call that triggers failure
	}{
		{"to_3D", 3},      // forward expand(1)+squeeze(2) ok, backward expand(3) fails
		{"back_to_2D", 4}, // forward(1,2)+backward expand(3) ok, backward squeeze(4) fails
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			real := compute.NewCPUEngine[float32](ops)
			eng := newErrEngine(real, map[string]int64{"Reshape": tt.threshold})
			attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
			m, err := NewHModule[float32](eng, ops, 16, 32, attn)
			if err != nil {
				t.Fatal(err)
			}

			input, _ := tensor.New[float32]([]int{1, 16}, nil)
			_, err = m.Forward(context.Background(), input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			dOut, _ := tensor.New[float32]([]int{1, 16}, nil)
			_, err = m.Backward(context.Background(), types.FullBackprop, dOut)
			if err == nil {
				t.Errorf("expected backward reshape error at threshold %d", tt.threshold)
			}
		})
	}
}

func TestHModule_Backward_BlockBackwardError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	ea := &errAttention{Node: attn, backwardErr: errors.New("attn backward fail")}
	m, err := NewHModule[float32](real, ops, 16, 32, ea)
	if err != nil {
		t.Fatal(err)
	}

	// Forward succeeds (forwardErr=nil)
	input, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Backward: Reshape to 3D succeeds, Block.Backward fails at attention.Backward
	dOut, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Backward(context.Background(), types.FullBackprop, dOut)
	if err == nil {
		t.Error("expected block backward error")
	}
}

// ---------- LModule Forward error paths ----------

func TestLModule_Forward_AddError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	eng := newErrEngine(real, map[string]int64{"Add": 1})
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	m, err := NewLModule[float32](eng, ops, 16, 32, attn)
	if err != nil {
		t.Fatal(err)
	}

	hState, _ := tensor.New[float32]([]int{1, 16}, nil)
	projected, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Forward(context.Background(), hState, projected)
	if err == nil {
		t.Error("expected Add error in LModule.Forward")
	}
}

func TestLModule_Forward_ReshapeTo3DError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	eng := newErrEngine(real, map[string]int64{"Reshape": 1})
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	m, err := NewLModule[float32](eng, ops, 16, 32, attn)
	if err != nil {
		t.Fatal(err)
	}

	hState, _ := tensor.New[float32]([]int{1, 16}, nil)
	projected, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Forward(context.Background(), hState, projected)
	if err == nil {
		t.Error("expected Reshape error in LModule.Forward")
	}
}

func TestLModule_Forward_BlockForwardError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	ea := &errAttention{Node: attn, forwardErr: errors.New("attn forward fail")}
	m, err := NewLModule[float32](real, ops, 16, 32, ea)
	if err != nil {
		t.Fatal(err)
	}

	// 3D input avoids module-level Reshape; Block.Forward fails at attention
	hState, _ := tensor.New[float32]([]int{1, 1, 16}, nil)
	projected, _ := tensor.New[float32]([]int{1, 1, 16}, nil)
	_, err = m.Forward(context.Background(), hState, projected)
	if err == nil {
		t.Error("expected block forward error in LModule.Forward")
	}
}

func TestLModule_Forward_ReshapeBackTo2DError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	eng := newErrEngine(real, map[string]int64{"Reshape": 2})
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	m, err := NewLModule[float32](eng, ops, 16, 32, attn)
	if err != nil {
		t.Fatal(err)
	}

	hState, _ := tensor.New[float32]([]int{1, 16}, nil)
	projected, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Forward(context.Background(), hState, projected)
	if err == nil {
		t.Error("expected Reshape back to 2D error in LModule.Forward")
	}
}

// ---------- LModule Backward error paths ----------

func TestLModule_Backward_ReshapeErrors(t *testing.T) {
	tests := []struct {
		name      string
		threshold int64
	}{
		{"to_3D", 3},
		{"back_to_2D", 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			real := compute.NewCPUEngine[float32](ops)
			eng := newErrEngine(real, map[string]int64{"Reshape": tt.threshold})
			attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
			m, err := NewLModule[float32](eng, ops, 16, 32, attn)
			if err != nil {
				t.Fatal(err)
			}

			hState, _ := tensor.New[float32]([]int{1, 16}, nil)
			projected, _ := tensor.New[float32]([]int{1, 16}, nil)
			_, err = m.Forward(context.Background(), hState, projected)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			dOut, _ := tensor.New[float32]([]int{1, 16}, nil)
			_, err = m.Backward(context.Background(), types.FullBackprop, dOut)
			if err == nil {
				t.Errorf("expected backward reshape error at threshold %d", tt.threshold)
			}
		})
	}
}

func TestLModule_Backward_BlockBackwardError(t *testing.T) {
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)
	attn, _ := attention.NewGlobalAttention[float32](real, ops, 16, 2, 2)
	ea := &errAttention{Node: attn, backwardErr: errors.New("attn backward fail")}
	m, err := NewLModule[float32](real, ops, 16, 32, ea)
	if err != nil {
		t.Fatal(err)
	}

	hState, _ := tensor.New[float32]([]int{1, 16}, nil)
	projected, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Forward(context.Background(), hState, projected)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dOut, _ := tensor.New[float32]([]int{1, 16}, nil)
	_, err = m.Backward(context.Background(), types.FullBackprop, dOut)
	if err == nil {
		t.Error("expected block backward error")
	}
}

// ---------- LModule Backward gradient duplication ----------

func TestLModule_Backward_EmptyGradientReturn(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	attn, _ := attention.NewGlobalAttention[float32](engine, ops, 16, 2, 2)
	m, err := NewLModule[float32](engine, ops, 16, 32, attn)
	if err != nil {
		t.Fatal(err)
	}

	// 3D forward so fwdNeedSqueeze=false
	m.HiddenState, _ = tensor.New[float32]([]int{1, 1, 16}, nil)
	hState, _ := tensor.New[float32]([]int{1, 1, 16}, nil)
	projected, _ := tensor.New[float32]([]int{1, 1, 16}, nil)
	_, err = m.Forward(context.Background(), hState, projected)
	if err != nil {
		t.Fatal(err)
	}

	// Backward returns 2 gradients (dInput[0] duplicated for both inputs)
	dOut, _ := tensor.New[float32]([]int{1, 1, 16}, nil)
	grads, err := m.Backward(context.Background(), types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if len(grads) != 2 {
		t.Errorf("expected 2 gradients, got %d", len(grads))
	}
}

// Statically assert interface implementations
var (
	_ graph.Node[float32] = (*HModule[float32])(nil)
	_ graph.Node[float32] = (*LModule[float32])(nil)
)
