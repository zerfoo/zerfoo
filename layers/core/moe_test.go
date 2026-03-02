package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// identityExpert returns its input unchanged.
type identityExpert struct{ shape []int }

func (n *identityExpert) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	n.shape = inputs[0].Shape()
	return inputs[0], nil
}

func (n *identityExpert) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *identityExpert) OpType() string                          { return "Identity" }
func (n *identityExpert) Attributes() map[string]interface{}      { return nil }
func (n *identityExpert) OutputShape() []int                      { return n.shape }
func (n *identityExpert) Parameters() []*graph.Parameter[float32] { return nil }

// scale2Expert multiplies each element by 2.
type scale2Expert struct{ shape []int }

func (n *scale2Expert) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	src := inputs[0].Data()
	out := make([]float32, len(src))
	for i, v := range src {
		out[i] = v * 2
	}
	t, err := tensor.New[float32](inputs[0].Shape(), out)
	if err != nil {
		return nil, err
	}
	n.shape = t.Shape()
	return t, nil
}

func (n *scale2Expert) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *scale2Expert) OpType() string                          { return "ScaleBy2" }
func (n *scale2Expert) Attributes() map[string]interface{}      { return nil }
func (n *scale2Expert) OutputShape() []int                      { return n.shape }
func (n *scale2Expert) Parameters() []*graph.Parameter[float32] { return nil }

// --- MoEGate tests ---

func TestMoEGate_ForwardShapeAndWeightsSumToOne(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2) // topK=2

	// hiddenStates [2, 3] (seq_len=2, model_dim=3)
	hs, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 0, 0, 0, 1, 0})
	// gateWeight [3, 3] (num_experts=3, model_dim=3) - identity matrix
	gw, _ := tensor.New[float32]([]int{3, 3}, []float32{1, 0, 0, 0, 1, 0, 0, 0, 1})

	out, err := gate.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("MoEGate.Forward failed: %v", err)
	}
	if len(out.Shape()) != 2 || out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Errorf("output shape = %v, want [2 2]", out.Shape())
	}
	data := out.Data()
	for row := 0; row < 2; row++ {
		sum := float64(0)
		for k := 0; k < 2; k++ {
			sum += float64(data[row*2+k])
		}
		if math.Abs(sum-1.0) > 1e-5 {
			t.Errorf("row %d weights sum = %f, want 1.0", row, sum)
		}
	}
}

func TestMoEGate_InvalidInputs(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2)

	_, err := gate.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}

	hs, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
	_, err = gate.Forward(context.Background(), hs)
	if err == nil {
		t.Error("expected error for 1 input")
	}
}

func TestMoEGate_OpType(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	if gate.OpType() != "MoEGate" {
		t.Errorf("OpType = %q, want MoEGate", gate.OpType())
	}
}

func TestMoEGate_Attributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2)
	attrs := gate.Attributes()
	if v, ok := attrs["top_k"]; !ok || v != 2 {
		t.Errorf("attributes[top_k] = %v, want 2", attrs["top_k"])
	}
}

func TestMoEGate_Backward(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	grads, err := gate.Backward(context.Background(), types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("MoEGate.Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil grads")
	}
}

func TestMoEGate_Parameters(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	if gate.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestMoEGate_OutputShape(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	if gate.OutputShape() != nil {
		t.Error("expected nil output shape before forward")
	}
}

func TestBuildMoEGate_Int64TopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{"top_k": int64(2)}
	node, err := BuildMoEGate(eng, ops, "gate", nil, attrs)
	if err != nil {
		t.Fatalf("BuildMoEGate failed: %v", err)
	}
	if node.OpType() != "MoEGate" {
		t.Errorf("OpType = %q, want MoEGate", node.OpType())
	}
}

func TestBuildMoEGate_IntTopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{"top_k": int(3)}
	node, err := BuildMoEGate(eng, ops, "gate", nil, attrs)
	if err != nil {
		t.Fatalf("BuildMoEGate with int top_k failed: %v", err)
	}
	if node.OpType() != "MoEGate" {
		t.Errorf("OpType = %q, want MoEGate", node.OpType())
	}
}

func TestBuildMoEGate_MissingTopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMoEGate(eng, ops, "gate", nil, map[string]interface{}{})
	if err == nil {
		t.Error("expected error for missing top_k")
	}
}

// --- MixtureOfExperts tests ---

func TestMixtureOfExperts_Forward(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1) // topK=1

	experts := []graph.Node[float32]{
		&identityExpert{},
		&scale2Expert{},
	}
	moe := NewMixtureOfExperts[float32](eng, ops, gate, experts, 2, 1)

	// hiddenStates [2, 2]: token 0 = [1,0], token 1 = [0,1]
	hs, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})
	// gateWeight [2, 2]: identity -> token 0 has high logit for expert 0, token 1 for expert 1
	gw, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})

	out, err := moe.Forward(context.Background(), hs, gw)
	if err != nil {
		t.Fatalf("MixtureOfExperts.Forward failed: %v", err)
	}
	if len(out.Shape()) != 2 || out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Errorf("output shape = %v, want [2 2]", out.Shape())
	}
	data := out.Data()
	// token 0 -> expert 0 (identity): output = [1, 0]
	if math.Abs(float64(data[0])-1.0) > 1e-5 || math.Abs(float64(data[1])) > 1e-5 {
		t.Errorf("token 0 output = [%f %f], want [1 0]", data[0], data[1])
	}
	// token 1 -> expert 1 (scale-by-2): output = [0, 2]
	if math.Abs(float64(data[2])) > 1e-5 || math.Abs(float64(data[3])-2.0) > 1e-5 {
		t.Errorf("token 1 output = [%f %f], want [0 2]", data[2], data[3])
	}
}

func TestMixtureOfExperts_InvalidInputs(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	_, err := moe.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestMixtureOfExperts_NoExperts(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)

	hs, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})
	gw, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 0, 0, 1})
	_, err := moe.Forward(context.Background(), hs, gw)
	if err == nil {
		t.Error("expected error when experts is nil")
	}
}

func TestMixtureOfExperts_OpType(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	if moe.OpType() != "MixtureOfExperts" {
		t.Errorf("OpType = %q, want MixtureOfExperts", moe.OpType())
	}
}

func TestMixtureOfExperts_Attributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 2)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 3, 2)
	attrs := moe.Attributes()
	if v, ok := attrs["num_experts"]; !ok || v != 3 {
		t.Errorf("attributes[num_experts] = %v, want 3", attrs["num_experts"])
	}
	if v, ok := attrs["top_k"]; !ok || v != 2 {
		t.Errorf("attributes[top_k] = %v, want 2", attrs["top_k"])
	}
}

func TestMixtureOfExperts_Backward(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	grads, err := moe.Backward(context.Background(), types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("MixtureOfExperts.Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil grads")
	}
}

func TestMixtureOfExperts_OutputShape(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	if moe.OutputShape() != nil {
		t.Error("expected nil output shape before forward")
	}
}

func TestMixtureOfExperts_Parameters(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	gate := NewMoEGate[float32](eng, ops, 1)
	moe := NewMixtureOfExperts[float32](eng, ops, gate, nil, 2, 1)
	if moe.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestBuildMixtureOfExperts_Int64Attrs(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{
		"num_experts": int64(4),
		"top_k":       int64(2),
	}
	node, err := BuildMixtureOfExperts(eng, ops, "moe", nil, attrs)
	if err != nil {
		t.Fatalf("BuildMixtureOfExperts failed: %v", err)
	}
	if node.OpType() != "MixtureOfExperts" {
		t.Errorf("OpType = %q, want MixtureOfExperts", node.OpType())
	}
}

func TestBuildMixtureOfExperts_MissingNumExperts(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMixtureOfExperts(eng, ops, "moe", nil, map[string]interface{}{"top_k": int64(2)})
	if err == nil {
		t.Error("expected error for missing num_experts")
	}
}

func TestBuildMixtureOfExperts_MissingTopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildMixtureOfExperts(eng, ops, "moe", nil, map[string]interface{}{"num_experts": int64(4)})
	if err == nil {
		t.Error("expected error for missing top_k")
	}
}
