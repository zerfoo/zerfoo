package activations

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// ---------- BuildSigmoid ----------

func TestBuildSigmoid(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	node, err := BuildSigmoid(eng, ops, "sigmoid", nil, nil)
	if err != nil {
		t.Fatalf("BuildSigmoid failed: %v", err)
	}
	if node.OpType() != "Sigmoid" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "Sigmoid")
	}
}

func TestBuildSigmoid_Forward(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	node, _ := BuildSigmoid(eng, ops, "sigmoid", nil, nil)
	input := makeTensor(t, []int{3}, []float32{0, 1, -1})
	out, err := node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if len(out.Data()) != 3 {
		t.Errorf("output length = %d, want 3", len(out.Data()))
	}
	// sigmoid(0) = 0.5
	if math.Abs(float64(out.Data()[0])-0.5) > 1e-5 {
		t.Errorf("sigmoid(0) = %f, want 0.5", out.Data()[0])
	}
}

// ---------- Softmax ----------

func TestSoftmax_Forward(t *testing.T) {
	eng := makeEngine()
	sm := NewSoftmax[float32](eng, -1)

	input := makeTensor(t, []int{4}, []float32{1, 2, 3, 4})
	out, err := sm.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Softmax.Forward failed: %v", err)
	}
	if len(out.Data()) != 4 {
		t.Errorf("output length = %d, want 4", len(out.Data()))
	}
	// Softmax outputs must sum to 1
	var sum float32
	for _, v := range out.Data() {
		sum += v
	}
	if math.Abs(float64(sum)-1.0) > 1e-5 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}
	// All values must be positive
	for i, v := range out.Data() {
		if v <= 0 {
			t.Errorf("softmax[%d] = %f, must be > 0", i, v)
		}
	}
}

func TestSoftmax_ForwardInputError(t *testing.T) {
	eng := makeEngine()
	sm := NewSoftmax[float32](eng, -1)
	_, err := sm.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestSoftmax_Backward(t *testing.T) {
	eng := makeEngine()
	sm := NewSoftmax[float32](eng, -1)
	grads, err := sm.Backward(context.Background(), types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("Softmax.Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil grads from Softmax.Backward")
	}
}

func TestSoftmax_OpType(t *testing.T) {
	eng := makeEngine()
	sm := NewSoftmax[float32](eng, -1)
	if sm.OpType() != "Softmax" {
		t.Errorf("OpType = %q, want %q", sm.OpType(), "Softmax")
	}
}

func TestSoftmax_Attributes(t *testing.T) {
	eng := makeEngine()
	sm := NewSoftmax[float32](eng, 1)
	attrs := sm.Attributes()
	if attrs["axis"] != 1 {
		t.Errorf("axis = %v, want 1", attrs["axis"])
	}
}

func TestSoftmax_OutputShape(t *testing.T) {
	eng := makeEngine()
	sm := NewSoftmax[float32](eng, -1)
	if sm.OutputShape() != nil {
		t.Error("OutputShape should be nil before forward")
	}
	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	_, _ = sm.Forward(context.Background(), input)
	shape := sm.OutputShape()
	if len(shape) != 2 || shape[0] != 1 || shape[1] != 4 {
		t.Errorf("OutputShape = %v, want [1, 4]", shape)
	}
}

func TestSoftmax_Parameters(t *testing.T) {
	eng := makeEngine()
	sm := NewSoftmax[float32](eng, -1)
	if sm.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestBuildSoftmax_DefaultAxis(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	node, err := BuildSoftmax(eng, ops, "softmax", nil, nil)
	if err != nil {
		t.Fatalf("BuildSoftmax failed: %v", err)
	}
	if node.OpType() != "Softmax" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "Softmax")
	}
	attrs := node.Attributes()
	if attrs["axis"] != -1 {
		t.Errorf("default axis = %v, want -1", attrs["axis"])
	}
}

func TestBuildSoftmax_WithAxis(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	attrs := map[string]interface{}{"axis": 2}
	node, err := BuildSoftmax(eng, ops, "softmax", nil, attrs)
	if err != nil {
		t.Fatalf("BuildSoftmax failed: %v", err)
	}
	if node.Attributes()["axis"] != 2 {
		t.Errorf("axis = %v, want 2", node.Attributes()["axis"])
	}
}

func TestBuildSoftmax_WithInt64Axis(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	attrs := map[string]interface{}{"axis": int64(1)}
	node, err := BuildSoftmax(eng, ops, "softmax", nil, attrs)
	if err != nil {
		t.Fatalf("BuildSoftmax failed: %v", err)
	}
	if node.Attributes()["axis"] != 1 {
		t.Errorf("axis = %v, want 1", node.Attributes()["axis"])
	}
}

// ---------- Erf ----------

func TestErf_Forward(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	erfLayer := NewErf[float32](eng, ops)

	input := makeTensor(t, []int{3}, []float32{0, 1, -1})
	out, err := erfLayer.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Erf.Forward failed: %v", err)
	}
	data := out.Data()
	// erf(0) = 0
	if math.Abs(float64(data[0])) > 1e-6 {
		t.Errorf("erf(0) = %f, want 0", data[0])
	}
	// erf(1) ≈ 0.8427008
	if math.Abs(float64(data[1])-0.8427008) > 1e-5 {
		t.Errorf("erf(1) = %f, want ~0.8427008", data[1])
	}
	// erf(-1) ≈ -0.8427008
	if math.Abs(float64(data[2])+0.8427008) > 1e-5 {
		t.Errorf("erf(-1) = %f, want ~-0.8427008", data[2])
	}
}

func TestErf_OpType(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	erfLayer := NewErf[float32](eng, ops)
	if erfLayer.OpType() != "Erf" {
		t.Errorf("OpType = %q, want %q", erfLayer.OpType(), "Erf")
	}
}

func TestErf_Backward(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}
	erfLayer := NewErf[float32](eng, ops)

	ctx := context.Background()
	input, _ := tensor.New[float32]([]int{3}, []float32{0, 0.5, -0.5})
	_, _ = erfLayer.Forward(ctx, input)

	grad, _ := tensor.New[float32]([]int{3}, []float32{1, 1, 1})
	grads, err := erfLayer.Backward(ctx, types.FullBackprop, grad)
	if err != nil {
		t.Fatalf("Erf.Backward failed: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}
	// d/dx erf(0) = 2/sqrt(pi) ≈ 1.1284
	expected0 := float32(2.0 / math.Sqrt(math.Pi))
	if math.Abs(float64(grads[0].Data()[0])-float64(expected0)) > 1e-5 {
		t.Errorf("erf'(0) = %f, want %f", grads[0].Data()[0], expected0)
	}
}

func TestBuildErf(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	node, err := BuildErf(eng, ops, "erf", nil, nil)
	if err != nil {
		t.Fatalf("BuildErf failed: %v", err)
	}
	if node.OpType() != "Erf" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "Erf")
	}
}
