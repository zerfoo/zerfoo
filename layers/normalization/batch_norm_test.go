// Package normalization_test tests the normalization layers.
package normalization_test

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestBatchNormalization_ZeroMean: when mean equals each channel value the output
// after scale=1, bias=0 is zero.
func TestBatchNormalization_ZeroMean(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// X: [1,2,1,1] two channels, values 3 and 7.
	X, _ := tensor.New[float32]([]int{1, 2, 1, 1}, []float32{3, 7})
	scale, _ := tensor.New[float32]([]int{2}, []float32{1, 1})
	B, _ := tensor.New[float32]([]int{2}, []float32{0, 0})
	mean, _ := tensor.New[float32]([]int{2}, []float32{3, 7})
	variance, _ := tensor.New[float32]([]int{2}, []float32{1, 1})

	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))
	out, err := layer.Forward(ctx, X, scale, B, mean, variance)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	wantShape := []int{1, 2, 1, 1}
	if !intSliceEq(out.Shape(), wantShape) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), wantShape)
	}
	for i, v := range out.Data() {
		if math.Abs(float64(v)) > 1e-4 {
			t.Errorf("out[%d] = %v, want ~0 (x==mean)", i, v)
		}
	}
}

// TestBatchNormalization_ScaleAndBias: scale=2, bias=1, mean=0, var=1 -> y = 2*x + 1.
func TestBatchNormalization_ScaleAndBias(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// X: [1,2,1,1] values 2 and 4.
	X, _ := tensor.New[float32]([]int{1, 2, 1, 1}, []float32{2, 4})
	scale, _ := tensor.New[float32]([]int{2}, []float32{2, 1})
	B, _ := tensor.New[float32]([]int{2}, []float32{1, 0})
	mean, _ := tensor.New[float32]([]int{2}, []float32{0, 0})
	variance, _ := tensor.New[float32]([]int{2}, []float32{1, 4})
	const eps = float32(1e-7)

	layer := normalization.NewBatchNormalization[float32](engine, &ops, eps)
	out, err := layer.Forward(ctx, X, scale, B, mean, variance)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	got := out.Data()
	const tol = float32(1e-4)

	// ch0: 2 * (2 - 0) / sqrt(1 + eps) + 1 = 4/~1 + 1 ≈ 5.0
	if math.Abs(float64(got[0]-5.0)) > float64(tol) {
		t.Errorf("ch0 = %v, want ~5.0", got[0])
	}
	// ch1: 1 * (4 - 0) / sqrt(4 + eps) + 0 = 4/2 = 2.0
	if math.Abs(float64(got[1]-2.0)) > float64(tol) {
		t.Errorf("ch1 = %v, want ~2.0", got[1])
	}
}

// TestBatchNormalization_Spatial: X has spatial dims [1,2,2,2].
func TestBatchNormalization_Spatial(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	// Two channels, each 2x2; channel 0: all 1, channel 1: all 2.
	data := []float32{1, 1, 1, 1, 2, 2, 2, 2}
	X, _ := tensor.New[float32]([]int{1, 2, 2, 2}, data)
	scale, _ := tensor.New[float32]([]int{2}, []float32{1, 1})
	B, _ := tensor.New[float32]([]int{2}, []float32{0, 0})
	mean, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	variance, _ := tensor.New[float32]([]int{2}, []float32{1, 1})

	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))
	out, err := layer.Forward(ctx, X, scale, B, mean, variance)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	wantShape := []int{1, 2, 2, 2}
	if !intSliceEq(out.Shape(), wantShape) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), wantShape)
	}
	// x==mean for every element, so all outputs should be ~0.
	for i, v := range out.Data() {
		if math.Abs(float64(v)) > 1e-4 {
			t.Errorf("out[%d] = %v, want ~0", i, v)
		}
	}
}

func TestBatchNormalization_InvalidInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))

	X, _ := tensor.New[float32]([]int{1, 1, 1, 1}, []float32{1})
	_, err := layer.Forward(context.Background(), X)
	if err == nil {
		t.Fatal("expected error for 1 input")
	}
}

func TestBatchNormalization_OpTypeAndMeta(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))

	if layer.OpType() != "BatchNormalization" {
		t.Errorf("OpType = %q, want BatchNormalization", layer.OpType())
	}
	if layer.Parameters() != nil {
		t.Error("Parameters should be nil")
	}
	grads, err := layer.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("Backward should return nil")
	}
}

func TestBuildBatchNormalization_DefaultEpsilon(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	node, err := normalization.BuildBatchNormalization[float32](engine, &ops, "bn", nil, map[string]interface{}{})
	if err != nil {
		t.Fatalf("BuildBatchNormalization failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildBatchNormalization returned nil")
	}
	if node.OpType() != "BatchNormalization" {
		t.Errorf("OpType = %q, want BatchNormalization", node.OpType())
	}
}

func TestBuildBatchNormalization_WithEpsilon(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	attrs := map[string]interface{}{"epsilon": float32(1e-3)}
	node, err := normalization.BuildBatchNormalization[float32](engine, &ops, "bn", nil, attrs)
	if err != nil {
		t.Fatalf("BuildBatchNormalization failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildBatchNormalization returned nil")
	}
}

func TestBuildBatchNormalization_WithFloat64Epsilon(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	attrs := map[string]interface{}{"epsilon": float64(1e-3)}
	node, err := normalization.BuildBatchNormalization[float32](engine, &ops, "bn", nil, attrs)
	if err != nil {
		t.Fatalf("BuildBatchNormalization failed: %v", err)
	}
	_ = node
}

// TestBatchNormalization_Attributes tests that Attributes returns epsilon.
func TestBatchNormalization_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	const eps = float32(1e-5)
	layer := normalization.NewBatchNormalization[float32](engine, &ops, eps)
	attrs := layer.Attributes()
	if attrs == nil {
		t.Fatal("Attributes returned nil")
	}
	if _, ok := attrs["epsilon"]; !ok {
		t.Error("missing 'epsilon' in Attributes")
	}
}

// TestBatchNormalization_OutputShape verifies OutputShape is populated after Forward.
func TestBatchNormalization_OutputShape(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	X, _ := tensor.New[float32]([]int{2, 3, 4, 4}, nil)
	data := make([]float32, 2*3*4*4)
	X.SetData(data)
	scale, _ := tensor.New[float32]([]int{3}, []float32{1, 1, 1})
	B, _ := tensor.New[float32]([]int{3}, []float32{0, 0, 0})
	mean, _ := tensor.New[float32]([]int{3}, []float32{0, 0, 0})
	variance, _ := tensor.New[float32]([]int{3}, []float32{1, 1, 1})

	layer := normalization.NewBatchNormalization[float32](engine, &ops, float32(1e-5))
	if _, err := layer.Forward(ctx, X, scale, B, mean, variance); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	want := []int{2, 3, 4, 4}
	if !intSliceEq(layer.OutputShape(), want) {
		t.Errorf("OutputShape = %v, want %v", layer.OutputShape(), want)
	}
}

// Intentionally avoid importing graph in test; use a local helper.
func intSliceEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Compile-time check that BatchNormalization satisfies graph.Node.
var _ graph.Node[float32] = normalization.NewBatchNormalization[float32](nil, nil, 0)
