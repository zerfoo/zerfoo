package activations

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/types"
)

func TestSoftmaxForward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	tests := []struct {
		name     string
		shape    []int
		input    []float32
		axis     int
		wantSums []float32 // expected row sums (each should be ~1.0)
	}{
		{
			name:     "simple_1_2_3",
			shape:    []int{1, 3},
			input:    []float32{1, 2, 3},
			axis:     -1,
			wantSums: []float32{1.0},
		},
		{
			name:     "two_rows",
			shape:    []int{2, 3},
			input:    []float32{1, 2, 3, 4, 5, 6},
			axis:     -1,
			wantSums: []float32{1.0, 1.0},
		},
		{
			name:     "single_element",
			shape:    []int{1, 1},
			input:    []float32{5.0},
			axis:     -1,
			wantSums: []float32{1.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			softmax := NewSoftmax(engine, tt.axis)
			input := makeTensor(t, tt.shape, tt.input)
			out, err := softmax.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			data := out.Data()
			// All outputs must be non-negative
			for i, v := range data {
				if v < 0 || v > 1 {
					t.Errorf("data[%d] = %v, want in [0, 1]", i, v)
				}
			}

			// Check row sums
			cols := tt.shape[len(tt.shape)-1]
			for row := 0; row < len(tt.wantSums); row++ {
				var sum float32
				for c := 0; c < cols; c++ {
					sum += data[row*cols+c]
				}
				if math.Abs(float64(sum-tt.wantSums[row])) > 1e-5 {
					t.Errorf("row %d sum = %v, want %v", row, sum, tt.wantSums[row])
				}
			}
		})
	}
}

func TestSoftmaxForwardUniform(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Equal inputs should produce uniform outputs (1/n for each element)
	n := 4
	input := makeTensor(t, []int{1, n}, []float32{5, 5, 5, 5})
	softmax := NewSoftmax(engine, -1)

	out, err := softmax.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expected := float32(1.0) / float32(n)
	for i, v := range out.Data() {
		if math.Abs(float64(v-expected)) > 1e-6 {
			t.Errorf("data[%d] = %v, want %v", i, v, expected)
		}
	}
}

func TestSoftmaxForwardLargeValues(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Large values should not cause overflow; softmax should still be valid
	input := makeTensor(t, []int{1, 3}, []float32{1000, 1001, 1002})
	softmax := NewSoftmax(engine, -1)

	out, err := softmax.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	data := out.Data()

	// Check no NaN or Inf
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("data[%d] = %v, expected finite value", i, v)
		}
	}

	// Sum should be 1.0
	var sum float32
	for _, v := range data {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("sum = %v, want 1.0", sum)
	}

	// Values should be monotonically increasing (1000 < 1001 < 1002)
	if !(data[0] < data[1] && data[1] < data[2]) {
		t.Errorf("expected monotonic increase, got %v", data)
	}
}

func TestSoftmaxForwardNegative(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	input := makeTensor(t, []int{1, 4}, []float32{-3, -2, -1, 0})
	softmax := NewSoftmax(engine, -1)

	out, err := softmax.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	data := out.Data()

	// All values must be positive
	for i, v := range data {
		if v <= 0 {
			t.Errorf("data[%d] = %v, want > 0", i, v)
		}
	}

	// Sum should be 1.0
	var sum float32
	for _, v := range data {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("sum = %v, want 1.0", sum)
	}

	// Values should be monotonically increasing (-3 < -2 < -1 < 0)
	for i := 1; i < len(data); i++ {
		if data[i] <= data[i-1] {
			t.Errorf("expected monotonic increase at index %d: %v", i, data)
			break
		}
	}
}

func TestSoftmaxForwardInputCountError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()
	softmax := NewSoftmax(engine, -1)

	// Zero inputs
	_, err := softmax.Forward(ctx)
	if err == nil {
		t.Error("expected error for 0 inputs")
	}

	// Two inputs
	a := makeTensor(t, []int{2}, []float32{1, 2})
	b := makeTensor(t, []int{2}, []float32{3, 4})
	_, err = softmax.Forward(ctx, a, b)
	if err == nil {
		t.Error("expected error for 2 inputs")
	}
}

func TestSoftmaxBackwardErrors(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Backward before Forward must error.
	softmax := NewSoftmax(engine, -1)
	dOut := makeTensor(t, []int{1, 3}, []float32{0.1, 0.2, 0.3})
	if _, err := softmax.Backward(ctx, types.FullBackprop, dOut); err == nil {
		t.Errorf("expected error when Backward is called before Forward")
	}

	// nil dOut must error.
	softmax2 := NewSoftmax(engine, -1)
	in := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
	if _, err := softmax2.Forward(ctx, in); err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if _, err := softmax2.Backward(ctx, types.FullBackprop, nil); err == nil {
		t.Errorf("expected error for nil dOut")
	}
}

// TestSoftmaxBackwardKnownValues verifies the analytical gradient for a
// hand-computable case. For input x = [1, 2, 3] with softmax output
// y = exp(x)/sum(exp(x)), and upstream gradient dOut = [1, 0, 0]:
//
//	dot     = sum(dOut * y) = y[0]
//	dInput  = y * (dOut - dot)
//	        = [y[0]*(1-y[0]), -y[0]*y[1], -y[0]*y[2]]
//
// which is the first row of the softmax Jacobian.
func TestSoftmaxBackwardKnownValues(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	softmax := NewSoftmax(engine, -1)
	in := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
	yT, err := softmax.Forward(ctx, in)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	y := yT.Data()

	dOut := makeTensor(t, []int{1, 3}, []float32{1, 0, 0})
	grads, err := softmax.Backward(ctx, types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}

	got := grads[0].Data()
	want := []float32{y[0] * (1 - y[0]), -y[0] * y[1], -y[0] * y[2]}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Errorf("dInput[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// Sum of dInput along the softmax axis must be ~0 because
	// sum_j d softmax_i / d x_j = 0 for each i, hence
	// sum_j (sum_i dOut_i * J_ij) = sum_i dOut_i * sum_j J_ij = 0.
	var s float32
	for _, v := range got {
		s += v
	}
	if math.Abs(float64(s)) > 1e-6 {
		t.Errorf("sum(dInput) = %v, want ~0", s)
	}
}

func TestSoftmaxMetadata(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	softmax := NewSoftmax(engine, -1)

	if softmax.OpType() != "Softmax" {
		t.Errorf("OpType = %q, want %q", softmax.OpType(), "Softmax")
	}

	attrs := softmax.Attributes()
	if attrs["axis"] != -1 {
		t.Errorf("axis = %v, want -1", attrs["axis"])
	}

	if softmax.Parameters() != nil {
		t.Error("expected nil parameters")
	}

	if softmax.OutputShape() != nil {
		t.Error("expected nil output shape before forward")
	}

	// After forward, OutputShape should be set
	ctx := context.Background()
	input := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	_, err := softmax.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	shape := softmax.OutputShape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Errorf("OutputShape = %v, want [2 3]", shape)
	}
}

func TestBuildSoftmax(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	tests := []struct {
		name       string
		attrs      map[string]interface{}
		expectAxis int
	}{
		{
			name:       "default_axis",
			attrs:      nil,
			expectAxis: -1,
		},
		{
			name:       "axis_int",
			attrs:      map[string]interface{}{"axis": 0},
			expectAxis: 0,
		},
		{
			name:       "axis_int64",
			attrs:      map[string]interface{}{"axis": int64(1)},
			expectAxis: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := BuildSoftmax(engine, ops, "softmax", nil, tt.attrs)
			if err != nil {
				t.Fatalf("BuildSoftmax: %v", err)
			}
			if node.OpType() != "Softmax" {
				t.Errorf("OpType = %q, want %q", node.OpType(), "Softmax")
			}
			attrs := node.Attributes()
			if attrs["axis"] != tt.expectAxis {
				t.Errorf("axis = %v, want %v", attrs["axis"], tt.expectAxis)
			}
		})
	}
}

// Statically assert that Softmax implements graph.Node.
func TestSoftmaxInterfaceConformance(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	var _ interface {
		OpType() string
		Attributes() map[string]interface{}
		OutputShape() []int
	} = NewSoftmax(engine, -1)

	// Also verify via New
	s := NewSoftmax(engine, 0)
	_ = s.OpType()
	_ = s.Attributes()
	_ = s.OutputShape()
	_ = s.Parameters()
}

// Verify specific softmax values for [1, 2, 3]
func TestSoftmaxForwardKnownValues(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	input := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
	softmax := NewSoftmax(engine, -1)

	out, err := softmax.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Compute expected: softmax([1,2,3])
	// e^1 / (e^1 + e^2 + e^3), e^2 / ..., e^3 / ...
	e1 := math.Exp(1)
	e2 := math.Exp(2)
	e3 := math.Exp(3)
	denom := e1 + e2 + e3
	expected := []float64{e1 / denom, e2 / denom, e3 / denom}

	data := out.Data()
	for i, want := range expected {
		got := float64(data[i])
		if math.Abs(got-want) > 1e-5 {
			t.Errorf("data[%d] = %v, want %v", i, got, want)
		}
	}
}
