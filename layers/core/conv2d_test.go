package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func newConv2dOnesFloat32(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = 1
	}
	t, _ := tensor.New[float32](shape, data)
	return t
}

func shapeEq(a, b []int) bool {
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

// TestConv2d_ForwardShapeAndValues exercises stride/kernel combinations using
// a table-driven approach to avoid structural code duplication (dupl linter).
func TestConv2d_ForwardShapeAndValues(t *testing.T) {
	type tc struct {
		name       string
		inputShape []int
		kernShape  []int
		strides    []int
		wantShape  []int
		wantVal    float32
	}
	cases := []tc{
		{
			name:       "5x5_3x3_stride1",
			inputShape: []int{1, 1, 5, 5},
			kernShape:  []int{1, 1, 3, 3},
			strides:    []int{1, 1},
			wantShape:  []int{1, 1, 3, 3},
			wantVal:    9, // 3*3 sum of ones
		},
		{
			name:       "4x4_2x2_stride2",
			inputShape: []int{1, 1, 4, 4},
			kernShape:  []int{1, 1, 2, 2},
			strides:    []int{2, 2},
			wantShape:  []int{1, 1, 2, 2},
			wantVal:    4, // 2*2 sum of ones
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](&ops)

			x := newConv2dOnesFloat32(c.inputShape)
			w := newConv2dOnesFloat32(c.kernShape)

			conv := NewConv2d[float32](engine, &ops, c.strides, []int{0, 0, 0, 0}, []int{1, 1}, 1)
			out, err := conv.Forward(context.Background(), x, w)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			if !shapeEq(out.Shape(), c.wantShape) {
				t.Fatalf("shape mismatch: got %v want %v", out.Shape(), c.wantShape)
			}
			for i, v := range out.Data() {
				if v != c.wantVal {
					t.Errorf("out[%d] = %v, want %v", i, v, c.wantVal)
				}
			}
		})
	}
}

// TestConv2d_WithBias: same as AllOnes test but adds bias=1 to each output channel.
func TestConv2d_WithBias(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x := newConv2dOnesFloat32([]int{1, 1, 5, 5})
	w := newConv2dOnesFloat32([]int{1, 1, 3, 3})
	b, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	out, err := conv.Forward(context.Background(), x, w, b)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	for i, v := range out.Data() {
		if v != 10 {
			t.Errorf("out[%d] = %v, want 10", i, v)
		}
	}
}

// TestConv2d_TwoOutputChannels: kernel [2,1,3,3] produces two output channels.
func TestConv2d_TwoOutputChannels(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x := newConv2dOnesFloat32([]int{1, 1, 5, 5})
	// First kernel all-ones (each out = 9), second kernel all-twos (each out = 18).
	wData := make([]float32, 2*1*3*3)
	for i := range 9 {
		wData[i] = 1
	}
	for i := 9; i < 18; i++ {
		wData[i] = 2
	}
	w, _ := tensor.New[float32]([]int{2, 1, 3, 3}, wData)

	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	out, err := conv.Forward(context.Background(), x, w)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	want := []int{1, 2, 3, 3}
	if !shapeEq(out.Shape(), want) {
		t.Fatalf("shape mismatch: got %v want %v", out.Shape(), want)
	}
	data := out.Data()
	// First 9 values are channel 0 (all 9), next 9 are channel 1 (all 18).
	for i, v := range data[:9] {
		if v != 9 {
			t.Errorf("ch0[%d] = %v, want 9", i, v)
		}
	}
	for i, v := range data[9:] {
		if v != 18 {
			t.Errorf("ch1[%d] = %v, want 18", i, v)
		}
	}
}

func TestConv2d_InvalidInputCount(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)

	_, err := conv.Forward(context.Background())
	if err == nil {
		t.Fatal("expected error for 0 inputs")
	}
}

func TestConv2d_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	if conv.OpType() != "Conv" {
		t.Errorf("OpType = %q, want %q", conv.OpType(), "Conv")
	}
}

func TestConv2d_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	conv := NewConv2d[float32](engine, &ops, []int{2, 3}, []int{1, 1, 1, 1}, []int{2, 2}, 4)
	attrs := conv.Attributes()
	if attrs == nil {
		t.Fatal("Attributes returned nil")
	}
	if _, ok := attrs["strides"]; !ok {
		t.Error("missing strides attribute")
	}
	if _, ok := attrs["pads"]; !ok {
		t.Error("missing pads attribute")
	}
	if _, ok := attrs["dilations"]; !ok {
		t.Error("missing dilations attribute")
	}
	if _, ok := attrs["group"]; !ok {
		t.Error("missing group attribute")
	}
}

func TestConv2d_ParametersAndBackward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	if conv.Parameters() != nil {
		t.Error("Parameters should be nil")
	}
	grads, err := conv.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("Backward should return nil grads for inference-only layer")
	}
}

func TestBuildConv2d_Defaults(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	// Build with empty attributes (should use defaults).
	node, err := BuildConv2d[float32](engine, &ops, "conv", nil, map[string]interface{}{})
	if err != nil {
		t.Fatalf("BuildConv2d failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildConv2d returned nil node")
	}
}

func TestBuildConv2d_WithAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)
	attrs := map[string]interface{}{
		"strides":   []int64{2, 2},
		"pads":      []int64{1, 1, 1, 1},
		"dilations": []int64{1, 1},
		"group":     int64(1),
	}
	node, err := BuildConv2d[float32](engine, &ops, "conv", nil, attrs)
	if err != nil {
		t.Fatalf("BuildConv2d failed: %v", err)
	}
	if node == nil {
		t.Fatal("BuildConv2d returned nil node")
	}
	if node.OpType() != "Conv" {
		t.Errorf("OpType = %q, want Conv", node.OpType())
	}
}

// TestConv2d_OutputShape verifies OutputShape is populated after Forward.
func TestConv2d_OutputShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](&ops)

	x := newConv2dOnesFloat32([]int{1, 1, 5, 5})
	w := newConv2dOnesFloat32([]int{1, 1, 3, 3})

	conv := NewConv2d[float32](engine, &ops, []int{1, 1}, []int{0, 0, 0, 0}, []int{1, 1}, 1)
	if _, err := conv.Forward(context.Background(), x, w); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	want := []int{1, 1, 3, 3}
	if !shapeEq(conv.OutputShape(), want) {
		t.Errorf("OutputShape = %v, want %v", conv.OutputShape(), want)
	}
}
