package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// ---------------------------------------------------------------------------
// Cos / Sin: Parameters() and Forward with int32 (default type switch branch)
// ---------------------------------------------------------------------------

func TestCos_Parameters(t *testing.T) {
	c := &Cos[float32]{}
	if p := c.Parameters(); p != nil {
		t.Errorf("Cos.Parameters() = %v, want nil", p)
	}
}

func TestSin_Parameters(t *testing.T) {
	s := &Sin[float32]{}
	if p := s.Parameters(); p != nil {
		t.Errorf("Sin.Parameters() = %v, want nil", p)
	}
}

func TestCos_Forward_UnsupportedType(t *testing.T) {
	c := &Cos[int32]{}
	in, _ := tensor.New[int32]([]int{2}, []int32{1, 2})
	_, err := c.Forward(context.Background(), in)
	if err == nil {
		t.Error("expected error for unsupported type int32")
	}
}

func TestSin_Forward_UnsupportedType(t *testing.T) {
	s := &Sin[int32]{}
	in, _ := tensor.New[int32]([]int{2}, []int32{1, 2})
	_, err := s.Forward(context.Background(), in)
	if err == nil {
		t.Error("expected error for unsupported type int32")
	}
}

func TestCos_Forward_Float64(t *testing.T) {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	c := &Cos[float64]{engine: engine}
	in, _ := tensor.New[float64]([]int{2}, []float64{0, 3.14159265358979})
	out, err := c.Forward(context.Background(), in)
	if err != nil {
		t.Fatalf("Cos Forward float64: %v", err)
	}
	if got := out.Data()[0]; got < 0.999 || got > 1.001 {
		t.Errorf("cos(0) = %v, want ~1.0", got)
	}
}

func TestSin_Forward_Float64(t *testing.T) {
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	s := &Sin[float64]{engine: engine}
	in, _ := tensor.New[float64]([]int{2}, []float64{1.5707963267948966, 0})
	out, err := s.Forward(context.Background(), in)
	if err != nil {
		t.Fatalf("Sin Forward float64: %v", err)
	}
	if got := out.Data()[0]; got < 0.999 || got > 1.001 {
		t.Errorf("sin(pi/2) = %v, want ~1.0", got)
	}
}

// ---------------------------------------------------------------------------
// Constant: Attributes with int32, int64, uint8 dtypes + unknown default
// ---------------------------------------------------------------------------

func TestConstant_Attributes_Int32(t *testing.T) {
	v, _ := tensor.New[int32]([]int{2}, []int32{1, 2})
	c := &Constant[int32]{name: "c", value: v}
	attrs := c.Attributes()
	if attrs["dtype"] != "int32" {
		t.Errorf("dtype = %v, want int32", attrs["dtype"])
	}
}

func TestConstant_Attributes_Int64(t *testing.T) {
	v, _ := tensor.New[int64]([]int{2}, []int64{1, 2})
	c := &Constant[int64]{name: "c", value: v}
	attrs := c.Attributes()
	if attrs["dtype"] != "int64" {
		t.Errorf("dtype = %v, want int64", attrs["dtype"])
	}
}

func TestConstant_Attributes_Uint8(t *testing.T) {
	v, _ := tensor.New[uint8]([]int{2}, []uint8{1, 2})
	c := &Constant[uint8]{name: "c", value: v}
	attrs := c.Attributes()
	if attrs["dtype"] != "uint8" {
		t.Errorf("dtype = %v, want uint8", attrs["dtype"])
	}
}

func TestConstant_Attributes_Float64(t *testing.T) {
	v, _ := tensor.New[float64]([]int{2}, []float64{1.0, 2.0})
	c := &Constant[float64]{name: "c", value: v}
	attrs := c.Attributes()
	if attrs["dtype"] != "float64" {
		t.Errorf("dtype = %v, want float64", attrs["dtype"])
	}
}

func TestConstant_Attributes_Unknown(t *testing.T) {
	v, _ := tensor.New[int16]([]int{2}, []int16{1, 2})
	c := &Constant[int16]{name: "c", value: v}
	attrs := c.Attributes()
	if attrs["dtype"] != "unknown" {
		t.Errorf("dtype = %v, want unknown", attrs["dtype"])
	}
}

// ---------------------------------------------------------------------------
// ReduceMean: axes from second input tensor (ONNX opset 18+)
// ---------------------------------------------------------------------------

func TestReduceMean_AxesFromSecondInput(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	// ReduceMean with empty axes, so it reads from second input.
	rm := &ReduceMean[float32]{engine: engine, axes: nil, keepDims: true}

	// Input: [2, 3] matrix.
	input, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	// Axes tensor: reduce along axis 1.
	axes, _ := tensor.New[float32]([]int{1}, []float32{1})

	out, err := rm.Forward(ctx, input, axes)
	if err != nil {
		t.Fatalf("ReduceMean with axes from input: %v", err)
	}

	// Mean along axis 1 with keepDims: [2,1] -> [[2], [5]]
	wantShape := []int{2, 1}
	if len(out.Shape()) != len(wantShape) {
		t.Fatalf("shape = %v, want %v", out.Shape(), wantShape)
	}
	for i, d := range wantShape {
		if out.Shape()[i] != d {
			t.Fatalf("shape = %v, want %v", out.Shape(), wantShape)
		}
	}

	got := out.Data()
	if got[0] < 1.99 || got[0] > 2.01 {
		t.Errorf("mean[0] = %v, want 2.0", got[0])
	}
	if got[1] < 4.99 || got[1] > 5.01 {
		t.Errorf("mean[1] = %v, want 5.0", got[1])
	}
}

// ---------------------------------------------------------------------------
// Pad: Attributes, OutputShape, BuildPad with constant_value
// ---------------------------------------------------------------------------

func TestPad_Attributes_OutputShape(t *testing.T) {
	engine := makeEngine()
	p := NewPad[float32](engine, []int64{1, 0, 1, 0}, 0)

	// Before Forward, OutputShape is nil.
	if p.OutputShape() != nil {
		t.Error("OutputShape before Forward should be nil")
	}

	attrs := p.Attributes()
	if attrs == nil {
		t.Fatal("Attributes should not be nil")
	}
	pads, ok := attrs["pads"].([]int64)
	if !ok {
		t.Fatal("pads attribute missing")
	}
	if len(pads) != 4 {
		t.Errorf("pads length = %d, want 4", len(pads))
	}

	// After Forward, OutputShape is populated.
	input, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	out, err := p.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Pad Forward: %v", err)
	}

	os := p.OutputShape()
	if os == nil {
		t.Fatal("OutputShape after Forward should not be nil")
	}
	if os[0] != out.Shape()[0] || os[1] != out.Shape()[1] {
		t.Errorf("OutputShape = %v, want %v", os, out.Shape())
	}
}

func TestBuildPad_WithConstantValue(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	tests := []struct {
		name  string
		attrs map[string]interface{}
	}{
		{
			name: "float64_constant",
			attrs: map[string]interface{}{
				"pads":           []int64{1, 0, 1, 0},
				"constant_value": float64(3.14),
			},
		},
		{
			name: "float32_constant",
			attrs: map[string]interface{}{
				"pads":           []int64{0, 1, 0, 1},
				"constant_value": float32(2.71),
			},
		},
		{
			name: "no_constant_value",
			attrs: map[string]interface{}{
				"pads": []int64{0, 0, 0, 0},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := BuildPad[float32](engine, ops, "pad", nil, tc.attrs)
			if err != nil {
				t.Fatalf("BuildPad: %v", err)
			}
			if node == nil {
				t.Fatal("BuildPad returned nil")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Resize: Attributes, OutputShape, BuildResize float32 scales, default mode
// ---------------------------------------------------------------------------

func TestResize_Attributes_OutputShape(t *testing.T) {
	ops := makeOps()
	engine := makeEngine()
	r := NewResize[float32](engine, ops, "nearest", []float64{1, 1, 2, 2}, nil)

	// Before Forward, OutputShape is nil.
	if r.OutputShape() != nil {
		t.Error("OutputShape before Forward should be nil")
	}

	attrs := r.Attributes()
	if attrs == nil {
		t.Fatal("Attributes should not be nil")
	}
	if attrs["mode"] != "nearest" {
		t.Errorf("mode = %v, want nearest", attrs["mode"])
	}

	// After Forward, OutputShape is populated.
	x, _ := tensor.New[float32]([]int{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	_, err := r.Forward(context.Background(), x)
	if err != nil {
		t.Fatalf("Resize Forward: %v", err)
	}
	os := r.OutputShape()
	if os == nil {
		t.Fatal("OutputShape after Forward should not be nil")
	}
}

func TestNewResize_DefaultMode(t *testing.T) {
	ops := makeOps()
	engine := makeEngine()
	// Empty mode should default to "nearest".
	r := NewResize[float32](engine, ops, "", []float64{1, 1, 2, 2}, nil)
	attrs := r.Attributes()
	if attrs["mode"] != "nearest" {
		t.Errorf("mode = %v, want nearest", attrs["mode"])
	}
}

func TestBuildResize_Float32Scales(t *testing.T) {
	ops := makeOps()
	engine := makeEngine()

	attrs := map[string]interface{}{
		"scales": []float32{1, 1, 2, 2},
	}
	node, err := BuildResize[float32](engine, ops, "resize", nil, attrs)
	if err != nil {
		t.Fatalf("BuildResize with float32 scales: %v", err)
	}
	if node == nil {
		t.Fatal("BuildResize returned nil")
	}
}

func TestResize_Non4DInput(t *testing.T) {
	ops := makeOps()
	engine := makeEngine()
	r := NewResize[float32](engine, ops, "nearest", []float64{1, 1, 2, 2}, nil)

	x, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	_, err := r.Forward(context.Background(), x)
	if err == nil {
		t.Error("expected error for non-4D input")
	}
}

// ---------------------------------------------------------------------------
// LMHead: OutputShape with tied weight
// ---------------------------------------------------------------------------

func TestLMHead_OutputShape_Tied(t *testing.T) {
	engine := makeEngine()
	embedWeight, _ := tensor.New[float32]([]int{4, 8}, make([]float32, 32))
	lmHead := NewTiedLMHead[float32](engine, embedWeight)

	// Tied LMHead returns nil for OutputShape.
	if os := lmHead.OutputShape(); os != nil {
		t.Errorf("TiedLMHead.OutputShape() = %v, want nil", os)
	}
}

func TestLMHead_OutputShape_Owned(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	lmHead, err := NewLMHead[float32](engine, ops, 8, 100)
	if err != nil {
		t.Fatalf("NewLMHead: %v", err)
	}
	os := lmHead.OutputShape()
	if os == nil {
		t.Fatal("Owned LMHead.OutputShape() should not be nil")
	}
}

// ---------------------------------------------------------------------------
// FFN: WithFFNNoBias option
// ---------------------------------------------------------------------------

func TestFFN_WithNoBias(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	f, err := NewFFN[float32]("ffn_nobias", engine, ops, 4, 8, 4, WithFFNNoBias[float32]())
	if err != nil {
		t.Fatalf("NewFFN with WithFFNNoBias: %v", err)
	}

	// Verify it can forward.
	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	out, err := f.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("FFN Forward no bias: %v", err)
	}
	if out == nil {
		t.Fatal("output nil")
	}
}

// ---------------------------------------------------------------------------
// BuildReduceMean: various attribute formats
// ---------------------------------------------------------------------------

func TestBuildReduceMean_Attributes(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	tests := []struct {
		name  string
		attrs map[string]any
	}{
		{
			name:  "axes_as_any_slice",
			attrs: map[string]any{"axes": []any{int64(0), int64(1)}, "keepdims": int64(1)},
		},
		{
			name:  "keepdims_false_int64",
			attrs: map[string]any{"axes": []int64{0}, "keepdims": int64(0)},
		},
		{
			name:  "keepdims_bool",
			attrs: map[string]any{"axes": []int64{0}, "keepdims": false},
		},
		{
			name:  "no_axes",
			attrs: map[string]any{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node, err := BuildReduceMean[float32](engine, ops, "rm", nil, tc.attrs)
			if err != nil {
				t.Fatalf("BuildReduceMean: %v", err)
			}
			if node == nil {
				t.Fatal("BuildReduceMean returned nil")
			}
		})
	}
}
