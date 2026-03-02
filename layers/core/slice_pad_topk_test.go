package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func makeFloat32Engine() compute.Engine[float32] {
	return compute.NewCPUEngine[float32](numeric.Float32Ops{})
}

// ---------- Slice ----------

func TestSlice_Forward_1D(t *testing.T) {
	eng := makeFloat32Engine()
	// slice [2:5] of a 1D tensor of length 6
	s := NewSlice[float32](eng, []int64{2}, []int64{5}, nil, nil)

	input, _ := tensor.New[float32]([]int{6}, []float32{0, 1, 2, 3, 4, 5})
	out, err := s.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Slice.Forward failed: %v", err)
	}
	want := []float32{2, 3, 4}
	got := out.Data()
	if len(got) != len(want) {
		t.Fatalf("output len = %d, want %d", len(got), len(want))
	}
	for i, v := range want {
		if got[i] != v {
			t.Errorf("out[%d] = %f, want %f", i, got[i], v)
		}
	}
}

func TestSlice_Forward_2D_WithAxes(t *testing.T) {
	eng := makeFloat32Engine()
	// slice axis 1 from 1 to 3 in a [2,4] tensor
	s := NewSlice[float32](eng, []int64{1}, []int64{3}, []int64{1}, nil)

	data := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	input, _ := tensor.New[float32]([]int{2, 4}, data)
	out, err := s.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Slice.Forward 2D failed: %v", err)
	}
	if len(out.Shape()) != 2 || out.Shape()[0] != 2 || out.Shape()[1] != 2 {
		t.Errorf("output shape = %v, want [2 2]", out.Shape())
	}
}

func TestSlice_Forward_NegativeStart(t *testing.T) {
	eng := makeFloat32Engine()
	// slice [-2:] of a 1D tensor of length 5 => indices [3,4]
	s := NewSlice[float32](eng, []int64{-2}, []int64{5}, nil, nil)

	input, _ := tensor.New[float32]([]int{5}, []float32{10, 20, 30, 40, 50})
	out, err := s.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Slice.Forward neg start failed: %v", err)
	}
	got := out.Data()
	if len(got) != 2 || got[0] != 40 || got[1] != 50 {
		t.Errorf("slice [-2:] = %v, want [40 50]", got)
	}
}

func TestSlice_Forward_EndBeyondDim(t *testing.T) {
	eng := makeFloat32Engine()
	// end=100 clamped to dim size 4
	s := NewSlice[float32](eng, []int64{2}, []int64{100}, nil, nil)

	input, _ := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	out, err := s.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Slice clamped end failed: %v", err)
	}
	if len(out.Data()) != 2 {
		t.Errorf("expected 2 elements, got %d", len(out.Data()))
	}
}

func TestSlice_ForwardInputError(t *testing.T) {
	eng := makeFloat32Engine()
	s := NewSlice[float32](eng, []int64{0}, []int64{1}, nil, nil)
	_, err := s.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestSlice_Backward(t *testing.T) {
	eng := makeFloat32Engine()
	s := NewSlice[float32](eng, []int64{0}, []int64{2}, nil, nil)
	grads, err := s.Backward(context.Background(), types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("Slice.Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil grads")
	}
}

func TestSlice_OpType(t *testing.T) {
	eng := makeFloat32Engine()
	s := NewSlice[float32](eng, nil, nil, nil, nil)
	if s.OpType() != "Slice" {
		t.Errorf("OpType = %q, want Slice", s.OpType())
	}
}

func TestSlice_Attributes(t *testing.T) {
	eng := makeFloat32Engine()
	s := NewSlice[float32](eng, []int64{1}, []int64{3}, []int64{0}, nil)
	attrs := s.Attributes()
	if attrs == nil {
		t.Error("Attributes should not be nil")
	}
}

func TestSlice_Parameters(t *testing.T) {
	eng := makeFloat32Engine()
	s := NewSlice[float32](eng, nil, nil, nil, nil)
	if s.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestBuildSlice_NoAttributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	node, err := BuildSlice(eng, ops, "slice", nil, map[string]interface{}{})
	if err != nil {
		t.Fatalf("BuildSlice failed: %v", err)
	}
	if node.OpType() != "Slice" {
		t.Errorf("OpType = %q, want Slice", node.OpType())
	}
}

func TestBuildSlice_WithAttributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{
		"starts": []int64{1},
		"ends":   []int64{4},
		"axes":   []int64{0},
	}
	node, err := BuildSlice(eng, ops, "slice", nil, attrs)
	if err != nil {
		t.Fatalf("BuildSlice with attrs failed: %v", err)
	}
	if node.OpType() != "Slice" {
		t.Errorf("OpType = %q, want Slice", node.OpType())
	}
}

// ---------- Pad ----------

func TestPad_Forward_1D(t *testing.T) {
	eng := makeFloat32Engine()
	// Pad [1, 2] on a 1D tensor of size 3 -> output size 1+3+2=6
	p := NewPad[float32](eng, []int64{1, 2}, 0)

	input, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	out, err := p.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Pad.Forward failed: %v", err)
	}
	got := out.Data()
	want := []float32{0, 1, 2, 3, 0, 0}
	if len(got) != len(want) {
		t.Fatalf("output len = %d, want %d", len(got), len(want))
	}
	for i, v := range want {
		if got[i] != v {
			t.Errorf("out[%d] = %f, want %f", i, got[i], v)
		}
	}
}

func TestPad_Forward_2D(t *testing.T) {
	eng := makeFloat32Engine()
	// Pad pads=[0,0,1,1] on a [2,2] tensor: pad 0 rows begin, 0 rows end, 1 col begin, 1 col end
	p := NewPad[float32](eng, []int64{0, 1, 0, 1}, 0)

	input, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	out, err := p.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Pad.Forward 2D failed: %v", err)
	}
	// Output shape should be [2, 4] (0+2+0 rows, 1+2+1 cols)
	if len(out.Shape()) != 2 || out.Shape()[0] != 2 || out.Shape()[1] != 4 {
		t.Errorf("output shape = %v, want [2 4]", out.Shape())
	}
}

func TestPad_Forward_WithConstantValue(t *testing.T) {
	eng := makeFloat32Engine()
	p := NewPad[float32](eng, []int64{2, 0}, -1)

	input, _ := tensor.New[float32]([]int{3}, []float32{10, 20, 30})
	out, err := p.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Pad constant value failed: %v", err)
	}
	got := out.Data()
	want := []float32{-1, -1, 10, 20, 30}
	if len(got) != len(want) {
		t.Fatalf("output len = %d, want %d", len(got), len(want))
	}
	for i, v := range want {
		if got[i] != v {
			t.Errorf("out[%d] = %f, want %f", i, got[i], v)
		}
	}
}

func TestPad_ForwardInputError(t *testing.T) {
	eng := makeFloat32Engine()
	p := NewPad[float32](eng, []int64{1, 1}, 0)
	_, err := p.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestPad_PadsMismatch(t *testing.T) {
	eng := makeFloat32Engine()
	// pads has 4 elements but input is 1D (expects 2)
	p := NewPad[float32](eng, []int64{1, 1, 2, 2}, 0)
	input, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	_, err := p.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected error for pads/ndim mismatch")
	}
}

func TestPad_Backward(t *testing.T) {
	eng := makeFloat32Engine()
	p := NewPad[float32](eng, []int64{1, 1}, 0)
	grads, err := p.Backward(context.Background(), types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("Pad.Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil grads")
	}
}

func TestPad_OpType(t *testing.T) {
	eng := makeFloat32Engine()
	p := NewPad[float32](eng, nil, 0)
	if p.OpType() != "Pad" {
		t.Errorf("OpType = %q, want Pad", p.OpType())
	}
}

func TestPad_Parameters(t *testing.T) {
	eng := makeFloat32Engine()
	p := NewPad[float32](eng, nil, 0)
	if p.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestBuildPad(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{
		"pads": []int64{1, 1},
	}
	node, err := BuildPad(eng, ops, "pad", nil, attrs)
	if err != nil {
		t.Fatalf("BuildPad failed: %v", err)
	}
	if node.OpType() != "Pad" {
		t.Errorf("OpType = %q, want Pad", node.OpType())
	}
}

func TestBuildPad_NoAttributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	node, err := BuildPad(eng, ops, "pad", nil, map[string]interface{}{})
	if err != nil {
		t.Fatalf("BuildPad no attrs failed: %v", err)
	}
	if node.OpType() != "Pad" {
		t.Errorf("OpType = %q, want Pad", node.OpType())
	}
}

// ---------- TopK ----------

func TestTopK_Forward_1D(t *testing.T) {
	eng := makeFloat32Engine()
	tk := NewTopK[float32](eng, 3, -1, true, true)

	input, _ := tensor.New[float32]([]int{5}, []float32{3, 1, 4, 1, 5})
	out, err := tk.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("TopK.Forward failed: %v", err)
	}
	got := out.Data()
	if len(got) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(got))
	}
	// Largest 3 values: 5, 4, 3 (sorted descending)
	if got[0] != 5 || got[1] != 4 || got[2] != 3 {
		t.Errorf("topk values = %v, want [5 4 3]", got)
	}
}

func TestTopK_Forward_SmallestFirst(t *testing.T) {
	eng := makeFloat32Engine()
	tk := NewTopK[float32](eng, 2, -1, false, true)

	input, _ := tensor.New[float32]([]int{4}, []float32{4, 2, 8, 1})
	out, err := tk.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("TopK.Forward (smallest) failed: %v", err)
	}
	got := out.Data()
	if len(got) != 2 {
		t.Fatalf("expected 2 elements, got %d", len(got))
	}
	// Smallest 2: 1, 2
	if got[0] != 1 || got[1] != 2 {
		t.Errorf("topk smallest values = %v, want [1 2]", got)
	}
}

func TestTopK_Forward_KLargerThanSize(t *testing.T) {
	eng := makeFloat32Engine()
	tk := NewTopK[float32](eng, 10, -1, true, true)

	input, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	out, err := tk.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("TopK large k failed: %v", err)
	}
	if len(out.Data()) != 3 {
		t.Errorf("expected 3 elements (clamped k), got %d", len(out.Data()))
	}
}

func TestTopK_ForwardInputError(t *testing.T) {
	eng := makeFloat32Engine()
	tk := NewTopK[float32](eng, 2, -1, true, true)
	_, err := tk.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestTopK_Backward(t *testing.T) {
	eng := makeFloat32Engine()
	tk := NewTopK[float32](eng, 2, -1, true, true)
	grads, err := tk.Backward(context.Background(), types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("TopK.Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil grads")
	}
}

func TestTopK_OpType(t *testing.T) {
	eng := makeFloat32Engine()
	tk := NewTopK[float32](eng, 2, -1, true, true)
	if tk.OpType() != "TopK" {
		t.Errorf("OpType = %q, want TopK", tk.OpType())
	}
}

func TestTopK_Parameters(t *testing.T) {
	eng := makeFloat32Engine()
	tk := NewTopK[float32](eng, 2, -1, true, true)
	if tk.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestBuildTopK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{"k": 3}
	node, err := BuildTopK(eng, ops, "topk", nil, attrs)
	if err != nil {
		t.Fatalf("BuildTopK failed: %v", err)
	}
	if node.OpType() != "TopK" {
		t.Errorf("OpType = %q, want TopK", node.OpType())
	}
}

func TestBuildTopK_MissingK(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	_, err := BuildTopK(eng, ops, "topk", nil, map[string]interface{}{})
	if err == nil {
		t.Error("expected error for missing k")
	}
}

func TestBuildTopK_WithAllAttributes(t *testing.T) {
	eng := makeFloat32Engine()
	ops := numeric.Float32Ops{}
	attrs := map[string]interface{}{
		"k":       int64(5),
		"axis":    int64(0),
		"largest": int64(0),
		"sorted":  int64(0),
	}
	node, err := BuildTopK(eng, ops, "topk", nil, attrs)
	if err != nil {
		t.Fatalf("BuildTopK with all attrs failed: %v", err)
	}
	if node.OpType() != "TopK" {
		t.Errorf("OpType = %q, want TopK", node.OpType())
	}
}
