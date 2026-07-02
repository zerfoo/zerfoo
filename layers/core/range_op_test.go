package core

import (
	"context"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
)

func TestRange_Forward(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()
	r := &Range[float32]{engine: engine}

	tests := []struct {
		name      string
		start     float32
		limit     float32
		delta     float32
		want      []float32
		wantShape []int
	}{
		{
			name:      "single element",
			start:     0, limit: 1, delta: 1,
			want:      []float32{0},
			wantShape: []int{1},
		},
		{
			name:      "ascending integers",
			start:     0, limit: 5, delta: 1,
			want:      []float32{0, 1, 2, 3, 4},
			wantShape: []int{5},
		},
		{
			name:      "ascending with step 2",
			start:     1, limit: 7, delta: 2,
			want:      []float32{1, 3, 5},
			wantShape: []int{3},
		},
		{
			name:      "fractional delta",
			start:     0, limit: 1, delta: 0.25,
			want:      []float32{0, 0.25, 0.5, 0.75},
			wantShape: []int{4},
		},
		{
			name:      "negative delta descending",
			start:     5, limit: 0, delta: -1,
			want:      []float32{5, 4, 3, 2, 1},
			wantShape: []int{5},
		},
		{
			name:      "negative delta descending fractional",
			start:     1, limit: -1, delta: -0.5,
			want:      []float32{1, 0.5, 0, -0.5},
			wantShape: []int{4},
		},
		{
			name:      "negative start ascending",
			start:     -3, limit: 3, delta: 2,
			want:      []float32{-3, -1, 1},
			wantShape: []int{3},
		},
		{
			name:      "empty range positive delta",
			start:     5, limit: 0, delta: 1,
			want:      []float32{},
			wantShape: []int{0},
		},
		{
			name:      "empty range negative delta",
			start:     0, limit: 5, delta: -1,
			want:      []float32{},
			wantShape: []int{0},
		},
		{
			name:      "start equals limit",
			start:     3, limit: 3, delta: 1,
			want:      []float32{},
			wantShape: []int{0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			start := makeTensor(t, []int{1}, []float32{tt.start})
			limit := makeTensor(t, []int{1}, []float32{tt.limit})
			delta := makeTensor(t, []int{1}, []float32{tt.delta})

			out, err := r.Forward(ctx, start, limit, delta)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			if got := out.Shape(); len(got) != 1 || got[0] != tt.wantShape[0] {
				t.Fatalf("Shape = %v, want %v", got, tt.wantShape)
			}

			if len(tt.want) > 0 {
				testutils.AssertFloat32SliceApproxEqual(t, tt.want, out.Data(), 1e-5, "output mismatch")
			} else if len(out.Data()) != 0 {
				t.Errorf("expected empty output, got %v", out.Data())
			}
		})
	}
}

func TestRange_Forward_ZeroDelta(t *testing.T) {
	engine := makeEngine()
	r := &Range[float32]{engine: engine}

	start := makeTensor(t, []int{1}, []float32{0})
	limit := makeTensor(t, []int{1}, []float32{5})
	delta := makeTensor(t, []int{1}, []float32{0})

	_, err := r.Forward(context.Background(), start, limit, delta)
	if err == nil {
		t.Fatal("expected error for zero delta, got nil")
	}
	if !strings.Contains(err.Error(), "zero") {
		t.Errorf("error should mention zero, got: %v", err)
	}
}

func TestRange_Forward_WrongInputCount(t *testing.T) {
	engine := makeEngine()
	r := &Range[float32]{engine: engine}
	ctx := context.Background()

	tests := []struct {
		name   string
		inputs int
	}{
		{"zero inputs", 0},
		{"one input", 1},
		{"two inputs", 2},
		{"four inputs", 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var inputs []*tensor.TensorNumeric[float32]
			for i := 0; i < tt.inputs; i++ {
				inputs = append(inputs, makeTensor(t, []int{1}, []float32{float32(i)}))
			}
			_, err := r.Forward(ctx, inputs...)
			if err == nil {
				t.Fatalf("expected error for %d inputs, got nil", tt.inputs)
			}
			if !strings.Contains(err.Error(), "3 inputs") {
				t.Errorf("error should mention 3 inputs, got: %v", err)
			}
		})
	}
}

func TestRange_Forward_NoPanic(t *testing.T) {
	engine := makeEngine()
	r := &Range[float32]{engine: engine}
	ctx := context.Background()

	// Various valid inputs should never panic.
	cases := []struct {
		start, limit, delta float32
	}{
		{0, 100, 1},
		{-50, 50, 0.5},
		{100, 0, -2},
		{0, 0.001, 0.001},
	}

	for _, c := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("panic on start=%v limit=%v delta=%v: %v", c.start, c.limit, c.delta, r)
				}
			}()
			start := makeTensor(t, []int{1}, []float32{c.start})
			limit := makeTensor(t, []int{1}, []float32{c.limit})
			delta := makeTensor(t, []int{1}, []float32{c.delta})
			_, _ = r.Forward(ctx, start, limit, delta)
		}()
	}
}

func TestRange_OpType(t *testing.T) {
	r := &Range[float32]{engine: makeEngine()}
	if got := r.OpType(); got != "Range" {
		t.Errorf("OpType() = %q, want %q", got, "Range")
	}
}

func TestRange_Attributes(t *testing.T) {
	r := &Range[float32]{engine: makeEngine()}
	if got := r.Attributes(); got != nil {
		t.Errorf("Attributes() = %v, want nil", got)
	}
}

func TestRange_OutputShape(t *testing.T) {
	r := &Range[float32]{engine: makeEngine()}
	if got := r.OutputShape(); got != nil {
		t.Errorf("OutputShape() = %v, want nil", got)
	}
}

func TestRange_Parameters(t *testing.T) {
	r := &Range[float32]{engine: makeEngine()}
	if got := r.Parameters(); got != nil {
		t.Errorf("Parameters() = %v, want nil", got)
	}
}

func TestRange_Backward(t *testing.T) {
	r := &Range[float32]{engine: makeEngine()}
	grad := makeTensor(t, []int{3}, []float32{1, 2, 3})

	_, err := r.Backward(context.Background(), types.FullBackprop, grad)
	if err == nil {
		t.Fatal("expected error from Backward, got nil")
	}
	if !strings.Contains(err.Error(), "not implemented") {
		t.Errorf("error should mention not implemented, got: %v", err)
	}
}

func TestRange_Forward_ZeroDimScalar(t *testing.T) {
	engine := makeEngine()
	r := &Range[float32]{engine: engine}
	ctx := context.Background()

	// 0-D tensors (shape []) are common in ONNX models for Range inputs.
	start, err := tensor.New([]int{}, []float32{0})
	if err != nil {
		t.Fatalf("create start: %v", err)
	}
	limit, err := tensor.New([]int{}, []float32{5})
	if err != nil {
		t.Fatalf("create limit: %v", err)
	}
	delta, err := tensor.New([]int{}, []float32{1})
	if err != nil {
		t.Fatalf("create delta: %v", err)
	}

	out, err := r.Forward(ctx, start, limit, delta)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	want := []float32{0, 1, 2, 3, 4}
	if got := out.Shape(); len(got) != 1 || got[0] != 5 {
		t.Fatalf("Shape = %v, want [5]", got)
	}
	testutils.AssertFloat32SliceApproxEqual(t, want, out.Data(), 1e-5, "output mismatch")
}

func TestRange_Forward_EmptyInput(t *testing.T) {
	engine := makeEngine()
	r := &Range[float32]{engine: engine}
	ctx := context.Background()

	// An input with no data should return an error, not panic.
	empty, err := tensor.New([]int{0}, []float32{})
	if err != nil {
		t.Fatalf("create empty tensor: %v", err)
	}
	limit := makeTensor(t, []int{1}, []float32{5})
	delta := makeTensor(t, []int{1}, []float32{1})

	_, err = r.Forward(ctx, empty, limit, delta)
	if err == nil {
		t.Fatal("expected error for empty start input, got nil")
	}
	if !strings.Contains(err.Error(), "no data") {
		t.Errorf("error should mention 'no data', got: %v", err)
	}
}

func TestBuildRange(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	node, err := BuildRange[float32](engine, ops, "test_range", nil, nil)
	if err != nil {
		t.Fatalf("BuildRange: %v", err)
	}
	if node == nil {
		t.Fatal("BuildRange returned nil node")
	}
	if got := node.OpType(); got != "Range" {
		t.Errorf("OpType() = %q, want %q", got, "Range")
	}
}
