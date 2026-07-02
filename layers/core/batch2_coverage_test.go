package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ---------- Range ----------

func TestRangeForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name     string
		start    float32
		limit    float32
		delta    float32
		wantData []float32
	}{
		{
			name:     "int_range_0_to_5",
			start:    0,
			limit:    5,
			delta:    1,
			wantData: []float32{0, 1, 2, 3, 4},
		},
		{
			name:     "float_range",
			start:    1.5,
			limit:    4,
			delta:    1,
			wantData: []float32{1.5, 2.5},
		},
		{
			name:     "negative_step",
			start:    5,
			limit:    0,
			delta:    -1,
			wantData: []float32{5, 4, 3, 2, 1},
		},
		{
			name:     "step_2",
			start:    0,
			limit:    10,
			delta:    3,
			wantData: []float32{0, 3, 6},
		},
		{
			name:     "empty_range",
			start:    5,
			limit:    0,
			delta:    1,
			wantData: []float32{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Range[float32]{engine: engine}
			start := makeTensor(t, []int{1}, []float32{tt.start})
			limit := makeTensor(t, []int{1}, []float32{tt.limit})
			delta := makeTensor(t, []int{1}, []float32{tt.delta})

			out, err := r.Forward(ctx, start, limit, delta)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			got := out.Data()
			if len(got) != len(tt.wantData) {
				t.Fatalf("len = %d, want %d", len(got), len(tt.wantData))
			}
			for i, v := range got {
				if math.Abs(float64(v-tt.wantData[i])) > 1e-5 {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	r := &Range[float32]{engine: engine}

	// Error: wrong number of inputs
	_, err := r.Forward(ctx, makeTensor(t, []int{1}, []float32{0}))
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Error: delta = 0
	start := makeTensor(t, []int{1}, []float32{0})
	limit := makeTensor(t, []int{1}, []float32{5})
	delta := makeTensor(t, []int{1}, []float32{0})
	_, err = r.Forward(ctx, start, limit, delta)
	if err == nil {
		t.Error("Forward with delta=0 should error")
	}

	// Backward returns error
	_, err = r.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if op := r.OpType(); op != "Range" {
		t.Errorf("OpType = %q, want %q", op, "Range")
	}
	if r.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if r.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if r.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildRange
	built, err := BuildRange(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildRange: %v", err)
	}
	if built.OpType() != "Range" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- ReduceMean ----------

func TestReduceMeanForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name      string
		shape     []int
		data      []float32
		axes      []int
		keepDims  bool
		wantShape []int
		wantData  []float32
	}{
		{
			name:      "single_axis_keepdims",
			shape:     []int{2, 3},
			data:      []float32{1, 2, 3, 4, 5, 6},
			axes:      []int{1},
			keepDims:  true,
			wantShape: []int{2, 1},
			wantData:  []float32{2, 5},
		},
		{
			name:      "single_axis_no_keepdims",
			shape:     []int{2, 3},
			data:      []float32{1, 2, 3, 4, 5, 6},
			axes:      []int{1},
			keepDims:  false,
			wantShape: []int{2},
			wantData:  []float32{2, 5},
		},
		{
			name:      "axis_0",
			shape:     []int{2, 3},
			data:      []float32{1, 2, 3, 4, 5, 6},
			axes:      []int{0},
			keepDims:  false,
			wantShape: []int{3},
			wantData:  []float32{2.5, 3.5, 4.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rm := &ReduceMean[float32]{engine: engine, axes: tt.axes, keepDims: tt.keepDims}
			input := makeTensor(t, tt.shape, tt.data)

			out, err := rm.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			gotShape := out.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("shape = %v, want %v", gotShape, tt.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], tt.wantShape[i])
				}
			}
			for i, v := range out.Data() {
				if math.Abs(float64(v-tt.wantData[i])) > 1e-5 {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	// Axes from second input (opset 18+)
	t.Run("axes_from_input", func(t *testing.T) {
		rm := &ReduceMean[float32]{engine: engine, axes: nil, keepDims: false}
		input := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
		axesTensor := makeTensor(t, []int{1}, []float32{1})

		out, err := rm.Forward(ctx, input, axesTensor)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		want := []float32{2, 5}
		for i, v := range out.Data() {
			if math.Abs(float64(v-want[i])) > 1e-5 {
				t.Errorf("data[%d] = %v, want %v", i, v, want[i])
			}
		}
	})

	rm := &ReduceMean[float32]{engine: engine, axes: []int{0}, keepDims: false}

	// Error: no inputs
	_, err := rm.Forward(ctx)
	if err == nil {
		t.Error("Forward with 0 inputs should error")
	}

	// Backward returns error
	_, err = rm.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if op := rm.OpType(); op != "ReduceMean" {
		t.Errorf("OpType = %q, want %q", op, "ReduceMean")
	}
	attrs := rm.Attributes()
	if attrs == nil {
		t.Fatal("Attributes should not be nil")
	}
	if rm.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if rm.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildReduceMean with []any axes and int64 keepdims
	built, err := BuildReduceMean(engine, ops, "test", nil, map[string]any{
		"axes":     []any{int64(1)},
		"keepdims": int64(0),
	})
	if err != nil {
		t.Fatalf("BuildReduceMean []any: %v", err)
	}
	if built.OpType() != "ReduceMean" {
		t.Errorf("built OpType = %q", built.OpType())
	}

	// BuildReduceMean with []int64 axes and bool keepdims
	_, err = BuildReduceMean(engine, ops, "test", nil, map[string]any{
		"axes":     []int64{0, 1},
		"keepdims": true,
	})
	if err != nil {
		t.Fatalf("BuildReduceMean []int64: %v", err)
	}

	// BuildReduceMean with no attributes
	_, err = BuildReduceMean(engine, ops, "test", nil, map[string]any{})
	if err != nil {
		t.Fatalf("BuildReduceMean no attrs: %v", err)
	}
}

// ---------- ScatterND ----------

func TestScatterNDForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name        string
		dataShape   []int
		data        []float32
		idxShape    []int
		indices     []float32
		updShape    []int
		updates     []float32
		wantData    []float32
	}{
		{
			name:      "basic_1d_scatter",
			dataShape: []int{5},
			data:      []float32{0, 0, 0, 0, 0},
			idxShape:  []int{2, 1},
			indices:   []float32{1, 3},
			updShape:  []int{2},
			updates:   []float32{10, 30},
			wantData:  []float32{0, 10, 0, 30, 0},
		},
		{
			name:      "overwrite_existing",
			dataShape: []int{4},
			data:      []float32{1, 2, 3, 4},
			idxShape:  []int{2, 1},
			indices:   []float32{0, 2},
			updShape:  []int{2},
			updates:   []float32{99, 88},
			wantData:  []float32{99, 2, 88, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &ScatterND[float32]{engine: engine}
			data := makeTensor(t, tt.dataShape, tt.data)
			indices := makeTensor(t, tt.idxShape, tt.indices)
			updates := makeTensor(t, tt.updShape, tt.updates)

			out, err := s.Forward(ctx, data, indices, updates)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			for i, v := range out.Data() {
				if v != tt.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	s := &ScatterND[float32]{engine: engine}

	// Error: wrong number of inputs
	_, err := s.Forward(ctx, makeTensor(t, []int{3}, []float32{1, 2, 3}))
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Backward returns error
	_, err = s.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if op := s.OpType(); op != "ScatterND" {
		t.Errorf("OpType = %q, want %q", op, "ScatterND")
	}
	if s.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if s.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if s.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildScatterND
	built, err := BuildScatterND(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildScatterND: %v", err)
	}
	if built.OpType() != "ScatterND" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Trilu ----------

func TestTriluForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name     string
		upper    bool
		k        *float32 // nil = no k input
		shape    []int
		data     []float32
		wantData []float32
	}{
		{
			name:  "upper_3x3",
			upper: true,
			shape: []int{3, 3},
			data:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			// Upper triangular: keep where c >= r
			wantData: []float32{1, 2, 3, 0, 5, 6, 0, 0, 9},
		},
		{
			name:  "lower_3x3",
			upper: false,
			shape: []int{3, 3},
			data:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			// Lower triangular: keep where c <= r
			wantData: []float32{1, 0, 0, 4, 5, 0, 7, 8, 9},
		},
		{
			name:  "upper_with_k1",
			upper: true,
			k:     ptrFloat32(1),
			shape: []int{3, 3},
			data:  []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			// Upper with k=1: keep where c >= r+1
			wantData: []float32{0, 2, 3, 0, 0, 6, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := &Trilu[float32]{engine: engine, upper: tt.upper}
			input := makeTensor(t, tt.shape, tt.data)

			var inputs []*tensor.TensorNumeric[float32]
			inputs = append(inputs, input)
			if tt.k != nil {
				kTensor := makeTensor(t, []int{1}, []float32{*tt.k})
				inputs = append(inputs, kTensor)
			}

			out, err := tr.Forward(ctx, inputs...)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			for i, v := range out.Data() {
				if v != tt.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	tr := &Trilu[float32]{engine: engine, upper: true}

	// Error: no inputs
	_, err := tr.Forward(ctx)
	if err == nil {
		t.Error("Forward with 0 inputs should error")
	}

	// Error: 1D input
	_, err = tr.Forward(ctx, makeTensor(t, []int{3}, []float32{1, 2, 3}))
	if err == nil {
		t.Error("Forward with 1D input should error")
	}

	// Backward returns error
	_, err = tr.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if op := tr.OpType(); op != "Trilu" {
		t.Errorf("OpType = %q, want %q", op, "Trilu")
	}
	attrs := tr.Attributes()
	if attrs == nil {
		t.Fatal("Attributes should not be nil")
	}
	if tr.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if tr.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildTrilu with int64 upper attribute
	built, err := BuildTrilu(engine, ops, "test", nil, map[string]any{"upper": int64(0)})
	if err != nil {
		t.Fatalf("BuildTrilu int64: %v", err)
	}
	if built.OpType() != "Trilu" {
		t.Errorf("built OpType = %q", built.OpType())
	}

	// BuildTrilu with bool upper attribute
	_, err = BuildTrilu(engine, ops, "test", nil, map[string]any{"upper": false})
	if err != nil {
		t.Fatalf("BuildTrilu bool: %v", err)
	}

	// BuildTrilu with default (no upper attribute)
	_, err = BuildTrilu(engine, ops, "test", nil, map[string]any{})
	if err != nil {
		t.Fatalf("BuildTrilu default: %v", err)
	}
}

// ---------- Where (additional coverage) ----------

func TestWhereBackwardAndBuild(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	w := &Where[float32]{engine: engine}

	// Error: wrong number of inputs
	_, err := w.Forward(ctx, makeTensor(t, []int{1}, []float32{1}))
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Backward returns error
	_, err = w.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if w.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if w.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if w.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildWhere
	built, err := BuildWhere(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildWhere: %v", err)
	}
	if built.OpType() != "Where" {
		t.Errorf("built OpType = %q", built.OpType())
	}

	// Scalar condition broadcast
	t.Run("scalar_cond", func(t *testing.T) {
		cond := makeTensor(t, []int{1}, []float32{1})
		x := makeTensor(t, []int{3}, []float32{10, 20, 30})
		y := makeTensor(t, []int{3}, []float32{-1, -2, -3})
		out, err := w.Forward(ctx, cond, x, y)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		want := []float32{10, 20, 30}
		for i, v := range out.Data() {
			if v != want[i] {
				t.Errorf("data[%d] = %v, want %v", i, v, want[i])
			}
		}
	})
}

// ---------- LessOrEqual (additional coverage) ----------

func TestLessOrEqualBackwardAndBuild(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	l := &LessOrEqual[float32]{engine: engine, ops: ops}

	// Backward returns error
	_, err := l.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if l.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if l.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if l.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildLessOrEqual
	built, err := BuildLessOrEqual(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildLessOrEqual: %v", err)
	}
	if built.OpType() != "LessOrEqual" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Mod (additional coverage) ----------

func TestModBackwardAndBuild(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	m := &Mod[float32]{engine: engine}

	// Backward returns error
	_, err := m.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if m.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if m.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if m.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildMod
	built, err := BuildMod(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildMod: %v", err)
	}
	if built.OpType() != "Mod" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Or (additional coverage) ----------

func TestOrBackwardAndBuild(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	o := &Or[float32]{engine: engine, ops: ops}

	// Backward returns error
	_, err := o.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if o.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if o.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if o.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildOr
	built, err := BuildOr(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildOr: %v", err)
	}
	if built.OpType() != "Or" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Helpers ----------

func ptrFloat32(v float32) *float32 { return &v }
