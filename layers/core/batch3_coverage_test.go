package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// ---------- Cos/Sin (additional coverage: Backward, Build, metadata) ----------

func TestCosSinBackwardBuildMetadata(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	trigTests := []struct {
		name    string
		node    interface {
			OpType() string
			Attributes() map[string]any
			OutputShape() []int
			Backward(context.Context, types.BackwardMode, *tensor.TensorNumeric[float32], ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error)
		}
		wantOp string
	}{
		{"Cos", &Cos[float32]{engine: engine}, "Cos"},
		{"Sin", &Sin[float32]{engine: engine}, "Sin"},
	}

	for _, tt := range trigTests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.node.Backward(ctx, types.FullBackprop, nil)
			if err == nil {
				t.Error("Backward should return error")
			}
			if op := tt.node.OpType(); op != tt.wantOp {
				t.Errorf("OpType = %q, want %q", op, tt.wantOp)
			}
			if tt.node.Attributes() != nil {
				t.Error("Attributes should be nil")
			}
			if tt.node.OutputShape() != nil {
				t.Error("OutputShape should be nil")
			}
		})
	}

	// BuildCos
	builtCos, err := BuildCos(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildCos: %v", err)
	}
	if builtCos.OpType() != "Cos" {
		t.Errorf("BuildCos OpType = %q", builtCos.OpType())
	}

	// BuildSin
	builtSin, err := BuildSin(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildSin: %v", err)
	}
	if builtSin.OpType() != "Sin" {
		t.Errorf("BuildSin OpType = %q", builtSin.OpType())
	}
}

// ---------- Gemm (additional coverage: Backward, Build, metadata, edge cases) ----------

func TestGemmBackwardBuildMetadata(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	g := &Gemm[float32]{engine: engine, ops: ops, alpha: 1, beta: 1}

	// Backward returns error
	_, err := g.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	attrs := g.Attributes()
	if attrs == nil {
		t.Fatal("Attributes should not be nil")
	}
	if g.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if g.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildGemm with various attribute types
	built, err := BuildGemm(engine, ops, "test", nil, map[string]any{
		"alpha":  float64(2.0),
		"beta":   float32(1.0),
		"transA": int64(1),
		"transB": true,
	})
	if err != nil {
		t.Fatalf("BuildGemm: %v", err)
	}
	if built.OpType() != "Gemm" {
		t.Errorf("built OpType = %q", built.OpType())
	}

	// BuildGemm with no attributes (defaults)
	_, err = BuildGemm(engine, ops, "test", nil, map[string]any{})
	if err != nil {
		t.Fatalf("BuildGemm defaults: %v", err)
	}

	// Gemm with scalar C broadcast
	t.Run("scalar_C", func(t *testing.T) {
		node := &Gemm[float32]{engine: engine, ops: ops, alpha: 1, beta: 1}
		a := makeTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
		b := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		c := makeTensor(t, []int{1}, []float32{10})

		out, err := node.Forward(ctx, a, b, c)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		want := []float32{11, 12, 13, 14}
		for i, v := range out.Data() {
			if math.Abs(float64(v-want[i])) > 1e-5 {
				t.Errorf("data[%d] = %v, want %v", i, v, want[i])
			}
		}
	})

	// Gemm with full matrix C
	t.Run("full_matrix_C", func(t *testing.T) {
		node := &Gemm[float32]{engine: engine, ops: ops, alpha: 1, beta: 1}
		a := makeTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
		b := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		c := makeTensor(t, []int{2, 2}, []float32{10, 20, 30, 40})

		out, err := node.Forward(ctx, a, b, c)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		want := []float32{11, 22, 33, 44}
		for i, v := range out.Data() {
			if math.Abs(float64(v-want[i])) > 1e-5 {
				t.Errorf("data[%d] = %v, want %v", i, v, want[i])
			}
		}
	})

	// Gemm: non-2D input error
	t.Run("non_2D_error", func(t *testing.T) {
		node := &Gemm[float32]{engine: engine, ops: ops, alpha: 1, beta: 0}
		a := makeTensor(t, []int{2, 2, 2}, make([]float32, 8))
		b := makeTensor(t, []int{2, 2}, make([]float32, 4))
		_, err := node.Forward(ctx, a, b)
		if err == nil {
			t.Error("Forward with 3D input should error")
		}
	})

	// Gemm: inner dim mismatch
	t.Run("dim_mismatch", func(t *testing.T) {
		node := &Gemm[float32]{engine: engine, ops: ops, alpha: 1, beta: 0}
		a := makeTensor(t, []int{2, 3}, make([]float32, 6))
		b := makeTensor(t, []int{4, 2}, make([]float32, 8))
		_, err := node.Forward(ctx, a, b)
		if err == nil {
			t.Error("Forward with dim mismatch should error")
		}
	})

	// Gemm: incompatible C shape
	t.Run("incompatible_C", func(t *testing.T) {
		node := &Gemm[float32]{engine: engine, ops: ops, alpha: 1, beta: 1}
		a := makeTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
		b := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		c := makeTensor(t, []int{3}, []float32{1, 2, 3})
		_, err := node.Forward(ctx, a, b, c)
		if err == nil {
			t.Error("Forward with incompatible C shape should error")
		}
	})
}

// ---------- Squeeze (additional coverage: Backward, Build, edge cases) ----------

func TestSqueezeBackwardBuildMetadata(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	s := &Squeeze[float32]{engine: engine, axes: []int{0}}

	// Backward returns error
	_, err := s.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	attrs := s.Attributes()
	if attrs == nil {
		t.Fatal("Attributes should not be nil")
	}
	if s.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if s.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildSqueeze with []int64
	built, err := BuildSqueeze(engine, ops, "test", nil, map[string]any{
		"axes": []int64{0},
	})
	if err != nil {
		t.Fatalf("BuildSqueeze []int64: %v", err)
	}
	if built.OpType() != "Squeeze" {
		t.Errorf("built OpType = %q", built.OpType())
	}

	// BuildSqueeze with []any
	_, err = BuildSqueeze(engine, ops, "test", nil, map[string]any{
		"axes": []any{int64(1)},
	})
	if err != nil {
		t.Fatalf("BuildSqueeze []any: %v", err)
	}

	// BuildSqueeze with unsupported axes type
	_, err = BuildSqueeze(engine, ops, "test", nil, map[string]any{
		"axes": "invalid",
	})
	if err == nil {
		t.Error("BuildSqueeze with invalid axes type should error")
	}

	// BuildSqueeze with no attributes
	_, err = BuildSqueeze(engine, ops, "test", nil, map[string]any{})
	if err != nil {
		t.Fatalf("BuildSqueeze no attrs: %v", err)
	}

	// Squeeze with axes from second input (opset 13+)
	t.Run("axes_from_input", func(t *testing.T) {
		sq := &Squeeze[float32]{engine: engine}
		input := makeTensor(t, []int{1, 3, 1}, make([]float32, 3))
		axesTensor := makeTensor(t, []int{1}, []float32{0})
		out, err := sq.Forward(ctx, input, axesTensor)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		gotShape := out.Shape()
		if len(gotShape) != 2 || gotShape[0] != 3 || gotShape[1] != 1 {
			t.Errorf("shape = %v, want [3 1]", gotShape)
		}
	})

	// Squeeze: squeeze all size-1 dims
	t.Run("squeeze_all", func(t *testing.T) {
		sq := &Squeeze[float32]{engine: engine}
		input := makeTensor(t, []int{1, 3, 1}, make([]float32, 3))
		out, err := sq.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		gotShape := out.Shape()
		if len(gotShape) != 1 || gotShape[0] != 3 {
			t.Errorf("shape = %v, want [3]", gotShape)
		}
	})

	// Squeeze: axis out of range
	t.Run("axis_out_of_range", func(t *testing.T) {
		sq := &Squeeze[float32]{engine: engine, axes: []int{5}}
		input := makeTensor(t, []int{1, 3}, make([]float32, 3))
		_, err := sq.Forward(ctx, input)
		if err == nil {
			t.Error("Forward with out-of-range axis should error")
		}
	})

	// Squeeze: dim not size 1
	t.Run("dim_not_1", func(t *testing.T) {
		sq := &Squeeze[float32]{engine: engine, axes: []int{1}}
		input := makeTensor(t, []int{1, 3}, make([]float32, 3))
		_, err := sq.Forward(ctx, input)
		if err == nil {
			t.Error("Forward with non-1 dim should error")
		}
	})

	// Squeeze: wrong number of inputs
	t.Run("wrong_inputs", func(t *testing.T) {
		sq := &Squeeze[float32]{engine: engine}
		_, err := sq.Forward(ctx)
		if err == nil {
			t.Error("Forward with 0 inputs should error")
		}
	})

	// Squeeze: negative axis
	t.Run("negative_axis", func(t *testing.T) {
		sq := &Squeeze[float32]{engine: engine, axes: []int{-1}}
		input := makeTensor(t, []int{3, 1}, make([]float32, 3))
		out, err := sq.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		gotShape := out.Shape()
		if len(gotShape) != 1 || gotShape[0] != 3 {
			t.Errorf("shape = %v, want [3]", gotShape)
		}
	})
}

// ---------- Tile (additional coverage: Backward, Build, metadata, errors) ----------

func TestTileBackwardBuildMetadata(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	ti := &Tile[float32]{engine: engine}

	// Backward returns error
	_, err := ti.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// Metadata
	if op := ti.OpType(); op != "Tile" {
		t.Errorf("OpType = %q, want %q", op, "Tile")
	}
	if ti.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if ti.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if ti.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildTile
	built, err := BuildTile(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildTile: %v", err)
	}
	if built.OpType() != "Tile" {
		t.Errorf("built OpType = %q", built.OpType())
	}

	// Tile: wrong number of inputs
	_, err = ti.Forward(ctx, makeTensor(t, []int{2}, []float32{1, 2}))
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Tile: repeats length mismatch
	t.Run("repeats_mismatch", func(t *testing.T) {
		input := makeTensor(t, []int{2, 3}, make([]float32, 6))
		repeats := makeTensor(t, []int{3}, []float32{1, 2, 3})
		_, err := ti.Forward(ctx, input, repeats)
		if err == nil {
			t.Error("Forward with mismatched repeats should error")
		}
	})

	// Tile: zero repeat
	t.Run("zero_repeat", func(t *testing.T) {
		input := makeTensor(t, []int{2}, []float32{1, 2})
		repeats := makeTensor(t, []int{1}, []float32{0})
		_, err := ti.Forward(ctx, input, repeats)
		if err == nil {
			t.Error("Forward with zero repeat should error")
		}
	})

	// Tile: normal 2D tiling
	t.Run("tile_2d", func(t *testing.T) {
		input := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		repeats := makeTensor(t, []int{2}, []float32{1, 2})
		out, err := ti.Forward(ctx, input, repeats)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		want := []float32{1, 2, 1, 2, 3, 4, 3, 4}
		for i, v := range out.Data() {
			if v != want[i] {
				t.Errorf("data[%d] = %v, want %v", i, v, want[i])
			}
		}
	})
}

// ---------- Max (additional coverage: Backward, Build, metadata) ----------

func TestMaxBackwardBuildMetadata(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	m := &Max[float32]{engine: engine}

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

	// BuildMax
	built, err := BuildMax(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildMax: %v", err)
	}
	if built.OpType() != "Max" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Slice (tensorToInt64 coverage) ----------

func TestSliceTensorToInt64(t *testing.T) {
	in := makeTensor(t, []int{3}, []float32{1, 2, 3})
	got := tensorToInt64(in)
	want := []int64{1, 2, 3}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("got[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// ---------- Slice Forward with dynamic inputs ----------

func TestSliceForwardDynamic(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	// Slice with no static starts/ends -- all provided via inputs
	s := NewSlice[float32](engine, nil, nil, nil, nil)

	// inputs: data, starts, ends, axes, steps
	data := makeTensor(t, []int{5}, []float32{10, 20, 30, 40, 50})
	starts := makeTensor(t, []int{1}, []float32{1})
	ends := makeTensor(t, []int{1}, []float32{4})
	axes := makeTensor(t, []int{1}, []float32{0})
	steps := makeTensor(t, []int{1}, []float32{1})

	out, err := s.Forward(ctx, data, starts, ends, axes, steps)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	want := []float32{20, 30, 40}
	got := out.Data()
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("data[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// ---------- Slice extractInt64Slice coverage ----------

func TestSliceExtractInt64Slice(t *testing.T) {
	// Missing key
	got := extractInt64Slice(map[string]any{}, "missing")
	if got != nil {
		t.Errorf("expected nil for missing key, got %v", got)
	}

	// Wrong type
	got = extractInt64Slice(map[string]any{"key": "not_int64"}, "key")
	if got != nil {
		t.Errorf("expected nil for wrong type, got %v", got)
	}

	// Correct type
	got = extractInt64Slice(map[string]any{"key": []int64{1, 2, 3}}, "key")
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Errorf("expected [1 2 3], got %v", got)
	}
}

// ---------- Where: size mismatch errors ----------

func TestWhereSizeMismatch(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()
	w := &Where[float32]{engine: engine}

	// cond and x have different sizes (non-scalar)
	cond := makeTensor(t, []int{3}, []float32{1, 0, 1})
	x := makeTensor(t, []int{4}, []float32{1, 2, 3, 4})
	y := makeTensor(t, []int{3}, []float32{-1, -2, -3})
	_, err := w.Forward(ctx, cond, x, y)
	if err == nil {
		t.Error("Forward with mismatched cond/x should error")
	}

	// cond and y have different sizes (non-scalar)
	x2 := makeTensor(t, []int{3}, []float32{1, 2, 3})
	y2 := makeTensor(t, []int{4}, []float32{-1, -2, -3, -4})
	_, err = w.Forward(ctx, cond, x2, y2)
	if err == nil {
		t.Error("Forward with mismatched cond/y should error")
	}
}

// ---------- Where: y shape larger than x ----------

func TestWhereYLargerThanX(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()
	w := &Where[float32]{engine: engine}

	cond := makeTensor(t, []int{1}, []float32{0})
	x := makeTensor(t, []int{1}, []float32{99})
	y := makeTensor(t, []int{3}, []float32{-1, -2, -3})

	out, err := w.Forward(ctx, cond, x, y)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	// cond=0, so all take from y
	want := []float32{-1, -2, -3}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("data[%d] = %v, want %v", i, v, want[i])
		}
	}
}

// ---------- GlobalAvgPool metadata ----------

func TestGlobalAvgPoolMetadata(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	g := NewGlobalAveragePool(engine, ops)

	if g.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
}

// ---------- TopK metadata ----------

func TestTopKMetadata(t *testing.T) {
	engine := makeEngine()
	tk := &TopK[float32]{engine: engine, k: 3}

	if tk.Attributes() == nil {
		t.Error("Attributes should not be nil")
	}
	if tk.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
}

// ---------- Slice OutputShape ----------

func TestSliceOutputShape(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	s := NewSlice[float32](engine, []int64{0}, []int64{2}, []int64{0}, nil)
	input := makeTensor(t, []int{4}, []float32{10, 20, 30, 40})
	_, err := s.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	os := s.OutputShape()
	if os == nil {
		t.Fatal("OutputShape should not be nil after Forward")
	}
	if len(os) != 1 || os[0] != 2 {
		t.Errorf("OutputShape = %v, want [2]", os)
	}
}

// ---------- Constant Attributes coverage ----------

func TestConstantAttributesDtypes(t *testing.T) {
	// float32 tensor
	v32, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	c32 := &Constant[float32]{value: v32}
	attrs := c32.Attributes()
	if attrs["dtype"] != "float32" {
		t.Errorf("dtype = %v, want float32", attrs["dtype"])
	}
}
