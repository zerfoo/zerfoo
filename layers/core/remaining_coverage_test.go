package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// ---------- Dense OpType ----------

func TestDense_OpType(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	d, err := NewDense[float32]("d", engine, ops, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if d.OpType() != "Dense" {
		t.Errorf("Dense OpType = %q, want %q", d.OpType(), "Dense")
	}
}

// ---------- FFN OpType, Attributes, OutputShape, Forward, Backward ----------

func TestFFN_Full(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	f, err := NewFFN[float32]("ffn", engine, ops, 4, 8, 4)
	if err != nil {
		t.Fatal(err)
	}

	// OpType
	if f.OpType() != "FFN" {
		t.Errorf("FFN OpType = %q, want %q", f.OpType(), "FFN")
	}

	// Attributes
	attr := f.Attributes()
	if attr == nil {
		t.Fatal("FFN Attributes should not be nil")
	}

	// OutputShape (delegates to w2)
	os := f.OutputShape()
	if os == nil {
		t.Fatal("FFN OutputShape should not be nil")
	}

	// Forward
	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	out, err := f.Forward(ctx, input)
	if err != nil {
		t.Fatalf("FFN Forward: %v", err)
	}
	if out == nil {
		t.Fatal("FFN Forward output nil")
	}

	// Forward error: wrong inputs
	_, err = f.Forward(ctx, input, input)
	if err == nil {
		t.Error("FFN Forward with 2 inputs should error")
	}

	// Backward
	gradShape := out.Shape()
	gradData := make([]float32, gradShape[0]*gradShape[1])
	for i := range gradData {
		gradData[i] = 0.1
	}
	grad := makeTensor(t, gradShape, gradData)
	grads, err := f.Backward(ctx, types.FullBackprop, grad)
	if err != nil {
		t.Fatalf("FFN Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("FFN Backward len = %d, want 1", len(grads))
	}
}

// ---------- FiLM ----------

func TestFiLM_Full(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	// NewFiLM
	film, err := NewFiLM[float32]("film", engine, ops, 4, 3)
	if err != nil {
		t.Fatal(err)
	}

	// OutputShape
	os := film.OutputShape()
	if os == nil {
		t.Fatal("FiLM OutputShape nil")
	}

	// OpType
	if film.OpType() != "FiLM" {
		t.Errorf("FiLM OpType = %q, want %q", film.OpType(), "FiLM")
	}

	// Attributes
	attr := film.Attributes()
	if attr == nil {
		t.Fatal("FiLM Attributes nil")
	}
	if attr["context_dim"] != 4 {
		t.Errorf("context_dim = %v, want 4", attr["context_dim"])
	}
	if attr["feature_dim"] != 3 {
		t.Errorf("feature_dim = %v, want 3", attr["feature_dim"])
	}

	// Forward
	feature := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
	ctxInput := makeTensor(t, []int{1, 4}, []float32{0.1, 0.2, 0.3, 0.4})
	out, err := film.Forward(ctx, feature, ctxInput)
	if err != nil {
		t.Fatalf("FiLM Forward: %v", err)
	}
	if out == nil {
		t.Fatal("FiLM Forward output nil")
	}

	// Backward
	grad := makeTensor(t, out.Shape(), make([]float32, out.Shape()[0]*out.Shape()[1]))
	for i := range grad.Data() {
		grad.Data()[i] = 1
	}
	grads, err := film.Backward(ctx, types.FullBackprop, grad, feature, ctxInput)
	if err != nil {
		t.Fatalf("FiLM Backward: %v", err)
	}
	if len(grads) != 2 {
		t.Fatalf("FiLM Backward len = %d, want 2", len(grads))
	}

	// Forward error: wrong input count
	_, err = film.Forward(ctx, feature)
	if err == nil {
		t.Error("FiLM Forward with 1 input should error")
	}

	// Backward error: wrong input count
	_, err = film.Backward(ctx, types.FullBackprop, grad, feature)
	if err == nil {
		t.Error("FiLM Backward with 1 input should error")
	}

	// NewFiLM error: empty name
	_, err = NewFiLM[float32]("", engine, ops, 4, 3)
	if err == nil {
		t.Error("NewFiLM with empty name should error")
	}

	// Parameters
	params := film.Parameters()
	if len(params) == 0 {
		t.Error("FiLM Parameters should not be empty")
	}

	// BuildFiLM
	tests := []struct {
		name  string
		attrs map[string]any
		want  string
	}{
		{
			name:  "success",
			attrs: map[string]any{"context_dim": 4, "feature_dim": 3},
		},
		{
			name:  "missing_context_dim",
			attrs: map[string]any{"feature_dim": 3},
			want:  "missing or invalid",
		},
		{
			name:  "missing_feature_dim",
			attrs: map[string]any{"context_dim": 4},
			want:  "missing or invalid",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := BuildFiLM(engine, ops, "test_film", nil, tt.attrs)
			if tt.want != "" {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("BuildFiLM: %v", err)
			}
			if node == nil {
				t.Fatal("BuildFiLM returned nil")
			}
		})
	}
}

// ---------- Linear OpType and Attributes ----------

func TestLinear_OpType_Attributes(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	l, err := NewLinear[float32]("lin", engine, ops, 4, 3)
	if err != nil {
		t.Fatal(err)
	}

	if l.OpType() != "Linear" {
		t.Errorf("Linear OpType = %q, want %q", l.OpType(), "Linear")
	}

	attr := l.Attributes()
	if attr == nil {
		t.Fatal("Linear Attributes nil")
	}
	if attr["input_features"] != 4 {
		t.Errorf("input_features = %v, want 4", attr["input_features"])
	}
	if attr["output_features"] != 3 {
		t.Errorf("output_features = %v, want 3", attr["output_features"])
	}
}

// ---------- MatMulNBits.String ----------

func TestMatMulNBits_String(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	// Create quantized weight: 2 rows, 2 cols (packed) -> actual 2x4
	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

	m, err := NewMatMulNBits("test", engine, ops, qw, scale, nil, 4, true)
	if err != nil {
		t.Fatal(err)
	}

	s := m.String()
	if s == "" {
		t.Error("String should not be empty")
	}
	if s == "MatMulNBits(uninitialized)" {
		t.Error("String should show dimensions")
	}

	// Test nil weights
	m2 := &MatMulNBits[float32]{}
	if m2.String() != "MatMulNBits(uninitialized)" {
		t.Errorf("nil weights String = %q, want uninitialized", m2.String())
	}
}

// ---------- Polynomial SetName ----------

func TestPolynomial_SetName(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	p, err := NewPolynomialExpansion("poly", engine, ops, 3)
	if err != nil {
		t.Fatal(err)
	}
	p.SetName("renamed")
	// SetName just sets the name field, verify no crash
}

// ---------- SpectralFingerprint error paths ----------

func TestSpectralFingerprint_Errors(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	// window <= 1
	_, err := NewSpectralFingerprint[float32](engine, ops, 1, 2)
	if err == nil {
		t.Error("window=1 should error")
	}

	// topK <= 0
	_, err = NewSpectralFingerprint[float32](engine, ops, 4, 0)
	if err == nil {
		t.Error("topK=0 should error")
	}

	// Forward: 2+ inputs
	sp, _ := NewSpectralFingerprint[float32](engine, ops, 4, 2)
	a := makeTensor(t, []int{1, 4}, make([]float32, 4))
	_, err = sp.Forward(context.Background(), a, a)
	if err == nil {
		t.Error("Forward with 2 inputs should error")
	}

	// Forward: non-2D input
	b := makeTensor(t, []int{4}, make([]float32, 4))
	_, err = sp.Forward(context.Background(), b)
	if err == nil {
		t.Error("Forward with 1D input should error")
	}

	// Forward: window too small
	c := makeTensor(t, []int{1, 2}, make([]float32, 2))
	_, err = sp.Forward(context.Background(), c)
	if err == nil {
		t.Error("Forward with small window should error")
	}

	// Forward: input larger than window (triggers start offset)
	sp2, _ := NewSpectralFingerprint[float32](engine, ops, 3, 2)
	d := makeTensor(t, []int{1, 5}, []float32{1, 2, 3, 4, 5})
	out, err := sp2.Forward(context.Background(), d)
	if err != nil {
		t.Fatalf("Forward with larger input: %v", err)
	}
	if out == nil {
		t.Fatal("output nil")
	}
}

// ---------- Panic recovery tests ----------

func TestMul_Forward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	m := NewMul(engine)
	a := makeTensor(t, []int{2}, []float32{1, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Mul Forward with 1 input should panic")
		}
	}()
	_, _ = m.Forward(context.Background(), a)
}

func TestMul_Backward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	m := NewMul(engine)
	g := makeTensor(t, []int{2}, []float32{1, 1})
	a := makeTensor(t, []int{2}, []float32{1, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Mul Backward with 1 input should panic")
		}
	}()
	_, _ = m.Backward(context.Background(), types.FullBackprop, g, a)
}

func TestSub_Backward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	s := NewSub(engine)
	g := makeTensor(t, []int{2}, []float32{1, 1})
	a := makeTensor(t, []int{2}, []float32{1, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Sub Backward with 1 input should panic")
		}
	}()
	_, _ = s.Backward(context.Background(), types.FullBackprop, g, a)
}

func TestReshape_Forward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	r := NewReshape(engine, []int{2})
	a := makeTensor(t, []int{2}, []float32{1, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Reshape Forward with 2 inputs should panic")
		}
	}()
	_, _ = r.Forward(context.Background(), a, a)
}

func TestReshape_Backward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	r := NewReshape(engine, []int{2})
	g := makeTensor(t, []int{2}, []float32{1, 1})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Reshape Backward with 0 inputs should panic")
		}
	}()
	_, _ = r.Backward(context.Background(), types.FullBackprop, g)
}

func TestCast_Forward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	c := NewCast(engine)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Cast Forward with 0 inputs should panic")
		}
	}()
	_, _ = c.Forward(context.Background())
}

func TestCast_Backward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	c := NewCast(engine)
	g := makeTensor(t, []int{2}, []float32{1, 1})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Cast Backward with 0 inputs should panic")
		}
	}()
	_, _ = c.Backward(context.Background(), types.FullBackprop, g)
}

func TestConcat_Forward_PanicNoInputs(t *testing.T) {
	engine := makeEngine()
	c := NewConcat[float32](engine, 0)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Concat Forward with 0 inputs should panic")
		}
	}()
	_, _ = c.Forward(context.Background())
}

func TestConcat_Backward_PanicNoInputs(t *testing.T) {
	engine := makeEngine()
	c := NewConcat[float32](engine, 0)
	g := makeTensor(t, []int{2}, []float32{1, 1})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Concat Backward with 0 inputs should panic")
		}
	}()
	_, _ = c.Backward(context.Background(), types.FullBackprop, g)
}

func TestUnsqueeze_Forward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	u := NewUnsqueeze(engine, []int{0})
	a := makeTensor(t, []int{2}, []float32{1, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Unsqueeze Forward with 2 inputs should panic")
		}
	}()
	_, _ = u.Forward(context.Background(), a, a)
}

func TestUnsqueeze_Backward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	u := NewUnsqueeze(engine, []int{0})
	g := makeTensor(t, []int{1, 2}, []float32{1, 2})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Unsqueeze Backward with 0 inputs should panic")
		}
	}()
	_, _ = u.Backward(context.Background(), types.FullBackprop, g)
}

func TestMatMul_Backward_PanicWrongInputs(t *testing.T) {
	engine := makeEngine()
	m := NewMatMul(engine)
	g := makeTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})

	defer func() {
		if r := recover(); r == nil {
			t.Error("MatMul Backward with 0 inputs should panic")
		}
	}()
	_, _ = m.Backward(context.Background(), types.FullBackprop, g)
}

// ---------- Constant Attributes dtype branches ----------

func TestConstant_Attributes_DTypes(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	// float32 is already covered, test that it works
	val := makeTensor(t, []int{2}, []float32{1, 2})
	c, _ := NewConstant("test", engine, ops, val)
	attr := c.Attributes()
	if attr["dtype"] != "float32" {
		t.Errorf("dtype = %v, want float32", attr["dtype"])
	}
}

// ---------- RotaryEmbedding Forward re-init path ----------

func TestRotaryEmbedding_ReInit(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	re := NewRotaryEmbedding[float32](engine)

	// First forward initializes inner with seqLen=2, headDim=4
	input1 := makeTensor(t, []int{1, 2, 4}, make([]float32, 8))
	_, err := re.Forward(ctx, input1)
	if err != nil {
		t.Fatalf("Forward 1: %v", err)
	}

	// Second forward with longer sequence should re-init
	input2 := makeTensor(t, []int{1, 4, 4}, make([]float32, 16))
	_, err = re.Forward(ctx, input2)
	if err != nil {
		t.Fatalf("Forward 2 (re-init): %v", err)
	}
}

// ---------- MatMulNBits Backward ----------

func TestMatMulNBits_Backward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	// Create quantized weight: 2 rows, 2 cols (packed) -> actual 2x4
	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

	m, err := NewMatMulNBits("test", engine, ops, qw, scale, nil, 4, true)
	if err != nil {
		t.Fatal(err)
	}

	// Forward first
	input := makeTensor(t, []int{1, 2}, []float32{1, 2})
	out, err := m.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Backward
	grad := makeTensor(t, out.Shape(), make([]float32, out.Shape()[0]*out.Shape()[1]))
	for i := range grad.Data() {
		grad.Data()[i] = 1
	}
	grads, err := m.Backward(ctx, types.FullBackprop, grad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("Backward len = %d, want 1", len(grads))
	}

	// Backward error: wrong inputs
	_, err = m.Backward(ctx, types.FullBackprop, grad, input, input)
	if err == nil {
		t.Error("Backward with 2 inputs should error")
	}
}

// ---------- MatMulNBits Forward errors ----------

func TestMatMulNBits_ForwardErrors(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

	m, _ := NewMatMulNBits("test", engine, ops, qw, scale, nil, 4, true)

	// Forward: wrong input count
	_, err := m.Forward(ctx)
	if err == nil {
		t.Error("Forward with 0 inputs should error")
	}

	// Forward: 1D input
	input1D := makeTensor(t, []int{4}, make([]float32, 4))
	_, err = m.Forward(ctx, input1D)
	if err == nil {
		t.Error("Forward with 1D input should error")
	}

	// Forward: dimension mismatch
	inputBad := makeTensor(t, []int{1, 5}, make([]float32, 5))
	_, err = m.Forward(ctx, inputBad)
	if err == nil {
		t.Error("Forward with mismatched dims should error")
	}
}

// ---------- MatMulNBits QuantizationInfo nil scale/zp ----------

func TestMatMulNBits_QuantizationInfo_NilFields(t *testing.T) {
	m := &MatMulNBits[float32]{}
	info := m.QuantizationInfo()
	if info["has_scale"] != false {
		t.Error("nil scale should report false")
	}
	if info["has_zero_point"] != false {
		t.Error("nil zeroPoint should report false")
	}
}

// ---------- Dense NewDense error paths ----------

func TestDense_Errors(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	// Empty name
	_, err := NewDense[float32]("", engine, ops, 4, 3)
	if err == nil {
		t.Error("empty name should error")
	}

	// Negative features
	_, err = NewDense[float32]("d", engine, ops, -1, 3)
	if err == nil {
		t.Error("negative input features should error")
	}
}

// ---------- Linear errors ----------

func TestLinear_Errors(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	// Forward: wrong inputs
	l, _ := NewLinear[float32]("l", engine, ops, 4, 3)
	input := makeTensor(t, []int{1, 4}, make([]float32, 4))
	_, err := l.Forward(context.Background(), input, input)
	if err == nil {
		t.Error("Forward with 2 inputs should error")
	}

	// Backward: wrong inputs
	grad := makeTensor(t, []int{1, 3}, make([]float32, 3))
	_, err = l.Backward(context.Background(), types.FullBackprop, grad, input, input)
	if err == nil {
		t.Error("Backward with 2 inputs should error")
	}

	// Negative features
	_, err = NewLinear[float32]("l", engine, ops, 0, 3)
	if err == nil {
		t.Error("zero input features should error")
	}
}

// ---------- MatMulNBits asymmetric with zero point ----------

func TestMatMulNBits_Asymmetric(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})
	zp, _ := tensor.New[uint8]([]int{2}, []uint8{8, 8})

	m, err := NewMatMulNBits("test", engine, ops, qw, scale, zp, 4, false)
	if err != nil {
		t.Fatal(err)
	}

	// Forward
	input := makeTensor(t, []int{1, 2}, []float32{1, 2})
	out, err := m.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out == nil {
		t.Fatal("output nil")
	}

	// QuantizationInfo with zero point
	info := m.QuantizationInfo()
	if info["has_zero_point"] != true {
		t.Error("should have zero point")
	}
}

// ---------- MatMulNBits global scale/zero point ----------

func TestMatMulNBits_GlobalScale(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale, _ := tensor.New[float32]([]int{1}, []float32{0.1}) // global scale
	zp, _ := tensor.New[uint8]([]int{1}, []uint8{8})

	m, err := NewMatMulNBits("test", engine, ops, qw, scale, zp, 4, false)
	if err != nil {
		t.Fatal(err)
	}

	input := makeTensor(t, []int{1, 2}, []float32{1, 2})
	out, err := m.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out == nil {
		t.Fatal("output nil")
	}
}

// ---------- MatMulNBits validation errors ----------

func TestMatMulNBits_ValidationErrors(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale1, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

	// Non-4-bit
	_, err := NewMatMulNBits("test", engine, ops, qw, scale1, nil, 8, true)
	if err == nil {
		t.Error("non-4-bit should error")
	}

	// Nil weights
	_, err = NewMatMulNBits("test", engine, ops, nil, scale1, nil, 4, true)
	if err == nil {
		t.Error("nil weights should error")
	}

	// Nil scale
	_, err = NewMatMulNBits("test", engine, ops, qw, nil, nil, 4, true)
	if err == nil {
		t.Error("nil scale should error")
	}

	// 1D weights
	qw1D, _ := tensor.New[uint8]([]int{4}, []uint8{0x12, 0x34, 0x56, 0x78})
	_, err = NewMatMulNBits("test", engine, ops, qw1D, scale1, nil, 4, true)
	if err == nil {
		t.Error("1D weights should error")
	}

	// 2D scale
	scale2D, _ := tensor.New[float32]([]int{2, 1}, []float32{0.1, 0.2})
	_, err = NewMatMulNBits("test", engine, ops, qw, scale2D, nil, 4, true)
	if err == nil {
		t.Error("2D scale should error")
	}

	// Mismatched scale rows
	scaleBad, _ := tensor.New[float32]([]int{3}, []float32{0.1, 0.2, 0.3})
	_, err = NewMatMulNBits("test", engine, ops, qw, scaleBad, nil, 4, true)
	if err == nil {
		t.Error("mismatched scale rows should error")
	}

	// 2D zero point
	zp2D, _ := tensor.New[uint8]([]int{2, 1}, []uint8{8, 8})
	_, err = NewMatMulNBits("test", engine, ops, qw, scale1, zp2D, 4, false)
	if err == nil {
		t.Error("2D zero point should error")
	}

	// Mismatched zero point
	zpBad, _ := tensor.New[uint8]([]int{3}, []uint8{8, 8, 8})
	_, err = NewMatMulNBits("test", engine, ops, qw, scale1, zpBad, 4, false)
	if err == nil {
		t.Error("mismatched zero point should error")
	}
}

// ---------- RotaryEmbedding Forward panic ----------

func TestRotaryEmbedding_Forward_PanicNoInputs(t *testing.T) {
	engine := makeEngine()
	re := NewRotaryEmbedding[float32](engine)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Forward with 0 inputs should panic")
		}
	}()
	_, _ = re.Forward(context.Background())
}

func TestRotaryEmbedding_Backward_PanicNoInputs(t *testing.T) {
	engine := makeEngine()
	re := NewRotaryEmbedding[float32](engine)
	g := makeTensor(t, []int{2}, []float32{1, 1})

	defer func() {
		if r := recover(); r == nil {
			t.Error("Backward with 0 inputs should panic")
		}
	}()
	_, _ = re.Backward(context.Background(), types.FullBackprop, g)
}

// ---------- Dense Backward with bias error (3D gradient) ----------

func TestBias_Backward_MultiDim(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	b, _ := NewBias("b", engine, ops, 4)

	// Forward with 2D input
	input := makeTensor(t, []int{2, 4}, make([]float32, 8))
	for i := range input.Data() {
		input.Data()[i] = float32(i)
	}
	_, err := b.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Bias Forward: %v", err)
	}

	// Backward with 2D gradient (1 Sum iteration)
	grad := makeTensor(t, []int{2, 4}, make([]float32, 8))
	for i := range grad.Data() {
		grad.Data()[i] = 1
	}
	grads, err := b.Backward(ctx, types.FullBackprop, grad)
	if err != nil {
		t.Fatalf("Bias Backward 2D: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("Bias Backward len = %d, want 1", len(grads))
	}
}

// ---------- Linear init register (partial coverage) ----------
// The init() function registers a Linear builder with float32. We can test it.

func TestLinear_InitRegistration(t *testing.T) {
	// Test that the init function registered correctly by trying to use the registered builder
	// This is implicitly tested through model.RegisterLayer, but we can at least verify the linear
	// package doesn't panic on init.
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	l, err := NewLinear[float32]("test", engine, ops, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	if l == nil {
		t.Fatal("NewLinear returned nil")
	}
}

// ---------- LMHead NewLMHead error ----------

func TestLMHead_NewError(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	// Zero dims should error (NewLinear requires positive features)
	_, err := NewLMHead(engine, ops, 0, 8)
	if err == nil {
		t.Error("NewLMHead with 0 hidden dim should error")
	}
}
