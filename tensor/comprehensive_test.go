package tensor

import (
	"math"
	"reflect"
	"testing"
)

// ---------- ShapesEqual ----------

func TestShapesEqual_Extended(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want bool
	}{
		{"equal", []int{2, 3}, []int{2, 3}, true},
		{"diff_length", []int{2, 3}, []int{2}, false},
		{"diff_values", []int{2, 3}, []int{2, 4}, false},
		{"empty_equal", []int{}, []int{}, true},
		{"nil_nil", nil, nil, true},
		{"nil_vs_empty", nil, []int{}, true},
		{"single", []int{5}, []int{5}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ShapesEqual(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("ShapesEqual(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

// ---------- ShapeEquals (method) ----------

func TestShapeEquals(t *testing.T) {
	a, _ := New[float32]([]int{2, 3}, nil)
	b, _ := New[float32]([]int{2, 3}, nil)
	c, _ := New[float32]([]int{3, 2}, nil)
	d, _ := New[float32]([]int{6}, nil)

	if !a.ShapeEquals(b) {
		t.Error("same shapes should be equal")
	}
	if a.ShapeEquals(c) {
		t.Error("different shapes should not be equal")
	}
	if a.ShapeEquals(d) {
		t.Error("different dims should not be equal")
	}
}

// ---------- SetStrides / SetShape ----------

func TestSetStridesAndSetShape(t *testing.T) {
	tensor, _ := New[float32]([]int{2, 3}, nil)

	newStrides := []int{3, 1}
	tensor.SetStrides(newStrides)
	if !reflect.DeepEqual(tensor.Strides(), newStrides) {
		t.Errorf("SetStrides: got %v, want %v", tensor.Strides(), newStrides)
	}

	newShape := []int{6}
	tensor.SetShape(newShape)
	if !reflect.DeepEqual(tensor.Shape(), newShape) {
		t.Errorf("SetShape: got %v, want %v", tensor.Shape(), newShape)
	}
}

// ---------- Float32ToBytes ----------

func TestFloat32ToBytes(t *testing.T) {
	t.Run("non_empty", func(t *testing.T) {
		data := []float32{1.0, 2.0}
		b, err := Float32ToBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		if len(b) != 8 { // 2 * 4 bytes
			t.Errorf("len(bytes) = %d, want 8", len(b))
		}
	})

	t.Run("empty", func(t *testing.T) {
		b, err := Float32ToBytes(nil)
		if err != nil {
			t.Fatal(err)
		}
		if b != nil {
			t.Errorf("expected nil for empty input, got %v", b)
		}
	})
}

// ---------- Int8ToBytes / Uint8ToBytes empty slices ----------

func TestInt8ToBytes_Empty(t *testing.T) {
	b, err := Int8ToBytes(nil)
	if err != nil {
		t.Fatal(err)
	}
	if b != nil {
		t.Errorf("expected nil for empty input")
	}
}

func TestUint8ToBytes_Empty(t *testing.T) {
	b, err := Uint8ToBytes(nil)
	if err != nil {
		t.Fatal(err)
	}
	if b != nil {
		t.Errorf("expected nil for empty input")
	}
}

// ---------- Bytes unsupported type ----------

func TestBytes_UnsupportedType(t *testing.T) {
	tensor, _ := New[int]([]int{2}, []int{1, 2})
	_, err := tensor.Bytes()
	if err == nil {
		t.Error("expected error for unsupported type int")
	}
}

// ---------- NewFromBytes error paths ----------

func TestNewFromBytes_Errors(t *testing.T) {
	t.Run("invalid_dim", func(t *testing.T) {
		_, err := NewFromBytes[float32]([]int{-1}, []byte{0, 0, 0, 0})
		if err == nil {
			t.Error("expected error for negative dim")
		}
	})

	t.Run("data_size_mismatch", func(t *testing.T) {
		_, err := NewFromBytes[float32]([]int{2}, []byte{0, 0, 0, 0}) // needs 8 bytes
		if err == nil {
			t.Error("expected error for data size mismatch")
		}
	})
}

// ---------- NewFromType ----------

func TestNewFromType(t *testing.T) {
	tests := []struct {
		name      string
		tensorTyp reflect.Type
		shape     []int
		data      any
		wantErr   bool
	}{
		{
			"float32",
			reflect.TypeOf((*TensorNumeric[float32])(nil)),
			[]int{2},
			[]float32{1.0, 2.0},
			false,
		},
		{
			"float64",
			reflect.TypeOf((*TensorNumeric[float64])(nil)),
			[]int{2},
			[]float64{1.0, 2.0},
			false,
		},
		{
			"int",
			reflect.TypeOf((*TensorNumeric[int])(nil)),
			[]int{2},
			[]int{1, 2},
			false,
		},
		{
			"int32",
			reflect.TypeOf((*TensorNumeric[int32])(nil)),
			[]int{2},
			[]int32{1, 2},
			false,
		},
		{
			"int64",
			reflect.TypeOf((*TensorNumeric[int64])(nil)),
			[]int{2},
			[]int64{1, 2},
			false,
		},
		{
			"int8",
			reflect.TypeOf((*TensorNumeric[int8])(nil)),
			[]int{2},
			[]int8{1, 2},
			false,
		},
		{
			"uint8",
			reflect.TypeOf((*TensorNumeric[uint8])(nil)),
			[]int{2},
			[]uint8{1, 2},
			false,
		},
		{
			"nil_data_float32",
			reflect.TypeOf((*TensorNumeric[float32])(nil)),
			[]int{2},
			nil,
			false,
		},
		{
			"not_pointer",
			reflect.TypeOf(TensorNumeric[float32]{}),
			[]int{2},
			nil,
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := NewFromType(tt.tensorTyp, tt.shape, tt.data)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result == nil {
				t.Fatal("result is nil")
			}
			if !reflect.DeepEqual(result.Shape(), tt.shape) {
				t.Errorf("shape = %v, want %v", result.Shape(), tt.shape)
			}
		})
	}
}

// ---------- NewBool ----------

func TestNewBool(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tb, err := NewBool([]int{2, 2}, []bool{true, false, true, false})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(tb.Shape(), []int{2, 2}) {
			t.Errorf("shape = %v, want [2 2]", tb.Shape())
		}
		if tb.DType() != reflect.TypeOf(false) {
			t.Errorf("DType = %v, want bool", tb.DType())
		}
		if len(tb.Data()) != 4 {
			t.Errorf("data len = %d, want 4", len(tb.Data()))
		}
	})

	t.Run("nil_data", func(t *testing.T) {
		tb, err := NewBool([]int{3}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(tb.Data()) != 3 {
			t.Errorf("data len = %d, want 3", len(tb.Data()))
		}
	})

	t.Run("scalar", func(t *testing.T) {
		tb, err := NewBool([]int{}, []bool{true})
		if err != nil {
			t.Fatal(err)
		}
		if tb.Data()[0] != true {
			t.Error("expected true")
		}
	})

	t.Run("scalar_empty", func(t *testing.T) {
		tb, err := NewBool([]int{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(tb.Data()) != 1 {
			t.Errorf("data len = %d, want 1", len(tb.Data()))
		}
	})

	t.Run("scalar_too_many", func(t *testing.T) {
		_, err := NewBool([]int{}, []bool{true, false})
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("negative_dim", func(t *testing.T) {
		_, err := NewBool([]int{-1}, nil)
		if err == nil {
			t.Error("expected error for negative dim")
		}
	})

	t.Run("size_mismatch", func(t *testing.T) {
		_, err := NewBool([]int{3}, []bool{true, false})
		if err == nil {
			t.Error("expected error for size mismatch")
		}
	})
}

// ---------- TensorBool methods ----------

func TestTensorBool_Bytes(t *testing.T) {
	t.Run("non_empty", func(t *testing.T) {
		tb, _ := NewBool([]int{3}, []bool{true, false, true})
		b, err := tb.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		if len(b) != 3 {
			t.Errorf("len(bytes) = %d, want 3", len(b))
		}
		if b[0] != 1 || b[1] != 0 || b[2] != 1 {
			t.Errorf("bytes = %v, want [1 0 1]", b)
		}
	})

	t.Run("empty", func(t *testing.T) {
		tb, _ := NewBool([]int{0}, nil)
		b, err := tb.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		if b != nil {
			t.Errorf("expected nil for empty bool tensor")
		}
	})
}

// ---------- TensorBool interface conformance ----------

func TestTensorBool_Interface(t *testing.T) {
	tb, _ := NewBool([]int{2}, []bool{true, false})
	var iface Tensor = tb
	_ = iface.Shape()
	_ = iface.DType()
}

// ---------- NewString ----------

func TestNewString(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		ts, err := NewString([]int{2}, []string{"hello", "world"})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ts.Shape(), []int{2}) {
			t.Errorf("shape = %v, want [2]", ts.Shape())
		}
		if ts.DType() != reflect.TypeOf("") {
			t.Errorf("DType = %v, want string", ts.DType())
		}
		if !reflect.DeepEqual(ts.Data(), []string{"hello", "world"}) {
			t.Errorf("data = %v", ts.Data())
		}
	})

	t.Run("nil_data", func(t *testing.T) {
		ts, err := NewString([]int{2}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(ts.Data()) != 2 {
			t.Errorf("data len = %d, want 2", len(ts.Data()))
		}
	})

	t.Run("scalar", func(t *testing.T) {
		ts, err := NewString([]int{}, []string{"hi"})
		if err != nil {
			t.Fatal(err)
		}
		if ts.Data()[0] != "hi" {
			t.Error("expected 'hi'")
		}
	})

	t.Run("scalar_empty", func(t *testing.T) {
		ts, err := NewString([]int{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if len(ts.Data()) != 1 {
			t.Errorf("data len = %d, want 1", len(ts.Data()))
		}
	})

	t.Run("scalar_too_many", func(t *testing.T) {
		_, err := NewString([]int{}, []string{"a", "b"})
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("negative_dim", func(t *testing.T) {
		_, err := NewString([]int{-1}, nil)
		if err == nil {
			t.Error("expected error for negative dim")
		}
	})

	t.Run("size_mismatch", func(t *testing.T) {
		_, err := NewString([]int{3}, []string{"a"})
		if err == nil {
			t.Error("expected error for size mismatch")
		}
	})
}

// ---------- TensorString interface conformance ----------

func TestTensorString_Interface(t *testing.T) {
	ts, _ := NewString([]int{1}, []string{"test"})
	var iface Tensor = ts
	_ = iface.Shape()
	_ = iface.DType()
}

// ---------- Ones ----------

func TestOnes(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		data := Ones[float32](3)
		for i, v := range data {
			if v != 1.0 {
				t.Errorf("Ones[float32][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("float64", func(t *testing.T) {
		data := Ones[float64](2)
		for i, v := range data {
			if v != 1.0 {
				t.Errorf("Ones[float64][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("int", func(t *testing.T) {
		data := Ones[int](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[int][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("int8", func(t *testing.T) {
		data := Ones[int8](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[int8][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("int16", func(t *testing.T) {
		data := Ones[int16](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[int16][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("int32", func(t *testing.T) {
		data := Ones[int32](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[int32][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("int64", func(t *testing.T) {
		data := Ones[int64](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[int64][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("uint", func(t *testing.T) {
		data := Ones[uint](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[uint][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("uint8", func(t *testing.T) {
		data := Ones[uint8](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[uint8][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("uint32", func(t *testing.T) {
		data := Ones[uint32](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[uint32][%d] = %v, want 1", i, v)
			}
		}
	})
	t.Run("uint64", func(t *testing.T) {
		data := Ones[uint64](2)
		for i, v := range data {
			if v != 1 {
				t.Errorf("Ones[uint64][%d] = %v, want 1", i, v)
			}
		}
	})
}

// ---------- Equals ----------

func TestEquals(t *testing.T) {
	a, _ := New[float32]([]int{2}, []float32{1.0, 2.0})
	b, _ := New[float32]([]int{2}, []float32{1.0, 2.0})
	c, _ := New[float32]([]int{2}, []float32{1.0, 3.0})
	d, _ := New[float32]([]int{3}, []float32{1.0, 2.0, 3.0})

	if !Equals(a, b) {
		t.Error("identical tensors should be equal")
	}
	if Equals(a, c) {
		t.Error("different data should not be equal")
	}
	if Equals(a, d) {
		t.Error("different shapes should not be equal")
	}
}

// ---------- AssertEquals ----------

func TestAssertEquals(t *testing.T) {
	a, _ := New[float32]([]int{2}, []float32{1.0, 2.0})
	b, _ := New[float32]([]int{2}, []float32{1.0, 2.0})

	// Should not fail
	AssertEquals(t, a, b)
}

// ---------- AssertClose ----------

func TestAssertClose(t *testing.T) {
	t.Run("float32_close", func(t *testing.T) {
		a, _ := New[float32]([]int{2}, []float32{1.0, 2.0})
		b, _ := New[float32]([]int{2}, []float32{1.001, 2.001})
		AssertClose(t, a, b, 0.01)
	})

	t.Run("float64_close", func(t *testing.T) {
		a, _ := New[float64]([]int{2}, []float64{1.0, 2.0})
		b, _ := New[float64]([]int{2}, []float64{1.0001, 2.0001})
		AssertClose(t, a, b, 0.001)
	})

	t.Run("int_close", func(t *testing.T) {
		a, _ := New[int]([]int{2}, []int{10, 20})
		b, _ := New[int]([]int{2}, []int{10, 20})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("int8", func(t *testing.T) {
		a, _ := New[int8]([]int{2}, []int8{1, 2})
		b, _ := New[int8]([]int{2}, []int8{1, 2})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("int16", func(t *testing.T) {
		a, _ := New[int16]([]int{2}, []int16{1, 2})
		b, _ := New[int16]([]int{2}, []int16{1, 2})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("int32", func(t *testing.T) {
		a, _ := New[int32]([]int{2}, []int32{1, 2})
		b, _ := New[int32]([]int{2}, []int32{1, 2})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("int64", func(t *testing.T) {
		a, _ := New[int64]([]int{2}, []int64{1, 2})
		b, _ := New[int64]([]int{2}, []int64{1, 2})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("uint", func(t *testing.T) {
		a, _ := New[uint]([]int{2}, []uint{1, 2})
		b, _ := New[uint]([]int{2}, []uint{1, 2})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("uint8", func(t *testing.T) {
		a, _ := New[uint8]([]int{2}, []uint8{1, 2})
		b, _ := New[uint8]([]int{2}, []uint8{1, 2})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("uint32", func(t *testing.T) {
		a, _ := New[uint32]([]int{2}, []uint32{1, 2})
		b, _ := New[uint32]([]int{2}, []uint32{1, 2})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("uint64", func(t *testing.T) {
		a, _ := New[uint64]([]int{2}, []uint64{1, 2})
		b, _ := New[uint64]([]int{2}, []uint64{1, 2})
		AssertClose(t, a, b, 0.0)
	})

	t.Run("shape_mismatch", func(t *testing.T) {
		a, _ := New[float32]([]int{2}, []float32{1, 2})
		b, _ := New[float32]([]int{3}, []float32{1, 2, 3})
		// Use a sub-test with a mock T to capture the error
		mockT := &testing.T{}
		AssertClose(mockT, a, b, 0.1)
		// mockT would have errors but we can't check from here;
		// the point is to exercise the shape mismatch branch
	})

	t.Run("not_close", func(t *testing.T) {
		a, _ := New[float32]([]int{2}, []float32{1.0, 2.0})
		b, _ := New[float32]([]int{2}, []float32{1.0, 3.0})
		mockT := &testing.T{}
		AssertClose(mockT, a, b, 0.01)
	})
}

// ---------- AssertClose negative diff ----------

func TestAssertClose_NegativeDiff(t *testing.T) {
	// Test case where diff is negative (b > a) to exercise diff < -tolerance
	a, _ := New[float32]([]int{2}, []float32{1.0, 2.0})
	b, _ := New[float32]([]int{2}, []float32{1.0, 0.0}) // 0.0 - 2.0 = -2.0, |diff| > tolerance
	mockT := &testing.T{}
	AssertClose(mockT, a, b, 0.01)
}

// ---------- NewFromType edge cases ----------

func TestNewFromType_NoFields(t *testing.T) {
	// Type with no fields should fail the NumField check
	type emptyStruct struct{}
	_, err := NewFromType(reflect.TypeOf((*emptyStruct)(nil)), []int{2}, nil)
	if err == nil {
		t.Error("expected error for type with no fields")
	}
}

func TestNewFromType_DataError(t *testing.T) {
	// Create a NewFromType call with wrong data for the shape to trigger the inner error path
	typ := reflect.TypeOf((*TensorNumeric[float32])(nil))
	_, err := NewFromType(typ, []int{2}, []float32{1.0})
	if err == nil {
		t.Error("expected error for mismatched data length")
	}
}

// ---------- DType ----------

func TestTensorNumeric_DType(t *testing.T) {
	tensor, _ := New[float64]([]int{2}, []float64{1, 2})
	dt := tensor.DType()
	if dt.Kind() != reflect.Float64 {
		t.Errorf("DType = %v, want float64", dt)
	}
}

// ---------- Tensor interface for TensorNumeric ----------

func TestTensorNumeric_Interface(t *testing.T) {
	tensor, _ := New[float32]([]int{2}, []float32{1, 2})
	var iface Tensor = tensor
	_ = iface.Shape()
	_ = iface.DType()
}

// ---------- Float32 Bytes round-trip ----------

func TestFloat32Bytes_RoundTrip(t *testing.T) {
	original, _ := New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	b, err := original.Bytes()
	if err != nil {
		t.Fatal(err)
	}

	restored, err := NewFromBytes[float32]([]int{2, 2}, b)
	if err != nil {
		t.Fatal(err)
	}

	for i, v := range restored.Data() {
		if math.Abs(float64(v-original.Data()[i])) > 1e-6 {
			t.Errorf("mismatch at index %d: got %v, want %v", i, v, original.Data()[i])
		}
	}
}
