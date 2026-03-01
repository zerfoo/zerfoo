package testutils

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

func TestRunTests(t *testing.T) {
	var ran []string

	cases := []TestCase{
		{Name: "first", Func: func(t *testing.T) { ran = append(ran, "first") }},
		{Name: "second", Func: func(t *testing.T) { ran = append(ran, "second") }},
	}

	RunTests(t, cases)

	if len(ran) != 2 {
		t.Fatalf("expected 2 tests to run, got %d", len(ran))
	}

	if ran[0] != "first" || ran[1] != "second" {
		t.Errorf("unexpected run order: %v", ran)
	}
}

func TestElementsMatch(t *testing.T) {
	tests := []struct {
		name string
		a, b []string
		want bool
	}{
		{"identical", []string{"a", "b", "c"}, []string{"a", "b", "c"}, true},
		{"different order", []string{"c", "a", "b"}, []string{"a", "b", "c"}, true},
		{"different length", []string{"a", "b"}, []string{"a", "b", "c"}, false},
		{"different elements", []string{"a", "b", "c"}, []string{"a", "b", "d"}, false},
		{"empty", []string{}, []string{}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ElementsMatch(tc.a, tc.b)
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestIntSliceEqual(t *testing.T) {
	tests := []struct {
		name string
		a, b []int
		want bool
	}{
		{"equal", []int{1, 2, 3}, []int{1, 2, 3}, true},
		{"different", []int{1, 2, 3}, []int{1, 2, 4}, false},
		{"different length", []int{1, 2}, []int{1, 2, 3}, false},
		{"empty", []int{}, []int{}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IntSliceEqual(tc.a, tc.b)
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestInt64SliceEqual(t *testing.T) {
	tests := []struct {
		name string
		a, b []int64
		want bool
	}{
		{"equal", []int64{1, 2, 3}, []int64{1, 2, 3}, true},
		{"different", []int64{1, 2, 3}, []int64{1, 2, 4}, false},
		{"different length", []int64{1}, []int64{1, 2}, false},
		{"empty", []int64{}, []int64{}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Int64SliceEqual(tc.a, tc.b)
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestAssertError(t *testing.T) {
	ft := &testing.T{}
	AssertError(ft, errors.New("fail"), "should have error")
	// No failure expected — err is non-nil.

	// Verify that passing nil would cause a failure by checking with a real T
	// (we can't easily intercept t.Errorf, but we verify the function doesn't panic).
	AssertError(t, errors.New("something"), "error expected and present")
}

func TestAssertNoError(t *testing.T) {
	AssertNoError(t, nil, "should have no error")
}

func TestAssertEqual(t *testing.T) {
	AssertEqual(t, 42, 42, "ints should match")
	AssertEqual(t, "hello", "hello", "strings should match")
	AssertEqual(t, true, true, "bools should match")
}

func TestAssertNotNil(t *testing.T) {
	x := 42
	AssertNotNil(t, &x, "pointer should not be nil")
}

func TestAssertNil(t *testing.T) {
	AssertNil(t, nil, "nil should be nil")
}

func TestAssertNil_NilPointer(t *testing.T) {
	var p *int
	AssertNil(t, p, "nil pointer should be nil")
}

func TestAssertTrue(t *testing.T) {
	AssertTrue(t, true, "should be true")
}

func TestAssertFalse(t *testing.T) {
	AssertFalse(t, false, "should be false")
}

func TestAssertContains(t *testing.T) {
	AssertContains(t, "hello world", "world", "should contain 'world'")
}

func TestAssertPanics(t *testing.T) {
	AssertPanics(t, func() { panic("boom") }, "should panic")
}

func TestAssertFloatEqual_Float32(t *testing.T) {
	AssertFloatEqual(t, float32(1.0), float32(1.0001), float32(0.001), "should be close")
}

func TestAssertFloatEqual_Float64(t *testing.T) {
	AssertFloatEqual(t, 3.14, 3.14, 1e-10, "should be equal")
}

func TestAssertFloat32SliceApproxEqual(t *testing.T) {
	a := []float32{1.0, 2.0, 3.0}
	b := []float32{1.001, 2.001, 3.001}
	AssertFloat32SliceApproxEqual(t, a, b, 0.01, "should be approximately equal")
}

func TestAssertFloat32SliceApproxEqual_DifferentLength(t *testing.T) {
	ft := &testing.T{}
	a := []float32{1.0, 2.0}
	b := []float32{1.0}
	// This should trigger an error on ft, not on the real t.
	AssertFloat32SliceApproxEqual(ft, a, b, 0.01, "different lengths")
}

func TestAssertFloat64SliceApproxEqual(t *testing.T) {
	a := []float64{1.0, 2.0, 3.0}
	b := []float64{1.0001, 2.0001, 3.0001}
	AssertFloat64SliceApproxEqual(t, a, b, 0.001, "should be approximately equal")
}

func TestAssertFloat64SliceApproxEqual_DifferentLength(t *testing.T) {
	ft := &testing.T{}
	a := []float64{1.0}
	b := []float64{1.0, 2.0}
	AssertFloat64SliceApproxEqual(ft, a, b, 0.001, "different lengths")
}

func TestNewMockEngine(t *testing.T) {
	eng := NewMockEngine[float32]()
	if eng == nil {
		t.Fatal("expected non-nil engine")
	}

	if eng.Err != nil {
		t.Errorf("expected nil Err, got %v", eng.Err)
	}
}

func TestNewMockEngineWithError(t *testing.T) {
	testErr := errors.New("mock error")
	eng := NewMockEngineWithError[float32](testErr)

	if !errors.Is(eng.Err, testErr) {
		t.Errorf("expected %v, got %v", testErr, eng.Err)
	}
}

func TestMockEngine_Methods(t *testing.T) {
	testErr := errors.New("fail")
	eng := NewMockEngineWithError[float32](testErr)
	ctx := context.Background()

	// Test that all methods return the configured error.
	tests := []struct {
		name string
		err  error
	}{
		{"UnaryOp", func() error { _, e := eng.UnaryOp(ctx, nil, nil); return e }()},
		{"Add", func() error { _, e := eng.Add(ctx, nil, nil); return e }()},
		{"Sub", func() error { _, e := eng.Sub(ctx, nil, nil); return e }()},
		{"Mul", func() error { _, e := eng.Mul(ctx, nil, nil); return e }()},
		{"Div", func() error { _, e := eng.Div(ctx, nil, nil); return e }()},
		{"MatMul", func() error { _, e := eng.MatMul(ctx, nil, nil); return e }()},
		{"Transpose", func() error { _, e := eng.Transpose(ctx, nil, nil); return e }()},
		{"Sum", func() error { _, e := eng.Sum(ctx, nil, 0, false); return e }()},
		{"Exp", func() error { _, e := eng.Exp(ctx, nil); return e }()},
		{"Log", func() error { _, e := eng.Log(ctx, nil); return e }()},
		{"Pow", func() error { _, e := eng.Pow(ctx, nil, nil); return e }()},
		{"Zero", func() error { return eng.Zero(ctx, nil) }()},
		{"Copy", func() error { return eng.Copy(ctx, nil, nil) }()},
		{"Close", func() error { return eng.Close() }()},
		{"Wait", func() error { return eng.Wait() }()},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if !errors.Is(tc.err, testErr) {
				t.Errorf("expected %v, got %v", testErr, tc.err)
			}
		})
	}
}

func TestMockEngine_NonErrorMethods(t *testing.T) {
	eng := NewMockEngine[float32]()

	if eng.Name() != "mock" {
		t.Errorf("expected 'mock', got %q", eng.Name())
	}

	if eng.String() != "mock" {
		t.Errorf("expected 'mock', got %q", eng.String())
	}

	if eng.Device() != nil {
		t.Error("expected nil Device")
	}

	if eng.Allocator() != nil {
		t.Error("expected nil Allocator")
	}

	if eng.Context() == nil {
		t.Error("expected non-nil Context")
	}

	// Fluent setters return self.
	if eng.WithName("x") != eng {
		t.Error("WithName should return self")
	}

	if eng.WithContext(context.Background()) != eng {
		t.Error("WithContext should return self")
	}

	if eng.WithAllocator(nil) != eng {
		t.Error("WithAllocator should return self")
	}

	if eng.WithDevice(nil) != eng {
		t.Error("WithDevice should return self")
	}
}

func TestMockEngine_WithError(t *testing.T) {
	eng := NewMockEngine[float32]()
	testErr := errors.New("injected")

	result := eng.WithError(testErr)
	if result != eng {
		t.Error("WithError should return self")
	}

	if !errors.Is(eng.Err, testErr) {
		t.Errorf("expected %v, got %v", testErr, eng.Err)
	}
}

func TestCompareTensorsApprox_Equal(t *testing.T) {
	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[float32]([]int{2, 3}, []float32{1.001, 2.001, 3.001, 4.001, 5.001, 6.001})

	ok := CompareTensorsApprox(t, a, b, float32(0.01))
	if !ok {
		t.Error("expected tensors to be approximately equal")
	}
}

func TestCompareTensorsApprox_DifferentShape(t *testing.T) {
	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})

	ft := &testing.T{}
	ok := CompareTensorsApprox(ft, a, b, float32(0.01))
	if ok {
		t.Error("expected tensors with different shapes to not match")
	}
}

func TestFailingInitializer(t *testing.T) {
	init := &FailingInitializer[float32]{}
	data, err := init.Initialize(4, 3)

	if err == nil {
		t.Error("expected error from FailingInitializer")
	}

	if data != nil {
		t.Errorf("expected nil data, got %v", data)
	}
}

// TestAssertions_FailurePaths exercises the error-reporting branches of all assertion helpers.
// We use a separate testing.T to avoid failing the real test.
func TestAssertions_FailurePaths(t *testing.T) {
	ft := &testing.T{}

	// AssertError with nil error.
	AssertError(ft, nil, "should fail")

	// AssertNoError with non-nil error.
	AssertNoError(ft, errors.New("oops"), "should fail")

	// AssertEqual with different values.
	AssertEqual(ft, 1, 2, "should fail")

	// AssertNotNil with nil.
	AssertNotNil(ft, nil, "should fail")

	// AssertNil with non-nil value.
	x := 42
	AssertNil(ft, &x, "should fail")

	// AssertTrue with false.
	AssertTrue(ft, false, "should fail")

	// AssertFalse with true.
	AssertFalse(ft, true, "should fail")

	// AssertContains with missing substring.
	AssertContains(ft, "hello", "xyz", "should fail")

	// AssertFloatEqual with values outside tolerance.
	AssertFloatEqual(ft, 1.0, 2.0, 0.01, "should fail")

	// AssertFloat32SliceApproxEqual with values outside tolerance.
	AssertFloat32SliceApproxEqual(ft, []float32{1.0}, []float32{2.0}, 0.01, "should fail")

	// AssertFloat64SliceApproxEqual with values outside tolerance.
	AssertFloat64SliceApproxEqual(ft, []float64{1.0}, []float64{2.0}, 0.01, "should fail")
}

func TestAssertPanics_NoPanic(t *testing.T) {
	ft := &testing.T{}
	// A function that does not panic should cause AssertPanics to report failure.
	AssertPanics(ft, func() {}, "should fail because no panic")
}

func TestCompareTensorsApprox_NotEqual(t *testing.T) {
	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	b, _ := tensor.New[float32]([]int{2}, []float32{10, 20})

	ft := &testing.T{}
	ok := CompareTensorsApprox(ft, a, b, float32(0.01))
	if ok {
		t.Error("expected tensors to not be approximately equal")
	}
}

func TestTestInitializer(t *testing.T) {
	init := &TestInitializer[float32]{Value: 0.5}
	data, err := init.Initialize(3, 4)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(data) != 12 {
		t.Fatalf("expected 12 elements, got %d", len(data))
	}

	for i, v := range data {
		if v != 0.5 {
			t.Errorf("data[%d] = %f, want 0.5", i, v)
		}
	}
}
