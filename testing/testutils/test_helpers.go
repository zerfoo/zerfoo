package testutils

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestCase represents a single test case with a name and a function to execute.
type TestCase struct {
	Name string
	Func func(t *testing.T)
}

// RunTests executes a slice of test cases.
func RunTests(t *testing.T, tests []TestCase) {
	for _, tt := range tests {
		t.Run(tt.Name, tt.Func)
	}
}

// ElementsMatch checks if two string slices contain the same elements, regardless of order.
func ElementsMatch(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	sort.Strings(a)
	sort.Strings(b)
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

// IntSliceEqual checks if two int slices are equal.
func IntSliceEqual(a, b []int) bool {
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

// Int64SliceEqual checks if two int64 slices are equal.
func Int64SliceEqual(a, b []int64) bool {
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

// AssertError checks if an error is not nil.
func AssertError(t *testing.T, err error, msg string) {
	t.Helper()
	if err == nil {
		t.Errorf("expected an error, but got nil: %s", msg)
	}
}

// AssertNoError checks if an error is nil.
func AssertNoError(t *testing.T, err error, msg string) {
	t.Helper()
	if err != nil {
		t.Errorf("expected no error, but got %v: %s", err, msg)
	}
}

// AssertEqual checks if two values are equal.
func AssertEqual[T comparable](t *testing.T, expected, actual T, msg string) {
	t.Helper()
	if actual != expected {
		t.Errorf("expected %v, got %v: %s", expected, actual, msg)
	}
}

// AssertNotNil checks if a value is not nil.
func AssertNotNil(t *testing.T, value interface{}, msg string) {
	t.Helper()
	if value == nil {
		t.Errorf("expected not nil, but got nil: %s", msg)
	}
}

// AssertNil checks if a value is nil.
func AssertNil(t *testing.T, value interface{}, msg string) {
	t.Helper()
	if value != nil && !reflect.ValueOf(value).IsNil() {
		t.Errorf("expected nil, but got %v: %s", value, msg)
	}
}

// AssertTrue checks if a boolean is true.
func AssertTrue(t *testing.T, condition bool, msg string) {
	t.Helper()
	if !condition {
		t.Errorf("expected true, but got false: %s", msg)
	}
}

// AssertFalse checks if a boolean is false.
func AssertFalse(t *testing.T, condition bool, msg string) {
	t.Helper()
	if condition {
		t.Errorf("expected false, but got true: %s", msg)
	}
}

// AssertContains checks if a string contains a substring.
func AssertContains(t *testing.T, s, substr, msg string) {
	t.Helper()
	if !strings.Contains(s, substr) {
		t.Errorf("expected %q to contain %q, but it did not: %s", s, substr, msg)
	}
}

// AssertPanics checks if a function panics.
func AssertPanics(t *testing.T, f func(), msg string) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected a panic, but none occurred: %s", msg)
		}
	}()
	f()
}

// AssertFloatEqual checks if two float values are approximately equal.
func AssertFloatEqual[T float32 | float64](t *testing.T, expected, actual T, tolerance T, msg string) {
	t.Helper()
	if math.Abs(float64(expected)-float64(actual)) > float64(tolerance) {
		t.Errorf("expected %v, got %v (tolerance %v): %s", expected, actual, tolerance, msg)
	}
}

// AssertFloat32SliceApproxEqual checks if two float32 slices are approximately equal element-wise.
func AssertFloat32SliceApproxEqual(t *testing.T, expected, actual []float32, tolerance float32, msg string) {
	t.Helper()
	if len(expected) != len(actual) {
		t.Errorf("slice lengths do not match: expected %d, got %d: %s", len(expected), len(actual), msg)
		return
	}
	for i := range expected {
		if math.Abs(float64(expected[i])-float64(actual[i])) > float64(tolerance) {
			t.Errorf("elements at index %d are not approximately equal: expected %v, got %v (tolerance %v): %s", i, expected[i], actual[i], tolerance, msg)
			return
		}
	}
}

// AssertFloat64SliceApproxEqual checks if two float64 slices are approximately equal element-wise.
func AssertFloat64SliceApproxEqual(t *testing.T, expected, actual []float64, tolerance float64, msg string) {
	t.Helper()
	if len(expected) != len(actual) {
		t.Errorf("slice lengths do not match: expected %d, got %d: %s", len(expected), len(actual), msg)
		return
	}
	for i := range expected {
		if math.Abs(expected[i]-actual[i]) > tolerance {
			t.Errorf("elements at index %d are not approximately equal: expected %v, got %v (tolerance %v): %s", i, expected[i], actual[i], tolerance, msg)
			return
		}
	}
}

// MockEngine is a mock implementation of the compute.Engine interface.
type MockEngine[T tensor.Numeric] struct {
	compute.Engine[T]
	Err error
}

// UnaryOp performs a unary operation on a tensor.
func (e *MockEngine[T]) UnaryOp(_ context.Context, _ *tensor.Tensor[T], _ func(T) T, _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Add performs element-wise addition of two tensors.
func (e *MockEngine[T]) Add(_ context.Context, _, _ *tensor.Tensor[T], _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Sub performs element-wise subtraction of two tensors.
func (e *MockEngine[T]) Sub(_ context.Context, _, _ *tensor.Tensor[T], _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Mul performs element-wise multiplication of two tensors.
func (e *MockEngine[T]) Mul(_ context.Context, _, _ *tensor.Tensor[T], _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Div performs element-wise division of two tensors.
func (e *MockEngine[T]) Div(_ context.Context, _, _ *tensor.Tensor[T], _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// MatMul performs matrix multiplication of two tensors.
func (e *MockEngine[T]) MatMul(_ context.Context, _, _ *tensor.Tensor[T], _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Transpose transposes a tensor along the specified axes.
func (e *MockEngine[T]) Transpose(_ context.Context, _ *tensor.Tensor[T], _ []int, _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Sum computes the sum of tensor elements along the specified axis.
func (e *MockEngine[T]) Sum(_ context.Context, _ *tensor.Tensor[T], _ int, _ bool, _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Exp computes the exponential of tensor elements.
func (e *MockEngine[T]) Exp(_ context.Context, _ *tensor.Tensor[T], _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Log computes the natural logarithm of tensor elements.
func (e *MockEngine[T]) Log(_ context.Context, _ *tensor.Tensor[T], _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Pow computes the power of base tensor raised to exponent tensor.
func (e *MockEngine[T]) Pow(_ context.Context, _, _ *tensor.Tensor[T], _ ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

// Zero sets all elements of the tensor to zero.
func (e *MockEngine[T]) Zero(_ context.Context, _ *tensor.Tensor[T]) error {
	return e.Err
}

// Copy copies data from source tensor to destination tensor.
func (e *MockEngine[T]) Copy(_ context.Context, _, _ *tensor.Tensor[T]) error {
	return e.Err
}

// WithName returns a new engine with the specified name.
func (e *MockEngine[T]) WithName(_ string) compute.Engine[T] {
	return e
}

// Name returns the name of the engine.
func (e *MockEngine[T]) Name() string {
	return "mock"
}

func (e *MockEngine[T]) String() string {
	return e.Name()
}

// Close closes the engine and releases resources.
func (e *MockEngine[T]) Close() error {
	return e.Err
}

// Wait waits for all pending operations to complete.
func (e *MockEngine[T]) Wait() error {
	return e.Err
}

// Device returns the device associated with the engine.
func (e *MockEngine[T]) Device() device.Device {
	return nil
}

// Allocator returns the memory allocator for the engine.
func (e *MockEngine[T]) Allocator() device.Allocator {
	return nil
}

// Context returns the context associated with the engine.
func (e *MockEngine[T]) Context() context.Context {
	return context.Background()
}

// WithContext returns a new engine with the specified context.
func (e *MockEngine[T]) WithContext(_ context.Context) compute.Engine[T] {
	return e
}

// WithAllocator returns a new engine with the specified allocator.
func (e *MockEngine[T]) WithAllocator(_ device.Allocator) compute.Engine[T] {
	return e
}

// WithDevice returns a new engine with the specified device.
func (e *MockEngine[T]) WithDevice(_ device.Device) compute.Engine[T] {
	return e
}

// WithError returns a mock engine that will return the specified error.
func (e *MockEngine[T]) WithError(err error) *MockEngine[T] {
	e.Err = err

	return e
}

// NewMockEngine creates a new mock engine for testing.
func NewMockEngine[T tensor.Numeric]() *MockEngine[T] {
	return &MockEngine[T]{}
}

// NewMockEngineWithError creates a new mock engine that returns the specified error.
func NewMockEngineWithError[T tensor.Numeric](err error) *MockEngine[T] {
	return &MockEngine[T]{Err: err}
}

// CompareTensorsApprox checks if two tensors are approximately equal element-wise.
func CompareTensorsApprox[T tensor.Numeric](t *testing.T, actual, expected *tensor.Tensor[T], epsilon T) bool {
	t.Helper()
	if !actual.ShapeEquals(expected) {
		t.Errorf("tensor shapes do not match: actual %v, expected %v", actual.Shape(), expected.Shape())
		return false
	}

	actualData := actual.Data()
	expectedData := expected.Data()

	if len(actualData) != len(expectedData) {
		t.Errorf("tensor data lengths do not match: actual %d, expected %d", len(actualData), len(expectedData))
		return false
	}

	for i := range actualData {
		if math.Abs(float64(actualData[i])-float64(expectedData[i])) > float64(epsilon) {
			t.Errorf("tensor elements at index %d are not approximately equal: actual %v, expected %v, epsilon %v", i, actualData[i], expectedData[i], epsilon)
			return false
		}
	}
	return true
}

// FailingInitializer is a test initializer that always returns an error.
type FailingInitializer[T tensor.Numeric] struct{}

// Initialize always returns an error for testing error paths.
func (f *FailingInitializer[T]) Initialize(inputSize, outputSize int) ([]T, error) {
	return nil, errors.New("test initializer failure")
}

// TestInitializer is a test initializer that sets all values to a specific value.
type TestInitializer[T tensor.Numeric] struct {
	Value T
}

// Initialize sets all values to the specified value.
func (t *TestInitializer[T]) Initialize(inputSize, outputSize int) ([]T, error) {
	size := inputSize * outputSize
	data := make([]T, size)
	for i := range data {
		data[i] = t.Value
	}
	return data, nil
}