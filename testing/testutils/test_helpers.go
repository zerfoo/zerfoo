package testutils

import (
	"context"
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/tensor"
)

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
	if value != nil {
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

// MockEngine is a mock implementation of the compute.Engine interface.
type MockEngine[T tensor.Numeric] struct {
	compute.Engine[T]
	Err error
}

func (e *MockEngine[T]) UnaryOp(ctx context.Context, a *tensor.Tensor[T], op func(T) T, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Add(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Sub(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Mul(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Div(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) MatMul(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Transpose(ctx context.Context, a *tensor.Tensor[T], axes []int, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Sum(ctx context.Context, a *tensor.Tensor[T], axis int, keepDims bool, dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Exp(ctx context.Context, a *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Log(ctx context.Context, a *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Pow(ctx context.Context, base, exponent *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return nil, e.Err
}

func (e *MockEngine[T]) Zero(ctx context.Context, a *tensor.Tensor[T]) error {
	return e.Err
}

func (e *MockEngine[T]) Copy(ctx context.Context, dst, src *tensor.Tensor[T]) error {
	return e.Err
}

func (e *MockEngine[T]) WithName(name string) compute.Engine[T] {
	return e
}

func (e *MockEngine[T]) Name() string {
	return "mock"
}

func (e *MockEngine[T]) String() string {
	return e.Name()
}

func (e *MockEngine[T]) Close() error {
	return e.Err
}

func (e *MockEngine[T]) Wait() error {
	return e.Err
}

func (e *MockEngine[T]) Device() device.Device {
	return nil
}

func (e *MockEngine[T]) Allocator() device.Allocator {
	return nil
}

func (e *MockEngine[T]) Context() context.Context {
	return context.Background()
}

func (e *MockEngine[T]) WithContext(ctx context.Context) compute.Engine[T] {
	return e
}

func (e *MockEngine[T]) WithAllocator(allocator device.Allocator) compute.Engine[T] {
	return e
}

func (e *MockEngine[T]) WithDevice(d device.Device) compute.Engine[T] {
	return e
}

func (e *MockEngine[T]) WithError(err error) *MockEngine[T] {
	e.Err = err
	return e
}

func NewMockEngine[T tensor.Numeric]() *MockEngine[T] {
	return &MockEngine[T]{}
}

func NewMockEngineWithError[T tensor.Numeric](err error) *MockEngine[T] {
	return &MockEngine[T]{Err: err}
}
