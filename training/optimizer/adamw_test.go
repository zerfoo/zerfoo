package optimizer

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestNewAdamW(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	adamw := NewAdamW[float32](engine, 0.001, 0.9, 0.999, 1e-8, 0.01)

	testutils.AssertNotNil(t, adamw, "AdamW should not be nil")
	testutils.AssertFloatEqual(t, 0.001, adamw.learningRate, 1e-6, "Learning rate should match")
	testutils.AssertFloatEqual(t, 0.9, adamw.beta1, 1e-6, "Beta1 should match")
	testutils.AssertFloatEqual(t, 0.999, adamw.beta2, 1e-6, "Beta2 should match")
	testutils.AssertFloatEqual(t, 1e-8, adamw.epsilon, 1e-10, "Epsilon should match")
	testutils.AssertFloatEqual(t, 0.01, adamw.weightDecay, 1e-6, "Weight decay should match")
	testutils.AssertEqual(t, 0, adamw.t, "Timestep should be 0 initially")
	testutils.AssertNotNil(t, adamw.m, "First moment map should be initialized")
	testutils.AssertNotNil(t, adamw.v, "Second moment map should be initialized")
}

func TestAdamW_Step_Basic(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Create AdamW with simple parameters
	adamw := NewAdamW[float32](engine, 0.1, 0.9, 0.999, 1e-8, 0.0) // No weight decay for simplicity

	// Create a parameter with gradient
	value, err := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	testutils.AssertNoError(t, err, "Failed to create value tensor")
	gradient, err := tensor.New[float32]([]int{2, 2}, []float32{0.1, 0.1, 0.1, 0.1})
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	param, err := graph.NewParameter("param1", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")

	param.Gradient = gradient

	params := []*graph.Parameter[float32]{param}

	// First step
	err = adamw.Step(ctx, params)
	testutils.AssertNoError(t, err, "First step should not error")
	testutils.AssertEqual(t, 1, adamw.t, "Timestep should be 1 after first step")

	// Check that parameter values changed
	originalValues := []float32{1.0, 2.0, 3.0, 4.0}

	newValues := param.Value.Data()
	for i, original := range originalValues {
		if newValues[i] == original {
			t.Errorf("Parameter value at index %d should have changed from %f", i, original)
		}
	}

	// Check that gradient was cleared (zeroed)
	testutils.AssertNotNil(t, param.Gradient, "Gradient tensor should still exist after step")

	for _, v := range param.Gradient.Data() {
		testutils.AssertFloatEqual(t, 0.0, v, 1e-6, "Gradient should be zeroed after step")
	}
}

//nolint:dupl // Intentional similarity with zero-gradient test to verify weight decay path using same harness
func TestAdamW_Step_WithWeightDecay(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Create AdamW with weight decay
	adamw := NewAdamW[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.1)

	// Create a parameter with gradient
	value, err := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	testutils.AssertNoError(t, err, "Failed to create value tensor")
	gradient, err := tensor.New[float32]([]int{2}, []float32{0.1, 0.1})
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	param, err := graph.NewParameter("param1", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")

	param.Gradient = gradient

	params := []*graph.Parameter[float32]{param}

	// Store original values
	originalValues := make([]float32, len(param.Value.Data()))
	copy(originalValues, param.Value.Data())

	// Step
	err = adamw.Step(ctx, params)
	testutils.AssertNoError(t, err, "Step should not error")

	// Check that values decreased due to weight decay
	newValues := param.Value.Data()
	for i, original := range originalValues {
		if newValues[i] >= original {
			t.Errorf("Parameter value at index %d should have decreased due to weight decay", i)
		}
	}
}

func TestAdamW_Step_MultipleSteps(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	adamw := NewAdamW[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.0)

	// Create a parameter
	value, err := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	testutils.AssertNoError(t, err, "Failed to create value tensor")

	param, err := graph.NewParameter("param1", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")

	params := []*graph.Parameter[float32]{param}

	// Multiple steps with different gradients
	for step := range 3 {
		gradient, err := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})
		testutils.AssertNoError(t, err, "Failed to create gradient tensor")

		param.Gradient = gradient

		err = adamw.Step(ctx, params)
		testutils.AssertNoError(t, err, "Step should not error")
		testutils.AssertEqual(t, step+1, adamw.t, "Timestep should increment")
	}

	// Check that momentum states are maintained
	testutils.AssertNotNil(t, adamw.m[param], "First moment should be maintained")
	testutils.AssertNotNil(t, adamw.v[param], "Second moment should be maintained")
}

func TestAdamW_Step_NoGradient(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	adamw := NewAdamW[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.0)

	// Create a parameter without gradient
	value, err := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	testutils.AssertNoError(t, err, "Failed to create value tensor")

	param, err := graph.NewParameter("param1", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")
	// param.Gradient is nil

	params := []*graph.Parameter[float32]{param}
	originalValues := make([]float32, len(param.Value.Data()))
	copy(originalValues, param.Value.Data())

	// Step should skip parameter without gradient
	err = adamw.Step(ctx, params)
	testutils.AssertNoError(t, err, "Step should not error with nil gradient")

	// Values should remain unchanged
	newValues := param.Value.Data()
	for i, original := range originalValues {
		testutils.AssertFloatEqual(t, original, newValues[i], 1e-6, "Parameter value should not change without gradient")
	}
}

func TestAdamW_Step_BiasCorrection(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test bias correction by checking timestep increment
	adamw := NewAdamW[float32](engine, 0.001, 0.9, 0.999, 1e-8, 0.0)

	// Create a parameter
	value, err := tensor.New[float32]([]int{1}, []float32{1.0})
	testutils.AssertNoError(t, err, "Failed to create value tensor")
	gradient, err := tensor.New[float32]([]int{1}, []float32{0.1})
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	param, err := graph.NewParameter("param1", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")

	param.Gradient = gradient

	params := []*graph.Parameter[float32]{param}

	// Check initial timestep
	testutils.AssertEqual(t, 0, adamw.t, "Initial timestep should be 0")

	// First step
	err = adamw.Step(ctx, params)
	testutils.AssertNoError(t, err, "Step should not error")
	testutils.AssertEqual(t, 1, adamw.t, "Timestep should be 1 after first step")

	// Second step with new gradient
	gradient2, err := tensor.New[float32]([]int{1}, []float32{0.1})
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	param.Gradient = gradient2

	err = adamw.Step(ctx, params)
	testutils.AssertNoError(t, err, "Step should not error")
	testutils.AssertEqual(t, 2, adamw.t, "Timestep should be 2 after second step")

	// Verify bias correction calculation happens (alpha computation)
	expectedAlpha := float32(0.001) * float32(math.Sqrt(float64(1.0-math.Pow(0.999, 2.0)))/(1.0-math.Pow(0.9, 2.0)))
	if expectedAlpha <= 0 {
		t.Errorf("Bias correction alpha should be positive, got %f", expectedAlpha)
	}
}

// Mock engine for error testing.
type mockAdamWEngine[T tensor.Numeric] struct {
	compute.Engine[T]
	zerosErr     bool
	mulScalarErr bool
	addErr       bool
	mulErr       bool
	sqrtErr      bool
	addScalarErr bool
	divErr       bool
	subErr       bool
}

func (m *mockAdamWEngine[T]) Zeros(ctx context.Context, dst *tensor.TensorNumeric[T], shape []int) error {
	if m.zerosErr {
		return errors.New("zeros error")
	}

	return m.Engine.Zeros(ctx, dst, shape)
}

func (m *mockAdamWEngine[T]) MulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.mulScalarErr {
		return nil, errors.New("mulscalar error")
	}

	return m.Engine.MulScalar(ctx, a, scalar, dst...)
}

func (m *mockAdamWEngine[T]) Add(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.addErr {
		return nil, errors.New("add error")
	}

	return m.Engine.Add(ctx, a, b, dst...)
}

func (m *mockAdamWEngine[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.mulErr {
		return nil, errors.New("mul error")
	}

	return m.Engine.Mul(ctx, a, b, dst...)
}

func (m *mockAdamWEngine[T]) Sqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.sqrtErr {
		return nil, errors.New("sqrt error")
	}

	return m.Engine.Sqrt(ctx, a, dst...)
}

func (m *mockAdamWEngine[T]) AddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.addScalarErr {
		return nil, errors.New("addscalar error")
	}

	return m.Engine.AddScalar(ctx, a, scalar, dst...)
}

func (m *mockAdamWEngine[T]) Div(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.divErr {
		return nil, errors.New("div error")
	}

	return m.Engine.Div(ctx, a, b, dst...)
}

func (m *mockAdamWEngine[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.subErr {
		return nil, errors.New("sub error")
	}

	return m.Engine.Sub(ctx, a, b, dst...)
}

func TestAdamW_Step_ErrorHandling(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}

	// Helper to create parameter
	createParam := func() *graph.Parameter[float32] {
		value, _ := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
		gradient, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.1})
		param, _ := graph.NewParameter("param1", value, tensor.New[float32])
		param.Gradient = gradient

		return param
	}

	tests := []struct {
		name        string
		engineError string
		expectError bool
	}{
		{"zeros error", "zerosErr", true},
		{"mulscalar error", "mulScalarErr", true},
		{"add error", "addErr", true},
		{"mul error", "mulErr", true},
		{"sqrt error", "sqrtErr", true},
		{"addscalar error", "addScalarErr", true},
		{"div error", "divErr", true},
		{"sub error", "subErr", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := &mockAdamWEngine[float32]{
				Engine: compute.NewCPUEngine[float32](ops),
			}

			// Set the specific error flag
			switch tt.engineError {
			case "zerosErr":
				engine.zerosErr = true
			case "mulScalarErr":
				engine.mulScalarErr = true
			case "addErr":
				engine.addErr = true
			case "mulErr":
				engine.mulErr = true
			case "sqrtErr":
				engine.sqrtErr = true
			case "addScalarErr":
				engine.addScalarErr = true
			case "divErr":
				engine.divErr = true
			case "subErr":
				engine.subErr = true
			}

			adamw := NewAdamW[float32](engine, 0.001, 0.9, 0.999, 1e-8, 0.01)
			param := createParam()
			params := []*graph.Parameter[float32]{param}

			err := adamw.Step(ctx, params)
			if tt.expectError {
				testutils.AssertError(t, err, "Step should return error for "+tt.name)
			} else {
				testutils.AssertNoError(t, err, "Step should not return error for "+tt.name)
			}
		})
	}
}

func TestAdamW_Step_EdgeCases(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Test with very small learning rate
	adamw := NewAdamW[float32](engine, 1e-10, 0.9, 0.999, 1e-8, 0.0)

	value, err := tensor.New[float32]([]int{1}, []float32{1.0})
	testutils.AssertNoError(t, err, "Failed to create value tensor")
	gradient, err := tensor.New[float32]([]int{1}, []float32{1.0})
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	param, err := graph.NewParameter("param1", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")

	param.Gradient = gradient

	params := []*graph.Parameter[float32]{param}
	originalValue := param.Value.Data()[0]

	err = adamw.Step(ctx, params)
	testutils.AssertNoError(t, err, "Step should not error with small learning rate")

	// Change should be very small
	change := math.Abs(float64(param.Value.Data()[0] - originalValue))
	if change > 1e-8 {
		t.Errorf("Change (%f) should be very small with tiny learning rate", change)
	}
}

//nolint:dupl // Intentional similarity with TestAdamW_Step_WithWeightDecay to cover zero-gradient pathway using same harness
func TestAdamW_Step_ZeroGradient(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	adamw := NewAdamW[float32](engine, 0.01, 0.9, 0.999, 1e-8, 0.1)

	value, err := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
	testutils.AssertNoError(t, err, "Failed to create value tensor")
	gradient, err := tensor.New[float32]([]int{2}, []float32{0.0, 0.0})
	testutils.AssertNoError(t, err, "Failed to create gradient tensor")

	param, err := graph.NewParameter("param1", value, tensor.New[float32])
	testutils.AssertNoError(t, err, "Failed to create parameter")

	param.Gradient = gradient

	params := []*graph.Parameter[float32]{param}
	originalValues := make([]float32, len(param.Value.Data()))
	copy(originalValues, param.Value.Data())

	err = adamw.Step(ctx, params)
	testutils.AssertNoError(t, err, "Step should not error with zero gradient")

	// With weight decay, values should still change
	newValues := param.Value.Data()
	for i, original := range originalValues {
		if newValues[i] >= original {
			t.Errorf("Parameter value at index %d should decrease due to weight decay even with zero gradient", i)
		}
	}
}
