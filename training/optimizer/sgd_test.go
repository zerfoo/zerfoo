package optimizer

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// testSGDClip is a generic helper function to test SGD gradient clipping for different numeric types.
func testSGDClip[T tensor.Numeric](ops numeric.Arithmetic[T]) func(t *testing.T) {
	return func(t *testing.T) {
		engine := compute.NewCPUEngine[T](ops)
		sgd := NewSGD[T](engine, ops, 0.1)
		value, _ := tensor.New[T]([]int{2, 2}, []T{ops.FromFloat32(1), ops.FromFloat32(2), ops.FromFloat32(3), ops.FromFloat32(4)})
		gradient, _ := tensor.New[T]([]int{2, 2}, []T{ops.FromFloat32(-10), ops.FromFloat32(0.5), ops.FromFloat32(10), ops.FromFloat32(-0.5)})
		param, _ := graph.NewParameter("param1", value, tensor.New[T])
		param.Gradient = gradient
		params := []*graph.Parameter[T]{param}
		threshold := float32(1.0)
		expected := []T{ops.FromFloat32(-1.0), ops.FromFloat32(0.5), ops.FromFloat32(1.0), ops.FromFloat32(-0.5)}

		sgd.Clip(context.Background(), params, threshold)

		if !reflect.DeepEqual(param.Gradient.Data(), expected) {
			t.Errorf("expected %v, got %v", expected, param.Gradient.Data())
		}
	}
}

type mockEngine[T tensor.Numeric] struct {
	compute.Engine[T]
	mulErr bool
	subErr bool
}

func (m *mockEngine[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.mulErr {
		return nil, errors.New("mul error")
	}

	return m.Engine.Mul(ctx, a, b, dst...)
}

func (m *mockEngine[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if m.subErr {
		return nil, errors.New("sub error")
	}

	return m.Engine.Sub(ctx, a, b, dst...)
}

func TestSGD_Step(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	sgd := NewSGD[int](engine, numeric.IntOps{}, 1.0)
	value, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	gradient, _ := tensor.New[int]([]int{2, 2}, []int{1, 1, 1, 1})
	param, _ := graph.NewParameter("param1", value, tensor.New[int])
	param.Gradient = gradient
	params := []*graph.Parameter[int]{param}
	if err := sgd.Step(context.Background(), params); err != nil {
		t.Fatalf("sgd.Step returned error: %v", err)
	}
	expected := []int{0, 1, 2, 3}
	if !reflect.DeepEqual(param.Value.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, param.Value.Data())
	}
}

func TestSGD_Clip(t *testing.T) {
	type testCase[T tensor.Numeric] struct {
		engine    compute.Engine[T]
		ops       numeric.Arithmetic[T]
		lr        float32
		value     []T
		gradient  []T
		threshold float32
		expected  []T
	}

	tests := []struct {
		name string
		test func(t *testing.T)
	}{
		{
			name: "float32",
			test: func(t *testing.T) {
				tc := testCase[float32]{
					engine:    compute.NewCPUEngine[float32](numeric.Float32Ops{}),
					ops:       numeric.Float32Ops{},
					lr:        0.1,
					value:     []float32{1, 2, 3, 4},
					gradient:  []float32{-10, 0.5, 10, -0.5},
					threshold: 1.0,
					expected:  []float32{-1.0, 0.5, 1.0, -0.5},
				}
				sgd := NewSGD[float32](tc.engine, tc.ops, tc.lr)
				value, _ := tensor.New[float32]([]int{2, 2}, tc.value)
				gradient, _ := tensor.New[float32]([]int{2, 2}, tc.gradient)
				param, _ := graph.NewParameter("param1", value, tensor.New[float32])
				param.Gradient = gradient
				params := []*graph.Parameter[float32]{param}

				sgd.Clip(context.Background(), params, tc.threshold)

				if !reflect.DeepEqual(param.Gradient.Data(), tc.expected) {
					t.Errorf("expected %v, got %v", tc.expected, param.Gradient.Data())
				}
			},
		},
		// Temporarily disabled due to float16/float8 compilation issues
		// {"float16", testSGDClip[float16.Float16](numeric.Float16Ops{})},
		// {"float8", testSGDClip[float8.Float8](numeric.Float8Ops{})},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.test)
	}
}

func TestSGD_Step_Error(t *testing.T) {
	t.Run("mul error", func(t *testing.T) {
		engine := &mockEngine[int]{
			Engine: compute.NewCPUEngine[int](numeric.IntOps{}),
			mulErr: true,
		}
		sgd := NewSGD[int](engine, numeric.IntOps{}, 1.0)
		value, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
		gradient, _ := tensor.New[int]([]int{2, 2}, []int{1, 1, 1, 1})
		param, _ := graph.NewParameter("param1", value, tensor.New[int])
		param.Gradient = gradient
		params := []*graph.Parameter[int]{param}
		err := sgd.Step(context.Background(), params)
		if err == nil {
			t.Errorf("The code did not return an error")
		}
	})

	t.Run("sub error", func(t *testing.T) {
		engine := &mockEngine[int]{
			Engine: compute.NewCPUEngine[int](numeric.IntOps{}),
			subErr: true,
		}
		sgd := NewSGD[int](engine, numeric.IntOps{}, 1.0)
		value, _ := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
		gradient, _ := tensor.New[int]([]int{2, 2}, []int{1, 1, 1, 1})
		param, _ := graph.NewParameter("param1", value, tensor.New[int])
		param.Gradient = gradient
		params := []*graph.Parameter[int]{param}
		err := sgd.Step(context.Background(), params)
		if err == nil {
			t.Errorf("The code did not return an error")
		}
	})
}
