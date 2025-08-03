package optimizer

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

type mockEngine[T tensor.Numeric] struct {
	compute.Engine[T]
	mulErr bool
	subErr bool
}

func (m *mockEngine[T]) Mul(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if m.mulErr {
		return nil, errors.New("mul error")
	}
	return m.Engine.Mul(ctx, a, b, dst...)
}

func (m *mockEngine[T]) Sub(ctx context.Context, a, b *tensor.Tensor[T], dst ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
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
	sgd.Step(params)
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

				sgd.Clip(params, tc.threshold)

				if !reflect.DeepEqual(param.Gradient.Data(), tc.expected) {
					t.Errorf("expected %v, got %v", tc.expected, param.Gradient.Data())
				}
			},
		},
		{
			name: "float16",
			test: func(t *testing.T) {
				ops := numeric.Float16Ops{}
				tc := testCase[float16.Float16]{
					engine:    compute.NewCPUEngine[float16.Float16](ops),
					ops:       ops,
					lr:        0.1,
					value:     []float16.Float16{ops.FromFloat32(1), ops.FromFloat32(2), ops.FromFloat32(3), ops.FromFloat32(4)},
					gradient:  []float16.Float16{ops.FromFloat32(-10), ops.FromFloat32(0.5), ops.FromFloat32(10), ops.FromFloat32(-0.5)},
					threshold: 1.0,
					expected:  []float16.Float16{ops.FromFloat32(-1.0), ops.FromFloat32(0.5), ops.FromFloat32(1.0), ops.FromFloat32(-0.5)},
				}
				sgd := NewSGD[float16.Float16](tc.engine, tc.ops, tc.lr)
				value, _ := tensor.New[float16.Float16]([]int{2, 2}, tc.value)
				gradient, _ := tensor.New[float16.Float16]([]int{2, 2}, tc.gradient)
				param, _ := graph.NewParameter("param1", value, tensor.New[float16.Float16])
				param.Gradient = gradient
				params := []*graph.Parameter[float16.Float16]{param}

				sgd.Clip(params, tc.threshold)

				if !reflect.DeepEqual(param.Gradient.Data(), tc.expected) {
					t.Errorf("expected %v, got %v", tc.expected, param.Gradient.Data())
				}
			},
		},
		{
			name: "float8",
			test: func(t *testing.T) {
				ops := numeric.Float8Ops{}
				tc := testCase[float8.Float8]{
					engine:    compute.NewCPUEngine[float8.Float8](ops),
					ops:       ops,
					lr:        0.1,
					value:     []float8.Float8{ops.FromFloat32(1), ops.FromFloat32(2), ops.FromFloat32(3), ops.FromFloat32(4)},
					gradient:  []float8.Float8{ops.FromFloat32(-10), ops.FromFloat32(0.5), ops.FromFloat32(10), ops.FromFloat32(-0.5)},
					threshold: 1.0,
					expected:  []float8.Float8{ops.FromFloat32(-1.0), ops.FromFloat32(0.5), ops.FromFloat32(1.0), ops.FromFloat32(-0.5)},
				}
				sgd := NewSGD[float8.Float8](tc.engine, tc.ops, tc.lr)
				value, _ := tensor.New[float8.Float8]([]int{2, 2}, tc.value)
				gradient, _ := tensor.New[float8.Float8]([]int{2, 2}, tc.gradient)
				param, _ := graph.NewParameter("param1", value, tensor.New[float8.Float8])
				param.Gradient = gradient
				params := []*graph.Parameter[float8.Float8]{param}

				sgd.Clip(params, tc.threshold)

				if !reflect.DeepEqual(param.Gradient.Data(), tc.expected) {
					t.Errorf("expected %v, got %v", tc.expected, param.Gradient.Data())
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.test)
	}
}

func TestSGD_Step_Panic(t *testing.T) {
	t.Run("mul error", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("The code did not panic")
			}
		}()
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
		sgd.Step(params)
	})

	t.Run("sub error", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("The code did not panic")
			}
		}()
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
		sgd.Step(params)
	})
}
