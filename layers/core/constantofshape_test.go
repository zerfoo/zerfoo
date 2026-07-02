package core

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestBuildConstantOfShape_TensorFillValue(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name      string
		attrs     map[string]any
		wantValue float32
	}{
		{
			name: "float32 tensor with -FLT_MAX",
			attrs: map[string]any{
				"value": makeFloat32Tensor(-3.4028235e+38),
			},
			wantValue: -3.4028235e+38,
		},
		{
			name: "float32 tensor with 0.0",
			attrs: map[string]any{
				"value": makeFloat32Tensor(0.0),
			},
			wantValue: 0.0,
		},
		{
			name: "float64 tensor with 1.5",
			attrs: map[string]any{
				"value": makeFloat64Tensor(1.5),
			},
			wantValue: 1.5,
		},
		{
			name: "int64 tensor with 42",
			attrs: map[string]any{
				"value": makeInt64Tensor(42),
			},
			wantValue: 42.0,
		},
		{
			name:      "no value attribute defaults to zero",
			attrs:     map[string]any{},
			wantValue: 0.0,
		},
		{
			name: "plain float64 still works",
			attrs: map[string]any{
				"value": float64(2.5),
			},
			wantValue: 2.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node, err := BuildConstantOfShape[float32](engine, ops, "", nil, tt.attrs)
			if err != nil {
				t.Fatalf("BuildConstantOfShape failed: %v", err)
			}

			// Create a shape input: [2, 3]
			shapeInput, err := tensor.New[float32]([]int{2}, []float32{2, 3})
			if err != nil {
				t.Fatalf("failed to create shape input: %v", err)
			}

			output, err := node.Forward(context.Background(), shapeInput)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			if len(output.Data()) != 6 {
				t.Fatalf("expected 6 elements, got %d", len(output.Data()))
			}

			for i, v := range output.Data() {
				if v != tt.wantValue {
					t.Errorf("element %d: got %v, want %v", i, v, tt.wantValue)
				}
			}
		})
	}
}

func makeFloat32Tensor(v float32) *tensorValue {
	data := make([]byte, 4)
	binary.LittleEndian.PutUint32(data, math.Float32bits(v))
	return &tensorValue{
		Dtype: tensorDTypeFloat32,
		Shape: []int64{1},
		Data:  data,
	}
}

func makeFloat64Tensor(v float64) *tensorValue {
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, math.Float64bits(v))
	return &tensorValue{
		Dtype: tensorDTypeFloat64,
		Shape: []int64{1},
		Data:  data,
	}
}

func makeInt64Tensor(v int64) *tensorValue {
	data := make([]byte, 8)
	binary.LittleEndian.PutUint64(data, uint64(v))
	return &tensorValue{
		Dtype: tensorDTypeInt64,
		Shape: []int64{1},
		Data:  data,
	}
}
