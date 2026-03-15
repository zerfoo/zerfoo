package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestCPUEngine_Sin(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	tests := []struct {
		name  string
		shape []int
		data  []float32
	}{
		{
			name:  "1D [4]",
			shape: []int{4},
			data:  []float32{0, 1, 2, 3},
		},
		{
			name:  "2D [2,3]",
			shape: []int{2, 3},
			data:  []float32{0, 0.5, 1.0, 1.5, 2.0, 2.5},
		},
		{
			name:  "3D [1,1,4]",
			shape: []int{1, 1, 4},
			data:  []float32{-1, 0, 1, 3.14159},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tensor.New[float32](tt.shape, tt.data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			result, err := engine.Sin(ctx, a)
			if err != nil {
				t.Fatalf("Sin: %v", err)
			}

			got := result.Data()
			if len(got) != len(tt.data) {
				t.Fatalf("expected %d elements, got %d", len(tt.data), len(got))
			}

			for i, v := range tt.data {
				want := float32(math.Sin(float64(v)))
				if math.Abs(float64(got[i]-want)) > 1e-6 {
					t.Errorf("index %d: sin(%f) = %f, got %f", i, v, want, got[i])
				}
			}
		})
	}
}

func TestCPUEngine_Sin_Nil(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	_, err := engine.Sin(context.Background(), nil)
	if err == nil {
		t.Error("expected error for nil input")
	}
}

func TestCPUEngine_Sin_LargeShape(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	n := 2048
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	a, err := tensor.New[float32]([]int{1, 1, n}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	result, err := engine.Sin(ctx, a)
	if err != nil {
		t.Fatalf("Sin: %v", err)
	}

	got := result.Data()
	for i, v := range data {
		want := float32(math.Sin(float64(v)))
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Errorf("index %d: sin(%f) = %f, got %f", i, v, want, got[i])
		}
	}
}
