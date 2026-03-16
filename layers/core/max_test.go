package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestMax_Forward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := &Max[float32]{engine: engine}

	tests := []struct {
		name string
		a, b []float32
		want []float32
	}{
		{"basic", []float32{1, 5, 3}, []float32{4, 2, 6}, []float32{4, 5, 6}},
		{"equal", []float32{3, 3}, []float32{3, 3}, []float32{3, 3}},
		{"negative", []float32{-1, -5}, []float32{-3, -2}, []float32{-1, -2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, _ := tensor.New[float32]([]int{len(tt.a)}, tt.a)
			b, _ := tensor.New[float32]([]int{len(tt.b)}, tt.b)
			out, err := m.Forward(context.Background(), a, b)
			if err != nil {
				t.Fatalf("Max Forward: %v", err)
			}
			got := out.Data()
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Errorf("out[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestMax_WrongInputCount(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := &Max[float32]{engine: engine}

	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	_, err := m.Forward(context.Background(), a)
	if err == nil {
		t.Error("expected error for 1 input")
	}
}

func TestMax_SizeMismatch(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := &Max[float32]{engine: engine}

	a, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	b, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	_, err := m.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected error for size mismatch")
	}
}

func TestMax_OpType(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := &Max[float32]{engine: engine}
	if m.OpType() != "Max" {
		t.Errorf("OpType = %q, want %q", m.OpType(), "Max")
	}
}

func TestBuildMax(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	node, err := BuildMax[float32](engine, numeric.Float32Ops{}, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildMax: %v", err)
	}
	if node.OpType() != "Max" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "Max")
	}
}
