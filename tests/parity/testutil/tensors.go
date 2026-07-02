package testutil

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// MakeTensor creates a tensor from golden data.
func MakeTensor(t *testing.T, data []float32, shape []int) *tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("create tensor shape=%v: %v", shape, err)
	}
	return tn
}

// MakeParam creates a graph.Parameter from golden data.
func MakeParam(t *testing.T, name string, data []float32, shape []int) *graph.Parameter[float32] {
	t.Helper()
	tn := MakeTensor(t, data, shape)
	p, err := graph.NewParameter[float32](name, tn, tensor.New[float32])
	if err != nil {
		t.Fatalf("create param %s: %v", name, err)
	}
	return p
}

// Setup creates a CPU compute engine and Float32Ops for parity tests.
func Setup() (compute.Engine[float32], *numeric.Float32Ops) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	return engine, ops
}
