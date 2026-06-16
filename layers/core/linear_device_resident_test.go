package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/gradcheck"
	"github.com/zerfoo/ztensor/types"
)

// TestLinear_Backward_Gradcheck pins the Linear backward (both dW and the
// dInput = grad @ W^T path) against float64 central finite differences. The
// CPU engine has no MatMulTransposeB, so this exercises the explicit-transpose
// fallback branch -- the byte-identical reference the GPU NT path (ADR 075 L1,
// device-resident operands) must match. A randomized upstream gradient catches
// a transposed-Jacobian error, which is exactly the failure mode a wrong
// MatMulTransposeB wiring would introduce.
func TestLinear_Backward_Gradcheck(t *testing.T) {
	ops := numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	ctx := context.Background()

	const inF, outF = 4, 3
	makeNode := func() (graph.Node[float64], error) {
		return NewLinear[float64]("gc_linear", engine, ops, inF, outF)
	}

	in, err := tensor.New[float64]([]int{2, inF}, []float64{
		0.1, -0.2, 0.3, 0.4,
		-0.5, 0.6, -0.7, 0.8,
	})
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}

	rep, err := gradcheck.Check(ctx, makeNode, []*tensor.TensorNumeric[float64]{in}, &gradcheck.Config{Seed: 42})
	if err != nil {
		t.Fatalf("gradcheck.Check: %v", err)
	}
	if !rep.OK() {
		t.Fatalf("Linear backward gradcheck failed:\n%s", rep.String())
	}
}

// TestLinear_Backward_DeviceResidentParity asserts the device-resident-operand
// dInput path (GPU MatMulTransposeB, reading the weight in its natural [in,out]
// layout with no explicit transpose) produces the SAME dInput as the CPU
// explicit-transpose reference. This is the GPU-vs-CPU half of the L1 gate:
// the optimization must not change the numerics. Skips cleanly on CPU-only
// machines / CI.
func TestLinear_Backward_DeviceResidentParity(t *testing.T) {
	ops := numeric.Float32Ops{}
	cpu := compute.NewCPUEngine[float32](ops)
	gpu, err := compute.NewGPUEngine[float32](ops, 0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer func() { _ = gpu.Close() }()
	ctx := context.Background()

	const inF, outF, batch = 8, 5, 4

	// Identical weights on both engines.
	wData := make([]float32, inF*outF)
	for i := range wData {
		wData[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
	}
	mkLinear := func(engine compute.Engine[float32]) (*Linear[float32], error) {
		l, e := NewLinear[float32]("dr_linear", engine, ops, inF, outF)
		if e != nil {
			return nil, e
		}
		copy(l.weights.Value.Data(), wData)
		return l, nil
	}
	lCPU, err := mkLinear(cpu)
	if err != nil {
		t.Fatalf("cpu linear: %v", err)
	}
	lGPU, err := mkLinear(gpu)
	if err != nil {
		t.Fatalf("gpu linear: %v", err)
	}

	inData := make([]float32, batch*inF)
	for i := range inData {
		inData[i] = float32(math.Cos(float64(i)*0.21)) * 0.7
	}
	gradData := make([]float32, batch*outF)
	for i := range gradData {
		gradData[i] = float32(math.Sin(float64(i)*0.53)+0.1) * 0.3
	}

	newPair := func() (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], error) {
		in, e := tensor.New[float32]([]int{batch, inF}, append([]float32(nil), inData...))
		if e != nil {
			return nil, nil, e
		}
		g, e := tensor.New[float32]([]int{batch, outF}, append([]float32(nil), gradData...))
		return in, g, e
	}

	inCPU, gradCPU, err := newPair()
	if err != nil {
		t.Fatalf("cpu tensors: %v", err)
	}
	inGPU, gradGPU, err := newPair()
	if err != nil {
		t.Fatalf("gpu tensors: %v", err)
	}

	dxCPU, err := lCPU.Backward(ctx, types.FullBackprop, gradCPU, inCPU)
	if err != nil {
		t.Fatalf("cpu backward: %v", err)
	}
	dxGPU, err := lGPU.Backward(ctx, types.FullBackprop, gradGPU, inGPU)
	if err != nil {
		t.Fatalf("gpu backward: %v", err)
	}

	cpuDX := dxCPU[0].Data()
	gpuDX := dxGPU[0].Data() // D2H on GPU storage.
	if len(cpuDX) != len(gpuDX) {
		t.Fatalf("dInput length mismatch: cpu=%d gpu=%d", len(cpuDX), len(gpuDX))
	}
	const tol = 1e-4
	var maxAbs float64
	for i := range cpuDX {
		d := math.Abs(float64(cpuDX[i] - gpuDX[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > tol {
		t.Fatalf("GPU NT dInput diverges from CPU transpose reference: max|diff|=%.3g > %g", maxAbs, tol)
	}
}
