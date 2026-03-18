package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// TestSDPABackward_FiniteDiff verifies SDPA backward against finite differences.
func TestSDPABackward_FiniteDiff(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	headDim := 4
	batch := 2
	seqLen := 3

	sdpa := NewScaledDotProductAttention[float32](engine, headDim)

	makeT := func(shape []int, seed int) *tensor.TensorNumeric[float32] {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(((i*7+seed)%19)-9) / 20.0
		}
		t, _ := tensor.New[float32](shape, data)
		return t
	}

	q := makeT([]int{batch, seqLen, headDim}, 3)
	k := makeT([]int{batch, seqLen, headDim}, 5)
	v := makeT([]int{batch, seqLen, headDim}, 11)

	out, err := sdpa.Forward(ctx, q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	dOutData := make([]float32, len(out.Data()))
	for i := range dOutData {
		dOutData[i] = float32(((i*11+5)%13)-6) / 10.0
	}
	dOut, _ := tensor.New[float32](out.Shape(), dOutData)

	grads, err := sdpa.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	dQ, dK, dV := grads[0], grads[1], grads[2]

	eps := float32(1e-3)
	tol := float32(5e-2)

	checkGrad(t, "dQ", q, dQ, func() float32 {
		o, _ := sdpa.Forward(ctx, q, k, v, nil)
		return dotProduct(o.Data(), dOutData)
	}, eps, tol)

	sdpa.Forward(ctx, q, k, v, nil)
	grads, _ = sdpa.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil)
	dK = grads[1]

	checkGrad(t, "dK", k, dK, func() float32 {
		o, _ := sdpa.Forward(ctx, q, k, v, nil)
		return dotProduct(o.Data(), dOutData)
	}, eps, tol)

	sdpa.Forward(ctx, q, k, v, nil)
	grads, _ = sdpa.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil)
	dV = grads[2]

	checkGrad(t, "dV", v, dV, func() float32 {
		o, _ := sdpa.Forward(ctx, q, k, v, nil)
		return dotProduct(o.Data(), dOutData)
	}, eps, tol)
}

// TestSDPABackward_Causal_FiniteDiff verifies SDPA backward with causal masking.
func TestSDPABackward_Causal_FiniteDiff(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	headDim := 4
	batch := 2
	seqLen := 3

	sdpa := NewScaledDotProductAttention[float32](engine, headDim)
	sdpa.SetCausal(true)

	makeT := func(shape []int, seed int) *tensor.TensorNumeric[float32] {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(((i*7+seed)%19)-9) / 20.0
		}
		t, _ := tensor.New[float32](shape, data)
		return t
	}

	q := makeT([]int{batch, seqLen, headDim}, 3)
	k := makeT([]int{batch, seqLen, headDim}, 5)
	v := makeT([]int{batch, seqLen, headDim}, 11)

	out, err := sdpa.Forward(ctx, q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	dOutData := make([]float32, len(out.Data()))
	for i := range dOutData {
		dOutData[i] = float32(((i*11+5)%13)-6) / 10.0
	}
	dOut, _ := tensor.New[float32](out.Shape(), dOutData)

	grads, err := sdpa.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	eps := float32(1e-3)
	tol := float32(5e-2)

	checkGrad(t, "causal_dQ", q, grads[0], func() float32 {
		o, _ := sdpa.Forward(ctx, q, k, v, nil)
		return dotProduct(o.Data(), dOutData)
	}, eps, tol)

	sdpa.Forward(ctx, q, k, v, nil)
	grads, _ = sdpa.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil)

	checkGrad(t, "causal_dK", k, grads[1], func() float32 {
		o, _ := sdpa.Forward(ctx, q, k, v, nil)
		return dotProduct(o.Data(), dOutData)
	}, eps, tol)

	sdpa.Forward(ctx, q, k, v, nil)
	grads, _ = sdpa.Backward(ctx, types.FullBackprop, dOut, nil, nil, nil)

	checkGrad(t, "causal_dV", v, grads[2], func() float32 {
		o, _ := sdpa.Forward(ctx, q, k, v, nil)
		return dotProduct(o.Data(), dOutData)
	}, eps, tol)
}

func checkGrad(t *testing.T, name string, param, grad *tensor.TensorNumeric[float32], lossFn func() float32, eps, tol float32) {
	t.Helper()
	analyticalGrad := make([]float32, len(grad.Data()))
	copy(analyticalGrad, grad.Data())

	data := param.Data()
	numFailed := 0
	for i := range data {
		orig := data[i]

		data[i] = orig + eps
		lPlus := lossFn()

		data[i] = orig - eps
		lMinus := lossFn()

		data[i] = orig

		numerical := (lPlus - lMinus) / (2 * eps)
		a := analyticalGrad[i]
		diff := float32(math.Abs(float64(a - numerical)))
		denom := float32(math.Max(1.0, math.Max(math.Abs(float64(a)), math.Abs(float64(numerical)))))

		if diff/denom > tol {
			numFailed++
			if numFailed <= 5 {
				t.Errorf("%s[%d]: analytical=%.6f numerical=%.6f relErr=%.4f",
					name, i, a, numerical, diff/denom)
			}
		}
	}
	if numFailed > 0 {
		t.Fatalf("%s: %d/%d exceeded tol=%.4f", name, numFailed, len(data), tol)
	}
	t.Logf("%s: %d elements passed", name, len(data))
}
