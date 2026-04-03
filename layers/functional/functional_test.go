package functional_test

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// ---------- LayerNorm ----------

func TestLayerNorm_Float32(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	eps := float32(1e-5)

	shapes := []struct {
		name  string
		shape []int
		dim   int
	}{
		{"1D", []int{4}, 4},
		{"2D", []int{2, 4}, 4},
		{"3D", []int{2, 3, 4}, 4},
	}

	for _, tc := range shapes {
		t.Run(tc.name, func(t *testing.T) {
			data := make([]float32, product(tc.shape))
			for i := range data {
				data[i] = float32(i+1) * 0.1
			}

			x, err := tensor.New[float32](tc.shape, data)
			assertNoErr(t, err)

			ln, err := normalization.NewLayerNormalization[float32](engine, tc.dim,
				normalization.WithLayerNormEpsilon[float32](eps))
			assertNoErr(t, err)

			want, err := ln.Forward(ctx, x)
			assertNoErr(t, err)

			params := ln.Parameters()
			gamma := params[0].Value
			beta := params[1].Value

			got, err := functional.LayerNorm(ctx, engine, x, gamma, beta, eps)
			assertNoErr(t, err)

			assertF32Close(t, want.Data(), got.Data(), 1e-6)
		})
	}
}

func TestLayerNorm_Float64(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](&numeric.Float64Ops{})
	eps := float64(1e-5)

	shapes := []struct {
		name  string
		shape []int
		dim   int
	}{
		{"1D", []int{4}, 4},
		{"2D", []int{3, 5}, 5},
		{"3D", []int{2, 3, 5}, 5},
	}

	for _, tc := range shapes {
		t.Run(tc.name, func(t *testing.T) {
			data := make([]float64, product(tc.shape))
			for i := range data {
				data[i] = float64(i+1) * 0.1
			}

			x, err := tensor.New[float64](tc.shape, data)
			assertNoErr(t, err)

			ln, err := normalization.NewLayerNormalization[float64](engine, tc.dim,
				normalization.WithLayerNormEpsilon[float64](eps))
			assertNoErr(t, err)

			want, err := ln.Forward(ctx, x)
			assertNoErr(t, err)

			params := ln.Parameters()
			gamma := params[0].Value
			beta := params[1].Value

			got, err := functional.LayerNorm(ctx, engine, x, gamma, beta, eps)
			assertNoErr(t, err)

			assertF64Close(t, want.Data(), got.Data(), 1e-10)
		})
	}
}

// ---------- RMSNorm ----------

func TestRMSNorm_Float32(t *testing.T) {
	ctx := context.Background()
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	eps := float32(1e-6)

	shapes := []struct {
		name  string
		shape []int
		dim   int
	}{
		{"1D", []int{4}, 4},
		{"2D", []int{2, 4}, 4},
		{"3D", []int{2, 3, 4}, 4},
	}

	for _, tc := range shapes {
		t.Run(tc.name, func(t *testing.T) {
			data := make([]float32, product(tc.shape))
			for i := range data {
				data[i] = float32(i+1) * 0.1
			}

			x, err := tensor.New[float32](tc.shape, data)
			assertNoErr(t, err)

			rms, err := normalization.NewRMSNorm[float32]("test", engine, ops, tc.dim,
				normalization.WithRMSNormEpsilon[float32](eps))
			assertNoErr(t, err)

			gain := rms.Parameters()[0].Value

			// Build reference with multi-step math (same ops as functional)
			// to avoid divergence from the fused float32 path in the layer.
			wantRef, err := rmsnormRef(ctx, engine, x, gain, eps)
			assertNoErr(t, err)

			got, err := functional.RMSNorm(ctx, engine, x, gain, eps)
			assertNoErr(t, err)

			assertF32Close(t, wantRef.Data(), got.Data(), 1e-6)
		})
	}
}

func TestRMSNorm_Float64(t *testing.T) {
	ctx := context.Background()
	ops := &numeric.Float64Ops{}
	engine := compute.NewCPUEngine[float64](ops)
	eps := float64(1e-6)

	shapes := []struct {
		name  string
		shape []int
		dim   int
	}{
		{"1D", []int{4}, 4},
		{"2D", []int{3, 5}, 5},
		{"3D", []int{2, 3, 5}, 5},
	}

	for _, tc := range shapes {
		t.Run(tc.name, func(t *testing.T) {
			data := make([]float64, product(tc.shape))
			for i := range data {
				data[i] = float64(i+1) * 0.1
			}

			x, err := tensor.New[float64](tc.shape, data)
			assertNoErr(t, err)

			rms, err := normalization.NewRMSNorm[float64]("test", engine, ops, tc.dim,
				normalization.WithRMSNormEpsilon[float64](eps))
			assertNoErr(t, err)

			gain := rms.Parameters()[0].Value

			// For float64 there's no fused path, so the layer's Forward
			// uses the same multi-step ops and we can compare directly.
			want, err := rms.Forward(ctx, x)
			assertNoErr(t, err)

			got, err := functional.RMSNorm(ctx, engine, x, gain, eps)
			assertNoErr(t, err)

			assertF64Close(t, want.Data(), got.Data(), 1e-10)
		})
	}
}

// ---------- LayerNorm nil-input errors ----------

func TestLayerNorm_NilInputs(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	x, _ := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	scale, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	bias, _ := tensor.New[float32]([]int{4}, []float32{0, 0, 0, 0})

	tests := []struct {
		name  string
		x     *tensor.TensorNumeric[float32]
		scale *tensor.TensorNumeric[float32]
		bias  *tensor.TensorNumeric[float32]
	}{
		{"nil x", nil, scale, bias},
		{"nil scale", x, nil, bias},
		{"nil bias", x, scale, nil},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := functional.LayerNorm(ctx, engine, tc.x, tc.scale, tc.bias, float32(1e-5))
			if err == nil {
				t.Fatal("expected error for nil input, got nil")
			}
		})
	}
}

// ---------- RMSNorm nil-input errors ----------

func TestRMSNorm_NilInputs(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})

	x, _ := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	scale, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})

	tests := []struct {
		name  string
		x     *tensor.TensorNumeric[float32]
		scale *tensor.TensorNumeric[float32]
	}{
		{"nil x", nil, scale},
		{"nil scale", x, nil},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := functional.RMSNorm(ctx, engine, tc.x, tc.scale, float32(1e-6))
			if err == nil {
				t.Fatal("expected error for nil input, got nil")
			}
		})
	}
}

// ---------- helpers ----------

func rmsnormRef[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	x, gain *tensor.TensorNumeric[T], eps T) (*tensor.TensorNumeric[T], error) {

	sq, err := engine.Mul(ctx, x, x, nil)
	if err != nil {
		return nil, err
	}
	meanSq, err := engine.ReduceMean(ctx, sq, len(x.Shape())-1, true)
	if err != nil {
		return nil, err
	}
	meanSqEps, err := engine.AddScalar(ctx, meanSq, eps, nil)
	if err != nil {
		return nil, err
	}
	rsqrt, err := engine.Rsqrt(ctx, meanSqEps, nil)
	if err != nil {
		return nil, err
	}
	normed, err := engine.Mul(ctx, x, rsqrt, nil)
	if err != nil {
		return nil, err
	}
	return engine.Mul(ctx, normed, gain, nil)
}

func product(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

func assertNoErr(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func assertF32Close(t *testing.T, want, got []float32, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("length mismatch: want %d, got %d", len(want), len(got))
	}
	for i := range want {
		d := math.Abs(float64(want[i]) - float64(got[i]))
		if d > tol {
			t.Errorf("index %d: want %v, got %v (diff %v > tol %v)", i, want[i], got[i], d, tol)
		}
	}
}

func assertF64Close(t *testing.T, want, got []float64, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("length mismatch: want %d, got %d", len(want), len(got))
	}
	for i := range want {
		d := math.Abs(want[i] - got[i])
		if d > tol {
			t.Errorf("index %d: want %v, got %v (diff %v > tol %v)", i, want[i], got[i], d, tol)
		}
	}
}
