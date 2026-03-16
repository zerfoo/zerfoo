package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// helper to create a filled tensor with sequential values starting at start.
func fillTensor(t *testing.T, shape []int, start float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = start + float32(i)*0.1
	}
	out, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("fillTensor: %v", err)
	}
	return out
}

// helper to create a constant tensor.
func constTensor(t *testing.T, shape []int, val float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	n := 1
	for _, d := range shape {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = val
	}
	out, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("constTensor: %v", err)
	}
	return out
}

func newTestEngine() compute.Engine[float32] {
	return compute.NewCPUEngine[float32](numeric.Float32Ops{})
}

// TestMul_BroadcastShapes tests Mul with various broadcast patterns used in ONNX
// decomposed normalization (e.g., RMSNorm: x * weight where weight is [2048]).
func TestMul_BroadcastShapes(t *testing.T) {
	engine := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name   string
		aShape []int
		bShape []int
	}{
		{"3D_times_1D", []int{1, 1, 4}, []int{4}},
		{"3D_times_3D_broadcast", []int{1, 1, 4}, []int{1, 1, 4}},
		{"2D_times_1D", []int{1, 4}, []int{4}},
		{"3D_times_scalar", []int{1, 1, 4}, []int{1}},
		{"2D_cross_broadcast", []int{3, 1}, []int{1, 4}},
		{"3D_batch_broadcast", []int{2, 1, 4}, []int{1, 3, 4}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := fillTensor(t, tc.aShape, 1.0)
			b := fillTensor(t, tc.bShape, 2.0)

			mul := NewMul[float32](engine)
			out, err := mul.Forward(ctx, a, b)
			if err != nil {
				t.Fatalf("Mul Forward: %v", err)
			}

			// Verify output has correct number of elements.
			outShape := out.Shape()
			outSize := 1
			for _, d := range outShape {
				outSize *= d
			}
			if len(out.Data()) != outSize {
				t.Errorf("output data len = %d, want %d", len(out.Data()), outSize)
			}

			// Verify no NaN or Inf in output.
			for i, v := range out.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v (NaN/Inf)", i, v)
					break
				}
			}
		})
	}
}

// TestAdd_BroadcastShapes tests Add with various broadcast patterns used in ONNX
// bias addition and residual connections.
func TestAdd_BroadcastShapes(t *testing.T) {
	engine := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name   string
		aShape []int
		bShape []int
	}{
		{"3D_plus_1D_bias", []int{1, 1, 4}, []int{4}},
		{"3D_plus_3D_same", []int{1, 1, 4}, []int{1, 1, 4}},
		{"2D_plus_1D", []int{1, 4}, []int{4}},
		{"3D_plus_scalar", []int{1, 1, 4}, []int{1}},
		{"2D_cross_broadcast", []int{3, 1}, []int{1, 4}},
		{"batch_broadcast", []int{2, 1, 4}, []int{1, 3, 4}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := fillTensor(t, tc.aShape, 1.0)
			b := fillTensor(t, tc.bShape, 10.0)

			add := NewAdd[float32](engine)
			out, err := add.Forward(ctx, a, b)
			if err != nil {
				t.Fatalf("Add Forward: %v", err)
			}

			outShape := out.Shape()
			outSize := 1
			for _, d := range outShape {
				outSize *= d
			}
			if len(out.Data()) != outSize {
				t.Errorf("output data len = %d, want %d", len(out.Data()), outSize)
			}

			for i, v := range out.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v (NaN/Inf)", i, v)
					break
				}
			}
		})
	}
}

// TestDiv_BroadcastShapes tests Div with various broadcast patterns used in ONNX
// decomposed normalization (e.g., x / sqrt(variance + eps)).
func TestDiv_BroadcastShapes(t *testing.T) {
	engine := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name   string
		aShape []int
		bShape []int
	}{
		{"3D_div_1D", []int{1, 1, 4}, []int{4}},
		{"3D_div_scalar", []int{1, 1, 4}, []int{1}},
		{"3D_div_3D_same", []int{1, 1, 4}, []int{1, 1, 4}},
		{"2D_div_1D", []int{1, 4}, []int{4}},
		{"batch_div_scalar", []int{2, 3, 4}, []int{1}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a := fillTensor(t, tc.aShape, 1.0)
			// Use non-zero values for divisor.
			b := constTensor(t, tc.bShape, 2.0)

			div := NewDiv[float32](engine)
			out, err := div.Forward(ctx, a, b)
			if err != nil {
				t.Fatalf("Div Forward: %v", err)
			}

			outShape := out.Shape()
			outSize := 1
			for _, d := range outShape {
				outSize *= d
			}
			if len(out.Data()) != outSize {
				t.Errorf("output data len = %d, want %d", len(out.Data()), outSize)
			}

			// Verify division results are half of the input values.
			aData := a.Data()
			outData := out.Data()
			// For same-shape case, check element-wise.
			if len(aData) == len(outData) {
				for i := range outData {
					want := aData[i] / 2.0
					if diff := float32(math.Abs(float64(outData[i] - want))); diff > 1e-6 {
						t.Errorf("output[%d] = %v, want %v", i, outData[i], want)
						break
					}
				}
			}

			for i, v := range out.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v (NaN/Inf)", i, v)
					break
				}
			}
		})
	}
}

// TestPow_ScalarExponent tests Pow with scalar exponent broadcast, the pattern
// used in ONNX decomposed RMSNorm: x^2 where exponent is a scalar tensor [1].
func TestPow_ScalarExponent(t *testing.T) {
	engine := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name     string
		baseShape []int
		expShape  []int
		expVal    float32
	}{
		{"3D_pow_scalar_2", []int{1, 1, 4}, []int{1}, 2.0},
		{"2D_pow_scalar_2", []int{1, 4}, []int{1}, 2.0},
		{"1D_pow_scalar_2", []int{4}, []int{1}, 2.0},
		{"3D_pow_scalar_half", []int{1, 1, 4}, []int{1}, 0.5},
		{"3D_pow_1D_broadcast", []int{1, 1, 4}, []int{4}, 2.0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base := fillTensor(t, tc.baseShape, 1.0)
			exp := constTensor(t, tc.expShape, tc.expVal)

			pow := NewPow[float32](engine)
			out, err := pow.Forward(ctx, base, exp)
			if err != nil {
				t.Fatalf("Pow Forward: %v", err)
			}

			outShape := out.Shape()
			outSize := 1
			for _, d := range outShape {
				outSize *= d
			}
			if len(out.Data()) != outSize {
				t.Errorf("output data len = %d, want %d", len(out.Data()), outSize)
			}

			// For scalar exponent=2, verify x^2.
			if tc.expVal == 2.0 && len(tc.expShape) == 1 && tc.expShape[0] == 1 {
				baseData := base.Data()
				outData := out.Data()
				if len(baseData) == len(outData) {
					for i := range outData {
						want := baseData[i] * baseData[i]
						if diff := float32(math.Abs(float64(outData[i] - want))); diff > 1e-5 {
							t.Errorf("output[%d] = %v, want %v (x^2)", i, outData[i], want)
							break
						}
					}
				}
			}

			for i, v := range out.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v (NaN/Inf)", i, v)
					break
				}
			}
		})
	}
}
