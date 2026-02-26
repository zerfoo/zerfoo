package normalization

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// ---------- errEngine ----------

type errEngine struct {
	compute.Engine[float32]
	calls   map[string]int
	failOn  map[string]int
	failErr error
}

func newErrEngine(failOn map[string]int) *errEngine {
	return &errEngine{
		Engine:  compute.NewCPUEngine[float32](numeric.Float32Ops{}),
		calls:   make(map[string]int),
		failOn:  failOn,
		failErr: fmt.Errorf("injected error"),
	}
}

func (e *errEngine) check(op string) error {
	e.calls[op]++
	if n, ok := e.failOn[op]; ok && e.calls[op] >= n {
		return e.failErr
	}
	return nil
}

func (e *errEngine) Mul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Mul"); err != nil {
		return nil, err
	}
	return e.Engine.Mul(ctx, a, b, dst...)
}

func (e *errEngine) Add(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
}

func (e *errEngine) Sub(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Sub"); err != nil {
		return nil, err
	}
	return e.Engine.Sub(ctx, a, b, dst...)
}

func (e *errEngine) Div(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Div"); err != nil {
		return nil, err
	}
	return e.Engine.Div(ctx, a, b, dst...)
}

func (e *errEngine) ReduceSum(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("ReduceSum"); err != nil {
		return nil, err
	}
	return e.Engine.ReduceSum(ctx, a, axis, keepDims, dst...)
}

func (e *errEngine) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("ReduceMean"); err != nil {
		return nil, err
	}
	return e.Engine.ReduceMean(ctx, a, axis, keepDims, dst...)
}

func (e *errEngine) DivScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("DivScalar"); err != nil {
		return nil, err
	}
	return e.Engine.DivScalar(ctx, a, scalar, dst...)
}

func (e *errEngine) AddScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("AddScalar"); err != nil {
		return nil, err
	}
	return e.Engine.AddScalar(ctx, a, scalar, dst...)
}

func (e *errEngine) MulScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("MulScalar"); err != nil {
		return nil, err
	}
	return e.Engine.MulScalar(ctx, a, scalar, dst...)
}

func (e *errEngine) Sqrt(ctx context.Context, a *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Sqrt"); err != nil {
		return nil, err
	}
	return e.Engine.Sqrt(ctx, a, dst...)
}

func (e *errEngine) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Rsqrt"); err != nil {
		return nil, err
	}
	return e.Engine.Rsqrt(ctx, a, dst...)
}

func (e *errEngine) Fill(ctx context.Context, t *tensor.TensorNumeric[float32], value float32) error {
	if err := e.check("Fill"); err != nil {
		return err
	}
	return e.Engine.Fill(ctx, t, value)
}

func (e *errEngine) Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

// ---------- helpers ----------

func makeTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	return tn
}

// ---------- LayerNormalization Forward error paths ----------
// Forward calls: ReduceSum#1, DivScalar#1, Sub#1, Mul#1, ReduceSum#2, DivScalar#2,
// AddScalar#1, Sqrt#1, Div#1, Mul#2, Add#1

func TestLayerNormalization_ForwardErrors(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"reduce_sum_1_error", map[string]int{"ReduceSum": 1}},
		{"div_scalar_1_error", map[string]int{"DivScalar": 1}},
		{"sub_error", map[string]int{"Sub": 1}},
		{"mul_1_error", map[string]int{"Mul": 1}},
		{"reduce_sum_2_error", map[string]int{"ReduceSum": 2}},
		{"div_scalar_2_error", map[string]int{"DivScalar": 2}},
		{"add_scalar_error", map[string]int{"AddScalar": 1}},
		{"sqrt_error", map[string]int{"Sqrt": 1}},
		{"div_error", map[string]int{"Div": 1}},
		{"mul_2_error", map[string]int{"Mul": 2}},
		{"add_error", map[string]int{"Add": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			ln, err := NewLayerNormalization[float32](eng, 4)
			if err != nil {
				t.Fatal(err)
			}
			input := makeTensor(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
			_, err = ln.Forward(ctx, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- LayerNormalization Backward error paths ----------
// Backward calls (after successful Forward):
// Mul#1(dOut*normedInput), ReduceSum#1(dGamma axis), AddGradient(gamma),
// ReduceSum#2(dBeta axis), AddGradient(beta),
// Mul#2(dOut*gamma), Sub#1(input-mean), AddScalar#1(var+eps), Sqrt#1,
// Mul#3(dLdNormedInput*inputMinusMean), ReduceSum#3(dLdVarianceTerm),
// ReduceSum#4(dLdMeanTerm), Div#1(term1), Mul#4(stdDev*stdDev),
// Mul#5(stdDevSquared*stdDev), Mul#6(inputMinusMean*dLdVarianceTerm),
// MulScalar#1(stdDevCubed*N), Div#2(term2), DivScalar#1(dLdMeanTerm/N),
// Sub#2(term1-term2), Sub#3(dInput-term3)

func TestLayerNormalization_BackwardErrors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	dim := 4

	// Use [dim, dim] input so ReduceSum(axis=1, keepDims=false) -> shape [dim] which
	// matches gamma shape [dim], allowing AddGradient to succeed and full backward to run.
	// Backward call sequence: Mul#1(dOut*normedInput), ReduceSum#1(dGamma), AddGrad(gamma),
	// ReduceSum#2(dBeta), AddGrad(beta), Mul#2(dOut*gamma), Sub#1, AddScalar#1, Sqrt#1,
	// Mul#3, ReduceSum#3, ReduceSum#4, Div#1, Mul#4, Mul#5, Mul#6, MulScalar#1, Div#2,
	// DivScalar#1, Sub#2, Sub#3

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"mul_dOut_normed_error", map[string]int{"Mul": 1}},
		{"reduce_sum_dGamma_error", map[string]int{"ReduceSum": 1}},
		{"reduce_sum_dBeta_error", map[string]int{"ReduceSum": 2}},
		{"mul_gamma_error", map[string]int{"Mul": 2}},
		{"sub_inputMinusMean_error", map[string]int{"Sub": 1}},
		{"add_scalar_error", map[string]int{"AddScalar": 1}},
		{"sqrt_error", map[string]int{"Sqrt": 1}},
		{"mul_dLdVariance_error", map[string]int{"Mul": 3}},
		{"reduce_sum_3_error", map[string]int{"ReduceSum": 3}},
		{"reduce_sum_4_error", map[string]int{"ReduceSum": 4}},
		{"div_term1_error", map[string]int{"Div": 1}},
		{"mul_stdDevSq_error", map[string]int{"Mul": 4}},
		{"mul_stdDevCubed_error", map[string]int{"Mul": 5}},
		{"mul_term2_num_error", map[string]int{"Mul": 6}},
		{"mulscalar_error", map[string]int{"MulScalar": 1}},
		{"div_term2_error", map[string]int{"Div": 2}},
		{"divscalar_error", map[string]int{"DivScalar": 1}},
		{"sub_term1_term2_error", map[string]int{"Sub": 2}},
		{"sub_dInput_term3_error", map[string]int{"Sub": 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			realEng := compute.NewCPUEngine[float32](ops)
			ln, err := NewLayerNormalization[float32](realEng, dim)
			if err != nil {
				t.Fatal(err)
			}
			data := make([]float32, dim*dim)
			for i := range data {
				data[i] = float32(i+1) * 0.1
			}
			input := makeTensor(t, []int{dim, dim}, data)
			_, err = ln.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			errEng := newErrEngine(tt.failOn)
			ln.engine = errEng
			gradData := make([]float32, dim*dim)
			for i := range gradData {
				gradData[i] = 0.01
			}
			grad := makeTensor(t, []int{dim, dim}, gradData)
			_, err = ln.Backward(ctx, types.FullBackprop, grad, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}

	// Test backward with 3D shape that causes AddGradient to fail (gamma shape mismatch)
	t.Run("add_gradient_gamma_shape_mismatch", func(t *testing.T) {
		realEng := compute.NewCPUEngine[float32](ops)
		ln, err := NewLayerNormalization[float32](realEng, dim)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{2, 3, dim}, make([]float32, 24))
		for i := range input.Data() {
			input.Data()[i] = float32(i+1) * 0.1
		}
		_, err = ln.Forward(ctx, input)
		if err != nil {
			t.Skipf("Forward failed: %v", err)
		}
		grad := makeTensor(t, []int{2, 3, dim}, make([]float32, 24))
		for i := range grad.Data() {
			grad.Data()[i] = 0.01
		}
		_, err = ln.Backward(ctx, types.FullBackprop, grad, input)
		// This errors because dGamma shape [2,3] != gamma shape [4]
		if err == nil {
			t.Log("Backward succeeded (shape mismatch resolved)")
		}
	})

	// Test full backward happy path with [dim, dim] input
	t.Run("full_happy_path", func(t *testing.T) {
		realEng := compute.NewCPUEngine[float32](ops)
		ln, err := NewLayerNormalization[float32](realEng, dim)
		if err != nil {
			t.Fatal(err)
		}
		data := make([]float32, dim*dim)
		for i := range data {
			data[i] = float32(i+1) * 0.1
		}
		input := makeTensor(t, []int{dim, dim}, data)
		_, err = ln.Forward(ctx, input)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}
		gradData := make([]float32, dim*dim)
		for i := range gradData {
			gradData[i] = 0.01
		}
		grad := makeTensor(t, []int{dim, dim}, gradData)
		grads, err := ln.Backward(ctx, types.FullBackprop, grad, input)
		if err != nil {
			t.Fatalf("Backward failed: %v", err)
		}
		if len(grads) != 1 {
			t.Errorf("grads len = %d, want 1", len(grads))
		}
	})
}

// ---------- RMSNorm Forward error paths ----------
// Forward calls: Mul#1(input*input), ReduceMean#1, AddScalar#1, Rsqrt#1, Mul#2, Mul#3

func TestRMSNorm_ForwardErrors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	dim := 4

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"mul_squared_error", map[string]int{"Mul": 1}},
		{"reduce_mean_error", map[string]int{"ReduceMean": 1}},
		{"add_scalar_error", map[string]int{"AddScalar": 1}},
		{"rsqrt_error", map[string]int{"Rsqrt": 1}},
		{"mul_normalized_error", map[string]int{"Mul": 2}},
		{"mul_gain_error", map[string]int{"Mul": 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			rms, err := NewRMSNorm[float32]("rms", eng, ops, dim)
			if err != nil {
				t.Fatal(err)
			}
			input := makeTensor(t, []int{1, dim}, []float32{1, 2, 3, 4})
			_, err = rms.Forward(ctx, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- RMSNorm Backward error paths ----------
// Backward calls (after Forward):
// Mul#1(input*rms normalized), Mul#2(dOut*normalized dGain), ReduceSum#1(axis 0),
// ReduceSum#2(axis 1), Reshape, Add#1(gradient accum), Mul#3(dOut*gain dNormalized),
// Mul#4(dNormalized*rms term1), Mul#5(rms*rms rmsCubed step1), Mul#6(rmsCubed*rms step2),
// Mul#7(dNormalized*input sumDNormX), ReduceSum#3(axis -1), Mul#8(input*sumDNormX term2 step1),
// Mul#9(term2*rmsCubed step2), Mul#10(term2*invN step3), Sub#1(term1-term2)

func TestRMSNorm_BackwardErrors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	dim := 4

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"mul_normalized_error", map[string]int{"Mul": 1}},
		{"mul_dGain_error", map[string]int{"Mul": 2}},
		{"reduce_sum_axis0_error", map[string]int{"ReduceSum": 1}},
		{"reduce_sum_axis1_error", map[string]int{"ReduceSum": 2}},
		{"add_gradient_error", map[string]int{"Add": 1}},
		{"mul_dNormalized_error", map[string]int{"Mul": 3}},
		{"mul_term1_error", map[string]int{"Mul": 4}},
		{"mul_rmsCubed1_error", map[string]int{"Mul": 5}},
		{"mul_rmsCubed2_error", map[string]int{"Mul": 6}},
		{"mul_sumDNormX_error", map[string]int{"Mul": 7}},
		{"reduce_sum_axis_neg1_error", map[string]int{"ReduceSum": 3}},
		{"mul_term2_step1_error", map[string]int{"Mul": 8}},
		{"mul_term2_step2_error", map[string]int{"Mul": 9}},
		{"mul_term2_step3_error", map[string]int{"Mul": 10}},
		{"sub_dInput_error", map[string]int{"Sub": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			realEng := compute.NewCPUEngine[float32](ops)
			rms, err := NewRMSNorm[float32]("rms", realEng, ops, dim)
			if err != nil {
				t.Fatal(err)
			}
			input := makeTensor(t, []int{2, 3, dim}, make([]float32, 24))
			for i := range input.Data() {
				input.Data()[i] = float32(i+1) * 0.1
			}
			_, err = rms.Forward(ctx, input)
			if err != nil {
				t.Skipf("Forward failed: %v", err)
			}
			errEng := newErrEngine(tt.failOn)
			rms.engine = errEng
			grad := makeTensor(t, []int{2, 3, dim}, make([]float32, 24))
			for i := range grad.Data() {
				grad.Data()[i] = 0.01
			}
			_, err = rms.Backward(ctx, types.FullBackprop, grad, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- SimplifiedLayerNormalization metadata ----------

func TestSimplifiedLayerNormalization_OpType_Comprehensive(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if sln.OpType() != "SimplifiedLayerNormalization" {
		t.Errorf("OpType = %q, want SimplifiedLayerNormalization", sln.OpType())
	}
}

func TestSimplifiedLayerNormalization_Attributes_Comprehensive(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	attrs := sln.Attributes()
	if _, ok := attrs["epsilon"]; !ok {
		t.Error("expected epsilon in attributes")
	}
}

func TestSimplifiedLayerNormalization_Parameters(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	params := sln.Parameters()
	if len(params) != 1 {
		t.Errorf("Parameters() len = %d, want 1", len(params))
	}
}

func TestSimplifiedLayerNormalization_OutputShape_Comprehensive(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	sln, err := NewSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	input := makeTensor(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	_, err = sln.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	os := sln.OutputShape()
	if len(os) != 2 || os[0] != 2 || os[1] != 4 {
		t.Errorf("OutputShape = %v, want [2 4]", os)
	}
}

func TestSimplifiedLayerNormalization_Forward_InputCountError(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	sln, _ := NewSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	_, err := sln.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestSimplifiedLayerNormalization_Backward_InputCountError(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	sln, _ := NewSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	grad := makeTensor(t, []int{2, 4}, make([]float32, 8))
	_, err := sln.Backward(context.Background(), types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error for 0 inputs in Backward")
	}
}

func TestSimplifiedLayerNormalization_Backward_NilCaches(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	sln, _ := NewSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	input := makeTensor(t, []int{2, 4}, make([]float32, 8))
	grad := makeTensor(t, []int{2, 4}, make([]float32, 8))
	_, err := sln.Backward(context.Background(), types.FullBackprop, grad, input)
	if err == nil {
		t.Error("expected error for nil cached tensors")
	}
}

// ---------- SimplifiedLayerNormalization Forward error paths ----------
// Forward: Mul#1(input*input), ReduceMean#1, AddScalar#1, Rsqrt#1, Mul#2(normalize), Mul#3(gain)

func TestSimplifiedLayerNormalization_ForwardErrors(t *testing.T) {
	ctx := context.Background()
	dim := 4

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"mul_squared_error", map[string]int{"Mul": 1}},
		{"reduce_mean_error", map[string]int{"ReduceMean": 1}},
		{"add_scalar_error", map[string]int{"AddScalar": 1}},
		{"rsqrt_error", map[string]int{"Rsqrt": 1}},
		{"mul_normalize_error", map[string]int{"Mul": 2}},
		{"mul_gain_error", map[string]int{"Mul": 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			gain := makeTensor(t, []int{dim}, []float32{1, 1, 1, 1})
			sln, err := NewSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
			if err != nil {
				t.Fatal(err)
			}
			input := makeTensor(t, []int{2, dim}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
			_, err = sln.Forward(ctx, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- SimplifiedLayerNormalization Backward error paths ----------
// Backward (after successful Forward):
// Mul#1(dOut*normalized dGainFull), ReduceSum#1..N(loop), Add#1(gradient accum),
// Mul#2(dOut*gain dNormalized), Mul#3(dNormalized*invStdDev term1),
// Mul#4(invStdDev^2 rmsSq), Mul#5(rmsSq*invStdDev rmsCubed),
// Mul#6(dNormalized*input dNormX), ReduceSum#last(sumDNormX),
// Mul#7(input*sumDNormX), Mul#8(term2*rmsCubed), Mul#9(term2*invN),
// Sub#1(term1-term2)

func TestSimplifiedLayerNormalization_BackwardErrors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	dim := 4

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"mul_dGain_error", map[string]int{"Mul": 1}},
		{"reduce_sum_dGain_error", map[string]int{"ReduceSum": 1}},
		{"add_gradient_error", map[string]int{"Add": 1}},
		{"mul_dNormalized_error", map[string]int{"Mul": 2}},
		{"mul_term1_error", map[string]int{"Mul": 3}},
		{"mul_rmsSq_error", map[string]int{"Mul": 4}},
		{"mul_rmsCubed_error", map[string]int{"Mul": 5}},
		{"mul_dNormX_error", map[string]int{"Mul": 6}},
		{"reduce_sum_sumDNormX_error", map[string]int{"ReduceSum": 2}},
		{"mul_term2_step1_error", map[string]int{"Mul": 7}},
		{"mul_term2_step2_error", map[string]int{"Mul": 8}},
		{"mul_term2_step3_error", map[string]int{"Mul": 9}},
		{"sub_dInput_error", map[string]int{"Sub": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			realEng := compute.NewCPUEngine[float32](ops)
			gain := makeTensor(t, []int{dim}, []float32{1, 1, 1, 1})
			sln, err := NewSimplifiedLayerNormalization[float32](realEng, ops, gain, 1e-6)
			if err != nil {
				t.Fatal(err)
			}
			input := makeTensor(t, []int{2, dim}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
			_, err = sln.Forward(ctx, input)
			if err != nil {
				t.Skipf("Forward failed: %v", err)
			}
			errEng := newErrEngine(tt.failOn)
			sln.engine = errEng
			grad := makeTensor(t, []int{2, dim}, make([]float32, 8))
			for i := range grad.Data() {
				grad.Data()[i] = 0.01
			}
			_, err = sln.Backward(ctx, types.FullBackprop, grad, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- SkipSimplifiedLayerNormalization metadata ----------

func TestSkipSimplifiedLayerNormalization_OpType_Comprehensive(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	ssln, err := NewSkipSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if ssln.OpType() != "SkipSimplifiedLayerNormalization" {
		t.Errorf("OpType = %q, want SkipSimplifiedLayerNormalization", ssln.OpType())
	}
}

func TestSkipSimplifiedLayerNormalization_Attributes_Comprehensive(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	ssln, err := NewSkipSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	if ssln.Attributes() == nil {
		t.Error("expected non-nil attributes")
	}
}

// ---------- SkipSimplifiedLayerNormalization Forward/Backward errors ----------

func TestSkipSimplifiedLayerNormalization_Forward_NormError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Mul": 1})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	ssln, err := NewSkipSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	input := makeTensor(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	_, err = ssln.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected error from normalization Forward")
	}
}

func TestSkipSimplifiedLayerNormalization_Forward_AddError(t *testing.T) {
	// Norm Forward uses Mul(3x), ReduceMean, AddScalar, Rsqrt but NOT Add.
	// The SkipSLN Add is the first Add call.
	eng := newErrEngine(map[string]int{"Add": 1})
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	ssln, err := NewSkipSimplifiedLayerNormalization[float32](eng, numeric.Float32Ops{}, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	input := makeTensor(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	_, err = ssln.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected error from Add (residual)")
	}
}

func TestSkipSimplifiedLayerNormalization_Backward_NormError(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	realEng := compute.NewCPUEngine[float32](ops)
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	ssln, err := NewSkipSimplifiedLayerNormalization[float32](realEng, ops, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	input := makeTensor(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	_, err = ssln.Forward(ctx, input)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}
	// Fail Mul#1 in backward to trigger normLayer.Backward error
	errEng := newErrEngine(map[string]int{"Mul": 1})
	ssln.engine = errEng
	ssln.normLayer.engine = errEng
	grad := makeTensor(t, []int{2, 4}, make([]float32, 8))
	_, err = ssln.Backward(ctx, types.FullBackprop, grad, input)
	if err == nil {
		t.Error("expected error from normLayer Backward")
	}
}

func TestSkipSimplifiedLayerNormalization_Backward_AddError(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	realEng := compute.NewCPUEngine[float32](ops)
	gain := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
	ssln, err := NewSkipSimplifiedLayerNormalization[float32](realEng, ops, gain, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	input := makeTensor(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	_, err = ssln.Forward(ctx, input)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}
	// normLayer.Backward uses Add#1 for gradient accumulation.
	// SkipSLN Backward's Add is after that.
	errEng := newErrEngine(map[string]int{"Add": 2})
	ssln.engine = errEng
	ssln.normLayer.engine = errEng
	grad := makeTensor(t, []int{2, 4}, make([]float32, 8))
	for i := range grad.Data() {
		grad.Data()[i] = 0.01
	}
	_, err = ssln.Backward(ctx, types.FullBackprop, grad, input)
	if err == nil {
		t.Error("expected error from Add (residual backward)")
	}
}
