package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestSSMLayer_OutputShape(t *testing.T) {
	engine := makeEngine()

	dState, dInput, dOutput := 4, 3, 2
	ssm, err := NewSSMLayer[float32](engine, dState, dInput, dOutput)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}

	batch, seqLen := 2, 5
	data := make([]float32, batch*seqLen*dInput)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dInput}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	output, err := ssm.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, seqLen, dOutput}
	if len(got) != len(want) {
		t.Fatalf("output shape rank = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestSSMLayer_ImpulseResponse(t *testing.T) {
	// Test with a known impulse response.
	// For a diagonal SSM with:
	//   A_diag = [-1], B = [[1]], C = [[1]], D = [[0]], dt = 1.0
	// The discretised system is:
	//   A_bar = exp(-1) ≈ 0.3679
	//   B_bar = (exp(-1) - 1) / (-1) = 1 - exp(-1) ≈ 0.6321
	//   x[0] = 0
	//   After impulse u=[1,0,0,...]:
	//     x[1] = A_bar*0 + B_bar*1 = 0.6321
	//     y[0] = C*x[0] + D*u[0] = 0 (x is updated then y uses new x... let's verify)
	// Actually in our implementation, x is updated first, then y is computed:
	//   Step 0: x = A_bar*0 + B_bar*1 = 0.6321, y = C*x + D*u = 0.6321 + 0 = 0.6321
	//   Step 1: x = A_bar*0.6321 + B_bar*0 = 0.2325, y = 0.2325
	//   Step 2: x = A_bar*0.2325 = 0.0855, y = 0.0855
	engine := makeEngine()

	dState, dInput, dOutput := 1, 1, 1
	ssm, err := NewSSMLayer[float32](engine, dState, dInput, dOutput)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}

	// Override parameters with known values.
	// A in log-space: A_diag = -exp(A_log). We want A_diag = -1, so A_log = 0.
	ssm.A.Value, _ = tensor.New[float32]([]int{1}, []float32{0.0})
	// B = [[1.0]]
	ssm.B.Value, _ = tensor.New[float32]([]int{1, 1}, []float32{1.0})
	// C = [[1.0]]
	ssm.C.Value, _ = tensor.New[float32]([]int{1, 1}, []float32{1.0})
	// D = [[0.0]]
	ssm.D.Value, _ = tensor.New[float32]([]int{1, 1}, []float32{0.0})
	// dt in log-space: dt = exp(log_dt). We want dt = 1.0, so log_dt = 0.
	ssm.Dt.Value, _ = tensor.New[float32]([]int{1}, []float32{0.0})

	// Impulse input: u = [1, 0, 0, 0]
	seqLen := 4
	inputData := []float32{1, 0, 0, 0}
	input, err := tensor.New[float32]([]int{1, seqLen, 1}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	output, err := ssm.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outData := output.Data()
	// Expected values:
	aBar := math.Exp(-1.0) // ≈ 0.3679
	bBar := 1.0 - aBar     // ≈ 0.6321
	expected := []float64{
		bBar,                // step 0: B_bar * 1
		aBar * bBar,         // step 1: A_bar * (B_bar)
		aBar * aBar * bBar,  // step 2: A_bar^2 * B_bar
		aBar * aBar * aBar * bBar, // step 3: A_bar^3 * B_bar
	}

	for i, want := range expected {
		got := float64(outData[i])
		if math.Abs(got-want) > 1e-4 {
			t.Errorf("y[%d] = %f, want %f (diff = %e)", i, got, want, math.Abs(got-want))
		}
	}
}

func TestSSMLayer_BatchGreaterThanOne(t *testing.T) {
	engine := makeEngine()

	dState, dInput, dOutput := 4, 3, 2
	ssm, err := NewSSMLayer[float32](engine, dState, dInput, dOutput)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}

	batch, seqLen := 3, 5
	data := make([]float32, batch*seqLen*dInput)
	for i := range data {
		data[i] = float32(i%7) * 0.1
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dInput}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	output, err := ssm.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, seqLen, dOutput}
	if len(got) != len(want) {
		t.Fatalf("output shape rank = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, got[i], want[i])
		}
	}

	// Verify batch independence: running each batch element separately should
	// produce the same result.
	for b := 0; b < batch; b++ {
		singleData := make([]float32, seqLen*dInput)
		copy(singleData, data[b*seqLen*dInput:(b+1)*seqLen*dInput])
		singleInput, err := tensor.New[float32]([]int{1, seqLen, dInput}, singleData)
		if err != nil {
			t.Fatalf("tensor.New single: %v", err)
		}
		singleOut, err := ssm.Forward(context.Background(), singleInput)
		if err != nil {
			t.Fatalf("Forward single batch %d: %v", b, err)
		}
		singleOutData := singleOut.Data()
		batchOutData := output.Data()
		for i := 0; i < seqLen*dOutput; i++ {
			batchVal := batchOutData[b*seqLen*dOutput+i]
			singleVal := singleOutData[i]
			if math.Abs(float64(batchVal-singleVal)) > 1e-5 {
				t.Errorf("batch %d, index %d: batched=%f, single=%f", b, i, batchVal, singleVal)
			}
		}
	}
}

func TestSSMLayer_DInputNotEqualDOutput(t *testing.T) {
	engine := makeEngine()

	// dInput=5, dOutput=3: different input and output dimensions.
	dState, dInput, dOutput := 8, 5, 3
	ssm, err := NewSSMLayer[float32](engine, dState, dInput, dOutput)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}

	batch, seqLen := 2, 6
	data := make([]float32, batch*seqLen*dInput)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, dInput}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	output, err := ssm.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, seqLen, dOutput}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestSSMLayer_Parameters(t *testing.T) {
	engine := makeEngine()

	ssm, err := NewSSMLayer[float32](engine, 4, 3, 2)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}

	params := ssm.Parameters()
	// A, B, C, D, dt = 5 parameters.
	if len(params) != 5 {
		t.Errorf("expected 5 parameters, got %d", len(params))
	}

	names := make(map[string]bool)
	for _, p := range params {
		names[p.Name] = true
	}
	for _, want := range []string{"ssm_A", "ssm_B", "ssm_C", "ssm_D", "ssm_dt"} {
		if !names[want] {
			t.Errorf("missing parameter %q", want)
		}
	}
}

func TestSSMLayer_OpType(t *testing.T) {
	engine := makeEngine()
	ssm, err := NewSSMLayer[float32](engine, 4, 3, 2)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}
	if got := ssm.OpType(); got != "SSM" {
		t.Errorf("OpType = %q, want %q", got, "SSM")
	}
}

func TestSSMLayer_Attributes(t *testing.T) {
	engine := makeEngine()
	ssm, err := NewSSMLayer[float32](engine, 4, 3, 2)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}
	attrs := ssm.Attributes()
	if attrs["d_state"] != 4 {
		t.Errorf("d_state = %v, want 4", attrs["d_state"])
	}
	if attrs["d_input"] != 3 {
		t.Errorf("d_input = %v, want 3", attrs["d_input"])
	}
	if attrs["d_output"] != 2 {
		t.Errorf("d_output = %v, want 2", attrs["d_output"])
	}
}

func TestNewSSMLayer_InvalidArgs(t *testing.T) {
	engine := makeEngine()

	tests := []struct {
		name    string
		dState  int
		dInput  int
		dOutput int
	}{
		{"zero dState", 0, 3, 2},
		{"negative dState", -1, 3, 2},
		{"zero dInput", 4, 0, 2},
		{"negative dInput", 4, -1, 2},
		{"zero dOutput", 4, 3, 0},
		{"negative dOutput", 4, 3, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSSMLayer[float32](engine, tt.dState, tt.dInput, tt.dOutput)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestSSMLayer_ForwardInputValidation(t *testing.T) {
	engine := makeEngine()
	ssm, err := NewSSMLayer[float32](engine, 4, 3, 2)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}

	ctx := context.Background()

	t.Run("wrong rank", func(t *testing.T) {
		input, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
		_, err := ssm.Forward(ctx, input)
		if err == nil {
			t.Error("expected error for 2D input")
		}
	})

	t.Run("wrong d_input", func(t *testing.T) {
		input, _ := tensor.New[float32]([]int{1, 4, 5}, make([]float32, 20))
		_, err := ssm.Forward(ctx, input)
		if err == nil {
			t.Error("expected error for wrong d_input")
		}
	})
}

func TestSSMLayer_FeedthroughD(t *testing.T) {
	// Verify that the D (feedthrough) matrix creates a direct input-to-output path.
	// Set A, B, C to zero so only D contributes.
	engine := makeEngine()

	dState, dInput, dOutput := 2, 2, 2
	ssm, err := NewSSMLayer[float32](engine, dState, dInput, dOutput)
	if err != nil {
		t.Fatalf("NewSSMLayer: %v", err)
	}

	// Zero out A (log-space), B, C. Set D to identity.
	ssm.A.Value, _ = tensor.New[float32]([]int{dState}, []float32{-100, -100}) // exp(-100)≈0
	ssm.B.Value, _ = tensor.New[float32]([]int{dState, dInput}, make([]float32, dState*dInput))
	ssm.C.Value, _ = tensor.New[float32]([]int{dOutput, dState}, make([]float32, dOutput*dState))
	ssm.D.Value, _ = tensor.New[float32]([]int{dOutput, dInput}, []float32{1, 0, 0, 1}) // identity
	ssm.Dt.Value, _ = tensor.New[float32]([]int{1}, []float32{0})

	input, _ := tensor.New[float32]([]int{1, 3, 2}, []float32{
		1, 2,
		3, 4,
		5, 6,
	})

	output, err := ssm.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outData := output.Data()
	expected := []float32{1, 2, 3, 4, 5, 6}
	for i, want := range expected {
		got := outData[i]
		if math.Abs(float64(got-want)) > 1e-4 {
			t.Errorf("output[%d] = %f, want %f", i, got, want)
		}
	}
}
