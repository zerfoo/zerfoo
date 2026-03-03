package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestAttentionHead_OpType(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ah := NewAttentionHead[float32](engine, 8, 4)

	if got := ah.OpType(); got != "AttentionHead" {
		t.Errorf("OpType() = %q, want %q", got, "AttentionHead")
	}
}

func TestAttentionHead_Attributes(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ah := NewAttentionHead[float32](engine, 8, 4)

	attrs := ah.Attributes()
	headDim, ok := attrs["head_dim"].(int)
	if !ok {
		t.Fatal("expected head_dim in attributes")
	}
	if headDim != 4 {
		t.Errorf("head_dim = %d, want 4", headDim)
	}
}

func TestAttentionHead_OutputShape(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ah := NewAttentionHead[float32](engine, 8, 4)

	shape := ah.OutputShape()
	if len(shape) != 3 {
		t.Fatalf("OutputShape length = %d, want 3", len(shape))
	}
	if shape[2] != 4 {
		t.Errorf("OutputShape[2] = %d, want 4 (headDim)", shape[2])
	}
}

func TestAttentionHead_Forward_WrongInputCount(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ah := NewAttentionHead[float32](engine, 8, 4)

	t1, _ := tensor.New[float32]([]int{1, 2, 8}, nil)
	t2, _ := tensor.New[float32]([]int{1, 2, 8}, nil)

	_, err := ah.Forward(context.Background(), t1, t2)
	if err == nil {
		t.Error("expected error for wrong input count")
	}
}

func TestAttentionHead_Forward_Non3DInput(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ah := NewAttentionHead[float32](engine, 8, 4)

	t1, _ := tensor.New[float32]([]int{8}, nil)
	_, err := ah.Forward(context.Background(), t1)
	if err == nil {
		t.Error("expected error for non-3D input")
	}
}

func TestNewAttentionHead_PanicOnInvalidInputDim(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for inputDim <= 0")
		}
	}()
	NewAttentionHead[float32](engine, 0, 4)
}

func TestNewAttentionHead_PanicOnInvalidHeadDim(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for headDim <= 0")
		}
	}()
	NewAttentionHead[float32](engine, 8, 0)
}

func TestAttentionHead_Backward_WrongInputCount(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ah := NewAttentionHead[float32](engine, 8, 4)

	dOut, _ := tensor.New[float32]([]int{1, 2, 4}, nil)
	_, err := ah.Backward(context.Background(), 0, dOut)
	if err == nil {
		t.Error("expected error for missing inputs in Backward")
	}
}

func TestAttentionHead_Backward_Non3DInput(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ah := NewAttentionHead[float32](engine, 8, 4)

	dOut, _ := tensor.New[float32]([]int{4}, nil)
	input2D, _ := tensor.New[float32]([]int{2, 8}, nil)
	_, err := ah.Backward(context.Background(), 0, dOut, input2D)
	if err == nil {
		t.Error("expected error for non-3D input in Backward")
	}
}

func TestNewAttentionHead_WithOptions(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	optCalled := false
	opt := func(o *AttentionHeadOptions[float32]) {
		optCalled = true
	}
	ah := NewAttentionHead[float32](engine, 8, 4, opt)
	if !optCalled {
		t.Error("option was not called")
	}
	if ah == nil {
		t.Error("expected non-nil attention head")
	}
}

func TestAttentionHead_Forward_EngineErrors(t *testing.T) {
	// Forward engine call sequence per Dense.Forward (3 projections):
	//   Linear.Forward: MatMul, Bias.Forward: Add
	// Then SDPA.Forward: Transpose, MatMul, MulScalar, Softmax, MatMul
	// Total: MatMul=5, Add=3, Transpose=1, MulScalar=1, Softmax=1
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"QProj_MatMul", map[string]int{"MatMul": 1}},
		{"QProj_BiasAdd", map[string]int{"Add": 1}},
		{"KProj_MatMul", map[string]int{"MatMul": 2}},
		{"KProj_BiasAdd", map[string]int{"Add": 2}},
		{"VProj_MatMul", map[string]int{"MatMul": 3}},
		{"VProj_BiasAdd", map[string]int{"Add": 3}},
		{"SDPA_Transpose", map[string]int{"Transpose": 1}},
		{"SDPA_QK_MatMul", map[string]int{"MatMul": 4}},
		{"SDPA_MulScalar", map[string]int{"MulScalar": 1}},
		{"SDPA_Softmax", map[string]int{"Softmax": 1}},
		{"SDPA_Final_MatMul", map[string]int{"MatMul": 5}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fe := newFailingEngine(tc.failOn)
			ah := NewAttentionHead[float32](fe, 8, 4)

			input, _ := tensor.New[float32]([]int{1, 3, 8}, nil)
			for i := range input.Data() {
				input.Data()[i] = float32(i%5+1) * 0.01
			}
			_, err := ah.Forward(context.Background(), input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestAttentionHead_Backward_EngineErrors(t *testing.T) {
	// Forward consumes: MatMul=5, Add=3, Transpose=1, MulScalar=1, Softmax=1
	// SDPA.Backward: Transpose x3, MatMul x4, Mul x2, ReduceSum x1, Sub x1, MulScalar x1
	// Dense.Backward (each): Bias.Backward(ReduceSum x1) + Linear.Backward(Reshape x2, Transpose x1, MatMul x1, Add x1, Transpose x1, MatMul x1)
	// AH.Backward.Add: x2
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"SDPA_Bwd_Transpose1", map[string]int{"Transpose": 2}},
		{"SDPA_Bwd_MatMul1", map[string]int{"MatMul": 6}},
		{"SDPA_Bwd_Transpose2", map[string]int{"Transpose": 3}},
		{"SDPA_Bwd_MatMul2", map[string]int{"MatMul": 7}},
		{"SDPA_Bwd_Mul", map[string]int{"Mul": 1}},
		{"SDPA_Bwd_ReduceSum", map[string]int{"ReduceSum": 1}},
		{"SDPA_Bwd_Sub", map[string]int{"Sub": 1}},
		{"SDPA_Bwd_MulScalar", map[string]int{"MulScalar": 2}},
		{"SDPA_Bwd_MatMul3", map[string]int{"MatMul": 8}},
		{"SDPA_Bwd_Transpose3", map[string]int{"Transpose": 4}},
		{"SDPA_Bwd_MatMul4", map[string]int{"MatMul": 9}},
		// VProj backward: Sum(bias), Reshape x2, Transpose, MatMul, Add, Transpose, MatMul
		{"VProj_Bwd_Sum", map[string]int{"Sum": 1}},
		{"VProj_Bwd_Reshape", map[string]int{"Reshape": 1}},
		{"VProj_Bwd_Transpose", map[string]int{"Transpose": 5}},
		{"VProj_Bwd_MatMul", map[string]int{"MatMul": 10}},
		{"VProj_Bwd_GradAdd", map[string]int{"Add": 4}},
		{"VProj_Bwd_Transpose2", map[string]int{"Transpose": 6}},
		{"VProj_Bwd_MatMul2", map[string]int{"MatMul": 11}},
		// KProj backward
		{"KProj_Bwd_Sum", map[string]int{"Sum": 2}},
		{"KProj_Bwd_Reshape", map[string]int{"Reshape": 3}},
		// QProj backward
		{"QProj_Bwd_Sum", map[string]int{"Sum": 3}},
		{"QProj_Bwd_Reshape", map[string]int{"Reshape": 5}},
		// AH sum gradients
		{"AH_Bwd_Add1", map[string]int{"Add": 7}},
		{"AH_Bwd_Add2", map[string]int{"Add": 8}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fe := newFailingEngine(tc.failOn)
			ah := NewAttentionHead[float32](fe, 8, 4)

			input, _ := tensor.New[float32]([]int{1, 3, 8}, nil)
			for i := range input.Data() {
				input.Data()[i] = float32(i%5+1) * 0.01
			}

			out, err := ah.Forward(context.Background(), input)
			if err != nil {
				t.Skipf("Forward failed (test targets Backward): %v", err)
			}

			dOut, _ := tensor.New[float32](out.Shape(), nil)
			for i := range dOut.Data() {
				dOut.Data()[i] = 1.0
			}

			_, err = ah.Backward(context.Background(), types.FullBackprop, dOut, input)
			if err == nil {
				t.Error("expected error in Backward")
			}
		})
	}
}
