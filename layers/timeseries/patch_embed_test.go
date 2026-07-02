package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func makeEngine() compute.Engine[float32] {
	return compute.NewCPUEngine(numeric.Float32Ops{})
}

func TestNewPatchEmbed_Valid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	pe, err := NewPatchEmbed[float32]("test_pe", engine, ops, 16, 32)
	if err != nil {
		t.Fatalf("NewPatchEmbed: %v", err)
	}
	if pe.OpType() != "PatchEmbed" {
		t.Errorf("OpType = %q, want %q", pe.OpType(), "PatchEmbed")
	}
	if pe.Name() != "test_pe" {
		t.Errorf("Name = %q, want %q", pe.Name(), "test_pe")
	}
}

func TestNewPatchEmbed_Invalid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name      string
		layerName string
		patchSize int
		embedDim  int
	}{
		{"empty name", "", 16, 32},
		{"zero patch_size", "pe", 0, 32},
		{"negative patch_size", "pe", -1, 32},
		{"zero embed_dim", "pe", 16, 0},
		{"negative embed_dim", "pe", 16, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewPatchEmbed[float32](tt.layerName, engine, ops, tt.patchSize, tt.embedDim)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestPatchEmbed_Attributes(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	pe, err := NewPatchEmbed[float32]("test_pe", engine, ops, 16, 32)
	if err != nil {
		t.Fatalf("NewPatchEmbed: %v", err)
	}
	attrs := pe.Attributes()
	if attrs["patch_size"] != 16 {
		t.Errorf("patch_size = %v, want 16", attrs["patch_size"])
	}
	if attrs["embed_dim"] != 32 {
		t.Errorf("embed_dim = %v, want 32", attrs["embed_dim"])
	}
}

func TestPatchEmbed_Forward(t *testing.T) {
	tests := []struct {
		name       string
		batch      int
		seqLen     int
		patchSize  int
		embedDim   int
		wantShape  []int
	}{
		{
			name:      "basic [2,96] patch_size=16",
			batch:     2,
			seqLen:    96,
			patchSize: 16,
			embedDim:  32,
			wantShape: []int{2, 6, 32},
		},
		{
			name:      "single batch [1,64] patch_size=8",
			batch:     1,
			seqLen:    64,
			patchSize: 8,
			embedDim:  16,
			wantShape: []int{1, 8, 16},
		},
		{
			name:      "patch_size equals seq_len",
			batch:     3,
			seqLen:    16,
			patchSize: 16,
			embedDim:  64,
			wantShape: []int{3, 1, 64},
		},
		{
			name:      "padding case [2,100] patch_size=16",
			batch:     2,
			seqLen:    100,
			patchSize: 16,
			embedDim:  32,
			wantShape: []int{2, 7, 32}, // ceil(100/16)=7
		},
		{
			name:      "padding case [1,10] patch_size=4",
			batch:     1,
			seqLen:    10,
			patchSize: 4,
			embedDim:  8,
			wantShape: []int{1, 3, 8}, // ceil(10/4)=3
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := makeEngine()
			ops := numeric.Float32Ops{}
			pe, err := NewPatchEmbed[float32]("pe", engine, ops, tt.patchSize, tt.embedDim)
			if err != nil {
				t.Fatalf("NewPatchEmbed: %v", err)
			}

			data := make([]float32, tt.batch*tt.seqLen)
			for i := range data {
				data[i] = float32(i) * 0.01
			}
			input, err := tensor.New[float32]([]int{tt.batch, tt.seqLen}, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			ctx := context.Background()
			out, err := pe.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			gotShape := out.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("output dims = %d, want %d", len(gotShape), len(tt.wantShape))
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], tt.wantShape[i])
				}
			}
		})
	}
}

func TestPatchEmbed_ForwardErrors(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	pe, err := NewPatchEmbed[float32]("pe", engine, ops, 16, 32)
	if err != nil {
		t.Fatalf("NewPatchEmbed: %v", err)
	}

	ctx := context.Background()

	t.Run("no inputs", func(t *testing.T) {
		_, err := pe.Forward(ctx)
		if err == nil {
			t.Error("expected error for no inputs")
		}
	})

	t.Run("two inputs", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 16}, make([]float32, 16))
		_, err := pe.Forward(ctx, x, x)
		if err == nil {
			t.Error("expected error for two inputs")
		}
	})

	t.Run("3D input", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 16, 4}, make([]float32, 64))
		_, err := pe.Forward(ctx, x)
		if err == nil {
			t.Error("expected error for 3D input")
		}
	})
}

func TestPatchEmbed_Backward(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch := 2
	seqLen := 32
	patchSize := 8
	embedDim := 16
	numPatches := seqLen / patchSize

	pe, err := NewPatchEmbed[float32]("pe", engine, ops, patchSize, embedDim)
	if err != nil {
		t.Fatalf("NewPatchEmbed: %v", err)
	}

	// Create input.
	inputData := make([]float32, batch*seqLen)
	for i := range inputData {
		inputData[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, seqLen}, inputData)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}

	ctx := context.Background()

	// Forward pass.
	out, err := pe.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Create output gradient (ones).
	gradData := make([]float32, batch*numPatches*embedDim)
	for i := range gradData {
		gradData[i] = 1.0
	}
	outputGrad, err := tensor.New[float32]([]int{batch, numPatches, embedDim}, gradData)
	if err != nil {
		t.Fatalf("tensor.New outputGrad: %v", err)
	}

	_ = out
	grads, err := pe.Backward(ctx, types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("expected 1 gradient, got %d", len(grads))
	}

	// Check input gradient shape matches input shape.
	gradShape := grads[0].Shape()
	if gradShape[0] != batch || gradShape[1] != seqLen {
		t.Errorf("input gradient shape = %v, want [%d, %d]", gradShape, batch, seqLen)
	}

	// Verify projection weight gradient was accumulated.
	if pe.proj.Gradient == nil {
		t.Fatal("projection weight gradient is nil")
	}
	wgShape := pe.proj.Gradient.Shape()
	if wgShape[0] != patchSize || wgShape[1] != embedDim {
		t.Errorf("weight gradient shape = %v, want [%d, %d]", wgShape, patchSize, embedDim)
	}
}

func TestPatchEmbed_BackwardFiniteDiff(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch := 1
	seqLen := 8
	patchSize := 4
	embedDim := 2

	pe, err := NewPatchEmbed[float32]("pe", engine, ops, patchSize, embedDim)
	if err != nil {
		t.Fatalf("NewPatchEmbed: %v", err)
	}

	inputData := make([]float32, batch*seqLen)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}

	ctx := context.Background()
	eps := float32(1e-3)
	numPatches := seqLen / patchSize

	// Compute analytical gradient.
	input, _ := tensor.New[float32]([]int{batch, seqLen}, append([]float32(nil), inputData...))
	out, err := pe.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Use ones as output gradient -> gradient of sum of all outputs.
	gradData := make([]float32, batch*numPatches*embedDim)
	for i := range gradData {
		gradData[i] = 1.0
	}
	outputGrad, _ := tensor.New[float32]([]int{batch, numPatches, embedDim}, gradData)
	_ = out

	grads, err := pe.Backward(ctx, types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	analyticalGrad := grads[0].Data()

	// Finite difference for each input element.
	for i := 0; i < batch*seqLen; i++ {
		// f(x + eps)
		dataPlus := append([]float32(nil), inputData...)
		dataPlus[i] += eps
		xPlus, _ := tensor.New[float32]([]int{batch, seqLen}, dataPlus)
		outPlus, err := pe.Forward(ctx, xPlus)
		if err != nil {
			t.Fatalf("Forward+: %v", err)
		}
		sumPlus := sum(outPlus.Data())

		// f(x - eps)
		dataMinus := append([]float32(nil), inputData...)
		dataMinus[i] -= eps
		xMinus, _ := tensor.New[float32]([]int{batch, seqLen}, dataMinus)
		outMinus, err := pe.Forward(ctx, xMinus)
		if err != nil {
			t.Fatalf("Forward-: %v", err)
		}
		sumMinus := sum(outMinus.Data())

		numerical := (sumPlus - sumMinus) / (2 * float64(eps))
		analytical := float64(analyticalGrad[i])

		if math.Abs(numerical-analytical) > 1e-2 {
			t.Errorf("input[%d]: numerical=%f, analytical=%f, diff=%f",
				i, numerical, analytical, math.Abs(numerical-analytical))
		}
	}
}

func TestPatchEmbed_Parameters(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	pe, err := NewPatchEmbed[float32]("pe", engine, ops, 8, 16)
	if err != nil {
		t.Fatalf("NewPatchEmbed: %v", err)
	}
	params := pe.Parameters()
	if len(params) != 1 {
		t.Fatalf("expected 1 parameter, got %d", len(params))
	}
	if params[0].Name != "pe_proj" {
		t.Errorf("parameter name = %q, want %q", params[0].Name, "pe_proj")
	}
	pShape := params[0].Value.Shape()
	if pShape[0] != 8 || pShape[1] != 16 {
		t.Errorf("parameter shape = %v, want [8, 16]", pShape)
	}
}

func TestPatchEmbed_SetName(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	pe, err := NewPatchEmbed[float32]("old", engine, ops, 8, 16)
	if err != nil {
		t.Fatalf("NewPatchEmbed: %v", err)
	}
	pe.SetName("new")
	if pe.Name() != "new" {
		t.Errorf("Name = %q, want %q", pe.Name(), "new")
	}
	if pe.proj.Name != "new_proj" {
		t.Errorf("proj.Name = %q, want %q", pe.proj.Name, "new_proj")
	}
}

func sum(data []float32) float64 {
	var s float64
	for _, v := range data {
		s += float64(v)
	}
	return s
}
