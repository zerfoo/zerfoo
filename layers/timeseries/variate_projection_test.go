package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestNewVariateProjection_Valid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vp, err := NewVariateProjection[float32]("test_vp", engine, ops, 16, 32, 10)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}
	if vp.OpType() != "VariateProjection" {
		t.Errorf("OpType = %q, want %q", vp.OpType(), "VariateProjection")
	}
	if vp.Name() != "test_vp" {
		t.Errorf("Name = %q, want %q", vp.Name(), "test_vp")
	}
}

func TestNewVariateProjection_Invalid(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name        string
		layerName   string
		inputDim    int
		embedDim    int
		maxVariates int
	}{
		{"empty name", "", 16, 32, 10},
		{"zero inputDim", "vp", 0, 32, 10},
		{"negative inputDim", "vp", -1, 32, 10},
		{"zero embedDim", "vp", 16, 0, 10},
		{"negative embedDim", "vp", 16, -1, 10},
		{"zero maxVariates", "vp", 16, 32, 0},
		{"negative maxVariates", "vp", 16, 32, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewVariateProjection[float32](tt.layerName, engine, ops, tt.inputDim, tt.embedDim, tt.maxVariates)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestVariateProjection_Attributes(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vp, err := NewVariateProjection[float32]("vp", engine, ops, 16, 32, 10)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}
	attrs := vp.Attributes()
	if attrs["input_dim"] != 16 {
		t.Errorf("input_dim = %v, want 16", attrs["input_dim"])
	}
	if attrs["embed_dim"] != 32 {
		t.Errorf("embed_dim = %v, want 32", attrs["embed_dim"])
	}
	if attrs["max_variates"] != 10 {
		t.Errorf("max_variates = %v, want 10", attrs["max_variates"])
	}
}

func TestVariateProjection_Forward(t *testing.T) {
	tests := []struct {
		name        string
		batch       int
		numVariates int
		inputDim    int
		embedDim    int
		maxVariates int
		wantShape   []int
	}{
		{
			name:        "single variate",
			batch:       2,
			numVariates: 1,
			inputDim:    16,
			embedDim:    32,
			maxVariates: 20,
			wantShape:   []int{2, 1, 32},
		},
		{
			name:        "5 variates",
			batch:       3,
			numVariates: 5,
			inputDim:    8,
			embedDim:    16,
			maxVariates: 20,
			wantShape:   []int{3, 5, 16},
		},
		{
			name:        "20 variates",
			batch:       1,
			numVariates: 20,
			inputDim:    32,
			embedDim:    64,
			maxVariates: 20,
			wantShape:   []int{1, 20, 64},
		},
		{
			name:        "single batch single variate",
			batch:       1,
			numVariates: 1,
			inputDim:    4,
			embedDim:    8,
			maxVariates: 10,
			wantShape:   []int{1, 1, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := makeEngine()
			ops := numeric.Float32Ops{}
			vp, err := NewVariateProjection[float32]("vp", engine, ops, tt.inputDim, tt.embedDim, tt.maxVariates)
			if err != nil {
				t.Fatalf("NewVariateProjection: %v", err)
			}

			data := make([]float32, tt.batch*tt.numVariates*tt.inputDim)
			for i := range data {
				data[i] = float32(i) * 0.01
			}
			input, err := tensor.New[float32]([]int{tt.batch, tt.numVariates, tt.inputDim}, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			ctx := context.Background()
			out, err := vp.Forward(ctx, input)
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

func TestVariateProjection_ForwardWithMask(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch := 2
	numVar := 5
	inputDim := 8
	embedDim := 16

	vp, err := NewVariateProjection[float32]("vp", engine, ops, inputDim, embedDim, 20)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}

	// Create input [2, 5, 8].
	data := make([]float32, batch*numVar*inputDim)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, numVar, inputDim}, data)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}

	// Create mask [2, 5]: first 3 variates valid, last 2 padded.
	maskData := []float32{
		1, 1, 1, 0, 0,
		1, 1, 1, 0, 0,
	}
	mask, err := tensor.New[float32]([]int{batch, numVar}, maskData)
	if err != nil {
		t.Fatalf("tensor.New mask: %v", err)
	}

	ctx := context.Background()
	out, err := vp.Forward(ctx, input, mask)
	if err != nil {
		t.Fatalf("Forward with mask: %v", err)
	}

	// Verify shape.
	gotShape := out.Shape()
	wantShape := []int{batch, numVar, embedDim}
	for i := range gotShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], wantShape[i])
		}
	}

	// Verify masked variates are zeroed.
	outData := out.Data()
	for b := range batch {
		for v := 3; v < numVar; v++ {
			base := (b*numVar + v) * embedDim
			for d := range embedDim {
				if outData[base+d] != 0 {
					t.Errorf("batch=%d variate=%d dim=%d: got %f, want 0 (masked)", b, v, d, outData[base+d])
				}
			}
		}
		// Verify at least some unmasked variates are non-zero.
		base := b * numVar * embedDim
		allZero := true
		for d := range embedDim {
			if outData[base+d] != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("batch=%d variate=0: all zeros, expected non-zero output for valid variate", b)
		}
	}
}

func TestVariateProjection_ForwardErrors(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vp, err := NewVariateProjection[float32]("vp", engine, ops, 16, 32, 10)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}

	ctx := context.Background()

	t.Run("no inputs", func(t *testing.T) {
		_, err := vp.Forward(ctx)
		if err == nil {
			t.Error("expected error for no inputs")
		}
	})

	t.Run("three inputs", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 1, 16}, make([]float32, 16))
		_, err := vp.Forward(ctx, x, x, x)
		if err == nil {
			t.Error("expected error for three inputs")
		}
	})

	t.Run("2D input", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 16}, make([]float32, 16))
		_, err := vp.Forward(ctx, x)
		if err == nil {
			t.Error("expected error for 2D input")
		}
	})

	t.Run("wrong inputDim", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 1, 8}, make([]float32, 8))
		_, err := vp.Forward(ctx, x)
		if err == nil {
			t.Error("expected error for wrong inputDim")
		}
	})

	t.Run("exceeds maxVariates", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{1, 11, 16}, make([]float32, 11*16))
		_, err := vp.Forward(ctx, x)
		if err == nil {
			t.Error("expected error for exceeding maxVariates")
		}
	})

	t.Run("wrong mask shape", func(t *testing.T) {
		x, _ := tensor.New[float32]([]int{2, 3, 16}, make([]float32, 2*3*16))
		mask, _ := tensor.New[float32]([]int{2, 5}, make([]float32, 10))
		_, err := vp.Forward(ctx, x, mask)
		if err == nil {
			t.Error("expected error for wrong mask shape")
		}
	})
}

func TestVariateProjection_DifferentVariateCounts(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	inputDim := 8
	embedDim := 16
	maxVar := 20

	vp, err := NewVariateProjection[float32]("vp", engine, ops, inputDim, embedDim, maxVar)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}

	ctx := context.Background()

	// Test with varying numbers of variates using the same layer.
	for _, numVar := range []int{1, 5, 20} {
		t.Run(intToName(numVar), func(t *testing.T) {
			batch := 2
			data := make([]float32, batch*numVar*inputDim)
			for i := range data {
				data[i] = float32(i) * 0.01
			}
			input, err := tensor.New[float32]([]int{batch, numVar, inputDim}, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			out, err := vp.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward with %d variates: %v", numVar, err)
			}

			gotShape := out.Shape()
			if gotShape[0] != batch || gotShape[1] != numVar || gotShape[2] != embedDim {
				t.Errorf("shape = %v, want [%d, %d, %d]", gotShape, batch, numVar, embedDim)
			}
		})
	}
}

func TestVariateProjection_FreqEmbeddingDiffers(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	inputDim := 4
	embedDim := 8

	vp, err := NewVariateProjection[float32]("vp", engine, ops, inputDim, embedDim, 10)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}

	// Two variates with identical input data should produce different outputs
	// because of different frequency embeddings.
	batch := 1
	numVar := 2
	data := make([]float32, batch*numVar*inputDim)
	for i := range inputDim {
		val := float32(i) * 0.1
		data[i] = val
		data[inputDim+i] = val // same data for both variates
	}
	input, err := tensor.New[float32]([]int{batch, numVar, inputDim}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	ctx := context.Background()
	out, err := vp.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	outData := out.Data()
	var1 := outData[:embedDim]
	var2 := outData[embedDim : 2*embedDim]

	allSame := true
	for d := range embedDim {
		if var1[d] != var2[d] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("identical variate inputs produced identical outputs; frequency embedding should differentiate them")
	}
}

func TestVariateProjection_Parameters(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vp, err := NewVariateProjection[float32]("vp", engine, ops, 8, 16, 10)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}
	params := vp.Parameters()
	if len(params) != 3 {
		t.Fatalf("expected 3 parameters, got %d", len(params))
	}

	wantNames := []string{"vp_proj", "vp_bias", "vp_freq_emb"}
	for i, p := range params {
		if p.Name != wantNames[i] {
			t.Errorf("params[%d].Name = %q, want %q", i, p.Name, wantNames[i])
		}
	}

	// Check shapes.
	projShape := params[0].Value.Shape()
	if projShape[0] != 8 || projShape[1] != 16 {
		t.Errorf("proj shape = %v, want [8, 16]", projShape)
	}
	biasShape := params[1].Value.Shape()
	if biasShape[0] != 16 {
		t.Errorf("bias shape = %v, want [16]", biasShape)
	}
	freqShape := params[2].Value.Shape()
	if freqShape[0] != 10 || freqShape[1] != 16 {
		t.Errorf("freq_emb shape = %v, want [10, 16]", freqShape)
	}
}

func TestVariateProjection_SetName(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vp, err := NewVariateProjection[float32]("old", engine, ops, 8, 16, 10)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}
	vp.SetName("new")
	if vp.Name() != "new" {
		t.Errorf("Name = %q, want %q", vp.Name(), "new")
	}
	if vp.proj.Name != "new_proj" {
		t.Errorf("proj.Name = %q, want %q", vp.proj.Name, "new_proj")
	}
	if vp.bias.Name != "new_bias" {
		t.Errorf("bias.Name = %q, want %q", vp.bias.Name, "new_bias")
	}
	if vp.freqEmb.Name != "new_freq_emb" {
		t.Errorf("freqEmb.Name = %q, want %q", vp.freqEmb.Name, "new_freq_emb")
	}
}

func TestVariateProjection_OutputShape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	vp, err := NewVariateProjection[float32]("vp", engine, ops, 16, 32, 10)
	if err != nil {
		t.Fatalf("NewVariateProjection: %v", err)
	}
	shape := vp.OutputShape()
	want := []int{-1, -1, 32}
	for i := range shape {
		if shape[i] != want[i] {
			t.Errorf("OutputShape[%d] = %d, want %d", i, shape[i], want[i])
		}
	}
}

func intToName(n int) string {
	switch n {
	case 1:
		return "1_variate"
	case 5:
		return "5_variates"
	case 20:
		return "20_variates"
	default:
		return "variates"
	}
}
