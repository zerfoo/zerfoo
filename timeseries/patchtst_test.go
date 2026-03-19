package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestPatchTST_Forward(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	tests := []struct {
		name      string
		config    PatchTSTConfig
		inputDims []int // shape of input tensor
		wantShape []int // expected output shape
		wantErr   bool
	}{
		{
			name: "univariate basic",
			config: PatchTSTConfig{
				InputLength:        24,
				PatchLength:        8,
				Stride:             4,
				DModel:             16,
				NHeads:             2,
				NLayers:            1,
				OutputDim:          4,
				ChannelIndependent: false,
			},
			inputDims: []int{2, 24},
			wantShape: []int{2, 4},
		},
		{
			name: "multivariate channel independent",
			config: PatchTSTConfig{
				InputLength:        24,
				PatchLength:        8,
				Stride:             4,
				DModel:             16,
				NHeads:             2,
				NLayers:            1,
				OutputDim:          4,
				ChannelIndependent: true,
			},
			inputDims: []int{2, 3, 24},
			wantShape: []int{2, 3, 4},
		},
		{
			name: "single batch single channel",
			config: PatchTSTConfig{
				InputLength:        16,
				PatchLength:        4,
				Stride:             4,
				DModel:             8,
				NHeads:             2,
				NLayers:            1,
				OutputDim:          2,
				ChannelIndependent: false,
			},
			inputDims: []int{1, 16},
			wantShape: []int{1, 2},
		},
		{
			name: "two encoder layers",
			config: PatchTSTConfig{
				InputLength:        16,
				PatchLength:        4,
				Stride:             4,
				DModel:             8,
				NHeads:             2,
				NLayers:            2,
				OutputDim:          3,
				ChannelIndependent: false,
			},
			inputDims: []int{1, 16},
			wantShape: []int{1, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := NewPatchTST(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewPatchTST: %v", err)
			}

			totalElems := 1
			for _, d := range tt.inputDims {
				totalElems *= d
			}
			data := make([]float32, totalElems)
			for i := range data {
				data[i] = float32(i) * 0.01
			}

			input, err := tensor.New[float32](tt.inputDims, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			output, err := model.Forward(ctx, input)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			gotShape := output.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("output shape rank = %d, want %d (got %v, want %v)", len(gotShape), len(tt.wantShape), gotShape, tt.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("output shape[%d] = %d, want %d (got %v, want %v)", i, gotShape[i], tt.wantShape[i], gotShape, tt.wantShape)
				}
			}

			// Verify output contains finite values.
			outData := output.Data()
			for i, v := range outData {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("output[%d] = %v, want finite", i, v)
					break
				}
			}
		})
	}
}

func TestPatchTST_Patching(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := PatchTSTConfig{
		InputLength:        12,
		PatchLength:        4,
		Stride:             2,
		DModel:             8,
		NHeads:             2,
		NLayers:            1,
		OutputDim:          2,
		ChannelIndependent: false,
	}

	// Verify NumPatches calculation: (12 - 4) / 2 + 1 = 5
	wantPatches := 5
	if got := config.NumPatches(); got != wantPatches {
		t.Fatalf("NumPatches() = %d, want %d", got, wantPatches)
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Create input [1, 12] with sequential values.
	data := make([]float32, 12)
	for i := range data {
		data[i] = float32(i)
	}
	input, err := tensor.New[float32]([]int{1, 12}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	// Extract patches directly to verify correctness.
	patches, err := model.extractPatches(ctx, input)
	if err != nil {
		t.Fatalf("extractPatches: %v", err)
	}

	// Expected shape: [1, 5, 4]
	gotShape := patches.Shape()
	wantShape := []int{1, 5, 4}
	if len(gotShape) != len(wantShape) {
		t.Fatalf("patches shape = %v, want %v", gotShape, wantShape)
	}
	for i := range gotShape {
		if gotShape[i] != wantShape[i] {
			t.Fatalf("patches shape[%d] = %d, want %d", i, gotShape[i], wantShape[i])
		}
	}

	// Verify patch contents: patches extracted at strides 0, 2, 4, 6, 8.
	pData := patches.Data()
	expected := [][]float32{
		{0, 1, 2, 3},   // offset 0
		{2, 3, 4, 5},   // offset 2
		{4, 5, 6, 7},   // offset 4
		{6, 7, 8, 9},   // offset 6
		{8, 9, 10, 11}, // offset 8
	}
	for p := range wantPatches {
		for j := range 4 {
			idx := p*4 + j
			if pData[idx] != expected[p][j] {
				t.Errorf("patch[%d][%d] = %v, want %v", p, j, pData[idx], expected[p][j])
			}
		}
	}

	// Verify model produces output with correct shape.
	output, err := model.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	outShape := output.Shape()
	if outShape[0] != 1 || outShape[1] != 2 {
		t.Errorf("output shape = %v, want [1, 2]", outShape)
	}
}

func TestPatchTST_ChannelIndependence(t *testing.T) {
	engine, ops := newTestEngine()
	ctx := context.Background()

	config := PatchTSTConfig{
		InputLength:        16,
		PatchLength:        4,
		Stride:             4,
		DModel:             8,
		NHeads:             2,
		NLayers:            1,
		OutputDim:          2,
		ChannelIndependent: true,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Create multivariate input [1, 3, 16] where channels have different data.
	channels := 3
	data := make([]float32, channels*16)
	for c := range channels {
		for i := range 16 {
			data[c*16+i] = float32(c*100 + i)
		}
	}
	input, err := tensor.New[float32]([]int{1, channels, 16}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	output, err := model.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Output should be [1, 3, 2] — each channel processed independently.
	gotShape := output.Shape()
	wantShape := []int{1, 3, 2}
	if len(gotShape) != len(wantShape) {
		t.Fatalf("output shape = %v, want %v", gotShape, wantShape)
	}
	for i := range gotShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, gotShape[i], wantShape[i])
		}
	}

	// Verify channel independence: running each channel separately should
	// produce the same result as running all channels together.
	outData := output.Data()
	for c := range channels {
		chData := data[c*16 : (c+1)*16]
		chInput, err := tensor.New[float32]([]int{1, 16}, chData)
		if err != nil {
			t.Fatalf("tensor.New channel %d: %v", c, err)
		}

		chOutput, err := model.Forward(ctx, chInput)
		if err != nil {
			t.Fatalf("Forward channel %d: %v", c, err)
		}

		chOutData := chOutput.Data()
		for i := range chOutData {
			multiIdx := c*config.OutputDim + i
			if math.Abs(float64(chOutData[i]-outData[multiIdx])) > 1e-5 {
				t.Errorf("channel %d output[%d] = %v (independent) vs %v (batched), diff > 1e-5",
					c, i, chOutData[i], outData[multiIdx])
			}
		}
	}
}

func TestPatchTST_Predict(t *testing.T) {
	engine, ops := newTestEngine()

	config := PatchTSTConfig{
		InputLength:        16,
		PatchLength:        4,
		Stride:             4,
		DModel:             8,
		NHeads:             2,
		NLayers:            1,
		OutputDim:          3,
		ChannelIndependent: true,
	}

	model, err := NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Single channel.
	input1 := [][]float64{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}
	out1, err := model.Predict(input1)
	if err != nil {
		t.Fatalf("Predict single channel: %v", err)
	}
	if len(out1) != 1 || len(out1[0]) != 3 {
		t.Fatalf("Predict single channel: got %d channels with %d outputs, want 1 channel with 3", len(out1), len(out1[0]))
	}

	// Multi-channel.
	input2 := [][]float64{
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		{100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115},
	}
	out2, err := model.Predict(input2)
	if err != nil {
		t.Fatalf("Predict multi-channel: %v", err)
	}
	if len(out2) != 2 || len(out2[0]) != 3 || len(out2[1]) != 3 {
		t.Fatalf("Predict multi-channel: unexpected shape")
	}

	// Verify finite values.
	for c, ch := range out2 {
		for i, v := range ch {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("Predict output[%d][%d] = %v, want finite", c, i, v)
			}
		}
	}
}

func TestNewPatchTST_Validation(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  PatchTSTConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 8, Stride: 4,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
		},
		{
			name: "zero input length",
			config: PatchTSTConfig{
				InputLength: 0, PatchLength: 8, Stride: 4,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "zero patch length",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 0, Stride: 4,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "patch longer than input",
			config: PatchTSTConfig{
				InputLength: 4, PatchLength: 8, Stride: 2,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "d_model not divisible by n_heads",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 8, Stride: 4,
				DModel: 15, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "zero stride",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 8, Stride: 0,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 4,
			},
			wantErr: true,
		},
		{
			name: "zero output dim",
			config: PatchTSTConfig{
				InputLength: 24, PatchLength: 8, Stride: 4,
				DModel: 16, NHeads: 2, NLayers: 1, OutputDim: 0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewPatchTST(tt.config, engine, ops)
			if tt.wantErr && err == nil {
				t.Fatal("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
