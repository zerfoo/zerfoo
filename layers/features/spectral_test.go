package features

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestSpectralFingerprint_Forward(t *testing.T) {
	tests := []struct {
		name      string
		outputDim int
		inputData []float32
		want      []float32
		wantErr   bool
	}{
		{
			name:      "simple case",
			outputDim: 4,
			inputData: []float32{1, 2, 3, 4, 5, 6, 7, 8},
			want:      []float32{10.452504, 5.656854, 4.329569, 4.0},
		},
		{
			name:      "output dim larger than input",
			outputDim: 8,
			inputData: []float32{1, 2, 3, 4},
			want:      []float32{2.828427, 2, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "empty input",
			outputDim: 4,
			inputData: []float32{},
			want:      []float32{0, 0, 0, 0},
		},
		{
			name:      "nil input",
			outputDim: 4,
			inputData: nil,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer := NewSpectralFingerprint[float32](tt.outputDim)
			var input *tensor.TensorNumeric[float32]
			var err error
			if tt.inputData != nil {
				input, err = tensor.New[float32]([]int{1, len(tt.inputData)}, tt.inputData)
				if err != nil {
					t.Fatalf("Failed to create input tensor: %v", err)
				}
			}

			output, err := layer.Forward(context.Background(), input)

			if (err != nil) != tt.wantErr {
				t.Errorf("SpectralFingerprint.Forward() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && !cmp.Equal(output.Data(), tt.want, cmpopts.EquateApprox(0, 1e-6)) {
				t.Errorf("SpectralFingerprint.Forward() = %v, want %v", output.Data(), tt.want)
			}
		})
	}
}

func TestSpectralFingerprint_Backward(t *testing.T) {
	layer := NewSpectralFingerprint[float32](4)
	input, _ := tensor.New[float32]([]int{1, 8}, make([]float32, 8))
	outputGrad, _ := tensor.New[float32]([]int{1, 4}, make([]float32, 4))

	inputGrad, err := layer.Backward(context.Background(), types.FullBackprop, outputGrad, input)
	if err != nil {
		t.Fatalf("Backward() error = %v", err)
	}

	if len(inputGrad) != 1 {
		t.Fatalf("Expected 1 input gradient, got %d", len(inputGrad))
	}

	want := make([]float32, 8)
	if !cmp.Equal(inputGrad[0].Data(), want) {
		t.Errorf("Backward() = %v, want %v", inputGrad[0].Data(), want)
	}
}

func TestSpectralFingerprint_Parameters(t *testing.T) {
	layer := NewSpectralFingerprint[float32](4)
	if params := layer.Parameters(); len(params) != 0 {
		t.Errorf("Parameters() = %v, want empty slice", params)
	}
}

func TestSpectralFingerprint_OpType(t *testing.T) {
	layer := NewSpectralFingerprint[float32](4)
	if opType := layer.OpType(); opType != "SpectralFingerprint" {
		t.Errorf("OpType() = %s, want SpectralFingerprint", opType)
	}
}

func TestSpectralFingerprint_Attributes(t *testing.T) {
	layer := NewSpectralFingerprint[float32](4)
	attrs := layer.Attributes()
	if len(attrs) != 1 {
		t.Fatalf("Expected 1 attribute, got %d", len(attrs))
	}
	if outputDim, ok := attrs["output_dim"]; !ok || outputDim != 4 {
		t.Errorf("Attributes()['output_dim'] = %v, want 4", outputDim)
	}
}
