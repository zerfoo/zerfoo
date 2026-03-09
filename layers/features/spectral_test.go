package features

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func newTestEngine() (compute.Engine[float32], numeric.Arithmetic[float32]) {
	ops := numeric.Float32Ops{}
	return compute.NewCPUEngine[float32](ops), ops
}

func TestSpectralFingerprint_Forward(t *testing.T) {
	engine, ops := newTestEngine()

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
			layer := NewSpectralFingerprint[float32](engine, ops, tt.outputDim)
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

			if !tt.wantErr {
				got := output.Data()
				for i, w := range tt.want {
					if i >= len(got) {
						t.Errorf("output too short: len %d, want at least %d", len(got), i+1)
						break
					}
					if diff := math.Abs(float64(got[i]) - float64(w)); diff > 1e-3 {
						t.Errorf("output[%d] = %v, want %v (diff %v)", i, got[i], w, diff)
					}
				}
			}
		})
	}
}

func TestSpectralFingerprint_Backward(t *testing.T) {
	engine, ops := newTestEngine()
	layer := NewSpectralFingerprint[float32](engine, ops, 4)
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
	got := inputGrad[0].Data()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("grad[%d] = %v, want 0", i, got[i])
		}
	}
}

func TestSpectralFingerprint_Parameters(t *testing.T) {
	engine, ops := newTestEngine()
	layer := NewSpectralFingerprint[float32](engine, ops, 4)
	if params := layer.Parameters(); len(params) != 0 {
		t.Errorf("Parameters() = %v, want empty slice", params)
	}
}

func TestSpectralFingerprint_OpType(t *testing.T) {
	engine, ops := newTestEngine()
	layer := NewSpectralFingerprint[float32](engine, ops, 4)
	if opType := layer.OpType(); opType != "SpectralFingerprint" {
		t.Errorf("OpType() = %s, want SpectralFingerprint", opType)
	}
}

func TestSpectralFingerprint_Attributes(t *testing.T) {
	engine, ops := newTestEngine()
	layer := NewSpectralFingerprint[float32](engine, ops, 4)
	attrs := layer.Attributes()
	if len(attrs) != 1 {
		t.Fatalf("Expected 1 attribute, got %d", len(attrs))
	}
	if outputDim, ok := attrs["output_dim"]; !ok || outputDim != 4 {
		t.Errorf("Attributes()['output_dim'] = %v, want 4", outputDim)
	}
}
