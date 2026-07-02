package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func testEngine() compute.Engine[float32] {
	return compute.NewCPUEngine(numeric.Float32Ops{})
}

func TestRegimeDetector(t *testing.T) {
	engine := testEngine()
	ops := numeric.Float32Ops{}

	cfg := RegimeConfig{
		InputDim:   10,
		HiddenDim:  32,
		NumLayers:  2,
		SeqLen:     60,
		NumClasses: 4,
	}

	rd, err := BuildRegimeDetector[float32](cfg, engine, ops)
	if err != nil {
		t.Fatalf("BuildRegimeDetector: %v", err)
	}

	// Synthetic input: [batch=2, seqLen=60, inputDim=10]
	batch, seqLen, inputDim := 2, 60, 10
	data := make([]float32, batch*seqLen*inputDim)
	for i := range data {
		data[i] = float32(i%7) * 0.1
	}
	input, err := tensor.New[float32]([]int{batch, seqLen, inputDim}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()
	output, err := rd.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Verify output shape is [2, 4]
	outShape := output.Shape()
	if len(outShape) != 2 || outShape[0] != batch || outShape[1] != cfg.NumClasses {
		t.Fatalf("output shape = %v, want [%d, %d]", outShape, batch, cfg.NumClasses)
	}

	// Verify softmax: each row sums to 1.0
	outData := output.Data()
	for b := 0; b < batch; b++ {
		var sum float64
		for c := 0; c < cfg.NumClasses; c++ {
			v := float64(outData[b*cfg.NumClasses+c])
			if v < 0 || v > 1 {
				t.Errorf("batch %d class %d: prob %f not in [0, 1]", b, c, v)
			}
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-5 {
			t.Errorf("batch %d: softmax sum = %f, want 1.0", b, sum)
		}
	}
}

func TestRegimeDetector_OpType(t *testing.T) {
	engine := testEngine()
	ops := numeric.Float32Ops{}
	cfg := RegimeConfig{InputDim: 5, HiddenDim: 8, NumLayers: 1, SeqLen: 10, NumClasses: 4}
	rd, err := BuildRegimeDetector[float32](cfg, engine, ops)
	if err != nil {
		t.Fatalf("BuildRegimeDetector: %v", err)
	}
	if got := rd.OpType(); got != "RegimeDetector" {
		t.Errorf("OpType = %q, want %q", got, "RegimeDetector")
	}
}

func TestRegimeDetector_OutputShape(t *testing.T) {
	engine := testEngine()
	ops := numeric.Float32Ops{}
	cfg := RegimeConfig{InputDim: 5, HiddenDim: 8, NumLayers: 1, SeqLen: 10, NumClasses: 4}
	rd, err := BuildRegimeDetector[float32](cfg, engine, ops)
	if err != nil {
		t.Fatalf("BuildRegimeDetector: %v", err)
	}
	shape := rd.OutputShape()
	if len(shape) != 2 || shape[1] != 4 {
		t.Errorf("OutputShape = %v, want [-1, 4]", shape)
	}
}

func TestRegimeDetector_Parameters(t *testing.T) {
	engine := testEngine()
	ops := numeric.Float32Ops{}
	cfg := RegimeConfig{InputDim: 5, HiddenDim: 8, NumLayers: 2, SeqLen: 10, NumClasses: 4}
	rd, err := BuildRegimeDetector[float32](cfg, engine, ops)
	if err != nil {
		t.Fatalf("BuildRegimeDetector: %v", err)
	}
	// 2 GRU layers x 6 params each + 2 classifier params = 14
	params := rd.Parameters()
	want := 2*6 + 2
	if len(params) != want {
		t.Errorf("Parameters count = %d, want %d", len(params), want)
	}
}

func TestBuildRegimeDetector_InvalidConfig(t *testing.T) {
	engine := testEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name string
		cfg  RegimeConfig
	}{
		{"zero InputDim", RegimeConfig{InputDim: 0, HiddenDim: 8, NumLayers: 1, SeqLen: 10}},
		{"zero HiddenDim", RegimeConfig{InputDim: 5, HiddenDim: 0, NumLayers: 1, SeqLen: 10}},
		{"zero NumLayers", RegimeConfig{InputDim: 5, HiddenDim: 8, NumLayers: 0, SeqLen: 10}},
		{"zero SeqLen", RegimeConfig{InputDim: 5, HiddenDim: 8, NumLayers: 1, SeqLen: 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := BuildRegimeDetector[float32](tt.cfg, engine, ops)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestRegimeDetector_InvalidInput(t *testing.T) {
	engine := testEngine()
	ops := numeric.Float32Ops{}
	cfg := RegimeConfig{InputDim: 5, HiddenDim: 8, NumLayers: 1, SeqLen: 10, NumClasses: 4}
	rd, err := BuildRegimeDetector[float32](cfg, engine, ops)
	if err != nil {
		t.Fatalf("BuildRegimeDetector: %v", err)
	}

	ctx := context.Background()

	// Wrong number of inputs
	input, _ := tensor.New[float32]([]int{2, 10, 5}, make([]float32, 100))
	_, err = rd.Forward(ctx, input, input)
	if err == nil {
		t.Error("expected error for multiple inputs, got nil")
	}

	// Wrong dimensionality (2D instead of 3D)
	input2d, _ := tensor.New[float32]([]int{2, 10}, make([]float32, 20))
	_, err = rd.Forward(ctx, input2d)
	if err == nil {
		t.Error("expected error for 2D input, got nil")
	}
}

func TestRegimeDetector_DefaultNumClasses(t *testing.T) {
	engine := testEngine()
	ops := numeric.Float32Ops{}
	cfg := RegimeConfig{InputDim: 5, HiddenDim: 8, NumLayers: 1, SeqLen: 10, NumClasses: 0}
	rd, err := BuildRegimeDetector[float32](cfg, engine, ops)
	if err != nil {
		t.Fatalf("BuildRegimeDetector: %v", err)
	}
	if rd.cfg.NumClasses != 4 {
		t.Errorf("NumClasses = %d, want 4 (default)", rd.cfg.NumClasses)
	}
}
