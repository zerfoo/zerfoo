package residual

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func newTestAttnRes(t *testing.T, modelDim int) *AttnRes[float32] {
	t.Helper()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ar, err := NewAttnRes[float32]("test", engine, ops, modelDim)
	if err != nil {
		t.Fatalf("NewAttnRes failed: %v", err)
	}
	return ar
}

func TestAttnResForward(t *testing.T) {
	ctx := context.Background()
	ar := newTestAttnRes(t, 4)

	// Three layer outputs, each [1, 4].
	layers := make([]*tensor.TensorNumeric[float32], 3)
	data := [][]float32{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
	}
	for i, d := range data {
		var err error
		layers[i], err = tensor.New[float32]([]int{1, 4}, d)
		if err != nil {
			t.Fatalf("failed to create layer tensor %d: %v", i, err)
		}
	}

	out, err := ar.Forward(ctx, layers...)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Verify output shape matches input shape.
	if len(out.Shape()) != 2 || out.Shape()[0] != 1 || out.Shape()[1] != 4 {
		t.Errorf("unexpected output shape: %v, want [1, 4]", out.Shape())
	}

	// Verify no NaN values.
	for i, v := range out.Data() {
		if math.IsNaN(float64(v)) {
			t.Errorf("NaN at index %d", i)
		}
	}
}

func TestAttnResWeightsSum(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ar, err := NewAttnRes[float32]("test", engine, ops, 4)
	if err != nil {
		t.Fatalf("NewAttnRes failed: %v", err)
	}

	// Create 5 layer outputs.
	layers := make([]*tensor.TensorNumeric[float32], 5)
	for i := range layers {
		d := make([]float32, 4)
		for j := range d {
			d[j] = float32(i*4 + j + 1)
		}
		layers[i], err = tensor.New[float32]([]int{1, 4}, d)
		if err != nil {
			t.Fatalf("failed to create layer tensor %d: %v", i, err)
		}
	}

	// We can verify the softmax weights sum to 1 by checking that the output
	// is a convex combination: each element should be between the min and max
	// of the corresponding elements across layers.
	out, err := ar.Forward(ctx, layers...)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	outData := out.Data()
	for j := 0; j < 4; j++ {
		minVal := float32(math.MaxFloat32)
		maxVal := float32(-math.MaxFloat32)
		for i := range layers {
			v := layers[i].Data()[j]
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
		if outData[j] < minVal-1e-5 || outData[j] > maxVal+1e-5 {
			t.Errorf("output[%d]=%f not in convex hull [%f, %f]", j, outData[j], minVal, maxVal)
		}
	}
}

func TestAttnResSelectiveAttention(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ar, err := NewAttnRes[float32]("test", engine, ops, 4)
	if err != nil {
		t.Fatalf("NewAttnRes failed: %v", err)
	}

	// Set the query to large positive values so that the layer with the largest
	// RMSNorm-aligned values gets the highest attention weight.
	queryData := []float32{100, 100, 100, 100}
	queryTensor, err := tensor.New[float32]([]int{1, 4}, queryData)
	if err != nil {
		t.Fatalf("failed to create query tensor: %v", err)
	}
	ar.query.Value = queryTensor

	// Layer 0: small values, layer 1: large aligned values, layer 2: negative.
	layer0, _ := tensor.New[float32]([]int{1, 4}, []float32{0.01, 0.01, 0.01, 0.01})
	layer1, _ := tensor.New[float32]([]int{1, 4}, []float32{10, 10, 10, 10})
	layer2, _ := tensor.New[float32]([]int{1, 4}, []float32{-10, -10, -10, -10})

	out, err := ar.Forward(ctx, layer0, layer1, layer2)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// With large positive query and RMSNorm, layer1 (all positive, large magnitude)
	// should dominate. Output should be much closer to layer1 than to layer0 or layer2.
	outData := out.Data()
	for j := 0; j < 4; j++ {
		distToLayer1 := math.Abs(float64(outData[j] - 10.0))
		distToLayer0 := math.Abs(float64(outData[j] - 0.01))
		if distToLayer1 >= distToLayer0 {
			t.Errorf("output[%d]=%f is closer to layer0 than layer1; expected layer1 to dominate", j, outData[j])
		}
	}
}

func TestAttnResSingleLayer(t *testing.T) {
	ctx := context.Background()
	ar := newTestAttnRes(t, 4)

	// With a single layer, softmax over 1 element gives weight 1.0,
	// so output should equal the input.
	input, err := tensor.New[float32]([]int{1, 4}, []float32{3, 5, 7, 9})
	if err != nil {
		t.Fatalf("failed to create input tensor: %v", err)
	}

	out, err := ar.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	outData := out.Data()
	inData := input.Data()
	for j := range inData {
		if math.Abs(float64(outData[j]-inData[j])) > 1e-5 {
			t.Errorf("single layer output[%d]=%f != input[%d]=%f", j, outData[j], j, inData[j])
		}
	}
}

func TestNewAttnResInvalidDim(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name     string
		modelDim int
	}{
		{"zero", 0},
		{"negative", -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewAttnRes[float32]("test", engine, ops, tt.modelDim)
			if err == nil {
				t.Error("expected error for invalid modelDim, got nil")
			}
		})
	}
}
