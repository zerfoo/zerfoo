//go:build cuda

package tabular

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestTrain_GPU(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine, err := compute.NewGPUEngine[float32](ops)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}

	// XOR problem on GPU.
	data := [][]float64{
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
		{0, 0}, {0, 1}, {1, 0}, {1, 1},
	}
	labels := []int{
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
	}

	mc := ModelConfig{
		HiddenDims:  []int{32, 16},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}

	tc := TrainConfig{
		Epochs:       1000,
		BatchSize:    16,
		LearningRate: 0.05,
	}

	model, err := Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		t.Fatalf("Train on GPU: %v", err)
	}

	// Verify convergence.
	tests := []struct {
		features []float64
		want     Direction
	}{
		{[]float64{0, 0}, Long},
		{[]float64{0, 1}, Short},
		{[]float64{1, 0}, Short},
		{[]float64{1, 1}, Long},
	}

	correct := 0
	for _, tt := range tests {
		dir, _, err := model.Predict(tt.features)
		if err != nil {
			t.Fatalf("Predict(%v): %v", tt.features, err)
		}
		if dir == tt.want {
			correct++
		}
	}

	if correct < 3 {
		t.Errorf("GPU XOR convergence: got %d/4 correct, want >= 3", correct)
	}
}
