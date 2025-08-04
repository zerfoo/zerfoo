package loss

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestMSE_Forward(t *testing.T) {
	var engine compute.Engine[float32] = compute.NewCPUEngine[float32](numeric.Float32Ops{})
	mse := NewMSE[float32](engine, numeric.Float32Ops{})

	predictions, _ := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 5.0})
	targets, _ := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	loss := mse.Forward(predictions, targets)

	expectedLoss := float32(0.25) // ( (1-1)^2 + (2-2)^2 + (3-3)^2 + (5-4)^2 ) / 4 = (0+0+0+1)/4 = 0.25
	if loss.Data()[0] != expectedLoss {
		t.Errorf("expected %f, got %f", expectedLoss, loss.Data()[0])
	}

	predictions2, _ := tensor.New[float32]([]int{1, 3}, []float32{1.0, 2.0, 3.0})
	targets2, _ := tensor.New[float32]([]int{1, 3}, []float32{1.0, 2.0, 3.0})
	loss2 := mse.Forward(predictions2, targets2)
	expectedLoss2 := float32(0.0)
	if loss2.Data()[0] != expectedLoss2 {
		t.Errorf("expected %f, got %f", expectedLoss2, loss2.Data()[0])
	}
}

func TestMSE_Backward(t *testing.T) {
	var engine compute.Engine[float32] = compute.NewCPUEngine[float32](numeric.Float32Ops{})
	mse := NewMSE[float32](engine, numeric.Float32Ops{})

	predictions, _ := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 5.0})
	targets, _ := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	gradient := mse.Backward(predictions, targets)

	// Gradient is (predictions - targets)
	// (1-1), (2-2), (3-3), (5-4) = 0, 0, 0, 1
	expected := []float32{0.0, 0.0, 0.0, 1.0}
	for i, v := range gradient.Data() {
		if v != expected[i] {
			t.Errorf("expected %v, got %v", expected, gradient.Data())
		}
	}

	predictions2, _ := tensor.New[float32]([]int{1, 3}, []float32{1.0, 2.0, 3.0})
	targets2, _ := tensor.New[float32]([]int{1, 3}, []float32{4.0, 5.0, 6.0})
	gradient2 := mse.Backward(predictions2, targets2)
	// (1-4), (2-5), (3-6) = -3, -3, -3
	expected2 := []float32{-3.0, -3.0, -3.0}
	for i, v := range gradient2.Data() {
		if v != expected2[i] {
			t.Errorf("expected %v, got %v", expected2, gradient2.Data())
		}
	}
}
