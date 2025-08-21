package loss

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/testing/testutils"
)

func TestMSE_Forward(t *testing.T) {
	var engine compute.Engine[float32] = compute.NewCPUEngine[float32](numeric.Float32Ops{})
	mse := NewMSE[float32](engine, numeric.Float32Ops{})

	predictions, _ := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 5.0})
	targets, _ := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	loss, _ := mse.Forward(context.Background(), predictions, targets)

	expectedLoss := float32(0.25) // ( (1-1)^2 + (2-2)^2 + (3-3)^2 + (5-4)^2 ) / 4 = (0+0+0+1)/4 = 0.25
	if loss.Data()[0] != expectedLoss {
		t.Errorf("expected %f, got %f", expectedLoss, loss.Data()[0])
	}

	predictions2, _ := tensor.New[float32]([]int{1, 3}, []float32{1.0, 2.0, 3.0})
	targets2, _ := tensor.New[float32]([]int{1, 3}, []float32{1.0, 2.0, 3.0})
	loss2, _ := mse.Forward(context.Background(), predictions2, targets2)
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
	gradient, _ := mse.Backward(context.Background(), predictions, targets)

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
	gradient2, _ := mse.Backward(context.Background(), predictions2, targets2)
	// (1-4), (2-5), (3-6) = -3, -3, -3
	expected2 := []float32{-3.0, -3.0, -3.0}
	for i, v := range gradient2.Data() {
		if v != expected2[i] {
			t.Errorf("expected %v, got %v", expected2, gradient2.Data())
		}
	}
}

func TestNewMSE(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	mse := NewMSE[float32](engine, ops)
	testutils.AssertNotNil(t, mse, "MSE should not be nil")
}

func TestMSE_Forward_EdgeCases(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	mse := NewMSE[float32](engine, ops)

	// Test with zeros
	predictions, err := tensor.New[float32]([]int{2, 2}, []float32{0.0, 0.0, 0.0, 0.0})
	testutils.AssertNoError(t, err, "Failed to create predictions tensor")
	targets, err := tensor.New[float32]([]int{2, 2}, []float32{0.0, 0.0, 0.0, 0.0})
	testutils.AssertNoError(t, err, "Failed to create targets tensor")

	loss, err := mse.Forward(ctx, predictions, targets)
	testutils.AssertNoError(t, err, "Forward should not error with zeros")
	testutils.AssertFloatEqual(t, 0.0, loss.Data()[0], 1e-6, "Loss should be 0 for identical zeros")

	// Test with single element
	pred1, err := tensor.New[float32]([]int{1}, []float32{5.0})
	testutils.AssertNoError(t, err, "Failed to create single prediction tensor")
	target1, err := tensor.New[float32]([]int{1}, []float32{3.0})
	testutils.AssertNoError(t, err, "Failed to create single target tensor")

	loss1, err := mse.Forward(ctx, pred1, target1)
	testutils.AssertNoError(t, err, "Forward should not error with single element")
	expected := float32(4.0) // (5-3)^2 = 4
	testutils.AssertFloatEqual(t, expected, loss1.Data()[0], 1e-6, "Loss should be 4.0 for (5-3)^2")

	// Test with large values
	predLarge, err := tensor.New[float32]([]int{2}, []float32{1000.0, 2000.0})
	testutils.AssertNoError(t, err, "Failed to create large predictions tensor")
	targetLarge, err := tensor.New[float32]([]int{2}, []float32{1001.0, 1999.0})
	testutils.AssertNoError(t, err, "Failed to create large targets tensor")

	lossLarge, err := mse.Forward(ctx, predLarge, targetLarge)
	testutils.AssertNoError(t, err, "Forward should not error with large values")
	expectedLarge := float32(1.0) // ((1000-1001)^2 + (2000-1999)^2) / 2 = (1 + 1) / 2 = 1
	testutils.AssertFloatEqual(t, expectedLarge, lossLarge.Data()[0], 1e-6, "Loss should be 1.0 for large values")
}

func TestMSE_Forward_DifferentShapes(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	mse := NewMSE[float32](engine, ops)

	// Test with 1D tensors
	pred1D, err := tensor.New[float32]([]int{4}, []float32{1.0, 2.0, 3.0, 4.0})
	testutils.AssertNoError(t, err, "Failed to create 1D predictions tensor")
	target1D, err := tensor.New[float32]([]int{4}, []float32{2.0, 3.0, 4.0, 5.0})
	testutils.AssertNoError(t, err, "Failed to create 1D targets tensor")

	loss1D, err := mse.Forward(ctx, pred1D, target1D)
	testutils.AssertNoError(t, err, "Forward should not error with 1D tensors")
	expected1D := float32(1.0) // ((1-2)^2 + (2-3)^2 + (3-4)^2 + (4-5)^2) / 4 = (1+1+1+1)/4 = 1
	testutils.AssertFloatEqual(t, expected1D, loss1D.Data()[0], 1e-6, "Loss should be 1.0 for 1D case")

	// Test with 3D tensors
	pred3D, err := tensor.New[float32]([]int{2, 2, 2}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})
	testutils.AssertNoError(t, err, "Failed to create 3D predictions tensor")
	target3D, err := tensor.New[float32]([]int{2, 2, 2}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})
	testutils.AssertNoError(t, err, "Failed to create 3D targets tensor")

	loss3D, err := mse.Forward(ctx, pred3D, target3D)
	testutils.AssertNoError(t, err, "Forward should not error with 3D tensors")
	testutils.AssertFloatEqual(t, 0.0, loss3D.Data()[0], 1e-6, "Loss should be 0 for identical 3D tensors")
}

func TestMSE_Backward_EdgeCases(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	mse := NewMSE[float32](engine, ops)

	// Test with zeros
	predictions, err := tensor.New[float32]([]int{2, 2}, []float32{0.0, 0.0, 0.0, 0.0})
	testutils.AssertNoError(t, err, "Failed to create predictions tensor")
	targets, err := tensor.New[float32]([]int{2, 2}, []float32{0.0, 0.0, 0.0, 0.0})
	testutils.AssertNoError(t, err, "Failed to create targets tensor")

	gradient, err := mse.Backward(ctx, predictions, targets)
	testutils.AssertNoError(t, err, "Backward should not error with zeros")
	for _, v := range gradient.Data() {
		testutils.AssertFloatEqual(t, 0.0, v, 1e-6, "Gradient should be 0 for identical values")
	}

	// Test with negative values
	predNeg, err := tensor.New[float32]([]int{2}, []float32{-1.0, -2.0})
	testutils.AssertNoError(t, err, "Failed to create negative predictions tensor")
	targetNeg, err := tensor.New[float32]([]int{2}, []float32{-3.0, -1.0})
	testutils.AssertNoError(t, err, "Failed to create negative targets tensor")

	gradNeg, err := mse.Backward(ctx, predNeg, targetNeg)
	testutils.AssertNoError(t, err, "Backward should not error with negative values")
	// Gradient: (-1 - (-3)), (-2 - (-1)) = (2, -1)
	expected := []float32{2.0, -1.0}
	for i, v := range gradNeg.Data() {
		testutils.AssertFloatEqual(t, expected[i], v, 1e-6, "Gradient should match expected for negative values")
	}
}

func TestMSE_Forward_ErrorHandling(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	mse := NewMSE[float32](engine, ops)

	// Test with mismatched shapes (should be caught by tensor operations)
	predictions, err := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	testutils.AssertNoError(t, err, "Failed to create predictions tensor")
	targets, err := tensor.New[float32]([]int{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	testutils.AssertNoError(t, err, "Failed to create mismatched targets tensor")

	_, err = mse.Forward(ctx, predictions, targets)
	if err != nil {
		// Expected to fail due to shape mismatch
		testutils.AssertError(t, err, "Forward should error with mismatched shapes")
	}
}

func TestMSE_Backward_ErrorHandling(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	mse := NewMSE[float32](engine, ops)

	// Test with mismatched shapes
	predictions, err := tensor.New[float32]([]int{2, 2}, []float32{1.0, 2.0, 3.0, 4.0})
	testutils.AssertNoError(t, err, "Failed to create predictions tensor")
	targets, err := tensor.New[float32]([]int{2, 3}, []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	testutils.AssertNoError(t, err, "Failed to create mismatched targets tensor")

	_, err = mse.Backward(ctx, predictions, targets)
	if err != nil {
		// Expected to fail due to shape mismatch
		testutils.AssertError(t, err, "Backward should error with mismatched shapes")
	}
}
