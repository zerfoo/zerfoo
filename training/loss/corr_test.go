package loss

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
)

// Statically assert CorrLoss implements graph.Node.
var _ graph.Node[float32] = (*CorrLoss[float32])(nil)

func TestNewCorrLoss(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)
	testutils.AssertNotNil(t, cl, "CorrLoss should not be nil")
}

func TestCorrLoss_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)
	if cl.OpType() != "CorrLoss" {
		t.Errorf("expected OpType CorrLoss, got %s", cl.OpType())
	}
}

func TestCorrLoss_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)
	if cl.Attributes() != nil {
		t.Error("expected nil Attributes")
	}
}

func TestCorrLoss_Parameters(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)
	if cl.Parameters() != nil {
		t.Error("expected nil Parameters")
	}
}

func TestCorrLoss_OutputShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)
	shape := cl.OutputShape()
	if len(shape) != 1 || shape[0] != 1 {
		t.Errorf("expected [1], got %v", shape)
	}
}

func TestCorrLoss_Forward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)

	tests := []struct {
		name     string
		preds    []float32
		targets  []float32
		shape    []int
		wantLoss float32
		tol      float32
	}{
		{
			name:     "perfect positive correlation",
			preds:    []float32{1, 2, 3, 4, 5},
			targets:  []float32{2, 4, 6, 8, 10},
			shape:    []int{5},
			wantLoss: -1.0, // -corr = -1.0
			tol:      1e-6,
		},
		{
			name:     "perfect negative correlation",
			preds:    []float32{1, 2, 3, 4, 5},
			targets:  []float32{10, 8, 6, 4, 2},
			shape:    []int{5},
			wantLoss: 1.0, // -corr = -(-1.0) = 1.0
			tol:      1e-6,
		},
		{
			name:     "zero correlation",
			preds:    []float32{1, 2, 3, 4, 5},
			targets:  []float32{3, 3, 3, 3, 3},
			shape:    []int{5},
			wantLoss: 0.0, // targets have zero variance -> corr = 0 (eps protects)
			tol:      1e-4,
		},
		{
			name:     "known values",
			preds:    []float32{1, 2, 3},
			targets:  []float32{2, 4, 5},
			shape:    []int{3},
			wantLoss: -pearson([]float32{1, 2, 3}, []float32{2, 4, 5}),
			tol:      1e-5,
		},
		{
			name:     "2D shape",
			preds:    []float32{0.1, 0.5, 0.3, 0.9},
			targets:  []float32{0.0, 0.4, 0.2, 1.0},
			shape:    []int{4, 1},
			wantLoss: -pearson([]float32{0.1, 0.5, 0.3, 0.9}, []float32{0.0, 0.4, 0.2, 1.0}),
			tol:      1e-5,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			preds, err := tensor.New[float32](tc.shape, tc.preds)
			testutils.AssertNoError(t, err, "create preds")
			targets, err := tensor.New[float32](tc.shape, tc.targets)
			testutils.AssertNoError(t, err, "create targets")

			loss, err := cl.Forward(ctx, preds, targets)
			testutils.AssertNoError(t, err, "Forward")
			testutils.AssertFloatEqual(t, tc.wantLoss, loss.Data()[0], tc.tol, "loss value")
		})
	}
}

func TestCorrLoss_Forward_ErrorHandling(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)

	// Wrong number of inputs
	pred, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	_, err := cl.Forward(ctx, pred)
	testutils.AssertError(t, err, "Forward with 1 input should error")

	_, err = cl.Forward(ctx, pred, pred, pred)
	testutils.AssertError(t, err, "Forward with 3 inputs should error")
}

func TestCorrLoss_Backward(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)

	// Use finite differences to verify analytical gradients.
	preds := []float32{0.2, 0.5, 0.1, 0.8, 0.4}
	targets := []float32{0.0, 0.4, 0.2, 1.0, 0.6}
	shape := []int{5}

	predT, _ := tensor.New[float32](shape, preds)
	targT, _ := tensor.New[float32](shape, targets)
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	// Run forward to cache, then backward
	_, err := cl.Forward(ctx, predT, targT)
	testutils.AssertNoError(t, err, "Forward")

	grads, err := cl.Backward(ctx, types.FullBackprop, dOut, predT, targT)
	testutils.AssertNoError(t, err, "Backward")

	if len(grads) != 2 {
		t.Fatalf("expected 2 gradient tensors, got %d", len(grads))
	}

	// Verify prediction gradients via finite differences
	eps := float32(1e-4)
	analyticalGrads := grads[0].Data()
	for i := range preds {
		perturbedPlus := make([]float32, len(preds))
		perturbedMinus := make([]float32, len(preds))
		copy(perturbedPlus, preds)
		copy(perturbedMinus, preds)
		perturbedPlus[i] += eps
		perturbedMinus[i] -= eps

		lossPlus := negPearson(perturbedPlus, targets)
		lossMinus := negPearson(perturbedMinus, targets)
		numericalGrad := (lossPlus - lossMinus) / (2 * eps)

		testutils.AssertFloatEqual(t, numericalGrad, analyticalGrads[i], 1e-3,
			"gradient mismatch at index")
	}

	// Target gradients should be zero
	for _, v := range grads[1].Data() {
		testutils.AssertFloatEqual(t, float32(0), v, 1e-8, "target gradient should be zero")
	}
}

func TestCorrLoss_Backward_NilInputs(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)

	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	// No cached inputs, no provided inputs
	_, err := cl.Backward(ctx, types.FullBackprop, dOut)
	testutils.AssertError(t, err, "Backward with nil inputs should error")
}

func TestCorrLoss_Backward_UseCachedInputs(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)

	predT, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	targT, _ := tensor.New[float32]([]int{3}, []float32{2, 4, 6})
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	// Forward caches inputs
	_, err := cl.Forward(ctx, predT, targT)
	testutils.AssertNoError(t, err, "Forward")

	// Backward without providing inputs (uses cached)
	grads, err := cl.Backward(ctx, types.FullBackprop, dOut)
	testutils.AssertNoError(t, err, "Backward with cached inputs")

	if len(grads) != 2 {
		t.Fatalf("expected 2 gradient tensors, got %d", len(grads))
	}
}

func TestCorrLoss_Backward_ScaledDOut(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	cl := NewCorrLoss[float32](engine, ops)

	predT, _ := tensor.New[float32]([]int{4}, []float32{0.1, 0.3, 0.7, 0.9})
	targT, _ := tensor.New[float32]([]int{4}, []float32{0.0, 0.2, 0.8, 1.0})

	// Get gradients with dOut=1.0
	dOut1, _ := tensor.New[float32]([]int{1}, []float32{1.0})
	_, _ = cl.Forward(ctx, predT, targT)
	grads1, _ := cl.Backward(ctx, types.FullBackprop, dOut1, predT, targT)

	// Get gradients with dOut=2.0
	dOut2, _ := tensor.New[float32]([]int{1}, []float32{2.0})
	_, _ = cl.Forward(ctx, predT, targT)
	grads2, _ := cl.Backward(ctx, types.FullBackprop, dOut2, predT, targT)

	// grads2 should be 2x grads1
	for i := range grads1[0].Data() {
		testutils.AssertFloatEqual(t, grads1[0].Data()[i]*2, grads2[0].Data()[i], 1e-5,
			"gradient should scale with dOut")
	}
}

// pearson computes Pearson correlation coefficient.
func pearson(x, y []float32) float32 {
	n := float64(len(x))
	var sumX, sumY float64
	for i := range x {
		sumX += float64(x[i])
		sumY += float64(y[i])
	}
	meanX := sumX / n
	meanY := sumY / n

	var sumXY, sumXX, sumYY float64
	for i := range x {
		dx := float64(x[i]) - meanX
		dy := float64(y[i]) - meanY
		sumXY += dx * dy
		sumXX += dx * dx
		sumYY += dy * dy
	}
	denom := math.Sqrt(sumXX * sumYY)
	if denom < 1e-8 {
		return 0
	}
	return float32(sumXY / denom)
}

// negPearson returns -pearson for use in finite difference gradient checks.
func negPearson(x, y []float32) float32 {
	return -pearson(x, y)
}
