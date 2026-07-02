package fp8

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func TestFP8Linear_New(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name    string
		lname   string
		in, out int
		data    []float32
		wantErr bool
	}{
		{name: "valid", lname: "fc1", in: 4, out: 3, data: make([]float32, 12)},
		{name: "nil_data_zero_init", lname: "fc2", in: 2, out: 2, data: nil},
		{name: "empty_name", lname: "", in: 4, out: 3, data: make([]float32, 12), wantErr: true},
		{name: "zero_in", lname: "fc3", in: 0, out: 3, data: nil, wantErr: true},
		{name: "wrong_data_len", lname: "fc4", in: 4, out: 3, data: make([]float32, 5), wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layer, err := NewFP8Linear[float32](tt.lname, engine, tt.in, tt.out, tt.data)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if layer.OpType() != "FP8Linear" {
				t.Errorf("OpType = %q, want FP8Linear", layer.OpType())
			}
			if layer.Name() != tt.lname {
				t.Errorf("Name = %q, want %q", layer.Name(), tt.lname)
			}
			params := layer.Parameters()
			if len(params) != 1 {
				t.Fatalf("Parameters() returned %d, want 1", len(params))
			}
			outShape := layer.OutputShape()
			if outShape[1] != tt.out {
				t.Errorf("OutputShape[1] = %d, want %d", outShape[1], tt.out)
			}
		})
	}
}

func TestFP8Linear_Forward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	// Weight [2, 3]: identity-like mapping for easy verification.
	wData := []float32{
		1, 0, 0,
		0, 1, 0,
	}
	layer, err := NewFP8Linear[float32]("test", engine, 3, 2, wData)
	if err != nil {
		t.Fatalf("NewFP8Linear: %v", err)
	}

	// Input [2, 3]
	x, err := tensor.New[float32]([]int{2, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
	})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	out, err := layer.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	shape := out.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 2 {
		t.Fatalf("output shape = %v, want [2, 2]", shape)
	}

	// Expected: x @ W^T = [[1,2],[4,5]]
	// FP8 quantization introduces small error, so we allow tolerance.
	data := out.Data()
	expected := []float32{1, 2, 4, 5}
	for i, want := range expected {
		if math.Abs(float64(data[i]-want)) > 0.5 {
			t.Errorf("out[%d] = %f, want ~%f (tol=0.5)", i, data[i], want)
		}
	}
}

func TestFP8Linear_Backward(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	inF, outF := 3, 2
	wData := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	layer, err := NewFP8Linear[float32]("test", engine, inF, outF, wData)
	if err != nil {
		t.Fatalf("NewFP8Linear: %v", err)
	}

	x, err := tensor.New[float32]([]int{1, inF}, []float32{1, 1, 1})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	// Forward to populate cache.
	_, err = layer.Forward(ctx, x)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Backward with unit gradient.
	grad, err := tensor.New[float32]([]int{1, outF}, []float32{1, 1})
	if err != nil {
		t.Fatalf("create grad: %v", err)
	}

	dxSlice, err := layer.Backward(ctx, types.FullBackprop, grad, x)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if len(dxSlice) != 1 {
		t.Fatalf("Backward returned %d tensors, want 1", len(dxSlice))
	}

	// dx = grad @ W = [1,1] @ [[1,2,3],[4,5,6]] = [5, 7, 9]
	dxData := dxSlice[0].Data()
	expectedDx := []float32{5, 7, 9}
	for i, want := range expectedDx {
		if math.Abs(float64(dxData[i]-want)) > 0.01 {
			t.Errorf("dx[%d] = %f, want %f", i, dxData[i], want)
		}
	}

	// Check weight gradient was accumulated.
	wGrad := layer.Parameters()[0].Gradient
	if wGrad == nil {
		t.Fatal("weight gradient is nil after backward")
	}
	wGradShape := wGrad.Shape()
	if wGradShape[0] != outF || wGradShape[1] != inF {
		t.Errorf("weight gradient shape = %v, want [%d, %d]", wGradShape, outF, inF)
	}
}

func TestFP8Linear_SyncFP8Weights(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	origW := []float32{0.5, -0.3, 0.8, -0.1}
	wData := make([]float32, len(origW))
	copy(wData, origW)

	layer, err := NewFP8Linear[float32]("test", engine, 2, 2, wData)
	if err != nil {
		t.Fatalf("NewFP8Linear: %v", err)
	}

	// Modify master weight data directly (simulating optimizer step).
	masterData := layer.masterWeight.Value.Data()
	for i := range masterData {
		masterData[i] *= 2
	}

	if err := layer.SyncFP8Weights(); err != nil {
		t.Fatalf("SyncFP8Weights: %v", err)
	}

	// After sync, FP8 weight should reflect the doubled master.
	// FP8 E4M3 has limited precision so we allow some tolerance.
	for i, v := range layer.fp8WeightData {
		want := origW[i] * 2
		if math.Abs(float64(v-want)) > 0.2 {
			t.Errorf("fp8Weight[%d] = %f, want ~%f after sync", i, v, want)
		}
	}
}

// TestFP8Linear_TrainingConvergence verifies that FP8 linear training on a
// synthetic regression task achieves loss within 2% of a FP32 baseline.
func TestFP8Linear_TrainingConvergence(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	const (
		inF      = 4
		outF     = 2
		nSamples = 64
		epochs   = 500
		lr       = float32(0.005)
	)

	// Generate deterministic synthetic data: y = x @ trueW^T
	rng := rand.New(rand.NewPCG(42, 0))
	trueW := make([]float32, outF*inF)
	for i := range trueW {
		trueW[i] = float32(rng.NormFloat64()) * 0.5
	}

	xData := make([]float32, nSamples*inF)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}
	x, err := tensor.New[float32]([]int{nSamples, inF}, xData)
	if err != nil {
		t.Fatalf("create X: %v", err)
	}

	// Compute targets: Y = X @ trueW^T
	trueWT, err := tensor.New[float32]([]int{outF, inF}, trueW)
	if err != nil {
		t.Fatalf("create trueW: %v", err)
	}
	trueWTTransposed, err := engine.Transpose(ctx, trueWT, []int{1, 0})
	if err != nil {
		t.Fatalf("transpose trueW: %v", err)
	}
	yTarget, err := engine.MatMul(ctx, x, trueWTTransposed)
	if err != nil {
		t.Fatalf("compute targets: %v", err)
	}

	// Train with FP8Linear.
	fp8FinalLoss := trainLinear(t, engine, ctx, x, yTarget, inF, outF, epochs, lr, true, rng)

	// Train with standard FP32 linear (baseline).
	rng2 := rand.New(rand.NewPCG(42, 0))
	f32FinalLoss := trainLinear(t, engine, ctx, x, yTarget, inF, outF, epochs, lr, false, rng2)

	t.Logf("FP8 final loss: %f, FP32 final loss: %f", fp8FinalLoss, f32FinalLoss)

	// Both losses should converge to near zero. FP8 quantization noise in the
	// forward pass creates a slightly higher loss floor, so we check:
	// 1. Both losses are small (< 0.05 for this synthetic task).
	// 2. The absolute difference between FP8 and FP32 loss is within 2% of
	//    the initial loss scale (which is on the order of 1.0).
	if fp8FinalLoss > 0.05 {
		t.Errorf("FP8 final loss (%f) too high, expected < 0.05", fp8FinalLoss)
	}
	if f32FinalLoss > 0.05 {
		t.Errorf("FP32 final loss (%f) too high, expected < 0.05", f32FinalLoss)
	}
	absDiff := math.Abs(float64(fp8FinalLoss - f32FinalLoss))
	if absDiff > 0.02 {
		t.Errorf("FP8-FP32 loss difference (%f) exceeds 2%% threshold", absDiff)
	}
}

// trainLinear runs a simple SGD training loop with either FP8Linear or a
// standard FP32 linear layer and returns the final MSE loss.
func trainLinear(
	t *testing.T,
	engine compute.Engine[float32],
	ctx context.Context,
	x, yTarget *tensor.TensorNumeric[float32],
	inF, outF, epochs int,
	lr float32,
	useFP8 bool,
	rng *rand.Rand,
) float32 {
	t.Helper()

	n := outF * inF
	initW := make([]float32, n)
	for i := range initW {
		initW[i] = float32(rng.NormFloat64()) * 0.1
	}

	var param *graph.Parameter[float32]
	var forwardFn func() (*tensor.TensorNumeric[float32], error)
	var backwardFn func(*tensor.TensorNumeric[float32]) error
	var syncFn func() error

	if useFP8 {
		layer, err := NewFP8Linear[float32]("fp8_train", engine, inF, outF, initW)
		if err != nil {
			t.Fatalf("NewFP8Linear: %v", err)
		}
		param = layer.Parameters()[0]

		forwardFn = func() (*tensor.TensorNumeric[float32], error) {
			return layer.Forward(ctx, x)
		}
		backwardFn = func(gradOut *tensor.TensorNumeric[float32]) error {
			_, err := layer.Backward(ctx, types.FullBackprop, gradOut, x)
			return err
		}
		syncFn = func() error {
			return layer.SyncFP8Weights()
		}
	} else {
		wTensor, err := tensor.New[float32]([]int{outF, inF}, initW)
		if err != nil {
			t.Fatalf("create weight: %v", err)
		}
		param, err = graph.NewParameter[float32]("f32_weight", wTensor, tensor.New[float32])
		if err != nil {
			t.Fatalf("create param: %v", err)
		}

		forwardFn = func() (*tensor.TensorNumeric[float32], error) {
			wT, err := engine.Transpose(ctx, param.Value, []int{1, 0})
			if err != nil {
				return nil, err
			}
			return engine.MatMul(ctx, x, wT)
		}
		backwardFn = func(gradOut *tensor.TensorNumeric[float32]) error {
			gT, err := engine.Transpose(ctx, gradOut, []int{1, 0})
			if err != nil {
				return err
			}
			dW, err := engine.MatMul(ctx, gT, x)
			if err != nil {
				return err
			}
			if param.Gradient != nil {
				param.Gradient, err = engine.Add(ctx, param.Gradient, dW)
			} else {
				param.Gradient = dW
			}
			return err
		}
		syncFn = func() error { return nil }
	}

	nSamples := x.Shape()[0]
	var finalLoss float32

	for epoch := 0; epoch < epochs; epoch++ {
		// Zero gradient.
		zg, err := tensor.New[float32]([]int{outF, inF}, make([]float32, n))
		if err != nil {
			t.Fatalf("zero grad: %v", err)
		}
		param.Gradient = zg

		// Forward.
		yPred, err := forwardFn()
		if err != nil {
			t.Fatalf("forward (epoch %d): %v", epoch, err)
		}

		// MSE loss = mean((yPred - yTarget)^2)
		diff, err := engine.Sub(ctx, yPred, yTarget)
		if err != nil {
			t.Fatalf("sub (epoch %d): %v", epoch, err)
		}
		sq, err := engine.Mul(ctx, diff, diff)
		if err != nil {
			t.Fatalf("mul (epoch %d): %v", epoch, err)
		}
		sqData := sq.Data()
		var lossVal float32
		for _, v := range sqData {
			lossVal += v
		}
		lossVal /= float32(len(sqData))
		finalLoss = lossVal

		// Backward: dL/dy = 2*(yPred - yTarget) / N
		scale := 2.0 / float32(nSamples*outF)
		gradOut, err := engine.UnaryOp(ctx, diff, func(v float32) float32 { return v * scale })
		if err != nil {
			t.Fatalf("grad scale (epoch %d): %v", epoch, err)
		}

		if err := backwardFn(gradOut); err != nil {
			t.Fatalf("backward (epoch %d): %v", epoch, err)
		}

		// SGD step: W -= lr * grad
		scaledGrad, err := engine.UnaryOp(ctx, param.Gradient, func(v float32) float32 { return v * lr })
		if err != nil {
			t.Fatalf("scale grad (epoch %d): %v", epoch, err)
		}
		param.Value, err = engine.Sub(ctx, param.Value, scaledGrad)
		if err != nil {
			t.Fatalf("sgd step (epoch %d): %v", epoch, err)
		}

		if err := syncFn(); err != nil {
			t.Fatalf("sync (epoch %d): %v", epoch, err)
		}
	}

	return finalLoss
}

func TestFP8Linear_QuantizeRoundtrip(t *testing.T) {
	data := []float32{1.0, -2.0, 3.5, -0.5, 100.0, -100.0}
	deq := quantizeDequantizeFP8(data)
	if len(deq) != len(data) {
		t.Fatalf("dequantized length = %d, want %d", len(deq), len(data))
	}

	mae := fp8RoundtripError(data, deq)
	// FP8 E4M3 has limited precision; MAE should be reasonable for this range.
	if mae > 5.0 {
		t.Errorf("roundtrip MAE = %f, too high", mae)
	}
	t.Logf("FP8 roundtrip MAE: %f", mae)
}
