package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
	"github.com/zerfoo/ztensor/types"
)

func newSeqClsEngine() (compute.Engine[float32], numeric.Arithmetic[float32]) {
	ops := numeric.Float32Ops{}
	return compute.NewCPUEngine[float32](ops), ops
}

func makeInput3D(t *testing.T, batch, seqLen, hiddenDim int) *tensor.TensorNumeric[float32] {
	t.Helper()
	data := make([]float32, batch*seqLen*hiddenDim)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	out, err := tensor.New[float32]([]int{batch, seqLen, hiddenDim}, data)
	testutils.AssertNoError(t, err, "failed to create 3D input tensor: %v")
	return out
}

func TestSeqClassification_CLS_Forward(t *testing.T) {
	engine, ops := newSeqClsEngine()
	batch, seqLen, hiddenDim, numClasses := 2, 8, 16, 3

	layer, err := NewSeqClassification[float32](engine, ops, PoolCLS, hiddenDim, numClasses)
	testutils.AssertNoError(t, err, "failed to create SeqClassification: %v")

	input := makeInput3D(t, batch, seqLen, hiddenDim)
	output, err := layer.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward failed: %v")
	testutils.AssertNotNil(t, output, "expected output to not be nil")
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{batch, numClasses}, output.Shape()),
		"expected output shape [2, 3]")
}

func TestSeqClassification_Mean_Forward(t *testing.T) {
	engine, ops := newSeqClsEngine()
	batch, seqLen, hiddenDim, numClasses := 2, 8, 16, 3

	layer, err := NewSeqClassification[float32](engine, ops, PoolMean, hiddenDim, numClasses)
	testutils.AssertNoError(t, err, "failed to create SeqClassification: %v")

	input := makeInput3D(t, batch, seqLen, hiddenDim)
	output, err := layer.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward failed: %v")
	testutils.AssertNotNil(t, output, "expected output to not be nil")
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{batch, numClasses}, output.Shape()),
		"expected output shape [2, 3]")
}

func TestSeqClassification_CLS_Backward(t *testing.T) {
	engine, ops := newSeqClsEngine()
	batch, seqLen, hiddenDim, numClasses := 2, 4, 8, 3

	layer, err := NewSeqClassification[float32](engine, ops, PoolCLS, hiddenDim, numClasses)
	testutils.AssertNoError(t, err, "failed to create SeqClassification: %v")

	input := makeInput3D(t, batch, seqLen, hiddenDim)

	// Run forward first to populate internal state.
	output, err := layer.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward failed: %v")

	// Create gradient matching output shape.
	gradData := make([]float32, batch*numClasses)
	for i := range gradData {
		gradData[i] = 1.0
	}
	gradOutput, err := tensor.New[float32](output.Shape(), gradData)
	testutils.AssertNoError(t, err, "failed to create gradient tensor: %v")

	grads, err := layer.Backward(context.Background(), types.FullBackprop, gradOutput, input)
	testutils.AssertNoError(t, err, "backward failed: %v")
	testutils.AssertNotNil(t, grads, "expected gradients to not be nil")
	testutils.AssertEqual(t, 1, len(grads), "expected 1 gradient tensor")
	testutils.AssertTrue(t, testutils.IntSliceEqual(input.Shape(), grads[0].Shape()),
		"expected gradient shape to match input shape")

	// For CLS pooling, gradient should be zero at positions > 0.
	gData := grads[0].Data()
	for b := range batch {
		for s := 1; s < seqLen; s++ {
			for d := range hiddenDim {
				idx := b*seqLen*hiddenDim + s*hiddenDim + d
				testutils.AssertTrue(t, gData[idx] == 0,
					"expected zero gradient at non-CLS position")
			}
		}
	}
}

func TestSeqClassification_Mean_Backward(t *testing.T) {
	engine, ops := newSeqClsEngine()
	batch, seqLen, hiddenDim, numClasses := 2, 4, 8, 3

	layer, err := NewSeqClassification[float32](engine, ops, PoolMean, hiddenDim, numClasses)
	testutils.AssertNoError(t, err, "failed to create SeqClassification: %v")

	input := makeInput3D(t, batch, seqLen, hiddenDim)

	output, err := layer.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward failed: %v")

	gradData := make([]float32, batch*numClasses)
	for i := range gradData {
		gradData[i] = 1.0
	}
	gradOutput, err := tensor.New[float32](output.Shape(), gradData)
	testutils.AssertNoError(t, err, "failed to create gradient tensor: %v")

	grads, err := layer.Backward(context.Background(), types.FullBackprop, gradOutput, input)
	testutils.AssertNoError(t, err, "backward failed: %v")
	testutils.AssertNotNil(t, grads, "expected gradients to not be nil")
	testutils.AssertTrue(t, testutils.IntSliceEqual(input.Shape(), grads[0].Shape()),
		"expected gradient shape to match input shape")
}

func TestSeqClassification_WithDropout(t *testing.T) {
	engine, ops := newSeqClsEngine()
	batch, seqLen, hiddenDim, numClasses := 2, 4, 8, 3

	layer, err := NewSeqClassification[float32](engine, ops, PoolCLS, hiddenDim, numClasses,
		WithDropout[float32](engine, ops, 0.1),
	)
	testutils.AssertNoError(t, err, "failed to create SeqClassification with dropout: %v")
	testutils.AssertNotNil(t, layer.dropout, "expected dropout to be set")

	input := makeInput3D(t, batch, seqLen, hiddenDim)

	// Eval mode (default): dropout is a no-op.
	output, err := layer.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward (eval) failed: %v")
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{batch, numClasses}, output.Shape()),
		"expected output shape [2, 3] in eval mode")

	// Training mode: dropout is active.
	layer.SetTraining(true)
	outputTrain, err := layer.Forward(context.Background(), input)
	testutils.AssertNoError(t, err, "forward (train) failed: %v")
	testutils.AssertTrue(t, testutils.IntSliceEqual([]int{batch, numClasses}, outputTrain.Shape()),
		"expected output shape [2, 3] in training mode")
}

func TestSeqClassification_WithoutDropout(t *testing.T) {
	engine, ops := newSeqClsEngine()
	hiddenDim, numClasses := 8, 3

	layer, err := NewSeqClassification[float32](engine, ops, PoolMean, hiddenDim, numClasses)
	testutils.AssertNoError(t, err, "failed to create SeqClassification: %v")
	testutils.AssertTrue(t, layer.dropout == nil, "expected dropout to be nil")
}

func TestSeqClassification_Parameters(t *testing.T) {
	engine, ops := newSeqClsEngine()
	hiddenDim, numClasses := 8, 3

	layer, err := NewSeqClassification[float32](engine, ops, PoolCLS, hiddenDim, numClasses)
	testutils.AssertNoError(t, err, "failed to create SeqClassification: %v")

	params := layer.Parameters()
	// Dense layer has weight + bias = 2 parameters.
	testutils.AssertEqual(t, 2, len(params), "expected 2 parameters (weight + bias)")
}

func TestSeqClassification_OpTypeAndAttributes(t *testing.T) {
	engine, ops := newSeqClsEngine()

	layer, err := NewSeqClassification[float32](engine, ops, PoolMean, 8, 3)
	testutils.AssertNoError(t, err, "failed to create SeqClassification: %v")

	testutils.AssertEqual(t, "SeqClassification", layer.OpType(), "unexpected OpType")

	attrs := layer.Attributes()
	testutils.AssertEqual(t, "mean", attrs["pool_mode"], "expected pool_mode=mean")
	testutils.AssertEqual(t, 3, attrs["num_classes"], "expected num_classes=3")
}

func TestSeqClassification_InvalidInputs(t *testing.T) {
	engine, ops := newSeqClsEngine()

	// Invalid dimensions.
	_, err := NewSeqClassification[float32](engine, ops, PoolCLS, 0, 3)
	testutils.AssertTrue(t, err != nil, "expected error for zero hiddenDim")

	_, err = NewSeqClassification[float32](engine, ops, PoolCLS, 8, 0)
	testutils.AssertTrue(t, err != nil, "expected error for zero numClasses")

	// Wrong number of inputs.
	layer, err := NewSeqClassification[float32](engine, ops, PoolCLS, 8, 3)
	testutils.AssertNoError(t, err, "failed to create SeqClassification: %v")

	_, err = layer.Forward(context.Background())
	testutils.AssertTrue(t, err != nil, "expected error for no inputs")

	// Wrong input rank.
	bad2D, err := tensor.New[float32]([]int{2, 8}, make([]float32, 16))
	testutils.AssertNoError(t, err, "failed to create 2D tensor: %v")
	_, err = layer.Forward(context.Background(), bad2D)
	testutils.AssertTrue(t, err != nil, "expected error for 2D input")
}
