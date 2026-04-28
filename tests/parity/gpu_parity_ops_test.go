package parity_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"
	"github.com/zerfoo/ztensor/types"
)

// gpuOpsSetup creates CPU and GPU engines for parity tests.
// Skips the test if GPU is not available.
func gpuOpsSetup(t *testing.T) (cpu compute.Engine[float32], gpu compute.Engine[float32], gpuRaw *compute.GPUEngine[float32], ops *numeric.Float32Ops) {
	t.Helper()
	ops = &numeric.Float32Ops{}
	cpu = compute.NewCPUEngine[float32](ops)
	var err error
	gpuRaw, err = compute.NewGPUEngine[float32](*ops, 0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	os.Setenv("ZERFOO_DISABLE_CUDA_GRAPH", "1")
	gpu = compute.Engine[float32](gpuRaw)
	return
}

// assertGPUClose compares CPU and GPU float32 slices element-by-element.
func assertGPUClose(t *testing.T, label string, cpuData, gpuData []float32, tol float64) {
	t.Helper()
	if len(cpuData) != len(gpuData) {
		t.Fatalf("%s: length mismatch: cpu=%d gpu=%d", label, len(cpuData), len(gpuData))
	}
	maxDiff := 0.0
	maxIdx := 0
	mismatches := 0
	for i := range cpuData {
		d := math.Abs(float64(cpuData[i] - gpuData[i]))
		if d > tol {
			mismatches++
		}
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	t.Logf("%s: maxDiff=%.6e at idx=%d, mismatches=%d/%d", label, maxDiff, maxIdx, mismatches, len(cpuData))
	if maxDiff > tol {
		t.Errorf("%s: FAIL maxDiff=%.6e exceeds tolerance %.1e (at idx=%d: cpu=%.6f gpu=%.6f)",
			label, maxDiff, tol, maxIdx, cpuData[maxIdx], gpuData[maxIdx])
	}
}

// deterministicData generates a reproducible float32 slice.
func deterministicData(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i%97)*0.03 - 1.5
	}
	return data
}

// cloneF32 returns an independent copy of a float32 slice.
func cloneF32(src []float32) []float32 {
	return append([]float32(nil), src...)
}

// ---------------------------------------------------------------------------
// T86.5.4 — Core Ops GPU parity
// ---------------------------------------------------------------------------

func TestGPUParity_Linear(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	inFeatures, outFeatures := 8, 4
	inputData := deterministicData(2 * inFeatures) // batch=2
	weightData := deterministicData(inFeatures * outFeatures)

	// CPU
	cpuInput := testutil.MakeTensor(t, inputData, []int{2, inFeatures})
	cpuWeightParam := testutil.MakeParam(t, "weight", weightData, []int{inFeatures, outFeatures})
	cpuLinear := core.NewLinearFromParam(cpuEng, cpuWeightParam)
	cpuOut, err := cpuLinear.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU
	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), []int{2, inFeatures})
	gpuWeightParam := testutil.MakeParam(t, "weight", cloneF32(weightData), []int{inFeatures, outFeatures})
	gpuLinear := core.NewLinearFromParam(gpuEng, gpuWeightParam)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput, gpuWeightParam.Value}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuLinear.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUClose(t, "linear_forward", cpuOut.Data(), gpuOut.Data(), 1e-3)
}

func TestGPUParity_MatMul(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	aData := deterministicData(2 * 4)
	bData := deterministicData(4 * 3)

	// CPU
	cpuA := testutil.MakeTensor(t, aData, []int{2, 4})
	cpuB := testutil.MakeTensor(t, bData, []int{4, 3})
	cpuMM := core.NewMatMul[float32](cpuEng)
	cpuOut, err := cpuMM.Forward(ctx, cpuA, cpuB)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU
	gpuA := testutil.MakeTensor(t, cloneF32(aData), []int{2, 4})
	gpuB := testutil.MakeTensor(t, cloneF32(bData), []int{4, 3})
	gpuMM := core.NewMatMul[float32](gpuEng)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuA, gpuB}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuMM.Forward(ctx, gpuA, gpuB)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUClose(t, "matmul_forward", cpuOut.Data(), gpuOut.Data(), 1e-3)
}

func TestGPUParity_Conv1D(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	batch, seqLen, inCh, outCh, kernel := 2, 8, 3, 4, 3
	inputData := deterministicData(batch * seqLen * inCh)

	// CPU
	cpuInput := testutil.MakeTensor(t, inputData, []int{batch, seqLen, inCh})
	cpuConv, err := core.NewConv1D[float32]("test_conv", cpuEng, ops, inCh, outCh, kernel)
	if err != nil {
		t.Fatalf("CPU NewConv1D: %v", err)
	}

	// Set deterministic weights
	cpuParams := cpuConv.Parameters()
	weightData := deterministicData(len(cpuParams[0].Value.Data()))
	copy(cpuParams[0].Value.Data(), weightData)
	if len(cpuParams) > 1 {
		biasData := deterministicData(len(cpuParams[1].Value.Data()))
		copy(cpuParams[1].Value.Data(), biasData)
	}

	cpuOut, err := cpuConv.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU
	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), []int{batch, seqLen, inCh})
	gpuConv, err := core.NewConv1D[float32]("test_conv", gpuEng, ops, inCh, outCh, kernel)
	if err != nil {
		t.Fatalf("GPU NewConv1D: %v", err)
	}

	// Copy same weights
	gpuParams := gpuConv.Parameters()
	copy(gpuParams[0].Value.Data(), weightData)
	if len(gpuParams) > 1 && len(cpuParams) > 1 {
		copy(gpuParams[1].Value.Data(), cpuParams[1].Value.Data())
	}

	// Upload
	toUpload := []*tensor.TensorNumeric[float32]{gpuInput}
	for _, p := range gpuParams {
		toUpload = append(toUpload, p.Value)
	}
	if err := gpuRaw.UploadWeights(toUpload); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	gpuOut, err := gpuConv.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUClose(t, "conv1d_forward", cpuOut.Data(), gpuOut.Data(), 1e-3)
}

func TestGPUParity_FFN(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	inputDim, hiddenDim, outputDim := 4, 16, 4
	inputData := deterministicData(2 * inputDim)

	// CPU
	cpuInput := testutil.MakeTensor(t, inputData, []int{2, inputDim})
	cpuFFN, err := core.NewFFN[float32]("test_ffn", cpuEng, ops, inputDim, hiddenDim, outputDim, core.WithFFNNoBias[float32]())
	if err != nil {
		t.Fatalf("CPU NewFFN: %v", err)
	}

	// Set deterministic weights
	cpuParams := cpuFFN.Parameters()
	paramWeights := make([][]float32, len(cpuParams))
	for i, p := range cpuParams {
		wd := deterministicData(len(p.Value.Data()))
		copy(p.Value.Data(), wd)
		paramWeights[i] = wd
	}

	cpuOut, err := cpuFFN.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU
	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), []int{2, inputDim})
	gpuFFN, err := core.NewFFN[float32]("test_ffn", gpuEng, ops, inputDim, hiddenDim, outputDim, core.WithFFNNoBias[float32]())
	if err != nil {
		t.Fatalf("GPU NewFFN: %v", err)
	}

	gpuParams := gpuFFN.Parameters()
	for i, p := range gpuParams {
		copy(p.Value.Data(), paramWeights[i])
	}

	toUpload := []*tensor.TensorNumeric[float32]{gpuInput}
	for _, p := range gpuParams {
		toUpload = append(toUpload, p.Value)
	}
	if err := gpuRaw.UploadWeights(toUpload); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	gpuOut, err := gpuFFN.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUClose(t, "ffn_forward", cpuOut.Data(), gpuOut.Data(), 1e-3)
}

// ---------------------------------------------------------------------------
// T86.5.5 — Attention GPU parity
// ---------------------------------------------------------------------------

func TestGPUParity_SDPA_Causal(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	batch, seq, headDim := 2, 4, 8
	n := batch * seq * headDim
	qData := deterministicData(n)
	kData := deterministicData(n)
	vData := deterministicData(n)

	// CPU
	cpuQ := testutil.MakeTensor(t, qData, []int{batch, seq, headDim})
	cpuK := testutil.MakeTensor(t, kData, []int{batch, seq, headDim})
	cpuV := testutil.MakeTensor(t, vData, []int{batch, seq, headDim})
	cpuSDPA := attention.NewScaledDotProductAttention[float32](cpuEng, headDim)
	cpuSDPA.SetCausal(true)
	cpuOut, err := cpuSDPA.Forward(ctx, cpuQ, cpuK, cpuV, nil)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU
	gpuQ := testutil.MakeTensor(t, cloneF32(qData), []int{batch, seq, headDim})
	gpuK := testutil.MakeTensor(t, cloneF32(kData), []int{batch, seq, headDim})
	gpuV := testutil.MakeTensor(t, cloneF32(vData), []int{batch, seq, headDim})
	gpuSDPA := attention.NewScaledDotProductAttention[float32](gpuEng, headDim)
	gpuSDPA.SetCausal(true)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuQ, gpuK, gpuV}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuSDPA.Forward(ctx, gpuQ, gpuK, gpuV, nil)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUClose(t, "sdpa_causal", cpuOut.Data(), gpuOut.Data(), 1e-3)
}

func TestGPUParity_SDPA_Bidirectional(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	batch, seq, headDim := 2, 4, 8
	n := batch * seq * headDim
	qData := deterministicData(n)
	kData := deterministicData(n)
	vData := deterministicData(n)

	// CPU
	cpuQ := testutil.MakeTensor(t, qData, []int{batch, seq, headDim})
	cpuK := testutil.MakeTensor(t, kData, []int{batch, seq, headDim})
	cpuV := testutil.MakeTensor(t, vData, []int{batch, seq, headDim})
	cpuSDPA := attention.NewBidirectionalSDPA[float32](cpuEng, headDim)
	cpuOut, err := cpuSDPA.Forward(ctx, cpuQ, cpuK, cpuV, nil)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU
	gpuQ := testutil.MakeTensor(t, cloneF32(qData), []int{batch, seq, headDim})
	gpuK := testutil.MakeTensor(t, cloneF32(kData), []int{batch, seq, headDim})
	gpuV := testutil.MakeTensor(t, cloneF32(vData), []int{batch, seq, headDim})
	gpuSDPA := attention.NewBidirectionalSDPA[float32](gpuEng, headDim)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuQ, gpuK, gpuV}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuSDPA.Forward(ctx, gpuQ, gpuK, gpuV, nil)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUClose(t, "sdpa_bidirectional", cpuOut.Data(), gpuOut.Data(), 1e-3)
}

func TestGPUParity_GQA(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	dModel, nQHeads, nKVHeads := 32, 4, 2
	inputData := deterministicData(2 * dModel)

	// CPU
	cpuGQA, err := attention.NewGroupedQueryAttention(cpuEng, *ops, dModel, nQHeads, nKVHeads,
		attention.WithNoRoPE[float32]())
	if err != nil {
		t.Fatalf("CPU NewGQA: %v", err)
	}

	// Set deterministic weights
	cpuParams := cpuGQA.Parameters()
	paramWeights := make([][]float32, len(cpuParams))
	for i, p := range cpuParams {
		wd := deterministicData(len(p.Value.Data()))
		copy(p.Value.Data(), wd)
		paramWeights[i] = wd
	}

	cpuInput := testutil.MakeTensor(t, inputData, []int{1, 2, dModel})
	cpuOut, err := cpuGQA.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU
	gpuGQA, err := attention.NewGroupedQueryAttention(gpuEng, *ops, dModel, nQHeads, nKVHeads,
		attention.WithNoRoPE[float32]())
	if err != nil {
		t.Fatalf("GPU NewGQA: %v", err)
	}

	gpuParams := gpuGQA.Parameters()
	for i, p := range gpuParams {
		copy(p.Value.Data(), paramWeights[i])
	}

	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), []int{1, 2, dModel})
	toUpload := []*tensor.TensorNumeric[float32]{gpuInput}
	for _, p := range gpuParams {
		toUpload = append(toUpload, p.Value)
	}
	if err := gpuRaw.UploadWeights(toUpload); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	gpuOut, err := gpuGQA.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUClose(t, "gqa_forward", cpuOut.Data(), gpuOut.Data(), 1e-3)
}

// ---------------------------------------------------------------------------
// T86.5.7 — Backward GPU parity
// ---------------------------------------------------------------------------

func TestGPUParity_ReLU_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	inputData := deterministicData(2 * 8)
	gradOutData := deterministicData(2 * 8)
	shape := []int{2, 8}

	// CPU
	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuRelu := activations.NewReLU(cpuEng, ops)
	cpuOut, err := cpuRelu.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, cpuOut.Shape())
	cpuGrads, err := cpuRelu.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), shape)
	gpuRelu := activations.NewReLU(gpuEng, ops)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuRelu.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), gpuOut.Shape())
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuRelu.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "relu_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_GELU_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	inputData := deterministicData(2 * 8)
	gradOutData := deterministicData(2 * 8)
	shape := []int{2, 8}

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuGelu := activations.NewGelu(cpuEng, ops)
	cpuOut, err := cpuGelu.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, cpuOut.Shape())
	cpuGrads, err := cpuGelu.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), shape)
	gpuGelu := activations.NewGelu(gpuEng, ops)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuGelu.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), gpuOut.Shape())
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuGelu.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "gelu_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_Sigmoid_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	inputData := deterministicData(2 * 8)
	gradOutData := deterministicData(2 * 8)
	shape := []int{2, 8}

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuSigmoid := activations.NewSigmoid(cpuEng, ops)
	cpuOut, err := cpuSigmoid.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, cpuOut.Shape())
	cpuGrads, err := cpuSigmoid.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), shape)
	gpuSigmoid := activations.NewSigmoid(gpuEng, ops)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuSigmoid.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), gpuOut.Shape())
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuSigmoid.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "sigmoid_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_Tanh_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	inputData := deterministicData(2 * 8)
	gradOutData := deterministicData(2 * 8)
	shape := []int{2, 8}

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuTanh := activations.NewTanh(cpuEng, ops)
	cpuOut, err := cpuTanh.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, cpuOut.Shape())
	cpuGrads, err := cpuTanh.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), shape)
	gpuTanh := activations.NewTanh(gpuEng, ops)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuTanh.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), gpuOut.Shape())
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuTanh.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "tanh_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_LeakyReLU_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	inputData := deterministicData(2 * 8)
	gradOutData := deterministicData(2 * 8)
	shape := []int{2, 8}
	alpha := 0.01

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLRelu := activations.NewLeakyReLU(cpuEng, ops, activations.WithAlpha[float32](alpha))
	cpuOut, err := cpuLRelu.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, cpuOut.Shape())
	cpuGrads, err := cpuLRelu.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), shape)
	gpuLRelu := activations.NewLeakyReLU(gpuEng, ops, activations.WithAlpha[float32](alpha))
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuLRelu.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), gpuOut.Shape())
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuLRelu.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "leaky_relu_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_SwiGLU_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	// SwiGLU expects last dim to be even (splits in half)
	inputData := deterministicData(2 * 8)
	shape := []int{2, 8}

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuSwiGLU := activations.NewSwiGLU[float32](cpuEng, ops)
	cpuOut, err := cpuSwiGLU.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gradOutData := deterministicData(len(cpuOut.Data()))
	cpuGradOut := testutil.MakeTensor(t, gradOutData, cpuOut.Shape())
	cpuGrads, err := cpuSwiGLU.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), shape)
	gpuSwiGLU := activations.NewSwiGLU[float32](gpuEng, ops)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuSwiGLU.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), gpuOut.Shape())
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuSwiGLU.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "swiglu_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_LayerNorm_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	features := 8
	inputData := deterministicData(2 * features)
	gradOutData := deterministicData(2 * features)
	shape := []int{2, features}
	eps := float32(1e-5)

	gammaData := deterministicData(features)
	betaData := deterministicData(features)

	// CPU
	cpuGammaParam := testutil.MakeParam(t, "gamma", gammaData, []int{features})
	cpuBetaParam := testutil.MakeParam(t, "beta", betaData, []int{features})
	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLN := normalization.NewLayerNormalizationFromParams(cpuEng, eps, cpuGammaParam, cpuBetaParam)
	if _, err := cpuLN.Forward(ctx, cpuInput); err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, shape)
	cpuGrads, err := cpuLN.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuGammaParam := testutil.MakeParam(t, "gamma", cloneF32(gammaData), []int{features})
	gpuBetaParam := testutil.MakeParam(t, "beta", cloneF32(betaData), []int{features})
	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), shape)
	gpuLN := normalization.NewLayerNormalizationFromParams(gpuEng, eps, gpuGammaParam, gpuBetaParam)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput, gpuGammaParam.Value, gpuBetaParam.Value}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	if _, err := gpuLN.Forward(ctx, gpuInput); err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), shape)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuLN.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "layer_norm_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_RMSNorm_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	features := 8
	inputData := deterministicData(2 * features)
	gradOutData := deterministicData(2 * features)
	shape := []int{2, features}
	eps := float32(1e-6)

	gainData := deterministicData(features)

	// CPU
	cpuGainParam := testutil.MakeParam(t, "gain", gainData, []int{features})
	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuRMS, err := normalization.NewRMSNormFromParam(cpuEng, ops, eps, cpuGainParam)
	if err != nil {
		t.Fatalf("CPU NewRMSNorm: %v", err)
	}
	if _, err := cpuRMS.Forward(ctx, cpuInput); err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, shape)
	cpuGrads, err := cpuRMS.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuGainParam := testutil.MakeParam(t, "gain", cloneF32(gainData), []int{features})
	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), shape)
	gpuRMS, err := normalization.NewRMSNormFromParam(gpuEng, ops, eps, gpuGainParam)
	if err != nil {
		t.Fatalf("GPU NewRMSNorm: %v", err)
	}
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput, gpuGainParam.Value}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	if _, err := gpuRMS.Forward(ctx, gpuInput); err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), shape)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuRMS.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "rms_norm_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_Linear_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	inFeatures, outFeatures := 8, 4
	inputData := deterministicData(2 * inFeatures)
	weightData := deterministicData(inFeatures * outFeatures)
	gradOutData := deterministicData(2 * outFeatures)

	// CPU
	cpuInput := testutil.MakeTensor(t, inputData, []int{2, inFeatures})
	cpuWeightParam := testutil.MakeParam(t, "weight", weightData, []int{inFeatures, outFeatures})
	cpuLinear := core.NewLinearFromParam(cpuEng, cpuWeightParam)
	if _, err := cpuLinear.Forward(ctx, cpuInput); err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, []int{2, outFeatures})
	cpuGrads, err := cpuLinear.Backward(ctx, types.FullBackprop, cpuGradOut, cpuInput)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuInput := testutil.MakeTensor(t, cloneF32(inputData), []int{2, inFeatures})
	gpuWeightParam := testutil.MakeParam(t, "weight", cloneF32(weightData), []int{inFeatures, outFeatures})
	gpuLinear := core.NewLinearFromParam(gpuEng, gpuWeightParam)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput, gpuWeightParam.Value}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	if _, err := gpuLinear.Forward(ctx, gpuInput); err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), []int{2, outFeatures})
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuLinear.Backward(ctx, types.FullBackprop, gpuGradOut, gpuInput)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "linear_backward_grad_input", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)

	// Compare weight gradients
	cpuParams := cpuLinear.Parameters()
	gpuParams := gpuLinear.Parameters()
	if cpuParams[0].Gradient != nil && gpuParams[0].Gradient != nil {
		assertGPUClose(t, "linear_backward_grad_weight", cpuParams[0].Gradient.Data(), gpuParams[0].Gradient.Data(), 1e-3)
	}
}

func TestGPUParity_MatMul_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	aData := deterministicData(2 * 4)
	bData := deterministicData(4 * 3)
	gradOutData := deterministicData(2 * 3)

	// CPU
	cpuA := testutil.MakeTensor(t, aData, []int{2, 4})
	cpuB := testutil.MakeTensor(t, bData, []int{4, 3})
	cpuMM := core.NewMatMul[float32](cpuEng)
	if _, err := cpuMM.Forward(ctx, cpuA, cpuB); err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuGradOut := testutil.MakeTensor(t, gradOutData, []int{2, 3})
	cpuGrads, err := cpuMM.Backward(ctx, types.FullBackprop, cpuGradOut, cpuA, cpuB)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuA := testutil.MakeTensor(t, cloneF32(aData), []int{2, 4})
	gpuB := testutil.MakeTensor(t, cloneF32(bData), []int{4, 3})
	gpuMM := core.NewMatMul[float32](gpuEng)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuA, gpuB}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	if _, err := gpuMM.Forward(ctx, gpuA, gpuB); err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), []int{2, 3})
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuMM.Backward(ctx, types.FullBackprop, gpuGradOut, gpuA, gpuB)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "matmul_backward_grad_a", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
	assertGPUClose(t, "matmul_backward_grad_b", cpuGrads[1].Data(), gpuGrads[1].Data(), 1e-3)
}

func TestGPUParity_MSELoss_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	predData := deterministicData(2 * 4)
	targetData := deterministicData(2 * 4)
	shape := []int{2, 4}

	// CPU
	cpuPred := testutil.MakeTensor(t, predData, shape)
	cpuTarget := testutil.MakeTensor(t, targetData, shape)
	cpuMSE := loss.NewMSE(cpuEng, ops)
	if _, err := cpuMSE.Forward(ctx, cpuPred, cpuTarget); err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuDOut := testutil.MakeTensor(t, []float32{1.0}, []int{1})
	cpuGrads, err := cpuMSE.Backward(ctx, types.FullBackprop, cpuDOut, cpuPred, cpuTarget)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuPred := testutil.MakeTensor(t, cloneF32(predData), shape)
	gpuTarget := testutil.MakeTensor(t, cloneF32(targetData), shape)
	gpuMSE := loss.NewMSE(gpuEng, ops)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuPred, gpuTarget}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	if _, err := gpuMSE.Forward(ctx, gpuPred, gpuTarget); err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuDOut := testutil.MakeTensor(t, []float32{1.0}, []int{1})
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuDOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuMSE.Backward(ctx, types.FullBackprop, gpuDOut, gpuPred, gpuTarget)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "mse_backward_grad", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_BCELoss_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	// BCE needs predictions in (0,1) range — use sigmoid-like values
	predData := []float32{0.2, 0.8, 0.5, 0.9, 0.1, 0.7, 0.3, 0.6}
	targetData := []float32{0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0}
	shape := []int{2, 4}

	// CPU
	cpuPred := testutil.MakeTensor(t, predData, shape)
	cpuTarget := testutil.MakeTensor(t, targetData, shape)
	cpuBCE := loss.NewBCELoss(cpuEng, ops)
	if _, err := cpuBCE.Forward(ctx, cpuPred, cpuTarget); err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuDOut := testutil.MakeTensor(t, []float32{1.0}, []int{1})
	cpuGrads, err := cpuBCE.Backward(ctx, types.FullBackprop, cpuDOut, cpuPred, cpuTarget)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuPred := testutil.MakeTensor(t, cloneF32(predData), shape)
	gpuTarget := testutil.MakeTensor(t, cloneF32(targetData), shape)
	gpuBCE := loss.NewBCELoss(gpuEng, ops)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuPred, gpuTarget}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	if _, err := gpuBCE.Forward(ctx, gpuPred, gpuTarget); err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuDOut := testutil.MakeTensor(t, []float32{1.0}, []int{1})
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuDOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuBCE.Backward(ctx, types.FullBackprop, gpuDOut, gpuPred, gpuTarget)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "bce_backward_grad", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_CrossEntropyLoss_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	// logits: [batch, classes]
	logitData := deterministicData(3 * 5)
	targetData := []float32{2, 0, 4} // class indices
	logitShape := []int{3, 5}

	// CPU
	cpuLogits := testutil.MakeTensor(t, logitData, logitShape)
	cpuTargets := testutil.MakeTensor(t, targetData, []int{3})
	cpuCEL := loss.NewCrossEntropyLoss[float32](cpuEng)
	if _, err := cpuCEL.Forward(ctx, cpuLogits, cpuTargets); err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	cpuDOut := testutil.MakeTensor(t, []float32{1.0}, []int{1})
	cpuGrads, err := cpuCEL.Backward(ctx, types.FullBackprop, cpuDOut)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuLogits := testutil.MakeTensor(t, cloneF32(logitData), logitShape)
	gpuTargets := testutil.MakeTensor(t, cloneF32(targetData), []int{3})
	gpuCEL := loss.NewCrossEntropyLoss[float32](gpuEng)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuLogits, gpuTargets}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	if _, err := gpuCEL.Forward(ctx, gpuLogits, gpuTargets); err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuDOut := testutil.MakeTensor(t, []float32{1.0}, []int{1})
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuDOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuCEL.Backward(ctx, types.FullBackprop, gpuDOut)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	assertGPUClose(t, "cross_entropy_backward_grad", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
}

func TestGPUParity_SDPA_Backward(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, _ := gpuOpsSetup(t)
	ctx := context.Background()

	batch, seq, headDim := 2, 4, 8
	n := batch * seq * headDim
	qData := deterministicData(n)
	kData := deterministicData(n)
	vData := deterministicData(n)
	shape := []int{batch, seq, headDim}

	// CPU
	cpuQ := testutil.MakeTensor(t, qData, shape)
	cpuK := testutil.MakeTensor(t, kData, shape)
	cpuV := testutil.MakeTensor(t, vData, shape)
	cpuSDPA := attention.NewScaledDotProductAttention[float32](cpuEng, headDim)
	cpuSDPA.SetCausal(true)
	cpuOut, err := cpuSDPA.Forward(ctx, cpuQ, cpuK, cpuV, nil)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}
	gradOutData := deterministicData(len(cpuOut.Data()))
	cpuGradOut := testutil.MakeTensor(t, gradOutData, cpuOut.Shape())
	cpuGrads, err := cpuSDPA.Backward(ctx, types.FullBackprop, cpuGradOut, cpuQ, cpuK, cpuV)
	if err != nil {
		t.Fatalf("CPU Backward: %v", err)
	}

	// GPU
	gpuQ := testutil.MakeTensor(t, cloneF32(qData), shape)
	gpuK := testutil.MakeTensor(t, cloneF32(kData), shape)
	gpuV := testutil.MakeTensor(t, cloneF32(vData), shape)
	gpuSDPA := attention.NewScaledDotProductAttention[float32](gpuEng, headDim)
	gpuSDPA.SetCausal(true)
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuQ, gpuK, gpuV}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuOut, err := gpuSDPA.Forward(ctx, gpuQ, gpuK, gpuV, nil)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}
	gpuGradOut := testutil.MakeTensor(t, cloneF32(gradOutData), gpuOut.Shape())
	if err := gpuRaw.UploadWeights([]*tensor.TensorNumeric[float32]{gpuGradOut}); err != nil {
		t.Fatalf("UploadWeights grad: %v", err)
	}
	gpuGrads, err := gpuSDPA.Backward(ctx, types.FullBackprop, gpuGradOut, gpuQ, gpuK, gpuV)
	if err != nil {
		t.Fatalf("GPU Backward: %v", err)
	}

	if len(cpuGrads) > 0 && len(gpuGrads) > 0 {
		assertGPUClose(t, "sdpa_backward_grad_q", cpuGrads[0].Data(), gpuGrads[0].Data(), 1e-3)
	}
	if len(cpuGrads) > 1 && len(gpuGrads) > 1 {
		assertGPUClose(t, "sdpa_backward_grad_k", cpuGrads[1].Data(), gpuGrads[1].Data(), 1e-3)
	}
	if len(cpuGrads) > 2 && len(gpuGrads) > 2 {
		assertGPUClose(t, "sdpa_backward_grad_v", cpuGrads[2].Data(), gpuGrads[2].Data(), 1e-3)
	}
}
