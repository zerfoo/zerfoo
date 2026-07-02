package parity_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"
)

// gpuSetup creates CPU and GPU engines for GPU parity tests.
// Tests are skipped when no GPU is available.
func gpuSetup(t *testing.T) (compute.Engine[float32], *compute.GPUEngine[float32], *numeric.Float32Ops) {
	t.Helper()
	ops := &numeric.Float32Ops{}
	cpu := compute.NewCPUEngine[float32](ops)
	gpu, err := compute.NewGPUEngine[float32](*ops, 0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	os.Setenv("ZERFOO_DISABLE_CUDA_GRAPH", "1")
	return cpu, gpu, ops
}

// assertGPUParity compares CPU and GPU output element-by-element.
func assertGPUParity(t *testing.T, name string, cpuData, gpuData []float32, tol float64) {
	t.Helper()
	if len(cpuData) != len(gpuData) {
		t.Fatalf("%s: length mismatch: cpu=%d gpu=%d", name, len(cpuData), len(gpuData))
	}
	maxDiff := float64(0)
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
	t.Logf("%s: maxDiff=%.6e at idx=%d (cpu=%.6f gpu=%.6f), mismatches=%d/%d",
		name, maxDiff, maxIdx, cpuData[maxIdx], gpuData[maxIdx], mismatches, len(cpuData))
	if mismatches > 0 {
		shown := 0
		for i := range cpuData {
			d := math.Abs(float64(cpuData[i] - gpuData[i]))
			if d > tol {
				t.Errorf("%s[%d]: cpu=%g gpu=%g diff=%g", name, i, cpuData[i], gpuData[i], d)
				shown++
				if shown >= 5 {
					break
				}
			}
		}
		t.Errorf("%s: %d/%d values exceed tolerance %g (maxDiff=%g)", name, mismatches, len(cpuData), tol, maxDiff)
	}
}

// deterministicInput generates deterministic float32 data.
func deterministicInput(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i%97)*0.03 - 1.5
	}
	return data
}

// ---------------------------------------------------------------------------
// T86.5.2 — Activation GPU parity tests (tolerance 1e-4)
// ---------------------------------------------------------------------------

func TestGPUParity_ReLU(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewReLU(cpuEng, ops)
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewReLU(compute.Engine[float32](gpuEng), ops)
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "relu", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_GELU(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewGelu(cpuEng, ops)
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewGelu(compute.Engine[float32](gpuEng), ops)
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "gelu", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_Sigmoid(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewSigmoid(cpuEng, ops)
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewSigmoid(compute.Engine[float32](gpuEng), ops)
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "sigmoid", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_Tanh(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewTanh(cpuEng, ops)
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewTanh(compute.Engine[float32](gpuEng), ops)
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "tanh", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_Softmax(t *testing.T) {
	cpuEng, gpuEng, _ := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewSoftmax[float32](cpuEng, -1)
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewSoftmax[float32](compute.Engine[float32](gpuEng), -1)
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "softmax", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_LeakyReLU(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewLeakyReLU(cpuEng, ops, activations.WithAlpha[float32](0.01))
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewLeakyReLU(compute.Engine[float32](gpuEng), ops, activations.WithAlpha[float32](0.01))
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "leaky_relu", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_SwiGLU(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	// SwiGLU splits on last dim, so last dim must be even
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewSwiGLU[float32](cpuEng, ops)
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewSwiGLU[float32](compute.Engine[float32](gpuEng), ops)
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "swiglu", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_Erf(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewErf(cpuEng, ops)
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewErf(compute.Engine[float32](gpuEng), ops)
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "erf", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_FastGelu(t *testing.T) {
	cpuEng, gpuEng, _ := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuLayer := activations.NewFastGelu[float32](cpuEng)
	cpuOut, err := cpuLayer.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLayer := activations.NewFastGelu[float32](compute.Engine[float32](gpuEng))
	gpuOut, err := gpuLayer.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "fast_gelu", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

// ---------------------------------------------------------------------------
// T86.5.3 — Normalization GPU parity tests (tolerance 1e-4)
// ---------------------------------------------------------------------------

func TestGPUParity_LayerNorm(t *testing.T) {
	cpuEng, gpuEng, _ := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	hiddenSize := 8
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	// Create gamma and beta parameters
	gammaData := make([]float32, hiddenSize)
	betaData := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		gammaData[i] = 1.0 + float32(i)*0.1
		betaData[i] = float32(i) * 0.05
	}

	// CPU path
	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuGamma := testutil.MakeParam(t, "gamma", gammaData, []int{hiddenSize})
	cpuBeta := testutil.MakeParam(t, "beta", betaData, []int{hiddenSize})
	cpuLN := normalization.NewLayerNormalizationFromParams(cpuEng, float32(1e-5), cpuGamma, cpuBeta)
	cpuOut, err := cpuLN.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU path
	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	gpuGamma := testutil.MakeParam(t, "gamma", append([]float32(nil), gammaData...), []int{hiddenSize})
	gpuBeta := testutil.MakeParam(t, "beta", append([]float32(nil), betaData...), []int{hiddenSize})
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{
		gpuInput, gpuGamma.Value, gpuBeta.Value,
	}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuLN := normalization.NewLayerNormalizationFromParams(compute.Engine[float32](gpuEng), float32(1e-5), gpuGamma, gpuBeta)
	gpuOut, err := gpuLN.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "layer_norm", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_RMSNorm(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	shape := []int{2, 4, 8}
	hiddenSize := 8
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	// Create gain parameter
	gainData := make([]float32, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		gainData[i] = 1.0 + float32(i)*0.1
	}

	// CPU path
	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuGain := testutil.MakeParam(t, "gain", gainData, []int{hiddenSize})
	cpuRMS, err := normalization.NewRMSNormFromParam(cpuEng, ops, float32(1e-5), cpuGain)
	if err != nil {
		t.Fatalf("NewRMSNormFromParam CPU: %v", err)
	}
	cpuOut, err := cpuRMS.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU path
	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	gpuGain := testutil.MakeParam(t, "gain", append([]float32(nil), gainData...), []int{hiddenSize})
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{
		gpuInput, gpuGain.Value,
	}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuRMS, err := normalization.NewRMSNormFromParam(compute.Engine[float32](gpuEng), ops, float32(1e-5), gpuGain)
	if err != nil {
		t.Fatalf("NewRMSNormFromParam GPU: %v", err)
	}
	gpuOut, err := gpuRMS.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "rms_norm", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

func TestGPUParity_BatchNorm(t *testing.T) {
	cpuEng, gpuEng, ops := gpuSetup(t)
	ctx := context.Background()
	// BatchNorm expects [N, C, ...] layout; use [2, 4, 8]
	shape := []int{2, 4, 8}
	numChannels := 4
	n := 2 * 4 * 8
	inputData := deterministicInput(n)

	// Create per-channel scale, bias, running mean, running var
	scaleData := make([]float32, numChannels)
	biasData := make([]float32, numChannels)
	meanData := make([]float32, numChannels)
	varData := make([]float32, numChannels)
	for i := 0; i < numChannels; i++ {
		scaleData[i] = 1.0 + float32(i)*0.1
		biasData[i] = float32(i) * 0.05
		meanData[i] = float32(i) * 0.01
		varData[i] = 1.0 + float32(i)*0.02
	}

	// CPU path
	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuScale := testutil.MakeTensor(t, scaleData, []int{numChannels})
	cpuBias := testutil.MakeTensor(t, biasData, []int{numChannels})
	cpuMean := testutil.MakeTensor(t, meanData, []int{numChannels})
	cpuVar := testutil.MakeTensor(t, varData, []int{numChannels})
	cpuBN := normalization.NewBatchNormalization[float32](cpuEng, ops, float32(1e-5))
	cpuOut, err := cpuBN.Forward(ctx, cpuInput, cpuScale, cpuBias, cpuMean, cpuVar)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU path
	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	gpuScale := testutil.MakeTensor(t, append([]float32(nil), scaleData...), []int{numChannels})
	gpuBias := testutil.MakeTensor(t, append([]float32(nil), biasData...), []int{numChannels})
	gpuMean := testutil.MakeTensor(t, append([]float32(nil), meanData...), []int{numChannels})
	gpuVar := testutil.MakeTensor(t, append([]float32(nil), varData...), []int{numChannels})
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{
		gpuInput, gpuScale, gpuBias, gpuMean, gpuVar,
	}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuBN := normalization.NewBatchNormalization[float32](compute.Engine[float32](gpuEng), ops, float32(1e-5))
	gpuOut, err := gpuBN.Forward(ctx, gpuInput, gpuScale, gpuBias, gpuMean, gpuVar)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "batch_norm", cpuOut.Data(), gpuOut.Data(), 1e-4)
}

// ---------------------------------------------------------------------------
// T86.5.6 — RotaryEmbedding GPU parity test (tolerance 1e-5)
// ---------------------------------------------------------------------------

func TestGPUParity_RotaryEmbedding(t *testing.T) {
	cpuEng, gpuEng, _ := gpuSetup(t)
	ctx := context.Background()
	headDim := 8
	seqLen := 4
	// RoPE input shape: [numHeads, seqLen, headDim]
	numHeads := 2
	shape := []int{numHeads, seqLen, headDim}
	n := numHeads * seqLen * headDim
	inputData := deterministicInput(n)

	// CPU path
	cpuInput := testutil.MakeTensor(t, inputData, shape)
	cpuRoPE, err := embeddings.NewRotaryPositionalEmbedding[float32](
		ctx, cpuEng, headDim, seqLen, embeddings.WithRotaryBase(10000.0),
	)
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding CPU: %v", err)
	}
	cpuOut, err := cpuRoPE.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	// GPU path
	gpuInput := testutil.MakeTensor(t, append([]float32(nil), inputData...), shape)
	if err := gpuEng.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}
	gpuRoPE, err := embeddings.NewRotaryPositionalEmbedding[float32](
		ctx, compute.Engine[float32](gpuEng), headDim, seqLen, embeddings.WithRotaryBase(10000.0),
	)
	if err != nil {
		t.Fatalf("NewRotaryPositionalEmbedding GPU: %v", err)
	}
	gpuOut, err := gpuRoPE.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	assertGPUParity(t, "rotary_embedding", cpuOut.Data(), gpuOut.Data(), 1e-5)
}
