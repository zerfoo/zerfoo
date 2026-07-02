package parity_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGQA_MultiKVDecode tests GQA decode (seqLen=1) with numKVHeads > 1.
// This is the configuration used by Llama 3.2 (24 query heads, 8 KV heads)
// where the decode path diverges between CPU and GPU.
//
// The test runs a prefill (seqLen=2) then a decode (seqLen=1) on both
// CPU and GPU and compares the decode output.
func TestGQA_MultiKVDecode(t *testing.T) {
	ops := numeric.Float32Ops{}
	hiddenSize := 256 // small for fast test
	numQueryHeads := 6
	numKVHeads := 2
	headDim := 32 // hiddenSize / numQueryHeads (approximately)
	qDim := numQueryHeads * headDim
	kDim := numKVHeads * headDim
	ctx := context.Background()

	// Create random but deterministic weights.
	makeWeight := func(rows, cols int) *tensor.TensorNumeric[float32] {
		data := make([]float32, rows*cols)
		for i := range data {
			data[i] = float32(i%73)*0.01 - 0.36
		}
		t, _ := tensor.New[float32]([]int{rows, cols}, data)
		return t
	}

	qW := makeWeight(qDim, hiddenSize)   // [192, 256]
	kW := makeWeight(kDim, hiddenSize)    // [64, 256]
	vW := makeWeight(kDim, hiddenSize)    // [64, 256]
	oW := makeWeight(hiddenSize, qDim)    // [256, 192]
	qNormData := make([]float32, headDim)
	kNormData := make([]float32, headDim)
	for i := range qNormData {
		qNormData[i] = 1.0
		kNormData[i] = 1.0
	}
	qNormW, _ := tensor.New[float32]([]int{headDim}, qNormData)
	kNormW, _ := tensor.New[float32]([]int{headDim}, kNormData)

	// Create input [1, 2, 256] for prefill.
	prefillData := make([]float32, 2*hiddenSize)
	for i := range prefillData {
		prefillData[i] = float32(i%41)*0.02 - 0.4
	}

	// Decode input [1, 1, 256].
	decodeData := make([]float32, hiddenSize)
	for i := range decodeData {
		decodeData[i] = float32(i%37)*0.03 - 0.5
	}

	// Helper to build a GQA layer.
	buildGQA := func(eng compute.Engine[float32]) *attention.GroupedQueryAttention[float32] {
		t.Helper()
		param := func(name string, w *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
			return &graph.Parameter[float32]{Name: name, Value: w}
		}
		// Virtual transpose weights.
		vt := func(w *tensor.TensorNumeric[float32]) *tensor.TensorNumeric[float32] {
			s := w.Shape()
			wt, _ := tensor.NewWithStorage[float32]([]int{s[1], s[0]}, w.GetStorage())
			return wt
		}
		wq := core.NewDenseFromParams[float32](core.NewLinearFromParam[float32](eng, param("q", vt(qW))), nil)
		wk := core.NewDenseFromParams[float32](core.NewLinearFromParam[float32](eng, param("k", vt(kW))), nil)
		wv := core.NewDenseFromParams[float32](core.NewLinearFromParam[float32](eng, param("v", vt(vW))), nil)
		wo := core.NewDenseFromParams[float32](core.NewLinearFromParam[float32](eng, param("o", vt(oW))), nil)

		rope, _ := embeddings.NewRotaryPositionalEmbedding[float32](
			ctx, eng, headDim, 1024, embeddings.WithRotaryBase(10000.0),
		)
		gqa, _ := attention.NewGroupedQueryAttentionFromParams[float32](
			eng, ops, hiddenSize, numQueryHeads, numKVHeads,
			wq, wk, wv, wo, rope, headDim,
		)
		qNorm, _ := normalization.NewRMSNormFromParam[float32](eng, ops, 1e-6, param("qn", qNormW))
		kNorm, _ := normalization.NewRMSNormFromParam[float32](eng, ops, 1e-6, param("kn", kNormW))
		gqa.SetQKNorms(qNorm, kNorm)
		return gqa
	}

	// --- CPU ---
	cpuEng := compute.NewCPUEngine[float32](ops)
	cpuGQA := buildGQA(compute.Engine[float32](cpuEng))

	cpuCache := generate.NewKVCache[float32](1, 128)
	cpuPrefillCtx := generate.WithCache[float32](ctx, cpuCache)

	cpuPrefillInput, _ := tensor.New[float32]([]int{1, 2, hiddenSize}, prefillData)
	cpuPrefillOut, err := cpuGQA.Forward(cpuPrefillCtx, cpuPrefillInput)
	if err != nil {
		t.Fatalf("CPU prefill: %v", err)
	}
	t.Logf("CPU prefill shape=%v sum=%.6f", cpuPrefillOut.Shape(), tensorSum(cpuPrefillOut))

	cpuDecodeInput, _ := tensor.New[float32]([]int{1, 1, hiddenSize}, decodeData)
	cpuDecodeOut, err := cpuGQA.Forward(cpuPrefillCtx, cpuDecodeInput)
	if err != nil {
		t.Fatalf("CPU decode: %v", err)
	}
	t.Logf("CPU decode shape=%v sum=%.6f", cpuDecodeOut.Shape(), tensorSum(cpuDecodeOut))

	// --- GPU ---
	gpuEngPtr, err := compute.NewGPUEngine[float32](ops, 0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	os.Setenv("ZERFOO_DISABLE_CUDA_GRAPH", "1")
	gpuGQA := buildGQA(compute.Engine[float32](gpuEngPtr))

	gpuCache := generate.NewKVCache[float32](1, 128)
	gpuPrefillCtx := generate.WithCache[float32](ctx, gpuCache)

	gpuPrefillInput, _ := tensor.New[float32]([]int{1, 2, hiddenSize}, append([]float32(nil), prefillData...))
	gpuPrefillOut, err := gpuGQA.Forward(gpuPrefillCtx, gpuPrefillInput)
	if err != nil {
		t.Fatalf("GPU prefill: %v", err)
	}
	t.Logf("GPU prefill shape=%v sum=%.6f", gpuPrefillOut.Shape(), tensorSum(gpuPrefillOut))

	gpuDecodeInput, _ := tensor.New[float32]([]int{1, 1, hiddenSize}, append([]float32(nil), decodeData...))
	gpuDecodeOut, err := gpuGQA.Forward(gpuPrefillCtx, gpuDecodeInput)
	if err != nil {
		t.Fatalf("GPU decode: %v", err)
	}
	t.Logf("GPU decode shape=%v sum=%.6f", gpuDecodeOut.Shape(), tensorSum(gpuDecodeOut))

	// --- Compare prefill ---
	compareTensors(t, "prefill", cpuPrefillOut, gpuPrefillOut)

	// --- Compare decode ---
	compareTensors(t, "decode", cpuDecodeOut, gpuDecodeOut)
}

func tensorSum(t *tensor.TensorNumeric[float32]) float64 {
	s := float64(0)
	for _, v := range t.Data() {
		s += float64(v)
	}
	return s
}

func compareTensors(t *testing.T, label string, cpu, gpu *tensor.TensorNumeric[float32]) {
	t.Helper()
	cD := cpu.Data()
	gD := gpu.Data()
	if len(cD) != len(gD) {
		t.Errorf("[%s] length mismatch: cpu=%d gpu=%d", label, len(cD), len(gD))
		return
	}
	maxDiff := float64(0)
	maxIdx := 0
	for i := range cD {
		d := math.Abs(float64(cD[i] - gD[i]))
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	status := "PASS"
	if maxDiff > 0.01 {
		status = "FAIL"
		t.Errorf("[%s] maxDiff=%.6e at idx=%d (cpu=%.6f gpu=%.6f)", label, maxDiff, maxIdx, cD[maxIdx], gD[maxIdx])
	}
	t.Logf("[%s] maxDiff=%.6e (%s)", label, maxDiff, status)
}
