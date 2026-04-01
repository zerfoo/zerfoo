package parity_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGQA_GPUParity runs the full GroupedQueryAttention Forward on both CPU
// and GPU engines with identical inputs and compares the output element-by-element.
// Set GEMMA3_GGUF_PATH to the Gemma3-1B GGUF model path to enable.
func TestGQA_GPUParity(t *testing.T) {
	modelPath := os.Getenv("GEMMA3_GGUF_PATH")
	if modelPath == "" {
		t.Skip("GEMMA3_GGUF_PATH not set; skipping GQA GPU parity test")
	}

	gm, err := inference.LoadGGUF(modelPath)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}

	prefix := "model.layers.0."
	qW := gm.Tensors[prefix+"self_attn.q_proj.weight"]
	kW := gm.Tensors[prefix+"self_attn.k_proj.weight"]
	vW := gm.Tensors[prefix+"self_attn.v_proj.weight"]
	oW := gm.Tensors[prefix+"self_attn.o_proj.weight"]
	qNormW := gm.Tensors[prefix+"self_attn.q_norm.weight"]
	kNormW := gm.Tensors[prefix+"self_attn.k_norm.weight"]

	for _, pair := range []struct {
		name string
		w    *tensor.TensorNumeric[float32]
	}{
		{"q_proj", qW}, {"k_proj", kW}, {"v_proj", vW}, {"o_proj", oW},
		{"q_norm", qNormW}, {"k_norm", kNormW},
	} {
		if pair.w == nil {
			t.Fatalf("missing %s%s.weight", prefix, pair.name)
		}
		t.Logf("%s: shape=%v storage=%T", pair.name, pair.w.Shape(), pair.w.GetStorage())
	}

	ops := numeric.Float32Ops{}
	hiddenSize := 1152
	numQueryHeads := 4
	numKVHeads := 1
	headDim := 256

	ctx := context.Background()

	inputData := make([]float32, 2*hiddenSize)
	for i := range inputData {
		inputData[i] = float32(i%97)*0.03 - 1.5
	}

	// --- CPU GQA ---
	cpuEng := compute.NewCPUEngine[float32](ops)
	cpuGQA := buildTestGQA(t, compute.Engine[float32](cpuEng), ops,
		qW, kW, vW, oW, qNormW, kNormW,
		hiddenSize, numQueryHeads, numKVHeads, headDim, false)

	cpuInput, err := tensor.New[float32]([]int{1, 2, hiddenSize}, inputData)
	if err != nil {
		t.Fatalf("CPU input: %v", err)
	}
	cpuOut, err := cpuGQA.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU GQA Forward: %v", err)
	}

	// --- GPU GQA ---
	gpuEngPtr, err := compute.NewGPUEngine[float32](ops, 0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	gpuGQA := buildTestGQA(t, compute.Engine[float32](gpuEngPtr), ops,
		qW, kW, vW, oW, qNormW, kNormW,
		hiddenSize, numQueryHeads, numKVHeads, headDim, true)

	gpuInput, err := tensor.New[float32]([]int{1, 2, hiddenSize}, append([]float32(nil), inputData...))
	if err != nil {
		t.Fatalf("GPU input: %v", err)
	}
	if err := gpuEngPtr.UploadWeights([]*tensor.TensorNumeric[float32]{gpuInput}); err != nil {
		t.Fatalf("UploadWeights (input): %v", err)
	}

	os.Setenv("ZERFOO_DISABLE_CUDA_GRAPH", "1")
	gpuOut, err := gpuGQA.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU GQA Forward: %v", err)
	}

	// --- Compare ---
	cpuData := cpuOut.Data()
	gpuData := gpuOut.Data()

	if len(cpuData) != len(gpuData) {
		t.Fatalf("length mismatch: cpu=%d gpu=%d", len(cpuData), len(gpuData))
	}

	maxDiff := float64(0)
	maxIdx := 0
	cpuSum, gpuSum := float64(0), float64(0)
	bigDiffs := 0
	for i := range cpuData {
		cpuSum += float64(cpuData[i])
		gpuSum += float64(gpuData[i])
		d := math.Abs(float64(cpuData[i] - gpuData[i]))
		if d > 0.01 {
			bigDiffs++
		}
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}

	t.Logf("output shape=%v len=%d", cpuOut.Shape(), len(cpuData))
	t.Logf("CPU sum=%.6f GPU sum=%.6f", cpuSum, gpuSum)
	t.Logf("maxDiff=%.6e at idx=%d (cpu=%.6f gpu=%.6f)", maxDiff, maxIdx, cpuData[maxIdx], gpuData[maxIdx])
	t.Logf("diffs>0.01: %d/%d", bigDiffs, len(cpuData))

	if maxDiff > 0.1 {
		t.Errorf("FAIL: maxDiff=%.4f exceeds threshold 0.1", maxDiff)
		cols := cpuOut.Shape()[len(cpuOut.Shape())-1]
		rows := len(cpuData) / cols
		for r := 0; r < rows; r++ {
			rowMax := float64(0)
			for c := 0; c < cols; c++ {
				d := math.Abs(float64(cpuData[r*cols+c] - gpuData[r*cols+c]))
				if d > rowMax {
					rowMax = d
				}
			}
			t.Logf("  row %d: maxDiff=%.6e", r, rowMax)
		}
	}
}

func buildTestGQA(
	t *testing.T,
	engine compute.Engine[float32],
	ops numeric.Arithmetic[float32],
	qW, kW, vW, oW, qNormW, kNormW *tensor.TensorNumeric[float32],
	hiddenSize, numQueryHeads, numKVHeads, headDim int,
	isGPU bool,
) *attention.GroupedQueryAttention[float32] {
	t.Helper()

	vt := func(w *tensor.TensorNumeric[float32]) *tensor.TensorNumeric[float32] {
		s := w.Shape()
		wt, err := tensor.NewWithStorage[float32]([]int{s[1], s[0]}, w.GetStorage())
		if err != nil {
			t.Fatalf("virtual transpose: %v", err)
		}
		return wt
	}

	param := func(name string, w *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: w}
	}

	qWT := vt(qW)
	kWT := vt(kW)
	vWT := vt(vW)
	oWT := vt(oW)

	if isGPU {
		type uploader interface {
			UploadWeights([]*tensor.TensorNumeric[float32]) error
		}
		if up, ok := engine.(uploader); ok {
			if err := up.UploadWeights([]*tensor.TensorNumeric[float32]{
				qWT, kWT, vWT, oWT, qNormW, kNormW,
			}); err != nil {
				t.Fatalf("UploadWeights: %v", err)
			}
		}
	}

	wq := core.NewDenseFromParams[float32](
		core.NewLinearFromParam[float32](engine, param("q_proj.weight", qWT)),
		nil,
	)
	wk := core.NewDenseFromParams[float32](
		core.NewLinearFromParam[float32](engine, param("k_proj.weight", kWT)),
		nil,
	)
	wv := core.NewDenseFromParams[float32](
		core.NewLinearFromParam[float32](engine, param("v_proj.weight", vWT)),
		nil,
	)
	wo := core.NewDenseFromParams[float32](
		core.NewLinearFromParam[float32](engine, param("o_proj.weight", oWT)),
		nil,
	)

	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, headDim, 8192,
		embeddings.WithRotaryBase(10000.0),
	)
	if err != nil {
		t.Fatalf("RoPE: %v", err)
	}

	gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
		engine, ops, hiddenSize, numQueryHeads, numKVHeads,
		wq, wk, wv, wo, rope, headDim,
	)
	if err != nil {
		t.Fatalf("GQA: %v", err)
	}

	qNorm, err := normalization.NewRMSNormFromParam[float32](engine, ops, 1e-6, param("q_norm.weight", qNormW))
	if err != nil {
		t.Fatalf("qNorm: %v", err)
	}
	kNorm, err := normalization.NewRMSNormFromParam[float32](engine, ops, 1e-6, param("k_norm.weight", kNormW))
	if err != nil {
		t.Fatalf("kNorm: %v", err)
	}
	gqa.SetQKNorms(qNorm, kNorm)

	return gqa
}
