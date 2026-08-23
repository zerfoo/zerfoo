package parity_test

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/tests/parity/testutil"
	"github.com/zerfoo/ztensor/tensor"
)

// TestGPUParity_GQA_FusedRepeatInterleave is the proof for T148.2: the fused
// GQA KV-head expansion path, disabled since 2026-08-09 by the ztensor#180
// mitigation, is genuinely back in use and still numerically correct.
//
// TestGPUParity_GQA on its own CANNOT establish that. GroupedQueryAttention
// falls back to Reshape -> Repeat -> Reshape whenever RepeatInterleave returns
// an error, and the fallback is correct, so a values-only parity assertion is
// green under both paths and can name neither. This test therefore asserts
// which path is in play BEFORE it asserts anything about the numbers.
func TestGPUParity_GQA_FusedRepeatInterleave(t *testing.T) {
	cpuEng, gpuEng, gpuRaw, ops := gpuOpsSetup(t)
	ctx := context.Background()

	// The claim under test. Not a skip: if the deployed libkernels.so lacks
	// launch_repeat_interleave_f32, GQA still produces correct output via the
	// fallback -- and "the fused path is restored" is then false while every
	// other assertion here passes. That silent green is the failure mode this
	// test exists to prevent.
	if !gpuRaw.FusedRepeatInterleaveAvailable() {
		t.Fatal("fused RepeatInterleave kernel is not available on this GPU; " +
			"GroupedQueryAttention would fall back to Reshape -> Repeat -> Reshape " +
			"and this test would prove nothing about the fused path (ztensor#180). " +
			"Rebuild libkernels.so from ztensor's internal/cuda/kernels " +
			"(make CUDA_ARCH=sm_121 shared) and redeploy it onto LD_LIBRARY_PATH")
	}

	// numQueryHeads != numKeyValueHeads and numKeyValueHeads > 1 is what makes
	// GroupedQueryAttention take the expansion path at all -- the condition
	// under which ztensor#180 crashed. A test with nQ == nKV would never reach
	// the code it means to cover.
	const dModel, nQHeads, nKVHeads = 32, 4, 2
	if nQHeads == nKVHeads || nKVHeads <= 1 {
		t.Fatal("head counts do not exercise the KV expansion path")
	}

	inputData := deterministicData(2 * dModel)

	cpuGQA, err := attention.NewGroupedQueryAttention(cpuEng, *ops, dModel, nQHeads, nKVHeads,
		attention.WithNoRoPE[float32]())
	if err != nil {
		t.Fatalf("CPU NewGQA: %v", err)
	}
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

	// Before ztensor#183 this call did not return an error, it killed the
	// process with SIGSEGV.
	gpuOut, err := gpuGQA.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward on the fused path: %v", err)
	}

	// Sensitivity: a saturated or degenerate CPU reference would make this
	// comparison pass against almost any GPU output (docs/lore.md L-0009 --
	// the same trap that hid the RoPE position bug behind a softmax that had
	// collapsed onto one key). Require the reference to actually vary before
	// trusting agreement with it.
	minV, maxV := cpuOut.Data()[0], cpuOut.Data()[0]
	for _, v := range cpuOut.Data() {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	const tol = 1e-3
	if float64(maxV-minV) <= tol {
		t.Fatalf("CPU reference is degenerate (range %.3e <= tolerance %.1e): "+
			"agreement with it would prove nothing", maxV-minV, tol)
	}
	t.Logf("cpu reference range=%.6e (min=%.6f max=%.6f) over %d elements",
		maxV-minV, minV, maxV, len(cpuOut.Data()))

	assertGPUClose(t, "gqa_forward_fused_repeat_interleave", cpuOut.Data(), gpuOut.Data(), tol)
}
