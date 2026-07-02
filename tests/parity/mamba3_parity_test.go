package parity_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/inference"
)

// TestMamba3Parity verifies that Mamba 3 MIMO SSM forward pass on GPU
// matches the CPU reference implementation within 1e-3 tolerance.
//
// The test builds a small Mamba 3 model with synthetic weights and runs
// the same forward pass on both CPU and CUDA engines. This validates that
// GPU kernel dispatch, memory transfers, and fused operations preserve
// numerical accuracy.
//
// Set MAMBA3_PARITY_CUDA=1 to enable CUDA testing on DGX Spark.
// Without CUDA, the test validates CPU-only determinism (two runs match).
func TestMamba3Parity(t *testing.T) {
	mc := inference.Mamba3Config{
		NumLayers:  2,
		DModel:     64,
		DState:     8,
		DConv:      4,
		DInner:     128,
		NumHeads:   4,
		VocabSize:  256,
		EOSTokenID: 0,
		RMSNormEps: 1e-5,
	}

	tensors := makeMamba3ParityTensors(mc)

	// CPU reference run.
	cpuEngine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	cpuGraph, _, err := inference.BuildMamba3MIMO(mc, cloneTensors(tensors), cpuEngine)
	if err != nil {
		t.Fatalf("CPU BuildMamba3MIMO: %v", err)
	}

	tokenIDs := []float32{1, 5, 10, 3, 7, 15}
	seqLen := len(tokenIDs)

	cpuInput, err := tensor.New([]int{1, seqLen}, tokenIDs)
	if err != nil {
		t.Fatalf("create CPU input: %v", err)
	}

	cpuOutput, err := cpuGraph.Forward(context.Background(), cpuInput)
	if err != nil {
		t.Fatalf("CPU forward: %v", err)
	}
	cpuData := cpuOutput.Data()

	// Sanity: CPU output is non-trivial.
	allZero := true
	for _, v := range cpuData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("CPU output is all zeros")
	}
	for i, v := range cpuData {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("CPU output[%d] is NaN/Inf", i)
		}
	}

	t.Run("cpu_determinism", func(t *testing.T) {
		// Second CPU run should produce identical results.
		cpuGraph2, _, err := inference.BuildMamba3MIMO(mc, cloneTensors(tensors), cpuEngine)
		if err != nil {
			t.Fatalf("CPU BuildMamba3MIMO (run 2): %v", err)
		}
		cpuInput2, _ := tensor.New([]int{1, seqLen}, tokenIDs)
		cpuOutput2, err := cpuGraph2.Forward(context.Background(), cpuInput2)
		if err != nil {
			t.Fatalf("CPU forward (run 2): %v", err)
		}
		cpuData2 := cpuOutput2.Data()
		if len(cpuData) != len(cpuData2) {
			t.Fatalf("output length mismatch: %d vs %d", len(cpuData), len(cpuData2))
		}
		for i := range cpuData {
			if cpuData[i] != cpuData2[i] {
				t.Errorf("CPU non-determinism at [%d]: %v vs %v", i, cpuData[i], cpuData2[i])
				break
			}
		}
	})

	t.Run("cuda_parity", func(t *testing.T) {
		if os.Getenv("MAMBA3_PARITY_CUDA") == "" {
			t.Skip("MAMBA3_PARITY_CUDA not set; skipping GPU parity test")
		}

		cudaEngine, err := compute.NewGPUEngine[float32](numeric.Float32Ops{}, 0)
		if err != nil {
			t.Skipf("CUDA engine unavailable: %v", err)
		}

		cudaGraph, _, err := inference.BuildMamba3MIMO(mc, cloneTensors(tensors), cudaEngine)
		if err != nil {
			t.Fatalf("CUDA BuildMamba3MIMO: %v", err)
		}

		cudaInput, _ := tensor.New([]int{1, seqLen}, tokenIDs)
		cudaOutput, err := cudaGraph.Forward(context.Background(), cudaInput)
		if err != nil {
			t.Fatalf("CUDA forward: %v", err)
		}

		cudaData := cudaOutput.Data()
		if len(cpuData) != len(cudaData) {
			t.Fatalf("output length mismatch: CPU=%d CUDA=%d", len(cpuData), len(cudaData))
		}

		const tol = 1e-3
		maxDiff := float64(0)
		failCount := 0
		for i := range cpuData {
			diff := math.Abs(float64(cpuData[i]) - float64(cudaData[i]))
			if diff > maxDiff {
				maxDiff = diff
			}
			if diff > tol {
				failCount++
				if failCount <= 5 {
					t.Errorf("parity violation at [%d]: CPU=%v CUDA=%v diff=%v (tol=%v)",
						i, cpuData[i], cudaData[i], diff, tol)
				}
			}
		}
		if failCount > 5 {
			t.Errorf("... and %d more parity violations", failCount-5)
		}
		t.Logf("Mamba 3 CPU/CUDA parity: max_diff=%.6e over %d values (tol=%.0e)",
			maxDiff, len(cpuData), tol)
	})

	t.Run("cuda_multi_head_parity", func(t *testing.T) {
		if os.Getenv("MAMBA3_PARITY_CUDA") == "" {
			t.Skip("MAMBA3_PARITY_CUDA not set; skipping GPU parity test")
		}

		// Test with different head counts to ensure multi-head MIMO
		// SSM dispatch is correct on GPU.
		for _, numHeads := range []int{1, 2, 4} {
			t.Run(fmt.Sprintf("heads_%d", numHeads), func(t *testing.T) {
				mc2 := mc
				mc2.NumHeads = numHeads
				if mc2.DInner%numHeads != 0 {
					t.Skipf("DInner=%d not divisible by numHeads=%d", mc2.DInner, numHeads)
				}

				tensors2 := makeMamba3ParityTensors(mc2)

				cpuG, _, err := inference.BuildMamba3MIMO(mc2, cloneTensors(tensors2), cpuEngine)
				if err != nil {
					t.Fatalf("CPU build: %v", err)
				}
				cpuIn, _ := tensor.New([]int{1, 4}, []float32{1, 5, 10, 3})
				cpuOut, err := cpuG.Forward(context.Background(), cpuIn)
				if err != nil {
					t.Fatalf("CPU forward: %v", err)
				}

				cudaEngine, err := compute.NewGPUEngine[float32](numeric.Float32Ops{}, 0)
				if err != nil {
					t.Skipf("CUDA engine unavailable: %v", err)
				}
				cudaG, _, err := inference.BuildMamba3MIMO(mc2, cloneTensors(tensors2), cudaEngine)
				if err != nil {
					t.Fatalf("CUDA build: %v", err)
				}
				cudaIn, _ := tensor.New([]int{1, 4}, []float32{1, 5, 10, 3})
				cudaOut, err := cudaG.Forward(context.Background(), cudaIn)
				if err != nil {
					t.Fatalf("CUDA forward: %v", err)
				}

				cpuD := cpuOut.Data()
				cudaD := cudaOut.Data()
				const tol = 1e-3
				maxDiff := float64(0)
				for i := range cpuD {
					diff := math.Abs(float64(cpuD[i]) - float64(cudaD[i]))
					if diff > maxDiff {
						maxDiff = diff
					}
					if diff > tol {
						t.Errorf("heads=%d parity violation at [%d]: CPU=%v CUDA=%v diff=%v",
							numHeads, i, cpuD[i], cudaD[i], diff)
						break
					}
				}
				t.Logf("heads=%d max_diff=%.6e", numHeads, maxDiff)
			})
		}
	})
}

// makeMamba3ParityTensors creates deterministic synthetic weights for parity testing.
func makeMamba3ParityTensors(mc inference.Mamba3Config) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	headDim := mc.DInner / mc.NumHeads
	dtRank := int(math.Ceil(float64(mc.DModel) / 16))

	fill := func(shape []int, scale float32) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = scale * float32(math.Sin(float64(i)*0.01))
		}
		t, _ := tensor.New(shape, data)
		return t
	}
	ones := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 1.0
		}
		t, _ := tensor.New(shape, data)
		return t
	}
	logInit := func(shape []int) *tensor.TensorNumeric[float32] {
		size := 1
		for _, d := range shape {
			size *= d
		}
		data := make([]float32, size)
		cols := shape[len(shape)-1]
		for i := range data {
			data[i] = float32(math.Log(float64(i%cols + 1)))
		}
		t, _ := tensor.New(shape, data)
		return t
	}

	tensors["token_embd.weight"] = fill([]int{mc.VocabSize, mc.DModel}, 0.02)
	tensors["output.weight"] = fill([]int{mc.VocabSize, mc.DModel}, 0.02)
	tensors["output_norm.weight"] = ones([]int{mc.DModel})

	for i := 0; i < mc.NumLayers; i++ {
		prefix := fmt.Sprintf("mamba3.%d.", i)
		tensors[prefix+"norm.weight"] = ones([]int{mc.DModel})
		tensors[prefix+"in_proj.weight"] = fill([]int{2 * mc.DInner, mc.DModel}, 0.02)
		tensors[prefix+"conv1d.weight"] = fill([]int{mc.DInner, 1, mc.DConv}, 0.02)
		tensors[prefix+"x_proj.weight"] = fill([]int{dtRank + 2*mc.DState*mc.NumHeads, mc.DInner}, 0.02)
		tensors[prefix+"dt_proj.weight"] = fill([]int{mc.DInner, dtRank}, 0.02)

		for h := 0; h < mc.NumHeads; h++ {
			tensors[fmt.Sprintf("%sA_log.%d", prefix, h)] = logInit([]int{headDim, mc.DState})
			tensors[fmt.Sprintf("%sD.%d", prefix, h)] = ones([]int{headDim})
		}

		tensors[prefix+"head_mix.weight"] = fill([]int{mc.DInner, mc.DInner}, 0.02)
		tensors[prefix+"out_proj.weight"] = fill([]int{mc.DModel, mc.DInner}, 0.02)
	}

	return tensors
}

// cloneTensors creates a deep copy of a tensor map so each engine gets
// independent tensor objects (avoids aliasing when uploading to GPU).
func cloneTensors(src map[string]*tensor.TensorNumeric[float32]) map[string]*tensor.TensorNumeric[float32] {
	dst := make(map[string]*tensor.TensorNumeric[float32], len(src))
	for k, v := range src {
		data := v.Data()
		cloned := make([]float32, len(data))
		copy(cloned, data)
		t, _ := tensor.New(v.Shape(), cloned)
		dst[k] = t
	}
	return dst
}
