package crossasset

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// TestTrainGPU_CPUParity trains the same CrossAsset model configuration with
// both CPU and GPU engines, then compares loss curves and final accuracy.
//
// On CPU-only machines (CI, laptops), NewGPUEngine returns an error and the
// test skips cleanly. On DGX / GPU machines, the test validates that GPU
// training produces numerically equivalent results to CPU training.
func TestTrainGPU_CPUParity(t *testing.T) {
	ops := numeric.Float32Ops{}
	gpuEng, err := compute.NewGPUEngine[float32](ops)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}

	cfg := Config{
		NSources:          3,
		FeaturesPerSource: 4,
		DModel:            8,
		NHeads:            2,
		NLayers:           1,
		DropoutRate:       0.0,
		LearningRate:      0.01,
	}

	tc := TrainConfig{
		Epochs:       10,
		BatchSize:    10,
		LearningRate: 0.01,
	}

	data, labels := testData(cfg, 20)

	// --- CPU training ---
	cpuModel := NewModel(cfg)
	cpuEng := compute.NewCPUEngine[float32](ops)

	cpuResult, err := cpuModel.TrainGPU(data, labels, tc, cpuEng)
	if err != nil {
		t.Fatalf("CPU TrainGPU: %v", err)
	}

	// --- GPU training ---
	// Fresh model with identical initial weights requires same seed.
	// NewModel uses deterministic initialization, so two NewModel calls
	// with the same config produce identical weights.
	gpuModel := NewModel(cfg)

	gpuResult, err := gpuModel.TrainGPU(data, labels, tc, gpuEng)
	if err != nil {
		t.Fatalf("GPU TrainGPU: %v", err)
	}

	// --- Compare loss curves ---
	if len(cpuResult.Losses) != len(gpuResult.Losses) {
		t.Fatalf("loss length mismatch: CPU=%d GPU=%d",
			len(cpuResult.Losses), len(gpuResult.Losses))
	}

	const lossTol = 1e-3
	for i := range cpuResult.Losses {
		diff := math.Abs(cpuResult.Losses[i] - gpuResult.Losses[i])
		if diff > lossTol {
			t.Errorf("epoch %d loss mismatch: CPU=%.6f GPU=%.6f diff=%.6f (tol=%.6f)",
				i, cpuResult.Losses[i], gpuResult.Losses[i], diff, lossTol)
		}
	}

	t.Logf("CPU final loss: %.6f, GPU final loss: %.6f",
		cpuResult.Losses[len(cpuResult.Losses)-1],
		gpuResult.Losses[len(gpuResult.Losses)-1])

	// --- Compare final accuracy ---
	accDiff := math.Abs(cpuResult.FinalAccuracy - gpuResult.FinalAccuracy)
	const accTol = 0.02 // 2%
	if accDiff > accTol {
		t.Errorf("accuracy mismatch: CPU=%.4f GPU=%.4f diff=%.4f (tol=%.4f)",
			cpuResult.FinalAccuracy, gpuResult.FinalAccuracy, accDiff, accTol)
	}
	t.Logf("CPU accuracy: %.4f, GPU accuracy: %.4f",
		cpuResult.FinalAccuracy, gpuResult.FinalAccuracy)

	// --- Sanity: losses should be finite ---
	for i, l := range gpuResult.Losses {
		if math.IsNaN(l) || math.IsInf(l, 0) {
			t.Errorf("GPU epoch %d: loss is %v", i, l)
		}
	}
}
