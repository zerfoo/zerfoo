package timeseries

import (
	"math"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// TestGPUTinyTrainingConvergence is the end-to-end regression test for T4.2:
// GPU PatchTST training must actually reduce loss on trivial synthetic data.
//
// Skip mechanism: attempt to construct a GPU engine via compute.NewGPUEngine.
// If CUDA is unavailable (CPU-only CI, no libcudart), the call returns an
// error and we skip cleanly — mirrors the dispatch used by cmd/bench_train.
//
// Pre-fix expectation (GPU convergence regression under investigation in
// docs/plans/gpu-training-convergence-regression.md): on GPU this test MUST
// FAIL because trainWindowedGPU never writes updated weights back to the
// device, so loss[1] ≈ loss[0]. Post-fix, loss[1] should drop well below
// loss[0] * 0.9 on data this trivial.
//
// Runtime budget: must complete in <= 30s on DGX Spark (GB10). The config
// below (256 samples, 2 channels, 2 epochs, tiny model) is sized for that
// budget with headroom.
func TestGPUTinyTrainingConvergence(t *testing.T) {
	ops := numeric.Float32Ops{}
	gpuEng, err := compute.NewGPUEngine[float32](ops)
	if err != nil {
		t.Skipf("GPU engine unavailable (CPU-only build or no CUDA runtime): %v", err)
	}
	defer gpuEng.Close()

	const (
		nSamples  = 256
		nChannels = 2
		inputLen  = 24
	)

	config := PatchTSTConfig{
		InputLength: inputLen,
		PatchLength: 8,
		Stride:      4,
		DModel:      32,
		NHeads:      2,
		NLayers:     1,
		OutputDim:   1,
	}

	model, err := NewPatchTST(config, gpuEng, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Deterministic synthetic input: per-sample linear ramp with a fixed
	// per-channel phase, plus a label that is a trivial linear function of
	// the input — a model with this capacity should fit it quickly.
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.OutputDim)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, nChannels)
		phase := float64(s) / float64(nSamples)
		var sum float64
		for c := 0; c < nChannels; c++ {
			windows[s][c] = make([]float64, inputLen)
			for i := 0; i < inputLen; i++ {
				// Deterministic: sine wave + channel offset + sample phase.
				v := math.Sin(2*math.Pi*(float64(i)/float64(inputLen)+phase)) + 0.25*float64(c)
				windows[s][c][i] = v
				sum += v
			}
		}
		labels[s] = sum / float64(nChannels*inputLen)
	}

	start := time.Now()
	result, err := model.TrainWindowed(windows, labels, TrainConfig{
		Epochs:    2,
		LR:        1e-3,
		GradClip:  1.0,
		BatchSize: 32,
	})
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}
	if elapsed > 30*time.Second {
		t.Errorf("runtime budget exceeded: %v > 30s", elapsed)
	}

	if len(result.LossHistory) < 2 {
		t.Fatalf("expected at least 2 epochs of loss history, got %d", len(result.LossHistory))
	}

	loss0 := result.LossHistory[0]
	loss1 := result.LossHistory[1]
	t.Logf("GPU tiny train: loss[0]=%.6f loss[1]=%.6f elapsed=%v", loss0, loss1, elapsed)

	if math.IsNaN(loss0) || math.IsInf(loss0, 0) || math.IsNaN(loss1) || math.IsInf(loss1, 0) {
		t.Fatalf("non-finite loss: loss[0]=%v loss[1]=%v", loss0, loss1)
	}
	if loss0 <= 0 {
		t.Fatalf("loss[0] must be positive to assert relative reduction, got %v", loss0)
	}

	// Core assertion: at least 10% loss reduction epoch-over-epoch.
	if !(loss1 < loss0*0.9) {
		t.Fatalf("GPU training did not converge: loss[1]=%.6f not < loss[0]*0.9=%.6f (loss[0]=%.6f)", loss1, loss0*0.9, loss0)
	}
}
