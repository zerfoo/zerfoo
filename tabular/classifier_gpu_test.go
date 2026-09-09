package tabular

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// This opt-in gate is run through Spark. Once explicitly requested, unavailable
// GPU dispatch is a failure rather than a skip or CPU fallback verdict.
func TestClassifierTrainingGPU(t *testing.T) {
	if os.Getenv("ZERFOO_R_MODEL_GPU") != "1" {
		t.Skip("requires explicit Spark GPU validation")
	}
	engine, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	})
	dataset := classifierFixture(t)
	manifest := dataset.Manifest()
	for _, hidden := range [][]int{nil, {16}} {
		for _, seed := range []uint64{17, 42, 91} {
			started := time.Now()
			config := ClassifierConfig{InputDim: 4, ClassCount: 3, Labels: manifest.Labels, HiddenDims: hidden, Seed: seed}
			probe, err := NewClassifier(config, engine)
			if err != nil {
				t.Fatal(err)
			}
			input, err := probe.inputTensor(context.Background(), [][]float64{{0, 0, 0, 0}, {1, 1, 1, 1}})
			if err != nil {
				t.Fatal(err)
			}
			logits, err := probe.forward(context.Background(), input)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := logits.GetStorage().(*tensor.GPUStorage[float32]); !ok {
				t.Fatalf("GPU forward returned %T, refusing CPU-only evidence", logits.GetStorage())
			}
			result, err := FitClassifier(context.Background(), dataset, config, FitOptions{Epochs: 200, BatchSize: 15, LearningRate: 0.01, WeightDecay: 0.0001, Seed: seed, MaxSteps: 1200}, engine, nil)
			if err != nil {
				t.Fatal(err)
			}
			if err := engine.Sync(); err != nil {
				t.Fatal(err)
			}
			rows, labels, err := dataset.Partition("test")
			if err != nil {
				t.Fatal(err)
			}
			report, err := result.Model.Evaluate(context.Background(), rows, labels)
			if err != nil {
				t.Fatal(err)
			}
			if report.Accuracy < 26.0/30 || report.MacroF1 < 0.85 || report.CrossEntropy > 0.6 {
				t.Fatalf("GPU frozen convergence failure: %+v", report)
			}
			output := filepath.Join(t.TempDir(), "bundle")
			id, err := SaveClassifierBundle(context.Background(), output, result.Model, dataset, "")
			if err != nil {
				t.Fatal(err)
			}
			cpu, err := LoadClassifierBundle(context.Background(), output, id, compute.NewCPUEngine(numeric.Float32Ops{}))
			if err != nil {
				t.Fatal(err)
			}
			cpuReport, err := cpu.Model.Evaluate(context.Background(), rows, labels)
			if err != nil {
				t.Fatal(err)
			}
			if cpuReport.Accuracy != report.Accuracy {
				t.Fatalf("GPU artifact CPU deployment changed labels: GPU=%g CPU=%g", report.Accuracy, cpuReport.Accuracy)
			}
			t.Logf("device=%d managed=%t hidden=%v seed=%d steps=%d accuracy=%g macro_f1=%g CE=%g elapsed=%s artifact=%s", engine.DeviceID(), engine.IsManagedMemory(), hidden, seed, result.Steps, report.Accuracy, report.MacroF1, report.CrossEntropy, time.Since(started), id)
		}
	}
}
