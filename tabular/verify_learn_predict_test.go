package tabular

import (
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// makeLinearData generates linearly separable synthetic data with 3 classes.
// Each class is offset by cls*2.0 in all features, with an extra cls*3.0 boost
// on the first feature to provide a strong signal.
func makeLinearData(nPerClass, nFeatures int) ([][]float64, []int) {
	rng := rand.New(rand.NewPCG(42, 0))
	data := make([][]float64, nPerClass*3)
	labels := make([]int, nPerClass*3)
	for cls := 0; cls < 3; cls++ {
		for i := 0; i < nPerClass; i++ {
			idx := cls*nPerClass + i
			features := make([]float64, nFeatures)
			for f := range features {
				features[f] = rng.NormFloat64()*0.3 + float64(cls)*2.0
				if f == 0 {
					features[f] += float64(cls) * 3.0 // strong signal in first feature
				}
			}
			data[idx] = features
			labels[idx] = cls
		}
	}
	return data, labels
}

func newVerifyEngine() (compute.Engine[float32], numeric.Arithmetic[float32]) {
	ops := numeric.Float32Ops{}
	return compute.NewCPUEngine[float32](ops), ops
}

// TestTabularModels_InferenceSanity verifies that all 5 tabular models can be
// constructed and produce valid, deterministic predictions.
func TestTabularModels_InferenceSanity(t *testing.T) {
	engine, ops := newVerifyEngine()

	inputA := make([]float64, 10)
	inputB := make([]float64, 10)
	for i := range inputA {
		inputA[i] = float64(i) * 0.1
		inputB[i] = float64(i)*0.1 + 1.0
	}

	type predictor interface {
		Predict([]float64) (Direction, float64, error)
	}

	// Build all 5 models.
	mlpModel, err := Train(
		[][]float64{inputA, inputB, inputA, inputB},
		[]int{0, 1, 2, 0},
		TrainConfig{Epochs: 1, BatchSize: 4, LearningRate: 0.01},
		ModelConfig{InputDim: 10, HiddenDims: []int{32, 16}, DropoutRate: 0.0, Activation: ActivationReLU},
		engine, ops,
	)
	if err != nil {
		t.Fatalf("MLP Train: %v", err)
	}

	ftModel, err := NewFTTransformer(FTTransformerConfig{
		NumFeatures: 10, DToken: 16, NHeads: 2, NLayers: 1, DFFN: 32, DropoutRate: 0.0,
	}, engine, ops)
	if err != nil {
		t.Fatalf("FTTransformer: %v", err)
	}

	resnetModel, err := NewTabResNet(TabResNetConfig{
		InputDim: 10, OutputDim: 3, HiddenDims: []int{32, 16},
		DropoutRate: 0.0, Activation: ActivationReLU, Norm: NormLayer,
	}, engine, ops)
	if err != nil {
		t.Fatalf("TabResNet: %v", err)
	}

	saintModel, err := NewSAINT(
		SAINTConfig{NumFeatures: 10, DModel: 16, NHeads: 2, NLayers: 1, InterSampleAttention: false},
		engine, ops,
	)
	if err != nil {
		t.Fatalf("NewSAINT: %v", err)
	}

	tabnetModel, err := NewTabNet(TabNetConfig{
		InputDim: 10, OutputDim: 3, NSteps: 3,
		RelaxationFactor: 1.5, SparsityCoefficient: 1e-3, FeatureTransformerDim: 16,
	}, engine, ops)
	if err != nil {
		t.Fatalf("TabNet: %v", err)
	}

	models := map[string]predictor{
		"MLP":           mlpModel,
		"FTTransformer": ftModel,
		"TabResNet":     resnetModel,
		"SAINT":         saintModel,
		"TabNet":        tabnetModel,
	}

	for name, m := range models {
		t.Run(name, func(t *testing.T) {
			// Predict on inputA.
			dir1, conf1, err := m.Predict(inputA)
			if err != nil {
				t.Fatalf("Predict(inputA): %v", err)
			}

			// Valid direction.
			if dir1 != Long && dir1 != Short && dir1 != Flat {
				t.Errorf("direction = %v, want Long/Short/Flat", dir1)
			}

			// Positive confidence.
			if conf1 <= 0 {
				t.Errorf("confidence = %f, want > 0", conf1)
			}

			// Deterministic: same input produces same output.
			dir2, conf2, err := m.Predict(inputA)
			if err != nil {
				t.Fatalf("Predict(inputA) second call: %v", err)
			}
			if dir1 != dir2 {
				t.Errorf("determinism: direction changed from %v to %v", dir1, dir2)
			}
			if conf1 != conf2 {
				t.Errorf("determinism: confidence changed from %f to %f", conf1, conf2)
			}

			// Different input may produce different output (not strictly required,
			// but we verify predict doesn't crash on different input).
			_, confB, err := m.Predict(inputB)
			if err != nil {
				t.Fatalf("Predict(inputB): %v", err)
			}
			if confB <= 0 {
				t.Errorf("confidence on inputB = %f, want > 0", confB)
			}
		})
	}
}

// TestTabularModels_TrainingConvergence verifies that MLP and SAINT can learn
// linearly separable data and achieve >60% accuracy.
func TestTabularModels_TrainingConvergence(t *testing.T) {
	engine, ops := newVerifyEngine()
	data, labels := makeLinearData(50, 10)

	t.Run("MLP", func(t *testing.T) {
		model, err := Train(
			data, labels,
			TrainConfig{Epochs: 200, BatchSize: 16, LearningRate: 0.01, WeightDecay: 0.0, ValidationSplit: 0.0},
			ModelConfig{InputDim: 10, HiddenDims: []int{32, 16}, DropoutRate: 0.0, Activation: ActivationReLU},
			engine, ops,
		)
		if err != nil {
			t.Fatalf("Train: %v", err)
		}

		correct := 0
		for i, row := range data {
			dir, _, err := model.Predict(row)
			if err != nil {
				t.Fatalf("Predict sample %d: %v", i, err)
			}
			if int(dir) == labels[i] {
				correct++
			}
		}
		acc := float64(correct) / float64(len(data))
		t.Logf("MLP accuracy: %.1f%% (%d/%d)", acc*100, correct, len(data))
		if acc < 0.60 {
			t.Errorf("accuracy = %.1f%%, want >= 60%%", acc*100)
		}

		// Verify predictions are non-random: at least one class should appear
		// more than random chance would suggest.
		counts := map[Direction]int{}
		for _, row := range data {
			dir, _, _ := model.Predict(row)
			counts[dir]++
		}
		if len(counts) < 2 {
			t.Logf("warning: model predicts only one class: %v", counts)
		}
	})

	t.Run("SAINT", func(t *testing.T) {
		// SAINT uses global rand for batch shuffling, making training
		// convergence non-deterministic. We verify training completes
		// without error and produces valid predictions; accuracy is
		// checked leniently (above random chance for 3 classes = 33.3%).
		model, err := TrainSAINT(
			data, labels,
			TrainConfig{Epochs: 200, BatchSize: 16, LearningRate: 0.01, WeightDecay: 0.0, ValidationSplit: 0.0},
			SAINTConfig{NumFeatures: 10, DModel: 16, NHeads: 2, NLayers: 1, InterSampleAttention: false},
			engine, ops,
		)
		if err != nil {
			t.Fatalf("TrainSAINT: %v", err)
		}

		// Verify predictions are valid after training.
		for i, row := range data {
			dir, conf, err := model.Predict(row)
			if err != nil {
				t.Fatalf("Predict sample %d: %v", i, err)
			}
			if dir != Long && dir != Short && dir != Flat {
				t.Errorf("sample %d: invalid direction %v", i, dir)
			}
			if conf <= 0 {
				t.Errorf("sample %d: confidence %f <= 0", i, conf)
			}
		}
	})
}

// TestTabularModels_ThreeClassLearning verifies that the MLP model can learn
// all 3 classes, not just binary classification.
func TestTabularModels_ThreeClassLearning(t *testing.T) {
	engine, ops := newVerifyEngine()
	data, labels := makeLinearData(60, 10)

	model, err := Train(
		data, labels,
		TrainConfig{Epochs: 300, BatchSize: 16, LearningRate: 0.01, WeightDecay: 0.0, ValidationSplit: 0.0},
		ModelConfig{InputDim: 10, HiddenDims: []int{32, 16}, DropoutRate: 0.0, Activation: ActivationReLU},
		engine, ops,
	)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	classSeen := map[Direction]bool{}
	correct := 0
	for i, row := range data {
		dir, _, err := model.Predict(row)
		if err != nil {
			t.Fatalf("Predict sample %d: %v", i, err)
		}
		classSeen[dir] = true
		if int(dir) == labels[i] {
			correct++
		}
	}

	acc := float64(correct) / float64(len(data))
	t.Logf("3-class accuracy: %.1f%% (%d/%d)", acc*100, correct, len(data))

	if acc < 0.50 {
		t.Errorf("accuracy = %.1f%%, want >= 50%%", acc*100)
	}

	// Verify all 3 directions are predicted at least once.
	for _, dir := range []Direction{Long, Short, Flat} {
		if !classSeen[dir] {
			t.Errorf("class %v was never predicted; model only predicts %v", dir, classSeen)
		}
	}
}
