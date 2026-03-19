package tabular

import (
	"testing"
)

// generateSourceData creates synthetic data for a source where feature[dim]
// determines the label. This creates a linearly separable problem that
// different sources share structurally but with different offsets.
func generateSourceData(n, inputDim, labelDim int, offset float64) ([][]float64, []int) {
	data := make([][]float64, n)
	labels := make([]int, n)
	for i := 0; i < n; i++ {
		row := make([]float64, inputDim)
		for j := 0; j < inputDim; j++ {
			row[j] = float64(i*inputDim+j)*0.01 + offset
		}
		// Label based on whether the key feature exceeds a threshold.
		val := row[labelDim]
		switch {
		case val < offset+0.3:
			labels[i] = 0 // Long
		case val < offset+0.6:
			labels[i] = 1 // Short
		default:
			labels[i] = 2 // Flat
		}
		data[i] = row
	}
	return data, labels
}

func TestPreTrain_Convergence(t *testing.T) {
	engine, ops := newTestEngine()

	inputDim := 4
	samplesPerSource := 30

	// Create 3 sources with different offsets but same structure.
	allData := make([][][]float64, 3)
	allLabels := make([][]int, 3)
	for s := 0; s < 3; s++ {
		allData[s], allLabels[s] = generateSourceData(samplesPerSource, inputDim, 0, float64(s)*0.1)
	}

	config := PreTrainConfig{
		Epochs:       200,
		BatchSize:    32,
		LearningRate: 0.01,
		HiddenDims:   []int{16, 8},
		DropoutRate:  0.0,
		Activation:   ActivationReLU,
	}

	bm, err := PreTrain(allData, allLabels, config, engine, ops)
	if err != nil {
		t.Fatalf("PreTrain: %v", err)
	}
	if bm == nil {
		t.Fatal("expected non-nil BaseModel")
	}
	if bm.Model == nil {
		t.Fatal("expected non-nil BaseModel.Model")
	}

	// Verify convergence: the pre-trained model should predict training data
	// with reasonable accuracy.
	correct := 0
	total := 0
	for s := 0; s < 3; s++ {
		for i, row := range allData[s] {
			dir, _, err := bm.Model.Predict(row)
			if err != nil {
				t.Fatalf("Predict source %d sample %d: %v", s, i, err)
			}
			if int(dir) == allLabels[s][i] {
				correct++
			}
			total++
		}
	}
	acc := float64(correct) / float64(total)
	if acc < 0.5 {
		t.Errorf("pre-trained model accuracy %.2f, want >= 0.5", acc)
	}
}

func TestPreTrain_TransferBenefit(t *testing.T) {
	engine, ops := newTestEngine()

	inputDim := 4
	samplesPerSource := 40

	// Create multiple sources for pre-training.
	allData := make([][][]float64, 3)
	allLabels := make([][]int, 3)
	for s := 0; s < 3; s++ {
		allData[s], allLabels[s] = generateSourceData(samplesPerSource, inputDim, 0, float64(s)*0.05)
	}

	preTrainConfig := PreTrainConfig{
		Epochs:       300,
		BatchSize:    32,
		LearningRate: 0.01,
		HiddenDims:   []int{16, 8},
		DropoutRate:  0.0,
		Activation:   ActivationReLU,
	}

	bm, err := PreTrain(allData, allLabels, preTrainConfig, engine, ops)
	if err != nil {
		t.Fatalf("PreTrain: %v", err)
	}

	// Create a target dataset (new source with similar structure).
	targetData, targetLabels := generateSourceData(samplesPerSource, inputDim, 0, 0.15)

	ftConfig := TrainConfig{
		Epochs:       50,
		BatchSize:    16,
		LearningRate: 0.005,
	}

	// Fine-tune from pre-trained base.
	ftModel, err := bm.FineTune(targetData, targetLabels, ftConfig, engine, ops)
	if err != nil {
		t.Fatalf("FineTune: %v", err)
	}

	// Train from scratch with the same budget.
	mc := ModelConfig{
		HiddenDims:  []int{16, 8},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}
	scratchConfig := TrainConfig{
		Epochs:       50,
		BatchSize:    16,
		LearningRate: 0.005,
	}
	scratchModel, err := Train(targetData, targetLabels, scratchConfig, mc, engine, ops)
	if err != nil {
		t.Fatalf("Train from scratch: %v", err)
	}

	// Evaluate both on the target data.
	ftCorrect := 0
	scratchCorrect := 0
	for i, row := range targetData {
		ftDir, _, err := ftModel.Predict(row)
		if err != nil {
			t.Fatalf("FineTuned Predict %d: %v", i, err)
		}
		if int(ftDir) == targetLabels[i] {
			ftCorrect++
		}

		scrDir, _, err := scratchModel.Predict(row)
		if err != nil {
			t.Fatalf("Scratch Predict %d: %v", i, err)
		}
		if int(scrDir) == targetLabels[i] {
			scratchCorrect++
		}
	}

	ftAcc := float64(ftCorrect) / float64(len(targetData))
	scratchAcc := float64(scratchCorrect) / float64(len(targetData))

	t.Logf("fine-tuned accuracy: %.2f, scratch accuracy: %.2f", ftAcc, scratchAcc)

	// The fine-tuned model should achieve at least comparable accuracy.
	// With pre-training on related data it should generally be equal or better.
	// We use a tolerance to account for randomness in weight initialisation.
	if ftAcc < scratchAcc-0.15 {
		t.Errorf("fine-tuned accuracy %.2f significantly worse than scratch %.2f (tolerance 0.15)", ftAcc, scratchAcc)
	}
}

func TestPreTrain_ErrorCases(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name      string
		allData   [][][]float64
		allLabels [][]int
		config    PreTrainConfig
	}{
		{
			name:      "no sources",
			allData:   nil,
			allLabels: nil,
			config:    PreTrainConfig{Epochs: 1, BatchSize: 1, HiddenDims: []int{4}},
		},
		{
			name:      "mismatched source counts",
			allData:   [][][]float64{{{1, 2}}},
			allLabels: [][]int{{0}, {1}},
			config:    PreTrainConfig{Epochs: 1, BatchSize: 1, HiddenDims: []int{4}},
		},
		{
			name:      "empty source",
			allData:   [][][]float64{{}},
			allLabels: [][]int{{}},
			config:    PreTrainConfig{Epochs: 1, BatchSize: 1, HiddenDims: []int{4}},
		},
		{
			name:      "source data/label mismatch",
			allData:   [][][]float64{{{1, 2}, {3, 4}}},
			allLabels: [][]int{{0}},
			config:    PreTrainConfig{Epochs: 1, BatchSize: 1, HiddenDims: []int{4}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PreTrain(tt.allData, tt.allLabels, tt.config, engine, ops)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func TestFineTune_DimMismatch(t *testing.T) {
	engine, ops := newTestEngine()

	// Pre-train with inputDim=4.
	allData := [][][]float64{
		{{1, 2, 3, 4}, {5, 6, 7, 8}},
	}
	allLabels := [][]int{{0, 1}}

	config := PreTrainConfig{
		Epochs:       10,
		BatchSize:    2,
		LearningRate: 0.01,
		HiddenDims:   []int{4},
		Activation:   ActivationReLU,
	}

	bm, err := PreTrain(allData, allLabels, config, engine, ops)
	if err != nil {
		t.Fatalf("PreTrain: %v", err)
	}

	// Try fine-tuning with inputDim=2 — should fail.
	_, err = bm.FineTune([][]float64{{1, 2}}, []int{0}, TrainConfig{Epochs: 1, BatchSize: 1}, engine, ops)
	if err == nil {
		t.Fatal("expected dimension mismatch error, got nil")
	}
}

