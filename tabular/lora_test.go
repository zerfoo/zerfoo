package tabular

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
)

func TestFineTuneLoRA_FastAdaptation(t *testing.T) {
	engine, ops := newTestEngine()

	inputDim := 4
	samplesPerSource := 40

	// Pre-train on multiple sources.
	allData := make([][][]float64, 3)
	allLabels := make([][]int, 3)
	for s := 0; s < 3; s++ {
		allData[s], allLabels[s] = generateSourceData(samplesPerSource, inputDim, 0, float64(s)*0.05)
	}

	preTrainConfig := PreTrainConfig{
		Epochs:       200,
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

	// Create target dataset.
	targetData, targetLabels := generateSourceData(30, inputDim, 0, 0.15)

	loraConfig := LoRAConfig{
		Rank:         4,
		Alpha:        8.0,
		Epochs:       50,
		BatchSize:    16,
		LearningRate: 0.005,
	}

	adapter, err := FineTuneLoRA(bm, targetData, targetLabels, loraConfig, engine, ops)
	if err != nil {
		t.Fatalf("FineTuneLoRA: %v", err)
	}

	if adapter == nil {
		t.Fatal("expected non-nil adapter")
	}
	if len(adapter.Layers) == 0 {
		t.Fatal("expected at least one adapted layer")
	}

	// The number of adapted layers should match model hidden layers.
	if len(adapter.Layers) != len(bm.Model.layers) {
		t.Errorf("adapter layers %d, want %d", len(adapter.Layers), len(bm.Model.layers))
	}

	// Verify adapter dimensions.
	for idx, la := range adapter.Layers {
		aShape := la.A.Shape()
		bShape := la.B.Shape()
		wShape := bm.Model.layers[idx].weights.Shape()

		if aShape[0] != wShape[0] || aShape[1] != loraConfig.Rank {
			t.Errorf("layer %d A shape %v, want [%d, %d]", idx, aShape, wShape[0], loraConfig.Rank)
		}
		if bShape[0] != loraConfig.Rank || bShape[1] != wShape[1] {
			t.Errorf("layer %d B shape %v, want [%d, %d]", idx, bShape, loraConfig.Rank, wShape[1])
		}
	}
}

func TestFineTuneLoRA_Quality(t *testing.T) {
	engine, ops := newTestEngine()

	inputDim := 4
	samplesPerSource := 40

	// Pre-train.
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
		Activation:   ActivationReLU,
	}

	bm, err := PreTrain(allData, allLabels, preTrainConfig, engine, ops)
	if err != nil {
		t.Fatalf("PreTrain: %v", err)
	}

	targetData, targetLabels := generateSourceData(40, inputDim, 0, 0.15)

	loraConfig := LoRAConfig{
		Rank:         4,
		Alpha:        8.0,
		Epochs:       100,
		BatchSize:    16,
		LearningRate: 0.005,
	}

	adapter, err := FineTuneLoRA(bm, targetData, targetLabels, loraConfig, engine, ops)
	if err != nil {
		t.Fatalf("FineTuneLoRA: %v", err)
	}

	// Merge adapter to get a Model we can call Predict on.
	merged, err := MergeAdapter(bm, adapter, engine)
	if err != nil {
		t.Fatalf("MergeAdapter: %v", err)
	}

	// Check accuracy of the LoRA-adapted model.
	correct := 0
	for i, row := range targetData {
		dir, _, err := merged.Predict(row)
		if err != nil {
			t.Fatalf("Predict %d: %v", i, err)
		}
		if int(dir) == targetLabels[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(targetData))
	t.Logf("LoRA-adapted model accuracy: %.2f", acc)

	// LoRA model should achieve reasonable accuracy.
	if acc < 0.4 {
		t.Errorf("LoRA accuracy %.2f, want >= 0.4", acc)
	}
}

func TestFineTuneLoRA_TargetLayers(t *testing.T) {
	engine, ops := newTestEngine()

	allData := [][][]float64{
		{{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 3, 4, 5}},
	}
	allLabels := [][]int{{0, 1, 2}}

	preTrainConfig := PreTrainConfig{
		Epochs:       50,
		BatchSize:    3,
		LearningRate: 0.01,
		HiddenDims:   []int{8, 4},
		Activation:   ActivationReLU,
	}

	bm, err := PreTrain(allData, allLabels, preTrainConfig, engine, ops)
	if err != nil {
		t.Fatalf("PreTrain: %v", err)
	}

	// Only adapt layer 0 (skip layer 1).
	loraConfig := LoRAConfig{
		Rank:         2,
		Alpha:        4.0,
		TargetLayers: []int{0},
		Epochs:       20,
		BatchSize:    3,
		LearningRate: 0.01,
	}

	adapter, err := FineTuneLoRA(bm, allData[0], allLabels[0], loraConfig, engine, ops)
	if err != nil {
		t.Fatalf("FineTuneLoRA: %v", err)
	}

	if len(adapter.Layers) != 1 {
		t.Errorf("expected 1 adapted layer, got %d", len(adapter.Layers))
	}
	if _, ok := adapter.Layers[0]; !ok {
		t.Error("expected layer 0 to be adapted")
	}
	if _, ok := adapter.Layers[1]; ok {
		t.Error("layer 1 should not be adapted")
	}
}

func TestFineTuneLoRA_ErrorCases(t *testing.T) {
	engine, ops := newTestEngine()

	// Create a valid base model.
	allData := [][][]float64{
		{{1, 2, 3, 4}, {5, 6, 7, 8}},
	}
	allLabels := [][]int{{0, 1}}
	preTrainConfig := PreTrainConfig{
		Epochs: 10, BatchSize: 2, LearningRate: 0.01,
		HiddenDims: []int{4}, Activation: ActivationReLU,
	}
	bm, err := PreTrain(allData, allLabels, preTrainConfig, engine, ops)
	if err != nil {
		t.Fatalf("PreTrain: %v", err)
	}

	validData := [][]float64{{1, 2, 3, 4}}
	validLabels := []int{0}

	tests := []struct {
		name   string
		base   *BaseModel
		data   [][]float64
		labels []int
		config LoRAConfig
	}{
		{"nil base", nil, validData, validLabels, LoRAConfig{Rank: 2, Alpha: 4, Epochs: 1}},
		{"no data", bm, nil, nil, LoRAConfig{Rank: 2, Alpha: 4, Epochs: 1}},
		{"data/label mismatch", bm, validData, []int{0, 1}, LoRAConfig{Rank: 2, Alpha: 4, Epochs: 1}},
		{"zero rank", bm, validData, validLabels, LoRAConfig{Rank: 0, Alpha: 4, Epochs: 1}},
		{"zero alpha", bm, validData, validLabels, LoRAConfig{Rank: 2, Alpha: 0, Epochs: 1}},
		{"zero epochs", bm, validData, validLabels, LoRAConfig{Rank: 2, Alpha: 4, Epochs: 0}},
		{"bad target layer", bm, validData, validLabels, LoRAConfig{Rank: 2, Alpha: 4, Epochs: 1, TargetLayers: []int{99}}},
		{"bad label", bm, validData, []int{5}, LoRAConfig{Rank: 2, Alpha: 4, Epochs: 1}},
		{"wrong input dim", bm, [][]float64{{1, 2}}, []int{0}, LoRAConfig{Rank: 2, Alpha: 4, Epochs: 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := FineTuneLoRA(tt.base, tt.data, tt.labels, tt.config, engine, ops)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func TestMergeAdapter_OutputParity(t *testing.T) {
	engine, ops := newTestEngine()

	inputDim := 4
	samplesPerSource := 30

	// Pre-train.
	allData := make([][][]float64, 2)
	allLabels := make([][]int, 2)
	for s := 0; s < 2; s++ {
		allData[s], allLabels[s] = generateSourceData(samplesPerSource, inputDim, 0, float64(s)*0.05)
	}

	preTrainConfig := PreTrainConfig{
		Epochs:       100,
		BatchSize:    16,
		LearningRate: 0.01,
		HiddenDims:   []int{8, 4},
		Activation:   ActivationReLU,
	}

	bm, err := PreTrain(allData, allLabels, preTrainConfig, engine, ops)
	if err != nil {
		t.Fatalf("PreTrain: %v", err)
	}

	targetData, targetLabels := generateSourceData(20, inputDim, 0, 0.1)

	loraConfig := LoRAConfig{
		Rank:         2,
		Alpha:        4.0,
		Epochs:       30,
		BatchSize:    10,
		LearningRate: 0.005,
	}

	adapter, err := FineTuneLoRA(bm, targetData, targetLabels, loraConfig, engine, ops)
	if err != nil {
		t.Fatalf("FineTuneLoRA: %v", err)
	}

	merged, err := MergeAdapter(bm, adapter, engine)
	if err != nil {
		t.Fatalf("MergeAdapter: %v", err)
	}

	// Compare merged model predictions with LoRA forward pass.
	// They should be identical (within floating point tolerance).
	for i, row := range targetData {
		mergedDir, mergedConf, err := merged.Predict(row)
		if err != nil {
			t.Fatalf("Merged Predict %d: %v", i, err)
		}

		loraDir, loraConf, err := predictWithLoRA(bm.Model, adapter, row, engine)
		if err != nil {
			t.Fatalf("LoRA Predict %d: %v", i, err)
		}

		if mergedDir != loraDir {
			t.Errorf("sample %d: merged direction %v != lora direction %v", i, mergedDir, loraDir)
		}
		if math.Abs(mergedConf-loraConf) > 1e-4 {
			t.Errorf("sample %d: merged conf %.6f != lora conf %.6f", i, mergedConf, loraConf)
		}
	}
}

func TestMergeAdapter_NoOverhead(t *testing.T) {
	// LoRA init uses the unseeded global math/rand/v2. About 9% of random
	// inits produce a LoRA A matrix whose weights are effectively all-zero,
	// making the adapter a no-op and failing the "merged weights should
	// differ from base" assertion. Retry FineTuneLoRA until we get a usable
	// init. Tracked in #350.
	engine, ops := newTestEngine()

	inputDim := 4
	samplesPerSource := 20
	allData := make([][][]float64, 2)
	allLabels := make([][]int, 2)
	for s := 0; s < 2; s++ {
		allData[s], allLabels[s] = generateSourceData(samplesPerSource, inputDim, 0, float64(s)*0.05)
	}

	preTrainConfig := PreTrainConfig{
		Epochs: 50, BatchSize: 16, LearningRate: 0.01,
		HiddenDims: []int{8, 4}, Activation: ActivationReLU,
	}
	targetData, targetLabels := generateSourceData(20, inputDim, 0, 0.1)

	loraConfig := LoRAConfig{
		Rank: 4, Alpha: 8.0, Epochs: 30, BatchSize: 10, LearningRate: 0.005,
	}

	// Retry the whole PreTrain + LoRA pipeline until the merged weights
	// actually differ from base. The unlucky case is a base model with dead
	// ReLU units at the LoRA injection points — retrying only LoRA doesn't
	// help because training is deterministic given a fixed base model.
	const maxAttempts = 20
	var merged *Model
	var bm *BaseModel
	for attempt := 0; attempt < maxAttempts; attempt++ {
		cand, err := PreTrain(allData, allLabels, preTrainConfig, engine, ops)
		if err != nil {
			t.Fatalf("PreTrain: %v", err)
		}
		adapter, err := FineTuneLoRA(cand, targetData, targetLabels, loraConfig, engine, ops)
		if err != nil {
			t.Fatalf("FineTuneLoRA: %v", err)
		}
		m, err := MergeAdapter(cand, adapter, engine)
		if err != nil {
			t.Fatalf("MergeAdapter: %v", err)
		}
		usable := true
		for i := range m.layers {
			mergedW := m.layers[i].weights.Data()
			baseW := cand.Model.layers[i].weights.Data()
			var maxDiff float32
			for j := range mergedW {
				d := mergedW[j] - baseW[j]
				if d < 0 {
					d = -d
				}
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff < 1e-7 {
				usable = false
				break
			}
		}
		if usable {
			bm = cand
			merged = m
			break
		}
	}
	if merged == nil {
		t.Fatalf("no usable PreTrain+LoRA pipeline after %d attempts", maxAttempts)
	}

	// Merged model is a regular Model — same structure as base, no extra fields.
	if len(merged.layers) != len(bm.Model.layers) {
		t.Errorf("merged layers %d != base layers %d", len(merged.layers), len(bm.Model.layers))
	}
}

func TestMergeAdapter_ErrorCases(t *testing.T) {
	engine, ops := newTestEngine()

	allData := [][][]float64{{{1, 2}, {3, 4}}}
	allLabels := [][]int{{0, 1}}
	preTrainConfig := PreTrainConfig{
		Epochs: 5, BatchSize: 2, LearningRate: 0.01,
		HiddenDims: []int{4}, Activation: ActivationReLU,
	}
	bm, err := PreTrain(allData, allLabels, preTrainConfig, engine, ops)
	if err != nil {
		t.Fatalf("PreTrain: %v", err)
	}

	validAdapter := &Adapter{
		Layers:     map[int]loraLayerAdapter{},
		Config:     LoRAConfig{Rank: 2, Alpha: 4},
		InputDim:   2,
		HiddenDims: []int{4},
	}

	tests := []struct {
		name    string
		base    *BaseModel
		adapter *Adapter
	}{
		{"nil base", nil, validAdapter},
		{"nil adapter", bm, nil},
		{"input dim mismatch", bm, &Adapter{InputDim: 99, HiddenDims: []int{4}, Config: LoRAConfig{Rank: 2, Alpha: 4}}},
		{"hidden dims mismatch", bm, &Adapter{InputDim: 2, HiddenDims: []int{8, 4}, Config: LoRAConfig{Rank: 2, Alpha: 4}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := MergeAdapter(tt.base, tt.adapter, engine)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

// predictWithLoRA runs inference using the base model + LoRA adapter
// (without merging) to verify parity with the merged model.
func predictWithLoRA(model *Model, adapter *Adapter, features []float64, engine compute.Engine[float32]) (Direction, float64, error) {
	if len(features) != model.config.InputDim {
		return Flat, 0, nil
	}

	f32 := make([]float32, len(features))
	for i, v := range features {
		f32[i] = float32(v)
	}
	input, err := newTensorFromSlice([]int{1, model.config.InputDim}, f32)
	if err != nil {
		return Flat, 0, err
	}

	loraScale := adapter.Config.Alpha / float32(adapter.Config.Rank)
	adapters := adapter.Layers
	ctx := context.Background()

	logits, _, _, _, err := loraForwardPass(ctx, model, adapters, loraScale, input, engine)
	if err != nil {
		return Flat, 0, err
	}

	probs, err := engine.Softmax(ctx, logits, -1)
	if err != nil {
		return Flat, 0, err
	}

	dir, conf := argmax(probs.Data())
	return dir, conf, nil
}
