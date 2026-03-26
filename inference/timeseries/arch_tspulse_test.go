package timeseries

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func newTestTSPulseModel(t *testing.T, numClasses int) *TSPulseModel {
	t.Helper()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	cfg := &TSPulseConfig{
		ContextLen:  32,
		NumChannels: 2,
		PatchLen:    8,
		DModel:      16,
		NumLayers:   1,
		MaskType:    "hybrid",
		HeadType:    "allhead",
		NumClasses:  numClasses,
	}
	model, err := newTSPulseModel(cfg, engine)
	if err != nil {
		t.Fatalf("newTSPulseModel: %v", err)
	}
	return model
}

func makeTSPulseInput(contextLen, numChannels int, scale float64) [][]float64 {
	input := make([][]float64, contextLen)
	for t := range contextLen {
		input[t] = make([]float64, numChannels)
		for c := range numChannels {
			input[t][c] = float64(t*numChannels+c) * scale
		}
	}
	return input
}

func TestTSPulseAnomalyDetect(t *testing.T) {
	model := newTestTSPulseModel(t, 0)

	// Normal data: smooth linear ramp.
	normal := makeTSPulseInput(32, 2, 0.01)

	// Anomalous data: same as normal but with a spike at position 16.
	anomalous := makeTSPulseInput(32, 2, 0.01)
	for c := range 2 {
		anomalous[16][c] = 100.0 // large spike
	}

	normalScores, err := model.AnomalyDetect(normal)
	if err != nil {
		t.Fatalf("AnomalyDetect (normal): %v", err)
	}
	anomalousScores, err := model.AnomalyDetect(anomalous)
	if err != nil {
		t.Fatalf("AnomalyDetect (anomalous): %v", err)
	}

	// Check output length.
	if len(normalScores) != 32 {
		t.Fatalf("normal scores length: got %d, want 32", len(normalScores))
	}
	if len(anomalousScores) != 32 {
		t.Fatalf("anomalous scores length: got %d, want 32", len(anomalousScores))
	}

	// The anomalous signal at position 16 should produce higher reconstruction
	// error than the normal signal at the same position.
	if anomalousScores[16] <= normalScores[16] {
		t.Errorf("anomaly score at spike: anomalous=%f should be > normal=%f", anomalousScores[16], normalScores[16])
	}

	// All scores should be non-negative (MSE).
	for i, s := range normalScores {
		if s < 0 {
			t.Errorf("normal score[%d] = %f is negative", i, s)
		}
	}
}

func TestTSPulseClassify(t *testing.T) {
	numClasses := 5
	model := newTestTSPulseModel(t, numClasses)
	input := makeTSPulseInput(32, 2, 0.01)

	probs, err := model.Classify(input)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}

	// Output should be [num_classes].
	if len(probs) != numClasses {
		t.Fatalf("output length: got %d, want %d", len(probs), numClasses)
	}

	// Probabilities should sum to ~1.0 (softmax output).
	var sum float64
	for _, p := range probs {
		sum += p
		if p < 0 || p > 1 {
			t.Errorf("probability %f is outside [0, 1]", p)
		}
	}
	if math.Abs(sum-1.0) > 1e-4 {
		t.Errorf("probabilities sum to %f, want ~1.0", sum)
	}
}

func TestTSPulseClassifyNoHead(t *testing.T) {
	model := newTestTSPulseModel(t, 0) // no classification head
	input := makeTSPulseInput(32, 2, 0.01)

	_, err := model.Classify(input)
	if err == nil {
		t.Fatal("expected error when classifying without classification head")
	}
}

func TestTSPulseImpute(t *testing.T) {
	model := newTestTSPulseModel(t, 0)
	input := makeTSPulseInput(32, 2, 0.01)

	// Mask positions 10-15 as missing.
	mask := make([]bool, 32)
	for i := 10; i < 16; i++ {
		mask[i] = true
	}

	result, err := model.Impute(input, mask)
	if err != nil {
		t.Fatalf("Impute: %v", err)
	}

	// Output should have same shape.
	if len(result) != 32 {
		t.Fatalf("output length: got %d, want 32", len(result))
	}
	for idx := range 32 {
		if len(result[idx]) != 2 {
			t.Fatalf("output channels at idx=%d: got %d, want 2", idx, len(result[idx]))
		}
	}

	// Non-masked positions should be unchanged.
	for i := 0; i < 10; i++ {
		for c := range 2 {
			if result[i][c] != input[i][c] {
				t.Errorf("non-masked position [%d][%d]: got %f, want %f", i, c, result[i][c], input[i][c])
			}
		}
	}

	// Masked positions should have been filled (may differ from original).
	for i := 10; i < 16; i++ {
		// Just check that values are finite.
		for c := range 2 {
			if math.IsNaN(result[i][c]) || math.IsInf(result[i][c], 0) {
				t.Errorf("masked position [%d][%d] is not finite: %f", i, c, result[i][c])
			}
		}
	}
}

func TestTSPulseImputeMaskLengthMismatch(t *testing.T) {
	model := newTestTSPulseModel(t, 0)
	input := makeTSPulseInput(32, 2, 0.01)
	mask := make([]bool, 10) // wrong length

	_, err := model.Impute(input, mask)
	if err == nil {
		t.Fatal("expected error for mask length mismatch")
	}
}

func TestTSPulseEmbed(t *testing.T) {
	model := newTestTSPulseModel(t, 0)
	input := makeTSPulseInput(32, 2, 0.01)

	emb, err := model.Embed(input)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	// Output should be [d_model] dimensional.
	if len(emb) != 16 {
		t.Fatalf("embedding length: got %d, want 16", len(emb))
	}

	// All values should be finite.
	for i, v := range emb {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("embedding[%d] is not finite: %f", i, v)
		}
	}

	// Embedding should not be all zeros (encoder produces non-trivial output).
	allZero := true
	for _, v := range emb {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("embedding is all zeros")
	}
}

func TestTSPulseEmbedSimilarity(t *testing.T) {
	model := newTestTSPulseModel(t, 0)

	// Two identical inputs should have cosine similarity of 1.
	input1 := makeTSPulseInput(32, 2, 0.01)
	input2 := makeTSPulseInput(32, 2, 0.01)

	emb1, err := model.Embed(input1)
	if err != nil {
		t.Fatalf("Embed input1: %v", err)
	}
	emb2, err := model.Embed(input2)
	if err != nil {
		t.Fatalf("Embed input2: %v", err)
	}

	sim := cosineSimilarity(emb1, emb2)
	if math.Abs(sim-1.0) > 1e-5 {
		t.Errorf("identical inputs cosine similarity: got %f, want 1.0", sim)
	}
}

func TestTSPulseConfigValidation(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	tests := []struct {
		name string
		cfg  *TSPulseConfig
	}{
		{"zero ContextLen", &TSPulseConfig{ContextLen: 0, NumChannels: 2, PatchLen: 8, DModel: 16, NumLayers: 1, MaskType: "hybrid", HeadType: "allhead"}},
		{"zero NumChannels", &TSPulseConfig{ContextLen: 32, NumChannels: 0, PatchLen: 8, DModel: 16, NumLayers: 1, MaskType: "hybrid", HeadType: "allhead"}},
		{"zero PatchLen", &TSPulseConfig{ContextLen: 32, NumChannels: 2, PatchLen: 0, DModel: 16, NumLayers: 1, MaskType: "hybrid", HeadType: "allhead"}},
		{"zero DModel", &TSPulseConfig{ContextLen: 32, NumChannels: 2, PatchLen: 8, DModel: 0, NumLayers: 1, MaskType: "hybrid", HeadType: "allhead"}},
		{"zero NumLayers", &TSPulseConfig{ContextLen: 32, NumChannels: 2, PatchLen: 8, DModel: 16, NumLayers: 0, MaskType: "hybrid", HeadType: "allhead"}},
		{"invalid MaskType", &TSPulseConfig{ContextLen: 32, NumChannels: 2, PatchLen: 8, DModel: 16, NumLayers: 1, MaskType: "unknown", HeadType: "allhead"}},
		{"invalid HeadType", &TSPulseConfig{ContextLen: 32, NumChannels: 2, PatchLen: 8, DModel: 16, NumLayers: 1, MaskType: "hybrid", HeadType: "unknown"}},
		{"negative NumClasses", &TSPulseConfig{ContextLen: 32, NumChannels: 2, PatchLen: 8, DModel: 16, NumLayers: 1, MaskType: "hybrid", HeadType: "allhead", NumClasses: -1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newTSPulseModel(tt.cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestTSPulseInputValidation(t *testing.T) {
	model := newTestTSPulseModel(t, 0)

	// Wrong context length.
	shortInput := makeTSPulseInput(16, 2, 0.01)
	_, err := model.AnomalyDetect(shortInput)
	if err == nil {
		t.Error("expected error for wrong context length")
	}

	// Wrong number of channels.
	wrongChannels := makeTSPulseInput(32, 3, 0.01)
	_, err = model.Embed(wrongChannels)
	if err == nil {
		t.Error("expected error for wrong number of channels")
	}
}

func TestTSPulseCosineSimilarity(t *testing.T) {
	a := []float64{1, 0, 0}
	b := []float64{1, 0, 0}
	if sim := cosineSimilarity(a, b); math.Abs(sim-1.0) > 1e-10 {
		t.Errorf("parallel vectors: got %f, want 1.0", sim)
	}

	c := []float64{0, 1, 0}
	if sim := cosineSimilarity(a, c); math.Abs(sim) > 1e-10 {
		t.Errorf("orthogonal vectors: got %f, want 0.0", sim)
	}

	d := []float64{-1, 0, 0}
	if sim := cosineSimilarity(a, d); math.Abs(sim+1.0) > 1e-10 {
		t.Errorf("anti-parallel vectors: got %f, want -1.0", sim)
	}

	// Empty vectors.
	if sim := cosineSimilarity([]float64{}, []float64{}); sim != 0 {
		t.Errorf("empty vectors: got %f, want 0.0", sim)
	}

	// Zero vector.
	if sim := cosineSimilarity([]float64{0, 0}, []float64{1, 1}); sim != 0 {
		t.Errorf("zero vector: got %f, want 0.0", sim)
	}
}
