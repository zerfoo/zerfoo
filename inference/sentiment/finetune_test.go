package sentiment

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// mockTrainableModel simulates a trainable model whose logits shift based on
// gradient updates, allowing loss to decrease over training.
type mockTrainableModel struct {
	weights    []float64 // bias per class
	numClasses int
}

func newMockTrainableModel(numClasses int) *mockTrainableModel {
	return &mockTrainableModel{
		weights:    make([]float64, numClasses),
		numClasses: numClasses,
	}
}

func (m *mockTrainableModel) Forward(inputIDs []int) ([]float32, error) {
	// Return weights as logits — the bias gets updated via UpdateParams.
	logits := make([]float32, m.numClasses)
	for i := range logits {
		logits[i] = float32(m.weights[i])
	}
	return logits, nil
}

func (m *mockTrainableModel) NumClasses() int { return m.numClasses }

func (m *mockTrainableModel) UpdateParams(grad []float64, lr float64) error {
	for i := range m.weights {
		m.weights[i] -= lr * grad[i]
	}
	return nil
}

func TestValidateConfig(t *testing.T) {
	valid := TrainingConfig{
		Epochs:       5,
		LearningRate: 0.001,
		BatchSize:    8,
		ValSplit:     0.2,
		LoRARank:     0,
		MaxSeqLen:    128,
		Labels:       []string{"negative", "positive"},
	}
	if err := validateConfig(valid); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*TrainingConfig)
	}{
		{"zero epochs", func(c *TrainingConfig) { c.Epochs = 0 }},
		{"negative lr", func(c *TrainingConfig) { c.LearningRate = -0.01 }},
		{"zero batch", func(c *TrainingConfig) { c.BatchSize = 0 }},
		{"val split 1.0", func(c *TrainingConfig) { c.ValSplit = 1.0 }},
		{"negative val split", func(c *TrainingConfig) { c.ValSplit = -0.1 }},
		{"negative lora rank", func(c *TrainingConfig) { c.LoRARank = -1 }},
		{"zero max seq len", func(c *TrainingConfig) { c.MaxSeqLen = 0 }},
		{"one label", func(c *TrainingConfig) { c.Labels = []string{"only"} }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := valid
			tt.mutate(&cfg)
			if err := validateConfig(cfg); err == nil {
				t.Error("expected validation error")
			}
		})
	}
}

func TestFineTuneLossDecreases(t *testing.T) {
	model := newMockTrainableModel(2)
	tok := &mockTokenizer{}

	// Create training data: all "good" texts are positive, all "bad" are negative.
	data := []TrainingData{
		{Text: "good movie", Label: "positive"},
		{Text: "great film", Label: "positive"},
		{Text: "loved it", Label: "positive"},
		{Text: "wonderful", Label: "positive"},
		{Text: "bad movie", Label: "negative"},
		{Text: "terrible", Label: "negative"},
		{Text: "awful film", Label: "negative"},
		{Text: "hated it", Label: "negative"},
	}

	cfg := TrainingConfig{
		Epochs:       10,
		LearningRate: 0.5,
		BatchSize:    4,
		ValSplit:     0.25,
		LoRARank:     0,
		MaxSeqLen:    128,
		Labels:       []string{"negative", "positive"},
	}

	result, err := FineTune(model, tok, data, cfg)
	if err != nil {
		t.Fatalf("FineTune: %v", err)
	}

	if len(result.EpochMetrics) != 10 {
		t.Fatalf("expected 10 epoch metrics, got %d", len(result.EpochMetrics))
	}

	// Loss should decrease from first to last epoch.
	firstLoss := result.EpochMetrics[0].TrainLoss
	lastLoss := result.EpochMetrics[len(result.EpochMetrics)-1].TrainLoss
	if lastLoss >= firstLoss {
		t.Errorf("loss did not decrease: first=%f, last=%f", firstLoss, lastLoss)
	}

	if result.FinalTrainLoss != lastLoss {
		t.Errorf("FinalTrainLoss=%f != last epoch loss=%f", result.FinalTrainLoss, lastLoss)
	}
}

func TestFineTuneNoData(t *testing.T) {
	model := newMockTrainableModel(2)
	tok := &mockTokenizer{}
	cfg := TrainingConfig{
		Epochs: 1, LearningRate: 0.01, BatchSize: 1, MaxSeqLen: 128,
		Labels: []string{"a", "b"},
	}
	_, err := FineTune(model, tok, nil, cfg)
	if err == nil {
		t.Fatal("expected error for empty data")
	}
}

func TestFineTuneUnknownLabel(t *testing.T) {
	model := newMockTrainableModel(2)
	tok := &mockTokenizer{}
	data := []TrainingData{{Text: "test", Label: "unknown"}}
	cfg := TrainingConfig{
		Epochs: 1, LearningRate: 0.01, BatchSize: 1, MaxSeqLen: 128,
		Labels: []string{"a", "b"},
	}
	_, err := FineTune(model, tok, data, cfg)
	if err == nil {
		t.Fatal("expected error for unknown label")
	}
}

func TestFineTuneZeroValSplit(t *testing.T) {
	model := newMockTrainableModel(2)
	tok := &mockTokenizer{}
	data := []TrainingData{
		{Text: "hello", Label: "positive"},
		{Text: "world", Label: "negative"},
	}
	cfg := TrainingConfig{
		Epochs: 1, LearningRate: 0.1, BatchSize: 2, ValSplit: 0, MaxSeqLen: 128,
		Labels: []string{"negative", "positive"},
	}
	result, err := FineTune(model, tok, data, cfg)
	if err != nil {
		t.Fatalf("FineTune: %v", err)
	}
	// With zero val split, val accuracy should be 0.
	if result.FinalValAcc != 0 {
		t.Errorf("expected 0 val acc with no val split, got %f", result.FinalValAcc)
	}
}

func TestSplitData(t *testing.T) {
	data := make([]TrainingData, 10)
	for i := range data {
		data[i] = TrainingData{Text: "t", Label: "l"}
	}

	train, val := splitData(data, 0.2)
	if len(train)+len(val) != 10 {
		t.Errorf("split lost data: train=%d val=%d", len(train), len(val))
	}
	if len(val) != 2 {
		t.Errorf("expected 2 val samples for 0.2 split of 10, got %d", len(val))
	}

	// Zero split: all training.
	train, val = splitData(data, 0)
	if len(val) != 0 {
		t.Errorf("expected 0 val with 0 split, got %d", len(val))
	}
	if len(train) != 10 {
		t.Errorf("expected 10 train with 0 split, got %d", len(train))
	}
}

func TestArgmax(t *testing.T) {
	tests := []struct {
		vals []float64
		want int
	}{
		{[]float64{1, 3, 2}, 1},
		{[]float64{5}, 0},
		{[]float64{-1, -2, -0.5}, 2},
	}
	for _, tt := range tests {
		got := argmax(tt.vals)
		if got != tt.want {
			t.Errorf("argmax(%v) = %d, want %d", tt.vals, got, tt.want)
		}
	}
}

func TestLoadTrainingDataCSV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")

	content := "text,label\n\"great movie\",positive\n\"bad film\",negative\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	data, err := LoadTrainingData(path)
	if err != nil {
		t.Fatalf("LoadTrainingData: %v", err)
	}
	if len(data) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(data))
	}
	if data[0].Text != "great movie" || data[0].Label != "positive" {
		t.Errorf("data[0] = %+v", data[0])
	}
	if data[1].Text != "bad film" || data[1].Label != "negative" {
		t.Errorf("data[1] = %+v", data[1])
	}
}

func TestLoadTrainingDataCSVReversedColumns(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")

	content := "label,text\npositive,hello world\nnegative,bad day\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	data, err := LoadTrainingData(path)
	if err != nil {
		t.Fatalf("LoadTrainingData: %v", err)
	}
	if len(data) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(data))
	}
	if data[0].Text != "hello world" || data[0].Label != "positive" {
		t.Errorf("data[0] = %+v", data[0])
	}
}

func TestLoadTrainingDataCSVMissingHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")

	content := "foo,bar\nhello,world\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadTrainingData(path)
	if err == nil {
		t.Fatal("expected error for missing text/label columns")
	}
}

func TestLoadTrainingDataCSVEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")

	content := "text,label\n"
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadTrainingData(path)
	if err == nil {
		t.Fatal("expected error for empty CSV")
	}
}

func TestLoadTrainingDataJSONL(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.jsonl")

	content := `{"text": "great movie", "label": "positive"}
{"text": "bad film", "label": "negative"}
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	data, err := LoadTrainingData(path)
	if err != nil {
		t.Fatalf("LoadTrainingData: %v", err)
	}
	if len(data) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(data))
	}
	if data[0].Text != "great movie" || data[0].Label != "positive" {
		t.Errorf("data[0] = %+v", data[0])
	}
}

func TestLoadTrainingDataJSONLBlankLines(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.jsonl")

	content := `{"text": "a", "label": "x"}

{"text": "b", "label": "y"}
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	data, err := LoadTrainingData(path)
	if err != nil {
		t.Fatalf("LoadTrainingData: %v", err)
	}
	if len(data) != 2 {
		t.Fatalf("expected 2, got %d", len(data))
	}
}

func TestLoadTrainingDataJSONLMissingField(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.jsonl")

	content := `{"text": "hello"}
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadTrainingData(path)
	if err == nil {
		t.Fatal("expected error for missing label field")
	}
}

func TestLoadTrainingDataJSONLEmpty(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.jsonl")

	if err := os.WriteFile(path, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadTrainingData(path)
	if err == nil {
		t.Fatal("expected error for empty JSONL")
	}
}

func TestLoadTrainingDataUnsupportedFormat(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "data.txt")
	if err := os.WriteFile(path, []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadTrainingData(path)
	if err == nil {
		t.Fatal("expected error for unsupported format")
	}
}

func TestLoadTrainingDataFileNotFound(t *testing.T) {
	_, err := LoadTrainingData("/nonexistent/path/data.csv")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestFineTuneEpochMetrics(t *testing.T) {
	model := newMockTrainableModel(3)
	tok := &mockTokenizer{}
	data := []TrainingData{
		{Text: "aaa", Label: "a"},
		{Text: "bbb", Label: "b"},
		{Text: "ccc", Label: "c"},
		{Text: "aab", Label: "a"},
	}
	cfg := TrainingConfig{
		Epochs: 3, LearningRate: 0.5, BatchSize: 2, ValSplit: 0.25, MaxSeqLen: 64,
		Labels: []string{"a", "b", "c"},
	}
	result, err := FineTune(model, tok, data, cfg)
	if err != nil {
		t.Fatalf("FineTune: %v", err)
	}
	if len(result.EpochMetrics) != 3 {
		t.Fatalf("expected 3 metrics, got %d", len(result.EpochMetrics))
	}
	for i, m := range result.EpochMetrics {
		if m.Epoch != i+1 {
			t.Errorf("metric[%d].Epoch = %d, want %d", i, m.Epoch, i+1)
		}
		if math.IsNaN(m.TrainLoss) || math.IsInf(m.TrainLoss, 0) {
			t.Errorf("metric[%d].TrainLoss is invalid: %f", i, m.TrainLoss)
		}
	}
}
