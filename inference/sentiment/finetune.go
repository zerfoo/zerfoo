package sentiment

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// TrainingConfig holds fine-tuning configuration.
type TrainingConfig struct {
	Epochs       int
	LearningRate float64
	BatchSize    int
	ValSplit     float64 // fraction for validation (0.0-1.0)
	LoRARank     int     // 0 = full fine-tuning, >0 = LoRA
	MaxSeqLen    int
	Labels       []string
}

// TrainingData represents a labeled text sample.
type TrainingData struct {
	Text  string `json:"text"`
	Label string `json:"label"`
}

// TrainingResult holds metrics from fine-tuning.
type TrainingResult struct {
	FinalTrainLoss float64
	FinalValAcc    float64
	EpochMetrics   []EpochMetric
}

// EpochMetric holds metrics for a single training epoch.
type EpochMetric struct {
	Epoch     int
	TrainLoss float64
	ValAcc    float64
}

// TrainableModel abstracts a model that supports forward pass and parameter
// updates during fine-tuning. This allows mock implementations for testing.
type TrainableModel interface {
	// Forward runs the model on tokenized input and returns logits.
	Forward(inputIDs []int) ([]float32, error)

	// NumClasses returns the number of output classes.
	NumClasses() int

	// UpdateParams applies gradient-based parameter updates.
	// grad is the loss gradient (softmax - one_hot) averaged over the batch.
	// lr is the learning rate.
	UpdateParams(grad []float64, lr float64) error
}

// validateConfig checks the training configuration for invalid values.
func validateConfig(cfg TrainingConfig) error {
	if cfg.Epochs <= 0 {
		return fmt.Errorf("sentiment finetune: epochs must be positive, got %d", cfg.Epochs)
	}
	if cfg.LearningRate <= 0 {
		return fmt.Errorf("sentiment finetune: learning rate must be positive, got %f", cfg.LearningRate)
	}
	if cfg.BatchSize <= 0 {
		return fmt.Errorf("sentiment finetune: batch size must be positive, got %d", cfg.BatchSize)
	}
	if cfg.ValSplit < 0 || cfg.ValSplit >= 1.0 {
		return fmt.Errorf("sentiment finetune: val split must be in [0, 1), got %f", cfg.ValSplit)
	}
	if cfg.LoRARank < 0 {
		return fmt.Errorf("sentiment finetune: LoRA rank must be non-negative, got %d", cfg.LoRARank)
	}
	if cfg.MaxSeqLen <= 0 {
		return fmt.Errorf("sentiment finetune: max seq len must be positive, got %d", cfg.MaxSeqLen)
	}
	if len(cfg.Labels) < 2 {
		return fmt.Errorf("sentiment finetune: at least 2 labels required, got %d", len(cfg.Labels))
	}
	return nil
}

// FineTune trains a sentiment model on labeled data using the provided
// TrainableModel, Tokenizer, and configuration. It returns training metrics.
func FineTune(model TrainableModel, tokenizer Tokenizer, data []TrainingData, cfg TrainingConfig) (*TrainingResult, error) {
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("sentiment finetune: no training data provided")
	}

	// Build label-to-index map.
	labelIdx := make(map[string]int, len(cfg.Labels))
	for i, l := range cfg.Labels {
		labelIdx[l] = i
	}

	// Validate all labels exist in the config.
	for i, d := range data {
		if _, ok := labelIdx[d.Label]; !ok {
			return nil, fmt.Errorf("sentiment finetune: data[%d] has unknown label %q", i, d.Label)
		}
	}

	// Split into train/val.
	trainData, valData := splitData(data, cfg.ValSplit)

	numClasses := model.NumClasses()
	result := &TrainingResult{
		EpochMetrics: make([]EpochMetric, 0, cfg.Epochs),
	}

	rng := rand.New(rand.NewSource(42))

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		// Shuffle training data.
		rng.Shuffle(len(trainData), func(i, j int) {
			trainData[i], trainData[j] = trainData[j], trainData[i]
		})

		epochLoss := 0.0
		batchCount := 0

		// Process batches.
		for i := 0; i < len(trainData); i += cfg.BatchSize {
			end := i + cfg.BatchSize
			if end > len(trainData) {
				end = len(trainData)
			}
			batch := trainData[i:end]

			batchLoss := 0.0
			batchGrad := make([]float64, numClasses)

			for _, sample := range batch {
				ids, err := tokenizer.Encode(sample.Text)
				if err != nil {
					return nil, fmt.Errorf("sentiment finetune: tokenize: %w", err)
				}
				if len(ids) > cfg.MaxSeqLen {
					ids = ids[:cfg.MaxSeqLen]
				}

				logits32, err := model.Forward(ids)
				if err != nil {
					return nil, fmt.Errorf("sentiment finetune: forward: %w", err)
				}

				logits := float32ToFloat64(logits32)
				probs := softmax(logits)
				target := labelIdx[sample.Label]

				// Cross-entropy loss: -log(prob[target])
				if probs[target] > 0 {
					batchLoss -= math.Log(probs[target])
				} else {
					batchLoss += 100.0 // large penalty for zero probability
				}

				// Gradient: softmax - one_hot
				for c := 0; c < numClasses; c++ {
					delta := probs[c]
					if c == target {
						delta -= 1.0
					}
					batchGrad[c] += delta
				}
			}

			// Average over batch.
			batchSize := float64(len(batch))
			batchLoss /= batchSize
			for c := range batchGrad {
				batchGrad[c] /= batchSize
			}

			if err := model.UpdateParams(batchGrad, cfg.LearningRate); err != nil {
				return nil, fmt.Errorf("sentiment finetune: update params: %w", err)
			}

			epochLoss += batchLoss
			batchCount++
		}

		avgLoss := epochLoss / float64(batchCount)

		// Compute validation accuracy.
		valAcc := 0.0
		if len(valData) > 0 {
			correct := 0
			for _, sample := range valData {
				ids, err := tokenizer.Encode(sample.Text)
				if err != nil {
					return nil, fmt.Errorf("sentiment finetune: tokenize val: %w", err)
				}
				if len(ids) > cfg.MaxSeqLen {
					ids = ids[:cfg.MaxSeqLen]
				}
				logits32, err := model.Forward(ids)
				if err != nil {
					return nil, fmt.Errorf("sentiment finetune: forward val: %w", err)
				}
				logits := float32ToFloat64(logits32)
				pred := argmax(logits)
				if cfg.Labels[pred] == sample.Label {
					correct++
				}
			}
			valAcc = float64(correct) / float64(len(valData))
		}

		result.EpochMetrics = append(result.EpochMetrics, EpochMetric{
			Epoch:     epoch + 1,
			TrainLoss: avgLoss,
			ValAcc:    valAcc,
		})
	}

	last := result.EpochMetrics[len(result.EpochMetrics)-1]
	result.FinalTrainLoss = last.TrainLoss
	result.FinalValAcc = last.ValAcc

	return result, nil
}

// splitData splits data into train and validation sets.
func splitData(data []TrainingData, valSplit float64) (train, val []TrainingData) {
	if valSplit <= 0 {
		return append([]TrainingData(nil), data...), nil
	}
	// Deterministic split: last valSplit fraction goes to validation.
	valCount := int(math.Round(float64(len(data)) * valSplit))
	if valCount == 0 {
		valCount = 1
	}
	if valCount >= len(data) {
		valCount = len(data) - 1
	}
	splitIdx := len(data) - valCount
	train = make([]TrainingData, splitIdx)
	copy(train, data[:splitIdx])
	val = make([]TrainingData, valCount)
	copy(val, data[splitIdx:])
	return train, val
}

// argmax returns the index of the maximum value.
func argmax(vals []float64) int {
	maxIdx := 0
	maxVal := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] > maxVal {
			maxVal = vals[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// LoadTrainingData reads labeled data from a CSV or JSONL file.
// CSV format: "text","label" columns (header required).
// JSONL format: {"text": "...", "label": "..."} per line.
// Format is auto-detected from file extension.
func LoadTrainingData(path string) ([]TrainingData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("sentiment load data: %w", err)
	}
	defer f.Close()

	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".csv":
		return loadCSV(f)
	case ".jsonl":
		return loadJSONL(f)
	default:
		return nil, fmt.Errorf("sentiment load data: unsupported format %q (use .csv or .jsonl)", ext)
	}
}

// loadCSV reads training data from CSV with "text" and "label" columns.
func loadCSV(r io.Reader) ([]TrainingData, error) {
	reader := csv.NewReader(r)

	// Read header.
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("sentiment load csv: read header: %w", err)
	}

	textCol, labelCol := -1, -1
	for i, col := range header {
		switch strings.TrimSpace(strings.ToLower(col)) {
		case "text":
			textCol = i
		case "label":
			labelCol = i
		}
	}
	if textCol < 0 || labelCol < 0 {
		return nil, fmt.Errorf("sentiment load csv: header must contain 'text' and 'label' columns, got %v", header)
	}

	var data []TrainingData
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("sentiment load csv: %w", err)
		}
		if textCol >= len(record) || labelCol >= len(record) {
			return nil, fmt.Errorf("sentiment load csv: row has %d columns, need at least %d", len(record), max(textCol, labelCol)+1)
		}
		data = append(data, TrainingData{
			Text:  record[textCol],
			Label: record[labelCol],
		})
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("sentiment load csv: no data rows found")
	}
	return data, nil
}

// loadJSONL reads training data from JSONL format.
func loadJSONL(r io.Reader) ([]TrainingData, error) {
	scanner := bufio.NewScanner(r)
	var data []TrainingData
	line := 0

	for scanner.Scan() {
		line++
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}
		var d TrainingData
		if err := json.Unmarshal([]byte(text), &d); err != nil {
			return nil, fmt.Errorf("sentiment load jsonl: line %d: %w", line, err)
		}
		if d.Text == "" {
			return nil, fmt.Errorf("sentiment load jsonl: line %d: missing 'text' field", line)
		}
		if d.Label == "" {
			return nil, fmt.Errorf("sentiment load jsonl: line %d: missing 'label' field", line)
		}
		data = append(data, d)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("sentiment load jsonl: %w", err)
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("sentiment load jsonl: no data found")
	}
	return data, nil
}
