package training

import (
	"testing"

	"github.com/zerfoo/zerfoo/data"
)

func TestEraSequencer_GenerateSequences(t *testing.T) {
	// Create test dataset with multiple eras
	dataset := &data.Dataset{
		Eras: []data.EraData{
			{Era: 100, Stocks: []data.StockData{{ID: "stock1", Features: []float64{1.0}, Target: 0.5}}},
			{Era: 101, Stocks: []data.StockData{{ID: "stock1", Features: []float64{2.0}, Target: 0.6}}},
			{Era: 102, Stocks: []data.StockData{{ID: "stock1", Features: []float64{3.0}, Target: 0.7}}},
			{Era: 103, Stocks: []data.StockData{{ID: "stock1", Features: []float64{4.0}, Target: 0.8}}},
			{Era: 104, Stocks: []data.StockData{{ID: "stock1", Features: []float64{5.0}, Target: 0.9}}},
		},
	}

	tests := []struct {
		name         string
		maxSeqLen    int
		numSequences int
		wantCount    int
		wantMaxLen   int
		wantMinLen   int
	}{
		{
			name:         "generate short sequences",
			maxSeqLen:    2,
			numSequences: 10,
			wantCount:    10,
			wantMaxLen:   2,
			wantMinLen:   1,
		},
		{
			name:         "generate longer sequences",
			maxSeqLen:    4,
			numSequences: 5,
			wantCount:    5,
			wantMaxLen:   4,
			wantMinLen:   1,
		},
		{
			name:         "single sequence",
			maxSeqLen:    3,
			numSequences: 1,
			wantCount:    1,
			wantMaxLen:   3,
			wantMinLen:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sequencer := NewEraSequencer(tt.maxSeqLen)
			sequences := sequencer.GenerateSequences(dataset, tt.numSequences)

			if len(sequences) != tt.wantCount {
				t.Errorf("EraSequencer.GenerateSequences() count = %v, want %v", len(sequences), tt.wantCount)
			}

			// Verify each sequence properties
			for i, seq := range sequences {
				if len(seq.Eras) > tt.wantMaxLen {
					t.Errorf("Sequence %d length = %v, want <= %v", i, len(seq.Eras), tt.wantMaxLen)
				}

				if len(seq.Eras) < tt.wantMinLen {
					t.Errorf("Sequence %d length = %v, want >= %v", i, len(seq.Eras), tt.wantMinLen)
				}

				// Verify chronological order
				for j := 1; j < len(seq.Eras); j++ {
					if seq.Eras[j].Era <= seq.Eras[j-1].Era {
						t.Errorf("Sequence %d not chronologically ordered: era %d <= era %d", i, seq.Eras[j].Era, seq.Eras[j-1].Era)
					}
				}

				// Verify eras are consecutive
				for j := 1; j < len(seq.Eras); j++ {
					if seq.Eras[j].Era != seq.Eras[j-1].Era+1 {
						t.Errorf("Sequence %d eras not consecutive: era %d != era %d + 1", i, seq.Eras[j].Era, seq.Eras[j-1].Era)
					}
				}
			}
		})
	}
}

func TestEraSequencer_GenerateSequences_EmptyDataset(t *testing.T) {
	dataset := &data.Dataset{
		Eras: []data.EraData{},
	}

	sequencer := NewEraSequencer(3)
	sequences := sequencer.GenerateSequences(dataset, 5)

	if len(sequences) != 0 {
		t.Errorf("EraSequencer.GenerateSequences() with empty dataset should return 0 sequences, got %v", len(sequences))
	}
}

func TestEraSequencer_GenerateSequences_SingleEra(t *testing.T) {
	dataset := &data.Dataset{
		Eras: []data.EraData{
			{Era: 100, Stocks: []data.StockData{{ID: "stock1", Features: []float64{1.0}, Target: 0.5}}},
		},
	}

	sequencer := NewEraSequencer(3)
	sequences := sequencer.GenerateSequences(dataset, 5)

	// Should generate sequences with just the single era
	if len(sequences) != 5 {
		t.Errorf("EraSequencer.GenerateSequences() count = %v, want 5", len(sequences))
	}

	for i, seq := range sequences {
		if len(seq.Eras) != 1 {
			t.Errorf("Sequence %d should have 1 era, got %v", i, len(seq.Eras))
		}
		if seq.Eras[0].Era != 100 {
			t.Errorf("Sequence %d era should be 100, got %v", i, seq.Eras[0].Era)
		}
	}
}

func TestEraSequencer_GenerateTrainValidationSplit(t *testing.T) {
	dataset := &data.Dataset{
		Eras: []data.EraData{
			{Era: 100, Stocks: []data.StockData{{ID: "stock1", Features: []float64{1.0}, Target: 0.5}}},
			{Era: 101, Stocks: []data.StockData{{ID: "stock1", Features: []float64{2.0}, Target: 0.6}}},
			{Era: 102, Stocks: []data.StockData{{ID: "stock1", Features: []float64{3.0}, Target: 0.7}}},
			{Era: 103, Stocks: []data.StockData{{ID: "stock1", Features: []float64{4.0}, Target: 0.8}}},
			{Era: 104, Stocks: []data.StockData{{ID: "stock1", Features: []float64{5.0}, Target: 0.9}}},
		},
	}

	tests := []struct {
		name           string
		validationEras int
		wantTrainEras  int
		wantValidEras  int
	}{
		{
			name:           "split with 2 validation eras",
			validationEras: 2,
			wantTrainEras:  3,
			wantValidEras:  2,
		},
		{
			name:           "split with 1 validation era",
			validationEras: 1,
			wantTrainEras:  4,
			wantValidEras:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sequencer := NewEraSequencer(3)
			trainData, validData := sequencer.GenerateTrainValidationSplit(dataset, tt.validationEras)

			if len(trainData.Eras) != tt.wantTrainEras {
				t.Errorf("Train data eras = %v, want %v", len(trainData.Eras), tt.wantTrainEras)
			}

			if len(validData.Eras) != tt.wantValidEras {
				t.Errorf("Validation data eras = %v, want %v", len(validData.Eras), tt.wantValidEras)
			}

			// Verify no overlap between train and validation
			trainEras := make(map[int]bool)
			for _, era := range trainData.Eras {
				trainEras[era.Era] = true
			}

			for _, era := range validData.Eras {
				if trainEras[era.Era] {
					t.Errorf("Era %v appears in both train and validation sets", era.Era)
				}
			}

			// Verify validation eras are the last ones
			if len(validData.Eras) > 0 {
				lastValidEra := validData.Eras[len(validData.Eras)-1].Era
				originalLastEra := dataset.Eras[len(dataset.Eras)-1].Era
				if lastValidEra != originalLastEra {
					t.Errorf("Last validation era = %v, want %v", lastValidEra, originalLastEra)
				}
			}
		})
	}
}

func TestEraSequencer_GenerateTrainValidationSplit_TooManyValidationEras(t *testing.T) {
	dataset := &data.Dataset{
		Eras: []data.EraData{
			{Era: 100, Stocks: []data.StockData{{ID: "stock1", Features: []float64{1.0}, Target: 0.5}}},
			{Era: 101, Stocks: []data.StockData{{ID: "stock1", Features: []float64{2.0}, Target: 0.6}}},
		},
	}

	sequencer := NewEraSequencer(3)
	trainData, validData := sequencer.GenerateTrainValidationSplit(dataset, 5) // More validation eras than available

	// Should use all eras for validation
	if len(trainData.Eras) != 0 {
		t.Errorf("Train data should be empty when requesting more validation eras than available, got %v eras", len(trainData.Eras))
	}

	if len(validData.Eras) != 2 {
		t.Errorf("Validation data should have 2 eras (all available), got %v", len(validData.Eras))
	}
}
