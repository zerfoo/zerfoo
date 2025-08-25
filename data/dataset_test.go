package data

import (
	"math"
	"testing"
)

func TestStockData_Create(t *testing.T) {
	tests := []struct {
		name     string
		id       string
		features []float64
		target   float64
		wantID   string
	}{
		{
			name:     "valid stock data",
			id:       "stock123",
			features: []float64{1.0, 2.0, 3.0},
			target:   0.5,
			wantID:   "stock123",
		},
		{
			name:     "empty features",
			id:       "stock456",
			features: []float64{},
			target:   0.0,
			wantID:   "stock456",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stock := StockData{
				ID:       tt.id,
				Features: tt.features,
				Target:   tt.target,
			}

			if stock.ID != tt.wantID {
				t.Errorf("StockData.ID = %v, want %v", stock.ID, tt.wantID)
			}

			if len(stock.Features) != len(tt.features) {
				t.Errorf("len(StockData.Features) = %v, want %v", len(stock.Features), len(tt.features))
			}

			if stock.Target != tt.target {
				t.Errorf("StockData.Target = %v, want %v", stock.Target, tt.target)
			}
		})
	}
}

func TestEraData_Create(t *testing.T) {
	tests := []struct {
		name     string
		era      int
		stocks   []StockData
		eraStats []float64
		wantEra  int
	}{
		{
			name: "valid era data",
			era:  100,
			stocks: []StockData{
				{ID: "stock1", Features: []float64{1.0}, Target: 0.5},
				{ID: "stock2", Features: []float64{2.0}, Target: 0.7},
			},
			eraStats: []float64{1.5, 0.1},
			wantEra:  100,
		},
		{
			name:     "empty era",
			era:      101,
			stocks:   []StockData{},
			eraStats: []float64{},
			wantEra:  101,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			era := EraData{
				Era:      tt.era,
				Stocks:   tt.stocks,
				EraStats: tt.eraStats,
			}

			if era.Era != tt.wantEra {
				t.Errorf("EraData.Era = %v, want %v", era.Era, tt.wantEra)
			}

			if len(era.Stocks) != len(tt.stocks) {
				t.Errorf("len(EraData.Stocks) = %v, want %v", len(era.Stocks), len(tt.stocks))
			}

			if len(era.EraStats) != len(tt.eraStats) {
				t.Errorf("len(EraData.EraStats) = %v, want %v", len(era.EraStats), len(tt.eraStats))
			}
		})
	}
}

func TestDataset_Create(t *testing.T) {
	tests := []struct {
		name     string
		eras     []EraData
		wantLen  int
	}{
		{
			name: "dataset with multiple eras",
			eras: []EraData{
				{Era: 100, Stocks: []StockData{{ID: "stock1", Features: []float64{1.0}, Target: 0.5}}},
				{Era: 101, Stocks: []StockData{{ID: "stock2", Features: []float64{2.0}, Target: 0.7}}},
			},
			wantLen: 2,
		},
		{
			name:    "empty dataset",
			eras:    []EraData{},
			wantLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataset := Dataset{
				Eras: tt.eras,
			}

			if len(dataset.Eras) != tt.wantLen {
				t.Errorf("len(Dataset.Eras) = %v, want %v", len(dataset.Eras), tt.wantLen)
			}
		})
	}
}

func TestNormalizeFeatures(t *testing.T) {
	tests := []struct {
		name     string
		dataset  *Dataset
		wantMean float64
		wantStd  float64
		epsilon  float64
	}{
		{
			name: "normalize features",
			dataset: &Dataset{
				Eras: []EraData{
					{
						Era: 100,
						Stocks: []StockData{
							{ID: "stock1", Features: []float64{1.0, 2.0}, Target: 0.5},
							{ID: "stock2", Features: []float64{2.0, 4.0}, Target: 0.7},
							{ID: "stock3", Features: []float64{3.0, 6.0}, Target: 0.9},
						},
					},
				},
			},
			wantMean: 0.0,  // After normalization, mean should be close to 0
			wantStd:  1.0,  // After normalization, std should be close to 1
			epsilon:  1e-10,
		},
		{
			name: "empty dataset",
			dataset: &Dataset{
				Eras: []EraData{},
			},
			wantMean: 0.0,
			wantStd:  0.0,
			epsilon:  1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a copy to avoid modifying the test data
			testDataset := *tt.dataset
			if len(tt.dataset.Eras) > 0 {
				testDataset.Eras = make([]EraData, len(tt.dataset.Eras))
				for i, era := range tt.dataset.Eras {
					testDataset.Eras[i] = EraData{
						Era:      era.Era,
						EraStats: make([]float64, len(era.EraStats)),
						Stocks:   make([]StockData, len(era.Stocks)),
					}
					copy(testDataset.Eras[i].EraStats, era.EraStats)
					for j, stock := range era.Stocks {
						testDataset.Eras[i].Stocks[j] = StockData{
							ID:       stock.ID,
							Target:   stock.Target,
							Features: make([]float64, len(stock.Features)),
						}
						copy(testDataset.Eras[i].Stocks[j].Features, stock.Features)
					}
				}
			}

			testDataset.NormalizeFeatures()

			if len(testDataset.Eras) == 0 || len(testDataset.Eras[0].Stocks) == 0 {
				return // Skip validation for empty datasets
			}

			// Calculate mean and std of normalized features
			numFeatures := len(testDataset.Eras[0].Stocks[0].Features)
			if numFeatures == 0 {
				return
			}

			for featureIdx := 0; featureIdx < numFeatures; featureIdx++ {
				var sum, sumSq float64
				var count int

				for _, era := range testDataset.Eras {
					for _, stock := range era.Stocks {
						if featureIdx < len(stock.Features) {
							val := stock.Features[featureIdx]
							sum += val
							sumSq += val * val
							count++
						}
					}
				}

				if count == 0 {
					continue
				}

				mean := sum / float64(count)

				// Check if mean is close to 0
				if abs(mean-tt.wantMean) > tt.epsilon {
					t.Errorf("Feature %d normalized mean = %v, want %v", featureIdx, mean, tt.wantMean)
				}

				// Calculate sample standard deviation to verify normalization
				if count > 1 {
					var variance float64
					for _, era := range testDataset.Eras {
						for _, stock := range era.Stocks {
							if featureIdx < len(stock.Features) {
								val := stock.Features[featureIdx]
								variance += (val - mean) * (val - mean)
							}
						}
					}
					sampleStd := math.Sqrt(variance / float64(count-1))
					
					// For normalized data, sample std should be close to 1
					if abs(sampleStd-1.0) > 0.1 { // More lenient epsilon for std
						t.Errorf("Feature %d normalized sample std = %v, want ~1.0", featureIdx, sampleStd)
					}
				}
			}
		})
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}