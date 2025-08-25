package features

import (
	"testing"

	"github.com/zerfoo/zerfoo/data"
)

func TestLaggedTransformer_Transform(t *testing.T) {
	tests := []struct {
		name           string
		dataset        *data.Dataset
		lags           []int
		wantFeatureLen int // Expected length of features after transformation
	}{
		{
			name: "single lag transformation",
			dataset: &data.Dataset{
				Eras: []data.EraData{
					{
						Era: 100,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{1.0, 2.0}, Target: 0.5},
						},
					},
					{
						Era: 101,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{3.0, 4.0}, Target: 0.7},
						},
					},
				},
			},
			lags:           []int{1},
			wantFeatureLen: 4, // 2 original + 2 lagged features
		},
		{
			name: "multiple lags transformation",
			dataset: &data.Dataset{
				Eras: []data.EraData{
					{
						Era: 100,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{1.0, 2.0}, Target: 0.5},
						},
					},
					{
						Era: 101,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{3.0, 4.0}, Target: 0.7},
						},
					},
					{
						Era: 102,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{5.0, 6.0}, Target: 0.9},
						},
					},
				},
			},
			lags:           []int{1, 2},
			wantFeatureLen: 6, // 2 original + 2*2 lagged features
		},
		{
			name: "empty dataset",
			dataset: &data.Dataset{
				Eras: []data.EraData{},
			},
			lags:           []int{1},
			wantFeatureLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transformer := NewLaggedTransformer(tt.lags)
			
			// Make a copy of the dataset to avoid modifying test data
			testDataset := copyDataset(tt.dataset)
			
			err := transformer.Transform(testDataset)
			if err != nil {
				t.Errorf("LaggedTransformer.Transform() error = %v", err)
				return
			}

			if len(testDataset.Eras) == 0 {
				return // Skip validation for empty datasets
			}

			// Check that all stocks have the expected feature length
			for _, era := range testDataset.Eras {
				for _, stock := range era.Stocks {
					if len(stock.Features) != tt.wantFeatureLen {
						t.Errorf("LaggedTransformer.Transform() feature length = %v, want %v", len(stock.Features), tt.wantFeatureLen)
					}
				}
			}
		})
	}
}

func TestRollingTransformer_Transform(t *testing.T) {
	tests := []struct {
		name           string
		dataset        *data.Dataset
		window         int
		wantFeatureLen int // Expected length of features after transformation
	}{
		{
			name: "rolling window transformation",
			dataset: &data.Dataset{
				Eras: []data.EraData{
					{
						Era: 100,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{1.0, 2.0}, Target: 0.5},
						},
					},
					{
						Era: 101,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{3.0, 4.0}, Target: 0.7},
						},
					},
					{
						Era: 102,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{5.0, 6.0}, Target: 0.9},
						},
					},
				},
			},
			window:         2,
			wantFeatureLen: 6, // 2 original + 2*2 rolling stats (mean, std) for each original feature
		},
		{
			name: "empty dataset",
			dataset: &data.Dataset{
				Eras: []data.EraData{},
			},
			window:         2,
			wantFeatureLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transformer := NewRollingTransformer(tt.window)
			
			// Make a copy of the dataset to avoid modifying test data
			testDataset := copyDataset(tt.dataset)
			
			err := transformer.Transform(testDataset)
			if err != nil {
				t.Errorf("RollingTransformer.Transform() error = %v", err)
				return
			}

			if len(testDataset.Eras) == 0 {
				return // Skip validation for empty datasets
			}

			// Check that all stocks have the expected feature length
			for _, era := range testDataset.Eras {
				for _, stock := range era.Stocks {
					if len(stock.Features) != tt.wantFeatureLen {
						t.Errorf("RollingTransformer.Transform() feature length = %v, want %v", len(stock.Features), tt.wantFeatureLen)
					}
				}
			}
		})
	}
}

func TestFFTTransformer_Transform(t *testing.T) {
	tests := []struct {
		name           string
		dataset        *data.Dataset
		window         int
		k              int // Number of top frequencies
		wantFeatureLen int // Expected length of features after transformation
	}{
		{
			name: "FFT transformation",
			dataset: &data.Dataset{
				Eras: []data.EraData{
					{
						Era: 100,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{1.0, 2.0}, Target: 0.5},
						},
					},
					{
						Era: 101,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{3.0, 4.0}, Target: 0.7},
						},
					},
					{
						Era: 102,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{5.0, 6.0}, Target: 0.9},
						},
					},
					{
						Era: 103,
						Stocks: []data.StockData{
							{ID: "stock1", Features: []float64{7.0, 8.0}, Target: 0.8},
						},
					},
				},
			},
			window:         3,
			k:              2,
			wantFeatureLen: 6, // 2 original + 2*2 FFT features (k=2 frequencies for each original feature)
		},
		{
			name: "empty dataset",
			dataset: &data.Dataset{
				Eras: []data.EraData{},
			},
			window:         3,
			k:              2,
			wantFeatureLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transformer := NewFFTTransformer(tt.window, tt.k)
			
			// Make a copy of the dataset to avoid modifying test data
			testDataset := copyDataset(tt.dataset)
			
			err := transformer.Transform(testDataset)
			if err != nil {
				t.Errorf("FFTTransformer.Transform() error = %v", err)
				return
			}

			if len(testDataset.Eras) == 0 {
				return // Skip validation for empty datasets
			}

			// Check that all stocks have the expected feature length
			for _, era := range testDataset.Eras {
				for _, stock := range era.Stocks {
					if len(stock.Features) != tt.wantFeatureLen {
						t.Errorf("FFTTransformer.Transform() feature length = %v, want %v", len(stock.Features), tt.wantFeatureLen)
					}
				}
			}
		})
	}
}

// copyDataset creates a deep copy of a dataset for testing
func copyDataset(original *data.Dataset) *data.Dataset {
	if original == nil {
		return nil
	}
	
	copyData := &data.Dataset{
		Eras: make([]data.EraData, len(original.Eras)),
	}
	
	for i, era := range original.Eras {
		copyData.Eras[i] = data.EraData{
			Era:      era.Era,
			EraStats: make([]float64, len(era.EraStats)),
			Stocks:   make([]data.StockData, len(era.Stocks)),
		}
		copy(copyData.Eras[i].EraStats, era.EraStats)
		
		for j, stock := range era.Stocks {
			copyData.Eras[i].Stocks[j] = data.StockData{
				ID:       stock.ID,
				Target:   stock.Target,
				Features: make([]float64, len(stock.Features)),
			}
			copy(copyData.Eras[i].Stocks[j].Features, stock.Features)
		}
	}
	
	return copyData
}