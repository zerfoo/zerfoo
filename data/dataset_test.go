package data

import (
	"math"
	"testing"
)

func TestSample_Create(t *testing.T) {
	tests := []struct {
		name     string
		id       string
		features []float64
		target   float64
		wantID   string
	}{
		{
			name:     "valid sample",
			id:       "sample123",
			features: []float64{1.0, 2.0, 3.0},
			target:   0.5,
			wantID:   "sample123",
		},
		{
			name:     "empty features",
			id:       "sample456",
			features: []float64{},
			target:   0.0,
			wantID:   "sample456",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sample := Sample{
				ID:       tt.id,
				Features: tt.features,
				Target:   tt.target,
			}

			if sample.ID != tt.wantID {
				t.Errorf("Sample.ID = %v, want %v", sample.ID, tt.wantID)
			}

			if len(sample.Features) != len(tt.features) {
				t.Errorf("len(Sample.Features) = %v, want %v", len(sample.Features), len(tt.features))
			}

			if sample.Target != tt.target {
				t.Errorf("Sample.Target = %v, want %v", sample.Target, tt.target)
			}
		})
	}
}

func TestBatch_Create(t *testing.T) {
	tests := []struct {
		name      string
		index     int
		samples   []Sample
		stats     []float64
		wantIndex int
	}{
		{
			name:  "valid batch",
			index: 100,
			samples: []Sample{
				{ID: "s1", Features: []float64{1.0}, Target: 0.5},
				{ID: "s2", Features: []float64{2.0}, Target: 0.7},
			},
			stats:     []float64{1.5, 0.1},
			wantIndex: 100,
		},
		{
			name:      "empty batch",
			index:     101,
			samples:   []Sample{},
			stats:     []float64{},
			wantIndex: 101,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batch := Batch{
				Index:   tt.index,
				Samples: tt.samples,
				Stats:   tt.stats,
			}

			if batch.Index != tt.wantIndex {
				t.Errorf("Batch.Index = %v, want %v", batch.Index, tt.wantIndex)
			}

			if len(batch.Samples) != len(tt.samples) {
				t.Errorf("len(Batch.Samples) = %v, want %v", len(batch.Samples), len(tt.samples))
			}

			if len(batch.Stats) != len(tt.stats) {
				t.Errorf("len(Batch.Stats) = %v, want %v", len(batch.Stats), len(tt.stats))
			}
		})
	}
}

func TestDataset_Create(t *testing.T) {
	tests := []struct {
		name    string
		batches []Batch
		wantLen int
	}{
		{
			name: "dataset with multiple batches",
			batches: []Batch{
				{Index: 100, Samples: []Sample{{ID: "s1", Features: []float64{1.0}, Target: 0.5}}},
				{Index: 101, Samples: []Sample{{ID: "s2", Features: []float64{2.0}, Target: 0.7}}},
			},
			wantLen: 2,
		},
		{
			name:    "empty dataset",
			batches: []Batch{},
			wantLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataset := Dataset{
				Batches: tt.batches,
			}

			if len(dataset.Batches) != tt.wantLen {
				t.Errorf("len(Dataset.Batches) = %v, want %v", len(dataset.Batches), tt.wantLen)
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
				Batches: []Batch{
					{
						Index: 100,
						Samples: []Sample{
							{ID: "s1", Features: []float64{1.0, 2.0}, Target: 0.5},
							{ID: "s2", Features: []float64{2.0, 4.0}, Target: 0.7},
							{ID: "s3", Features: []float64{3.0, 6.0}, Target: 0.9},
						},
					},
				},
			},
			wantMean: 0.0, // After normalization, mean should be close to 0
			wantStd:  1.0, // After normalization, std should be close to 1
			epsilon:  1e-10,
		},
		{
			name: "empty dataset",
			dataset: &Dataset{
				Batches: []Batch{},
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
			if len(tt.dataset.Batches) > 0 {
				testDataset.Batches = make([]Batch, len(tt.dataset.Batches))
				for i, batch := range tt.dataset.Batches {
					testDataset.Batches[i] = Batch{
						Index:   batch.Index,
						Stats:   make([]float64, len(batch.Stats)),
						Samples: make([]Sample, len(batch.Samples)),
					}
					copy(testDataset.Batches[i].Stats, batch.Stats)
					for j, sample := range batch.Samples {
						testDataset.Batches[i].Samples[j] = Sample{
							ID:       sample.ID,
							Target:   sample.Target,
							Features: make([]float64, len(sample.Features)),
						}
						copy(testDataset.Batches[i].Samples[j].Features, sample.Features)
					}
				}
			}

			testDataset.NormalizeFeatures()

			if len(testDataset.Batches) == 0 || len(testDataset.Batches[0].Samples) == 0 {
				return // Skip validation for empty datasets
			}

			// Calculate mean and std of normalized features
			numFeatures := len(testDataset.Batches[0].Samples[0].Features)
			if numFeatures == 0 {
				return
			}

			for featureIdx := 0; featureIdx < numFeatures; featureIdx++ {
				var sum, sumSq float64
				var count int

				for _, batch := range testDataset.Batches {
					for _, sample := range batch.Samples {
						if featureIdx < len(sample.Features) {
							val := sample.Features[featureIdx]
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
					for _, batch := range testDataset.Batches {
						for _, sample := range batch.Samples {
							if featureIdx < len(sample.Features) {
								val := sample.Features[featureIdx]
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
