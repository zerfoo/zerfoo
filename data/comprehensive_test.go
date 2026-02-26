package data

import (
	"testing"
)

// ---------- NormalizeFeatures edge cases ----------

func TestNormalizeFeatures_SingleSample(t *testing.T) {
	// Single sample: counts[i]=1, so stdDevs[i]=0, features set to 0
	ds := &Dataset{
		Batches: []Batch{
			{
				Index: 100,
				Samples: []Sample{
					{ID: "s1", Features: []float64{5.0, 10.0}, Target: 0.5},
				},
			},
		},
	}

	ds.NormalizeFeatures()

	for i, f := range ds.Batches[0].Samples[0].Features {
		if f != 0.0 {
			t.Errorf("feature[%d] = %v, want 0.0 (single sample, zero stddev)", i, f)
		}
	}
}

func TestNormalizeFeatures_ConstantFeatures(t *testing.T) {
	// All samples have the same feature values → variance=0, stdDev=0 → features set to 0
	ds := &Dataset{
		Batches: []Batch{
			{
				Index: 100,
				Samples: []Sample{
					{ID: "s1", Features: []float64{7.0, 3.0}, Target: 0.1},
					{ID: "s2", Features: []float64{7.0, 3.0}, Target: 0.2},
					{ID: "s3", Features: []float64{7.0, 3.0}, Target: 0.3},
				},
			},
		},
	}

	ds.NormalizeFeatures()

	for j, sample := range ds.Batches[0].Samples {
		for i, f := range sample.Features {
			if f != 0.0 {
				t.Errorf("sample[%d].feature[%d] = %v, want 0.0 (constant values, zero stddev)", j, i, f)
			}
		}
	}
}

func TestNormalizeFeatures_MixedVariance(t *testing.T) {
	// Feature 0 has variance (1,2,3), feature 1 is constant (5,5,5)
	ds := &Dataset{
		Batches: []Batch{
			{
				Index: 100,
				Samples: []Sample{
					{ID: "s1", Features: []float64{1.0, 5.0}, Target: 0.1},
					{ID: "s2", Features: []float64{2.0, 5.0}, Target: 0.2},
					{ID: "s3", Features: []float64{3.0, 5.0}, Target: 0.3},
				},
			},
		},
	}

	ds.NormalizeFeatures()

	// Feature 1 (constant) should be zero
	for j, sample := range ds.Batches[0].Samples {
		if sample.Features[1] != 0.0 {
			t.Errorf("sample[%d].feature[1] = %v, want 0.0 (constant feature)", j, sample.Features[1])
		}
	}

	// Feature 0 should have been normalized (non-zero for non-mean values)
	if ds.Batches[0].Samples[0].Features[0] == 0.0 && ds.Batches[0].Samples[2].Features[0] == 0.0 {
		t.Error("feature[0] should have been normalized with non-zero values for non-mean samples")
	}
}

func TestNormalizeFeatures_EmptySamples(t *testing.T) {
	// Batches exist but have no samples → early return
	ds := &Dataset{
		Batches: []Batch{
			{Index: 100, Samples: []Sample{}},
		},
	}

	// Should not panic
	ds.NormalizeFeatures()
}

func TestNormalizeFeatures_MultipleBatches(t *testing.T) {
	// Single sample spread across two batches still triggers stdDev=0
	ds := &Dataset{
		Batches: []Batch{
			{
				Index: 100,
				Samples: []Sample{
					{ID: "s1", Features: []float64{4.0}, Target: 0.5},
				},
			},
			// Second batch has no samples, so counts stay at 1
		},
	}

	ds.NormalizeFeatures()

	if ds.Batches[0].Samples[0].Features[0] != 0.0 {
		t.Errorf("feature = %v, want 0.0 (single sample across batches)", ds.Batches[0].Samples[0].Features[0])
	}
}
