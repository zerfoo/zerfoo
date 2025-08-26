package metrics

import (
	"math"
	"testing"
)

func TestCalculateMetrics(t *testing.T) {
	tests := []struct {
		name        string
		predictions []float64
		targets     []float64
		wantNil     bool
	}{
		{
			name:        "valid inputs",
			predictions: []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			targets:     []float64{1.1, 1.9, 3.1, 3.9, 5.1},
			wantNil:     false,
		},
		{
			name:        "empty inputs",
			predictions: []float64{},
			targets:     []float64{},
			wantNil:     true,
		},
		{
			name:        "mismatched length",
			predictions: []float64{1.0, 2.0},
			targets:     []float64{1.0, 2.0, 3.0},
			wantNil:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateMetrics(tt.predictions, tt.targets)
			
			if (result == nil) != tt.wantNil {
				t.Errorf("CalculateMetrics() nil result = %v, wantNil %v", result == nil, tt.wantNil)
				return
			}

			if result != nil {
				// Check that all metrics are calculated (not NaN)
				if math.IsNaN(result.PearsonCorrelation) && len(tt.predictions) > 1 {
					t.Error("PearsonCorrelation should not be NaN for valid inputs")
				}
				if math.IsNaN(result.SpearmanCorrelation) && len(tt.predictions) > 1 {
					t.Error("SpearmanCorrelation should not be NaN for valid inputs")
				}
				if result.MSE < 0 {
					t.Error("MSE should not be negative")
				}
				if result.RMSE < 0 {
					t.Error("RMSE should not be negative")
				}
				if result.MAE < 0 {
					t.Error("MAE should not be negative")
				}
			}
		})
	}
}

func TestPearsonCorrelation(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		y    []float64
		want float64
	}{
		{
			name: "perfect positive correlation",
			x:    []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			y:    []float64{2.0, 4.0, 6.0, 8.0, 10.0},
			want: 1.0,
		},
		{
			name: "perfect negative correlation",
			x:    []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			y:    []float64{5.0, 4.0, 3.0, 2.0, 1.0},
			want: -1.0,
		},
		{
			name: "no correlation",
			x:    []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			y:    []float64{3.0, 3.0, 3.0, 3.0, 3.0},
			want: math.NaN(),
		},
		{
			name: "empty slices",
			x:    []float64{},
			y:    []float64{},
			want: math.NaN(),
		},
		{
			name: "mismatched length",
			x:    []float64{1.0, 2.0},
			y:    []float64{1.0, 2.0, 3.0},
			want: math.NaN(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PearsonCorrelation(tt.x, tt.y)
			
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("PearsonCorrelation() = %v, want NaN", got)
				}
			} else {
				if math.Abs(got-tt.want) > 1e-10 {
					t.Errorf("PearsonCorrelation() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestSpearmanCorrelation(t *testing.T) {
	tests := []struct {
		name string
		x    []float64
		y    []float64
		want float64
	}{
		{
			name: "monotonic increasing",
			x:    []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			y:    []float64{1.0, 4.0, 9.0, 16.0, 25.0}, // x^2
			want: 1.0,
		},
		{
			name: "monotonic decreasing",
			x:    []float64{1.0, 2.0, 3.0, 4.0, 5.0},
			y:    []float64{25.0, 16.0, 9.0, 4.0, 1.0},
			want: -1.0,
		},
		{
			name: "tied values",
			x:    []float64{1.0, 2.0, 2.0, 3.0},
			y:    []float64{1.0, 2.5, 2.5, 4.0},
			want: 1.0,
		},
		{
			name: "empty slices",
			x:    []float64{},
			y:    []float64{},
			want: math.NaN(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SpearmanCorrelation(tt.x, tt.y)
			
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("SpearmanCorrelation() = %v, want NaN", got)
				}
			} else {
				if math.Abs(got-tt.want) > 1e-10 {
					t.Errorf("SpearmanCorrelation() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestCalculateRanks(t *testing.T) {
	tests := []struct {
		name   string
		values []float64
		want   []float64
	}{
		{
			name:   "no ties",
			values: []float64{3.0, 1.0, 4.0, 2.0},
			want:   []float64{3.0, 1.0, 4.0, 2.0},
		},
		{
			name:   "with ties",
			values: []float64{1.0, 2.0, 2.0, 3.0},
			want:   []float64{1.0, 2.5, 2.5, 4.0},
		},
		{
			name:   "all same values",
			values: []float64{2.0, 2.0, 2.0},
			want:   []float64{2.0, 2.0, 2.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateRanks(tt.values)
			
			if len(got) != len(tt.want) {
				t.Errorf("calculateRanks() length = %v, want %v", len(got), len(tt.want))
				return
			}
			
			for i := range got {
				if math.Abs(got[i]-tt.want[i]) > 1e-10 {
					t.Errorf("calculateRanks()[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestCalculateMSE(t *testing.T) {
	tests := []struct {
		name        string
		predictions []float64
		targets     []float64
		want        float64
	}{
		{
			name:        "perfect predictions",
			predictions: []float64{1.0, 2.0, 3.0},
			targets:     []float64{1.0, 2.0, 3.0},
			want:        0.0,
		},
		{
			name:        "some error",
			predictions: []float64{1.0, 2.0, 3.0},
			targets:     []float64{1.1, 1.9, 3.1},
			want:        0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateMSE(tt.predictions, tt.targets)
			if math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("calculateMSE() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCalculateMAE(t *testing.T) {
	tests := []struct {
		name        string
		predictions []float64
		targets     []float64
		want        float64
	}{
		{
			name:        "perfect predictions",
			predictions: []float64{1.0, 2.0, 3.0},
			targets:     []float64{1.0, 2.0, 3.0},
			want:        0.0,
		},
		{
			name:        "some error",
			predictions: []float64{1.0, 2.0, 3.0},
			targets:     []float64{1.1, 1.9, 3.1},
			want:        0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateMAE(tt.predictions, tt.targets)
			if math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("calculateMAE() = %v, want %v", got, tt.want)
			}
		})
	}
}