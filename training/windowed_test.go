package training

import (
	"testing"
)

// mockFlatBackend implements only Backend (not WindowedBackend).
type mockFlatBackend struct {
	trainCalled bool
}

func (m *mockFlatBackend) Train(features [][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	m.trainCalled = true
	return &TrainResult{FinalLoss: 0.1, BestLoss: 0.05, BestEpoch: 5, TotalEpochs: 10}, nil
}

// mockFlatPredictor implements only Predictor.
type mockFlatPredictor struct{}

func (m *mockFlatPredictor) Predict(modelPath string, features [][]float64) ([]float64, error) {
	out := make([]float64, len(features))
	for i := range out {
		out[i] = 1.0
	}
	return out, nil
}

// mockWindowedBackend implements both Backend and WindowedBackend.
type mockWindowedBackend struct {
	trainCalled         bool
	trainWindowedCalled bool
}

func (m *mockWindowedBackend) Train(features [][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	m.trainCalled = true
	return &TrainResult{FinalLoss: 0.2, BestLoss: 0.1, BestEpoch: 3, TotalEpochs: 10}, nil
}

func (m *mockWindowedBackend) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	m.trainWindowedCalled = true
	return &TrainResult{FinalLoss: 0.05, BestLoss: 0.02, BestEpoch: 7, TotalEpochs: 10}, nil
}

// mockWindowedPredictor implements both Predictor and WindowedPredictor.
type mockWindowedPredictor struct{}

func (m *mockWindowedPredictor) Predict(modelPath string, features [][]float64) ([]float64, error) {
	return make([]float64, len(features)), nil
}

func (m *mockWindowedPredictor) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	out := make([]float64, len(windows))
	for i := range out {
		out[i] = 2.0
	}
	return out, nil
}

func TestCreateWindows(t *testing.T) {
	tests := []struct {
		name       string
		data       [][]float64
		windowLen  int
		wantN      int
		wantLabels []float64
	}{
		{
			name:       "basic 3-row window from 5 rows",
			data:       [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}},
			windowLen:  3,
			wantN:      3,
			wantLabels: []float64{6, 8, 10},
		},
		{
			name:       "window equals data length",
			data:       [][]float64{{1, 2}, {3, 4}},
			windowLen:  2,
			wantN:      1,
			wantLabels: []float64{4},
		},
		{
			name:      "window larger than data",
			data:      [][]float64{{1, 2}},
			windowLen: 5,
			wantN:     0,
		},
		{
			name:      "zero window length",
			data:      [][]float64{{1, 2}, {3, 4}},
			windowLen: 0,
			wantN:     0,
		},
		{
			name:      "negative window length",
			data:      [][]float64{{1, 2}},
			windowLen: -1,
			wantN:     0,
		},
		{
			name:      "nil data",
			data:      nil,
			windowLen: 3,
			wantN:     0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			windows, labels := CreateWindows(tt.data, tt.windowLen)
			if len(windows) != tt.wantN {
				t.Fatalf("got %d windows, want %d", len(windows), tt.wantN)
			}
			if len(labels) != tt.wantN {
				t.Fatalf("got %d labels, want %d", len(labels), tt.wantN)
			}
			for i, wl := range tt.wantLabels {
				if labels[i] != wl {
					t.Errorf("label[%d] = %v, want %v", i, labels[i], wl)
				}
			}
			// Verify windows are deep copies.
			if tt.wantN > 0 && tt.data != nil {
				windows[0][0][0] = -999
				if tt.data[0][0] == -999 {
					t.Error("window data is not a deep copy of input")
				}
			}
		})
	}
}

func TestParseWindowSizes(t *testing.T) {
	tests := []struct {
		name string
		s    string
		want []int
	}{
		{"basic", "15,30,60,120", []int{15, 30, 60, 120}},
		{"single", "42", []int{42}},
		{"with spaces", " 10 , 20 , 30 ", []int{10, 20, 30}},
		{"empty string", "", nil},
		{"invalid entries skipped", "10,abc,20", []int{10, 20}},
		{"all invalid", "foo,bar", nil},
		{"trailing comma", "5,10,", []int{5, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseWindowSizes(tt.s)
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("got[%d] = %d, want %d", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// TestWindowedBackendCompilesBothInterfaces verifies that a type satisfying both
// Backend and WindowedBackend compiles and can be assigned to both interface types.
func TestWindowedBackendCompilesBothInterfaces(t *testing.T) {
	var b Backend = &mockWindowedBackend{}
	var wb WindowedBackend = &mockWindowedBackend{}
	if b == nil || wb == nil {
		t.Fatal("expected non-nil interface values")
	}
}

// TestWindowedPredictorCompilesBothInterfaces verifies that a type satisfying both
// Predictor and WindowedPredictor compiles and can be assigned to both.
func TestWindowedPredictorCompilesBothInterfaces(t *testing.T) {
	var p Predictor = &mockWindowedPredictor{}
	var wp WindowedPredictor = &mockWindowedPredictor{}
	if p == nil || wp == nil {
		t.Fatal("expected non-nil interface values")
	}
}

// TestDispatchTrainWindowedBackend verifies that DispatchTrain calls
// TrainWindowed when the backend implements WindowedBackend.
func TestDispatchTrainWindowedBackend(t *testing.T) {
	wb := &mockWindowedBackend{}
	features := [][]float64{{1, 2}, {3, 4}}
	labels := []float64{0, 1}
	config := TrainConfig{Epochs: 5}

	result, err := DispatchTrain(wb, features, labels, config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wb.trainWindowedCalled {
		t.Error("expected TrainWindowed to be called")
	}
	if wb.trainCalled {
		t.Error("expected Train NOT to be called")
	}
	if result.FinalLoss != 0.05 {
		t.Errorf("expected FinalLoss 0.05, got %f", result.FinalLoss)
	}
}

// TestDispatchTrainFlatBackend verifies that DispatchTrain falls back to
// Backend.Train when the backend does not implement WindowedBackend.
func TestDispatchTrainFlatBackend(t *testing.T) {
	fb := &mockFlatBackend{}
	features := [][]float64{{1, 2}, {3, 4}}
	labels := []float64{0, 1}
	config := TrainConfig{Epochs: 5}

	result, err := DispatchTrain(fb, features, labels, config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !fb.trainCalled {
		t.Error("expected Train to be called")
	}
	if result.FinalLoss != 0.1 {
		t.Errorf("expected FinalLoss 0.1, got %f", result.FinalLoss)
	}
}

// TestDispatchTrainWindowedWithWindows verifies that DispatchTrainWindowed
// calls TrainWindowed on a WindowedBackend.
func TestDispatchTrainWindowedWithWindows(t *testing.T) {
	wb := &mockWindowedBackend{}
	windows := [][][]float64{
		{{1, 2}, {3, 4}},
		{{5, 6}, {7, 8}},
	}
	labels := []float64{0, 1}
	config := TrainConfig{Epochs: 5}

	result, err := DispatchTrainWindowed(wb, windows, labels, config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !wb.trainWindowedCalled {
		t.Error("expected TrainWindowed to be called")
	}
	if wb.trainCalled {
		t.Error("expected Train NOT to be called")
	}
	if result.BestLoss != 0.02 {
		t.Errorf("expected BestLoss 0.02, got %f", result.BestLoss)
	}
}

// TestDispatchTrainWindowedFallback verifies that DispatchTrainWindowed
// flattens windows and falls back to Train for non-windowed backends.
func TestDispatchTrainWindowedFallback(t *testing.T) {
	fb := &mockFlatBackend{}
	windows := [][][]float64{
		{{1, 2}, {3, 4}},
		{{5, 6}, {7, 8}},
	}
	labels := []float64{0, 1}
	config := TrainConfig{Epochs: 5}

	result, err := DispatchTrainWindowed(fb, windows, labels, config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !fb.trainCalled {
		t.Error("expected Train to be called for flat backend")
	}
	if result.FinalLoss != 0.1 {
		t.Errorf("expected FinalLoss 0.1, got %f", result.FinalLoss)
	}
}

// TestTypeAssertionDispatch verifies the type assertion pattern used by
// walk-forward validators.
func TestTypeAssertionDispatch(t *testing.T) {
	backends := []Backend{
		&mockFlatBackend{},
		&mockWindowedBackend{},
	}

	for _, b := range backends {
		_, isWindowed := b.(WindowedBackend)
		switch b.(type) {
		case *mockFlatBackend:
			if isWindowed {
				t.Error("mockFlatBackend should not satisfy WindowedBackend")
			}
		case *mockWindowedBackend:
			if !isWindowed {
				t.Error("mockWindowedBackend should satisfy WindowedBackend")
			}
		}
	}
}
