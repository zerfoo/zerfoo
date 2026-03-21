package training

import (
	"strconv"
	"strings"
)

// CreateWindows converts flat rows into overlapping temporal windows.
// Each window contains windowLen consecutive rows. Labels come from the last row's last element.
func CreateWindows(data [][]float64, windowLen int) (windows [][][]float64, labels []float64) {
	if windowLen <= 0 || len(data) < windowLen {
		return nil, nil
	}
	n := len(data) - windowLen + 1
	windows = make([][][]float64, n)
	labels = make([]float64, n)
	for i := 0; i < n; i++ {
		window := make([][]float64, windowLen)
		for j := 0; j < windowLen; j++ {
			row := data[i+j]
			window[j] = make([]float64, len(row))
			copy(window[j], row)
		}
		windows[i] = window
		lastRow := data[i+windowLen-1]
		labels[i] = lastRow[len(lastRow)-1]
	}
	return windows, labels
}

// ParseWindowSizes parses a comma-separated string of window sizes.
// Example: "15,30,60,120" -> []int{15, 30, 60, 120}
func ParseWindowSizes(s string) []int {
	parts := strings.Split(s, ",")
	var sizes []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			continue
		}
		sizes = append(sizes, v)
	}
	return sizes
}

// WindowedBackend is implemented by time-series models that consume temporal
// windows rather than flat feature vectors. Walk-forward validators check for
// this interface via type assertion before falling back to standard training.
type WindowedBackend interface {
	TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error)
}

// WindowedPredictor is implemented by models that predict from temporal windows
// instead of flat feature vectors.
type WindowedPredictor interface {
	PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error)
}

// TrainConfig holds hyperparameters shared across flat and windowed backends.
type TrainConfig struct {
	Epochs       int
	BatchSize    int
	LearningRate float64
	WeightDecay  float64
}

// TrainResult holds the outcome of a training run.
type TrainResult struct {
	FinalLoss   float64
	BestLoss    float64
	BestEpoch   int
	TotalEpochs int
}

// Backend is implemented by models that train on flat tabular data.
// Walk-forward validators dispatch to this interface when the model does not
// implement WindowedBackend.
type Backend interface {
	Train(features [][]float64, labels []float64, config TrainConfig) (*TrainResult, error)
}

// Predictor is implemented by models that predict from flat feature vectors.
type Predictor interface {
	Predict(modelPath string, features [][]float64) ([]float64, error)
}

// DispatchTrain checks whether backend implements WindowedBackend and calls
// TrainWindowed if so; otherwise it falls back to Backend.Train with flat
// features. This is the dispatch logic used by walk-forward validators.
func DispatchTrain(backend Backend, features [][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	if wb, ok := backend.(WindowedBackend); ok {
		// Promote each flat row to a single-step window [1][features].
		windows := make([][][]float64, len(features))
		for i, row := range features {
			windows[i] = [][]float64{row}
		}
		return wb.TrainWindowed(windows, labels, config)
	}
	return backend.Train(features, labels, config)
}

// DispatchTrainWindowed dispatches a windowed training call. If the backend
// implements WindowedBackend it calls TrainWindowed directly; otherwise it
// flattens the windows and falls back to Backend.Train.
func DispatchTrainWindowed(backend Backend, windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	if wb, ok := backend.(WindowedBackend); ok {
		return wb.TrainWindowed(windows, labels, config)
	}
	// Flatten: take the last timestep of each window as the feature vector.
	features := make([][]float64, len(windows))
	for i, w := range windows {
		if len(w) > 0 {
			features[i] = w[len(w)-1]
		}
	}
	return backend.Train(features, labels, config)
}
