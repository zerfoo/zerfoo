package timeseries

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestITransformer_TrainWindowed_Engine(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	channels := 3
	inputLen := 12
	outputLen := 4
	nSamples := 50

	config := ITransformerConfig{
		Channels: channels, InputLen: inputLen, OutputLen: outputLen,
		DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
	}
	m, err := NewITransformer(config, eng, numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*channels*outputLen)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, channels)
		for c := 0; c < channels; c++ {
			windows[i][c] = make([]float64, inputLen)
			for t := 0; t < inputLen; t++ {
				windows[i][c][t] = math.Sin(float64(i+t+c) * 0.3)
			}
			for o := 0; o < outputLen; o++ {
				labels[i*channels*outputLen+c*outputLen+o] = math.Sin(float64(i+inputLen+o+c) * 0.3)
			}
		}
	}

	res, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   50,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if res.FinalLoss >= res.LossHistory[0] {
		t.Errorf("expected loss to decrease: initial=%f, final=%f", res.LossHistory[0], res.FinalLoss)
	}
	if math.IsNaN(res.FinalLoss) || math.IsInf(res.FinalLoss, 0) {
		t.Errorf("final loss is not finite: %f", res.FinalLoss)
	}
}

func TestITransformer_TrainWindowed_NilEngine(t *testing.T) {
	channels := 3
	inputLen := 12
	outputLen := 4
	nSamples := 30

	config := ITransformerConfig{
		Channels: channels, InputLen: inputLen, OutputLen: outputLen,
		DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
	}
	m, err := NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	if m.engine != nil {
		t.Fatal("expected nil engine")
	}

	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*channels*outputLen)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, channels)
		for c := 0; c < channels; c++ {
			windows[i][c] = make([]float64, inputLen)
			for tt := 0; tt < inputLen; tt++ {
				windows[i][c][tt] = rand.Float64()
			}
			for o := 0; o < outputLen; o++ {
				labels[i*channels*outputLen+c*outputLen+o] = rand.Float64()
			}
		}
	}

	res, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   20,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if math.IsNaN(res.FinalLoss) || math.IsInf(res.FinalLoss, 0) {
		t.Errorf("final loss is not finite: %f", res.FinalLoss)
	}
}
