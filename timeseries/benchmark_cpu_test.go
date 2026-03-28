package timeseries

import (
	"testing"
	"time"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestAllBackends_CPUTrainingBenchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping CPU training benchmark in short mode")
	}
	const (
		nSamples  = 1000
		channels  = 5
		inputLen  = 10
		outputLen = 1
		epochs    = 5
	)

	rng := rand.New(rand.NewPCG(42, 0))
	windows := make([][][]float64, nSamples)
	for i := range windows {
		windows[i] = make([][]float64, channels)
		for c := range windows[i] {
			windows[i][c] = make([]float64, inputLen)
			for t := range windows[i][c] {
				windows[i][c][t] = math.Sin(float64(t+i)*0.3+float64(c)*0.5) + rng.Float64()*0.1
			}
		}
	}

	config := TrainConfig{
		Epochs:       epochs,
		LR:           1e-3,
		WeightDecay:  1e-4,
		GradClip:     1.0,
		WarmupEpochs: 1,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	type benchCase struct {
		name      string
		labels    []float64
		train     func([][][]float64, []float64, TrainConfig) (*TrainResult, error)
	}

	labelsFlat := func(n, ch, out int) []float64 {
		r := rand.New(rand.NewPCG(99, 0))
		l := make([]float64, n*ch*out)
		for i := range l {
			l[i] = r.Float64()
		}
		return l
	}

	cases := []benchCase{
		{
			name:   "DLinear",
			labels: labelsFlat(nSamples, channels, outputLen),
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, error) {
				m, err := NewDLinear(inputLen, outputLen, channels, 3)
				if err != nil {
					return nil, err
				}
				return m.TrainWindowed(w, l, cfg)
			},
		},
		{
			name:   "NHiTS",
			labels: labelsFlat(nSamples, 1, outputLen),
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, error) {
				m, err := NewNHiTS(NHiTSConfig{
					InputLength: inputLen, OutputLength: outputLen, Channels: channels,
					PoolKernels: []int{2, 4}, HiddenSize: 16, NumMLPLayers: 2,
				}, engine, ops)
				if err != nil {
					return nil, err
				}
				return m.TrainWindowed(w, l, cfg)
			},
		},
		{
			name:   "FreTS",
			labels: labelsFlat(nSamples, channels, outputLen),
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, error) {
				m, err := NewFreTS(FreTSConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					TopK: 3, HiddenSize: 16,
				})
				if err != nil {
					return nil, err
				}
				return m.TrainWindowed(w, l, cfg)
			},
		},
		{
			name:   "ITransformer",
			labels: labelsFlat(nSamples, channels, outputLen),
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, error) {
				m, err := NewITransformer(ITransformerConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
				}, nil, nil)
				if err != nil {
					return nil, err
				}
				return m.TrainWindowed(w, l, cfg)
			},
		},
		{
			name:   "CfC",
			labels: labelsFlat(nSamples, 1, channels*outputLen),
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, error) {
				m, err := NewCfC(CfCConfig{
					InputSize: channels, HiddenSize: 16, OutputSize: channels,
					NumLayers: 1, OutputLen: outputLen,
				})
				if err != nil {
					return nil, err
				}
				return m.TrainWindowed(w, l, cfg)
			},
		},
		{
			name:   "PatchTST",
			labels: labelsFlat(nSamples, 1, outputLen),
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, error) {
				m, err := NewPatchTST(PatchTSTConfig{
					InputLength: inputLen, PatchLength: 5, Stride: 3,
					DModel: 16, NHeads: 2, NLayers: 1,
					OutputDim: outputLen, ChannelIndependent: true,
				}, nil, ops)
				if err != nil {
					return nil, err
				}
				return m.TrainWindowed(w, l, cfg)
			},
		},
		{
			name:   "Mamba",
			labels: labelsFlat(nSamples, channels, outputLen),
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, error) {
				m, err := NewMamba(MambaConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					DModel: 16, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1,
				}, nil, ops)
				if err != nil {
					return nil, err
				}
				return m.TrainWindowed(w, l, cfg)
			},
		},
	}

	totalStart := time.Now()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			start := time.Now()
			res, err := tc.train(windows, tc.labels, config)
			elapsed := time.Since(start)
			if err != nil {
				t.Fatalf("training failed: %v", err)
			}
			if math.IsNaN(res.FinalLoss) || math.IsInf(res.FinalLoss, 0) {
				t.Fatalf("final loss is NaN/Inf: %v", res.FinalLoss)
			}
			samplesPerSec := float64(nSamples*epochs) / elapsed.Seconds()
			t.Logf("%s: %v (%.0f samples/sec, loss=%.6f)", tc.name, elapsed, samplesPerSec, res.FinalLoss)
		})
	}
	totalElapsed := time.Since(totalStart)
	t.Logf("TOTAL: %v for all 7 backends", totalElapsed)
	if totalElapsed > 30*time.Second {
		t.Errorf("total training time %v exceeds 30s budget", totalElapsed)
	}
}
