package timeseries

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)


// TestITransformer_ForwardBatchEngine_Parity verifies that the batched engine
// forward pass produces output matching sample-by-sample forward within 1e-5.
func TestITransformer_ForwardBatchEngine_Parity(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	channels := 4
	inputLen := 10
	outputLen := 3
	batch := 5

	config := ITransformerConfig{
		Channels: channels, InputLen: inputLen, OutputLen: outputLen,
		DModel: 16, DFF: 32, NHeads: 2, NLayers: 2,
	}
	m, err := NewITransformer(config, eng, numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	rng := rand.New(rand.NewPCG(42, 0))

	// Build input data.
	inputData := make([]float32, batch*channels*inputLen)
	samples := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		samples[b] = make([][]float64, channels)
		for c := 0; c < channels; c++ {
			samples[b][c] = make([]float64, inputLen)
			for i := 0; i < inputLen; i++ {
				v := rng.NormFloat64()
				samples[b][c][i] = v
				inputData[b*channels*inputLen+c*inputLen+i] = float32(v)
			}
		}
	}

	// Sample-by-sample forward (reference).
	sampleOutputs := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		sampleOutputs[b] = m.forward(samples[b])
	}

	// Batched forward.
	inputTensor, err := tensor.New[float32]([]int{batch, channels, inputLen}, inputData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	batchOut, err := m.forwardBatchEngine(context.Background(), inputTensor)
	if err != nil {
		t.Fatalf("forwardBatchEngine: %v", err)
	}

	outShape := batchOut.Shape()
	if outShape[0] != batch || outShape[1] != channels || outShape[2] != outputLen {
		t.Fatalf("output shape = %v, want [%d, %d, %d]", outShape, batch, channels, outputLen)
	}

	outData := batchOut.Data()
	const tol = 1e-5
	for b := 0; b < batch; b++ {
		for c := 0; c < channels; c++ {
			for o := 0; o < outputLen; o++ {
				got := float64(outData[b*channels*outputLen+c*outputLen+o])
				want := sampleOutputs[b][c][o]
				diff := math.Abs(got - want)
				if diff > tol {
					t.Errorf("batch[%d] chan[%d] out[%d]: got=%f want=%f diff=%f > tol=%f",
						b, c, o, got, want, diff, tol)
				}
			}
		}
	}
}

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

// TestITransformer_EngineForwardParity verifies that the engine-based forward
// pass produces output matching the CPU forward pass within float32 tolerance.
func TestITransformer_EngineForwardParity(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	channels := 5
	inputLen := 16
	outputLen := 6

	config := ITransformerConfig{
		Channels: channels, InputLen: inputLen, OutputLen: outputLen,
		DModel: 32, DFF: 64, NHeads: 4, NLayers: 2,
	}
	m, err := NewITransformer(config, eng, numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	rng := rand.New(rand.NewPCG(99, 0))
	input := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		input[c] = make([]float64, inputLen)
		for i := range input[c] {
			input[c][i] = rng.NormFloat64()
		}
	}

	// CPU forward.
	cpuOut, cpuCache := m.forwardWithCache(input)

	// Engine forward.
	engineOut, engineCache := m.forwardWithCacheEngine(context.Background(), input)

	// Compare outputs.
	const tol = 1e-3
	for c := 0; c < channels; c++ {
		for o := 0; o < outputLen; o++ {
			diff := math.Abs(cpuOut[c][o] - engineOut[c][o])
			if diff > tol {
				t.Errorf("output[%d][%d]: cpu=%f engine=%f diff=%f > tol=%f",
					c, o, cpuOut[c][o], engineOut[c][o], diff, tol)
			}
		}
	}

	// Compare cache preProj (tokens entering output projection).
	for c := 0; c < channels; c++ {
		for d := 0; d < config.DModel; d++ {
			diff := math.Abs(cpuCache.preProj[c][d] - engineCache.preProj[c][d])
			if diff > tol {
				t.Errorf("preProj[%d][%d]: cpu=%f engine=%f diff=%f > tol=%f",
					c, d, cpuCache.preProj[c][d], engineCache.preProj[c][d], diff, tol)
			}
		}
	}
}
