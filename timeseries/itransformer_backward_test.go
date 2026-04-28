package timeseries

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

// The bespoke finite-difference gradient check that used to live here
// (TestITransformer_BackwardBatchEngine_GradientCheck) was migrated to
// ztensor's shared gradcheck harness; see TestTimeseriesBackward_Gradcheck in
// gradcheck_test.go (plan T1.6). gradcheck verifies the float64 CPU backward,
// and TestITransformer_BackwardBatchEngine_ParityWithCPU below pins the
// engine-batched backward to that CPU path.

func TestITransformer_BackwardBatchEngine_ParityWithCPU(t *testing.T) {
	engine, ops := newTestEngine()
	config := ITransformerConfig{
		Channels:  2,
		InputLen:  4,
		OutputLen: 2,
		DModel:    4,
		DFF:       8,
		NHeads:    2,
		NLayers:   1,
		Seed:      1,
	}

	m, err := NewITransformer(config, engine, ops)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	ctx := context.Background()
	batch := 3
	channels := config.Channels
	inputLen := config.InputLen
	outputLen := config.OutputLen

	rng := rand.New(rand.NewPCG(42, 0))

	// Build input data as both [][]float64 slices and float32 tensors.
	windows := make([][][]float64, batch)
	inFlat := make([]float32, batch*channels*inputLen)
	for b := 0; b < batch; b++ {
		windows[b] = make([][]float64, channels)
		for c := 0; c < channels; c++ {
			windows[b][c] = make([]float64, inputLen)
			for i := 0; i < inputLen; i++ {
				v := rng.NormFloat64() * 0.5
				windows[b][c][i] = v
				inFlat[b*channels*inputLen+c*inputLen+i] = float32(v)
			}
		}
	}

	labels := make([]float64, batch*channels*outputLen)
	tgtFlat := make([]float32, batch*channels*outputLen)
	for i := range labels {
		v := rng.NormFloat64() * 0.5
		labels[i] = v
		tgtFlat[i] = float32(v)
	}

	// CPU path: per-sample forward/backward.
	scale := 1.0 / float64(batch*channels*outputLen)
	cpuGrads := newITransformerGrads(config)
	for s := 0; s < batch; s++ {
		pred, cache := m.forwardWithCache(windows[s])
		dOutput := make([][]float64, channels)
		for c := 0; c < channels; c++ {
			dOutput[c] = make([]float64, outputLen)
			for o := 0; o < outputLen; o++ {
				labelIdx := s*channels*outputLen + c*outputLen + o
				diff := pred[c][o] - labels[labelIdx]
				dOutput[c][o] = 2.0 * diff * scale
			}
		}
		m.backward(dOutput, cache, &cpuGrads)
	}
	cpuFlat := cpuGrads.collectGrads(config)

	// Engine batched path.
	inT, err := tensor.New[float32]([]int{batch, channels, inputLen}, inFlat)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}
	tgtT, err := tensor.New[float32]([]int{batch, channels, outputLen}, tgtFlat)
	if err != nil {
		t.Fatalf("tensor.New target: %v", err)
	}

	engineGrads, _, err := m.backwardBatchEngine(ctx, inT, tgtT)
	if err != nil {
		t.Fatalf("backwardBatchEngine: %v", err)
	}

	if len(cpuFlat) != len(engineGrads) {
		t.Fatalf("grad length mismatch: cpu=%d, engine=%d", len(cpuFlat), len(engineGrads))
	}

	maxRelErr := 0.0
	failCount := 0
	for i := range cpuFlat {
		denom := math.Max(math.Abs(cpuFlat[i]), math.Abs(engineGrads[i]))
		if denom < 1e-10 {
			continue
		}
		relErr := math.Abs(cpuFlat[i]-engineGrads[i]) / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > 1e-3 {
			failCount++
			if failCount <= 5 {
				t.Errorf("grad[%d]: cpu=%.8e, engine=%.8e, relErr=%.4e",
					i, cpuFlat[i], engineGrads[i], relErr)
			}
		}
	}

	if failCount > 0 {
		t.Errorf("%d/%d gradients exceed 0.1%% relative error", failCount, len(cpuFlat))
	}
	t.Logf("parity check: %d grads, maxRelErr=%.4e, failures=%d", len(cpuFlat), maxRelErr, failCount)
}

func TestITransformer_BackwardBatchEngine_LossReduction(t *testing.T) {
	engine, ops := newTestEngine()
	config := ITransformerConfig{
		Channels:  2,
		InputLen:  4,
		OutputLen: 2,
		DModel:    4,
		DFF:       8,
		NHeads:    2,
		NLayers:   1,
		Seed:      1,
	}

	m, err := NewITransformer(config, engine, ops)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	ctx := context.Background()
	batch := 4
	channels := config.Channels
	inputLen := config.InputLen
	outputLen := config.OutputLen

	rng := rand.New(rand.NewPCG(77, 0))

	inFlat := make([]float32, batch*channels*inputLen)
	for i := range inFlat {
		inFlat[i] = float32(rng.NormFloat64() * 0.5)
	}
	tgtFlat := make([]float32, batch*channels*outputLen)
	for i := range tgtFlat {
		tgtFlat[i] = float32(rng.NormFloat64() * 0.5)
	}

	inT, err := tensor.New[float32]([]int{batch, channels, inputLen}, inFlat)
	if err != nil {
		t.Fatalf("tensor.New input: %v", err)
	}
	tgtT, err := tensor.New[float32]([]int{batch, channels, outputLen}, tgtFlat)
	if err != nil {
		t.Fatalf("tensor.New target: %v", err)
	}

	_, loss, err := m.backwardBatchEngine(ctx, inT, tgtT)
	if err != nil {
		t.Fatalf("backwardBatchEngine: %v", err)
	}

	// Verify loss is finite and positive.
	if !isFinite(loss) {
		t.Fatalf("loss is not finite: %v", loss)
	}
	if loss < 0 {
		t.Fatalf("MSE loss should be non-negative, got %v", loss)
	}

	// Compute expected loss manually.
	expected := 0.0
	for b := 0; b < batch; b++ {
		sampleInput := make([][]float64, channels)
		for c := 0; c < channels; c++ {
			sampleInput[c] = make([]float64, inputLen)
			off := b*channels*inputLen + c*inputLen
			for i := 0; i < inputLen; i++ {
				sampleInput[c][i] = float64(inFlat[off+i])
			}
		}
		pred := m.forward(sampleInput)
		for c := 0; c < channels; c++ {
			for o := 0; o < outputLen; o++ {
				tgtIdx := b*channels*outputLen + c*outputLen + o
				diff := pred[c][o] - float64(tgtFlat[tgtIdx])
				expected += diff * diff
			}
		}
	}
	expected /= float64(batch * channels * outputLen)

	relErr := math.Abs(loss-expected) / math.Max(math.Abs(expected), 1e-10)
	if relErr > 1e-3 {
		t.Errorf("loss mismatch: batched=%.8e, manual=%.8e, relErr=%.4e", loss, expected, relErr)
	}
	t.Logf("loss parity: batched=%.8e, manual=%.8e, relErr=%.4e", loss, expected, relErr)
}

func TestITransformer_BackwardBatchEngine_InputValidation(t *testing.T) {
	engine, ops := newTestEngine()
	config := ITransformerConfig{
		Channels:  2,
		InputLen:  4,
		OutputLen: 2,
		DModel:    4,
		DFF:       8,
		NHeads:    2,
		NLayers:   1,
		Seed:      1,
	}

	m, err := NewITransformer(config, engine, ops)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	ctx := context.Background()

	tests := []struct {
		name     string
		inShape  []int
		tgtShape []int
		wantErr  string
	}{
		{
			name:     "2D input",
			inShape:  []int{2, 4},
			tgtShape: []int{1, 2, 2},
			wantErr:  "3D input",
		},
		{
			name:     "wrong channels",
			inShape:  []int{1, 3, 4},
			tgtShape: []int{1, 2, 2},
			wantErr:  "expected 2 channels",
		},
		{
			name:     "wrong inputLen",
			inShape:  []int{1, 2, 5},
			tgtShape: []int{1, 2, 2},
			wantErr:  "expected inputLen 4",
		},
		{
			name:     "2D target",
			inShape:  []int{1, 2, 4},
			tgtShape: []int{2, 2},
			wantErr:  "3D target",
		},
		{
			name:     "target shape mismatch",
			inShape:  []int{1, 2, 4},
			tgtShape: []int{2, 2, 2},
			wantErr:  "target shape",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inSize := 1
			for _, d := range tt.inShape {
				inSize *= d
			}
			tgtSize := 1
			for _, d := range tt.tgtShape {
				tgtSize *= d
			}
			inT, _ := tensor.New[float32](tt.inShape, make([]float32, inSize))
			tgtT, _ := tensor.New[float32](tt.tgtShape, make([]float32, tgtSize))

			_, _, err := m.backwardBatchEngine(ctx, inT, tgtT)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.wantErr)
			}
			if got := err.Error(); !contains(got, tt.wantErr) {
				t.Errorf("error %q does not contain %q", got, tt.wantErr)
			}
		})
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchSubstr(s, substr)
}

func searchSubstr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
