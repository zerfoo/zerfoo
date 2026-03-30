package timeseries

import (
	"context"
	"fmt"
	"math"
	"testing"

	its "github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// chronosTestConfig returns a small Chronos config suitable for tests.
func chronosTestConfig() *its.ChronosConfig {
	return &its.ChronosConfig{
		NumEncoderLayers: 2,
		NumDecoderLayers: 2,
		DModel:           16,
		NumHeads:         2,
		DFF:              32,
		VocabSize:        8,
		Horizon:          4,
	}
}

// makeChronosInput creates a [batch, seqLen] token ID tensor with values in [0, vocabSize).
func makeChronosInput(t *testing.T, batch, seqLen, vocabSize int) *tensor.TensorNumeric[float32] {
	t.Helper()
	data := make([]float32, batch*seqLen)
	for i := range data {
		data[i] = float32(i % vocabSize)
	}
	input, err := tensor.New[float32]([]int{batch, seqLen}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	return input
}

func TestChronosParity_OutputShapeBatchSizes(t *testing.T) {
	cfg := chronosTestConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	g, err := its.BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	ctx := context.Background()

	for _, tc := range []struct {
		name   string
		batch  int
		seqLen int
	}{
		{"batch1_seq4", 1, 4},
		{"batch2_seq5", 2, 5},
		{"batch4_seq8", 4, 8},
		{"batch8_seq3", 8, 3},
		{"batch1_seq1", 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := makeChronosInput(t, tc.batch, tc.seqLen, cfg.VocabSize)
			output, err := g.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			got := output.Shape()
			want := []int{tc.batch, cfg.Horizon, cfg.VocabSize}
			if len(got) != len(want) {
				t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("shape[%d]: got %d, want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func TestChronosParity_Deterministic(t *testing.T) {
	cfg := chronosTestConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	g, err := its.BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	ctx := context.Background()
	batch, seqLen := 2, 6

	// Run forward 10 times on the same input and verify identical outputs.
	input := makeChronosInput(t, batch, seqLen, cfg.VocabSize)

	firstOutput, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward (first): %v", err)
	}
	firstData := firstOutput.Data()

	for run := 1; run < 10; run++ {
		// Recreate input to avoid any tensor caching issues.
		inp := makeChronosInput(t, batch, seqLen, cfg.VocabSize)
		output, err := g.Forward(ctx, inp)
		if err != nil {
			t.Fatalf("Forward (run %d): %v", run, err)
		}

		data := output.Data()
		if len(data) != len(firstData) {
			t.Fatalf("run %d: output length %d != first %d", run, len(data), len(firstData))
		}
		for i, v := range data {
			if v != firstData[i] {
				t.Fatalf("run %d: output[%d] = %v, first = %v (not deterministic)", run, i, v, firstData[i])
			}
		}
	}
}

func TestChronosParity_NonDegenerate(t *testing.T) {
	cfg := chronosTestConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	g, err := its.BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	ctx := context.Background()

	// Run forward on 10 synthetic tokenized input series.
	for series := 0; series < 10; series++ {
		batch := 1
		seqLen := 5 + series // varying lengths
		data := make([]float32, batch*seqLen)
		for i := range data {
			data[i] = float32((i + series*3) % cfg.VocabSize)
		}
		input, err := tensor.New[float32]([]int{batch, seqLen}, data)
		if err != nil {
			t.Fatalf("series %d: create input: %v", series, err)
		}

		output, fErr := g.Forward(ctx, input)
		if fErr != nil {
			t.Fatalf("series %d: Forward: %v", series, fErr)
		}

		outData := output.Data()

		// Check all values are finite.
		for i, v := range outData {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("series %d: output[%d] is not finite: %v", series, i, v)
			}
		}

		// Check not all values are identical (degenerate output).
		allSame := true
		first := outData[0]
		for _, v := range outData[1:] {
			if v != first {
				allSame = false
				break
			}
		}
		if allSame && len(outData) > 1 {
			t.Errorf("series %d: all output values are identical (%v) — degenerate", series, first)
		}
	}
}

func TestChronosParity_DifferentInputsDifferentOutputs(t *testing.T) {
	cfg := chronosTestConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	g, err := its.BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	ctx := context.Background()
	batch, seqLen := 1, 6

	// Create two distinct inputs.
	dataA := make([]float32, batch*seqLen)
	dataB := make([]float32, batch*seqLen)
	for i := range dataA {
		dataA[i] = float32(i % cfg.VocabSize)
		dataB[i] = float32((i + 3) % cfg.VocabSize) // shifted tokens
	}

	inputA, err := tensor.New[float32]([]int{batch, seqLen}, dataA)
	if err != nil {
		t.Fatalf("create inputA: %v", err)
	}
	inputB, err := tensor.New[float32]([]int{batch, seqLen}, dataB)
	if err != nil {
		t.Fatalf("create inputB: %v", err)
	}

	outA, err := g.Forward(ctx, inputA)
	if err != nil {
		t.Fatalf("Forward A: %v", err)
	}
	outB, err := g.Forward(ctx, inputB)
	if err != nil {
		t.Fatalf("Forward B: %v", err)
	}

	// The encoder-decoder cross-attention should produce different outputs
	// for different encoder inputs.
	dA := outA.Data()
	dB := outB.Data()
	if len(dA) != len(dB) {
		t.Fatalf("output lengths differ: %d vs %d", len(dA), len(dB))
	}

	allSame := true
	for i := range dA {
		if dA[i] != dB[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("different inputs produced identical outputs — encoder-decoder attention is not discriminating")
	}
}

func TestChronosParity_BatchedVsSingle(t *testing.T) {
	cfg := chronosTestConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	g, err := its.BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	ctx := context.Background()
	seqLen := 5

	// Create a batch=2 input where both sequences are identical.
	singleData := make([]float32, seqLen)
	for i := range singleData {
		singleData[i] = float32(i % cfg.VocabSize)
	}
	batchData := make([]float32, 2*seqLen)
	copy(batchData[:seqLen], singleData)
	copy(batchData[seqLen:], singleData)

	singleInput, err := tensor.New[float32]([]int{1, seqLen}, singleData)
	if err != nil {
		t.Fatalf("create single input: %v", err)
	}
	batchInput, err := tensor.New[float32]([]int{2, seqLen}, batchData)
	if err != nil {
		t.Fatalf("create batch input: %v", err)
	}

	singleOut, err := g.Forward(ctx, singleInput)
	if err != nil {
		t.Fatalf("Forward single: %v", err)
	}
	batchOut, err := g.Forward(ctx, batchInput)
	if err != nil {
		t.Fatalf("Forward batch: %v", err)
	}

	singleData2 := singleOut.Data()
	batchData2 := batchOut.Data()

	horizonVocab := cfg.Horizon * cfg.VocabSize

	// batch[0] should match single output.
	for i := 0; i < horizonVocab; i++ {
		if singleData2[i] != batchData2[i] {
			t.Errorf("batch[0] output[%d] = %v, single = %v", i, batchData2[i], singleData2[i])
			break
		}
	}
	// batch[1] should also match since input is identical.
	for i := 0; i < horizonVocab; i++ {
		if batchData2[horizonVocab+i] != singleData2[i] {
			t.Errorf("batch[1] output[%d] = %v, single = %v", i, batchData2[horizonVocab+i], singleData2[i])
			break
		}
	}
}

func TestChronosParity_VaryingSequenceLengths(t *testing.T) {
	cfg := chronosTestConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	g, err := its.BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	ctx := context.Background()

	// Test various sequence lengths produce correct output shape.
	for _, seqLen := range []int{1, 2, 4, 8, 16, 32} {
		t.Run(fmt.Sprintf("seq%d", seqLen), func(t *testing.T) {
			input := makeChronosInput(t, 1, seqLen, cfg.VocabSize)
			output, err := g.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward seqLen=%d: %v", seqLen, err)
			}

			got := output.Shape()
			want := []int{1, cfg.Horizon, cfg.VocabSize}
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("seqLen=%d shape[%d]: got %d, want %d", seqLen, i, got[i], want[i])
				}
			}

			// Verify finite output.
			for _, v := range output.Data() {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("seqLen=%d: non-finite output", seqLen)
				}
			}
		})
	}
}

func TestChronosParity_LogitsRangeReasonable(t *testing.T) {
	cfg := chronosTestConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	g, err := its.BuildChronos[float32](tensors, cfg, engine)
	if err != nil {
		t.Fatalf("BuildChronos: %v", err)
	}

	ctx := context.Background()
	input := makeChronosInput(t, 2, 6, cfg.VocabSize)

	output, err := g.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Logits from a zero-initialized model should be relatively small in magnitude.
	// A reasonable range is [-1000, 1000]; anything beyond suggests numerical issues.
	for i, v := range output.Data() {
		fv := float64(v)
		if fv > 1000 || fv < -1000 {
			t.Errorf("logit[%d] = %v exceeds reasonable range [-1000, 1000]", i, v)
		}
	}
}
