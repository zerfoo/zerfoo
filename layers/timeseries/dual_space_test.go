package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestDualSpaceEncoder_ForwardShapes(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	dModel := 16
	patchLen := 8
	numLayers := 1
	batch := 2
	seqLen := 32 // 32 / 8 = 4 patches

	enc, err := NewDualSpaceEncoder[float32](engine, ops, dModel, patchLen, numLayers)
	if err != nil {
		t.Fatalf("NewDualSpaceEncoder: %v", err)
	}

	input, err := tensor.New[float32]([]int{batch, seqLen}, randomSlice[float32](batch*seqLen))
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	out, err := enc.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// FineGrained: [batch, numPatches, dModel]
	numPatches := seqLen / patchLen
	fgShape := out.FineGrained.Shape()
	wantFG := []int{batch, numPatches, dModel}
	if len(fgShape) != len(wantFG) {
		t.Fatalf("FineGrained rank = %d, want %d", len(fgShape), len(wantFG))
	}
	for i := range wantFG {
		if fgShape[i] != wantFG[i] {
			t.Errorf("FineGrained shape[%d] = %d, want %d", i, fgShape[i], wantFG[i])
		}
	}

	// Semantic: [batch, dModel]
	semShape := out.Semantic.Shape()
	wantSem := []int{batch, dModel}
	if len(semShape) != len(wantSem) {
		t.Fatalf("Semantic rank = %d, want %d", len(semShape), len(wantSem))
	}
	for i := range wantSem {
		if semShape[i] != wantSem[i] {
			t.Errorf("Semantic shape[%d] = %d, want %d", i, semShape[i], wantSem[i])
		}
	}
}

func TestDualSpaceEncoder_DFTRoundTrip(t *testing.T) {
	patchLen := 16
	numPatches := 3
	input := randomSlice[float32](numPatches * patchLen)

	output := dftRoundTrip(input, patchLen)

	if len(output) != len(input) {
		t.Fatalf("output length = %d, want %d", len(output), len(input))
	}

	const tol = 1e-4
	for i := range input {
		diff := math.Abs(float64(output[i] - input[i]))
		if diff > tol {
			t.Errorf("DFT round-trip mismatch at index %d: got %f, want %f (diff=%e)",
				i, output[i], input[i], diff)
		}
	}
}

func TestDualSpaceEncoder_TimeAndFreqDiffer(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	dModel := 8
	patchLen := 4
	numLayers := 1
	batch := 1
	seqLen := 8

	enc, err := NewDualSpaceEncoder[float32](engine, ops, dModel, patchLen, numLayers)
	if err != nil {
		t.Fatalf("NewDualSpaceEncoder: %v", err)
	}

	input, err := tensor.New[float32]([]int{batch, seqLen}, randomSlice[float32](batch*seqLen))
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	// Run only the time path.
	timeEmb, err := enc.timePatchEmbed.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("time patch embed: %v", err)
	}
	for i, block := range enc.timeEncoder {
		timeEmb, err = block.Forward(context.Background(), timeEmb)
		if err != nil {
			t.Fatalf("time encoder block %d: %v", i, err)
		}
	}

	// Run only the freq path.
	numPatches := seqLen / patchLen
	freqPatches, err := enc.applyFrequencyPath(context.Background(), input, batch, numPatches)
	if err != nil {
		t.Fatalf("freq path: %v", err)
	}
	freqEmb, err := enc.freqPatchEmbed.Forward(context.Background(), freqPatches)
	if err != nil {
		t.Fatalf("freq patch embed: %v", err)
	}
	for i, block := range enc.freqEncoder {
		freqEmb, err = block.Forward(context.Background(), freqEmb)
		if err != nil {
			t.Fatalf("freq encoder block %d: %v", i, err)
		}
	}

	// The two paths should produce different outputs since they have
	// different weights and the freq path applies DFT/IDFT transformation.
	timeData := timeEmb.Data()
	freqData := freqEmb.Data()

	allSame := true
	for i := range timeData {
		if math.Abs(float64(timeData[i]-freqData[i])) > 1e-6 {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("time and frequency paths produced identical outputs; expected different outputs")
	}
}

func TestDualSpaceEncoder_FusionOutputDModel(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	dModel := 16
	patchLen := 8
	numLayers := 1
	batch := 1
	seqLen := 16

	enc, err := NewDualSpaceEncoder[float32](engine, ops, dModel, patchLen, numLayers)
	if err != nil {
		t.Fatalf("NewDualSpaceEncoder: %v", err)
	}

	input, err := tensor.New[float32]([]int{batch, seqLen}, randomSlice[float32](batch*seqLen))
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	out, err := enc.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Verify fusion produces dModel, not 2*dModel.
	fgShape := out.FineGrained.Shape()
	if fgShape[2] != dModel {
		t.Errorf("FineGrained last dim = %d, want %d (not 2*dModel=%d)", fgShape[2], dModel, 2*dModel)
	}
}

func TestDualSpaceEncoder_BatchGreaterThanOne(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	dModel := 8
	patchLen := 4
	numLayers := 1
	batch := 4
	seqLen := 12 // 12 / 4 = 3 patches

	enc, err := NewDualSpaceEncoder[float32](engine, ops, dModel, patchLen, numLayers)
	if err != nil {
		t.Fatalf("NewDualSpaceEncoder: %v", err)
	}

	input, err := tensor.New[float32]([]int{batch, seqLen}, randomSlice[float32](batch*seqLen))
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	out, err := enc.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	numPatches := seqLen / patchLen
	fgShape := out.FineGrained.Shape()
	if fgShape[0] != batch {
		t.Errorf("FineGrained batch = %d, want %d", fgShape[0], batch)
	}
	if fgShape[1] != numPatches {
		t.Errorf("FineGrained numPatches = %d, want %d", fgShape[1], numPatches)
	}

	semShape := out.Semantic.Shape()
	if semShape[0] != batch {
		t.Errorf("Semantic batch = %d, want %d", semShape[0], batch)
	}
}

func TestDualSpaceEncoder_Parameters(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	dModel := 8
	patchLen := 4
	numLayers := 2

	enc, err := NewDualSpaceEncoder[float32](engine, ops, dModel, patchLen, numLayers)
	if err != nil {
		t.Fatalf("NewDualSpaceEncoder: %v", err)
	}

	params := enc.Parameters()
	if len(params) == 0 {
		t.Fatal("Parameters() returned empty slice")
	}

	// Expected parameter count:
	// Per transformer block: Q, K, V projections (3) + norm1 gamma+beta (2) + FFN1, FFN2 (2) + norm2 gamma+beta (2) = 9
	// Time path: 1 (patch embed) + numLayers*9
	// Freq path: 2 (freq linear1+2) + 1 (patch embed) + numLayers*9
	// Fusion: 1
	// Total = 1 + numLayers*9 + 2 + 1 + numLayers*9 + 1 = 5 + 18*numLayers
	expected := 5 + 18*numLayers
	if len(params) != expected {
		t.Errorf("Parameters() count = %d, want %d", len(params), expected)
	}

	// Verify all parameters are non-nil.
	for i, p := range params {
		if p == nil {
			t.Errorf("Parameter %d is nil", i)
		}
		if p.Value == nil {
			t.Errorf("Parameter %d (%s) has nil Value", i, p.Name)
		}
	}
}

func TestDualSpaceEncoder_InvalidInputRank(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	enc, err := NewDualSpaceEncoder[float32](engine, ops, 8, 4, 1)
	if err != nil {
		t.Fatalf("NewDualSpaceEncoder: %v", err)
	}

	// 3D input should be rejected.
	input3d, err := tensor.New[float32]([]int{1, 4, 2}, make([]float32, 8))
	if err != nil {
		t.Fatalf("create 3D input: %v", err)
	}

	_, err = enc.Forward(context.Background(), input3d)
	if err == nil {
		t.Error("expected error for 3D input, got nil")
	}
}

func TestDualSpaceEncoder_PaddedInput(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	dModel := 8
	patchLen := 4
	numLayers := 1
	batch := 1
	seqLen := 10 // not divisible by 4: will be padded to 12

	enc, err := NewDualSpaceEncoder[float32](engine, ops, dModel, patchLen, numLayers)
	if err != nil {
		t.Fatalf("NewDualSpaceEncoder: %v", err)
	}

	input, err := tensor.New[float32]([]int{batch, seqLen}, randomSlice[float32](batch*seqLen))
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	out, err := enc.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// After padding 10 -> 12, numPatches = 12/4 = 3.
	expectedPatches := 3
	fgShape := out.FineGrained.Shape()
	if fgShape[1] != expectedPatches {
		t.Errorf("FineGrained numPatches = %d, want %d (after padding)", fgShape[1], expectedPatches)
	}
}
