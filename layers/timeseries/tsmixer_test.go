package timeseries

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestTSMixerBlock_ChannelIndependent_OutputShape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch, numPatches, dModel := 2, 8, 16
	block, err := NewTSMixerBlock[float32](engine, ops, numPatches, dModel, 4, false)
	if err != nil {
		t.Fatalf("NewTSMixerBlock: %v", err)
	}

	input, err := tensor.New[float32]([]int{batch, numPatches, dModel}, make([]float32, batch*numPatches*dModel))
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := block.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, numPatches, dModel}
	if len(got) != len(want) {
		t.Fatalf("output shape rank = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTSMixerBlock_ChannelMixing_OutputShape(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch, numPatches, dModel := 2, 8, 16
	block, err := NewTSMixerBlock[float32](engine, ops, numPatches, dModel, 4, true)
	if err != nil {
		t.Fatalf("NewTSMixerBlock: %v", err)
	}

	input, err := tensor.New[float32]([]int{batch, numPatches, dModel}, make([]float32, batch*numPatches*dModel))
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := block.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, numPatches, dModel}
	if len(got) != len(want) {
		t.Fatalf("output shape rank = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTSMixerBlock_ChannelMixing_DiffersFromIndependent(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch, numPatches, dModel := 1, 4, 8

	// Create input with non-zero values so the MLPs produce different outputs.
	data := make([]float32, batch*numPatches*dModel)
	for i := range data {
		data[i] = float32(i+1) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, numPatches, dModel}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	blockIndep, err := NewTSMixerBlock[float32](engine, ops, numPatches, dModel, 2, false)
	if err != nil {
		t.Fatalf("NewTSMixerBlock (independent): %v", err)
	}

	blockMixed, err := NewTSMixerBlock[float32](engine, ops, numPatches, dModel, 2, true)
	if err != nil {
		t.Fatalf("NewTSMixerBlock (mixed): %v", err)
	}

	outIndep, err := blockIndep.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward (independent): %v", err)
	}
	outMixed, err := blockMixed.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward (mixed): %v", err)
	}

	// Weights are random, so outputs will differ between blocks and modes.
	indepData := outIndep.Data()
	mixedData := outMixed.Data()
	allSame := true
	for i := range indepData {
		if indepData[i] != mixedData[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("channel-mixing and channel-independent outputs are identical; expected different results")
	}
}

func TestTSMixerBlock_ResidualConnection(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch, numPatches, dModel := 1, 4, 8

	// Use non-zero input so residual connection is visible.
	data := make([]float32, batch*numPatches*dModel)
	for i := range data {
		data[i] = 1.0
	}
	input, err := tensor.New[float32]([]int{batch, numPatches, dModel}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	block, err := NewTSMixerBlock[float32](engine, ops, numPatches, dModel, 2, false)
	if err != nil {
		t.Fatalf("NewTSMixerBlock: %v", err)
	}

	output, err := block.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Due to the residual connection, the output should be non-zero even if
	// the MLP branch were to produce zeros (it won't with random weights,
	// but the residual guarantees non-zero output regardless).
	outData := output.Data()
	allZero := true
	for _, v := range outData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("output is all zeros; residual connection should prevent this")
	}
}

func TestTSMixerBlock_BatchSizeGreaterThanOne(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	batch, numPatches, dModel := 4, 6, 12
	block, err := NewTSMixerBlock[float32](engine, ops, numPatches, dModel, 2, true)
	if err != nil {
		t.Fatalf("NewTSMixerBlock: %v", err)
	}

	data := make([]float32, batch*numPatches*dModel)
	for i := range data {
		data[i] = float32(i%7) * 0.1
	}
	input, err := tensor.New[float32]([]int{batch, numPatches, dModel}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := block.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, numPatches, dModel}
	if len(got) != len(want) {
		t.Fatalf("output shape rank = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTSMixerBlock_OpType(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	block, err := NewTSMixerBlock[float32](engine, ops, 4, 8, 2, false)
	if err != nil {
		t.Fatalf("NewTSMixerBlock: %v", err)
	}
	if got := block.OpType(); got != "TSMixerBlock" {
		t.Errorf("OpType = %q, want %q", got, "TSMixerBlock")
	}
}

func TestTSMixerBlock_Parameters(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	// Channel-independent: 2 time MLPs + 1 time norm (gamma + beta) = 4 params
	block, err := NewTSMixerBlock[float32](engine, ops, 4, 8, 2, false)
	if err != nil {
		t.Fatalf("NewTSMixerBlock: %v", err)
	}
	params := block.Parameters()
	if len(params) != 4 {
		t.Errorf("channel-independent params = %d, want 4", len(params))
	}

	// Channel-mixing: 4 (time) + 2 feat MLPs + 1 feat norm (gamma + beta) = 8 params
	block2, err := NewTSMixerBlock[float32](engine, ops, 4, 8, 2, true)
	if err != nil {
		t.Fatalf("NewTSMixerBlock: %v", err)
	}
	params2 := block2.Parameters()
	if len(params2) != 8 {
		t.Errorf("channel-mixing params = %d, want 8", len(params2))
	}
}

func TestNewTSMixerBlock_InvalidArgs(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name       string
		numPatches int
		dModel     int
		expansion  int
	}{
		{"zero numPatches", 0, 8, 2},
		{"negative numPatches", -1, 8, 2},
		{"zero dModel", 4, 0, 2},
		{"negative dModel", 4, -1, 2},
		{"zero expansion", 4, 8, 0},
		{"negative expansion", 4, 8, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewTSMixerBlock[float32](engine, ops, tt.numPatches, tt.dModel, tt.expansion, false)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}
