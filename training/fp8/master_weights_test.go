package fp8

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestMasterWeights(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	const (
		nLayers = 3
		dim     = 16
	)

	// Create 3 FP8Linear layers (16x16 each).
	layers := make([]*FP8Linear[float32], nLayers)
	for i := range layers {
		initData := make([]float32, dim*dim)
		for j := range initData {
			initData[j] = float32(j+1) * 0.01 * float32(i+1)
		}
		var err error
		layers[i], err = NewFP8Linear[float32](
			"layer_"+string(rune('a'+i)), engine, dim, dim, initData,
		)
		if err != nil {
			t.Fatalf("NewFP8Linear[%d]: %v", i, err)
		}
	}

	store, err := NewMasterWeightStore[float32](layers)
	if err != nil {
		t.Fatalf("NewMasterWeightStore: %v", err)
	}

	// Verify FP32Params count and shapes.
	fp32Params := store.FP32Params()
	if len(fp32Params) != nLayers {
		t.Fatalf("FP32Params() returned %d, want %d", len(fp32Params), nLayers)
	}
	for i, p := range fp32Params {
		shape := p.Shape()
		if len(shape) != 2 || shape[0] != dim || shape[1] != dim {
			t.Errorf("FP32Params[%d] shape = %v, want [%d, %d]", i, shape, dim, dim)
		}
	}

	// Record initial FP8 weight snapshots.
	initialFP8 := make([][]float32, nLayers)
	for i, layer := range layers {
		initialFP8[i] = make([]float32, len(layer.fp8WeightData))
		copy(initialFP8[i], layer.fp8WeightData)
	}

	// Simulate optimizer step: add delta to FP32 params.
	delta := float32(0.5)
	for _, p := range fp32Params {
		data := p.Data()
		for j := range data {
			data[j] += delta
		}
	}

	// Sync back to FP8.
	if err := store.SyncToFP8(); err != nil {
		t.Fatalf("SyncToFP8: %v", err)
	}

	// Verify FP8 weights changed.
	for i, layer := range layers {
		changed := false
		for j, v := range layer.fp8WeightData {
			if v != initialFP8[i][j] {
				changed = true
				break
			}
		}
		if !changed {
			t.Errorf("layer %d FP8 weights did not change after SyncToFP8", i)
		}
	}

	// Verify MemoryBytes.
	totalParams := int64(nLayers * dim * dim)
	expectedBytes := totalParams * 4
	gotBytes := store.MemoryBytes()
	if gotBytes != expectedBytes {
		t.Errorf("MemoryBytes() = %d, want %d", gotBytes, expectedBytes)
	}
	if gotBytes <= 0 {
		t.Error("MemoryBytes() should be > 0")
	}
}

func TestMasterWeightsMemoryDoc(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Create a single layer to measure memory overhead ratio.
	const dim = 64
	initData := make([]float32, dim*dim)
	for i := range initData {
		initData[i] = float32(i) * 0.001
	}

	layer, err := NewFP8Linear[float32]("bench", engine, dim, dim, initData)
	if err != nil {
		t.Fatalf("NewFP8Linear: %v", err)
	}

	store, err := NewMasterWeightStore[float32]([]*FP8Linear[float32]{layer})
	if err != nil {
		t.Fatalf("NewMasterWeightStore: %v", err)
	}

	masterBytes := store.MemoryBytes()
	// FP8 model weights use 1 byte per parameter.
	nParams := int64(dim * dim)
	fp8Bytes := nParams // 1 byte per FP8 param

	ratio := float64(masterBytes) / float64(fp8Bytes)
	t.Logf("FP32 master weight memory: %d bytes (%d params x 4 bytes)", masterBytes, nParams)
	t.Logf("FP8 model weight memory:   %d bytes (%d params x 1 byte)", fp8Bytes, nParams)
	t.Logf("Memory overhead ratio:     %.1fx (FP32 master / FP8 model)", ratio)

	// The ratio should be exactly 4.0 (4 bytes FP32 / 1 byte FP8).
	if math.Abs(ratio-4.0) > 0.01 {
		t.Errorf("expected memory ratio ~4.0x, got %.2fx", ratio)
	}
}

func TestMasterWeightsEmpty(t *testing.T) {
	store, err := NewMasterWeightStore[float32](nil)
	if err != nil {
		t.Fatalf("NewMasterWeightStore(nil): %v", err)
	}
	if len(store.FP32Params()) != 0 {
		t.Error("FP32Params() should be empty for nil layers")
	}
	if store.MemoryBytes() != 0 {
		t.Error("MemoryBytes() should be 0 for empty store")
	}
	if err := store.SyncToFP8(); err != nil {
		t.Errorf("SyncToFP8 on empty store: %v", err)
	}
}
