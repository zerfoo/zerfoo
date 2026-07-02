package fsdp

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

func TestDistributedCheckpoint(t *testing.T) {
	const worldSize = 2
	const paramSize = 128 // must be divisible by worldSize

	paramSizes := map[string]int{
		"weight_a": paramSize,
		"weight_b": paramSize * 2,
	}

	// Create a ShardedModule for each rank.
	makeModule := func(rank int) (*ShardedModule[float32], *mockModel) {
		model := newMockModel(paramSizes)
		sm := NewShardedModule[float32](model, rank, worldSize, nil)
		return sm, model
	}

	t.Run("save_rank0_creates_file", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "checkpoint.gguf")

		sm, _ := makeModule(0)
		if err := SaveCheckpoint(path, sm, 0); err != nil {
			t.Fatalf("SaveCheckpoint: %v", err)
		}

		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("checkpoint file not created: %v", err)
		}
		if info.Size() == 0 {
			t.Fatal("checkpoint file is empty")
		}
	})

	t.Run("save_rank1_does_not_write", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "checkpoint.gguf")

		sm, _ := makeModule(1)
		if err := SaveCheckpoint(path, sm, 1); err != nil {
			t.Fatalf("SaveCheckpoint: %v", err)
		}

		if _, err := os.Stat(path); err == nil {
			t.Fatal("rank 1 should not create checkpoint file")
		}
	})

	t.Run("round_trip_preserves_shards", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "checkpoint.gguf")

		// Create rank-0 module, save checkpoint.
		sm0, _ := makeModule(0)

		// Record original shards before save.
		origShards := make(map[string][]float32)
		for name, shard := range sm0.shards {
			cp := make([]float32, len(shard))
			copy(cp, shard)
			origShards[name] = cp
		}

		if err := SaveCheckpoint(path, sm0, 0); err != nil {
			t.Fatalf("SaveCheckpoint: %v", err)
		}

		// Create a fresh rank-0 module with different data and load checkpoint.
		sm0Load, model0 := makeModule(0)
		// Corrupt the shard data to ensure LoadCheckpoint overwrites it.
		for _, p := range model0.Parameters() {
			data := p.Value.Data()
			for i := range data {
				data[i] = 999.0
			}
		}

		if err := LoadCheckpoint(path, sm0Load, 0); err != nil {
			t.Fatalf("LoadCheckpoint rank 0: %v", err)
		}

		// Verify shards match the original.
		for name, origShard := range origShards {
			loadedShard := sm0Load.shards[name]
			if len(loadedShard) != len(origShard) {
				t.Errorf("param %q: shard size mismatch: got %d, want %d", name, len(loadedShard), len(origShard))
				continue
			}
			for i, v := range origShard {
				if loadedShard[i] != v {
					t.Errorf("param %q: shard[%d] = %v, want %v", name, i, loadedShard[i], v)
					break
				}
			}
		}
	})

	t.Run("round_trip_all_ranks", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "checkpoint.gguf")

		// Save from rank 0.
		sm0, _ := makeModule(0)
		if err := SaveCheckpoint(path, sm0, 0); err != nil {
			t.Fatalf("SaveCheckpoint: %v", err)
		}

		// Load on both ranks and verify each gets correct shard.
		for rank := 0; rank < worldSize; rank++ {
			smLoad, _ := makeModule(rank)
			if err := LoadCheckpoint(path, smLoad, rank); err != nil {
				t.Fatalf("LoadCheckpoint rank %d: %v", rank, err)
			}

			// Verify shard sizes.
			for name, fullSize := range paramSizes {
				shard := smLoad.shards[name]
				expectedShardSize := fullSize / worldSize
				if len(shard) != expectedShardSize {
					t.Errorf("rank %d param %q: shard size = %d, want %d",
						rank, name, len(shard), expectedShardSize)
				}
			}

			// Verify originalSizes are correct.
			for name, fullSize := range paramSizes {
				if smLoad.originalSizes[name] != fullSize {
					t.Errorf("rank %d param %q: originalSize = %d, want %d",
						rank, name, smLoad.originalSizes[name], fullSize)
				}
			}
		}
	})

	t.Run("forward_after_load_matches_original", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "checkpoint.gguf")

		// Save from rank 0.
		sm0, _ := makeModule(0)
		input, _ := tensor.New[float32]([]int{1}, []float32{1.0})

		origOut, err := sm0.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("original Forward: %v", err)
		}

		if err := SaveCheckpoint(path, sm0, 0); err != nil {
			t.Fatalf("SaveCheckpoint: %v", err)
		}

		// Load into fresh module, run forward.
		smLoad, _ := makeModule(0)
		if err := LoadCheckpoint(path, smLoad, 0); err != nil {
			t.Fatalf("LoadCheckpoint: %v", err)
		}

		loadedOut, err := smLoad.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("loaded Forward: %v", err)
		}

		// Compare outputs.
		origData := origOut.Data()
		loadedData := loadedOut.Data()
		if len(origData) != len(loadedData) {
			t.Fatalf("output size mismatch: %d vs %d", len(origData), len(loadedData))
		}
		for i, v := range origData {
			if loadedData[i] != v {
				t.Errorf("output[%d] = %v, want %v", i, loadedData[i], v)
			}
		}
	})
}

// mockCheckpointModel is a model that uses parameter data in forward pass
// to verify checkpoint correctness through output.
type mockCheckpointModel struct {
	params []*graph.Parameter[float32]
}

func newMockCheckpointModel(sizes map[string]int) *mockCheckpointModel {
	m := &mockCheckpointModel{}
	for name, size := range sizes {
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(i+1) * 0.01
		}
		t, _ := tensor.New[float32]([]int{size}, data)
		p, _ := graph.NewParameter[float32](name, t, tensor.New[float32])
		m.params = append(m.params, p)
	}
	return m
}

func (m *mockCheckpointModel) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Sum all parameter values as a simple deterministic output.
	var sum float32
	for _, p := range m.params {
		for _, v := range p.Value.Data() {
			sum += v
		}
	}
	out, _ := tensor.New[float32]([]int{1}, []float32{sum})
	return out, nil
}

func (m *mockCheckpointModel) Backward(_ context.Context, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (m *mockCheckpointModel) Parameters() []*graph.Parameter[float32] {
	return m.params
}
