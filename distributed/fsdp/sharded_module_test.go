package fsdp

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// mockModel implements training.Model[float32] for testing.
type mockModel struct {
	params []*graph.Parameter[float32]
}

func newMockModel(paramSizes map[string]int) *mockModel {
	m := &mockModel{}
	for name, size := range paramSizes {
		data := make([]float32, size)
		for i := range data {
			data[i] = float32(i + 1)
		}
		t, _ := tensor.New[float32]([]int{size}, data)
		p, _ := graph.NewParameter[float32](name, t, tensor.New[float32])
		m.params = append(m.params, p)
	}
	return m
}

func (m *mockModel) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Return a dummy output using the first parameter's data length.
	out, _ := tensor.New[float32]([]int{1}, []float32{1.0})
	return out, nil
}

func (m *mockModel) Backward(_ context.Context, grad *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	// Simulate gradient computation: set gradients on each parameter.
	for _, p := range m.params {
		fullSize := p.Value.Size()
		gradData := make([]float32, fullSize)
		for i := range gradData {
			gradData[i] = 0.1
		}
		g, _ := tensor.New[float32]([]int{fullSize}, gradData)
		p.Gradient = g
	}
	return nil, nil
}

func (m *mockModel) Parameters() []*graph.Parameter[float32] {
	return m.params
}

func TestShardedModule(t *testing.T) {
	const worldSize = 8
	const paramSize = 1024 // must be divisible by worldSize

	paramSizes := map[string]int{
		"weight_1": paramSize,
		"weight_2": paramSize * 2,
		"weight_3": paramSize * 4,
	}

	// Total params across all layers.
	var totalParams int
	for _, s := range paramSizes {
		totalParams += s
	}

	t.Run("shard_size_equals_total_div_worldSize", func(t *testing.T) {
		model := newMockModel(paramSizes)
		sm := NewShardedModule[float32](model, 0, worldSize, nil)

		for name, fullSize := range paramSizes {
			shard, ok := sm.shards[name]
			if !ok {
				t.Fatalf("shard for %q not found", name)
			}
			expected := fullSize / worldSize
			if len(shard) != expected {
				t.Errorf("shard %q: got size %d, want %d", name, len(shard), expected)
			}
		}
	})

	t.Run("memory_reduction_8x", func(t *testing.T) {
		model := newMockModel(paramSizes)
		sm := NewShardedModule[float32](model, 0, worldSize, nil)

		shardMem := sm.ShardMemoryBytes()
		replicatedMem := sm.ReplicatedMemoryBytes()

		if replicatedMem == 0 {
			t.Fatal("replicated memory is zero")
		}
		ratio := float64(replicatedMem) / float64(shardMem)
		if ratio < float64(worldSize) {
			t.Errorf("memory reduction ratio = %.2f, want >= %d", ratio, worldSize)
		}
		// Verify exact 8x for these evenly-divisible sizes.
		if shardMem*int64(worldSize) != replicatedMem {
			t.Errorf("ShardMemoryBytes=%d * %d = %d != ReplicatedMemoryBytes=%d",
				shardMem, worldSize, shardMem*int64(worldSize), replicatedMem)
		}
	})

	t.Run("forward_completes_without_error", func(t *testing.T) {
		model := newMockModel(paramSizes)
		sm := NewShardedModule[float32](model, 0, worldSize, nil)

		input, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		output, err := sm.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}
		if output == nil {
			t.Fatal("Forward returned nil output")
		}
	})

	t.Run("parameters_restored_after_forward", func(t *testing.T) {
		model := newMockModel(paramSizes)
		sm := NewShardedModule[float32](model, 0, worldSize, nil)

		input, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		_, err := sm.Forward(context.Background(), input)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		// After forward, parameters should be back to sharded state.
		for _, p := range model.Parameters() {
			expectedSize := paramSizes[p.Name] / worldSize
			if p.Value.Size() != expectedSize {
				t.Errorf("param %q size after forward = %d, want %d (sharded)",
					p.Name, p.Value.Size(), expectedSize)
			}
		}
	})

	t.Run("backward_completes_and_scatters_gradients", func(t *testing.T) {
		model := newMockModel(paramSizes)
		sm := NewShardedModule[float32](model, 0, worldSize, nil)

		input, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		grad, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		_, err := sm.Backward(context.Background(), grad, input)
		if err != nil {
			t.Fatalf("Backward failed: %v", err)
		}

		// After backward, gradients should be scattered (sharded).
		for _, p := range model.Parameters() {
			if p.Gradient == nil {
				t.Errorf("param %q gradient is nil after backward", p.Name)
				continue
			}
			expectedSize := paramSizes[p.Name] / worldSize
			if p.Gradient.Size() != expectedSize {
				t.Errorf("param %q gradient size = %d, want %d (scattered)",
					p.Name, p.Gradient.Size(), expectedSize)
			}
		}
	})

	t.Run("all_ranks_get_correct_shards", func(t *testing.T) {
		for rank := 0; rank < worldSize; rank++ {
			model := newMockModel(map[string]int{"w": paramSize})
			sm := NewShardedModule[float32](model, rank, worldSize, nil)

			shard := sm.shards["w"]
			shardSize := paramSize / worldSize
			if len(shard) != shardSize {
				t.Errorf("rank %d: shard size = %d, want %d", rank, len(shard), shardSize)
			}

			// Verify shard contains the correct slice of the original data.
			expectedStart := float32(rank*shardSize + 1)
			if shard[0] != expectedStart {
				t.Errorf("rank %d: shard[0] = %v, want %v", rank, shard[0], expectedStart)
			}
		}
	})
}

func TestShardedModuleEdgeCases(t *testing.T) {
	t.Run("worldSize_1_no_sharding_effect", func(t *testing.T) {
		model := newMockModel(map[string]int{"w": 128})
		sm := NewShardedModule[float32](model, 0, 1, nil)

		if sm.ShardMemoryBytes() != sm.ReplicatedMemoryBytes() {
			t.Error("with worldSize=1, shard and replicated memory should be equal")
		}
	})

	t.Run("multiple_forward_backward_cycles", func(t *testing.T) {
		model := newMockModel(map[string]int{"w": 64})
		sm := NewShardedModule[float32](model, 0, 2, nil)

		input, _ := tensor.New[float32]([]int{1}, []float32{1.0})
		grad, _ := tensor.New[float32]([]int{1}, []float32{1.0})

		for i := 0; i < 5; i++ {
			_, err := sm.Forward(context.Background(), input)
			if err != nil {
				t.Fatalf("cycle %d: Forward failed: %v", i, err)
			}
			_, err = sm.Backward(context.Background(), grad, input)
			if err != nil {
				t.Fatalf("cycle %d: Backward failed: %v", i, err)
			}
		}
	})
}
